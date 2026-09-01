"""gepa_data.py - the optimiser's dev pool, the judge-prompt shape, and the reward function.

Everything here is about DATA and SCORING; the search itself lives in gepa_optimize.py.

THE ONE DISCIPLINE THAT MATTERS: the optimizer sees the GEPA dev pool and nothing
else. `load_examples()` asserts, every time it is called, that no consultation in
the pool appears in the evaluation set (`master/pairs_master_frozen.json`, and
`master/dataset_v2.json` when the factorial builder lands it). The assertion is not
decorative - it is the reason the finished prompt can be handed to the grid as an
arm whose numbers mean something.

The pool (52 pairs / 22 consultations carved out at the ground-truth freeze, plus the 5
gepa_dev-tagged partial-omission pairs) becomes 79 judgements:

    22 clean notes        one per consultation (the ideal note; the errored notes'
                          shared twin), the false-alarm side of the objective
    16 add    errored     fabricated content
    22 change errored     altered content
    14 omit-complete      the removed fact survives nowhere in the note
     5 omit-partial       a residual mention survives, the load-bearing part is gone

The judge prompt has three parts and GEPA only ever rewrites the middle one:

    PREAMBLE          transcript then note, fixed, FIRST so every candidate shares
                      a byte-identical cache prefix per note (the main cost lever)
    <instruction>     the evolving block - the whole artifact of the optimisation
    OUTPUT_CONTRACT   fixed JSON shape + score direction, so w2_arms.parse_score_json
                      keeps working and the metric can never invert under the optimizer

Scoring policy (how many points a critical omission costs, where the flag line sits,
what counts as normal compression) lives INSIDE the evolvable block on purpose: that
calibration is exactly what we want the optimizer to discover.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path

import w2_common as W  # noqa: E402
from w2_arms import parse_score_json  # noqa: E402

MASTER = os.path.join(ROOT, "master")
DEV_POOL = os.path.join(MASTER, "gepa_dev_pool.json")
DEV_FACTORIAL = os.path.join(MASTER, "gepa_dev_factorial.json")   # v2 densification
PARTIAL_SEED = os.path.join(MASTER, "partial_omission_seed_set.json")
EVAL_SETS = [os.path.join(MASTER, "pairs_master_frozen.json"),
             os.path.join(MASTER, "dataset_v2.json")]

FLAG_THRESHOLD = 7          # w2_arms: the gepa-optimized arm flags on score <= 7
SEVERITY_WEIGHT = {"critical": 1.0, "supporting": 0.5, "peripheral": 0.25}

# ---------------------------------------------------------------- prompt shape
PREAMBLE = ("CONSULTATION TRANSCRIPT:\n{transcript}\n\n"
            "CLINICAL NOTE (AI-generated from the transcript above):\n{note}\n\n---\n\n")

OUTPUT_CONTRACT = """
Return ONLY a JSON object on one line, with no other text before or after it:
{"score": <integer 0-10>, "omissions": [{"missing": "<what the note is missing, or what it \
asserts wrongly>", "severity": "critical|supporting|peripheral", "why": "<one line: what could \
go wrong because of it>"}]}

The score runs one way and one way only: 10 = the note fully satisfies the criterion, \
0 = a severe violation. Leave "omissions" empty if you are penalising nothing."""


def build_prompt(instruction):
    """The full judge prompt for a candidate: fixed preamble + evolved block + fixed contract."""
    return PREAMBLE + instruction.strip() + "\n\n" + OUTPUT_CONTRACT.strip() + "\n"


def render(prompt_template, transcript, note):
    out = prompt_template.replace("{transcript}", transcript).replace("{note}", note)
    assert "{transcript}" not in out and "{note}" not in out
    return out


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------- the dev pool
def _eval_consultations():
    """{(stratum, id)} appearing in any evaluation set on disk. The guard's ground truth.

    A pair carrying `eval_set: false` does NOT count: dataset_v2 keeps the pairs the
    optimiser trained on in the file, tagged and held out of the evaluation set for every
    arm, precisely so this guard and the grid see the same item set. Reading them as
    evaluation items would make the guard refuse the dev pool over consultations that were
    never evaluated on.
    """
    ids = set()
    for path in EVAL_SETS:
        if not os.path.exists(path):
            continue
        blob = json.load(open(path))
        pairs = blob["pairs"] if isinstance(blob, dict) else blob
        for p in pairs:
            if p.get("eval_set") is False:
                continue
            ids.add((p["stratum"], p["id"]))
    return ids


def load_examples(verbose=True):
    """The dev judgements, in a stable order, with the eval-disjointness guard applied.

    Returns (examples, provenance). Every example carries what the reward function
    needs to write a SPECIFIC critique: which fact went missing, how the freeze graded
    its severity and why, and - for the partial class - the residual mention that
    survives and makes the omission easy to miss.
    """
    dev = json.load(open(DEV_POOL))
    transcripts, tprov = W.transcript_index()
    partial = json.load(open(PARTIAL_SEED))
    partial_dev = [p for p in partial["pairs"] if p.get("split") == "gepa_dev"]
    # v2: the severity x residual factorial applied to the dev split. v1 trained on five
    # omit-partial judgements, which is why its 40% partial detection was 2-of-5 and why the
    # minibatch and full-pool scores kept disagreeing. These are built on the same 22
    # consultations, so the evaluation set loses nothing by their existence.
    factorial_dev = []
    if os.path.exists(DEV_FACTORIAL):
        factorial_dev = [p for p in json.load(open(DEV_FACTORIAL))["pairs"]
                         if p.get("split") == "gepa_dev"]

    dev_ids = {(c["stratum"], c["id"]) for c in dev["consultations"]}
    dev_ids |= {(p["stratum"], p["id"]) for p in partial_dev}
    dev_ids |= {(p["stratum"], p["id"]) for p in factorial_dev}
    leak = sorted(dev_ids & _eval_consultations())
    if leak:
        raise SystemExit(
            "EVAL LEAK: these consultations are in both the GEPA dev pool and an evaluation "
            f"set - refusing to optimise on them: {leak}")

    by_consult = {(c["stratum"], c["id"]): c for c in dev["consultations"]}
    examples = []

    for (stratum, cid), c in sorted(by_consult.items()):
        transcript = transcripts[(stratum, cid)]
        clean_texts = {sha(p["clean"])[:12] for p in c["pairs"]}
        assert len(clean_texts) == 1, f"{cid}: {len(clean_texts)} distinct clean notes"
        examples.append({
            "key": f"{stratum}|{cid}|clean", "kind": "clean", "gold_is_error": False,
            "stratum": stratum, "consultation": cid, "transcript": transcript,
            "note": c["pairs"][0]["clean"], "clean_key": f"{stratum}|{cid}|clean",
            "severity": None, "fact": None, "severity_rationale": None, "residual": None})
        for p in c["pairs"]:
            kind = "omit-complete" if p["type"] == "omit" else p["type"]
            examples.append({
                "key": p["pair_id"], "kind": kind, "gold_is_error": True,
                "stratum": stratum, "consultation": cid, "transcript": transcript,
                "note": p["errored"], "clean_key": f"{stratum}|{cid}|clean",
                "severity": p.get("severity"), "fact": p.get("what") or p.get("change"),
                "severity_rationale": p.get("severity_rationale"), "residual": None,
                "mc_idx": p.get("mc_idx"),
                "must_contain": (c["fact_sheet"]["must_contain"][p["mc_idx"]]
                                 if isinstance(p.get("mc_idx"), int)
                                 and p["mc_idx"] < len(c["fact_sheet"]["must_contain"]) else None)})

    for p in partial_dev:  # the graded-omission design's partial class
        key = (p["stratum"], p["id"])
        sites = (p.get("residual") or {}).get("sites") or []
        examples.append({
            "key": p["pair_id"] + "|partial", "kind": "omit-partial", "gold_is_error": True,
            "stratum": p["stratum"], "consultation": p["id"],
            "transcript": transcripts[key], "note": p["errored"],
            "clean_key": f"{p['stratum']}|{p['id']}|clean",
            "severity": p.get("severity"), "fact": p.get("fact") or p.get("what"),
            "severity_rationale": None, "mc_idx": p.get("mc_idx"), "must_contain": None,
            "residual": [{"quote": s.get("quote"), "section": s.get("section"),
                          "strength": s.get("strength")} for s in sites[:3]]})

    for p in factorial_dev:  # v2 densification: complete + partial, graded on both axes
        key = (p["stratum"], p["id"])
        sites = (p.get("residual") or {}).get("verified_sites") or []
        examples.append({
            "key": p["pair_id"], "kind": p["class"], "gold_is_error": True,
            "stratum": p["stratum"], "consultation": p["id"],
            "transcript": transcripts[key], "note": p["errored"],
            "clean_key": f"{p['stratum']}|{p['id']}|clean",
            "severity": p.get("severity"), "fact": p.get("fact") or p.get("what"),
            "severity_rationale": p.get("severity_rationale"),
            "residual_level": p.get("residual_level"),
            "residual_strength": (p.get("residual") or {}).get("verified_strongest"),
            "mc_idx": None, "must_contain": None,
            "residual": [{"quote": s.get("quote"), "section": s.get("section"),
                          "strength": s.get("strength")} for s in sites[:3]] or None})

    examples.sort(key=lambda e: e["key"])
    for i, e in enumerate(examples):  # the grid's item_index, for the grid's seed formula
        e["item_index"] = i
    if verbose:
        n = {}
        for e in examples:
            n[e["kind"]] = n.get(e["kind"], 0) + 1
        print(f"dev pool: {len(examples)} judgements over {len(by_consult)} consultations "
              f"({', '.join(f'{k} {v}' for k, v in sorted(n.items()))})")
    prov = {"dev_pool": os.path.relpath(DEV_POOL, ROOT), "dev_pool_sha256": _file_sha(DEV_POOL),
            "dev_factorial": (os.path.relpath(DEV_FACTORIAL, ROOT)
                              if os.path.exists(DEV_FACTORIAL) else None),
            "dev_factorial_sha256": (_file_sha(DEV_FACTORIAL)
                                     if os.path.exists(DEV_FACTORIAL) else None),
            "n_factorial_dev": len(factorial_dev),
            "partial_seed": os.path.relpath(PARTIAL_SEED, ROOT),
            "partial_seed_sha256": _file_sha(PARTIAL_SEED),
            "transcript_sources": tprov, "n_examples": len(examples),
            "n_consultations": len(by_consult),
            "eval_sets_checked": [os.path.relpath(p, ROOT) for p in EVAL_SETS
                                  if os.path.exists(p)],
            "eval_consultations_seen": len(_eval_consultations()),
            "eval_overlap": 0}
    return examples, prov


def _file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---------------------------------------------------------------- reward + critique
def score_example(ex, raw_text):
    """One judgement -> (per-example reward in [0,1], parsed score, flagged, critique).

    Reward is 0.7 * the hard flag decision + 0.3 * magnitude-aware soft credit, so the
    optimizer sees a gradient rather than a 0/1 cliff: an 8 on a gold-errored note is a
    nearer miss than a 10 even though both sit the same side of the flag line.
    The critique is the natural-language text the reflection model reads.
    """
    val = parse_score_json(raw_text)
    if val is None:
        return 0.0, None, False, ("PARSE FAILURE - no usable {\"score\": n} JSON came back. "
                                  "Gold: " + _gold_summary(ex) + ". Emitted: "
                                  + (raw_text or "<empty>")[:400])
    flagged = val <= FLAG_THRESHOLD
    hard = 1.0 if flagged == ex["gold_is_error"] else 0.0
    soft = (1.0 - val / 10.0) if ex["gold_is_error"] else (val / 10.0)
    return 0.7 * hard + 0.3 * soft, val, flagged, _critique(ex, val, flagged)


def _gold_summary(ex):
    if ex["kind"] == "clean":
        return "a clean note - faithful and complete"
    if ex["kind"] == "omit-partial":
        return f"errored (omission, PARTIAL - a residual mention survives), severity {ex['severity']}"
    if ex["kind"] == "omit-complete":
        return f"errored (omission, complete removal), severity {ex['severity']}"
    return f"errored ({ex['kind']}), severity {ex['severity']}"


def _critique(ex, val, flagged):
    """The outcome-specific coaching the reflection model reads. False negatives on
    omissions NAME the missing fact - the one thing an earlier optimisation run could not
    do, because its training data had no omission category to name anything from."""
    head = (f"[{ex['key']}] gold: {_gold_summary(ex)}. Judge scored {val}/10 "
            f"(flags at <= {FLAG_THRESHOLD}, so this note was "
            f"{'FLAGGED' if flagged else 'PASSED'}).")
    if flagged == ex["gold_is_error"]:
        if not ex["gold_is_error"]:
            return (head + " TRUE NEGATIVE. Keep whatever tolerance let the note through: "
                    "paraphrase into clinical language, standard normal-exam phrasing, "
                    "conversational colour dropped, a fact recorded once instead of three times.")
        if ex["kind"].startswith("omit"):
            return (head + f" TRUE POSITIVE (omission). Missing: \"{_fact(ex)}\". Keep whatever "
                    "made the judge read the note against everything the transcript established "
                    "rather than only checking the note for internal correctness.")
        return head + " TRUE POSITIVE. Keep this calibration for altered/fabricated content."

    if flagged and not ex["gold_is_error"]:
        return (head + " FALSE ALARM. This note is faithful and complete; the judge penalised it "
                "anyway. The likely cause is ordinary summarisation being read as loss - a "
                "phrase-level detail, small talk, a restated point, a normal finding written in "
                "conventional shorthand. The instruction needs a sharper line between an omission "
                "that changes a clinician's understanding or the patient's care and normal "
                "clinical compression. False alarms are capped at 10% of clean notes: a judge "
                "that flags everything has solved nothing.")

    if ex["kind"] == "omit-partial":
        res = "; ".join(f'"{r["quote"]}" (in {r["section"]})' for r in (ex["residual"] or [])[:2])
        return (head + f" FALSE NEGATIVE (PARTIAL omission, {ex['severity']}). What was removed: "
                f"\"{_fact(ex)}\". The note still mentions the topic nearby - {res or 'a related '
                'phrase survives elsewhere'} - and that surviving mention is almost certainly why "
                "the judge let it pass. A topic that is still named is not the same as a fact that "
                "is still recorded. Ask what a reader would actually KNOW after reading the note: "
                "if the load-bearing part (the working diagnosis, the dose, the reason for urgency, "
                "the red flag, the pertinent negative) is gone, it is an omission even though a "
                "related phrase survives somewhere else.")
    if ex["kind"] == "omit-complete":
        return (head + f" FALSE NEGATIVE (omission, {ex['severity']}). The transcript establishes "
                f"content the note simply does not carry: \"{_fact(ex)}\"."
                + (f" Why it matters: {ex['severity_rationale'][:400]}"
                   if ex.get("severity_rationale") else "")
                + " The judge treated the note as acceptably complete. A note with zero wrong "
                "statements can still fail. The instruction needs an explicit step that checks the "
                "note AGAINST the transcript's own content - go looking for a working diagnosis, a "
                "red flag or safety-netting instruction, a drug dose or allergy, an examination "
                "finding, or a stated pertinent negative that never made it into the note.")
    if ex["kind"] == "add":
        return (head + f" FALSE NEGATIVE (fabrication, {ex['severity']}). The note asserts content "
                f"the transcript does not support: \"{_fact(ex)}\". It reads as plausible, "
                "professional documentation, which is exactly why it slipped through.")
    return (head + f" FALSE NEGATIVE (alteration, {ex['severity']}). The note changes an assertion "
            f"the transcript makes: \"{_fact(ex)}\" - a flipped negation, a wrong value, a "
            "hardened tentative decision, a wrong subject or side. Distinct from both fabrication "
            "and omission, and it needs its own cue.")


def _fact(ex):
    return (ex.get("fact") or ex.get("must_contain") or "the removed content")[:300]


# ---------------------------------------------------------------- objective vector
KINDS = ["clean", "add", "change", "omit-complete", "omit-partial"]


def objective(examples, results):
    """results = {key: {"score": int|None, "flagged": bool, "reward": float}} -> the vector.

    Detection is the arm's own flag rule (score <= 7), so the numbers here mean the same
    thing the released judge will do in production. Discrimination (errored scored below
    its clean twin) is computed alongside because that is the benchmark's headline metric, but it is
    NOT what the search optimises - a judge you can only use by comparing a note against a
    counterfactual clean twin is not a judge you can deploy.
    """
    by_kind = {k: [] for k in KINDS}
    for e in examples:
        r = results.get(e["key"])
        if r is not None:
            by_kind[e["kind"]].append((e, r))

    def rate(kind):
        rows = by_kind[kind]
        return (sum(1 for _, r in rows if r["flagged"]) / len(rows)) if rows else None

    def sw_rate(kinds):
        num = den = 0.0
        for k in kinds:
            for e, r in by_kind[k]:
                w = SEVERITY_WEIGHT.get(e["severity"] or "supporting", 0.5)
                den += w
                num += w * (1.0 if r["flagged"] else 0.0)
        return (num / den) if den else None

    clean_scores = {e["key"]: results[e["key"]]["score"]
                    for e in examples if e["kind"] == "clean" and e["key"] in results}
    disc = {}
    for k in ("add", "change", "omit-complete", "omit-partial"):
        rows = [(e, r) for e, r in by_kind[k]
                if clean_scores.get(e["clean_key"]) is not None and r["score"] is not None]
        disc[k] = (sum(1 for e, r in rows if r["score"] < clean_scores[e["clean_key"]]) / len(rows)
                   if rows else None)

    fa = rate("clean")
    od_omit_sw = sw_rate(["omit-complete", "omit-partial"])
    od_ac = sw_rate(["add", "change"])
    n_parsed = sum(1 for e in examples if results.get(e["key"], {}).get("score") is not None)
    vec = {
        "n_judged": len(results), "n_parse_failures": len(results) - n_parsed,
        "fa": fa,
        "od_omit_complete": rate("omit-complete"), "od_omit_partial": rate("omit-partial"),
        "od_add": rate("add"), "od_change": rate("change"),
        "od_omit_severity_weighted": od_omit_sw, "od_add_change_severity_weighted": od_ac,
        "discrimination": disc,
        "mean_reward": (sum(r["reward"] for r in results.values()) / len(results)
                        if results else None),
        "feasible": (fa is not None and fa <= 0.10),
    }
    vec["J"] = (0.45 * (od_omit_sw or 0.0) + 0.20 * (od_ac or 0.0)
                + 0.35 * (1.0 - (fa if fa is not None else 1.0)))
    return vec


def objective_line(vec):
    def pct(x):
        return "  n/a" if x is None else f"{100 * x:5.1f}"
    return (f"J={vec['J']:.3f} {'' if vec['feasible'] else '(INFEASIBLE) '}"
            f"| omit-complete {pct(vec['od_omit_complete'])}%  "
            f"omit-partial {pct(vec['od_omit_partial'])}%  "
            f"add {pct(vec['od_add'])}%  change {pct(vec['od_change'])}%  "
            f"FA {pct(vec['fa'])}%  omit-sw {pct(vec['od_omit_severity_weighted'])}%")
