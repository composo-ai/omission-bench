"""Consolidate a fact sheet down to its core assertions.

Matching the blind-extracted fact sheets against the authored ones showed that blind
extraction recovers 99.4% of authored facts but produces sheets ~1.8x fatter on
must_contain; the PriMock critiques showed a chunk of the fat is over-strictness.
This pass brings the extracted strata down toward the authored sheets' strictness WITHOUT
deleting anything:

  - per kept extracted sheet (primock v2.1, aci, trapblind, authored_extracted), ONE
    claude-opus-5 plan-path call classifies every must_contain item
        core        - a competent GP would consider the note DEFICIENT without it (the
                      severity rubric's action / safety / record-quality lens)
        contextual  - legitimate content whose absence does not make the note deficient
    The FULL sheet is retained; items gain a "scope" field. Ideal notes and pairs build
    from core items only (downstream, not here).
  - salience_traps gain "pair_eligible": false for importance == "peripheral" (critical +
    supporting stay eligible) - mechanical, no LLM.
  - Items are judged on their own clinical merit - the classifier is NEVER shown a target
    count and is told not to curve to one (instrument integrity; the authored-envelope
    comparison is a reported result, not an instruction).

Modes:
  --source primock|aci|trapblind|authored   classify that stratum -> master/fact_sheets_<stem>_core.json
  --gaps      reverse-recovery check: for the authored consultations, every
              extracted-only must_contain item classified core is a candidate authored-sheet
              coverage gap; one opus-5 call per consultation double-checks each candidate is
              genuinely absent from the authored sheet (semantic, not string).
              -> master/authored_coverage_gaps.json. authored_scenarios.json is NOT modified.
              Candidates come from the per-item match table between the blind-extracted and
              authored sheets (master/wd_provenance_matches.json, keys kept|<id>|mc, 1-based
              indices, verified aligned with the current sheets).
  --report    distribution stats + the gaps summary -> master/consolidation_report.json

Idempotent by id + fact-sheet sha (state: master/consolidate_state_<stem>.json); --max-seconds
bounds one invocation so a stratum can run as several blocking chunks; rerun until complete.
Model: claude-opus-5 on the plan path, as for all construction work. Usage:
  python corpus/consolidate_sheets.py --source primock [--workers 6] [--max-seconds S]
  python corpus/consolidate_sheets.py --gaps [--workers 6] [--max-seconds S]
  python corpus/consolidate_sheets.py --report
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import hashlib, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from common import claude_json, HERE

MODEL = "claude-opus-5"
EFFORT = "medium"
TIMEOUT = 420
ATTEMPTS = 3
T0 = time.time()
M = lambda *p: os.path.join(HERE, "master", *p)


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


SOURCE = _arg("--source")
WORKERS = int(_arg("--workers", "6"))
MAX_SECONDS = float(_arg("--max-seconds", "0"))
STEMS = {"primock": "primock", "aci": "aci", "trapblind": "trapblind",
         "authored": "authored_extracted"}
AUTHORED_TARGET_NOTE = ("authored sheets' mean must_contain (the strictness envelope the pass "
                        "aims toward: comparable, not identical; no hard cap)")

CLASSIFY_PROMPT = """You are a senior UK GP acting as a clinical auditor for an AI-evaluation research study.
Below is a primary-care consultation TRANSCRIPT and the numbered list of `must_contain` facts
extracted from it (each with its stated load_bearing grade). The list will be used to score AI
scribe notes for completeness. Blind extraction tends to over-include, so the study needs each
fact classified by whether the note truly NEEDS it.

Classify EVERY numbered fact as exactly one of:

- "core": a competent GP would consider a note of THIS consultation DEFICIENT without this
  fact. Apply the study's severity lens in order: (1) ACTION - the fact gates, licenses or
  directs what happens next (working impression, plan elements including drug names, doses,
  frequencies and durations, referrals, examination findings that drove the decision, the
  decisive pertinent negatives behind it); (2) SAFETY - it carries a safety function even if
  action today is unchanged (safety-netting instructions, red flags, allergies, medication
  changes, follow-up triggers); (3) RECORD QUALITY - its absence would materially weaken the
  note as a clinical record (the presenting complaint itself, key positives, duration/course
  where the assessment turns on it, material risk or social history the plan relies on).
- "contextual": legitimate content a good note MAY include, but its absence would not make the
  note deficient - descriptive colour, secondary detail that neither supports nor threatens
  the assessment, soft social context with no action attached, granular restatement where
  another listed fact already carries the clinical load (if two items overlap, the sharper
  clinical statement is core and the duplicative gloss is contextual), reassurance or
  education phrasing with no specific safety content.

Rules: judge each fact on its own clinical merit in the context of THIS consultation. Do NOT
aim for any particular count or ratio - never curve to a target. The load_bearing grade is
context, not the answer: a "high" fact can be contextual and a "medium" fact can be core.
If genuinely torn, choose "contextual" (the conservative direction for a strictness filter).

TRANSCRIPT:
{transcript}

MUST_CONTAIN FACTS (numbered):
{facts}

Output ONLY JSON: {"classifications": [ {"item": <number>, "scope": "core" | "contextual",
"reason": "<one short line>"} ]} - one entry per numbered fact, covering every number exactly once."""

GAPS_PROMPT = """You are a senior UK GP auditing the COVERAGE of a hand-authored clinical fact sheet for an
AI-evaluation research study. For the consultation below there are two artifacts: the AUTHORED
fact sheet (written with the transcript, the study's reference) and a list of CANDIDATE facts
that a separate blind extraction produced and graded clinically core, but which an automatic
match pass could not find in the authored sheet's must_contain list. Your job: for each
candidate, decide whether the authored sheet genuinely lacks that clinical content, or in fact
carries it somewhere (semantic equivalence, not string match - check its must_contain facts,
its must_not_contain assertions, and its salience_traps' correct_handling text).

- "present": the clinical substance of the candidate is captured somewhere in the authored
  sheet (say where in one line). Partial coverage that captures the clinical point counts as
  present; a different granularity of the same fact is present.
- "absent": the authored sheet genuinely does not carry this clinical content anywhere - a
  real coverage gap.

AUTHORED FACT SHEET:
{authored}

CANDIDATE FACTS (numbered, blind-extracted, graded core):
{candidates}

Output ONLY JSON: {"verdicts": [ {"candidate": <number>, "verdict": "present" | "absent",
"where": "<one line: where it is captured, or why it is genuinely missing>"} ]} - one entry
per candidate, every number exactly once."""


def fs_sha(rec):
    return hashlib.sha256(json.dumps(rec["fact_sheet"], sort_keys=True).encode()).hexdigest()[:16]


def out_of_budget():
    return MAX_SECONDS and time.time() - T0 > MAX_SECONDS


def numbered(lines):
    return "\n".join(f"{i + 1}. {t}" for i, t in enumerate(lines))


def classify_sheet(rec):
    """One plan-path call -> per-item scope classifications, or None on failure."""
    mc = rec["fact_sheet"]["must_contain"]
    facts = numbered([f"{it['fact']} (load_bearing: {it.get('load_bearing', '?')})" for it in mc])
    prompt = (CLASSIFY_PROMPT.replace("{transcript}", rec["transcript"])
              .replace("{facts}", facts))
    for _ in range(ATTEMPTS):
        r = claude_json(prompt, model=MODEL, effort=EFFORT, timeout=TIMEOUT, retries=1)
        cls = (r or {}).get("classifications")
        if (isinstance(cls, list) and len(cls) == len(mc)
                and all(isinstance(c, dict) and c.get("scope") in ("core", "contextual")
                        and isinstance(c.get("item"), int) for c in cls)
                and {c["item"] for c in cls} == set(range(1, len(mc) + 1))):
            return cls
        if out_of_budget():
            break
    return None


def run_source(source):
    stem = STEMS[source]
    infile = M(f"fact_sheets_{stem}.json")
    outfile = M(f"fact_sheets_{stem}_core.json")
    statefile = M(f"consolidate_state_{stem}.json")
    recs = json.load(open(infile))
    state = json.load(open(statefile)) if os.path.exists(statefile) else {}
    # drop stale state (sheet changed since classification - e.g. a stratum re-run)
    state = {rid: e for rid, e in state.items()
             if any(r["id"] == rid and e.get("fs_sha") == fs_sha(r) for r in recs)}
    todo = [r for r in recs if r["id"] not in state]
    print(f"[{source}] {len(recs)} kept sheets | {len(state)} already classified | "
          f"{len(todo)} to run on {MODEL} (plan path, effort={EFFORT}, {WORKERS} workers"
          + (f", budget {MAX_SECONDS:.0f}s" if MAX_SECONDS else "") + ")", flush=True)

    lock = threading.Lock()
    n = [len(state)]

    def save_state():
        tmp = statefile + ".tmp"
        json.dump(state, open(tmp, "w"), ensure_ascii=False, indent=1)
        os.replace(tmp, statefile)

    def one(rec):
        if out_of_budget():
            return
        cls = classify_sheet(rec)
        with lock:
            if cls is None:
                print(f"  {rec['id']}: classify FAILED (rerun to retry)", flush=True)
                return
            state[rec["id"]] = {"fs_sha": fs_sha(rec), "classifications": cls,
                                "model": MODEL, "route": "plan", "effort": EFFORT}
            n[0] += 1
            n_core = sum(1 for c in cls if c["scope"] == "core")
            print(f"  [{n[0]}/{len(recs)}] {rec['id']:28} core {n_core:2d} / "
                  f"{len(cls):2d} mc", flush=True)
            save_state()

    if todo:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(one, todo))
    missing = [r["id"] for r in recs if r["id"] not in state]
    if missing:
        print(f"[{source}] INCOMPLETE - {len(missing)} sheets unclassified (rerun): "
              f"{missing[:6]}{'...' if len(missing) > 6 else ''}", flush=True)
        return False

    # ---- write the _core view: full sheets + per-item scope + trap pair_eligible ----
    out = []
    for rec in recs:
        cls = {c["item"]: c["scope"] for c in state[rec["id"]]["classifications"]}
        r2 = json.loads(json.dumps(rec, ensure_ascii=False))   # deep copy, nothing shared
        for i, it in enumerate(r2["fact_sheet"]["must_contain"], 1):
            it["scope"] = cls[i]
        for t in r2["fact_sheet"]["salience_traps"]:
            t["pair_eligible"] = t.get("importance") != "peripheral"
        mc_core = sum(1 for v in cls.values() if v == "core")
        r2["consolidation_provenance"] = {
            "model": MODEL, "route": "plan", "effort": EFFORT,
            "pass": "must_contain items classified core vs contextual",
            "counts": {"mc_total": len(cls), "mc_core": mc_core,
                       "mc_contextual": len(cls) - mc_core,
                       "traps_pair_eligible": sum(
                           1 for t in r2["fact_sheet"]["salience_traps"] if t["pair_eligible"]),
                       "traps_peripheral_excluded": sum(
                           1 for t in r2["fact_sheet"]["salience_traps"] if not t["pair_eligible"])}}
        out.append(r2)
    tmp = outfile + ".tmp"
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, outfile)
    core_counts = [r["consolidation_provenance"]["counts"]["mc_core"] for r in out]
    print(f"[{source}] complete -> {os.path.relpath(outfile, HERE)} | mean core mc "
          f"{sum(core_counts)/len(core_counts):.1f} (of {sum(len(r['fact_sheet']['must_contain']) for r in out)/len(out):.1f} total)", flush=True)
    return True


# ---------------------------------------------------------------- reverse-recovery
def run_gaps():
    """Candidates = extracted-only core must_contain items (from the match table against the
    authored sheets); one call per consultation verifies each is genuinely absent from the
    authored sheet (semantic)."""
    outfile = M("authored_coverage_gaps.json")
    core_rows = {r["id"]: r for r in json.load(open(M("fact_sheets_authored_extracted_core.json")))}
    authored = {r["id"]: r for r in json.load(open(os.path.join(HERE, "authored_scenarios.json")))}
    matches = json.load(open(M("wd_provenance_matches.json")))
    raw_ids = [r["id"] for r in json.load(open(M("fact_sheets_raw_authored_extracted.json")))]
    not_assessable = [i for i in raw_ids if i not in core_rows]

    prior = json.load(open(outfile)) if os.path.exists(outfile) else {}
    per = prior.get("per_consultation", {})

    jobs = []
    for cid, rec in core_rows.items():
        key = f"kept|{cid}|mc"
        if key not in matches:
            print(f"  {cid}: no match-table entry - skipped", flush=True)
            continue
        mc = rec["fact_sheet"]["must_contain"]
        matched = {i for mt in matches[key]["matches"] for i in (mt.get("b") or [])}
        cand_idx = [i for i in range(1, len(mc) + 1)
                    if i not in matched and mc[i - 1].get("scope") == "core"]
        sha = hashlib.sha256(json.dumps(
            [mc[i - 1]["fact"] for i in cand_idx]).encode()).hexdigest()[:16]
        if per.get(cid, {}).get("cand_sha") == sha:
            continue   # idempotent: already verified this exact candidate set
        jobs.append((cid, rec, cand_idx, sha))
    print(f"[gaps] {len(core_rows)} assessable consultations | {len(jobs)} to verify | "
          f"not assessable (extracted sheet dropped at critique): {not_assessable}", flush=True)

    lock = threading.Lock()

    def save():
        obj = {"what": "reverse-recovery check - candidate authored-sheet coverage gaps",
               "note": "candidates are blind-extracted must_contain items classified core that "
                       "the automatic match pass found in no authored must_contain item; each was "
                       "then double-checked semantically against the FULL authored sheet "
                       "(must_contain + must_not_contain + trap correct_handling) by one "
                       "claude-opus-5 plan-path call per consultation. verdict=absent means a "
                       "genuine coverage gap. authored_scenarios.json is NOT modified; the 90 "
                       "authored pairs are untouched - this is a reported result, not a repair.",
               "not_assessable": {"ids": not_assessable,
                                  "reason": "no kept blind-extracted sheet (dropped at critique)"},
               "per_consultation": per}
        gaps = [(cid, g) for cid, d in per.items() for g in d.get("gaps", [])]
        obj["summary"] = {
            "consultations_assessed": len(per),
            "candidates_checked": sum(len(d.get("candidates", [])) for d in per.values()),
            "confirmed_gaps": len(gaps),
            "consultations_with_gaps": len({c for c, _ in gaps})}
        tmp = outfile + ".tmp"
        json.dump(obj, open(tmp, "w"), ensure_ascii=False, indent=1)
        os.replace(tmp, outfile)

    def one(job):
        cid, rec, cand_idx, sha = job
        if out_of_budget():
            return
        mc = rec["fact_sheet"]["must_contain"]
        if not cand_idx:
            with lock:
                per[cid] = {"cand_sha": sha, "candidates": [], "gaps": []}
                print(f"  {cid}: no extracted-only core items", flush=True)
                save()
            return
        afs = {k: authored[cid]["fact_sheet"][k]
               for k in ("must_contain", "must_not_contain", "salience_traps")}
        prompt = (GAPS_PROMPT
                  .replace("{authored}", json.dumps(afs, ensure_ascii=False, indent=1))
                  .replace("{candidates}", numbered([mc[i - 1]["fact"] for i in cand_idx])))
        verdicts = None
        for _ in range(ATTEMPTS):
            r = claude_json(prompt, model=MODEL, effort=EFFORT, timeout=TIMEOUT, retries=1)
            v = (r or {}).get("verdicts")
            if (isinstance(v, list) and len(v) == len(cand_idx)
                    and all(isinstance(x, dict) and x.get("verdict") in ("present", "absent")
                            and isinstance(x.get("candidate"), int) for x in v)
                    and {x["candidate"] for x in v} == set(range(1, len(cand_idx) + 1))):
                verdicts = v
                break
            if out_of_budget():
                break
        with lock:
            if verdicts is None:
                print(f"  {cid}: gap check FAILED (rerun to retry)", flush=True)
                return
            cands = [{"extracted_mc_index": i, "fact": mc[i - 1]["fact"],
                      "verdict": verdicts[k]["verdict"], "where": verdicts[k].get("where", "")}
                     for k, i in enumerate(cand_idx)]
            per[cid] = {"cand_sha": sha, "candidates": cands,
                        "gaps": [c for c in cands if c["verdict"] == "absent"],
                        "model": MODEL, "route": "plan"}
            print(f"  {cid}: {len(cand_idx)} candidates -> "
                  f"{sum(1 for c in cands if c['verdict'] == 'absent')} confirmed absent", flush=True)
            save()

    if jobs:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(one, jobs))
    save()
    missing = [cid for cid in core_rows if cid not in per]
    if missing:
        print(f"[gaps] INCOMPLETE - rerun for: {missing}", flush=True)
        return False
    print(f"[gaps] complete -> {os.path.relpath(outfile, HERE)}", flush=True)
    return True


# ---------------------------------------------------------------- report
def pctl(xs, q):
    xs = sorted(xs)
    i = (len(xs) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (i - lo), 1)


def dist(xs):
    n = len(xs)
    mean = sum(xs) / n
    sd = (sum((x - mean) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return {"n": n, "mean": round(mean, 1), "sd": round(sd, 2), "min": min(xs),
            "p25": pctl(xs, .25), "median": pctl(xs, .5), "p75": pctl(xs, .75), "max": max(xs)}


def run_report():
    authored_rows = json.load(open(os.path.join(HERE, "authored_scenarios.json")))
    authored_mc_all = [len(r["fact_sheet"]["must_contain"]) for r in authored_rows]
    kept_ext_ids = {r["id"] for r in json.load(open(M("fact_sheets_authored_extracted.json")))}
    authored_mc_kept27 = [len(r["fact_sheet"]["must_contain"]) for r in authored_rows
                          if r["id"] in kept_ext_ids]
    target = round(sum(authored_mc_kept27) / len(authored_mc_kept27), 1)

    strata = {}
    for source, stem in STEMS.items():
        path = M(f"fact_sheets_{stem}_core.json")
        if not os.path.exists(path):
            print(f"[report] {stem}_core.json missing - run --source {source} first")
            return False
        rows = json.load(open(path))
        total = [len(r["fact_sheet"]["must_contain"]) for r in rows]
        core = [r["consolidation_provenance"]["counts"]["mc_core"] for r in rows]
        from collections import Counter
        imp = Counter(t["importance"] for r in rows for t in r["fact_sheet"]["salience_traps"])
        eligible = sum(1 for r in rows for t in r["fact_sheet"]["salience_traps"]
                       if t["pair_eligible"])
        n_traps = sum(len(r["fact_sheet"]["salience_traps"]) for r in rows)
        strata[stem] = {
            "n_sheets": len(rows),
            "must_contain_total": dist(total),
            "must_contain_core": dist(core),
            "must_contain_contextual": dist([t - c for t, c in zip(total, core)]),
            "core_share": round(sum(core) / sum(total), 3),
            "vs_authored_target": {"target_mean": target,
                                   "core_mean_minus_target": round(sum(core) / len(core) - target, 1)},
            "traps": {"total": n_traps, "importance": dict(imp),
                      "pair_eligible": eligible, "peripheral_excluded": n_traps - eligible}}

    gaps_path = M("authored_coverage_gaps.json")
    gaps_summary = (json.load(open(gaps_path))["summary"] if os.path.exists(gaps_path)
                    else "NOT RUN - python corpus/consolidate_sheets.py --gaps")

    all_core = [c for s in STEMS.values() for c in
                [r["consolidation_provenance"]["counts"]["mc_core"]
                 for r in json.load(open(M(f"fact_sheets_{s}_core.json")))]]
    report = {
        "what": "consolidation pass report",
        "generated_by": "consolidate_sheets.py --report",
        "method": "one claude-opus-5 plan-path call per kept extracted sheet classifying every "
                  "must_contain item core vs contextual under the severity rubric's action/"
                  "safety/record-quality lens; full sheets retained (nothing deleted); ideal "
                  "notes and pairs will build from core items only. Traps: peripheral "
                  "importance -> pair_eligible false. The classifier never sees a target "
                  "count and is instructed not to curve toward one.",
        "authored_reference": {
            "mean_mc_all_30": round(sum(authored_mc_all) / len(authored_mc_all), 1),
            "mean_mc_over_the_27_with_kept_extracted_sheets": target,
            "note": AUTHORED_TARGET_NOTE},
        "per_stratum": strata,
        "pooled_extracted_core_mean": round(sum(all_core) / len(all_core), 1),
        "reverse_recovery_check": gaps_summary,
    }
    out = M("consolidation_report.json")
    json.dump(report, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"[report] authored target mean {target}")
    for stem, s in strata.items():
        print(f"  {stem:20} mc {s['must_contain_total']['mean']:5.1f} -> core "
              f"{s['must_contain_core']['mean']:5.1f} (share {s['core_share']:.0%}) | traps "
              f"pair-eligible {s['traps']['pair_eligible']}/{s['traps']['total']}")
    print(f"  gaps: {gaps_summary}")
    print(f"-> {os.path.relpath(out, HERE)}")
    return True


if __name__ == "__main__":
    if "--report" in sys.argv:
        ok = run_report()
    elif "--gaps" in sys.argv:
        ok = run_gaps()
    else:
        assert SOURCE in STEMS, "usage: --source primock|aci|trapblind|authored | --gaps | --report"
        ok = run_source(SOURCE)
    sys.exit(0 if ok is not False else 0)   # incomplete chunks exit 0; rerun until complete
