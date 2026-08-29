"""w2_pipeline.py - Arm B of W-F v2: the PIPELINE JUDGE (gepa/V2-DESIGN.md, "Arm B").

The idea Arm B tests is that our ground-truth CONSTRUCTION machinery, run at inference
time, is the strongest judge we know how to build: the transcript is available in
production, "reference-free" means no verified answer key rather than no source, and
every construction stage has an inference-time analogue. Three tiers, so the paper can
show where any gain comes from:

  B1  single-prompt checklist        - already built, `arm_checklist` in w2_baselines.py
  B2  pipeline-lite   (this file)    - extract facts from the TRANSCRIPT ONLY (cached per
                                       consultation), then one KEYED present/absent call
                                       per note. Score = share of facts present.
  B3  full pipeline   (this file)    - extraction -> critic audit (drop unsupported and
                                       duplicate facts, grade severity on
                                       specs/severity-rubric.md) -> keyed
                                       full/partial/absent TRICHOTOMY per note -> REFUTE
                                       pass over the flagged facts only -> severity-
                                       weighted aggregate + a legible missing-facts list.

Roles are the pinned ones (models.lock.json), each at its established settings:

  stage 1   extract  constructor   anthropic/claude-opus-5  effort medium   cached to disk
  stage 1b  audit    auditor       openai/gpt-5.5           effort high     cached to disk
  stage 2   check    judge-primary openai/gpt-5.4           effort medium   per note per tier
  stage 3   refute   auditor       openai/gpt-5.5           effort high     per note, flags only

DEVIATION ON RECORD (2026-08-13). The grid ran judge-primary at reasoning_effort "none", and
the first instinct was to match it exactly so any difference would be attributable to task
restructuring alone. A 2-consultation wiring pilot killed that: at "none" the checker answers a
36-way keyed question in ~190 output tokens with ZERO reasoning tokens - about 7 tokens per
verdict - and across four notes it never once used the trichotomy's middle label. Re-running the
same prompt on the two CLEAN notes (no omission pair was read to make this call) at "medium"
produced 1,100-1,200 reasoning tokens and did use "partial". "none" is calibrated for the grid's
single 2-6 sentence judgement; it is structurally incapable of per-fact work on a 40-item list,
the same class of mechanical inadequacy as the 1,024-token truncation that killed the RAGAS arm
in its own smoke. So the check stage runs at "medium" and this is disclosed rather than hidden:
if a tier beats the grid here, part of the gain may be deliberation rather than decomposition,
and that confound is for the confirmation batch to separate.

BLINDNESS (the honesty rail). Stages 1 and 1b must never see the note under judgement -
otherwise the checklist is contaminated by the thing it is used to score. That is enforced
structurally, not by convention: `build_extract_prompt` and `build_audit_prompt` take a
transcript (and a fact list) and NOTHING else - no note text is in scope inside them - and
`assert_blind` additionally checks the built prompt against every note of that consultation
before the call is made. A stage-1 prompt that contained a note would raise, not run.

KEYED VERDICTS (the RAGAS-fix pattern, w2_baselines.py header). Facts carry ids; the checker
returns a verdict per id; a mis-count is a set difference rather than a silent misalignment.
Missing ids trigger ONE targeted retry asking only about them; if any id is still unanswered
after that, the record is a parse_failure rather than a fraction over a mystery denominator.

REFUTE FLIPS ARE QUOTE-VERIFIED. The refuter must return the verbatim note text that carries
a fact it claims the checker missed; a flip is accepted only if that quote actually occurs in
the note (whitespace/case-normalised, >= 8 characters). Unverifiable quotes are counted
(`refute_unverified`) and the absence stands - a hallucinated rescue must never look like a
correct one.

SCORES. B2 aggregate = n_present / n_facts. B3 aggregate = severity-weighted CAPTURE,
sum(w * credit) / sum(w) with credit full 1.0 / partial 0.5 / absent 0.0 - so both tiers
share the grid's orientation (higher = better note, an omission pushes the score DOWN) and
w2_pipeline_analyze.py can sweep one threshold rule across everything. `flagged` is written
as None on purpose: no threshold is pre-registered for these tiers, the analysis sweeps it.
The severity weights (critical 4 / supporting 2 / peripheral 1) are INDICATIVE - a stated
guess at relative clinical cost, not a canon number; the analysis also reports the unweighted
variant so nothing rests on them.

    .venv/bin/python w2_pipeline.py --plan --dataset master/arms_smoke_subset.json
    .venv/bin/python w2_pipeline.py --dataset master/arms_smoke_subset.json --tiers B2,B3 -y
"""
import argparse, json, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor

from common import HERE, RESULTS, Run
import w2_common as W

EXPERIMENT = "w2-pipeline"
SPEC = "gepa/V2-DESIGN.md"
# The cache is pinned to w2-pipeline REGARDLESS of where a run writes its records. Extraction
# and audit are the instrument, bought once (2026-08-14) and shared by every later run; a
# second-family checker re-runs stage 2 against exactly those artifacts, so pointing the cache
# at the output directory would silently re-buy them on a different model and destroy the
# comparison. --experiment moves the OUTPUT only (added 2026-08-17).
CACHE_DIR = os.path.join(RESULTS, "w2-pipeline", "_cache")
TIERS = ["B2", "B3"]

# Pinned roles + the settings each role is established on elsewhere in the study.
EXTRACT_ROLE, EXTRACT_EFFORT = "constructor", "medium"    # W-A construction row
AUDIT_ROLE, AUDIT_EFFORT = "auditor", "high"              # models.lock auditor notes
CHECK_ROLE, CHECK_EFFORT = "judge-primary", "medium"      # the grid's judge; effort per the
#                                                           DEVIATION ON RECORD above
REFUTE_ROLE, REFUTE_EFFORT = AUDIT_ROLE, AUDIT_EFFORT

GEN_SEED = 11                 # cached per-consultation stages, shared across tiers + replicates
RETRY_SEED_OFFSET = 5         # targeted missing-id retry: a distinct sample, same item decade
REFUTE_SEED_OFFSET = 6
# Reasoning tokens count against max_tokens on both routes, so these budgets have to cover the
# thinking as well as the JSON. A truncated reply is a parse failure, which is the one failure
# mode that would silently cost the arm records rather than money.
EXTRACT_MAX_TOKENS = 12000
# 24000, not 12000: the auditor at effort high spends 5-12k tokens THINKING before it writes,
# and one consultation (aci/D2N155) burned 11,912 of a 12,000 budget and truncated mid-JSON on
# both attempts - a parse failure bought twice at full price. Mechanical fix, 2026-08-14.
AUDIT_MAX_TOKENS = 24000
# 16000 for the same reason, learned twice: at effort high the auditor can spend EVERY token
# of an 8,000 budget thinking and return an empty completion (finish_reason='length'), which
# llm() correctly refuses to treat as a verdict - it raised after 5 attempts and killed a chunk
# on 2026-08-14. The budget is the fix; the graceful degradation below is the backstop.
REFUTE_MAX_TOKENS = 16000

# INDICATIVE weights (see module docstring): a stated guess at relative clinical cost, not canon.
SEVERITY_WEIGHTS = {"critical": 4.0, "supporting": 2.0, "peripheral": 1.0}
PARTIAL_CREDIT = 0.5
CREDIT = {"full": 1.0, "partial": PARTIAL_CREDIT, "absent": 0.0,
          "present": 1.0}  # B2's binary verdicts share the scale


def check_budget(n_facts, per_fact=16, effort="medium"):
    """~8-16 output tokens per keyed verdict, plus slack; capped so nothing runs away.

    Reasoning tokens count against max_tokens, and the deliberating checker spends 1.1-1.2k of
    them on a 40-fact list (measured), so the reasoning configurations get their own headroom.
    """
    body = 512 + per_fact * max(1, n_facts)
    if effort in (None, "none"):
        return int(min(4096, max(1024, body)))
    return int(min(9000, max(3000, 2500 + body)))


def refute_budget(n_flagged, effort=REFUTE_EFFORT):
    """The refuter returns a verdict AND a verbatim quote per flagged fact, after reasoning."""
    body = 1024 + 90 * max(1, n_flagged)
    if effort in (None, "none"):
        return int(min(4096, max(1024, body)))
    return int(min(REFUTE_MAX_TOKENS, max(6000, 4000 + body)))


# ---------------------------------------------------------------- prompt builders
def facts_block(facts):
    """The numbered fact list every keyed prompt reads. Severity is deliberately NOT shown:
    it is an aggregation weight, and a checker told a fact is 'critical' is a checker being
    nudged about how hard to look."""
    return "\n".join(f"{f['id']}: {str(f['fact']).strip()}" for f in facts)


def build_extract_prompt(prompts, transcript):
    """Stage 1. TRANSCRIPT ONLY - no note text is in scope in this function, by construction."""
    return W.render(prompts["pipeline_extract"], transcript=transcript)


def build_audit_prompt(prompts, transcript, facts):
    """Stage 1b. TRANSCRIPT + the candidate fact list ONLY - same blindness rail."""
    return W.render(prompts["pipeline_audit"], transcript=transcript,
                    facts_block=facts_block(facts))


def build_check_prompt(prompts, tier, note_text, facts):
    """Stage 2. The only stage that may see the note."""
    name = "pipeline_check_tri" if tier == "B3" else "pipeline_check_binary"
    return W.render(prompts[name], note=note_text, facts_block=facts_block(facts))


def build_refute_prompt(prompts, note_text, facts):
    """Stage 3. Note + the claimed-missing facts; no transcript, so the refuter can only
    rescue a fact by finding it in the note."""
    return W.render(prompts["pipeline_refute"], note=note_text, facts_block=facts_block(facts))


def _norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def assert_blind(prompt, note_texts, stage):
    """Structural check that a transcript-only stage never saw a note under judgement.

    Compares normalised slices of each note (start, middle, end) against the built prompt.
    A note is a generated SOAP document, not a span of the transcript, so any hit here is a
    real leak rather than an incidental overlap."""
    p = _norm(prompt)
    for t in note_texts:
        n = _norm(t)
        if len(n) < 80:
            continue
        for start in (0, max(0, len(n) // 2 - 60), max(0, len(n) - 120)):
            probe = n[start:start + 120]
            if len(probe) >= 80 and probe in p:
                raise AssertionError(
                    f"BLINDNESS VIOLATION in {stage}: the note under judgement appears in the "
                    f"prompt ({probe[:60]!r}...)")


# ---------------------------------------------------------------- parsing
def parse_facts(text):
    """{"facts": [{"id","fact","evidence"}]} -> a normalised fact list, or None."""
    j = W.parse_json_blob(text)
    if not isinstance(j, dict):
        return None
    raw = j.get("facts")
    if not isinstance(raw, list) or not raw:
        return None
    out, seen = [], set()
    for i, item in enumerate(raw):
        if isinstance(item, str):
            item = {"fact": item}
        if not isinstance(item, dict):
            continue
        fact = str(item.get("fact") or "").strip()
        if not fact:
            continue
        fid = str(item.get("id") or "").strip().lower() or f"f{len(out) + 1}"
        if fid in seen:  # a repeated id would collapse two facts into one keyed slot
            fid = f"{fid}_{len(out) + 1}"
        seen.add(fid)
        out.append({"id": fid, "fact": fact,
                    "evidence": str(item.get("evidence") or "").strip()[:400]})
    return out or None


def parse_audit(text, candidate_ids):
    """{"facts": [{"id","fact","severity"}], "dropped": [...]} -> (kept, dropped, n_defaulted)."""
    j = W.parse_json_blob(text)
    if not isinstance(j, dict):
        return None
    raw = j.get("facts")
    if not isinstance(raw, list):
        return None
    want = {i.lower() for i in candidate_ids}
    kept, seen, defaulted = [], set(), 0
    for item in raw:
        if not isinstance(item, dict):
            continue
        fid = str(item.get("id") or "").strip().lower()
        fact = str(item.get("fact") or "").strip()
        if fid not in want or fid in seen or not fact:
            continue  # hallucinated / repeated ids are dropped, never invented into the sheet
        seen.add(fid)
        sev = str(item.get("severity") or "").strip().lower()
        if sev not in SEVERITY_WEIGHTS:
            sev, defaulted = "supporting", defaulted + 1
        kept.append({"id": fid, "fact": fact, "severity": sev,
                     "why": str(item.get("why") or "").strip()[:200]})
    dropped = []
    for item in (j.get("dropped") if isinstance(j.get("dropped"), list) else []):
        if isinstance(item, dict) and str(item.get("id") or "").strip().lower() in want:
            dropped.append({"id": str(item["id"]).strip().lower(),
                            "reason": str(item.get("reason") or "").strip()[:60]})
    if not kept:
        return None
    return kept, dropped, defaulted


def parse_keyed_verdicts(text, want_ids, allowed, stats=None):
    """{"verdicts": {"f1": "full", ...}} -> {id: label} for the ids we asked about.

    Tolerates a bare keyed object (no wrapper), booleans (B2's true/false), and the obvious
    synonyms. Returns None only when there is no usable keyed verdict at all.

    B2 is asked a binary question and occasionally answers "partial" anyway. That is coerced to
    "present" and counted (`coerced_partial`): B2's contract is coverage - "does the note carry
    this fact" - and scoring a partly-carried fact as absent would smuggle B3's trichotomy
    strictness into the tier whose whole purpose is to lack it."""
    j = W.parse_json_blob(text)
    if not isinstance(j, dict):
        return None
    v = j.get("verdicts")
    if not isinstance(v, dict):
        v = j.get("coverage") if isinstance(j.get("coverage"), dict) else None
    if not isinstance(v, dict):
        v = j
    syn = {"true": "present", "yes": "present", "covered": "present", "captured": "present",
           "false": "absent", "no": "absent", "missing": "absent", "not covered": "absent",
           "complete": "full", "fully": "full", "partially": "partial", "partial_": "partial"}
    want = {str(i).lower() for i in want_ids}
    out = {}
    for k, val in v.items():
        key = str(k).strip().lower()
        if key not in want:
            continue
        if isinstance(val, bool):
            lab = "present" if val else "absent"
        elif isinstance(val, dict):  # {"verdict": "full"} shaped replies
            lab = str(val.get("verdict") or val.get("label") or "").strip().lower()
        else:
            lab = str(val).strip().lower()
        lab = syn.get(lab, lab)
        if lab == "present" and "present" not in allowed:
            lab = "full"
        if lab == "full" and "full" not in allowed:
            lab = "present"
        if lab == "partial" and "partial" not in allowed:
            lab = "present"
            if stats is not None:
                stats["coerced_partial"] = stats.get("coerced_partial", 0) + 1
        if lab in allowed:
            out[key] = lab
    return out or None


def parse_refute(text, want_ids):
    """{"verdicts": {"f3": {"found": "full", "quote": "..."}}} -> {id: (found, quote)}."""
    j = W.parse_json_blob(text)
    if not isinstance(j, dict):
        return None
    v = j.get("verdicts") if isinstance(j.get("verdicts"), dict) else j
    want = {str(i).lower() for i in want_ids}
    out = {}
    for k, val in v.items():
        key = str(k).strip().lower()
        if key not in want:
            continue
        if isinstance(val, str):
            found, quote = val.strip().lower(), ""
        elif isinstance(val, dict):
            found = str(val.get("found") or val.get("verdict") or "").strip().lower()
            quote = str(val.get("quote") or "").strip()
        else:
            continue
        if found in ("yes", "true", "present"):
            found = "full"
        if found in ("false", "absent", "none", "not found"):
            found = "no"
        if found in ("full", "partial", "no"):
            out[key] = (found, quote)
    return out or None


def quote_in_note(quote, note_text, min_chars=8):
    """A refute flip counts only if its quote really is in the note (normalised)."""
    q, n = _norm(quote), _norm(note_text)
    if len(q) < min_chars:
        return False
    if q in n:
        return True
    # tolerate a trailing ellipsis or a trimmed clause: require a long verbatim head
    head = q[:60]
    return len(head) >= min_chars and head in n


# ---------------------------------------------------------------- aggregation
def aggregate_note(verdicts, facts, weighted):
    """(aggregate, detail) for one note. Higher = better note, as everywhere else in W2."""
    by_id = {f["id"]: f for f in facts}
    counts = {"full": 0, "partial": 0, "absent": 0}
    num = den = 0.0
    unum = uden = 0.0
    missing = []
    for fid, lab in sorted(verdicts.items()):
        lab_n = "full" if lab == "present" else lab
        counts[lab_n] = counts.get(lab_n, 0) + 1
        f = by_id.get(fid, {})
        w = SEVERITY_WEIGHTS.get(f.get("severity"), 2.0) if weighted else 1.0
        num += w * CREDIT.get(lab_n, 0.0)
        den += w
        unum += CREDIT.get(lab_n, 0.0)
        uden += 1.0
        if lab_n != "full":
            missing.append({"id": fid, "verdict": lab_n, "severity": f.get("severity"),
                            "fact": str(f.get("fact") or "")[:220]})
    agg = (num / den) if den else None
    detail = {"n_facts": len(verdicts), "n_full": counts["full"], "n_partial": counts["partial"],
              "n_absent": counts["absent"],
              "aggregate_unweighted": round(unum / uden, 6) if uden else None,
              "weighted_missingness": round(1.0 - agg, 6) if agg is not None else None,
              "missing_facts": missing[:60], "scale": [0, 1]}
    return (round(agg, 6) if agg is not None else None), detail


# ---------------------------------------------------------------- caches
class StageCache:
    """results/w2-pipeline/_cache/<kind>_<consultation>.json - the once-per-consultation
    artifacts, shared across tiers, replicates and reruns."""

    def __init__(self, kind):
        self.kind = kind
        os.makedirs(CACHE_DIR, exist_ok=True)
        self._locks, self._glock = {}, threading.Lock()

    def path(self, key):
        return os.path.join(CACHE_DIR, f"{self.kind}_{key}.json")

    def lock(self, key):
        with self._glock:
            return self._locks.setdefault(key, threading.Lock())

    def get(self, key):
        p = self.path(key)
        if not os.path.exists(p):
            return None
        try:
            return json.load(open(p))
        except json.JSONDecodeError:
            return None

    def put(self, key, obj):
        json.dump(obj, open(self.path(key), "w"), indent=1)


def cache_key(block):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{block['stratum']}__{block['consultation']}")


# ---------------------------------------------------------------- spend
def ledger_experiment_total(experiment):
    """Credits already booked to one experiment (completed Run rows in the ledger)."""
    path = os.path.join(RESULTS, "cost_ledger.jsonl")
    if not os.path.exists(path):
        return 0.0
    tot = 0.0
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("experiment") == experiment:
                tot += d.get("cost_usd") or 0.0
    return tot


class PipelineGuard:
    """Experiment-scoped tripwire: this smoke was authorized at a HARD CAP of $60 for the
    whole of Arm B, so the guard counts w2-pipeline credits only (the project-wide ledger is
    already past w2_common.SpendGuard's coded defaults and would refuse to start).

    Checked after every note and at every chunk boundary; past `stop` the runner finishes the
    chunk it is in, closes cleanly and reports partial coverage rather than dying mid-record."""

    def __init__(self, warn=45.0, stop=55.0, hard=60.0, experiment=None):
        self.warn, self.stop, self.hard = warn, stop, hard
        self.experiment = experiment or EXPERIMENT
        self.baseline = ledger_experiment_total(self.experiment)
        self.spent_here = 0.0
        self._warned = False
        self._lock = threading.Lock()

    @property
    def total(self):
        return self.baseline + self.spent_here

    def add(self, usd):
        with self._lock:
            self.spent_here += usd or 0.0
            if self.total >= self.warn and not self._warned:
                self._warned = True
                print(f"  ! SPEND WARNING: {self.experiment} credits {self.total:.2f} USD "
                      f">= {self.warn:.0f}", flush=True)
        if self.total >= self.hard:
            raise SystemExit(f"\nHARD CAP: {self.experiment} credits {self.total:.2f} USD >= "
                             f"{self.hard:.0f}. Stopping immediately.")

    def should_stop(self):
        return self.total >= self.stop


# ---------------------------------------------------------------- stage runners
class StageLog:
    """One JSONL row per cached stage call, so cost per note is computable per tier
    (extraction is shared by B2 and B3; the audit belongs to B3 alone)."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._lock = threading.Lock()

    def add(self, row):
        with self._lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(row) + "\n")


def run_stage(prompt, role, effort, parser, seed, max_tokens):
    """One cached-stage call through W.judge (llm() + the pre-registered single retry)."""
    vals, texts, metas, retried = W.judge(prompt, role, 1, seed, temperature=1.0,
                                          reasoning_effort=effort, parser=parser,
                                          max_tokens=max_tokens)
    rcs = W.receipts(metas)
    return vals[0], texts[0], W.receipt_totals(rcs), rcs, bool(retried)


def ensure_facts(block, prompts, cache, guard, stagelog, run_id, force=False):
    """Stage 1, once per consultation, cached to disk and reused everywhere."""
    key = cache_key(block)
    with cache.lock(key):
        hit = cache.get(key)
        if hit and hit.get("facts") and not force:
            return hit["facts"]
        prompt = build_extract_prompt(prompts, block["transcript"])
        assert_blind(prompt, [n["text"] for n in block["notes"]], "stage1-extract")
        facts, text, tot, rcs, retried = run_stage(
            prompt, EXTRACT_ROLE, EXTRACT_EFFORT, parse_facts, GEN_SEED, EXTRACT_MAX_TOKENS)
        guard.add(tot["cost_usd"])
        stagelog.add({"stage": "extract", "consultation": block["consultation"],
                      "stratum": block["stratum"], "role": EXTRACT_ROLE, "run_id": run_id,
                      "retried": retried, "totals": tot, "t": round(time.time(), 3)})
        if not facts:
            raise RuntimeError(f"extraction unparseable for {key} after retry")
        cache.put(key, {"consultation": block["consultation"], "stratum": block["stratum"],
                        "role": EXTRACT_ROLE, "effort": EXTRACT_EFFORT, "seed": GEN_SEED,
                        "prompt_sha256": W.sha256_text(prompts["pipeline_extract"]),
                        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "n_facts": len(facts), "totals": tot, "facts": facts})
        return facts


def ensure_audit(block, facts, prompts, cache, guard, stagelog, run_id, force=False):
    """Stage 1b, once per consultation, cached to disk. Transcript + fact list only."""
    key = cache_key(block)
    with cache.lock(key):
        hit = cache.get(key)
        if hit and hit.get("facts") and not force:
            return hit["facts"], hit
        prompt = build_audit_prompt(prompts, block["transcript"], facts)
        assert_blind(prompt, [n["text"] for n in block["notes"]], "stage1b-audit")
        ids = [f["id"] for f in facts]
        parsed, text, tot, rcs, retried = run_stage(
            prompt, AUDIT_ROLE, AUDIT_EFFORT, lambda t: parse_audit(t, ids), GEN_SEED,
            AUDIT_MAX_TOKENS)
        guard.add(tot["cost_usd"])
        stagelog.add({"stage": "audit", "consultation": block["consultation"],
                      "stratum": block["stratum"], "role": AUDIT_ROLE, "run_id": run_id,
                      "retried": retried, "totals": tot, "t": round(time.time(), 3)})
        if not parsed:
            raise RuntimeError(f"audit unparseable for {key} after retry")
        kept, dropped, defaulted = parsed
        rec = {"consultation": block["consultation"], "stratum": block["stratum"],
               "role": AUDIT_ROLE, "effort": AUDIT_EFFORT, "seed": GEN_SEED,
               "prompt_sha256": W.sha256_text(prompts["pipeline_audit"]),
               "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "n_candidates": len(facts), "n_kept": len(kept), "n_dropped": len(dropped),
               "severity_defaulted": defaulted, "totals": tot,
               "by_severity": {s: sum(1 for f in kept if f["severity"] == s)
                               for s in SEVERITY_WEIGHTS},
               "dropped": dropped, "facts": kept}
        cache.put(key, rec)
        return kept, rec


# ---------------------------------------------------------------- the per-note judgement
def judge_note(tier, note, facts, prompts, call, refute_effort=REFUTE_EFFORT,
               check_effort=CHECK_EFFORT, check_role=CHECK_ROLE):
    """Stage 2 (+ stage 3 for B3) for one note. Returns (aggregate, detail).

    `call(name, prompt, parser, role, effort, seed_offset, max_tokens)` runs one LLM call and
    accumulates its receipts - the runner owns cost and stores, this owns the contract.
    """
    ids = [f["id"] for f in facts]
    allowed = ("full", "partial", "absent") if tier == "B3" else ("present", "absent")
    per_fact_tokens = 16 if tier == "B3" else 14
    pstats = {}

    def check(subset_ids, seed_offset):
        subset = [f for f in facts if f["id"] in set(subset_ids)]
        prompt = build_check_prompt(prompts, tier, note["text"], subset)
        return call("check", prompt,
                    lambda t: parse_keyed_verdicts(t, subset_ids, allowed, pstats),
                    check_role, check_effort, seed_offset,
                    check_budget(len(subset_ids), per_fact_tokens, check_effort))

    verdicts = check(ids, 0) or {}
    missing = [i for i in ids if i not in verdicts]
    retried_missing = bool(missing)
    if missing:  # ONE targeted retry, asking only about the ids that came back missing
        extra = check(missing, RETRY_SEED_OFFSET) or {}
        verdicts.update({k: v for k, v in extra.items() if k in set(missing)})
    unknown = [i for i in ids if i not in verdicts]
    if unknown:  # brief's contract: every id answered, or the record is a parse failure
        return None, {"n_facts": len(ids), "unknown_ids": unknown[:40],
                      "n_unknown": len(unknown), "retried_missing_ids": retried_missing,
                      "parse_failure_reason": "unanswered fact ids after one targeted retry"}

    pre = dict(verdicts)
    flips, unverified, refuted, refute_error = [], [], [], None
    if tier == "B3":
        flagged = [i for i in ids if verdicts.get(i) == "absent"]
        if flagged:
            subset = [f for f in facts if f["id"] in set(flagged)]
            prompt = build_refute_prompt(prompts, note["text"], subset)
            try:
                got = call("refute", prompt, lambda t: parse_refute(t, flagged), REFUTE_ROLE,
                           refute_effort, REFUTE_SEED_OFFSET,
                           refute_budget(len(flagged), refute_effort)) or {}
            except Exception as ex:
                # A dead refuter must not cost us the note. Stage 3 only ever RESCUES an
                # absence, so losing it degrades to the pre-refute verdicts - the conservative
                # direction - and the record says so rather than looking like a clean run.
                got, refute_error = {}, str(ex)[:300]
            for fid, (found, quote) in got.items():
                if found == "no":
                    continue
                if quote_in_note(quote, note["text"]):
                    verdicts[fid] = found          # rescued: absent -> full / partial
                    flips.append({"id": fid, "to": found, "quote": quote[:200]})
                else:
                    unverified.append({"id": fid, "claimed": found, "quote": quote[:120]})
            refuted = flagged

    agg, detail = aggregate_note(verdicts, facts, weighted=(tier == "B3"))
    detail.update({"verdicts": verdicts, "retried_missing_ids": retried_missing,
                   "n_refuted": len(refuted), "refute_flips": len(flips),
                   "refute_flip_detail": flips[:40], "refute_unverified": len(unverified),
                   "refute_unverified_detail": unverified[:20],
                   "n_absent_pre_refute": sum(1 for v in pre.values() if v == "absent"),
                   "refute_error": refute_error,
                   "weighted": tier == "B3", "coerced_partial": pstats.get("coerced_partial", 0),
                   "severity_weights": SEVERITY_WEIGHTS if tier == "B3" else None})
    return agg, detail


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="W-F v2 Arm B: the pipeline judge (B2 + B3)")
    ap.add_argument("--tiers", default="B2,B3", help="comma list of " + ",".join(TIERS))
    ap.add_argument("--dataset", default="master/arms_smoke_subset.json")
    ap.add_argument("--subset-spec", default=None,
                    help="opt-in JSON: judge only the listed note_keys, and merge in any transcripts the benchmark index lacks (see w2_common.load_subset_spec). Used by the real-error arm to point these judges at census notes; off by default")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--run", type=int, default=None)
    ap.add_argument("--skip-blocks", type=int, default=0,
                    help="skip the first N consultations. With --max-blocks this slices the "
                         "corpus, so two processes can judge disjoint halves concurrently "
                         "(distinct --tag each, merged in analysis). Stage 1/1b runs at only "
                         "min(workers,3), so slicing is the only lever on a cold-cache arm")
    ap.add_argument("--max-blocks", type=int, default=None,
                    help="bound a live pilot to the first N consultations")
    ap.add_argument("--workers", type=int, default=4, help="clamped to 5 (shared account)")
    ap.add_argument("--chunk", type=int, default=8, help="consultations per Run/manifest")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--plan", action="store_true", help="plan the run (no calls) and exit")
    ap.add_argument("--warn-usd", type=float, default=45.0)
    ap.add_argument("--stop-usd", type=float, default=55.0)
    ap.add_argument("--hard-usd", type=float, default=60.0)
    ap.add_argument("--refresh-cache", action="store_true",
                    help="re-run extraction/audit even when cached (instrument change only)")
    ap.add_argument("--refute-effort", default=REFUTE_EFFORT,
                    choices=["none", "low", "medium", "high"],
                    help="reasoning effort for stage 3 (auditor's pinned setting is high)")
    ap.add_argument("--check-role", default=CHECK_ROLE,
                    help="models.lock role for stage 2 (the ONLY judge stage). Default is the "
                         "grid's judge; point it at another family to test whether the "
                         "restructuring result is judge-specific. Extraction, audit and the "
                         "refute pass are unaffected and stay as bought.")
    ap.add_argument("--experiment", default=EXPERIMENT,
                    help="results/<experiment>/ for records, manifests and the spend guard's "
                         "ledger scope. The stage cache stays pinned to w2-pipeline so the "
                         "bought extraction/audit artifacts are reused, never re-bought.")
    ap.add_argument("--check-effort", default=CHECK_EFFORT,
                    choices=["none", "minimal", "low", "medium", "high"],
                    help="reasoning effort for stage 2 (see DEVIATION ON RECORD in the header: "
                         "the grid's 'none' cannot do per-fact work on a 40-item list)")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    tiers = [t.strip().upper() for t in args.tiers.split(",") if t.strip()]
    bad = [t for t in tiers if t not in TIERS]
    if bad:
        raise SystemExit(f"unknown tier(s) {bad}; known: {TIERS}")
    workers = max(1, min(5, args.workers))

    names = ["pipeline_extract", "pipeline_check_binary"] if tiers == ["B2"] else \
            ["pipeline_extract", "pipeline_audit", "pipeline_check_binary",
             "pipeline_check_tri", "pipeline_refute"]
    prompts = W.load_prompts(sorted(set(names)))

    pairs, info = W.load_dataset(path=args.dataset)
    info["n_pairs_selected"] = len(pairs)
    transcripts, tprov = W.transcript_index()
    subset = W.load_subset_spec(args.subset_spec)
    if subset:
        transcripts.update(subset["transcripts"])
    blocks, notes = W.note_units(pairs, transcripts)
    blocks, notes = W.apply_subset_spec(subset, blocks, notes)
    if args.skip_blocks:
        blocks = blocks[args.skip_blocks:]
    if args.max_blocks:
        blocks = blocks[:args.max_blocks]
        notes = [n for b in blocks for n in b["notes"]]

    replicates = [args.run] if args.run else list(range(1, args.runs + 1))
    base_tag = args.tag or f"pipeline-{info['sha256'][:8]}"
    experiment = args.experiment
    from common import resolve_model
    resolve_model(args.check_role)   # fail fast: an unpinned check role is not a run
    stores = {t: W.RecordStore(experiment, f"{base_tag}-{t}") for t in tiers}
    stagelog = StageLog(os.path.join(RESULTS, experiment, "_state", f"{base_tag}-stages.jsonl"))
    facts_cache, audit_cache = StageCache("facts"), StageCache("audit")

    def done(tier, rep, note):
        return stores[tier].has(f"{tier}|r{rep}|{note['note_key']}")

    todo = [(t, rep, n) for rep in replicates for b in blocks for n in b["notes"] for t in tiers
            if not done(t, rep, n)]
    n_cached_facts = sum(1 for b in blocks if facts_cache.get(cache_key(b)))
    n_cached_audit = sum(1 for b in blocks if audit_cache.get(cache_key(b)))
    print(f"w2 pipeline judge | tiers={tiers} x {len(replicates)} replicate(s) x {len(notes)} "
          f"notes over {len(blocks)} consultations")
    print(f"  dataset: {info['pairs_file']} version={info['dataset_version']} "
          f"sha {info['sha256'][:12]} | {info['n_pairs']} pairs, by class {info['by_class']}")
    print(f"  {len(todo)} note-judgements to run; extraction cached for {n_cached_facts}/"
          f"{len(blocks)} consultations, audit for {n_cached_audit}/{len(blocks)}")
    print(f"  stores: " + ", ".join(os.path.relpath(s.path, HERE) for s in stores.values()))
    if args.plan:
        need_x = len(blocks) - n_cached_facts
        need_a = (len(blocks) - n_cached_audit) if "B3" in tiers else 0
        n_check = len(todo) + sum(1 for t, r, n in todo if t == "B3")  # + a refute call each
        print(f"  would buy ~{need_x} extraction + {need_a} audit + ~{n_check} check/refute calls")
        return
    if not todo and not args.refresh_cache:
        print("nothing to do - every judgement is already in the record store")
        return
    W.confirm("proceed?", args.yes)

    guard = PipelineGuard(args.warn_usd, args.stop_usd, args.hard_usd, experiment=experiment)
    print(f"  check stage: role {args.check_role} -> "
          f"{resolve_model(args.check_role)['resolved']} at effort {args.check_effort!r}; "
          f"extraction/audit reused from {os.path.relpath(CACHE_DIR, HERE)}")
    print(f"  spend guard: {experiment} baseline ${guard.baseline:.2f}, "
          f"warn ${guard.warn:.0f} / stop ${guard.stop:.0f} / hard ${guard.hard:.0f}")

    t0, made, stopped = time.time(), 0, False
    for rep in replicates:
        if stopped:
            break
        seed = W.RUN_SEEDS.get(rep, 11 * rep)
        pending = [b for b in blocks
                   if any(not done(t, rep, n) for t in tiers for n in b["notes"])]
        for ci, chunk in enumerate(W.chunked(pending, args.chunk), 1):
            if guard.should_stop():
                stopped = True
                print(f"\nSPEND STOP: {experiment} credits ${guard.total:.2f} >= "
                      f"${guard.stop:.0f} - stopping cleanly with partial coverage.")
                break
            params = W.run_params(
                {"arm": "B-pipeline", "tiers": tiers, "chunk": ci, "replicate": rep,
                 "roles": {"extract": [EXTRACT_ROLE, EXTRACT_EFFORT],
                           "audit": [AUDIT_ROLE, AUDIT_EFFORT],
                           "check": [args.check_role, args.check_effort],
                           "refute": [REFUTE_ROLE, args.refute_effort]},
                 "check_effort_deviation": (
                     "the grid ran judge-primary at reasoning_effort 'none'; measured in the "
                     "wiring pilot, 'none' answers a 36-way keyed question in ~190 output "
                     "tokens with 0 reasoning tokens and never uses the trichotomy's middle "
                     "label, so the check stage runs at "
                     f"'{args.check_effort}'. Disclosed: a win here may owe something to "
                     "deliberation as well as to decomposition."),
                 "gen_seed": GEN_SEED, "workers": workers,
                 "severity_weights": SEVERITY_WEIGHTS, "partial_credit": PARTIAL_CREDIT,
                 "severity_weights_status": "INDICATIVE - stated guess at relative clinical "
                                            "cost, not a canon number; the unweighted variant "
                                            "is recorded alongside",
                 "keyed_contract": "verdict per fact id; ONE targeted retry for missing ids, "
                                   "then parse_failure (no unknown denominator)",
                 "refute_contract": "one batched auditor call per note over the ABSENT ids "
                                    "only; a flip needs a verbatim quote verified to occur "
                                    "in the note",
                 "blindness": "stages 1/1b take transcript(+facts) only; assert_blind checks "
                              "every built prompt against every note of the consultation",
                 "flag_rule": "none - flagged is null, thresholds are swept in analysis",
                 "record_stores": {t: os.path.relpath(s.path, HERE) for t, s in stores.items()},
                 "stage_log": os.path.relpath(stagelog.path, HERE),
                 "cache_dir": os.path.relpath(CACHE_DIR, HERE)}, info)
            W.assert_dataset_stable(info)
            with Run(experiment, params=params, replicate=rep, seed=seed,
                     inputs=[info["pairs_file"]], spec=SPEC, allow_dirty=True) as run:
                run.register_prompts(prompts)
                chunk_records = []
                failed_blocks, failed_notes = [], []

                # ---- stage 1 / 1b for the whole chunk (cached; independent per consultation)
                def prepare(b):
                    try:
                        facts = ensure_facts(b, prompts, facts_cache, guard, stagelog,
                                             run.run_id, args.refresh_cache)
                        b["_facts"] = facts
                        if "B3" in tiers:
                            kept, _ = ensure_audit(b, facts, prompts, audit_cache, guard,
                                                   stagelog, run.run_id, args.refresh_cache)
                            b["_audited"] = kept
                    except Exception as ex:
                        failed_blocks.append((f"{b['stratum']}/{b['consultation']}", str(ex)[:200]))

                with ThreadPoolExecutor(max_workers=min(workers, 3)) as ex:
                    list(ex.map(prepare, chunk))
                for name, why in failed_blocks:
                    print(f"  ! stage-1 failure {name}: {why}", flush=True)

                # ---- stage 2 (+3) for every outstanding note-judgement in the chunk
                def worker(job):
                    """One note-judgement. Wrapped: ThreadPoolExecutor.map re-raises on
                    consumption, so before 2026-08-14 one dead call took the whole chunk down
                    with it (it did, twice). A failed note writes NO record, so the resume
                    retries it; the chunk carries on."""
                    try:
                        return judge_one(job)
                    except Exception as ex:
                        failed_notes.append({"tier": job[0], "note_key": job[1]["note_key"],
                                             "error": str(ex)[:300]})
                        print(f"    ! note failed {job[0]}|{job[1]['note_key']}: "
                              f"{str(ex)[:160]}", flush=True)

                def judge_one(job):
                    tier, note, b = job
                    facts = b["_audited"] if tier == "B3" else b["_facts"]
                    base_seed = seed * 100000 + note["item_index"] * 10
                    metas, calls = [], []

                    def call(name, prompt, parser, role, effort, seed_offset, max_tokens):
                        vals, _, ms, retried = W.judge(
                            prompt, role, 1, base_seed + seed_offset, temperature=1.0,
                            reasoning_effort=effort, parser=parser, max_tokens=max_tokens)
                        metas.extend(ms)
                        calls.append({"stage": name, "role": role, "retried": bool(retried)})
                        return vals[0]

                    agg, detail = judge_note(tier, note, facts, prompts, call,
                                             refute_effort=args.refute_effort,
                                             check_effort=args.check_effort,
                                             check_role=args.check_role)
                    rcs = W.receipts(metas)
                    tot = W.receipt_totals(rcs)
                    rec = {"key": f"{tier}|r{rep}|{note['note_key']}", "cell": tier, "arm": tier,
                           "arm_family": "pipeline", "tier": tier, "format": "score", "k": 1,
                           "temperature": 1.0, "replicate": rep, "run_seed": seed,
                           "base_seed": base_seed, "model_key": "pipeline",
                           "role": args.check_role, "model": metas[0]["model"],
                           **W.note_fields(note),
                           "n_facts": detail.get("n_facts"), "n_full": detail.get("n_full"),
                           "n_present": detail.get("n_full"), "n_partial": detail.get("n_partial"),
                           "n_absent": detail.get("n_absent"),
                           "refute_flips": detail.get("refute_flips", 0),
                           "aggregate": agg, "samples": [agg], "verdict": None, "flagged": None,
                           "parse_failure": agg is None, "detail": detail, "calls": calls,
                           "receipts": rcs, "totals": tot, "run_id": run.run_id,
                           "note_chars": len(note["text"]),
                           "transcript_chars": len(b["transcript"]),
                           "t": round(time.time(), 3), "substrate_frozen": info["frozen"],
                           "dataset_version": info["dataset_version"]}
                    stores[tier].put(rec)
                    chunk_records.append(rec)
                    guard.add(tot["cost_usd"])

                jobs = [(t, n, b) for b in chunk if b.get("_facts")
                        for n in sorted(b["notes"], key=lambda n: (n["note_role"] != "clean",
                                                                   n["note_key"]))
                        for t in tiers
                        if not done(t, rep, n) and (t != "B3" or b.get("_audited"))]
                W.run_block(jobs, worker, workers, warmup=False)
                made += len(jobs)
                print(f"  r{rep} chunk{ci}: {len(chunk)} consultations, {len(jobs)} judgements "
                      f"| {experiment} credits ${guard.total:.2f} "
                      f"(+${guard.spent_here:.2f} this session) | {time.time() - t0:.0f}s",
                      flush=True)
                run.save("w2_pipeline.json",
                         {"tag": base_tag, "tiers": tiers, "replicate": rep, "seed": seed,
                          "params": params, "transcript_sources": tprov,
                          "failed_blocks": failed_blocks, "failed_notes": failed_notes,
                          "n_records": len(chunk_records),
                          "records": chunk_records})
    for s in stores.values():
        s.close()
    print(f"\ndone: {made} judgements this session; {experiment} credits "
          f"${guard.spent_here:.2f} this session, ${guard.total:.2f} total")
    for t, s in stores.items():
        print(f"  {t}: {len(s.records)} records -> {os.path.relpath(s.path, HERE)}")
    if stopped:
        print("  PARTIAL COVERAGE - the spend stop fired; rerun to resume (idempotent by key)")


if __name__ == "__main__":
    main()
