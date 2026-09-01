"""The third optimisation campaign: canonical GEPA, a fair inner split, two legs.

    python3 gepa/gepa_v3.py --selftest
    python3 gepa/gepa_v3.py --leg pipeline --phase prep -y
    python3 gepa/gepa_v3.py --leg pipeline --phase seeds -y
    python3 gepa/gepa_v3.py --leg pipeline --phase iterate --n 3 -y
    python3 gepa/gepa_v3.py --leg pipeline --phase test -y
    python3 gepa/gepa_v3.py --phase report

WHY v3 EXISTS. v1 and v2 both drew the reflection traces and the acceptance decision from
ONE small pool - the overspecialisation the DSPy GEPA docs name outright, and it showed:
v1's winner read 0.685-equivalent on its dev pool and 0.549 paired on data it had never
seen. v3 fixes that at real scale (gepa/gepa_v3_data.py: TEST held out entirely, TRAIN
split REFLECT/VALID at consultation level), runs the mechanics the GEPA
paper (arXiv 2507.19457) and the DSPy implementation actually specify, and points the
search at a COMPOUND system - the place GEPA claims its advantage - rather than only at a
single monolithic prompt.

CANONICAL MECHANICS, and where each one lives here:

  per-instance Pareto frontier   `pareto_parent`: a candidate stays in contention if it is
                                 the best anyone has managed on at least ONE VALID
                                 instance, and parents are sampled in proportion to how
                                 many instances they lead. Aggregate-best selection would
                                 discard the candidate that fixed two hard partials and
                                 lost on average - which is GEPA's whole selection argument.
  promiscuous acceptance         `_one_iteration`: a child enters the pool on a MINIBATCH
                                 improvement. Exploration is the point; the frontier prunes
                                 later. (v2 did the opposite - full-pool accept, minibatch
                                 as a pre-filter - and accepted 1 mutation in 11 iterations.)
  system-aware merge             `merge_candidates`: enabled once >= 6 candidates exist,
                                 capped at 4 invocations. DISCLOSED ADAPTATION: the paper's
                                 merge swaps whole MODULES between two descendants of a
                                 common ancestor. Both legs here evolve exactly one module
                                 (leg A freezes extraction on purpose), so module-swapping
                                 is undefined and merge is implemented as a reflector-
                                 mediated crossover of two frontier candidates chosen for
                                 leading on DISJOINT instances. Same intent, one level down.
  rich textual feedback          `traces_*`: every failure names the gold fact that was
                                 removed, the surviving mention for partials, the severity
                                 and residual grades, the note's rank against its own clean
                                 twin, and - leg A only, and the sharpest signal in the run -
                                 the DIFF of per-fact verdicts between the errored note and
                                 its twin, i.e. exactly which fact the checker did or did
                                 not notice going missing.
  reflection LM                  auditor (gpt-5.5) at reasoning_effort MEDIUM, not high:
                                 cost, deliberately.

THE TWO LEGS

  leg `pipeline` (primary)  the B2 tier of the pipeline judge with its CHECK module
                            evolved and everything else frozen: the same transcript-only
                            extraction, from the same on-disk cache, at the same
                            reasoning_effort "none", scored by the same
                            w2_pipeline.judge_note contract. Only w2_prompts/
                            pipeline_check_binary.txt's instruction paragraph moves. This
                            is the one place in the study a real gain is plausible, because
                            a per-fact closed lookup is a question wording can genuinely
                            sharpen.
  leg `mono` (secondary)    the grid's FC-score judge at k=1, seeded with the grid's own
                            criterion and with the engineered prompt. Its confirm is gated:
                            nothing is bought on TEST unless the winner clears the
                            pre-registered margin on VALID.

THE OBJECTIVE, on VALID, for both legs:
  PRIMARY   paired tie-adjusted discrimination on omissions - errored note scores strictly
            below its OWN clean twin, ties count half. Chance 0.500.
  SECONDARY absolute detection at the most sensitive threshold whose false-alarm rate on
            VALID's 20 clean twins stays <= 10%.
Both are reported everywhere. Primary is paired rather than absolute because VALID carries
20 clean notes: the false-alarm axis moves in 5pp steps, so an absolute operating point is
too coarse to steer a search with. It is still reported, and the TEST read-out quotes both.

BUDGET. Hard cap $80 across BOTH legs, clean stop at $72, enforced off
results/cost_ledger.jsonl filtered to experiment `gepa-v3` plus a live sidecar so a killed
invocation cannot hide spend. Every phase is resumable: state in gepa/v3_state_<leg>.json,
per-note results in gepa/cache/v3_<leg>_cache.jsonl, so a killed invocation costs nothing
to redo and the whole run proceeds as short bounded synchronous calls.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path

import common as C          # noqa: E402
import w2_common as W       # noqa: E402
import w2_arms as A         # noqa: E402
import w2_pipeline as P     # noqa: E402
import gepa_v3_data as D    # noqa: E402

SPEC = "gepa/V2-DESIGN.md"
EXPERIMENT = "gepa-v3"
STUDENT_ROLE = "judge-primary"     # gpt-5.4, the grid's judge and the pipeline's checker
REFLECTION_ROLE = "auditor"        # gpt-5.5
REFLECTION_EFFORT = "medium"       # not high: cost
RUN_SEED = 11                      # the grid's replicate-1 seed, and w2_pipeline's GEN_SEED
FA_TARGET = 0.10

STATE = os.path.join(HERE, "v3_state_{leg}.json")
CACHE = os.path.join(HERE, "cache", "v3_{leg}_cache.jsonl")
SPEND_SIDECAR = os.path.join(HERE, "v3_spend.json")
LINEAGE = os.path.join(HERE, "v3_lineage.jsonl")
LOGFILE = os.path.join(HERE, "v3_run.log")
RESULTS_JSON = os.path.join(HERE, "v3_results.json")
WINNER_PROMPT = os.path.join(ROOT, "w2_prompts", "pipeline_check_v3.txt")

OMIT_KINDS = ("omit-complete", "omit-partial")
ERR_KINDS = ("omit-complete", "omit-partial", "add", "change")


def log(msg=""):
    print(msg, flush=True)
    with open(LOGFILE, "a") as f:
        f.write(msg + "\n")


def lineage(rec):
    with open(LINEAGE, "a") as f:
        f.write(json.dumps({"t_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            **rec}) + "\n")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


# ================================================================ prompt shapes
class Shape:
    """A student prompt is PREAMBLE + the evolved INSTRUCTION + a fixed CONTRACT.

    The split is taken from the shipped prompt file itself and asserted to round-trip byte
    for byte (`--selftest`), so the seed candidate IS the production prompt rather than a
    paraphrase of it, and the optimizer cannot rewrite the output contract out from under
    the parser. Everything the harness depends on structurally - the slots, the keyed-
    verdict contract, the score range - lives in the fixed parts; everything that is a
    JUDGEMENT about the task lives in the evolved part.
    """

    def __init__(self, name, path, split_after, contract_starts_with):
        self.name, self.path = name, os.path.join(ROOT, path)
        raw = open(self.path).read()
        head, rest = raw.split(split_after, 1)
        self.preamble = head + split_after
        i = rest.index(contract_starts_with)
        self.instruction = rest[:i].rstrip()
        self.contract = rest[i:]
        assert self.build(self.instruction) == raw, f"{path}: shape does not round-trip"
        self.source_sha256 = sha(raw)

    def build(self, instruction):
        return self.preamble + instruction.strip() + "\n\n" + self.contract


def pipeline_shape():
    return Shape("pipeline_check_binary", "w2_prompts/pipeline_check_binary.txt",
                 "{facts_block}\n\n", "Answer with one entry per fact ID")


def mono_shape():
    return Shape("grid_FC_score", "w2_prompts/grid_FC_score.txt",
                 "---\n\n", "Assess the note strictly against this criterion.")


def engineered_instruction():
    """The engineered completeness judge re-expressed in the grid's shape: its body and its
    scoring rule become the instruction block; its own JSON contract is dropped because the
    harness owns the contract. A DISCLOSED reshaping - the prompt's ORDER changes (transcript and
    note move to the front), so this seed is 'the engineered judge's content in the grid's
    frame', not that judge's own prompt byte for byte."""
    raw = open(os.path.join(ROOT, "w2_prompts", "engineered_completeness.txt")).read()
    body = raw.split("CONSULTATION TRANSCRIPT:")[0].strip()
    tail = raw.split("Work through the note against the transcript,", 1)[1]
    scoring = tail.strip().split("\n\n")[-1].strip()
    scoring = scoring.split("List in \"omissions\"")[0].strip()
    return body + "\n\n" + scoring


LEGS = {
    "pipeline": {
        "what": "the B2 pipeline judge's CHECK module (extraction frozen and cached)",
        "shape": pipeline_shape, "max_score": 1.0, "margin": 0.05,
        "seeds": ["seed_check_binary"], "k_search": 1, "k_test": 1,
    },
    "mono": {
        "what": "the grid's FC-score monolithic judge at k=1",
        "shape": mono_shape, "max_score": 10.0, "margin": 3.0,
        "seeds": ["seed_fc_score", "seed_engineered"], "k_search": 1, "k_test": 8,
    },
}


def seed_instructions(leg):
    shape = LEGS[leg]["shape"]()
    if leg == "pipeline":
        return [("seed_check_binary", shape.instruction)]
    return [("seed_fc_score", shape.instruction), ("seed_engineered", engineered_instruction())]


# ================================================================ cache
class Cache:
    """(template sha, note key, seed, k) -> the per-note result. Flushed on every write, so
    a killed invocation loses nothing it paid for."""

    def __init__(self, path):
        self.path, self.hits, self.misses, self.d = path, 0, 0, {}
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            for line in open(path):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.d[r["k"]] = r["v"]
        self._fh = None

    def get(self, key):
        v = self.d.get(key)
        self.hits += v is not None
        self.misses += v is None
        return v

    def put(self, key, val):
        with self._lock:
            self.d[key] = val
            if self._fh is None:
                self._fh = open(self.path, "a")
            self._fh.write(json.dumps({"k": key, "v": val}) + "\n")
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


# ================================================================ budget
class BudgetStop(Exception):
    pass


def experiment_spend():
    """Credits booked to THIS experiment, ledger + live sidecar.

    results/cost_ledger.jsonl is shared with every sibling experiment (a MEDEC run was in
    flight while this one started), so the cap counts `gepa-v3` rows only. A Run writes its
    row on exit, so an invocation killed mid-flight would otherwise spend real credits and
    report zero, an under-count of real spend. The sidecar is written as the calls come back
    and the larger of the two is taken per run_id.
    """
    path = os.path.join(C.RESULTS, "cost_ledger.jsonl")
    led = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("experiment") == EXPERIMENT:
                led[d.get("run_id")] = led.get(d.get("run_id"), 0.0) + (d.get("cost_usd") or 0.0)
    side = json.load(open(SPEND_SIDECAR)) if os.path.exists(SPEND_SIDECAR) else {}
    tot = sum(max(led.get(r, 0.0), side.get(r, 0.0)) for r in set(led) | set(side))
    return round(tot, 6), led, side


class Budget:
    def __init__(self, cap, stop, prior, run_id=None, leg_cap=None, leg_prior=0.0):
        self.cap, self.stop, self.prior = cap, stop, prior
        self.leg_cap, self.leg_prior = leg_cap, leg_prior
        self.spent, self.calls, self.by_role = 0.0, 0, {}
        self.run_id, self._since_flush = run_id, 0
        self._lock = threading.Lock()

    def bind(self, run_id):
        self.run_id = run_id

    def _flush(self):
        if not self.run_id:
            return
        side = json.load(open(SPEND_SIDECAR)) if os.path.exists(SPEND_SIDECAR) else {}
        side[self.run_id] = round(self.spent, 6)
        tmp = SPEND_SIDECAR + ".tmp"
        json.dump(side, open(tmp, "w"), indent=1)
        os.replace(tmp, SPEND_SIDECAR)

    def add(self, role, usd):
        with self._lock:
            self.spent += usd or 0.0
            self.calls += 1
            self.by_role[role] = round(self.by_role.get(role, 0.0) + (usd or 0.0), 6)
            self._since_flush += 1
            if self._since_flush >= 8:
                self._since_flush = 0
                self._flush()

    @property
    def total(self):
        return self.prior + self.spent

    @property
    def leg_total(self):
        return self.leg_prior + self.spent

    @property
    def exhausted(self):
        return self.total >= self.cap

    def should_stop(self):
        if self.total >= self.stop:
            return f"run stop reached (${self.total:.2f} of ${self.cap:.0f}, stop ${self.stop:.0f})"
        if self.leg_cap and self.leg_total >= self.leg_cap:
            return f"leg cap reached (${self.leg_total:.2f} of ${self.leg_cap:.0f})"
        return None


# ================================================================ the students
class StageGuard:
    """w2_pipeline.ensure_facts wants an object with .add(usd); this is that, wired to the
    run budget so a cached-stage call counts against the same cap as everything else."""

    def __init__(self, budget):
        self.budget = budget

    def add(self, usd):
        self.budget.add("constructor", usd)
        if self.budget.exhausted:
            raise BudgetStop()


_FACTS = {}


def facts_for(block, budget, run_id, prompts=None):
    """Stage 1 of the pipeline, from w2_pipeline's own on-disk cache - one extraction per
    consultation, shared with the pipeline judge's own runs, so the TEST comparison is against
    the identical fact list the B2 baseline was scored on."""
    key = P.cache_key(block)
    if key in _FACTS:
        return _FACTS[key]
    prompts = prompts or W.load_prompts(["pipeline_extract"], quiet=True)
    cache = P.StageCache("facts")
    stagelog = P.StageLog(os.path.join(C.RESULTS, EXPERIMENT, "_state", "v3-stages.jsonl"))
    facts = P.ensure_facts(block, prompts, cache, StageGuard(budget), stagelog, run_id)
    _FACTS[key] = facts
    return facts


def score_pipeline(template, ex, facts, base_seed, budget):
    """One note through the B2 contract: one keyed present/absent call over the frozen fact
    list, ONE targeted retry for unanswered ids, score = share of facts present. This calls
    w2_pipeline.judge_note itself rather than re-implementing it, so the tier being optimised
    is literally the one the published pipeline results were measured on."""
    metas = []

    def call(name, prompt, parser, role, effort, seed_offset, max_tokens):
        if budget.exhausted:
            raise BudgetStop()
        vals, _texts, ms, _r = W.judge(prompt, role, 1, base_seed + seed_offset,
                                       temperature=1.0, reasoning_effort=effort,
                                       parser=parser, max_tokens=max_tokens)
        metas.extend(ms)
        for m in ms:
            budget.add(STUDENT_ROLE, m.get("cost_usd_reported"))
        return vals[0]

    agg, detail = P.judge_note("B2", {"text": ex["note"]}, facts,
                               {"pipeline_check_binary": template}, call, check_effort="none")
    return {"score": agg,
            "detail": {"verdicts": detail.get("verdicts") or {},
                       "n_facts": detail.get("n_facts"),
                       "n_absent": detail.get("n_absent"),
                       "coerced_partial": detail.get("coerced_partial", 0),
                       "parse_failure_reason": detail.get("parse_failure_reason")}}


def score_mono(template, ex, k, base_seed, budget):
    """One note through the grid's scored-judge contract: transcript + note + the evolved
    criterion, a 2-6 sentence assessment, `Score: <0-10>`; k=1 during the search, winsorized
    mean at k=8 for the TEST confirm (w2_common.aggregate_cell's rule for score cells)."""
    if budget.exhausted:
        raise BudgetStop()
    prompt = A.render_arm(template, ex["transcript"], ex["note"])
    vals, texts, metas, _r = W.judge(prompt, STUDENT_ROLE, k, base_seed, temperature=1.0,
                                     reasoning_effort="none", parser=A.parse_score_json,
                                     max_tokens=W.MAX_TOKENS)
    for m in metas:
        budget.add(STUDENT_ROLE, m.get("cost_usd_reported"))
    if any(v is None for v in vals) or len(vals) != k:
        return {"score": None, "detail": {"parse_failure_reason": "unparseable score"}}
    score = float(vals[0]) if k == 1 else C.winsorize8([float(v) for v in vals])
    return {"score": score, "detail": {"vals": vals, "text0": (texts[0] or "")[:1200]}}


def evaluate(leg, instruction, examples, k, cache, budget, workers, blocks_by_key=None,
             label="", run_id=None):
    """Run one candidate over `examples`. Returns {key: result}.

    Consultation-major with a serial warm-up call per consultation: for the mono leg every
    note of a consultation re-sends the same transcript, so the first call warms the
    provider-side prefix cache for the rest.
    """
    shape = LEGS[leg]["shape"]()
    template = shape.build(instruction)
    tsha = sha(template)[:32]
    out, lock = {}, threading.Lock()

    def one(ex):
        base_seed = RUN_SEED * 100000 + ex["item_index"] * 10
        facts = None
        if leg == "pipeline":
            facts = facts_for(blocks_by_key[(ex["stratum"], ex["consultation"])], budget, run_id)
            ckey = f"{tsha}|{ex['key']}|{base_seed}|k{k}|f{sha(json.dumps(facts))[:8]}"
        else:
            ckey = f"{tsha}|{ex['key']}|{base_seed}|k{k}"
        hit = cache.get(ckey)
        if hit is None:
            hit = (score_pipeline(template, ex, facts, base_seed, budget) if leg == "pipeline"
                   else score_mono(template, ex, k, base_seed, budget))
            cache.put(ckey, hit)
        with lock:
            out[ex["key"]] = hit

    by_consult = {}
    for ex in examples:
        by_consult.setdefault((ex["stratum"], ex["consultation"]), []).append(ex)

    def do_group(group):
        group.sort(key=lambda e: (e["kind"] != "clean", e["key"]))
        one(group[0])
        for ex in group[1:]:
            one(ex)

    groups = [g for _k, g in sorted(by_consult.items())]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(do_group, groups))
    if label:
        log(f"    {label}: {len(out)}/{len(examples)} judged k={k} | cache "
            f"{cache.hits}h/{cache.misses}m | ${budget.total:.2f}")
    return out


# ================================================================ measures
def _sweep_best_at_fa(err, clean, cap=FA_TARGET):
    """(threshold, FA, detection) at the most sensitive cut-off whose false-alarm rate on the
    clean twins is still <= cap. Flag rule `score < threshold`, the convention
    w2_pipeline_analyze uses, so a number here is comparable with one there.

    Invariant to adding a constant to every score and to any monotone relabelling, which is
    the loophole a fixed flag line leaves open: a prompt cannot buy detection by pushing the
    whole score distribution down."""
    if not err or not clean:
        return None, None, None
    # Round the VALUES as well as the candidate thresholds: comparing an unrounded score
    # against a rounded threshold makes the sweep sensitive to the last floating-point bit,
    # which is exactly what breaks the shift-invariance this function is chosen for.
    err = [round(v, 9) for v in err]
    clean = [round(v, 9) for v in clean]
    cand = sorted(set(err) | set(clean))
    best = (None, None, -1.0)
    for t in cand + [cand[-1] + 1e-6]:
        fa = sum(1 for v in clean if v < t) / len(clean)
        if fa <= cap + 1e-12:
            det = sum(1 for v in err if v < t) / len(err)
            if det > best[2] or (det == best[2] and best[1] is not None and fa < best[1]):
                best = (t, fa, det)
    return best if best[0] is not None else (None, None, None)


def _paired(rows):
    """P(errored strictly below its own clean twin) + half the tie mass. Chance 0.500."""
    rows = [(e, c) for e, c in rows if e is not None and c is not None]
    if not rows:
        return None
    return sum(1.0 if e < c else 0.5 if e == c else 0.0 for e, c in rows) / len(rows)


def _paired_full(rows):
    rows = [(e, c) for e, c in rows if e is not None and c is not None]
    if not rows:
        return None
    return {"paired": round(_paired(rows), 4), "n": len(rows),
            "wins": sum(1 for e, c in rows if e < c),
            "ties": sum(1 for e, c in rows if e == c),
            "losses": sum(1 for e, c in rows if e > c)}


def objective(examples, scores):
    """The full read-out vector for one candidate on one split."""
    by_kind, clean_by_key = {}, {}
    for e in examples:
        s = scores.get(e["key"])
        if isinstance(s, dict):
            s = s.get("score")
        if s is None:
            continue
        by_kind.setdefault(e["kind"], []).append((e, s))
        if e["kind"] == "clean":
            clean_by_key[e["key"]] = s

    def vals(kinds):
        return [s for k in kinds for _e, s in by_kind.get(k, [])]

    def paired(kinds, pred=None):
        rows = [(s, clean_by_key.get(e["clean_key"])) for k in kinds
                for e, s in by_kind.get(k, []) if pred is None or pred(e)]
        return _paired([(a, c) for a, c in rows if c is not None])

    clean = vals(("clean",))
    omit = vals(OMIT_KINDS)
    thr, fa, det = _sweep_best_at_fa(omit, clean)

    def det_at(kinds, pred=None):
        rows = [s for k in kinds for e, s in by_kind.get(k, []) if pred is None or pred(e)]
        return (sum(1 for s in rows if thr is not None and s < thr) / len(rows)) if rows else None

    def by_res(level):
        return paired(OMIT_KINDS, lambda e: e.get("residual_level") == level)

    n_pf = sum(1 for e in examples
               if (scores.get(e["key"]) or {}).get("score") is None) if scores else 0
    return {
        "n_judged": sum(len(v) for v in by_kind.values()), "n_parse_failures": n_pf,
        "n_clean": len(clean), "n_omit": len(omit),
        # PRIMARY
        "paired_omit": paired(OMIT_KINDS),
        "paired_omit_detail": _paired_full([(s, clean_by_key.get(e["clean_key"]))
                                            for k in OMIT_KINDS for e, s in by_kind.get(k, [])]),
        # SECONDARY
        "det10": det, "thr10": thr, "fa10": fa,
        "det10_complete": det_at(OMIT_KINDS, lambda e: e.get("residual_level") == "complete"),
        "det10_partial": det_at(OMIT_KINDS,
                                lambda e: (e.get("residual_level") or "").startswith("partial")),
        # conditioning + controls
        "paired_complete": by_res("complete"), "paired_partial_weak": by_res("partial-weak"),
        "paired_partial_strong": by_res("partial-strong"),
        "paired_commissions": paired(("add", "change")),
        "paired_add": paired(("add",)), "paired_change": paired(("change",)),
        "det10_addchange": det_at(("add", "change")),
        "mean_clean": (sum(clean) / len(clean)) if clean else None,
        "mean_omit": (sum(omit) / len(omit)) if omit else None,
    }


def objective_line(v):
    def pct(x):
        return "  n/a" if x is None else f"{100 * x:5.1f}"

    def num(x, f="{:.3f}"):
        return " n/a " if x is None else f.format(x)
    return (f"paired-omit {num(v['paired_omit'])} (cmpl {num(v['paired_complete'])} "
            f"pw {num(v['paired_partial_weak'])} ps {num(v['paired_partial_strong'])}) | "
            f"det@FA<=10% {pct(v['det10'])}% (FA {pct(v['fa10'])}%) | "
            f"comm {num(v['paired_commissions'])}")


def rewards_of(leg, examples, scores):
    """Per-instance reward for the Pareto frontier and for the minibatch statistic.

    Threshold-free and aligned with the objective: an errored note is rewarded for sitting
    BELOW its own clean twin (tie-adjusted) plus a margin term so a near miss is visibly
    nearer; a clean note is rewarded for sitting high. This is what lets a candidate that
    fixes two hard partials and loses on average survive as a parent.
    """
    cfg = LEGS[leg]
    top, margin = cfg["max_score"], cfg["margin"]
    clean_by_key = {e["key"]: (scores.get(e["key"]) or {}).get("score")
                    for e in examples if e["kind"] == "clean"}
    out = {}
    for e in examples:
        s = (scores.get(e["key"]) or {}).get("score")
        if s is None:
            out[e["key"]] = 0.0
            continue
        if e["kind"] == "clean":
            out[e["key"]] = max(0.0, min(1.0, s / top))
            continue
        c = clean_by_key.get(e["clean_key"])
        if c is None:
            out[e["key"]] = max(0.0, min(1.0, 1.0 - s / top))
            continue
        base = 1.0 if s < c else 0.5 if s == c else 0.0
        out[e["key"]] = 0.5 * base + 0.5 * max(0.0, min(1.0, (c - s) / margin))
    return out


def batch_score(leg, examples, scores):
    """The minibatch statistic acceptance runs on: the mean per-instance reward over the
    batch - the same quantity the frontier is built from, so the pre-filter and the frontier
    point the same way."""
    r = rewards_of(leg, examples, scores)
    vals = [r[e["key"]] for e in examples if e["key"] in r]
    return (sum(vals) / len(vals)) if vals else None


def batch_separation(leg, examples, scores):
    """Recorded alongside: tie-adjusted P(an errored note scores below a clean note) over
    every errored x clean cross pair in the batch. Finer-grained than the paired statistic
    on 8 pairs, and threshold-free."""
    clean = [(scores.get(e["key"]) or {}).get("score") for e in examples if e["kind"] == "clean"]
    err = [(scores.get(e["key"]) or {}).get("score") for e in examples if e["kind"] in ERR_KINDS]
    clean = [c for c in clean if c is not None]
    err = [a for a in err if a is not None]
    if not clean or not err:
        return None
    return _paired([(a, c) for a in err for c in clean])


# ================================================================ selection
def pareto_parent(pool, rng):
    """GEPA's selection rule: a candidate stays in contention if it is the best anyone has
    managed on at least one VALID instance, and is sampled in proportion to how many
    instances it leads."""
    keys = sorted({k for c in pool for k in c["rewards"]})
    leads = {c["id"]: 0 for c in pool}
    for k in keys:
        best = max((c["rewards"].get(k, -1.0) for c in pool), default=None)
        if best is None:
            continue
        for c in pool:
            if abs(c["rewards"].get(k, -1.0) - best) < 1e-9:
                leads[c["id"]] += 1
    frontier = [c for c in pool if leads[c["id"]] > 0]
    if not frontier:
        return max(pool, key=lambda c: c["vec"]["paired_omit"] or 0.0), leads
    return rng.choices(frontier, weights=[leads[c["id"]] for c in frontier], k=1)[0], leads


def minibatch(blocks, rng, size=12, n_consult=4):
    """12 notes drawn CONSULTATION-first from REFLECT: each drawn consultation contributes
    its clean twin plus two errored notes, partials preferred.

    Consultation-first rather than note-first for one reason that matters more than the
    stratification: the reflector's sharpest trace is the comparison of an errored note with
    ITS OWN clean twin (leg A can then diff the per-fact verdicts), and a flat draw over a
    pool that is 80% errored routinely fails to contain the matching twin.
    """
    with_partial = [b for b in blocks if any(e["kind"] == "omit-partial" for e in b["examples"])]
    without = [b for b in blocks if b not in with_partial]
    rng.shuffle(with_partial)
    rng.shuffle(without)
    picked_blocks = with_partial[:3] + (with_partial[3:] + without)[:max(0, n_consult - 3)]
    picked = []
    for b in picked_blocks:
        clean = [e for e in b["examples"] if e["kind"] == "clean"]
        errs = [e for e in b["examples"] if e["kind"] in ERR_KINDS]
        part = [e for e in errs if e["kind"] == "omit-partial"]
        rest = [e for e in errs if e["kind"] != "omit-partial"]
        rng.shuffle(part)
        rng.shuffle(rest)
        take = (part[:1] + rest[:1]) if part else rest[:2]
        if len(take) < 2:
            take = (part + rest)[:2]
        picked += clean[:1] + take
    return picked[:size]


# ================================================================ reflection
REFLECT_HEAD = {
    "pipeline": """You are improving one module of a two-stage system that checks whether an \
AI-generated clinical note is missing anything the consultation established.

HOW THE SYSTEM WORKS.
  Stage 1 (FROZEN - not yours to change, and its output is not something you can influence): a \
separate model reads the consultation TRANSCRIPT on its own, with the note hidden from it, and \
writes a numbered list of the facts a correct note of that consultation ought to carry. Typically \
30-45 facts. It is imperfect: some entries are minor, some overlap, some ask for detail a good \
note would reasonably compress.
  Stage 2 (YOURS): a model is shown the clinical NOTE and that numbered fact list - never the \
transcript - and must return one verdict per fact id, "present" or "absent". Your instruction \
block is what tells it how to decide.
  The note's score is then simply the FRACTION OF FACTS marked present. A note is sent for human \
review when its score falls below a cut-off, and that cut-off is chosen afterwards from the data: \
the most sensitive value that still leaves at most 10% of genuinely clean notes flagged.

THE ONE THING THAT WILL NOT WORK. Marking more facts absent across the board buys NOTHING. Every \
note's score moves down together and the cut-off moves down with it. The only thing that helps is \
ORDER: a note that really is missing something must end up with a LOWER fraction than a note that \
is not. That means precision on the individual fact - marking absent exactly those facts the note \
genuinely does not carry, and marking present the ones it does carry in different words.""",
    "mono": """You are improving the instruction block of an LLM judge that reviews AI-generated \
clinical notes against the consultation transcript they came from.

HOW THE JUDGE IS USED. The full prompt is, in this order: the consultation transcript, then the \
clinical note, then YOUR instruction block, then a fixed output contract you do not write and \
cannot change:

--- fixed output contract (do not restate it, do not contradict it) ---
{contract}
--- end fixed output contract ---

HOW IT IS SCORED. The judge returns one 0-10 score per note. A review cut-off is then chosen \
AFTERWARDS from the data: the most sensitive value that still leaves at most 10% of genuinely \
clean notes flagged.

THE ONE THING THAT WILL NOT WORK. Making the judge harsher - lowering every score, penalising \
more things, lowering the bar for a violation - buys NOTHING, because every score moving down \
together simply moves the cut-off down with it. The only thing that improves this judge is \
putting notes in a better ORDER: notes that really are missing something the consultation \
established must score BELOW notes that are faithful and complete."""}

REFLECT_TAIL = """
WHAT COUNTS AS A FAILURE. Two kinds, and either one alone is worthless:
  * an errored note scoring at or above its own clean twin. The errors are omitted content above \
all (the consultation established something and the note does not carry it), plus fabricated and \
altered content. Omissions come in two flavours: the fact is gone entirely (complete), or the \
topic still surfaces somewhere while the load-bearing part of it - the working diagnosis, the \
dose, the reason for urgency, the red flag, the pertinent negative - has been dropped (partial). \
The partial case is far harder and is where this whole family of methods fails, because the \
surviving mention reads like coverage.
  * a clean note scoring low. Clinical notes legitimately compress: paraphrase into clinical \
language, standard abbreviations, conventional normal-exam phrasing, dropped conversational \
colour, and a fact recorded once rather than three times are all correct behaviour, not loss.

THE CURRENT INSTRUCTION BLOCK
=============================
{instruction}
=============================

HOW IT JUST PERFORMED on {n} notes from {m} consultations. Each entry gives the ground truth, what \
the module returned, how the note ranked against its own clean twin, and a diagnosis:

{traces}

YOUR TASK. Write an improved instruction block. Requirements:
  * It REPLACES the block above entirely and is read by a model that sees only its own inputs, \
your block and the fixed contract. It cannot refer to these examples, to this critique, to "the \
previous instruction", or to any consultation you have just been shown - those notes will never be \
seen again. Generalise the lesson; never encode the instance.
  * Aim at the ORDERING failures above, not at overall strictness. If an errored note scored the \
same as its clean twin, the fix is a discrimination the module can actually make on an unseen \
note, not a louder instruction to care more.
  * Keep what is working. The diagnoses mark correct decisions explicitly: whatever produced them \
must survive your rewrite.
  * Do not restate the output contract or the JSON shape - they are appended after your block \
automatically. No placeholders, no {{note}} / {{transcript}} / {{facts_block}} tokens.

Return the new instruction block inside these exact markers and nothing else outside them:
<INSTRUCTION>
...your improved instruction block...
</INSTRUCTION>"""

MERGE_PROMPT = """You are combining two competing instruction blocks for the same module of the \
same system, described below. Both descend from the same original block. Each is the best \
performer on a DIFFERENT set of notes, which is the reason to combine them rather than pick one.

{head}

BLOCK A - currently the best of any candidate on {n_a} of the validation notes
=============================
{a}
=============================

BLOCK B - currently the best of any candidate on {n_b} of the validation notes
=============================
{b}
=============================

WHERE THEY DIFFER IN OUTCOME. These are notes one block ranks correctly against its own clean \
twin and the other does not:

{contrast}

YOUR TASK. Write ONE instruction block that keeps what makes A right where it is right and what \
makes B right where it is right. This is a synthesis, not a concatenation: resolve the conflicts \
between them explicitly rather than including both sides of a contradiction, and keep the result \
readable - a module that is asked to hold twenty competing rules in mind follows none of them. \
Do not restate the output contract or the JSON shape. No placeholders.

Return the new instruction block inside these exact markers and nothing else outside them:
<INSTRUCTION>
...your combined instruction block...
</INSTRUCTION>"""


def _fmt(x, f="{:.3f}"):
    return "n/a" if x is None else f.format(x)


def trace_pipeline(ex, res, twin_res, thr):
    """The per-note trace for leg A. The diff of per-fact verdicts against the note's own
    clean twin is the sharpest signal available anywhere in this study: it says exactly which
    facts the checker noticed going missing and which it did not."""
    s = res.get("score")
    d = res.get("detail") or {}
    v = d.get("verdicts") or {}
    head = f"[{ex['key']}] CLASS={ex['kind']}"
    for f in ("severity", "residual_level", "residual_strength"):
        if ex.get(f):
            head += f" {f}={ex[f]}"
    if s is None:
        return head + " -> PARSE FAILURE: no usable keyed verdict came back."
    n_abs = sum(1 for lab in v.values() if lab == "absent")
    if ex["kind"] == "clean":
        absent = [k for k, lab in sorted(v.items()) if lab == "absent"]
        lines = [f"{head}. Gold: a clean, faithful, COMPLETE note - nothing was removed from it. "
                 f"You marked {n_abs} of {len(v)} facts absent, score {_fmt(s, '{:.4f}')}."]
        if absent:
            lines.append("  Every one of those is a FALSE ABSENCE by construction. The facts you "
                         "said this complete note does not carry: "
                         + "; ".join(f'{k} "{_fact_text(d, k)}"' for k in absent[:8]))
            lines.append("  These are what drives the false-alarm rate, and they also drown out "
                         "the one real absence in an errored note.")
        else:
            lines.append("  No false absences - exactly right.")
        return "\n".join(lines)

    tw = (twin_res or {}).get("score")
    twv = ((twin_res or {}).get("detail") or {}).get("verdicts") or {}
    rank = ("no clean twin available" if tw is None else
            f"its own clean twin scored {_fmt(tw, '{:.4f}')}, so this note ranked "
            + ("BELOW its twin - CORRECT" if s < tw else
               "LEVEL WITH its twin - a tie, which is a failure of ordering" if s == tw
               else "ABOVE its twin - the wrong way round"))
    lines = [f"{head}. Gold: errored. You marked {n_abs} of {len(v)} facts absent, score "
             f"{_fmt(s, '{:.4f}')}. {rank}."]
    if ex["kind"].startswith("omit"):
        lines.append(f'  What was actually removed from this note: "{(ex.get("fact") or "")[:280]}"')
        sites = ex.get("residual") or []
        if sites and sites[0].get("quote"):
            lines.append(f'  The topic still surfaces in the note as: "{sites[0]["quote"][:180]}" '
                         f'(in {sites[0].get("section") or "the note"}) - a '
                         f'{sites[0].get("strength") or "surviving"} mention. A topic that is '
                         f'still named is not the same as a fact that is still recorded.')
        elif ex.get("residual_level") == "complete":
            lines.append("  Nothing of this fact survives anywhere in the note - a complete "
                         "removal, the easiest case there is.")
    else:
        lines.append(f'  What the note gets wrong (fabricated or altered): '
                     f'"{(ex.get("fact") or "")[:280]}"')
    if twv:
        gained = [k for k in sorted(v) if v[k] == "absent" and twv.get(k) == "present"]
        lost = [k for k in sorted(v) if v[k] == "present" and twv.get(k) == "absent"]
        both = [k for k in sorted(v) if v[k] == "absent" and twv.get(k) == "absent"]
        if gained:
            lines.append("  Facts you called absent HERE but present in the clean twin (i.e. the "
                         "difference you did detect): "
                         + "; ".join(f'{k} "{_fact_text(d, k)}"' for k in gained[:6]))
        else:
            lines.append("  You called absent NOTHING here that you called present in the clean "
                         "twin. The removed content did not register at all.")
        if both:
            lines.append(f"  Facts you called absent in BOTH this note and its complete twin "
                         f"({len(both)}): " + ", ".join(both[:10])
                         + " - these are constant noise: they cost false alarms and they hide the "
                           "one difference that matters.")
        if lost:
            lines.append(f"  Facts called present here but absent in the twin ({len(lost)}) - "
                         "inconsistency between two nearly identical notes: " + ", ".join(lost[:6]))
    if thr is not None:
        lines.append(f"  At the current review cut-off ({_fmt(thr, '{:.4f}')}) this note would be "
                     + ("CAUGHT." if s < thr else "MISSED."))
    return "\n".join(lines)


def _fact_text(detail, fid):
    return (detail.get("fact_text") or {}).get(fid, "")[:150]


def trace_mono(ex, res, twin_res, thr):
    s = res.get("score")
    said = ((res.get("detail") or {}).get("text0") or "").strip().replace("\n", " ")
    head = f"[{ex['key']}] CLASS={ex['kind']}"
    for f in ("severity", "residual_level", "residual_strength"):
        if ex.get(f):
            head += f" {f}={ex[f]}"
    if s is None:
        return head + " -> PARSE FAILURE: no usable score came back. Emitted: " + said[:300]
    if ex["kind"] == "clean":
        tw = "CORRECT (above the review cut-off)" if (thr is None or s >= thr) else (
            "FALSE ALARM (at or below the review cut-off, so it would be sent for human review). "
            "It is faithful and complete; the judge penalised ordinary summarisation.")
        return (f"{head}. Gold: a clean, faithful, complete note. Scored {s:.1f}/10. {tw}\n"
                f"  judge said: {said[:380]}")
    tw = (twin_res or {}).get("score")
    rank = ("no clean twin available" if tw is None else
            f"its own clean twin scored {tw:.1f}, so this errored note ranked "
            + ("BELOW its twin - CORRECT" if s < tw else
               "LEVEL WITH its twin - a tie, which is a failure of ordering" if s == tw
               else "ABOVE its twin - the wrong way round"))
    lines = [f"{head}. Gold: errored. Scored {s:.1f}/10. {rank}."]
    lines.append(f'  What the note is missing / gets wrong: "{(ex.get("fact") or "")[:280]}"')
    sites = ex.get("residual") or []
    if sites and sites[0].get("quote"):
        lines.append(f'  The topic still surfaces as: "{sites[0]["quote"][:180]}" (in '
                     f'{sites[0].get("section") or "the note"}) - a '
                     f'{sites[0].get("strength") or "surviving"} mention.')
    elif ex.get("residual_level") == "complete":
        lines.append("  Nothing of this fact survives anywhere in the note.")
    if ex.get("severity_rationale"):
        lines.append(f"  Why it matters: {ex['severity_rationale'][:250]}")
    lines.append(f"  judge said: {said[:380]}")
    return "\n".join(lines)


def build_traces(leg, mb, res, thr):
    by_key = {e["key"]: e for e in mb}
    fn = trace_pipeline if leg == "pipeline" else trace_mono
    out = []
    for e in sorted(mb, key=lambda e: (e["consultation"], e["kind"] != "clean", e["key"])):
        twin = res.get(e["clean_key"]) if e["kind"] != "clean" else None
        out.append(fn(e, res.get(e["key"]) or {}, twin, thr))
    return "\n\n".join(out), len({e["consultation"] for e in mb}), len(by_key)


def _reflect_call(prompt, budget):
    texts, meta = C.llm(prompt, REFLECTION_ROLE, temperature=1.0, k=1,
                        reasoning_effort=REFLECTION_EFFORT, max_tokens=16000, timeout=900)
    budget.add(REFLECTION_ROLE, meta.get("cost_usd_reported"))
    m = re.search(r"<INSTRUCTION>(.*?)</INSTRUCTION>", texts[0], re.S)
    if not m:
        return None, texts[0][:400]
    new = m.group(1).strip()
    for bad in ("{transcript}", "{note}", "{summary}", "{facts_block}"):
        new = new.replace(bad, {"{transcript}": "the transcript", "{note}": "the note",
                                "{summary}": "the note",
                                "{facts_block}": "the fact list"}[bad])
    return (new if len(new) > 200 else None), None


def reflect(leg, instruction, traces, n, m, budget):
    head = REFLECT_HEAD[leg]
    if leg == "mono":
        head = head.replace("{contract}", LEGS["mono"]["shape"]().contract.strip())
    prompt = head + REFLECT_TAIL.replace("{instruction}", instruction).replace(
        "{traces}", traces).replace("{n}", str(n)).replace("{m}", str(m))
    return _reflect_call(prompt, budget)


def merge_candidates(leg, a, b, contrast, n_a, n_b, budget):
    head = REFLECT_HEAD[leg]
    if leg == "mono":
        head = head.replace("{contract}", LEGS["mono"]["shape"]().contract.strip())
    prompt = (MERGE_PROMPT.replace("{head}", head).replace("{a}", a["instruction"])
              .replace("{b}", b["instruction"]).replace("{contrast}", contrast)
              .replace("{n_a}", str(n_a)).replace("{n_b}", str(n_b)))
    return _reflect_call(prompt, budget)


# ================================================================ state
def load_state(leg):
    path = STATE.format(leg=leg)
    if os.path.exists(path):
        return json.load(open(path))
    return {"leg": leg, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_ids": [], "pool": [], "history": [], "next_id": 0, "iteration": 0,
            "n_merges": 0, "stopped": None, "test": {}, "spend_note": None}


def save_state(leg, st):
    path = STATE.format(leg=leg)
    tmp = path + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def _slim(c):
    return {k: v for k, v in c.items() if k not in ("rewards", "scores")}


def new_run(leg, params):
    return C.Run(EXPERIMENT, params={"leg": leg, "leg_what": LEGS[leg]["what"], **params},
                 replicate=1, seed=RUN_SEED,
                 inputs=["master/dataset_v2.json", "master/arms_confirm_subset.json",
                         "gepa/v3_partition.json"],
                 spec=SPEC, allow_dirty=True)


def make_budget(args, st):
    prior, led, _side = experiment_spend()
    leg_prior = sum(max(led.get(r, 0.0), 0.0) for r in st["run_ids"])
    return Budget(args.cap, args.stop, prior, leg_cap=args.leg_cap, leg_prior=leg_prior)


# ================================================================ phases
def blocks_index(blocks):
    out = {}
    for b in blocks:
        out[(b["stratum"], b["consultation"])] = b
    return out


def _attach(examples, blocks):
    by = {}
    for b in blocks:
        b["examples"] = []
        by[(b["stratum"], b["consultation"])] = b
    for e in examples:
        by[(e["stratum"], e["consultation"])]["examples"].append(e)
    return list(by.values())


def phase_prep(args, leg, st):
    """Leg A only: extraction for every TRAIN consultation not already in w2_pipeline's
    cache. One call per consultation, cached to disk, shared with the pipeline judge's runs."""
    _ex, blocks, _m = D.load_split("train")
    cache = P.StageCache("facts")
    todo = [b for b in blocks if not cache.get(P.cache_key(b))][:args.n if args.n else None]
    log(f"  extraction: {len(todo)} consultations to buy "
        f"(~${0.0868 * len(todo):.2f} at the measured extraction rate)")
    if not todo:
        return
    budget = make_budget(args, st)
    with new_run(leg, {"phase": "prep", "n_consultations": len(todo)}) as run:
        st["run_ids"].append(run.run_id)
        budget.bind(run.run_id)
        save_state(leg, st)
        prompts = W.load_prompts(["pipeline_extract"], quiet=True)
        run.register_prompts(prompts)
        errs = []

        def one(b):
            try:
                facts_for(b, budget, run.run_id, prompts)
            except Exception as ex:
                errs.append((f"{b['stratum']}/{b['consultation']}", str(ex)[:200]))
        try:
            with ThreadPoolExecutor(max_workers=min(args.workers, 4)) as pool:
                list(pool.map(one, todo))
        except BudgetStop:
            log("  ! budget stop during extraction")
        finally:
            budget._flush()
        for name, why in errs:
            log(f"  ! extraction failed {name}: {why}")
        log(f"  extraction done: {len(todo) - len(errs)} bought, ${budget.spent:.2f} this run")
    save_state(leg, st)


def _eval_valid(leg, instruction, st, cache, budget, args, label):
    ex, blocks, _m = D.load_split("valid")
    bidx = blocks_index(blocks)
    res = evaluate(leg, instruction, ex, LEGS[leg]["k_search"], cache, budget, args.workers,
                   bidx, label, run_id=None)
    return res, objective(ex, res), rewards_of(leg, ex, res), ex


def phase_seeds(args, leg, st, cache):
    done = {c["name"] for c in st["pool"]}
    todo = [(n, i) for n, i in seed_instructions(leg) if n not in done]
    if not todo:
        log("  all seeds already evaluated")
        return
    budget = make_budget(args, st)
    with new_run(leg, {"phase": "seeds", "seeds": [n for n, _ in todo],
                       "k": LEGS[leg]["k_search"]}) as run:
        st["run_ids"].append(run.run_id)
        budget.bind(run.run_id)
        save_state(leg, st)
        try:
            for name, instr in todo:
                res, vec, rew, _ex = _eval_valid(leg, instr, st, cache, budget, args,
                                                 f"seed {name}")
                cand = {"id": st["next_id"], "name": name, "kind": "seed", "parent": None,
                        "iter": 0, "instruction": instr, "vec": vec, "rewards": rew,
                        "scores": {k: v.get("score") for k, v in res.items()},
                        "chars": len(instr)}
                st["pool"].append(cand)
                st["next_id"] += 1
                log(f"  seed #{cand['id']} {name:<22} {objective_line(vec)}")
                lineage({"event": "candidate", "leg": leg, "run_id": run.run_id, **_slim(cand)})
                save_state(leg, st)
                why = budget.should_stop()
                if why:
                    st["stopped"] = why
                    break
        except BudgetStop:
            st["stopped"] = "hard cap hit mid-evaluation"
            log("  ! hard cap hit mid-evaluation - state saved, cache is warm")
        finally:
            budget._flush()
            cache.close()
    save_state(leg, st)


def _contrast_lines(leg, a, b, examples, limit=8):
    """Notes one candidate ranks correctly against its own twin and the other does not - the
    evidence the merge prompt is built on."""
    lines, na, nb = [], 0, 0
    for e in examples:
        if e["kind"] not in ERR_KINDS:
            continue
        sa, sb = a["scores"].get(e["key"]), b["scores"].get(e["key"])
        ca, cb = a["scores"].get(e["clean_key"]), b["scores"].get(e["clean_key"])
        if None in (sa, sb, ca, cb):
            continue
        wa, wb = sa < ca, sb < cb
        if wa == wb:
            continue
        na += wa
        nb += wb
        if len(lines) < limit:
            lines.append(f"  [{e['kind']}"
                         + (f"/{e['residual_level']}" if e.get("residual_level") else "")
                         + f"] {'A' if wa else 'B'} ranks it correctly, "
                           f"{'B' if wa else 'A'} does not. Removed: "
                           f"\"{(e.get('fact') or '')[:150]}\"")
    return ("\n".join(lines) or "  (no clean contrast - they differ only in margin)"), na, nb


def phase_iterate(args, leg, st, cache):
    budget = make_budget(args, st)
    why = budget.should_stop()
    if why or st["iteration"] >= args.iterations:
        st["stopped"] = why or "iterations exhausted"
        save_state(leg, st)
        log(f"  STOP: {st['stopped']}")
        return
    rex, rblocks, _m = D.load_split("reflect")
    rblocks = _attach(rex, rblocks)
    ridx = blocks_index(rblocks)
    vex, vblocks, _m2 = D.load_split("valid")
    vidx = blocks_index(vblocks)
    with new_run(leg, {"phase": "iterate", "from_iter": st["iteration"] + 1, "n": args.n,
                       "k": LEGS[leg]["k_search"], "minibatch": args.minibatch}) as run:
        st["run_ids"].append(run.run_id)
        budget.bind(run.run_id)
        save_state(leg, st)
        try:
            for _ in range(args.n):
                if st["iteration"] >= args.iterations:
                    st["stopped"] = "iterations exhausted"
                    break
                why = budget.should_stop()
                if why:
                    st["stopped"] = why
                    break
                _one_iteration(args, leg, st, cache, budget, run, rblocks, ridx, vex, vidx)
                save_state(leg, st)
        except BudgetStop:
            st["stopped"] = f"hard cap hit mid-evaluation (${budget.total:.2f})"
            log("  ! hard cap hit mid-evaluation - state saved, cache is warm")
        finally:
            budget._flush()
            cache.close()
    if st.get("stopped"):
        log(f"  STOP: {st['stopped']}")
    save_state(leg, st)


def _one_iteration(args, leg, st, cache, budget, run, rblocks, ridx, vex, vidx):
    it = st["iteration"] + 1
    rng = random.Random(args.seed + 1013 * it + (0 if leg == "pipeline" else 77))
    pool = st["pool"]
    parent, leads = pareto_parent(pool, rng)
    do_merge = (args.merge and len(pool) >= 6 and st["n_merges"] < args.max_merges
                and rng.random() < 0.4)

    mb = minibatch(rblocks, rng, args.minibatch)
    p_res = evaluate(leg, parent["instruction"], mb, LEGS[leg]["k_search"], cache, budget,
                     args.workers, ridx)
    p_stat = batch_score(leg, mb, p_res)
    thr = parent["vec"].get("thr10")

    if do_merge:
        frontier = sorted([c for c in pool if leads[c["id"]] > 0],
                          key=lambda c: -leads[c["id"]])[:4]
        pick = None
        for i in range(len(frontier)):
            for j in range(i + 1, len(frontier)):
                lines, na, nb = _contrast_lines(leg, frontier[i], frontier[j], vex)
                if na and nb:
                    pick = (frontier[i], frontier[j], lines, na, nb)
                    break
            if pick:
                break
        if pick is None:
            do_merge = False
        else:
            a, b, lines, na, nb = pick
            parent = a
            new_instr, err = merge_candidates(leg, a, b, lines, leads[a["id"]], leads[b["id"]],
                                              budget)
            st["n_merges"] += 1
            kind, origin = "merge", f"#{a['id']}+#{b['id']}"
            log(f"  it{it:02d} MERGE #{a['id']}({a['name']}) x #{b['id']}({b['name']}) "
                f"| A wins {na} / B wins {nb} discordant notes")
    if not do_merge:
        traces, m, n = build_traces(leg, mb, p_res, thr)
        new_instr, err = reflect(leg, parent["instruction"], traces, n, m, budget)
        kind, origin = "mutation", f"#{parent['id']}"

    st["iteration"] = it
    rec = {"iter": it, "leg": leg, "kind": kind, "origin": origin, "parent": parent["id"],
           "parent_leads": leads[parent["id"]], "mb_parent": round(p_stat or 0, 5)}
    if new_instr is None:
        st["history"].append({**rec, "outcome": "no_block", "why": (err or "")[:200]})
        log(f"  it{it:02d} {kind} from {origin} -> reflection produced no block")
        lineage({"event": "mutation_failed", **rec, "why": (err or "")[:200]})
        return
    if any(sha(new_instr) == sha(c["instruction"]) for c in pool):
        st["history"].append({**rec, "outcome": "duplicate"})
        log(f"  it{it:02d} {kind} from {origin} -> duplicate instruction, skipped")
        lineage({"event": "mutation_duplicate", **rec})
        return

    c_res = evaluate(leg, new_instr, mb, LEGS[leg]["k_search"], cache, budget, args.workers, ridx)
    c_stat = batch_score(leg, mb, c_res)
    rec.update({"mb_child": round(c_stat or 0, 5), "chars": len(new_instr),
                "instruction_sha256": sha(new_instr),
                "mb_sep_parent": round(batch_separation(leg, mb, p_res) or 0, 4),
                "mb_sep_child": round(batch_separation(leg, mb, c_res) or 0, 4)})
    accepted = c_stat is not None and p_stat is not None and c_stat > p_stat + 1e-9
    log(f"  it{it:02d} {kind} from {origin} (leads {leads[parent['id']]}/{len(vex)}) "
        f"minibatch {p_stat:.4f} -> {c_stat:.4f} {'ACCEPT' if accepted else 'reject'} "
        f"| ${budget.total:.2f}")
    if not accepted:
        st["history"].append({**rec, "outcome": "rejected_minibatch"})
        lineage({"event": "rejected_minibatch", **rec})
        return

    # PROMISCUOUS acceptance: the minibatch alone admits it. VALID is then evaluated so the
    # frontier has per-instance rewards for it - the frontier, not the accept rule, is what
    # prunes a child that only looked good on 12 notes.
    res, vec, rew, _ex = _eval_valid(leg, new_instr, st, cache, budget, args,
                                     f"it{it:02d} VALID")
    cand = {"id": st["next_id"], "name": f"v3{leg[0]}-{st['next_id']:02d}", "kind": kind,
            "parent": parent["id"], "origin": origin, "iter": it, "instruction": new_instr,
            "vec": vec, "rewards": rew, "scores": {k: v.get("score") for k, v in res.items()},
            "chars": len(new_instr)}
    st["pool"].append(cand)
    st["next_id"] += 1
    pv = parent["vec"]
    log(f"       VALID: {objective_line(vec)}")
    log(f"       vs parent#{parent['id']}: paired {_fmt(pv['paired_omit'])} -> "
        f"{_fmt(vec['paired_omit'])}, det10 {100 * (pv['det10'] or 0):.1f}% -> "
        f"{100 * (vec['det10'] or 0):.1f}% -> accepted as #{cand['id']} {cand['name']}")
    st["history"].append({**rec, "outcome": "accepted", "id": cand["id"],
                          "valid_paired": vec["paired_omit"], "valid_det10": vec["det10"]})
    lineage({"event": "candidate", "leg": leg, "run_id": run.run_id, **_slim(cand)})


# ================================================================ TEST
def winner_of(st):
    """Best VALID aggregate: primary paired discrimination on omissions, tie-broken on the
    secondary (detection at FA <= 10%)."""
    if not st["pool"]:
        return None
    return sorted(st["pool"], key=lambda c: (-(c["vec"]["paired_omit"] or 0),
                                             -(c["vec"]["det10"] or 0), c["id"]))[0]


def baseline_of(st, leg):
    name = "seed_check_binary" if leg == "pipeline" else "seed_fc_score"
    return next((c for c in st["pool"] if c["name"] == name), None)


def phase_test(args, leg, st, cache):
    """The winner, ONCE, on the held-out TEST subset. Nothing else in this file ever touches
    it."""
    win = winner_of(st)
    base = baseline_of(st, leg)
    if win is None:
        log("  no candidate to test")
        return
    if leg == "mono":
        gate = gate_decision(st)
        st["gate"] = gate
        log(f"  GATE: {gate['decision']} - {gate['why']}")
        if not gate["passed"] and not args.force_test:
            save_state(leg, st)
            return
    if win["name"] == (base or {}).get("name") and not args.force_test:
        log(f"  the VALID winner is the baseline seed itself ({win['name']}) - the search "
            f"produced nothing to confirm; no TEST spend.")
        st["test"]["skipped"] = "winner is the seed baseline"
        save_state(leg, st)
        return
    ex, blocks, meta = D.load_split("test")
    bidx = blocks_index(blocks)
    k = LEGS[leg]["k_test"]
    budget = make_budget(args, st)
    todo = [c for c in ([win] + ([base] if args.test_baseline and base else []))
            if str(c["id"]) not in st["test"]]
    with new_run(leg, {"phase": "test", "k": k, "split": "test",
                       "candidates": [c["name"] for c in todo],
                       "test_file": meta["source"], "test_sha256": meta["sha256"]}) as run:
        st["run_ids"].append(run.run_id)
        budget.bind(run.run_id)
        save_state(leg, st)
        try:
            for c in todo:
                res = evaluate(leg, c["instruction"], ex, k, cache, budget, args.workers, bidx,
                               f"TEST {c['name']} k={k}", run_id=run.run_id)
                vec = objective(ex, res)
                st["test"][str(c["id"])] = {
                    "name": c["name"], "kind": c["kind"], "iter": c["iter"], "k": k, "vec": vec,
                    "scores": {kk: v.get("score") for kk, v in res.items()}}
                log(f"  TEST {c['name']:<22} {objective_line(vec)}")
                lineage({"event": "test", "leg": leg, "run_id": run.run_id, "name": c["name"],
                         "k": k, "vec": vec})
                save_state(leg, st)
        except BudgetStop:
            log("  ! hard cap hit during the TEST run")
        finally:
            budget._flush()
            cache.close()
    save_state(leg, st)


def gate_decision(st):
    """Leg B's pre-registered gate: nothing is bought on TEST unless the winner clears the
    FC-score seed on VALID by more than 2pp on BOTH measures."""
    win, base = winner_of(st), baseline_of(st, "mono")
    if win is None or base is None:
        return {"passed": False, "decision": "no candidate", "why": "pool incomplete"}
    dp = (win["vec"]["paired_omit"] or 0) - (base["vec"]["paired_omit"] or 0)
    dd = (win["vec"]["det10"] or 0) - (base["vec"]["det10"] or 0)
    ok = win["kind"] != "seed" and dp > 0.02 and dd > 0.02
    return {"passed": bool(ok), "winner": win["name"], "baseline": base["name"],
            "delta_paired_pp": round(100 * dp, 2), "delta_det10_pp": round(100 * dd, 2),
            "threshold_pp": 2.0,
            "decision": "confirm on TEST at k=8" if ok else "no candidate met the gate",
            "why": (f"{win['name']} beats {base['name']} on VALID by "
                    f"{100 * dp:+.1f}pp paired and {100 * dd:+.1f}pp det@FA<=10%; the gate "
                    f"needs >2pp on both" + ("" if ok else " - not met"))}


# ================================================================ baselines on TEST
def test_baseline_pipeline():
    """The B2 baseline on TEST, free, out of the pipeline judge's own record store."""
    path = os.path.join(C.RESULTS, "w2-pipeline", "_state", "confirm-B2.jsonl")
    if not os.path.exists(path):
        return None
    scores, seeds = {}, {}
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        scores[r["note_key"]] = r.get("aggregate")
        seeds[r["note_key"]] = r.get("base_seed")
    return {"source": "results/w2-pipeline/_state/confirm-B2.jsonl", "scores": scores,
            "base_seeds": seeds, "n": len(scores)}


def test_baseline_grid(cell="FC-score-k8", replicate=None):
    """The grid's cell on TEST, free, restricted to the TEST pairs out of grid-main2.jsonl."""
    path = os.path.join(C.RESULTS, "w2-ablation", "_state", "grid-main2.jsonl")
    if not os.path.exists(path):
        return None
    ex, _b, _m = D.load_split("test")
    want = {e["key"] for e in ex}
    vals = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("cell") != cell or r["note_key"] not in want:
            continue
        if replicate and r.get("replicate") != replicate:
            continue
        vals.setdefault(r["note_key"], []).append(r.get("aggregate"))
    scores = {k: (sum(v) / len(v)) if all(x is not None for x in v) else None
              for k, v in vals.items()}
    return {"source": f"results/w2-ablation/_state/grid-main2.jsonl [{cell}"
                      + (f", r{replicate}]" if replicate else ", mean of replicates]"),
            "cell": cell, "replicate": replicate, "scores": scores, "n": len(scores),
            "n_missing": len(want) - len(scores)}


# ================================================================ statistics
def _sign_test(wins, losses):
    n = wins + losses
    if not n:
        return {"n_informative": 0, "p": 1.0}
    from scipy import stats
    return {"n_informative": n, "wins": wins, "losses": losses,
            "p": round(float(stats.binomtest(wins, n, 0.5).pvalue), 8)}


def _wilson(k, n, z=1.959963984540054):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": k, "n": n, "p": round(p, 4), "lo": round(max(0.0, c - h), 4),
            "hi": round(min(1.0, c + h), 4)}


def _two_prop_z(k1, n1, k2, n2):
    if not n1 or not n2:
        return None
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if not se:
        return None
    from scipy import stats
    z = (k1 / n1 - k2 / n2) / se
    return {"z": round(z, 3), "p": round(float(2 * (1 - stats.norm.cdf(abs(z)))), 6)}


def compare_on_test(examples, cand_scores, base_scores, n_boot=4000):
    """Winner vs baseline on TEST, on both measures, with the tests this study's conventions ask
    for: an exact sign test on the paired wins, an exact McNemar on the flag decisions at each
    side's own FA <= 10% operating point, and a cluster bootstrap over CONSULTATIONS (the unit
    of independence) on the paired delta."""
    cln = {e["key"]: e for e in examples if e["kind"] == "clean"}
    omis = [e for e in examples if e["kind"] in OMIT_KINDS]

    def pack(scores):
        clean = [scores.get(k) for k in cln if scores.get(k) is not None]
        rows = [(e, scores.get(e["key"]), scores.get(e["clean_key"])) for e in omis]
        rows = [(e, a, c) for e, a, c in rows if a is not None and c is not None]
        pd = _paired_full([(a, c) for _e, a, c in rows])
        err = [a for _e, a, _c in rows]
        thr, fa, det = _sweep_best_at_fa(err, clean)
        return {"paired": pd, "rows": rows, "clean": clean, "thr": thr, "fa": fa, "det": det,
                "n_clean": len(clean)}

    ca, ba = pack(cand_scores), pack(base_scores)
    keys = [e["key"] for e, _a, _c in ca["rows"]] if ca["rows"] else []
    common = [e for e in omis if cand_scores.get(e["key"]) is not None
              and base_scores.get(e["key"]) is not None
              and cand_scores.get(e["clean_key"]) is not None
              and base_scores.get(e["clean_key"]) is not None]

    def win(scores, e):
        a, c = scores[e["key"]], scores[e["clean_key"]]
        return 1.0 if a < c else 0.5 if a == c else 0.0

    cw = [win(cand_scores, e) for e in common]
    bw = [win(base_scores, e) for e in common]
    disc_c = sum(1 for x, y in zip(cw, bw) if x > y)
    disc_b = sum(1 for x, y in zip(cw, bw) if x < y)
    # cluster bootstrap over consultations on the paired delta
    by_c = {}
    for i, e in enumerate(common):
        by_c.setdefault(e["consultation"], []).append(i)
    ids, rng, diffs = list(by_c), random.Random(RUN_SEED), []
    for _ in range(n_boot if common else 0):
        idx = [i for c in rng.choices(ids, k=len(ids)) for i in by_c[c]]
        diffs.append(sum(cw[i] for i in idx) / len(idx) - sum(bw[i] for i in idx) / len(idx))
    diffs.sort()

    def flags(pk, scores):
        thr = pk["thr"]
        return {e["key"]: (scores[e["key"]] < thr) for e in common if thr is not None}
    fc, fb = flags(ca, cand_scores), flags(ba, base_scores)
    mc_c = sum(1 for k in fc if fc[k] and not fb.get(k))
    mc_b = sum(1 for k in fc if fb.get(k) and not fc[k])
    n = mc_c + mc_b
    mcp = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, x) for x in
                                              range(0, min(mc_c, mc_b) + 1)) / 2 ** n)
    return {
        "n_omissions_compared": len(common), "n_consultations": len(ids),
        "candidate": {"paired": ca["paired"], "det10": ca["det"], "fa10": ca["fa"],
                      "thr10": ca["thr"], "n_clean": ca["n_clean"],
                      "det_ci": _wilson(round((ca["det"] or 0) * len(ca["rows"])),
                                        len(ca["rows"])),
                      "det_vs_fa_z": _two_prop_z(round((ca["det"] or 0) * len(ca["rows"])),
                                                 len(ca["rows"]),
                                                 round((ca["fa"] or 0) * ca["n_clean"]),
                                                 ca["n_clean"])},
        "baseline": {"paired": ba["paired"], "det10": ba["det"], "fa10": ba["fa"],
                     "thr10": ba["thr"], "n_clean": ba["n_clean"],
                     "det_ci": _wilson(round((ba["det"] or 0) * len(ba["rows"])),
                                       len(ba["rows"])),
                     "det_vs_fa_z": _two_prop_z(round((ba["det"] or 0) * len(ba["rows"])),
                                                len(ba["rows"]),
                                                round((ba["fa"] or 0) * ba["n_clean"]),
                                                ba["n_clean"])},
        "delta_paired": round((ca["paired"]["paired"] if ca["paired"] else 0)
                              - (ba["paired"]["paired"] if ba["paired"] else 0), 4),
        "delta_det10_pp": round(100 * ((ca["det"] or 0) - (ba["det"] or 0)), 2),
        "sign_test_candidate_vs_baseline": _sign_test(disc_c, disc_b),
        "cluster_bootstrap_95ci_paired": ([round(diffs[int(.025 * len(diffs))], 4),
                                           round(diffs[int(.975 * len(diffs))], 4)]
                                          if diffs else None),
        "mcnemar_flags": {"candidate_only": mc_c, "baseline_only": mc_b,
                          "exact_p": round(mcp, 5)},
        "n_keys": len(keys)}


def fabrication_check(examples, scores):
    """The add/change control, free with the TEST run: a coverage score has nothing to say
    about a fabricated sentence, and B3, the full pipeline tier, scored `add` notes ABOVE
    their own twins."""
    out = {}
    for kind in ("add", "change"):
        rows = [(scores.get(e["key"]), scores.get(e["clean_key"]))
                for e in examples if e["kind"] == kind]
        out[kind] = _paired_full(rows)
    rows = [(scores.get(e["key"]), scores.get(e["clean_key"]))
            for e in examples if e["kind"] in ("add", "change")]
    out["commissions"] = _paired_full(rows)
    return out


# ================================================================ report
def _artifact_decision(block):
    """Write w2_prompts/pipeline_check_v3.txt, or say precisely why not."""
    t = block.get("test") or {}
    cands = list((t.get("candidates") or {}).values())
    if not cands:
        return {"written": False, "why": "no TEST run"}
    c = cands[0]
    v, b = c["vec"], c["baseline_vec"]
    beats_primary = (v["paired_omit"] or 0) > (b["paired_omit"] or 0) + 1e-9
    beats_secondary = (v["det10"] or 0) > (b["det10"] or 0) + 1e-9
    dec = {"written": bool(beats_primary), "candidate": c["name"],
           "beats_baseline_primary_paired": beats_primary,
           "beats_baseline_secondary_det10": beats_secondary,
           "delta_paired": round((v["paired_omit"] or 0) - (b["paired_omit"] or 0), 4),
           "delta_det10_pp": round(100 * ((v["det10"] or 0) - (b["det10"] or 0)), 2),
           "path": os.path.relpath(WINNER_PROMPT, ROOT)}
    if beats_primary:
        st = json.load(open(STATE.format(leg="pipeline")))
        instr = next(x["instruction"] for x in st["pool"] if x["name"] == c["name"])
        open(WINNER_PROMPT, "w").write(LEGS["pipeline"]["shape"]().build(instr))
        log(f"  wrote {dec['path']} - the v3 winner beat the B2 baseline on held-out TEST")
    else:
        if os.path.exists(WINNER_PROMPT):
            os.remove(WINNER_PROMPT)
        dec["why"] = ("the winner does not beat the B2 baseline on the primary measure on "
                      f"held-out TEST ({dec['delta_paired']:+.4f} paired), so no v3 prompt "
                      "file is written - only v3_results.json records what the search found")
        log(f"  {dec['path']} deliberately NOT written: {dec['why']}")
    return dec


def write_results(args):
    out = {"arm": "third optimisation campaign - canonical GEPA on a fair inner split",
           "spec": SPEC, "experiment": EXPERIMENT,
           "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "partition": json.load(open(D.PARTITION)) if os.path.exists(D.PARTITION) else None,
           "student_model": C.resolve_model(STUDENT_ROLE)["resolved"],
           "reflection_model": C.resolve_model(REFLECTION_ROLE)["resolved"],
           "reflection_effort": REFLECTION_EFFORT,
           "objective": {
               "primary": "paired tie-adjusted discrimination on omissions, on VALID",
               "secondary": "absolute detection at the best threshold with FA <= 10% on "
                            "VALID's clean twins",
               "acceptance": "PROMISCUOUS - a child enters the pool on a minibatch "
                             "improvement in mean per-instance reward; the per-instance "
                             "Pareto frontier prunes afterwards",
               "selection": "per-instance Pareto frontier on VALID, parents sampled in "
                            "proportion to instances led",
               "merge": "reflector-mediated crossover of two frontier candidates leading on "
                        "disjoint instances (see module docstring: the paper's module-swap "
                        "is undefined for a one-module system)"},
           "legs": {}}
    spend, led, side = experiment_spend()
    for leg in ("pipeline", "mono"):
        path = STATE.format(leg=leg)
        if not os.path.exists(path):
            continue
        st = json.load(open(path))
        win, base = winner_of(st), baseline_of(st, leg)
        ex, _b, meta = D.load_split("test")
        block = {
            "what": LEGS[leg]["what"], "k_search": LEGS[leg]["k_search"],
            "k_test": LEGS[leg]["k_test"],
            "trajectory": {
                "iterations_run": st["iteration"], "stopped_because": st.get("stopped"),
                "n_candidates": len(st["pool"]),
                "n_seeds": sum(1 for c in st["pool"] if c["kind"] == "seed"),
                "n_accepted_mutations": sum(1 for c in st["pool"] if c["kind"] == "mutation"),
                "n_accepted_merges": sum(1 for c in st["pool"] if c["kind"] == "merge"),
                "n_merge_invocations": st["n_merges"],
                "outcomes": {o: sum(1 for h in st["history"] if h.get("outcome") == o)
                             for o in sorted({h.get("outcome") for h in st["history"]})},
                "winner_appeared_at_iteration": (win or {}).get("iter"),
                "history": st["history"]},
            "valid": {"baseline": {"name": (base or {}).get("name"),
                                   "vec": (base or {}).get("vec")},
                      "winner": {"name": (win or {}).get("name"), "kind": (win or {}).get("kind"),
                                 "iter": (win or {}).get("iter"), "vec": (win or {}).get("vec")},
                      "all_candidates": [{"id": c["id"], "name": c["name"], "kind": c["kind"],
                                          "iter": c["iter"], "parent": c.get("parent"),
                                          "chars": c["chars"], "vec": c["vec"]}
                                         for c in st["pool"]]},
            "run_ids": st["run_ids"],
            "spend_usd": round(sum(led.get(r, 0.0) for r in st["run_ids"]), 4),
        }
        # The VALID gain, tested rather than quoted. VALID carries 48 omissions and 20 clean
        # twins, so a several-point difference there is well inside sampling noise - and
        # saying so needs no extra spend, because every candidate's per-note scores are
        # already on disk.
        if win and base and win["id"] != base["id"]:
            vex, _vb, _vm = D.load_split("valid")
            block["valid"]["winner_vs_baseline"] = compare_on_test(vex, win["scores"],
                                                                   base["scores"])
        if leg == "mono":
            block["gate"] = st.get("gate") or gate_decision(st)
        tests = st.get("test") or {}
        if tests and any(k.isdigit() for k in tests):
            # The primary comparator is a SINGLE run of the baseline, because the candidate
            # is a single run: the grid's replicate 1 (its seed-11 run, the same run seed
            # this harness uses). The other two replicates and their mean are recorded
            # beside it so the read-out can say how much of any gap is run-to-run noise.
            base_pack = (test_baseline_pipeline() if leg == "pipeline"
                         else test_baseline_grid("FC-score-k8", replicate=1))
            block["test"] = {"n_examples": len(ex), "source": meta["source"],
                             "sha256": meta["sha256"],
                             "baseline_source": (base_pack or {}).get("source"),
                             "candidates": {}}
            if leg == "mono":
                block["test"]["baseline_replicates"] = {}
                for rep in (1, 2, 3, None):
                    bp = test_baseline_grid("FC-score-k8", replicate=rep)
                    if bp:
                        block["test"]["baseline_replicates"][str(rep or "mean")] = {
                            "n": bp["n"], "vec": objective(
                                ex, {k: {"score": v} for k, v in bp["scores"].items()})}
            for cid, t in tests.items():
                if not cid.isdigit():
                    continue
                entry = {"name": t["name"], "kind": t["kind"], "k": t["k"], "vec": t["vec"],
                         "fabrication_check": fabrication_check(ex, t["scores"])}
                if base_pack:
                    entry["vs_baseline"] = compare_on_test(ex, t["scores"], base_pack["scores"])
                    entry["baseline_vec"] = objective(
                        ex, {k: {"score": v} for k, v in base_pack["scores"].items()})
                block["test"]["candidates"][cid] = entry
        # The artifact rule, v2's discipline kept: a prompt file is written ONLY when the
        # winner beat its baseline on the PRIMARY measure on held-out TEST data. Anything
        # else on disk under a v3 name would falsely imply that v3 found something.
        if leg == "pipeline":
            block["artifact"] = _artifact_decision(block)
        out["legs"][leg] = block
    out["spend"] = {"experiment_total_usd": spend, "by_run_id": led,
                    "sidecar_in_flight": side,
                    "cap_usd": args.cap, "stop_usd": args.stop}
    json.dump(out, open(RESULTS_JSON, "w"), indent=1)
    log(f"  wrote {os.path.relpath(RESULTS_JSON, ROOT)} | experiment spend ${spend:.2f}")
    return out


# ================================================================ selftest
def selftest():
    for leg in LEGS:
        shape = LEGS[leg]["shape"]()
        raw = open(shape.path).read()
        assert shape.build(shape.instruction) == raw
        print(f"  {leg}: shape round-trips, instruction {len(shape.instruction)} chars, "
              f"contract {len(shape.contract)} chars")
    ei = engineered_instruction()
    assert len(ei) > 3000 and "{transcript}" not in ei and "{note}" not in ei
    p = A.render_arm(LEGS["mono"]["shape"]().build(ei), "T", "N")
    assert "{" not in p.split("Score:")[0][-200:]
    # det@FA is invariant to a uniform shift and to a monotone rescaling
    clean, err = [0.99, 0.98, 0.97, 1.0, 0.95], [0.9, 0.98, 0.7, 0.6]
    t, fa, det = _sweep_best_at_fa(err, clean, 0.25)
    t2, fa2, det2 = _sweep_best_at_fa([e - 0.3 for e in err], [c - 0.3 for c in clean], 0.25)
    assert abs(det - det2) < 1e-12 and abs(fa - fa2) < 1e-12
    t3, fa3, det3 = _sweep_best_at_fa([e * 2 for e in err], [c * 2 for c in clean], 0.25)
    assert abs(det - det3) < 1e-12 and abs(fa - fa3) < 1e-12
    assert _paired([(1, 2), (2, 2), (3, 2)]) == 0.5
    # the partition, and the seeds the objective is computed over
    for split in ("reflect", "valid", "test"):
        ex, blocks, meta = D.load_split(split)
        assert ex and blocks
        print(f"  {split}: {len(ex)} notes / {len(blocks)} consultations {meta['by_kind']}")
    vex, _vb, _ = D.load_split("valid")
    rex, rblocks, _ = D.load_split("reflect")
    assert not ({e["consultation"] for e in vex} & {e["consultation"] for e in rex})
    tex, _tb, _ = D.load_split("test")
    assert not ({e["consultation"] for e in tex} & {e["consultation"] for e in rex + vex})
    rb = _attach(rex, rblocks)
    for trial in range(20):
        mb = minibatch(rb, random.Random(trial), 12)
        nc = sum(1 for e in mb if e["kind"] == "clean")
        npart = sum(1 for e in mb if e["kind"] == "omit-partial")
        assert len(mb) == 12 and nc >= 3 and npart >= 3, (len(mb), nc, npart)
        assert all(e["clean_key"] in {x["key"] for x in mb} for e in mb)
    print(f"  minibatch: 12 notes, >=3 clean, >=3 partial, every twin present (20 draws)")
    # the TEST seed formula must reproduce the B2 baseline's seeds exactly, or the TEST
    # comparison is not a matched one
    bp = test_baseline_pipeline()
    if bp:
        mism = [e["key"] for e in tex
                if bp["base_seeds"].get(e["key"]) not in (None, RUN_SEED * 100000
                                                          + e["item_index"] * 10)]
        assert not mism, f"seed mismatch vs confirm-B2 on {len(mism)} notes, e.g. {mism[:3]}"
        cov = sum(1 for e in tex if e["key"] in bp["scores"])
        print(f"  TEST seed parity with confirm-B2: exact on {cov}/{len(tex)} notes")
    bg = test_baseline_grid("FC-score-k8")
    if bg:
        print(f"  TEST grid baseline FC-score-k8: {bg['n']} notes "
              f"({bg['n_missing']} missing)")
    spend, led, side = experiment_spend()
    print(f"selftest OK | experiment spend so far ${spend:.2f} over {len(led)} ledger rows")


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leg", choices=sorted(LEGS), default="pipeline")
    ap.add_argument("--phase", choices=["prep", "seeds", "iterate", "test", "report", "status"],
                    default="status")
    ap.add_argument("--n", type=int, default=1, help="units of work this invocation")
    ap.add_argument("--iterations", type=int, default=15)
    ap.add_argument("--minibatch", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cap", type=float, default=80.0, help="hard USD cap, BOTH legs")
    ap.add_argument("--stop", type=float, default=72.0, help="clean-stop USD, BOTH legs")
    ap.add_argument("--leg-cap", type=float, default=None, help="USD cap for this leg")
    ap.add_argument("--merge", action="store_true", default=True)
    ap.add_argument("--no-merge", dest="merge", action="store_false")
    ap.add_argument("--max-merges", type=int, default=4)
    ap.add_argument("--test-baseline", action="store_true",
                    help="also re-run the seed baseline on TEST (normally free from records)")
    ap.add_argument("--force-test", action="store_true")
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.phase == "report":
        write_results(args)
        return

    leg, st = args.leg, load_state(args.leg)
    spend, led, _side = experiment_spend()
    leg_spend = sum(led.get(r, 0.0) for r in st["run_ids"])
    if args.phase == "status":
        win = winner_of(st)
        log(f"v3 {leg} | pool {len(st['pool'])} | iter {st['iteration']}/{args.iterations} | "
            f"merges {st['n_merges']}/{args.max_merges} | ${leg_spend:.2f} this leg, "
            f"${spend:.2f} experiment | stopped: {st.get('stopped')}")
        for c in st["pool"]:
            log(f"  #{c['id']:<2} {c['kind']:<8} {c['name']:<14} it{c['iter']:02d} "
                f"{objective_line(c['vec'])}")
        if win:
            log(f"  VALID winner: #{win['id']} {win['name']} ({win['kind']}, it{win['iter']})")
        for cid, t in (st.get("test") or {}).items():
            if cid.isdigit():
                log(f"  TEST {t['name']:<14} k={t['k']} {objective_line(t['vec'])}")
        return

    log(f"\n===== {datetime.now(timezone.utc).isoformat(timespec='seconds')} leg={leg} "
        f"phase={args.phase} n={args.n} | experiment ${spend:.2f} (leg ${leg_spend:.2f}) | "
        f"cap ${args.cap:.0f} stop ${args.stop:.0f}"
        + (f" leg-cap ${args.leg_cap:.0f}" if args.leg_cap else "") + " =====")
    W.confirm(f"leg {leg} phase {args.phase}: proceed against a ${args.cap:.0f} hard cap?",
              args.yes)
    cache = Cache(CACHE.format(leg=leg))
    if args.phase == "prep":
        phase_prep(args, leg, st)
    elif args.phase == "seeds":
        phase_seeds(args, leg, st, cache)
    elif args.phase == "iterate":
        phase_iterate(args, leg, st, cache)
    elif args.phase == "test":
        phase_test(args, leg, st, cache)
    spend, led, _s = experiment_spend()
    log(f"  spend now ${spend:.2f} experiment total "
        f"(${sum(led.get(r, 0.0) for r in st['run_ids']):.2f} this leg), cap ${args.cap:.0f}")


if __name__ == "__main__":
    main()
