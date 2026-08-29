"""gepa_v2.py - W-F v2 Arm A: GEPA on the strongest monolithic judge architecture.

    .venv/bin/python gepa/gepa_v2.py --selftest             # no spend: metric + seed checks
    .venv/bin/python gepa/gepa_v2.py --phase seeds --n 1 -y # evaluate one seed, full pool, k=4
    .venv/bin/python gepa/gepa_v2.py --phase iterate --n 1 -y
    .venv/bin/python gepa/gepa_v2.py --phase final -y       # k=8 confirms + v2_results.json

WHY THIS EXISTS (gepa/V2-DESIGN.md, "Arm A"). v1 optimised a k=1 scored judge, so the
sentence "you never optimised the best architecture" was still sayable about the wall
claim. The completed grid (docs/FINDINGS.md sec 17) says the best design is FC-score-k8:
completeness criterion, 0-10 score, 8 samples, winsorized mean. v2 optimises THAT, and
scores it on an objective that a threshold shift cannot game.

WHAT IS DIFFERENT FROM v1 (gepa_optimize.py), point by point:

  student            k=4 ensemble during the search, k=8 for the final confirms. The
                     aggregator is the shared winsorized mean (common.winsorize8 at k=8;
                     the same operator applied to 4 values at k=4 - asserted equal in
                     --selftest).
  objective          PRIMARY: absolute detection on omissions at a false-alarm rate <= 10%
                     on the 22 clean twins, with the operating threshold chosen POST HOC
                     from the pool. Adding or subtracting a constant from every score
                     leaves this exactly unchanged, which is the whole point: v1's J
                     rewarded any prompt that shifted the score mass down past a fixed
                     flag line. SECONDARY: paired tie-adjusted discrimination (errored
                     note below its own clean twin, ties count half). A candidate that
                     improves neither is rejected.
  frontier/accept    full-pool scores at k=4, never the minibatch (a colleague's v23b
                     correction). The 16-item stratified minibatch is a pre-filter only.
  reflection         every trace names the class, the severity stratum, the residual
                     level and strength, the surviving mention (and whether the judge's
                     own text quoted it), and the note's rank against its clean twin.
                     The reflector is told outright that a uniform score shift is worth
                     nothing, because that is the loophole the objective closes.
  budget             hard cap enforced between iterations off results/cost_ledger.jsonl,
                     filtered to THIS run's run_ids (the ledger is shared with sibling
                     experiments and their rows must not count against this cap), plus a
                     live in-process count so a single iteration cannot overshoot.

THE DATA RULE IS UNCHANGED. gepa_data.load_examples() refuses to run if any dev
consultation appears in an evaluation set; that assertion is what makes the winner
quotable. 176 dev judgements over 22 consultations, eval_overlap 0.

EXECUTION MODEL. Every phase is resumable: state lives in gepa/v2_state.json, judge
completions in gepa/cache/judge_cache_v2.jsonl (keyed on prompt sha + seed + k, flushed
on every write). A killed invocation therefore costs nothing to redo - which is what lets
the whole run proceed as a series of short, bounded, synchronous calls.
"""
import argparse
import json
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
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import common as C          # noqa: E402
import w2_common as W       # noqa: E402
import gepa_data as D       # noqa: E402

SPEC = "gepa/V2-DESIGN.md"
EXPERIMENT = "gepa-omission-v2"
STUDENT_ROLE = "judge-primary"      # gpt-5.4, the grid's judge
REFLECTION_ROLE = "auditor"         # gpt-5.5, reasoning high
RUN_SEED = 11                       # the grid's replicate-1 seed

SEARCH_K = 4                        # ensemble size during the search (cost)
FINAL_K = 8                         # ensemble size for the confirms (fidelity)
FA_TARGET = 0.10                    # primary operating point
FA_HARD = 0.15                      # soft band: n=22 clean twins carries ~+-9pp

SEED_DIR = os.path.join(HERE, "seeds")
CACHE_PATH = os.path.join(HERE, "cache", "judge_cache_v2.jsonl")
STATE_PATH = os.path.join(HERE, "v2_state.json")
SPEND_SIDECAR = os.path.join(HERE, "v2_spend.json")
LINEAGE = os.path.join(HERE, "v2_lineage.jsonl")
LOGFILE = os.path.join(HERE, "v2_run.log")
RESULTS_JSON = os.path.join(HERE, "v2_results.json")
WINNER_PATH = os.path.join(HERE, "judge_prompt_v2.txt")
V1_WINNER = os.path.join(HERE, "judge_prompt.txt")

OMIT_KINDS = ("omit-complete", "omit-partial")
ERR_KINDS = ("omit-complete", "omit-partial", "add", "change")


# ---------------------------------------------------------------- logging
def log(msg=""):
    print(msg, flush=True)
    with open(LOGFILE, "a") as f:
        f.write(msg + "\n")


def lineage(rec):
    with open(LINEAGE, "a") as f:
        f.write(json.dumps({"t_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            **rec}) + "\n")


# ---------------------------------------------------------------- seeds
SEED_ORDER = ["seed_fc_score", "seed_v1_winner", "seed_engineered", "seed_v14_incl_lowfa"]

LOWFA_BRIEF = (
    "Above everything else, keep false alarms low: a note that is faithful and complete must "
    "land in the top band, so never penalise ordinary clinical compression - paraphrase into "
    "clinical language, standard abbreviations, conventional normal-exam phrasing, dropped "
    "conversational colour, or a fact recorded once instead of three times.")


def ensure_seeds():
    """The four v2 seeds on disk. Three already exist from v1 (--init-seeds); this writes
    the two v2-specific ones and leaves the others byte-identical.

      seed_fc_score        the grid's FC-score criterion verbatim - the architecture's own
                           baseline, and the number every claim in the read-out is against.
      seed_v1_winner       gepa-04, v1's winner, lifted out of gepa/judge_prompt.txt (the
                           fixed preamble and output contract stripped: the harness owns
                           those, and they are byte-identical here).
      seed_engineered      the best prompt a human wrote for this task.
      seed_v14_incl_lowfa  the high-recall production judge (54.5% FA in v1) plus a single
                           explicit brief to drive false alarms down at retained recall -
                           the one seed that starts on the wrong side of the FA gate, so
                           the search has a high-recall parent to pull back rather than
                           only low-recall parents to push out.
    """
    os.makedirs(SEED_DIR, exist_ok=True)
    made = []
    p_v1 = os.path.join(SEED_DIR, "seed_v1_winner.txt")
    if not os.path.exists(p_v1):
        raw = open(V1_WINNER).read()
        body = raw.split("\n---\n\n", 1)[1]
        body = body.split("Return ONLY a JSON object")[0].strip()
        assert len(body) > 2000 and "{transcript}" not in body, "v1 winner extraction failed"
        open(p_v1, "w").write(
            f"<!-- seed: seed_v1_winner | source gepa/judge_prompt.txt (sha256 "
            f"{D.sha(raw)[:16]}) | v1's winner gepa-04, instruction block only -->\n"
            + body + "\n")
        made.append("seed_v1_winner")
    p_lowfa = os.path.join(SEED_DIR, "seed_v14_incl_lowfa.txt")
    if not os.path.exists(p_lowfa):
        base = open(os.path.join(SEED_DIR, "seed_v14_incl.txt")).read()
        body = "\n".join(l for l in base.splitlines() if not l.startswith("<!-- seed:")).strip()
        open(p_lowfa, "w").write(
            "<!-- seed: seed_v14_incl_lowfa | source gepa/seeds/seed_v14_incl.txt + one "
            "false-alarm brief (V2-DESIGN Arm A seed 4) -->\n"
            + body + "\n\n" + LOWFA_BRIEF + "\n")
        made.append("seed_v14_incl_lowfa")
    return made


def load_seeds():
    out = []
    for name in SEED_ORDER:
        path = os.path.join(SEED_DIR, name + ".txt")
        raw = open(path).read()
        body = "\n".join(l for l in raw.splitlines() if not l.startswith("<!-- seed:")).strip()
        out.append({"name": name, "instruction": body})
    return out


# ---------------------------------------------------------------- cache
class Cache:
    """(prompt sha, base seed, k) -> the k completion texts.

    Keyed on k as well as the prompt, because a k=8 judgement is not four more samples on
    top of a k=4 one - llm() issues its own n-sampling request per call, and pretending
    otherwise would silently mix two sampling implementations inside one ensemble.
    """

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
                self.d[r["k"]] = r["t"]
        self._fh = None

    def get(self, key):
        v = self.d.get(key)
        self.hits += v is not None
        self.misses += v is None
        return v

    def put(self, key, texts):
        with self._lock:
            self.d[key] = texts
            if self._fh is None:
                self._fh = open(self.path, "a")
            self._fh.write(json.dumps({"k": key, "t": texts}) + "\n")
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


# ---------------------------------------------------------------- budget
class BudgetStop(Exception):
    pass


def _ledger_rows(run_ids):
    path = os.path.join(C.RESULTS, "cost_ledger.jsonl")
    if not os.path.exists(path) or not run_ids:
        return {}
    want, out = set(run_ids), {}
    for line in open(path):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("run_id") in want:
            out[d["run_id"]] = out.get(d["run_id"], 0.0) + (d.get("cost_usd") or 0.0)
    return out


def ledger_spend(run_ids):
    """This run's spend, and ONLY this run's.

    results/cost_ledger.jsonl is shared with every other experiment in the harness (Arm B
    is writing to it while this runs), so the cap is enforced against the subset of rows
    whose run_id this process created.

    A Run only writes its ledger row when the context EXITS, so a killed invocation would
    otherwise spend real credits and report zero - the exact under-count docs/FINDINGS.md
    flags. gepa/v2_spend.json is a live sidecar written as the calls come back; per run_id
    the larger of the two is taken, so an interrupted invocation still counts against the
    cap and a completed one uses OpenRouter's own accounting.
    """
    led = _ledger_rows(run_ids)
    side = json.load(open(SPEND_SIDECAR)) if os.path.exists(SPEND_SIDECAR) else {}
    tot = sum(max(led.get(r, 0.0), side.get(r, 0.0)) for r in set(run_ids))
    return round(tot, 6), len(led)


class Budget:
    """Prior spend (ledger + sidecar) + this invocation's live spend, against the run's hard
    cap. The live half matters because a single full-pool evaluation must not be able to
    overshoot between two ledger writes."""

    def __init__(self, cap, prior, reserve=0.0, run_id=None):
        self.cap, self.prior, self.reserve = cap, prior, reserve
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
            if self._since_flush >= 10:
                self._since_flush = 0
                self._flush()

    @property
    def total(self):
        return self.prior + self.spent

    @property
    def exhausted(self):
        return self.total >= self.cap

    @property
    def search_exhausted(self):
        return self.total >= self.cap - self.reserve


# ---------------------------------------------------------------- aggregation
def winsorized_mean(vals):
    """The shared k=8 operator (common.winsorize8) stated for any k >= 4: sort, replace the
    min with the 2nd-lowest and the max with the 2nd-highest, take the mean. At k=8 this is
    winsorize8 exactly (asserted in --selftest); at k=4 it is the same operator on 4 draws,
    which is what "run the k=8 architecture at half the samples" has to mean."""
    s = sorted(float(v) for v in vals)
    if len(s) >= 4:
        s[0], s[-1] = s[1], s[-2]
    return sum(s) / len(s)


def aggregate(vals, k):
    if len(vals) != k or any(v is None for v in vals):
        return None
    return float(vals[0]) if k == 1 else winsorized_mean(vals)


# ---------------------------------------------------------------- objective
def _det_at_fa(clean, errored, fa_cap):
    """(threshold, achieved FA, detection) at the most sensitive cut-off whose false-alarm
    rate on the clean twins is still <= fa_cap. Flag rule is `aggregate <= threshold`.

    Both rates are monotone in the threshold, so the best detection under the constraint is
    always at the largest feasible threshold - no search, just the largest candidate that
    keeps FA inside the cap. A threshold below every observed score is always feasible
    (FA = 0, detection = 0), so this is defined for every candidate, which is exactly why
    it cannot be gamed by relocating the score mass.
    """
    if not clean or not errored:
        return None, None, None
    best = (float("-inf"), 0.0, 0.0)
    for t in sorted(set(clean) | set(errored)):
        fa = sum(1 for a in clean if a <= t) / len(clean)
        if fa <= fa_cap + 1e-12 and t > best[0]:
            best = (t, fa, sum(1 for a in errored if a <= t) / len(errored))
    return best


def _tie_adjusted_paired(rows):
    """P(errored below its own clean twin) + 0.5 * P(tie). Chance = 0.5."""
    if not rows:
        return None
    n = sum(1.0 if e < c else 0.5 if e == c else 0.0 for e, c in rows)
    return n / len(rows)


def objective(examples, aggs):
    """The full read-out vector for one candidate on one pool. `aggs` = {key: aggregate}."""
    by_kind = {}
    for e in examples:
        a = aggs.get(e["key"])
        if a is not None:
            by_kind.setdefault(e["kind"], []).append((e, a))
    clean = [a for _e, a in by_kind.get("clean", [])]
    clean_by_key = {e["key"]: a for e, a in by_kind.get("clean", [])}

    def scores(kinds):
        return [a for k in kinds for _e, a in by_kind.get(k, [])]

    def sev_weighted_det(kinds, thr):
        num = den = 0.0
        for k in kinds:
            for e, a in by_kind.get(k, []):
                w = D.SEVERITY_WEIGHT.get(e["severity"] or "supporting", 0.5)
                den += w
                num += w * (1.0 if a <= thr else 0.0)
        return (num / den) if den else None

    omit = scores(OMIT_KINDS)
    t10, fa10, det10 = _det_at_fa(clean, omit, FA_TARGET)
    t15, fa15, det15 = _det_at_fa(clean, omit, FA_HARD)

    def paired(kinds):
        rows = []
        for k in kinds:
            for e, a in by_kind.get(k, []):
                c = clean_by_key.get(e["clean_key"])
                if c is not None:
                    rows.append((a, c))
        return _tie_adjusted_paired(rows)

    def det_at(kind, thr):
        rows = by_kind.get(kind, [])
        return (sum(1 for _e, a in rows if a <= thr) / len(rows)) if rows else None

    vec = {
        "n_judged": sum(len(v) for v in by_kind.values()),
        "n_parse_failures": sum(1 for e in examples if aggs.get(e["key"]) is None),
        "n_clean": len(clean), "n_omit": len(omit),
        # PRIMARY
        "det10": det10, "thr10": t10 if t10 != float("-inf") else None, "fa10": fa10,
        "det10_complete": det_at("omit-complete", t10) if t10 is not None else None,
        "det10_partial": det_at("omit-partial", t10) if t10 is not None else None,
        "det10_sev_weighted": sev_weighted_det(OMIT_KINDS, t10) if t10 is not None else None,
        "det10_addchange": (_det_at_fa(clean, scores(("add", "change")), FA_TARGET)[2]),
        # soft band
        "det15": det15, "thr15": t15 if t15 != float("-inf") else None, "fa15": fa15,
        # SECONDARY
        "disc_omit": paired(OMIT_KINDS), "disc_complete": paired(("omit-complete",)),
        "disc_partial": paired(("omit-partial",)), "disc_addchange": paired(("add", "change")),
        # the architecture's nominal deployment rule, for continuity with the grid
        "det_nominal": (sum(1 for a in omit if a < 8.0) / len(omit)) if omit else None,
        "fa_nominal": (sum(1 for a in clean if a < 8.0) / len(clean)) if clean else None,
        "mean_clean": (sum(clean) / len(clean)) if clean else None,
        "mean_omit": (sum(omit) / len(omit)) if omit else None,
    }
    return vec


def objective_line(v):
    def pct(x):
        return "  n/a" if x is None else f"{100 * x:5.1f}"

    def num(x, fmt="{:.3f}"):
        return " n/a " if x is None else fmt.format(x)
    return (f"det@FA<=10% {pct(v['det10'])}% (thr {num(v['thr10'], '{:.2f}')}, "
            f"FA {pct(v['fa10'])}%) | disc-omit {num(v['disc_omit'])} | "
            f"complete {pct(v['det10_complete'])}% partial {pct(v['det10_partial'])}% | "
            f"det@FA<=15% {pct(v['det15'])}%")


def rewards_of(examples, aggs):
    """Per-example reward for the Pareto frontier, aligned with the objective rather than
    with a flag line: an errored note is rewarded for sitting BELOW its own clean twin
    (tie-adjusted, plus a margin term so a near miss is visibly nearer), a clean note for
    sitting high. A candidate that fixes two hard partials and loses on average therefore
    still leads on those two examples and survives as a parent - GEPA's whole selection
    argument, restated on a threshold-free reward."""
    clean_by_key = {e["key"]: aggs.get(e["key"]) for e in examples if e["kind"] == "clean"}
    out = {}
    for e in examples:
        a = aggs.get(e["key"])
        if a is None:
            out[e["key"]] = 0.0
            continue
        if e["kind"] == "clean":
            out[e["key"]] = max(0.0, min(1.0, a / 10.0))
            continue
        c = clean_by_key.get(e["clean_key"])
        if c is None:
            out[e["key"]] = max(0.0, min(1.0, 1.0 - a / 10.0))
            continue
        base = 1.0 if a < c else 0.5 if a == c else 0.0
        margin = max(0.0, min(1.0, (c - a) / 3.0))
        out[e["key"]] = 0.5 * base + 0.5 * margin
    return out


def batch_separation(examples, aggs):
    """The minibatch PRE-FILTER statistic: tie-adjusted P(an errored note scores below a
    clean note) over every errored-clean cross pair in the batch. Threshold-free, needs no
    twin inside the batch, and is the same quantity the primary objective is derived from -
    so the pre-filter and the accept rule point the same way. It decides nothing on its own:
    every promotion is re-scored on the full pool."""
    clean = [aggs[e["key"]] for e in examples
             if e["kind"] == "clean" and aggs.get(e["key"]) is not None]
    err = [aggs[e["key"]] for e in examples
           if e["kind"] in ERR_KINDS and aggs.get(e["key"]) is not None]
    if not clean or not err:
        return None
    return _tie_adjusted_paired([(a, c) for a in err for c in clean])


# ---------------------------------------------------------------- evaluation
def evaluate(instruction, examples, k, cache, budget, workers, label=""):
    """Run one candidate over `examples` at ensemble size k. Returns {key: {...}}.

    Ordering is consultation-major with a serial warm-up call, so the shared transcript
    prefix is in the provider's cache before the fan-out (docs/COSTS.md lever 1).
    """
    template = D.build_prompt(instruction)
    out, lock = {}, threading.Lock()

    def one(ex):
        prompt = D.render(template, ex["transcript"], ex["note"])
        base_seed = RUN_SEED * 100000 + ex["item_index"] * 10
        ckey = f"{D.sha(prompt)[:32]}|{base_seed}|k{k}"
        texts = cache.get(ckey)
        if texts is None:
            if budget.exhausted:
                raise BudgetStop()
            _vals, texts, metas, _r = W.judge(
                prompt, STUDENT_ROLE, k, base_seed, temperature=1.0,
                reasoning_effort="none", parser=D.parse_score_json)
            for m in metas:
                budget.add(STUDENT_ROLE, m.get("cost_usd_reported"))
            cache.put(ckey, texts)
        vals = [D.parse_score_json(t) for t in texts]
        with lock:
            out[ex["key"]] = {"agg": aggregate(vals, k), "vals": vals,
                              "text0": (texts[0] or "")[:1500]}

    by_consult = {}
    for ex in examples:
        by_consult.setdefault((ex["stratum"], ex["consultation"]), []).append(ex)

    def do_group(group):
        group.sort(key=lambda e: (e["kind"] != "clean", e["key"]))
        one(group[0])                                    # warm the shared transcript prefix
        for ex in group[1:]:
            one(ex)

    groups = [g for _k, g in sorted(by_consult.items())]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(do_group, groups))   # map re-raises the first worker exception here
    if label:
        log(f"    {label}: {len(out)}/{len(examples)} judged k={k} | cache "
            f"{cache.hits}h/{cache.misses}m | ${budget.total:.2f}")
    return out


# ---------------------------------------------------------------- selection
def pareto_parent(pool, rng):
    """A candidate stays in contention if it is the best anyone has managed on at least one
    example, and is sampled in proportion to how many examples it leads."""
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
        return max(pool, key=lambda c: c["vec"]["det10"] or 0.0), leads
    return rng.choices(frontier, weights=[leads[c["id"]] for c in frontier], k=1)[0], leads


MB_QUOTA = [("clean", 4), ("omit-partial", 4), ("omit-complete", 4), ("add", 2), ("change", 2)]


def minibatch(examples, rng, size=16):
    """16 items, stratified so every draw carries both sides of the objective: >= 4 clean
    (the false-alarm side) and >= 4 partial omissions (the class the grid says collapses).
    A uniform draw of 16 from a pool that is 87% errored routinely contains one clean note
    or none, and the false-alarm side then gets no gradient that round."""
    by_kind = {}
    for e in examples:
        by_kind.setdefault(e["kind"], []).append(e)
    picked = []
    for kind, n in MB_QUOTA:
        pool = by_kind.get(kind, [])
        picked += rng.sample(pool, min(n, len(pool)))
    taken = {e["key"] for e in picked}
    rest = [e for e in examples if e["key"] not in taken]
    while len(picked) < size and rest:
        picked.append(rest.pop(rng.randrange(len(rest))))
    rng.shuffle(picked)
    return picked[:size]


# ---------------------------------------------------------------- reflection
REFLECT = """You are improving the instruction block of an LLM judge that reviews AI-generated \
clinical notes against the consultation transcript they came from.

HOW THE JUDGE IS USED. The full prompt sent to the judge is, in this order: the consultation \
transcript, then the clinical note, then YOUR instruction block, then a fixed output contract \
that you do not write and cannot change:

--- fixed output contract (do not restate it, do not contradict it) ---
{contract}
--- end fixed output contract ---

HOW IT IS SCORED, AND THE ONE THING THAT WILL NOT WORK. The judge is run {k} times \
independently on each note and the {k} scores are combined into one number by a winsorized \
mean. A single review threshold is then chosen AFTERWARDS, from the data: the most sensitive \
cut-off that still leaves at most 10% of genuinely clean notes flagged. Detection is then \
measured at that cut-off.

The consequence is worth stating plainly, because it is the trap in this task: making the \
judge harsher - lowering every score, penalising more things, lowering the bar for what counts \
as a violation - buys NOTHING. Every score moving down together simply moves the cut-off down \
with it. The only thing that improves this judge is putting notes in a better ORDER: notes that \
really are missing something the consultation established must score BELOW notes that are \
faithful and complete. Sharper discriminations help. More severity does not.

WHAT COUNTS AS A FAILURE. Two kinds, and either alone is worthless:
  * an errored note scoring at or above a clean note. The errors are fabricated content, \
altered content, and above all OMITTED content the consultation established. Omissions come in \
two flavours: the fact is gone entirely (complete), or the topic is still mentioned somewhere \
while the load-bearing part of it - the working diagnosis, the dose, the reason for urgency, \
the red flag, the pertinent negative - has been dropped (partial). The partial case is far \
harder and is where judges fail, because the surviving mention reads like coverage.
  * a clean note scoring low. Clinical notes legitimately compress: paraphrase into clinical \
language, standard abbreviations, conventional normal-exam phrasing, dropped conversational \
colour, and a fact recorded once rather than three times are all correct behaviour, not loss.

THE CURRENT INSTRUCTION BLOCK
=============================
{instruction}
=============================

HOW IT JUST PERFORMED on {n} examples. Each entry gives the ground truth, the ensemble's \
scores, how the note ranked against its own clean twin, and a diagnosis:

{traces}

YOUR TASK. Write an improved instruction block. Requirements:
  * It replaces the block above entirely and is read by a model that sees ONLY the transcript, \
the note, your block and the fixed contract. It cannot refer to these examples, to this \
critique, to "the previous instruction", or to any consultation you have just been shown - \
those notes will never be seen again. Generalise the lesson; never encode the instance.
  * Aim at the ORDERING failures above, not at overall strictness. If a partial omission scored \
the same as its clean twin, the fix is a discrimination the judge can actually make on an \
unseen note - not a louder instruction to care more.
  * Keep what is working. The diagnoses mark correct rankings explicitly: whatever produced \
them must survive your rewrite.
  * Do not restate the output contract, the JSON shape, or the score range - they are appended \
after your block automatically.
  * Self-contained prose and lists. No placeholders, no {{transcript}} or {{note}} tokens.

Return the new instruction block inside these exact markers and nothing else outside them:
<INSTRUCTION>
...your improved instruction block...
</INSTRUCTION>"""


def _norm(s):
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())


def critique_v2(ex, res, clean_agg, thr):
    """The trace one example contributes to the reflection prompt.

    Names, per the v2 design: the class, the severity stratum, the residual level and the
    strength of the surviving mention, the surviving mention itself, and - the thing v1
    could not say - whether the judge's OWN text quoted that surviving mention, which is
    the direct evidence for "it saw the topic and read it as covered".
    """
    a = res.get("agg")
    vals = res.get("vals") or []
    said = (res.get("text0") or "").strip().replace("\n", " ")
    bits = [f"[{ex['key']}] CLASS={ex['kind']}"]
    if ex.get("severity"):
        bits.append(f"severity={ex['severity']}")
    if ex.get("residual_level"):
        bits.append(f"residual_level={ex['residual_level']}")
    if ex.get("residual_strength"):
        bits.append(f"residual_strength={ex['residual_strength']}")
    head = " ".join(bits)

    if a is None:
        return head + " -> PARSE FAILURE: no usable score came back. Emitted: " + said[:300]

    ens = f"ensemble {vals} -> {a:.2f}"
    if ex["kind"] == "clean":
        verdict = ("CORRECT (this clean note sits above the review cut-off)" if a > thr
                   else "FALSE ALARM (this clean note sits at or below the review cut-off, so it "
                        "would be sent for human review). It is faithful and complete; the judge "
                        "penalised ordinary summarisation - a phrase-level detail, small talk, a "
                        "restated point, a normal finding in conventional shorthand.")
        return f"{head}. Gold: a clean, faithful, complete note. {ens}. {verdict}"

    rank = ("no clean twin available" if clean_agg is None else
            f"its own clean twin scored {clean_agg:.2f}, so this errored note ranked "
            + ("BELOW its twin - correct" if a < clean_agg else
               "LEVEL WITH its twin - a tie, which is a failure of ordering" if a == clean_agg
               else "ABOVE its twin - the wrong way round"))
    caught = "CAUGHT (at or below the review cut-off)" if a <= thr else \
             "MISSED (above the review cut-off)"
    lines = [f"{head}. Gold: errored. {ens}. {rank}. {caught}."]
    fact = (ex.get("fact") or ex.get("must_contain") or "")[:300]
    if ex["kind"].startswith("omit"):
        lines.append(f'  What the note is missing: "{fact}"')
        sites = ex.get("residual") or []
        if sites:
            quote = sites[0].get("quote") or ""
            sect = sites[0].get("section") or "the note"
            echoed = _norm(quote)[:40] and _norm(quote)[:40] in _norm(said)
            lines.append(
                f'  The topic still surfaces in the note as: "{quote}" (in {sect}) - a '
                f'{sites[0].get("strength") or "surviving"} mention. The judge\'s own assessment '
                + ("DID quote that surviving mention, so it saw the topic and read it as covered."
                   if echoed else
                   "did NOT quote that surviving mention.")
                + " A topic that is still named is not the same as a fact that is still recorded.")
        elif ex.get("residual_level") == "complete":
            lines.append("  Nothing of this fact survives anywhere in the note - a complete "
                         "removal, the easiest case there is.")
    else:
        lines.append(f'  What the note gets wrong: "{fact}"')
    if ex.get("severity_rationale"):
        lines.append(f"  Why it matters: {ex['severity_rationale'][:300]}")
    lines.append(f"  judge said: {said[:420]}")
    return "\n".join(lines)


def reflect(instruction, traces, n, budget):
    prompt = REFLECT.format(contract=D.OUTPUT_CONTRACT.strip(), instruction=instruction,
                            traces=traces, n=n, k=SEARCH_K)
    texts, meta = C.llm(prompt, REFLECTION_ROLE, temperature=1.0, k=1,
                        reasoning_effort="high", max_tokens=16000, timeout=900)
    budget.add(REFLECTION_ROLE, meta.get("cost_usd_reported"))
    m = re.search(r"<INSTRUCTION>(.*?)</INSTRUCTION>", texts[0], re.S)
    if not m:
        return None, texts[0][:400]
    new = m.group(1).strip()
    for bad in ("{transcript}", "{note}", "{summary}"):
        new = new.replace(bad, "the transcript" if bad == "{transcript}" else "the note")
    return (new if len(new) > 200 else None), None


# ---------------------------------------------------------------- state
def load_state():
    if os.path.exists(STATE_PATH):
        return json.load(open(STATE_PATH))
    return {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_ids": [], "pool": [], "rejected": [], "next_id": 0, "iteration": 0,
            "reject_streak": 0, "stopped": None, "finals": {}, "search_k": SEARCH_K,
            "final_k": FINAL_K, "calls_k4": 0, "calls_k8": 0}


def save_state(st):
    tmp = STATE_PATH + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, STATE_PATH)


def new_run(params):
    return C.Run(EXPERIMENT, params=params, replicate=1, seed=RUN_SEED,
                 inputs=["master/gepa_dev_pool.json", "master/gepa_dev_factorial.json",
                         "master/partial_omission_seed_set.json"],
                 spec=SPEC, allow_dirty=True)


# ---------------------------------------------------------------- phases
def phase_seeds(args, examples, st, cache):
    done = {c["name"] for c in st["pool"]}
    todo = [s for s in load_seeds() if s["name"] not in done][:args.n or 99]
    if not todo:
        log("  all seeds already evaluated")
        return
    prior, _ = ledger_spend(st["run_ids"])
    budget = Budget(args.cap, prior, args.finals_reserve)
    with new_run({"phase": "seeds", "k": SEARCH_K, "seeds": [s["name"] for s in todo],
                  "cap_usd": args.cap}) as run:
        st["run_ids"].append(run.run_id)
        budget.bind(run.run_id)
        save_state(st)
        try:
            for s in todo:
                res = evaluate(s["instruction"], examples, SEARCH_K, cache, budget,
                               args.workers, f"seed {s['name']}")
                aggs = {k: v["agg"] for k, v in res.items()}
                vec = objective(examples, aggs)
                cand = {"id": st["next_id"], "name": s["name"], "kind": "seed", "parent": None,
                        "iter": 0, "instruction": s["instruction"], "vec": vec,
                        "rewards": rewards_of(examples, aggs), "aggs": aggs,
                        "chars": len(s["instruction"]), "k": SEARCH_K}
                st["pool"].append(cand)
                st["next_id"] += 1
                st["calls_k4"] += len(res) * SEARCH_K
                log(f"  seed #{cand['id']} {s['name']:<22} {objective_line(vec)}")
                lineage({"event": "candidate", "run_id": run.run_id, **_slim(cand)})
                save_state(st)
                if budget.search_exhausted:
                    log("  ! search budget reserve reached during seeds")
                    break
        except BudgetStop:
            log("  ! budget cap hit mid-evaluation - state saved, nothing lost (cache is warm)")
        finally:
            budget._flush()
            cache.close()
    save_state(st)


def phase_iterate(args, examples, st, cache):
    prior, _ = ledger_spend(st["run_ids"])
    budget = Budget(args.cap, prior, args.finals_reserve)
    if budget.search_exhausted:
        st["stopped"] = f"budget reserve reached (${budget.total:.2f} of ${args.cap:.0f})"
        save_state(st)
        log(f"  STOP: {st['stopped']}")
        return
    if st["reject_streak"] >= args.stall:
        st["stopped"] = f"stall-stop ({args.stall} consecutive rejects)"
        save_state(st)
        log(f"  STOP: {st['stopped']}")
        return
    if st["iteration"] >= args.iterations:
        st["stopped"] = "iterations exhausted"
        save_state(st)
        log(f"  STOP: {st['stopped']}")
        return

    with new_run({"phase": "iterate", "k": SEARCH_K, "from_iter": st["iteration"] + 1,
                  "n": args.n, "cap_usd": args.cap}) as run:
        st["run_ids"].append(run.run_id)
        budget.bind(run.run_id)
        save_state(st)
        try:
            for _ in range(args.n):
                if st["iteration"] >= args.iterations:
                    st["stopped"] = "iterations exhausted"
                    break
                if st["reject_streak"] >= args.stall:
                    st["stopped"] = f"stall-stop ({args.stall} consecutive rejects)"
                    break
                if budget.search_exhausted:
                    st["stopped"] = (f"budget reserve reached (${budget.total:.2f} of "
                                     f"${args.cap:.0f}, ${args.finals_reserve:.0f} held for k=8)")
                    break
                _one_iteration(args, examples, st, cache, budget, run)
                save_state(st)
        except BudgetStop:
            st["stopped"] = f"budget cap hit mid-evaluation (${budget.total:.2f})"
            log("  ! budget cap hit mid-evaluation - state saved, cache is warm")
        finally:
            budget._flush()
            cache.close()
    if st.get("stopped"):
        log(f"  STOP: {st['stopped']}")
    save_state(st)


def _one_iteration(args, examples, st, cache, budget, run):
    it = st["iteration"] + 1
    rng = random.Random(args.seed + 1013 * it)
    parent, leads = pareto_parent(st["pool"], rng)
    mb = minibatch(examples, rng, args.minibatch)

    p_res = evaluate(parent["instruction"], mb, SEARCH_K, cache, budget, args.workers)
    p_aggs = {k: v["agg"] for k, v in p_res.items()}
    p_stat = batch_separation(mb, p_aggs)
    thr = parent["vec"].get("thr10")
    thr = thr if thr is not None else 8.0
    clean_by_key = {e["key"]: p_aggs.get(e["key"]) for e in mb if e["kind"] == "clean"}
    traces = "\n\n".join(critique_v2(e, p_res[e["key"]], clean_by_key.get(e["clean_key"]), thr)
                         for e in mb if e["key"] in p_res)

    new_instr, why = reflect(parent["instruction"], traces, len(mb), budget)
    st["iteration"] = it
    if new_instr is None:
        st["reject_streak"] += 1
        log(f"  it{it:02d} parent#{parent['id']} -> reflection produced no block ({why})")
        lineage({"event": "mutation_failed", "iter": it, "parent": parent["id"], "reason": why})
        return
    if any(D.sha(new_instr) == D.sha(c["instruction"]) for c in st["pool"]):
        st["reject_streak"] += 1
        log(f"  it{it:02d} parent#{parent['id']} -> duplicate instruction, skipped")
        lineage({"event": "mutation_duplicate", "iter": it, "parent": parent["id"]})
        return

    c_res = evaluate(new_instr, mb, SEARCH_K, cache, budget, args.workers)
    c_stat = batch_separation(mb, {k: v["agg"] for k, v in c_res.items()})
    st["calls_k4"] += (len(p_res) + len(c_res)) * SEARCH_K
    passed = c_stat is not None and p_stat is not None and c_stat > p_stat + 1e-9
    log(f"  it{it:02d} parent#{parent['id']} ({parent['name']}, leads "
        f"{leads[parent['id']]}/{len(examples)}) minibatch sep {p_stat:.3f} -> {c_stat:.3f} "
        f"{'promote' if passed else 'PRE-FILTERED OUT'} | ${budget.total:.2f}")
    if not passed:
        st["reject_streak"] += 1
        lineage({"event": "prefiltered", "iter": it, "parent": parent["id"],
                 "mb_parent": round(p_stat or 0, 4), "mb_child": round(c_stat or 0, 4),
                 "chars": len(new_instr), "instruction_sha256": D.sha(new_instr),
                 "instruction": new_instr})
        return

    full = evaluate(new_instr, examples, SEARCH_K, cache, budget, args.workers, f"it{it:02d} full")
    aggs = {k: v["agg"] for k, v in full.items()}
    vec = objective(examples, aggs)
    st["calls_k4"] += len(full) * SEARCH_K
    pv = parent["vec"]
    better_primary = (vec["det10"] or 0) > (pv["det10"] or 0) + 1e-9
    same_primary = abs((vec["det10"] or 0) - (pv["det10"] or 0)) <= 1e-9
    better_secondary = (vec["disc_omit"] or 0) > (pv["disc_omit"] or 0) + 1e-9
    accepted = better_primary or (same_primary and better_secondary)
    log(f"       full pool: {objective_line(vec)}")
    log(f"       vs parent#{parent['id']}: det10 {100*(pv['det10'] or 0):.1f}% -> "
        f"{100*(vec['det10'] or 0):.1f}%, disc {pv['disc_omit']:.3f} -> {vec['disc_omit']:.3f} "
        f"=> {'ACCEPT' if accepted else 'reject'}")
    rec = {"id": st["next_id"] if accepted else None, "iter": it, "parent": parent["id"],
           "vec": vec, "chars": len(new_instr), "instruction_sha256": D.sha(new_instr),
           "mb_parent": round(p_stat, 4), "mb_child": round(c_stat, 4),
           "spend_usd": round(budget.total, 4)}
    if not accepted:
        st["reject_streak"] += 1
        st["rejected"].append({**rec, "instruction": new_instr})
        lineage({"event": "rejected_full_pool", "run_id": run.run_id, **rec})
        return
    cand = {"id": st["next_id"], "name": f"gepa2-{st['next_id']:02d}", "kind": "mutation",
            "parent": parent["id"], "iter": it, "instruction": new_instr, "vec": vec,
            "rewards": rewards_of(examples, aggs), "aggs": aggs,
            "chars": len(new_instr), "k": SEARCH_K}
    st["pool"].append(cand)
    st["next_id"] += 1
    st["reject_streak"] = 0
    log(f"       -> accepted as #{cand['id']} {cand['name']}")
    lineage({"event": "candidate", "run_id": run.run_id, **_slim(cand)})


def phase_final(args, examples, st, cache):
    """k=8 on the full dev pool for the confirm set: the FC-score seed (the architecture's
    own baseline - the number every claim is against), the k=4 winner, and the next best
    frontier candidates the reserve can pay for."""
    baseline = "seed_fc_score"
    ranked = sorted(st["pool"], key=lambda c: (-(c["vec"]["det10"] or 0),
                                               -(c["vec"]["disc_omit"] or 0)))
    # Confirm order is not simply "top of the k=4 leaderboard". Three slots are reserved,
    # because they are the three numbers the read-out has to quote: the architecture's own
    # baseline, the best thing prompt SEARCH produced (a mutation - it answers "did GEPA
    # find anything", and a leaderboard sorted on det10 alone can crowd it out behind
    # seeds), and the best candidate overall. The rest fill by rank if the reserve stretches.
    order, seen = [], set()
    for pick in ([c for c in ranked if c["name"] == baseline][:1]
                 + [c for c in ranked if c["kind"] == "mutation"][:1]
                 + ranked):
        if pick["id"] not in seen:
            seen.add(pick["id"])
            order.append(pick)
    if args.confirm_name:
        order = [c for c in st["pool"] if c["name"] == args.confirm_name]
    order = order[:args.finals_max]
    prior, _ = ledger_spend(st["run_ids"])
    budget = Budget(args.cap, prior, 0.0)
    with new_run({"phase": "final", "k": FINAL_K, "cap_usd": args.cap,
                  "confirm": [c["name"] for c in order]}) as run:
        st["run_ids"].append(run.run_id)
        budget.bind(run.run_id)
        save_state(st)
        try:
            for c in order:
                if str(c["id"]) in st["finals"]:
                    continue
                if budget.exhausted:
                    log("  ! cap reached - stopping confirms with what we have")
                    break
                res = evaluate(c["instruction"], examples, FINAL_K, cache, budget,
                               args.workers, f"k8 {c['name']}")
                aggs = {k: v["agg"] for k, v in res.items()}
                vec = objective(examples, aggs)
                st["calls_k8"] += len(res) * FINAL_K
                st["finals"][str(c["id"])] = {"name": c["name"], "kind": c["kind"],
                                              "iter": c["iter"], "vec": vec, "aggs": aggs}
                log(f"  k=8 #{c['id']} {c['name']:<22} {objective_line(vec)}")
                lineage({"event": "final_k8", "run_id": run.run_id, "id": c["id"],
                         "name": c["name"], "vec": vec})
                save_state(st)
        except BudgetStop:
            log("  ! budget cap hit during confirms")
        finally:
            budget._flush()
            cache.close()
    save_state(st)


def _slim(c):
    return {k: v for k, v in c.items() if k not in ("rewards", "aggs")}


# ---------------------------------------------------------------- report
def significance(st, examples, n_boot=4000):
    """Every k=8 confirm against the FC-score baseline, on the primary measure.

    Detection at FA <= 10% is a rate over 116 omissions but only 22 clean twins set the
    operating point, and the notes cluster inside 22 consultations - so a raw pp difference
    means very little on its own. Two tests, both paired on the same omissions: an exact
    McNemar on the discordant notes, and a cluster bootstrap resampling CONSULTATIONS (not
    notes), which is the unit of independence here.
    """
    F = st.get("finals") or {}
    base_id = next((i for i, f in F.items() if f["name"] == "seed_fc_score"), None)
    if base_id is None:
        return {}
    consult = {e["key"]: e["consultation"] for e in examples}
    omissions = [e["key"] for e in examples if e["kind"] in OMIT_KINDS]

    def flags(f):
        thr = f["vec"]["thr10"]
        thr = float("-inf") if thr is None else thr
        return {k: (f["aggs"][k] <= thr) for k in omissions if f["aggs"].get(k) is not None}

    bf = flags(F[base_id])
    out = {}
    for i, f in F.items():
        if i == base_id:
            continue
        cf = flags(f)
        keys = sorted(set(bf) & set(cf))
        b_only = sum(1 for k in keys if bf[k] and not cf[k])
        c_only = sum(1 for k in keys if cf[k] and not bf[k])
        n = b_only + c_only
        import math
        p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, x)
                                                for x in range(0, min(b_only, c_only) + 1)) / 2 ** n)
        cl = {}
        for k in keys:
            cl.setdefault(consult[k], []).append(k)
        ids, rng, diffs = list(cl), random.Random(RUN_SEED), []
        for _ in range(n_boot):
            samp = [k for c in rng.choices(ids, k=len(ids)) for k in cl[c]]
            diffs.append(sum(cf[k] for k in samp) / len(samp)
                         - sum(bf[k] for k in samp) / len(samp))
        diffs.sort()
        out[f["name"]] = {
            "n_omissions": len(keys), "kind": f["kind"],
            "det10_candidate": round(100 * sum(cf[k] for k in keys) / len(keys), 2),
            "det10_baseline": round(100 * sum(bf[k] for k in keys) / len(keys), 2),
            "delta_pp": round(100 * (sum(cf[k] for k in keys) - sum(bf[k] for k in keys))
                              / len(keys), 2),
            "discordant_candidate_only": c_only, "discordant_baseline_only": b_only,
            "mcnemar_exact_p": round(p, 4),
            "cluster_bootstrap_95ci_pp": [round(100 * diffs[int(.025 * n_boot)], 2),
                                          round(100 * diffs[int(.975 * n_boot)], 2)],
            "fa_candidate": f["vec"]["fa10"], "fa_baseline": F[base_id]["vec"]["fa10"],
            "disc_omit_candidate": f["vec"]["disc_omit"],
            "disc_omit_baseline": F[base_id]["vec"]["disc_omit"]}
    return out


def _best_mutation_block(st, base):
    """The number the arm actually exists to report: the best thing the v2 SEARCH produced,
    at k=8, beside the architecture's own baseline. Seeds are comparators, not findings."""
    muts = {i: f for i, f in st["finals"].items() if f["kind"] == "mutation"}
    if not muts or not base:
        return {"confirmed_at_k8": bool(muts), "note": "no mutation confirmed at k=8"}
    i, f = max(muts.items(), key=lambda kv: (kv[1]["vec"]["det10"] or 0,
                                             kv[1]["vec"]["disc_omit"] or 0))
    return {"id": i, "name": f["name"], "iter": f["iter"], "k8": f["vec"],
            "delta_det10_pp": round(100 * ((f["vec"]["det10"] or 0)
                                           - (base["vec"]["det10"] or 0)), 2),
            "delta_disc_omit": round((f["vec"]["disc_omit"] or 0)
                                     - (base["vec"]["disc_omit"] or 0), 4),
            "beats_baseline": (f["vec"]["det10"] or 0) > (base["vec"]["det10"] or 0) + 1e-9}


def write_results(args, examples, st):
    finals = st["finals"]
    base_id = next((i for i, f in finals.items() if f["name"] == "seed_fc_score"), None)
    base = finals.get(base_id) if base_id else None
    winner_id, winner = None, None
    for i, f in sorted(finals.items(), key=lambda kv: (-(kv[1]["vec"]["det10"] or 0),
                                                       -(kv[1]["vec"]["disc_omit"] or 0))):
        winner_id, winner = i, f
        break
    beats = bool(base and winner and (
        (winner["vec"]["det10"] or 0) > (base["vec"]["det10"] or 0) + 1e-9 or
        (abs((winner["vec"]["det10"] or 0) - (base["vec"]["det10"] or 0)) <= 1e-9 and
         (winner["vec"]["disc_omit"] or 0) > (base["vec"]["disc_omit"] or 0) + 1e-9)))
    # Writing the artifact. The rule is not "whoever tops the k=8 table" - a SEED topping it
    # means prompt search found nothing, and the pool's seeds include prompts that already
    # exist on disk as their own artifacts. Emitting a byte-identical copy of v1's
    # judge_prompt.txt under a v2 name would tell a later session that v2 discovered
    # something, which it did not. So: the file is written only when the winner is a
    # MUTATION (a v2 discovery) that beats the baseline, and every other outcome is recorded
    # in v2_results.json instead of in a misleading file.
    winner_kind = winner["kind"] if winner else None
    duplicate_of = None
    if winner:
        built = D.build_prompt(next(c["instruction"] for c in st["pool"]
                                    if str(c["id"]) == winner_id))
        for other in (V1_WINNER,):
            if os.path.exists(other) and open(other).read() == built:
                duplicate_of = os.path.relpath(other, ROOT)
    write_it = bool(beats and winner_kind == "mutation" and not duplicate_of)
    if write_it:
        open(WINNER_PATH, "w").write(built)
        log(f"  wrote {os.path.relpath(WINNER_PATH, ROOT)} - a v2 mutation beat the baseline")
    else:
        if os.path.exists(WINNER_PATH):
            os.remove(WINNER_PATH)
        why = ("no candidate beat the FC-score seed baseline at k=8" if not beats else
               f"the k=8 top scorer is a {winner_kind}, not something the v2 search produced"
               + (f", and its prompt is byte-identical to {duplicate_of}" if duplicate_of else ""))
        log(f"  judge_prompt_v2.txt deliberately NOT written: {why}")

    spend, rows = ledger_spend(st["run_ids"])
    out = {
        "arm": "W-F v2 Arm A - GEPA on FC-score-k8",
        "spec": SPEC, "experiment": EXPERIMENT,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "student_model": C.resolve_model(STUDENT_ROLE)["resolved"],
        "reflection_model": C.resolve_model(REFLECTION_ROLE)["resolved"],
        "student_settings": {"temperature": 1.0, "reasoning_effort": "none",
                             "max_tokens": W.MAX_TOKENS, "search_k": SEARCH_K,
                             "final_k": FINAL_K, "aggregator": "winsorized mean",
                             "prompt_shape": "gepa_data.build_prompt (preamble + evolved "
                                             "instruction + fixed JSON contract)"},
        "objective": {"primary": "absolute detection on omissions at FA <= 10% on the 22 "
                                 "clean twins, threshold chosen post hoc from the pool",
                      "soft_band_hard_ceiling": FA_HARD,
                      "secondary": "paired tie-adjusted discrimination (errored below its "
                                   "own clean twin, ties half)",
                      "accept_rule": "primary improves, or primary ties and secondary improves"},
        "provenance": st.get("provenance"),
        "params": {k: v for k, v in vars(args).items() if k not in ("yes",)},
        "run_ids": st["run_ids"], "n_ledger_rows": rows, "spend_usd_ledger": spend,
        # judgements_served counts every judgement the search asked for, cache hits included;
        # api_* are the calls actually bought, taken from the ledger rows this run wrote.
        "calls": {"k4_judgements_served_x_samples": st["calls_k4"],
                  "k8_judgements_served_x_samples": st["calls_k8"],
                  "api": st.get("api_calls")},
        "trajectory": {"iterations_run": st["iteration"], "stopped_because": st.get("stopped"),
                       "n_candidates": len(st["pool"]), "n_seeds": sum(
                           1 for c in st["pool"] if c["kind"] == "seed"),
                       "n_accepted": sum(1 for c in st["pool"] if c["kind"] == "mutation"),
                       "n_rejected_full_pool": len(st["rejected"]),
                       "final_reject_streak": st["reject_streak"]},
        "k4_candidates": [_slim(c) for c in st["pool"]],
        "k4_rejected": [{k: v for k, v in r.items() if k != "instruction"}
                        for r in st["rejected"]],
        "k8_confirms": {i: {"name": f["name"], "kind": f["kind"], "iter": f["iter"],
                            "vec": f["vec"]} for i, f in finals.items()},
        "k8_not_confirmed": [c["name"] for c in st["pool"] if str(c["id"]) not in finals],
        "significance_vs_baseline_k8": significance(st, examples),
        "prereg_stall_stop": st.get("prereg_stall_stop"),
        "headline": {
            "baseline_name": base["name"] if base else None,
            "baseline_k8": base["vec"] if base else None,
            "winner_id": winner_id, "winner_name": winner["name"] if winner else None,
            "winner_kind": winner_kind, "winner_k8": winner["vec"] if winner else None,
            "winner_beats_baseline_primary": beats,
            "winner_prompt_identical_to": duplicate_of,
            "artifact_written": write_it,
            "best_search_product": _best_mutation_block(st, base),
            "delta_det10_pp": (round(100 * ((winner["vec"]["det10"] or 0)
                                            - (base["vec"]["det10"] or 0)), 2)
                               if base and winner else None),
            "delta_disc_omit": (round((winner["vec"]["disc_omit"] or 0)
                                      - (base["vec"]["disc_omit"] or 0), 4)
                                if base and winner else None)},
        "per_example": {
            "examples": [{"key": e["key"], "kind": e["kind"], "severity": e["severity"],
                          "residual_level": e.get("residual_level"),
                          "residual_strength": e.get("residual_strength"),
                          "consultation": e["consultation"], "stratum": e["stratum"]}
                         for e in examples],
            "k4_aggregates": {str(c["id"]): c["aggs"] for c in st["pool"]},
            "k8_aggregates": {i: f["aggs"] for i, f in finals.items()}},
    }
    json.dump(out, open(RESULTS_JSON, "w"), indent=1)
    log(f"  wrote {os.path.relpath(RESULTS_JSON, ROOT)}")
    return out


# ---------------------------------------------------------------- selftest
def selftest():
    for trial in range(200):
        rng = random.Random(trial)
        v = [rng.randint(0, 10) for _ in range(8)]
        assert abs(winsorized_mean(v) - C.winsorize8([float(x) for x in v])) < 1e-12
    # det@FA is invariant to a uniform shift, and to any monotone relabelling
    clean, err = [9.0, 9.0, 8.0, 10.0, 9.5], [8.0, 9.0, 7.0, 6.0]
    t, fa, det = _det_at_fa(clean, err, 0.25)
    t2, fa2, det2 = _det_at_fa([c - 3 for c in clean], [e - 3 for e in err], 0.25)
    assert abs(det - det2) < 1e-12 and abs(fa - fa2) < 1e-12, (det, det2)
    t3, fa3, det3 = _det_at_fa([c * 2 for c in clean], [e * 2 for e in err], 0.25)
    assert abs(det - det3) < 1e-12
    assert abs(t2 - (t - 3)) < 1e-12
    # 1 of 5 clean allowed at 25%: threshold 8.0 -> det = 3/4
    assert abs(det - 0.75) < 1e-12 and abs(fa - 0.2) < 1e-12, (t, fa, det)
    assert _tie_adjusted_paired([(1, 2), (2, 2), (3, 2)]) == 0.5
    ex, prov = D.load_examples(verbose=False)
    assert len(ex) == 176 and prov["eval_overlap"] == 0
    made = ensure_seeds()
    seeds = load_seeds()
    assert len(seeds) == 4 and all(len(s["instruction"]) > 300 for s in seeds)
    rng = random.Random(7)
    mb = minibatch(ex, rng, 16)
    n_clean = sum(1 for e in mb if e["kind"] == "clean")
    n_part = sum(1 for e in mb if e["kind"] == "omit-partial")
    assert len(mb) == 16 and n_clean >= 3 and n_part >= 3, (len(mb), n_clean, n_part)
    # the prompt shape must round-trip and stay parseable
    p = D.render(D.build_prompt(seeds[0]["instruction"]), "T", "N")
    assert "{transcript}" not in p and "Return ONLY a JSON object" in p
    assert D.parse_score_json('{"score": 7, "omissions": []}') == 7
    print(f"selftest OK | seeds written: {made or 'none (all present)'} | "
          f"{len(ex)} dev judgements | minibatch {n_clean} clean / {n_part} partial")
    for s in seeds:
        print(f"  seed {s['name']:<22} {len(s['instruction']):>6} chars")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["seeds", "iterate", "final", "report", "status"],
                    default="status")
    ap.add_argument("--n", type=int, default=1, help="units of work this invocation")
    ap.add_argument("--iterations", type=int, default=15)
    ap.add_argument("--stall", type=int, default=5, help="stop after N consecutive rejects")
    ap.add_argument("--minibatch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cap", type=float, default=80.0, help="hard USD cap for the whole run")
    ap.add_argument("--finals-reserve", type=float, default=20.0,
                    help="USD held back out of the cap for the k=8 confirms")
    ap.add_argument("--finals-max", type=int, default=4)
    ap.add_argument("--confirm-name", default=None,
                    help="confirm exactly this candidate at k=8 (overrides the rank order)")
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    ensure_seeds()
    examples, prov = D.load_examples(verbose=False)
    st = load_state()
    st["provenance"] = prov
    spend, rows = ledger_spend(st["run_ids"])

    if args.phase == "status":
        log(f"W-F v2 | {len(examples)} dev judgements, eval_overlap {prov['eval_overlap']} | "
            f"pool {len(st['pool'])} | iter {st['iteration']}/{args.iterations} | "
            f"rejects {st['reject_streak']}/{args.stall} | ${spend:.2f} over {rows} ledger rows "
            f"| stopped: {st.get('stopped')}")
        for c in st["pool"]:
            log(f"  #{c['id']:<2} {c['kind']:<8} {c['name']:<22} it{c['iter']:02d} "
                f"{objective_line(c['vec'])}")
        for i, f in st["finals"].items():
            log(f"  k8 #{i:<2} {f['name']:<22} {objective_line(f['vec'])}")
        return

    log(f"\n===== {datetime.now(timezone.utc).isoformat(timespec='seconds')} phase={args.phase} "
        f"n={args.n} | prior ${spend:.2f} ({rows} rows) | cap ${args.cap:.0f} "
        f"(reserve ${args.finals_reserve:.0f}) =====")
    W.confirm(f"phase {args.phase}: proceed against a ${args.cap:.0f} hard cap?", args.yes)

    cache = Cache(CACHE_PATH)
    if args.phase == "seeds":
        phase_seeds(args, examples, st, cache)
    elif args.phase == "iterate":
        phase_iterate(args, examples, st, cache)
    elif args.phase == "final":
        phase_final(args, examples, st, cache)
        write_results(args, examples, st)
    elif args.phase == "report":
        write_results(args, examples, st)
    spend, rows = ledger_spend(st["run_ids"])
    log(f"  spend now ${spend:.2f} over {rows} ledger rows (cap ${args.cap:.0f})")


if __name__ == "__main__":
    main()
