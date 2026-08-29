"""gepa_optimize.py - W-F: reflective prompt evolution for an omission-aware judge.

    .venv/bin/python gepa/gepa_optimize.py --init-seeds       # write gepa/seeds/*.txt
    .venv/bin/python gepa/gepa_optimize.py --smoke -y         # 1 seed, 8 examples, 1 iteration
    .venv/bin/python gepa/gepa_optimize.py --iterations 40 -y # the real run

WHAT THIS IS. GEPA (reflective prompt evolution): run a candidate prompt on a batch,
read its own failures back in natural language, ask a stronger model to rewrite the
instruction in light of them, keep what survives, and select parents from a Pareto
frontier over individual examples rather than a single average. The loop here is the
one a colleague's v23b run used - `experiments/clinical-hallucination-detection/methods/
v23b_gepa-judge-fulltrain/{optimize,feedback,program}.py` - reimplemented over this
harness's own transport instead of DSPy. Why, in full, in README.md; the short version
is that the artifact this workstream owes the grid is a raw prompt file at
`gepa/judge_prompt.txt` which `w2_arms.py` renders and `parse_score_json` parses, and
optimising a prompt inside DSPy's field-formatting would optimise it for a wrapper the
grid does not use.

WHAT IT OPTIMISES (gepa_data.py): detection on omissions - complete AND partial -
severity-weighted, detection on fabrications and alterations, and false alarms on clean
notes under a hard 10% ceiling. The June pilot's GEPA judge was trained against a
label scheme with no omission category at all and kept v14's "omissions are NOT errors"
exclusion; this objective is the correction.

THE DATA RULE. Dev pool only. gepa_data.load_examples() refuses to run if any dev
consultation appears in an evaluation set. The winner reaches the grid untested on
eval data - that is the entire value of the arm.
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

import common as C  # noqa: E402
import w2_common as W  # noqa: E402
import gepa_data as D  # noqa: E402

SPEC = "specs/w-f-gepa-omission-judge.md"
SEED_DIR = os.path.join(HERE, "seeds")
CACHE_PATH = os.path.join(HERE, "cache", "judge_cache.jsonl")
LINEAGE = os.path.join(HERE, "lineage.jsonl")
WINNER = os.path.join(HERE, "judge_prompt.txt")
RESULTS_JSON = os.path.join(HERE, "results.json")

STUDENT_ROLE = "judge-primary"     # gpt-5.4, exactly the grid's judge - direct comparability
REFLECTION_ROLE = "auditor"        # gpt-5.5 at reasoning_effort=high, the pinned strong role
RUN_SEED = 11                      # the grid's replicate-1 seed, so seeds line up with W2 r1


# ---------------------------------------------------------------- seeds
SEED_SPECS = [
    ("seed_fc_score", "grid_FC_score.txt",
     "W2's FC-score-k1 cell - the completeness-aware criterion in its minimal-pair form. "
     "The pivotal contrast: what did optimisation buy over the grid's own best-shaped cell?"),
    ("seed_v14_incl", "v14_incl.txt",
     "v14 with its omission-exclusion hunks removed and one affirmative omission bullet added - "
     "a hand-tuned production judge pointed at the right target, ~9.7KB of accumulated rules."),
    ("seed_engineered", "engineered_completeness.txt",
     "the engineered completeness judge: shared criterion + the severity decision procedure + "
     "8 hand-authored salience few-shots. The strongest prompt a human wrote for this task."),
]


def init_seeds():
    """Lift the three seed instruction blocks out of w2_prompts into gepa/seeds/.

    Each grid prompt interleaves its instructions with the transcript/note slots and ends
    with its own output-format line. The seed is that prompt with the slots and the format
    line removed - the format is the harness's now (gepa_data.OUTPUT_CONTRACT), so a
    candidate can never evolve its way out of being parseable.
    """
    os.makedirs(SEED_DIR, exist_ok=True)
    written = []
    for name, src, why in SEED_SPECS:
        path = os.path.join(W.PROMPT_DIR, src)
        if not os.path.exists(path):
            print(f"  ! {src} missing - skipping seed {name}")
            continue
        text = open(path).read()
        body = _strip_to_instruction(text, src)
        out = os.path.join(SEED_DIR, name + ".txt")
        open(out, "w").write(
            f"<!-- seed: {name} | source w2_prompts/{src} (sha256 {D.sha(text)[:16]}) | {why} -->\n"
            + body.strip() + "\n")
        written.append((name, src, len(body)))
        print(f"  wrote seeds/{name}.txt  <- w2_prompts/{src}  ({len(body)} chars)")
    return written


def _strip_to_instruction(text, src):
    """Drop the transcript/note blocks and the arm's own output-format tail."""
    if src == "grid_FC_score.txt":
        body = text.split("---", 1)[1]
        return body.split("Assess the note strictly against this criterion.")[0].strip() + (
            "\n\nAssess the note strictly against this criterion.")
    if src == "engineered_completeness.txt":
        return text.split("CONSULTATION TRANSCRIPT:")[0].strip()
    if src == "v14_incl.txt":
        head = text.split("### Transcript")[0].strip()
        tail = text.split("### Instructions", 1)[1]
        tail = tail.split("4. End your response")[0].rstrip()
        body = head + "\n\n### Instructions" + tail
        # v14 is a binary YES/NO judge and this arm is a scored one. Port its decision
        # idiom onto the score, minimally: the rules, the few-shots and the reasoning are
        # v14's verbatim, only the word for "flag" changes. Anything more would stop it
        # being a v14 seed.
        for a, b in (("Output a single binary decision: YES (flag for review) or NO.",
                      "Decide whether the note should be flagged for review, and express that "
                      "decision through the score: a flag-worthy note scores at or below the "
                      "flag line, a note you would not flag scores above it."),
                     ("If yes for any single item in the note → flag YES.",
                      "If yes for any single item in the note → the note is flag-worthy."),
                     ("If no for every item → flag NO.",
                      "If no for every item → the note is not flag-worthy."),
                     ("Verdict: YES", "Judgement: flag-worthy"),
                     ("Verdict: NO", "Judgement: not flag-worthy"),
                     ("flag the\n   doc YES.", "the note is\n   flag-worthy."),
                     ("flag\n   the doc NO.", "the note is\n   not flag-worthy.")):
            body = body.replace(a, b)
        return body
    raise ValueError(src)


def load_prior_candidates():
    """Every accepted candidate from previous runs, out of lineage.jsonl.

    Lets a run that stopped on its iteration count carry on from the population it built
    instead of restarting from the seeds. Re-scoring them is free: the judge cache is keyed
    on (prompt sha, seed), so a candidate already evaluated on the dev pool costs nothing
    to evaluate again.
    """
    if not os.path.exists(LINEAGE):
        return []
    seen, out = set(), []
    for line in open(LINEAGE):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") != "candidate" or not r.get("instruction"):
            continue
        h = D.sha(r["instruction"])
        if h in seen:
            continue
        seen.add(h)
        out.append({"name": r["name"], "kind": r["kind"], "instruction": r["instruction"],
                    "prior_id": r["id"], "prior_parent": r.get("parent")})
    return out


def load_seeds(names=None):
    out = []
    for name, _src, why in SEED_SPECS:
        path = os.path.join(SEED_DIR, name + ".txt")
        if not os.path.exists(path) or (names and name not in names):
            continue
        raw = open(path).read()
        body = "\n".join(l for l in raw.splitlines() if not l.startswith("<!-- seed:"))
        out.append({"name": name, "instruction": body.strip(), "why": why})
    if not out:
        raise SystemExit("no seeds on disk - run with --init-seeds first")
    return out


# ---------------------------------------------------------------- judge cache
class Cache:
    """(prompt sha, seed) -> completion text. Judge calls are pure, candidates share the
    same 79 notes, and a surviving parent gets re-scored every iteration - so without this
    the search would re-buy the same answer dozens of times."""

    def __init__(self, path):
        self.path = path
        self.hits = self.misses = 0
        self.d = {}
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

    def put(self, key, text):
        with self._lock:
            self.d[key] = text
            if self._fh is None:
                self._fh = open(self.path, "a")
            self._fh.write(json.dumps({"k": key, "t": text}) + "\n")
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


# ---------------------------------------------------------------- spend
class Budget:
    """This run's own credit cap, and the bridge to the project-cumulative tripwires.

    w2_common.SpendGuard watches the whole project's ledger (warn $700 / stop $900) and
    only sees committed lines; this run's ledger line is written at the end, so every
    call is fed to the guard live as well. Both bite; this cap bites first.
    """

    def __init__(self, cap, guard=None):
        self.cap, self.spent, self.calls = cap, 0.0, 0
        self.by_role, self.guard = {}, guard
        self._lock = threading.Lock()

    def add(self, role, usd):
        with self._lock:
            self.spent += usd or 0.0
            self.calls += 1
            self.by_role[role] = round(self.by_role.get(role, 0.0) + (usd or 0.0), 6)
            if self.guard is not None:
                self.guard.add(usd or 0.0)   # raises SystemExit at the project hard tripwire

    @property
    def exhausted(self):
        return self.spent >= self.cap


# ---------------------------------------------------------------- evaluation
def evaluate(instruction, examples, cache, budget, workers=10, label=""):
    """Run one candidate over `examples`. Returns (results, per-example rewards, critiques).

    Same transport, model, temperature and seed formula as the grid's own arms
    (w2_common.judge -> common.llm), so a candidate is measured under the conditions its
    released form will be evaluated under. Notes are ordered consultation-major with a
    serial warm-up call so the shared transcript prefix is cached before the fan-out.
    """
    template = D.build_prompt(instruction)
    results, rewards, critiques = {}, {}, {}

    def one(ex):
        prompt = D.render(template, ex["transcript"], ex["note"])
        base_seed = RUN_SEED * 100000 + ex["item_index"] * 10
        ckey = f"{D.sha(prompt)[:32]}|{base_seed}"
        text = cache.get(ckey)
        if text is None:
            if budget.exhausted:
                raise BudgetStop()
            _vals, texts, metas, _r = W.judge(
                prompt, STUDENT_ROLE, 1, base_seed, temperature=1.0,
                reasoning_effort="none", parser=D.parse_score_json)
            text = texts[0]
            for m in metas:
                budget.add(STUDENT_ROLE, m.get("cost_usd_reported"))
            cache.put(ckey, text)
        reward, score, flagged, critique = D.score_example(ex, text)
        results[ex["key"]] = {"score": score, "flagged": flagged, "reward": reward,
                              "text": text[:1200]}
        rewards[ex["key"]] = reward
        critiques[ex["key"]] = critique

    by_consult = {}
    for ex in examples:
        by_consult.setdefault((ex["stratum"], ex["consultation"]), []).append(ex)

    def do_group(group):
        """One consultation: judge its first note alone to warm the shared transcript
        prefix, then fan the rest out against the now-warm cache (docs/COSTS.md lever 1)."""
        group.sort(key=lambda e: (e["kind"] != "clean", e["key"]))
        one(group[0])
        if len(group) > 1:
            with ThreadPoolExecutor(max_workers=min(3, len(group) - 1)) as inner:
                list(inner.map(one, group[1:]))

    groups = [g for _k, g in sorted(by_consult.items())]
    with ThreadPoolExecutor(max_workers=max(1, workers // 3)) as outer:
        list(outer.map(do_group, groups))     # consultations in parallel, warm-up within each
    if label:
        print(f"    {label}: {len(results)} judged | cache {cache.hits}h/{cache.misses}m "
              f"| ${budget.spent:.2f}", flush=True)
    return results, rewards, critiques


class BudgetStop(Exception):
    pass


# ---------------------------------------------------------------- Pareto selection
def pareto_parent(pool, rng):
    """GEPA's selection rule: a candidate stays in contention if it is the best anyone has
    managed on at least ONE example, and is sampled in proportion to how many examples it
    leads. A candidate that fixes the two hardest partial omissions and loses on average
    therefore survives to be a parent - which is the whole reason GEPA uses a frontier over
    instances rather than a leaderboard over means.
    """
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
        return max(pool, key=lambda c: c["vec"]["J"]), leads
    weights = [leads[c["id"]] for c in frontier]
    return rng.choices(frontier, weights=weights, k=1)[0], leads


def minibatch(examples, rng, size):
    """Stratified, not uniform. The pool is 72% errored and only 28% clean, so a uniform
    draw of 8 routinely contains no clean note at all and the false-alarm side of the
    objective gets no gradient that round. Every minibatch carries both sides and both
    omission classes."""
    by_kind = {}
    for e in examples:
        by_kind.setdefault(e["kind"], []).append(e)
    quota = [("clean", 2), ("omit-complete", 2), ("omit-partial", 1), ("add", 1), ("change", 2)]
    picked = []
    for kind, n in quota:
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

A score of {threshold} or below counts as FLAGGING the note for human review. Anything above \
{threshold} passes it.

WHAT IT IS GRADED ON. Two things at once, and either alone is worthless:
  * catching notes that are wrong or incomplete - fabricated content, altered content, and above \
all OMITTED content the consultation established. Omissions come in two flavours: the fact is \
gone entirely, or the topic is still mentioned somewhere while the load-bearing part of it \
(the working diagnosis, the dose, the reason for urgency, the red flag, the pertinent negative) \
has been dropped. The second kind is far harder and is where judges fail.
  * NOT flagging clean notes. False alarms are capped at 10% of clean notes. A judge that flags \
everything has detected nothing. Clinical notes legitimately compress: paraphrase into clinical \
language, standard abbreviations, conventional normal-exam phrasing, dropped conversational \
colour and a fact recorded once rather than three times are all correct behaviour, not loss.

THE CURRENT INSTRUCTION BLOCK
=============================
{instruction}
=============================

HOW IT JUST PERFORMED on {n} examples. Each entry gives the ground truth, what the judge scored, \
what it said, and a diagnosis of the outcome:

{traces}

YOUR TASK. Write an improved instruction block. Requirements:
  * It replaces the block above entirely and is read by a model that sees ONLY the transcript, \
the note, your block and the fixed contract. It cannot refer to these examples, to this critique, \
to "the previous instruction", or to any consultation you have just been shown - those specific \
notes will never be seen again. Generalise the lesson; never encode the instance.
  * Keep what is working. The diagnoses above mark true positives and true negatives explicitly: \
whatever calibration produced them must survive your rewrite.
  * Be concrete about the discriminations that just went wrong. Vague exhortations to "be \
thorough" cost false alarms and buy nothing.
  * Do not restate the output contract, the JSON shape, or the score range - they are appended \
after your block automatically.
  * Self-contained prose and lists. No placeholders, no {{transcript}} or {{note}} tokens.

Return the new instruction block inside these exact markers and nothing else outside them:
<INSTRUCTION>
...your improved instruction block...
</INSTRUCTION>"""


def reflect(instruction, traces, budget, n):
    prompt = REFLECT.format(contract=D.OUTPUT_CONTRACT.strip(), threshold=D.FLAG_THRESHOLD,
                            instruction=instruction, traces=traces, n=n)
    texts, meta = C.llm(prompt, REFLECTION_ROLE, temperature=1.0, k=1,
                        reasoning_effort="high", max_tokens=16000, timeout=900)
    budget.add(REFLECTION_ROLE, meta.get("cost_usd_reported"))
    m = re.search(r"<INSTRUCTION>(.*?)</INSTRUCTION>", texts[0], re.S)
    if not m:
        return None, texts[0][:400]
    new = m.group(1).strip()
    for bad in ("{transcript}", "{note}", "{summary}"):
        new = new.replace(bad, "the note" if bad != "{transcript}" else "the transcript")
    return (new if len(new) > 200 else None), None


def trace_block(examples, results, critiques, limit=8):
    out = []
    for ex in examples[:limit]:
        r = results.get(ex["key"])
        if r is None:
            continue
        said = (r["text"] or "").strip().replace("\n", " ")[:400]
        out.append(f"{critiques[ex['key']]}\n  judge said: {said}")
    return "\n\n".join(out)


# ---------------------------------------------------------------- the loop
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init-seeds", action="store_true", help="write gepa/seeds/*.txt and exit")
    ap.add_argument("--iterations", type=int, default=40, help="reflective mutation rounds")
    ap.add_argument("--cap-usd", type=float, default=120.0, help="hard credit cap for this run")
    ap.add_argument("--minibatch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260728, help="search RNG (project seed)")
    ap.add_argument("--seeds", default=None, help="comma list of seed names (default: all)")
    ap.add_argument("--resume", action="store_true",
                    help="carry on from the population in lineage.jsonl (free: cache-backed)")
    ap.add_argument("--smoke", action="store_true", help="1 seed, 8 examples, 1 iteration")
    ap.add_argument("--limit", type=int, default=None, help="cap the dev pool (debug only)")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    if args.init_seeds:
        init_seeds()
        return

    t0 = time.time()
    examples, prov = D.load_examples()
    seeds = load_seeds(args.seeds.split(",") if args.seeds else None)
    if args.smoke:
        rng0 = random.Random(args.seed)
        examples = minibatch(examples, rng0, 8)
        seeds, args.iterations = seeds[:1], 1
    elif args.limit:
        examples = examples[:args.limit]

    print(f"W-F GEPA | {len(seeds)} seed(s), {len(examples)} dev judgements, "
          f"{args.iterations} iterations, cap ${args.cap_usd:.0f}")
    print(f"  student  {STUDENT_ROLE:<14} {C.resolve_model(STUDENT_ROLE)['resolved']} "
          f"temp 1.0 reasoning=none max_tokens {W.MAX_TOKENS} (the grid's own settings)")
    print(f"  reflect  {REFLECTION_ROLE:<14} {C.resolve_model(REFLECTION_ROLE)['resolved']} "
          f"temp 1.0 reasoning=high max_tokens 16000")
    print(f"  eval-set overlap: {prov['eval_overlap']} consultations "
          f"(checked against {', '.join(prov['eval_sets_checked'])})")
    est = (len(seeds) + args.iterations * 1.4) * len(examples)
    print(f"  ~{int(est)} judge calls worst case before caching")
    W.confirm(f"proceed? hard cap ${args.cap_usd:.0f} of OpenRouter credits", args.yes)

    guard = W.SpendGuard(700.0, 900.0, False)   # the project-cumulative tripwires still apply
    budget = Budget(args.cap_usd, guard)
    cache = Cache(CACHE_PATH)
    rng = random.Random(args.seed)
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{C._git_state()[0][:7]}"
    lineage = open(LINEAGE, "a")

    def log(rec):
        lineage.write(json.dumps({"run_id": run_id, "t": round(time.time() - t0, 1), **rec}) + "\n")
        lineage.flush()

    log({"event": "run_start", "spec": SPEC, "provenance": prov,
         "student": C.resolve_model(STUDENT_ROLE)["resolved"],
         "reflection": C.resolve_model(REFLECTION_ROLE)["resolved"],
         "params": {k: v for k, v in vars(args).items()},
         "output_contract_sha256": D.sha(D.OUTPUT_CONTRACT), "preamble_sha256": D.sha(D.PREAMBLE)})

    pool, next_id = [], 0
    stopped = None
    start = [dict(s, kind="seed") for s in seeds]
    if args.resume:
        prior = load_prior_candidates()
        have = {D.sha(s["instruction"]) for s in start}
        start += [p for p in prior if D.sha(p["instruction"]) not in have]
        print(f"  resuming: {len(start) - len(seeds)} candidate(s) recovered from lineage.jsonl")
    try:
        for s in start:
            res, rew, _crit = evaluate(s["instruction"], examples, cache, budget,
                                       args.workers, f"{s['kind']} {s['name']}")
            vec = D.objective(examples, res)
            cand = {"id": next_id, "name": s["name"], "kind": s["kind"], "parent": None,
                    "instruction": s["instruction"], "vec": vec, "rewards": rew, "iter": 0,
                    "scores": {k: v["score"] for k, v in res.items()}}
            pool.append(cand)
            next_id += 1
            print(f"  {s['kind']} {s['name']:<16} {D.objective_line(vec)}")
            log({"event": "candidate", "id": cand["id"], "name": s["name"], "kind": s["kind"],
                 "parent": None, "vec": vec, "chars": len(s["instruction"]),
                 "instruction_sha256": D.sha(s["instruction"]), "instruction": s["instruction"],
                 "resumed_from": s.get("prior_id"), "spend_usd": round(budget.spent, 4)})

        for it in range(1, args.iterations + 1):
            if budget.exhausted:
                stopped = "budget cap"
                break
            parent, leads = pareto_parent(pool, rng)
            mb = minibatch(examples, rng, args.minibatch)
            p_res, p_rew, p_crit = evaluate(parent["instruction"], mb, cache, budget, args.workers)
            p_mb = sum(p_rew.values()) / len(p_rew)
            traces = trace_block(mb, p_res, p_crit, args.minibatch)
            new_instr, why = reflect(parent["instruction"], traces, budget, len(mb))
            if new_instr is None:
                print(f"  it{it:02d} parent#{parent['id']} -> reflection produced no block ({why})")
                log({"event": "mutation_failed", "iter": it, "parent": parent["id"],
                     "reason": why, "spend_usd": round(budget.spent, 4)})
                continue
            if any(D.sha(new_instr) == D.sha(c["instruction"]) for c in pool):
                print(f"  it{it:02d} parent#{parent['id']} -> duplicate instruction, skipped")
                log({"event": "mutation_duplicate", "iter": it, "parent": parent["id"]})
                continue

            c_res, c_rew, _cc = evaluate(new_instr, mb, cache, budget, args.workers)
            c_mb = sum(c_rew.values()) / len(c_rew)
            accepted = c_mb > p_mb + 1e-9
            print(f"  it{it:02d} parent#{parent['id']} ({parent['name']}, leads "
                  f"{leads[parent['id']]}/{len(examples)}) minibatch {p_mb:.3f} -> {c_mb:.3f} "
                  f"{'ACCEPT' if accepted else 'reject'} | ${budget.spent:.2f}", flush=True)
            if not accepted:
                log({"event": "mutation_rejected", "iter": it, "parent": parent["id"],
                     "minibatch_parent": round(p_mb, 4), "minibatch_child": round(c_mb, 4),
                     "chars": len(new_instr), "instruction_sha256": D.sha(new_instr),
                     "instruction": new_instr, "spend_usd": round(budget.spent, 4)})
                continue

            res, rew, _c = evaluate(new_instr, examples, cache, budget, args.workers,
                                    f"it{it:02d} full")
            vec = D.objective(examples, res)
            cand = {"id": next_id, "name": f"gepa-{next_id:02d}", "kind": "mutation",
                    "parent": parent["id"], "instruction": new_instr, "vec": vec,
                    "rewards": rew, "iter": it,
                    "scores": {k: v["score"] for k, v in res.items()}}
            pool.append(cand)
            next_id += 1
            print(f"       -> #{cand['id']} {D.objective_line(vec)}")
            log({"event": "candidate", "id": cand["id"], "name": cand["name"],
                 "kind": "mutation", "parent": parent["id"], "iter": it, "vec": vec,
                 "minibatch_parent": round(p_mb, 4), "minibatch_child": round(c_mb, 4),
                 "chars": len(new_instr), "instruction_sha256": D.sha(new_instr),
                 "instruction": new_instr, "spend_usd": round(budget.spent, 4)})

    except BudgetStop:
        stopped = "budget cap (mid-evaluation)"
    except KeyboardInterrupt:
        stopped = "interrupted"
    finally:
        cache.close()

    feasible = [c for c in pool if c["vec"]["feasible"]]
    ranked = sorted(feasible or pool,
                    key=lambda c: (c["vec"]["J"], c["vec"]["od_omit_severity_weighted"] or 0))
    winner = ranked[-1]
    open(WINNER, "w").write(D.build_prompt(winner["instruction"]))

    out = {
        "run_id": run_id, "spec": SPEC, "generated_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds"),
        "stopped_because": stopped or "iterations exhausted",
        "student_model": C.resolve_model(STUDENT_ROLE)["resolved"],
        "reflection_model": C.resolve_model(REFLECTION_ROLE)["resolved"],
        "provenance": prov, "params": vars(args),
        "spend_usd": round(budget.spent, 4), "spend_by_role": budget.by_role,
        "judge_calls": budget.calls, "cache": {"hits": cache.hits, "misses": cache.misses},
        "wall_s": round(time.time() - t0),
        "n_candidates": len(pool), "n_feasible": len(feasible),
        "winner": {"id": winner["id"], "name": winner["name"], "kind": winner["kind"],
                   "parent": winner["parent"], "iter": winner["iter"],
                   "chars": len(winner["instruction"]),
                   "instruction_sha256": D.sha(winner["instruction"]),
                   "judge_prompt_sha256": D.sha(D.build_prompt(winner["instruction"])),
                   "vec": winner["vec"]},
        "candidates": [{"id": c["id"], "name": c["name"], "kind": c["kind"],
                        "parent": c["parent"], "iter": c["iter"],
                        "chars": len(c["instruction"]), "vec": c["vec"]} for c in pool],
        # per-example 0-10 scores, so "which omissions does the winner still miss, and did
        # the seeds miss the same ones?" is answerable from this file alone.
        "per_example": {"examples": [{"key": e["key"], "kind": e["kind"],
                                      "severity": e["severity"], "consultation": e["consultation"],
                                      "stratum": e["stratum"]} for e in examples],
                        "scores": {str(c["id"]): c.get("scores", {}) for c in pool}},
    }
    json.dump(out, open(RESULTS_JSON, "w"), indent=1)
    log({"event": "run_end", **{k: v for k, v in out.items() if k != "candidates"}})
    lineage.close()
    _write_manifest(run_id, out)

    print(f"\ndone in {out['wall_s']}s | {len(pool)} candidates | ${budget.spent:.2f} credits "
          f"| stopped: {out['stopped_because']}")
    print(f"WINNER #{winner['id']} {winner['name']} (from {winner['kind']}"
          f"{'' if winner['parent'] is None else ' of #%d' % winner['parent']})")
    print(f"  {D.objective_line(winner['vec'])}")
    for c in sorted(pool, key=lambda c: c["id"]):
        if c["kind"] == "seed":
            print(f"  seed {c['name']:<16} {D.objective_line(c['vec'])}")
    print(f"\n  {os.path.relpath(WINNER, ROOT)}  <- the arm slot w2_arms.py reads")


def _write_manifest(run_id, out):
    """The spec's documented manifest divergence (section 4): the optimisation step owns
    its own manifest + ledger line, because a DSPy-shaped optimiser cannot be wrapped in
    Run() per call. The subsequent EVALUATION of the winning prompt is an ordinary
    w2_arms.py cell and goes through llm()/Run() like every other arm."""
    d = os.path.join(C.RESULTS, "gepa-omission", run_id)
    os.makedirs(d, exist_ok=True)
    commit, dirty = C._git_state()
    man = {"manifest_version": 2, "experiment": "gepa-omission", "run_id": run_id,
           "status": "complete", "spec": SPEC, "script": "gepa/gepa_optimize.py",
           "finished_utc": out["generated_utc"], "git_commit": commit,
           "git_dirty": bool(dirty), "git_dirty_paths": dirty,
           "params": out["params"], "seed": out["params"]["seed"],
           "inputs": [{"file": out["provenance"]["dev_pool"],
                       "sha256": out["provenance"]["dev_pool_sha256"]},
                      {"file": out["provenance"]["partial_seed"],
                       "sha256": out["provenance"]["partial_seed_sha256"]}],
           "models": [{"role": STUDENT_ROLE, "resolved": out["student_model"],
                       "provider": "openrouter", "temperature": 1.0,
                       "reasoning_effort": "none", "max_tokens": W.MAX_TOKENS},
                      {"role": REFLECTION_ROLE, "resolved": out["reflection_model"],
                       "provider": "openrouter", "temperature": 1.0,
                       "reasoning_effort": "high", "max_tokens": 16000}],
           "calls": {"total": out["judge_calls"]}, "cost_usd": out["spend_usd"],
           "cost_source": "openrouter_usage_reported", "spend": "openrouter_credits",
           "package_versions": C._package_versions(),
           "outputs": ["../../../gepa/judge_prompt.txt", "../../../gepa/lineage.jsonl",
                       "../../../gepa/results.json"],
           "note": "optimisation step - manifest written by the script itself per "
                   "specs/w-f-gepa-omission-judge.md section 4 (manifest divergence)."}
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"), indent=1)
    with open(os.path.join(C.RESULTS, "cost_ledger.jsonl"), "a") as f:
        f.write(json.dumps({"run_id": run_id, "experiment": "gepa-omission",
                            "date": out["generated_utc"], "status": "complete",
                            "calls": out["judge_calls"], "cost_usd": out["spend_usd"],
                            "spend": "openrouter_credits", "wall_s": out["wall_s"]}) + "\n")


if __name__ == "__main__":
    main()
