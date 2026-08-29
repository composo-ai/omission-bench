"""grade_partial_seed_severity.py - rubric severity for the seed set's ungraded 10.

Completes the second gap in master/partial_omission_seed_set.json. Ten of the 34 relabelled
partials (all authored stratum) carry no severity grade: those pairs were preserved verbatim
from the pre-master build so they have no injector target to inherit an `importance` from,
and W-A's severity backfill ran only over PASSING pairs - these were excluded before it
reached them (docs/OMISSION-DESIGN-QA.md section 3; the seed set's own `gaps` block).

Instrument: `hard_negatives_master.RUBRIC_GRADE`, imported verbatim, so the grades sit on
exactly the same prompt and the same rubric text (specs/severity-rubric.md, instrument v1)
that graded every other pair in the corpus. Nothing about the rubric is re-worded here.

Two arms, cross-family and independent, mirroring the Amendment 2026-08-09 part R regrade:

  A  constructor  anthropic/claude-opus-5, reasoning_effort medium
  B  auditor      openai/gpt-5.5,          reasoning_effort high

Both route through OpenRouter (common.llm). The regrade's arm A ran the same Anthropic model
down the plan path (`claude -p`); this run keeps it on the API transport instead, which is
what models.lock's `constructor` role exists for - its notes name "severity rubric grades"
as its job, on the same model as the plan-path construction row (Amendment 2026-08-10).
Neither arm sees the other's answer, or any prior grade.

Consensus follows the regrade verbatim: agreement gives the grade; disagreement takes the
LOWER of the two (the rubric's own conservative tie-break) and sets `flagged`, so a contested
grade is visible rather than averaged away. Flagged pairs are exactly the ones worth a human
look, and they go into the sitting pack.

Output: master/partial_seed_severity_grades.json (state + report in one file; idempotent per
(pair, arm) so a rerun costs nothing for work already done). relabel_partial_seed.py folds
the grades into the seed set on its next run.

Usage:
  .venv/bin/python grade_partial_seed_severity.py [--budget 4] [--workers 4] [--report] -y
Exit: 0 complete | 2 work left (rerun) | 3 budget stop.
"""
import argparse, json, os, sys, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from common import HERE, Run, resolve_model
import hard_negatives_master as HN
import omission_sites as OS

SPEC = OS.SPEC
EXPERIMENT = "wd-partial-seed-severity"
MASTER = OS.MASTER
SEED_SET_FILE = os.path.join(MASTER, "partial_omission_seed_set.json")
OUT_FILE = os.path.join(MASTER, "partial_seed_severity_grades.json")

ARMS = {                          # arm -> (models.lock role, reasoning_effort)
    "opus5": ("constructor", "medium"),
    "gpt55": ("auditor", "high"),
}
SEED = 20260809                   # the severity-regrade seed (regrade_severity.GPT_SEED0)
VALID = ("critical", "supporting", "peripheral")
RANK = {"critical": 2, "supporting": 1, "peripheral": 0}    # lower rank = lower grade

_lock = threading.RLock()


def load_out():
    if os.path.exists(OUT_FILE):
        return json.load(open(OUT_FILE))
    return {"created": OS.now_utc(), "spend": {"cost_usd": 0.0, "calls": 0, "by_stage": {}},
            "grades": {}}


def save_out(out):
    with _lock:
        tmp = OUT_FILE + ".tmp"
        json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1)
        os.replace(tmp, OUT_FILE)


def targets():
    """The seed-set pairs with no severity, each with its consultation transcript."""
    doc = json.load(open(SEED_SET_FILE))
    tx = {r["key"]: r["transcript"] for r in OS.consultations(("eval", "gepa_dev"))}
    out = []
    for p in doc["pairs"]:
        if p.get("severity"):
            continue
        t = tx.get(p["key"])
        if not t:
            sys.exit(f"{p['pair_id']}: no transcript for {p['key']}")
        out.append({"pair_id": p["pair_id"], "stratum": p["stratum"], "id": p["id"],
                    "key": p["key"], "split": p["split"], "type": p["type"],
                    "change": p.get("change") or "", "what": p.get("what"),
                    "fact": p.get("fact"), "transcript": t})
    return out


def grade(pair, arm, budget, out):
    """One arm's grade for one pair. Returns the record, or None on a failed call."""
    role, effort = ARMS[arm]
    what = pair["what"]
    prompt = HN.RUBRIC_GRADE.format(
        rubric=HN.RUBRIC, t=pair["transcript"][:40000], typ=pair["type"],
        change=pair["change"],
        what=what if isinstance(what, str) else json.dumps(what, ensure_ascii=False))
    obj, meta = OS.call_json(budget, f"rubric_grade_{arm}", prompt, role, effort, seed=SEED)
    if not isinstance(obj, dict) or obj.get("importance") not in VALID:
        return {"arm": arm, "error": (meta or {}).get("error") or "bad grade",
                "raw": str(obj)[:200]}
    return {"arm": arm, "role": role, "reasoning_effort": effort, "seed": SEED,
            "importance": obj["importance"], "test_fired": obj.get("test_fired"),
            "rationale": obj.get("rationale"),
            "generation_ids": (meta or {}).get("generation_ids"),
            "cost_usd": (meta or {}).get("cost"), "utc": OS.now_utc()}


def consensus(recs):
    """Regrade rule, verbatim: agreement stands; disagreement takes the LOWER grade (the
    rubric's conservative tie-break) and flags the pair for a human look."""
    g = {a: r["importance"] for a, r in recs.items() if r.get("importance")}
    if len(g) < 2:
        only = next(iter(g.values()), None)
        return {"grade": only, "flagged": None if only else True, "n_arms": len(g),
                "note": "single arm returned - not a two-model consensus" if only
                        else "no arm returned a valid grade"}
    a, b = g["opus5"], g["gpt55"]
    agree = a == b
    return {"grade": a if agree else min((a, b), key=lambda x: RANK[x]),
            "flagged": not agree, "n_arms": 2,
            "note": "both arms agree" if agree else
                    f"arms disagree (opus5 {a} / gpt55 {b}) - lower grade taken"}


def run(pairs, out, budget, args):
    todo = [p for p in pairs
            if any(a not in (out["grades"].get(p["pair_id"], {}).get("arms") or {})
                   for a in ARMS)]
    arms = ", ".join(f"{a} = {resolve_model(role)['resolved']} effort {eff}"
                     for a, (role, eff) in ARMS.items())
    print(f"pairs: {len(pairs)} ungraded in the seed set, {len(todo)} with work left | "
          f"{arms} | seed {SEED} | budget ${budget.spent:.2f}/${budget.cap:.0f}", flush=True)

    def one(pair):
        pid = pair["pair_id"]
        rec = out["grades"].setdefault(pid, {"pair_id": pid, "stratum": pair["stratum"],
                                             "id": pair["id"], "key": pair["key"],
                                             "split": pair["split"], "arms": {}})
        for arm in ARMS:
            if arm in rec["arms"] and rec["arms"][arm].get("importance"):
                continue
            if not budget.ok():
                return
            got = grade(pair, arm, budget, out)
            if got:
                with _lock:
                    rec["arms"][arm] = got
                    save_out(out)
        rec["consensus"] = consensus(rec["arms"])
        rec["fact"] = pair["fact"]
        rec["change"] = pair["change"]
        with _lock:
            save_out(out)
        c = rec["consensus"]
        arms = " / ".join(f"{a}:{rec['arms'].get(a, {}).get('importance', 'ERR')}" for a in ARMS)
        print(f"  {pid:44} {arms:34} -> {c['grade']}"
              f"{'  FLAGGED' if c['flagged'] else ''} ${budget.spent:.2f}", flush=True)

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(one, todo))
    save_out(out)
    return sum(1 for p in pairs
               if any(a not in (out["grades"].get(p["pair_id"], {}).get("arms") or {})
                      for a in ARMS))


def report(pairs, out):
    recs = [out["grades"][p["pair_id"]] for p in pairs if p["pair_id"] in out["grades"]]
    done = [r for r in recs if (r.get("consensus") or {}).get("grade")]
    flagged = [r for r in done if r["consensus"]["flagged"]]
    out.update({
        "generated_utc": OS.now_utc(), "spec": SPEC,
        "what": "rubric severity grades for the 10 relabelled partial pairs that carried "
                "none. Two independent cross-family arms on the verbatim RUBRIC_GRADE "
                "prompt and the verbatim specs/severity-rubric.md; consensus = agreement, "
                "else the lower grade (the rubric's tie-break), with disagreement flagged.",
        "models": {a: {"arm": a, "role": r, **resolve_model(r), "reasoning_effort": e,
                       "seed": SEED} for a, (r, e) in ARMS.items()},
        "prompt_sha256_16": {"rubric_grade (hard_negatives_master, verbatim)":
                             HN.sha16(HN.RUBRIC_GRADE),
                             "severity_rubric_md": HN.sha16(HN.RUBRIC)},
        "consensus_rule": "agree -> that grade; disagree -> min(grade) by "
                          "critical>supporting>peripheral, flagged for human review "
                          "(regrade_severity.analyse, verbatim)",
        "counts": {
            "n": len(recs), "graded": len(done),
            "grades": dict(Counter(r["consensus"]["grade"] for r in done)),
            "agreement": f"{len(done) - len(flagged)}/{len(done)}",
            "flagged": [r["pair_id"] for r in flagged],
            "by_arm": {a: dict(Counter(r["arms"].get(a, {}).get("importance") for r in recs))
                       for a in ARMS},
            "test_fired": {a: dict(Counter(r["arms"].get(a, {}).get("test_fired")
                                           for r in recs)) for a in ARMS},
        },
    })
    save_out(out)
    c = out["counts"]
    print(f"\nmaster/partial_seed_severity_grades.json - {c['graded']}/{c['n']} graded")
    print(f"  consensus {c['grades']} | arms agree {c['agreement']} | flagged {c['flagged']}")
    for a in ARMS:
        print(f"  {a}: {c['by_arm'][a]} | tests {c['test_fired'][a]}")
    print(f"  spend ${out['spend']['cost_usd']:.2f} over {out['spend']['calls']} calls")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budget", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--report", action="store_true", help="rewrite the report, free")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    pairs = targets()
    out = load_out()
    if args.report:
        report(pairs, out)
        return 0
    if not pairs:
        print("nothing ungraded in the seed set")
        return 0
    if not args.yes:
        try:
            if input(f"grade {len(pairs)} pairs x 2 arms (cap ${args.budget:.0f})? [y/N] "
                     ).strip().lower() != "y":
                sys.exit("aborted")
        except EOFError:
            sys.exit("aborted (non-interactive)")

    budget = OS.Budget(out, args.budget)
    with Run(EXPERIMENT, params={"n_pairs": len(pairs), "arms": list(ARMS),
                                 "seed": SEED, "budget_cap_usd": args.budget},
             seed=SEED, inputs=[SEED_SET_FILE], spec=SPEC,
             allow_dirty=args.allow_dirty) as r:
        r.register_prompts({"RUBRIC_GRADE": HN.RUBRIC_GRADE, "SEVERITY_RUBRIC": HN.RUBRIC})
        left = run(pairs, out, budget, args)
        report(pairs, out)
        r.save("partial_seed_severity_summary.json", out["counts"])
    if budget.hit.is_set():
        print(f"\nBUDGET STOP at ${budget.spent:.2f} (cap ${budget.cap:.0f}) - state saved.")
        return 3
    return 0 if left == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
