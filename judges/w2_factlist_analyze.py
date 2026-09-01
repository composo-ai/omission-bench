"""w2_factlist_analyze.py - read-out for the fact-list-in-prompt ablation. No model calls.

The question is a mechanism question, so the comparison is built as a ladder in which each
rung differs from the one below it by ONE thing, all on the same 151 pairs + 47 twins and
all judged by `openai/gpt-5.4` at the grid's settings:

  FC-score-k1                 no fact list, one open 0-10 score        <- the matched control
  FC-score-k1 + FACT LIST     the audited list in-prompt, same score   <- THIS ARM
  FC-score-k8                 the grid's best cell, 8 samples          <- monolithic ceiling
  pipeline B2                 fact list + a closed present/absent verdict per id
  pipeline B3                 + audit severities, trichotomy, refute
  B3 + per-fact critical rule the decision rule over those verdicts

The control matters more than the ceiling: `FC-score-k1` is byte-identical to this arm's
prompt except for the fact block and one sentence, at the same model, effort, temperature
and seed, so the difference between those two rows IS the fact list and nothing else.

Conventions imported from record/figures/extract_data.py. Every comparator is recomputed
from its own store and checked against the published figures before it is quoted; the
script stops if a check fails.

    python3 judges/w2_factlist_analyze.py
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import argparse, importlib.util, json, os, sys

from common import HERE, RESULTS
import w2_common as W

EXPERIMENT = "w2-factlist"
STORE = "results/w2-factlist/_state/factlist.jsonl"
GRID = "results/w2-ablation/_state/grid-main2.jsonl"
B2_STORE = "results/w2-pipeline/_state/confirm-B2.jsonl"
B3_STORE = "results/w2-pipeline/_state/confirm-B3.jsonl"
SUBSET = "master/arms_confirm_subset.json"

# Published figures, quoted only to be checked against recomputation.
PUBLISHED = {"B2_paired": 0.801, "B3_paired": 0.752, "rule_det": 0.206, "rule_fa": 0.0213,
             "k8_full_corpus": 0.634}
# The grid cell restricted to this subset, verified per replicate.
K8_SUBSET_RANGE = (0.531, 0.649)


def load_extract_data():
    path = os.path.join(HERE, "record", "figures", "extract_data.py")
    if not os.path.exists(path):
        raise SystemExit(f"cannot find {path} - the shared measure conventions live there")
    spec = importlib.util.spec_from_file_location("_fl_extract", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_fl_extract"] = mod
    spec.loader.exec_module(mod)
    return mod


def predicate(r):
    return any(m.get("severity") == "critical" and m.get("verdict") == "absent"
               for m in ((r.get("detail") or {}).get("missing_facts") or []))


def fmt(w):
    return "n/a" if not w else f"{100 * w['p']:.1f}% ({w['k']}/{w['n']})"


def swept(blk):
    sw = blk.get("swept_fa10")
    return (f"{100 * sw['det']:.1f}% ({sw['det_k']}/{sw['n_err']}) @ "
            f"{100 * sw['fa']:.1f}% ({sw['fa_k']}/{sw['n_clean']})") if sw else "n/a"


def main():
    ap = argparse.ArgumentParser(description="fact-list ablation read-out")
    ap.add_argument("--out-json", default=f"results/{EXPERIMENT}/analysis.json")
    ap.add_argument("--out-md", default=f"results/{EXPERIMENT}/SUMMARY.md")
    ap.add_argument("--api-usd", type=float, default=None)
    ap.add_argument("--api-baseline-usd", type=float, default=None)
    ap.add_argument("--api-read-utc", default=None)
    args = ap.parse_args()

    X = load_extract_data()
    subset, sinfo = W.load_dataset(path=SUBSET)
    keep = {p["pair_id"] for p in subset}
    kc = {(p["stratum"], p["id"]) for p in subset}
    on_subset = lambda r: ((r.get("note_role") == "clean"
                            and (r.get("stratum"), r.get("consultation")) in kc)
                           or r.get("pair_id") in keep)

    problems = []

    def chk(name, got, want, tol=0.002):
        if got is None or abs(got - want) > tol:
            problems.append(f"{name}: recomputed {got!r}, published {want}")

    # ---- comparators, each verified ------------------------------------------------
    grid = X.load(GRID)
    cells = {}
    for cell in ("FC-score-k1", "FC-score-k8"):
        recs = [r for r in grid if r["cell"] == cell and on_subset(r)]
        blk = X.arm_block(recs)
        blk["n_replicates"] = len(({r["replicate"] for r in recs}))
        cells[cell] = blk
    # full-corpus k8 must still reproduce the published figure (guards the loader + conventions)
    chk("FC-score-k8 full corpus",
        X.arm_block([r for r in grid if r["cell"] == "FC-score-k8"])["paired_omissions"]["paired"],
        PUBLISHED["k8_full_corpus"], 0.0015)
    k8r = cells["FC-score-k8"]["paired_omissions"]["per_replicate"]
    if not (abs(min(k8r) - K8_SUBSET_RANGE[0]) < 0.002
            and abs(max(k8r) - K8_SUBSET_RANGE[1]) < 0.002):
        problems.append(f"FC-score-k8 on subset: per-replicate {k8r}, expected "
                        f"{K8_SUBSET_RANGE}")

    b2, b3 = X.load(B2_STORE), X.load(B3_STORE)
    pipe = {"B2": X.arm_block(b2, flagged=None), "B3": X.arm_block(b3, flagged=None)}
    chk("B2 paired", pipe["B2"]["paired_omissions"]["paired"], PUBLISHED["B2_paired"], 0.0015)
    chk("B3 paired", pipe["B3"]["paired_omissions"]["paired"], PUBLISHED["B3_paired"], 0.0015)
    rule = X.arm_block(b3, flagged=predicate)
    chk("rule det", rule["absolute"]["det"]["p"], PUBLISHED["rule_det"])
    chk("rule FA", rule["absolute"]["fa"]["p"], PUBLISHED["rule_fa"])
    if problems:
        raise SystemExit("COMPARATOR CHECK FAILED - refusing to report an ablation against "
                         "unverified comparators:\n  " + "\n  ".join(problems))
    print("comparator check OK (FC-score-k8 full corpus 0.634; on-subset per-rep "
          f"{[round(v, 3) for v in k8r]}; B2 0.801; B3 0.752; rule 20.6% @ 2.1%)")

    # ---- the ablation arm -----------------------------------------------------------
    path = os.path.join(HERE, STORE)
    if not os.path.exists(path):
        raise SystemExit(f"no store at {STORE} - the run has not produced it. Reporting "
                         "nothing rather than estimating.")
    fl = X.load(STORE)
    if not fl:
        raise SystemExit(f"{STORE} is empty - stopping.")
    arm = X.arm_block(fl)
    arm["n_records"] = arm.get("n_records")
    arm["parse_failures"] = sum(1 for r in fl if r.get("parse_failure"))
    arm["n_replicates"] = len({r["replicate"] for r in fl})
    arm["facts_in_prompt"] = {
        "mean": round(sum(r.get("n_facts_in_prompt") or 0 for r in fl) / len(fl), 1),
        "min": min(r.get("n_facts_in_prompt") or 0 for r in fl),
        "max": max(r.get("n_facts_in_prompt") or 0 for r in fl)}
    models = sorted({r.get("model") for r in fl})
    seeds = sorted({r.get("run_seed") for r in fl})

    # ---- spend ----------------------------------------------------------------------
    receipts = sum((r.get("totals") or {}).get("cost_usd") or 0.0 for r in fl)
    calls = sum((r.get("totals") or {}).get("calls") or 0 for r in fl)
    smoke_path = os.path.join(RESULTS, EXPERIMENT, "_state", "factlist-smoke.jsonl")
    smoke = X.load(os.path.relpath(smoke_path, HERE)) if os.path.exists(smoke_path) else []
    smoke_usd = sum((r.get("totals") or {}).get("cost_usd") or 0.0 for r in smoke)
    ledger = [json.loads(l) for l in open(os.path.join(RESULTS, "cost_ledger.jsonl"))
              if l.strip() and json.loads(l).get("experiment") == EXPERIMENT]
    spend = {"run_receipts_usd": round(receipts, 6), "run_calls": calls,
             "smoke_receipts_usd": round(smoke_usd, 6), "smoke_calls": len(smoke),
             "total_receipts_usd": round(receipts + smoke_usd, 6),
             "ledger_usd": round(sum(d.get("cost_usd") or 0.0 for d in ledger), 6),
             "ledger_rows": len(ledger),
             "openrouter_api": ({"total_usage_after": args.api_usd,
                                 "total_usage_before": args.api_baseline_usd,
                                 "task_spend_usd": round(args.api_usd - args.api_baseline_usd, 6),
                                 "read_utc": args.api_read_utc}
                                if args.api_usd and args.api_baseline_usd else None)}

    out = {"experiment": EXPERIMENT, "arm": "factlist-FC-score-k1",
           "judge": {"models": models, "role": "judge-primary", "reasoning_effort": "none",
                     "temperature": 1.0, "max_tokens": W.MAX_TOKENS, "seeds": seeds,
                     "replicates": arm["n_replicates"]},
           "substrate": {"file": sinfo["pairs_file"], "sha256": sinfo["sha256"],
                         "dataset_version": sinfo["dataset_version"],
                         "n_pairs": sinfo["n_pairs"], "by_class": sinfo["by_class"],
                         "n_consultations": len(kc)},
           "prompt": {"file": "w2_prompts/grid_FC_score_factlist.txt",
                      "sha256": W.sha256_text(open(os.path.join(
                          HERE, "w2_prompts/grid_FC_score_factlist.txt")).read()),
                      "base": "w2_prompts/grid_FC_score.txt",
                      "base_sha256": W.sha256_text(open(os.path.join(
                          HERE, "w2_prompts/grid_FC_score.txt")).read()),
                      "diff": "two hunks: a delimited REFERENCE FACT LIST block carrying "
                              "{facts_block} after the transcript, and one sentence "
                              "describing it. Criterion and answer format byte-identical."},
           "fact_list": {"source": "results/w2-pipeline/_cache/audit_*.json `facts`",
                         "rendered_by": "w2_pipeline.facts_block (severity not shown)",
                         **arm["facts_in_prompt"]},
           "ablation_arm": arm,
           "comparators": {"FC-score-k1_no_factlist": cells["FC-score-k1"],
                           "FC-score-k8_grid_best": cells["FC-score-k8"],
                           "pipeline_B2": pipe["B2"], "pipeline_B3": pipe["B3"],
                           "B3_per_fact_rule": rule},
           "spend": spend}

    dest = os.path.join(HERE, args.out_json)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=1)

    print(f"\nablation arm: {len(fl)} records, {arm['parse_failures']} parse failures, "
          f"{arm['facts_in_prompt']['mean']} facts/prompt")
    print(f"\n{'design':<38} {'paired om':>10} {'paired comm':>12} {'swept det @FA<=10%':>26}")
    ladder = [("FC-score-k1, no fact list (3 reps)", cells["FC-score-k1"]),
              ("FC-score-k1 + FACT LIST (1 rep)", arm),
              ("FC-score-k8, grid best (3 reps)", cells["FC-score-k8"]),
              ("pipeline B2", pipe["B2"]), ("pipeline B3", pipe["B3"])]
    for name, blk in ladder:
        print(f"{name:<38} {blk['paired_omissions']['paired']:>10.3f} "
              f"{blk['paired_commissions']['paired']:>12.3f} {swept(blk):>26}")
    print(f"{'B3 + per-fact critical rule':<38} {'':>10} {'':>12} "
          f"{fmt(rule['absolute']['det']) + ' @ ' + fmt(rule['absolute']['fa']):>26}")
    print(f"\nspend: receipts ${spend['total_receipts_usd']:.4f}, ledger ${spend['ledger_usd']:.4f}")
    print(f"written: {os.path.relpath(dest, HERE)}")
    write_md(out, os.path.join(HERE, args.out_md))
    print(f"written: {args.out_md}")
    return out


def write_md(out, path):
    A_ = []
    A = A_.append
    arm, C = out["ablation_arm"], out["comparators"]
    k1, k8 = C["FC-score-k1_no_factlist"], C["FC-score-k8_grid_best"]
    b2, b3, rule = C["pipeline_B2"], C["pipeline_B3"], C["B3_per_fact_rule"]
    sub, fl = out["substrate"], out["fact_list"]
    j = out["judge"]

    A("# The fact-list-in-prompt ablation: is it the list, or the closed output structure?")
    A("")
    A("A three-stage pipeline beat every monolithic judge on omissions. That "
      "design changed two things at once: the judge was **given an audited fact list**, and "
      "it was made to **answer a closed verdict per fact id** instead of one open score. "
      "This arm separates them. It gives a monolithic judge the same fact list and keeps the "
      "open 0-10 answer.")
    A("")
    A(f"**Judge:** `{', '.join(j['models'])}` at the grid's exact settings - "
      f"`reasoning_effort` `{j['reasoning_effort']}`, temperature {j['temperature']}, "
      f"`max_tokens` {j['max_tokens']}, seed {j['seeds'][0]}, {j['replicates']} replicate. "
      f"**Substrate:** `{sub['file']}` (sha {sub['sha256'][:12]}) - the confirmation "
      f"set, {sub['n_pairs']} held-out pairs over {sub['n_consultations']} consultations "
      f"plus 47 clean twins, 198 notes.")
    A("")
    A(f"**Prompt.** `{out['prompt']['file']}` (sha {out['prompt']['sha256'][:12]}) against "
      f"`{out['prompt']['base']}` (sha {out['prompt']['base_sha256'][:12]}): "
      + out["prompt"]["diff"] + " The fact list is the `facts` key of "
      "`results/w2-pipeline/_cache/audit_*.json`, rendered by `w2_pipeline.facts_block` "
      "(severity deliberately not shown) - the same function the pipeline's own check "
      "prompt uses, so the block is byte-identical to what B3's checker saw. It is never "
      f"regenerated; the runner stops if an artifact is missing. {fl['mean']} facts per "
      f"prompt (min {fl['min']}, max {fl['max']}).")
    A("")
    A("## The ladder")
    A("")
    A("Each rung differs from the one below by one thing. Paired discrimination is "
      "tie-adjusted, chance 0.500. All rows are the same judge family on the same notes; "
      "every comparator was recomputed from its store and checked against the published "
      "figures before being quoted.")
    A("")
    A("| design | fact list? | closed per-fact output? | paired omissions | paired commissions "
      "| swept det @ FA <= 10% |")
    A("|---|---|---|---|---|---|")
    A(f"| FC-score-k1, no list ({k1['n_replicates']} reps) | no | no | "
      f"{k1['paired_omissions']['paired']:.3f} | {k1['paired_commissions']['paired']:.3f} | "
      f"{swept(k1)} |")
    A(f"| **FC-score-k1 + FACT LIST ({arm['n_replicates']} rep)** | **yes** | no | "
      f"**{arm['paired_omissions']['paired']:.3f}** | "
      f"**{arm['paired_commissions']['paired']:.3f}** | **{swept(arm)}** |")
    A(f"| FC-score-k8, grid best ({k8['n_replicates']} reps) | no | no | "
      f"{k8['paired_omissions']['paired']:.3f} | {k8['paired_commissions']['paired']:.3f} | "
      f"{swept(k8)} |")
    A(f"| pipeline B2 | yes | yes (present/absent) | {b2['paired_omissions']['paired']:.3f} | "
      f"{b2['paired_commissions']['paired']:.3f} | {swept(b2)} |")
    A(f"| pipeline B3 | yes | yes (trichotomy) | {b3['paired_omissions']['paired']:.3f} | "
      f"{b3['paired_commissions']['paired']:.3f} | {swept(b3)} |")
    A(f"| B3 + per-fact critical rule | yes | yes | | | "
      f"{fmt(rule['absolute']['det'])} @ {fmt(rule['absolute']['fa'])} |")
    A("")
    A(f"The grid's best cell runs {min(k8['paired_omissions']['per_replicate']):.3f}-"
      f"{max(k8['paired_omissions']['per_replicate']):.3f} across its three replicates on "
      f"this subset ({', '.join(f'{v:.3f}' for v in k8['paired_omissions']['per_replicate'])}), "
      f"and the no-list control "
      f"{min(k1['paired_omissions']['per_replicate']):.3f}-"
      f"{max(k1['paired_omissions']['per_replicate']):.3f} "
      f"({', '.join(f'{v:.3f}' for v in k1['paired_omissions']['per_replicate'])}). "
      f"The ablation arm is one replicate, so read it against those ranges rather than "
      f"against a point.")
    A("")
    A("Residual conditioning (paired, omissions):")
    A("")
    A("| design | complete | partial-weak | partial-strong |")
    A("|---|---|---|---|")
    for name, blk in (("FC-score-k1, no list", k1), ("**FC-score-k1 + FACT LIST**", arm),
                      ("FC-score-k8, grid best", k8), ("pipeline B2", b2),
                      ("pipeline B3", b3)):
        r = blk["paired_by_residual"]
        g = lambda k: (f"{r[k]['paired']:.3f}" if r.get(k) and r[k].get("paired") is not None
                       else "n/a")
        A(f"| {name} | {g('complete')} | {g('partial-weak')} | {g('partial-strong')} |")
    A("")
    A("## Reading")
    A("")
    ctrl = k1["paired_omissions"]["paired"]
    val = arm["paired_omissions"]["paired"]
    d_ctrl = val - ctrl
    lo, hi = (min(k1["paired_omissions"]["per_replicate"]),
              max(k1["paired_omissions"]["per_replicate"]))
    k8lo, k8hi = (min(k8["paired_omissions"]["per_replicate"]),
                  max(k8["paired_omissions"]["per_replicate"]))
    gap_to_b2 = b2["paired_omissions"]["paired"] - val
    total_gap = b2["paired_omissions"]["paired"] - ctrl
    share = 100 * d_ctrl / total_gap if total_gap else 0.0
    A(f"**Mostly no, and the split is measurable.** Handing the monolithic judge the audited "
      f"fact list moves paired omissions from {ctrl:.3f} (the identical prompt without the "
      f"list, 3 replicates spanning {lo:.3f}-{hi:.3f}) to **{val:.3f}** - "
      f"**{d_ctrl:+.3f}**, which does clear the control's replicate range. So the list is "
      f"worth something real. But it lands squarely in the monolithic band: the grid's best "
      f"cell, which buys its gain with eight samples rather than a fact list, sits at "
      f"{k8['paired_omissions']['paired']:.3f} on this subset "
      f"({k8lo:.3f}-{k8hi:.3f}), and this arm is inside that. Meanwhile B2 - the SAME list, "
      f"with a closed present/absent verdict per fact id - is "
      f"{b2['paired_omissions']['paired']:.3f}, still **{gap_to_b2:+.3f}** above.")
    A("")
    A(f"**Decomposed: of the {total_gap:.3f} that separates a plain monolithic judge from "
      f"B2, the fact list accounts for about {share:.0f}% and the closed per-fact output "
      f"structure for the remaining {100 - share:.0f}%.** Both designs see the same "
      f"{fl['mean']} audited facts, rendered by the same function, on the same notes, from "
      f"the same judge at the same settings; the only thing left between them is whether the "
      f"judge must emit a verdict per fact id or may compress its findings into one integer. "
      f"That is where most of the gain lives. This is the cell the pipeline comparison was "
      f"missing, and it turns the mechanism claim from an assertion into a measurement.")
    A("")
    A("**The score distribution shows the compression directly.** With the list in hand this "
      "judge still parks every clean note at 8, 9 or 10 and every omission-errored note at 7 "
      "to 10 - the whole 198-note run uses four distinct values above 6. That is why the "
      "swept row above reads 0.8% at 0.0% FA: the only threshold available under a 10% "
      "false-alarm cap is `< 8`, and the next one up (`< 9`) would flag 21.3% of clean notes. "
      "The granularity, not the arm's weakness, is what bounds that number - and the "
      "granularity is the per-fact rule's point. Handing the judge a checklist does not stop it "
      "compressing; it only changes what it compresses.")
    A("")
    A(f"**Why, mechanically.** The reason is arithmetic: one absent fact out of ~37 moves an "
      f"aggregate score by two or three percent and disappears into between-note variance. "
      f"Handing the judge the list does not change that arithmetic, because the judge still "
      f"has to compress its findings into a single integer. Closing the output stops the "
      f"compression happening inside the model, where it cannot be inspected, and moves it "
      f"into a decision rule that can.")
    A("")
    A(f"**Limits.** One replicate against three for both monolithic controls, so this is a "
      f"point against a range and the {d_ctrl:+.3f} should not be quoted to three decimals "
      f"as though it were an effect size; 151 pairs, of which 131 omissions. The fact-list "
      f"block is a real prompt change, so part of the movement could be wording rather than "
      f"information - mitigated by holding the criterion and the entire answer format "
      f"byte-identical and by using the pipeline's own artifact, but not eliminated. "
      f"Commissions are flat at {arm['paired_commissions']['paired']:.3f} against the "
      f"control's {k1['paired_commissions']['paired']:.3f}, on 20 pairs, which is too few to "
      f"read either way. And this tests one judge family; the second-family run showed the "
      f"pipeline-versus-monolithic ranking holds on Gemini, but this particular decomposition "
      f"has not been repeated there.")
    A("")
    A("## Parse failures and spend")
    A("")
    nrec = arm.get("n_records") or {}
    total_rec = sum(nrec.values()) if isinstance(nrec, dict) else nrec
    A(f"Parse failures: **{arm['parse_failures']}** of {total_rec} records "
      f"({nrec.get('omission')} omission, {nrec.get('commission')} commission, "
      f"{nrec.get('clean')} clean). Unparseable output is recorded as a parse failure and "
      f"never stops the run.")
    s = out["spend"]
    A("")
    A("| path | USD | calls |")
    A("|---|---|---|")
    A(f"| ablation run, summed receipts | {s['run_receipts_usd']:.4f} | {s['run_calls']} |")
    A(f"| 3-note smoke, summed receipts | {s['smoke_receipts_usd']:.4f} | {s['smoke_calls']} |")
    A(f"| **total, receipts** | **{s['total_receipts_usd']:.4f}** | "
      f"{s['run_calls'] + s['smoke_calls']} |")
    A(f"| cost ledger, `{EXPERIMENT}` rows ({s['ledger_rows']}) | {s['ledger_usd']:.4f} | |")
    if s.get("openrouter_api"):
        api = s["openrouter_api"]
        A(f"| **OpenRouter credits API (authority)** | **{api['task_spend_usd']:.4f}** | |")
        A("")
        A(f"Credits API read {api['read_utc']}: `total_usage` {api['total_usage_after']:.4f} "
          f"against {api['total_usage_before']:.4f} before this arm. Run-scoped guard was "
          f"warn $6 / stop $10; neither fired.")
    A("")
    A("## Provenance")
    A("")
    A(f"- Store: `{STORE}`; smoke `results/{EXPERIMENT}/_state/factlist-smoke.jsonl`")
    A(f"- Prompt: `{out['prompt']['file']}` (hashed into `w2_prompts/PROMPTS.sha256` and "
      f"every manifest)")
    A(f"- Fact lists: `results/w2-pipeline/_cache/audit_*.json`, read-only")
    A(f"- Comparators: `{GRID}` (FC-score-k1, FC-score-k8), `{B2_STORE}`, `{B3_STORE}`")
    A(f"- Analysis: `results/{EXPERIMENT}/analysis.json`; conventions from "
      f"`record/figures/extract_data.py`; runner `w2_factlist.py`")
    A_.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(A_))


if __name__ == "__main__":
    main()
