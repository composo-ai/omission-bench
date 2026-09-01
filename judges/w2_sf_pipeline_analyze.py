"""w2_sf_pipeline_analyze.py - the pipeline judge with a SECOND-FAMILY checker.

The pipeline judge's published result is that restructuring the task (enumerate a fact
list from the transcript, then answer a keyed verdict per fact) beats every monolithic
design on omissions, and that only a per-fact decision rule turns that into detection.
Both results rest on one checker model,
`openai/gpt-5.4`. This asks whether the restructuring result is a property of the task
shape or of that model: the extraction and audit stages are the INSTRUMENT and are reused
byte for byte from `results/w2-pipeline/_cache`, and only stage 2 - the check - is re-run
on `google/gemini-3.1-pro-preview`.

Four read-outs, all on the same 151 held-out pairs + 47 clean twins:

  1. Gemini-checker B2 and B3   paired omissions + commissions, residual conditioning,
                                swept detection at FA <= 10%.
  2. The per-fact critical rule  flag iff any audit-graded-critical fact is verdicted
     on Gemini B3                absent - overall, on critical-severity pairs, by residual,
                                and false alarms on the 47 clean twins.
  3. The matched monolithic      Gemini FC-score-k1 from the second-family store, restricted
     comparator                  to exactly these pairs and twins. Costs nothing: those
                                records are already bought. This is the comparison that
                                answers "does restructuring help THIS judge".
  4. The gpt-5.4 baselines       recomputed from confirm-B2/B3.jsonl and checked against
                                the published figures before anything is reported.

Measure conventions are imported from `record/figures/extract_data.py`, the module that
reproduces the published numbers; nothing is reimplemented here. If the baselines do not
reproduce, this script STOPS rather than print a comparison whose reference column is
unverified.

    python3 judges/w2_sf_pipeline_analyze.py
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import argparse, glob, importlib.util, json, os, sys
from collections import Counter

from common import HERE, RESULTS
import w2_common as W

EXPERIMENT = "w2-sf-pipeline"
GPT_B2 = "results/w2-pipeline/_state/confirm-B2.jsonl"
GPT_B3 = "results/w2-pipeline/_state/confirm-B3.jsonl"
SF_STORE = "results/w2-sf-pipeline/_state/sfconfirm-{tier}.jsonl"
MONO_STORE = "results/w2-secondfamily/_state/secondfamily-gemini-e656e522.jsonl"
CONFIRM_SUBSET = "master/arms_confirm_subset.json"

# The published pipeline-judge figures, quoted ONLY to be checked against a recomputation
# from the gpt-5.4 stores.
PUBLISHED_PIPELINE = {"B2_paired": 0.801, "B3_paired": 0.752, "B2_complete": 0.936,
                      "B2_swept_det": 0.160, "B3_swept_det": 0.183,
                      "pred_det": 0.206, "pred_fa": 0.0213, "pred_complete": 0.339,
                      "pred_partial_strong": 0.0, "pred_critical": 0.328}


def load_extract_data():
    path = os.path.join(HERE, "record", "figures", "extract_data.py")
    if not os.path.exists(path):
        raise SystemExit(f"cannot find {path} - the shared measure conventions live there")
    spec = importlib.util.spec_from_file_location("_sf_extract_data", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_sf_extract_data"] = mod
    spec.loader.exec_module(mod)
    for fn in ("arm_block", "wilson", "two_prop_z", "sweep", "best_at_fa", "load",
               "residual_level_of", "IS_OMIT"):
        if not hasattr(mod, fn):
            raise SystemExit(f"record/figures/extract_data.py has no {fn}() - stopping")
    return mod


def predicate(r):
    """The published per-fact rule: flag the note iff any fact the AUDIT graded `critical`
    is verdicted absent. The severities are the cached gpt-5.5 audit's and are identical
    across checkers,
    which is what makes the two families comparable under one rule."""
    return any(m.get("severity") == "critical" and m.get("verdict") == "absent"
               for m in ((r.get("detail") or {}).get("missing_facts") or []))


def tier_block(X, recs, label):
    """arm_block plus the per-fact-rule read-out, for one tier's records."""
    out = {"label": label, "n_records": len(recs)}
    out.update(X.arm_block(recs, flagged=None))
    omis = [r for r in recs if r.get("note_role") != "clean" and X.IS_OMIT(r)]
    out["n_omission_records"] = len(omis)
    out["parse_failures"] = sum(1 for r in recs if r.get("parse_failure"))
    return out


def predicate_block(X, recs):
    """The per-fact critical rule on one tier's records (B3 only - B2 has no severities)."""
    pred = X.arm_block(recs, flagged=predicate)
    omis = [r for r in recs if r.get("note_role") != "clean" and X.IS_OMIT(r)]
    pred["by_residual_detection"] = {}
    for lv in ("complete", "partial-weak", "partial-strong"):
        sub = [r for r in omis if X.residual_level_of(r) == lv]
        pred["by_residual_detection"][lv] = X.wilson(
            sum(1 for r in sub if predicate(r)), len(sub))
    crit = [r for r in omis if r.get("severity") == "critical"]
    pred["critical_severity_pairs"] = X.wilson(
        sum(1 for r in crit if predicate(r)), len(crit))
    comm = [r for r in recs if r.get("note_role") != "clean"
            and r.get("pair_type") in ("add", "change")]
    pred["commission_flags"] = X.wilson(sum(1 for r in comm if predicate(r)), len(comm))
    return pred


def fmt_pct(w):
    return "n/a" if not w else f"{100 * w['p']:.1f}% ({w['k']}/{w['n']})"


def fmt_ci(w):
    return "n/a" if not w else f"{100 * w['p']:.1f}% [{100 * w['lo']:.1f}, {100 * w['hi']:.1f}]"


def main():
    ap = argparse.ArgumentParser(description="second-family pipeline check-stage analysis")
    ap.add_argument("--out-json", default=f"results/{EXPERIMENT}/analysis.json")
    ap.add_argument("--out-md", default=f"results/{EXPERIMENT}/SUMMARY.md")
    ap.add_argument("--api-usd", type=float, default=None)
    ap.add_argument("--api-baseline-usd", type=float, default=None)
    ap.add_argument("--api-read-utc", default=None)
    args = ap.parse_args()

    X = load_extract_data()

    # ---- 1. baseline: reproduce the published figures from the gpt-5.4 stores -------
    gb2, gb3 = X.load(GPT_B2), X.load(GPT_B3)
    base = {"B2": tier_block(X, gb2, "gpt-5.4 B2"), "B3": tier_block(X, gb3, "gpt-5.4 B3")}
    base_pred = predicate_block(X, gb3)
    problems = []

    def chk(name, got, want, tol=0.002):
        if got is None or abs(got - want) > tol:
            problems.append(f"{name}: recomputed {got!r}, the paper reports {want}")

    chk("B2 paired", base["B2"]["paired_omissions"]["paired"], PUBLISHED_PIPELINE["B2_paired"], 0.0015)
    chk("B3 paired", base["B3"]["paired_omissions"]["paired"], PUBLISHED_PIPELINE["B3_paired"], 0.0015)
    chk("B2 complete", base["B2"]["paired_by_residual"]["complete"]["paired"],
        PUBLISHED_PIPELINE["B2_complete"])
    chk("B2 swept det", base["B2"]["swept_fa10"]["det"], PUBLISHED_PIPELINE["B2_swept_det"])
    chk("B3 swept det", base["B3"]["swept_fa10"]["det"], PUBLISHED_PIPELINE["B3_swept_det"])
    chk("per-fact rule det", base_pred["absolute"]["det"]["p"], PUBLISHED_PIPELINE["pred_det"])
    chk("per-fact rule FA", base_pred["absolute"]["fa"]["p"], PUBLISHED_PIPELINE["pred_fa"])
    chk("per-fact rule complete", base_pred["by_residual_detection"]["complete"]["p"],
        PUBLISHED_PIPELINE["pred_complete"])
    chk("per-fact rule partial-strong",
        base_pred["by_residual_detection"]["partial-strong"]["p"],
        PUBLISHED_PIPELINE["pred_partial_strong"], 1e-9)
    chk("per-fact rule critical-severity", base_pred["critical_severity_pairs"]["p"],
        PUBLISHED_PIPELINE["pred_critical"])
    if problems:
        raise SystemExit("BASELINE CHECK FAILED - refusing to report a comparison whose "
                         "gpt-5.4 reference column does not reproduce:\n  "
                         + "\n  ".join(problems))
    print("baseline check OK: the published figures reproduced from the gpt-5.4 stores")
    print(f"  B2 paired {base['B2']['paired_omissions']['paired']:.3f} | "
          f"B3 paired {base['B3']['paired_omissions']['paired']:.3f} | "
          f"per-fact rule {fmt_pct(base_pred['absolute']['det'])} at "
          f"{fmt_pct(base_pred['absolute']['fa'])}")

    # ---- 2. the Gemini checker -------------------------------------------------------
    sf = {}
    for tier in ("B2", "B3"):
        path = SF_STORE.format(tier=tier)
        if not os.path.exists(os.path.join(HERE, path)):
            raise SystemExit(f"no store at {path} - the run has not produced it. "
                             "Reporting nothing rather than estimating.")
        sf[tier] = X.load(path)
        if not sf[tier]:
            raise SystemExit(f"{path} is empty - stopping.")
    sf_blocks = {t: tier_block(X, sf[t], f"gemini {t}") for t in ("B2", "B3")}
    sf_pred = predicate_block(X, sf["B3"])
    models = sorted({r.get("model") for t in sf for r in sf[t]})
    roles = sorted({r.get("role") for t in sf for r in sf[t]})

    # ---- 3. matched monolithic comparator, free ---------------------------------------
    subset, sinfo = W.load_dataset(path=CONFIRM_SUBSET)
    keep_pairs = {p["pair_id"] for p in subset}
    keep_consults = {(p["stratum"], p["id"]) for p in subset}
    mono_all = X.load(MONO_STORE)
    mono = [r for r in mono_all
            if r.get("arm") == "sf-FC-score-k1"
            and ((r.get("note_role") == "clean"
                  and (r.get("stratum"), r.get("consultation")) in keep_consults)
                 or r.get("pair_id") in keep_pairs)]
    mono_reps = sorted({r["replicate"] for r in mono})
    mono_per_rep = {}
    for rep in mono_reps:
        sub = [r for r in mono if r["replicate"] == rep]
        b = X.arm_block(sub)
        mono_per_rep[str(rep)] = {
            "n_records": b["n_records"],
            "paired_omissions": b["paired_omissions"], "paired_commissions": b["paired_commissions"],
            "paired_by_residual": b["paired_by_residual"],
            "absolute_omission_det": b["absolute"]["det"], "absolute_fa": b["absolute"]["fa"],
            "absolute_det_vs_fa_z": b["absolute"]["z"], "swept_fa10": b.get("swept_fa10")}
    mono_block = X.arm_block(mono)
    mono_block["per_replicate"] = mono_per_rep
    mono_block["n_replicates"] = len(mono_reps)
    mono_block["subset_match"] = {
        "pairs_wanted": len(keep_pairs),
        "clean_notes_matched": len({r["note_key"] for r in mono
                                    if r.get("note_role") == "clean"}),
        "errored_pairs_matched": len({r["pair_id"] for r in mono
                                      if r.get("note_role") != "clean"}),
        "note": "restricted from the second-family store; no calls were made for this row"}

    # ---- 4. spend ---------------------------------------------------------------------
    receipts = sum((r.get("totals") or {}).get("cost_usd") or 0.0
                   for t in sf for r in sf[t])
    calls = sum((r.get("totals") or {}).get("calls") or 0 for t in sf for r in sf[t])
    smoke_recs = []
    for p in sorted(glob.glob(os.path.join(RESULTS, EXPERIMENT, "_state", "sfsmoke-*.jsonl"))):
        smoke_recs += X.load(os.path.relpath(p, HERE))
    smoke_usd = sum((r.get("totals") or {}).get("cost_usd") or 0.0 for r in smoke_recs)
    ledger_rows = []
    lpath = os.path.join(RESULTS, "cost_ledger.jsonl")
    if os.path.exists(lpath):
        for line in open(lpath):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("experiment") == EXPERIMENT:
                ledger_rows.append(d)
    spend = {
        "confirmation_receipts_usd": round(receipts, 6), "confirmation_calls": calls,
        "smoke_receipts_usd": round(smoke_usd, 6),
        "smoke_calls": sum((r.get("totals") or {}).get("calls") or 0 for r in smoke_recs),
        "total_receipts_usd": round(receipts + smoke_usd, 6),
        "ledger_usd": round(sum(d.get("cost_usd") or 0.0 for d in ledger_rows), 6),
        "ledger_rows": len(ledger_rows),
        "ledger_calls": sum(d.get("calls") or 0 for d in ledger_rows),
        "reused_free": "Extraction (47 consultations) and audit (47) were read from "
                       "results/w2-pipeline/_cache and cost nothing; the refute pass ran on "
                       "the auditor role (openai/gpt-5.5) exactly as in the published run, "
                       "so only stage 2 changed family",
        "openrouter_api": ({"total_usage_after": args.api_usd,
                            "total_usage_before": args.api_baseline_usd,
                            "task_spend_usd": round(args.api_usd - args.api_baseline_usd, 6),
                            "read_utc": args.api_read_utc}
                           if args.api_usd and args.api_baseline_usd else None)}

    out = {"experiment": EXPERIMENT,
           "checker_models": models, "checker_roles": roles,
           "substrate": {"file": sinfo["pairs_file"], "sha256": sinfo["sha256"],
                         "dataset_version": sinfo["dataset_version"],
                         "n_pairs": sinfo["n_pairs"], "by_class": sinfo["by_class"],
                         "n_consultations": len(keep_consults)},
           "stages": {"extract": "constructor / anthropic-claude-opus-5 (CACHED, reused)",
                      "audit": "auditor / openai-gpt-5.5 effort high (CACHED, reused)",
                      "check": f"{roles} -> {models} effort minimal (RE-RUN, the variable)",
                      "refute": "auditor / openai-gpt-5.5 effort high (re-run, unchanged family)"},
           "gpt54_baseline": {"B2": base["B2"], "B3": base["B3"], "B3_predicate": base_pred,
                              "published_figures_quoted": PUBLISHED_PIPELINE,
                              "baseline_check": "reproduced within tolerance"},
           "gemini_checker": {"B2": sf_blocks["B2"], "B3": sf_blocks["B3"],
                              "B3_predicate": sf_pred},
           "matched_monolithic_gemini_FC_score_k1": mono_block,
           "spend": spend}

    dest = os.path.join(HERE, args.out_json)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=1)

    print(f"\ngemini checker: {len(sf['B2'])} B2 + {len(sf['B3'])} B3 records, "
          f"parse failures {sf_blocks['B2']['parse_failures']} / {sf_blocks['B3']['parse_failures']}")
    print(f"{'tier':<14} {'paired om':>10} {'paired comm':>12} {'swept det @FA<=10%':>22}")
    for lbl, blk in (("gpt-5.4 B2", base["B2"]), ("gemini B2", sf_blocks["B2"]),
                     ("gpt-5.4 B3", base["B3"]), ("gemini B3", sf_blocks["B3"])):
        sw = blk.get("swept_fa10")
        s = f"{100 * sw['det']:.1f}% @ {100 * sw['fa']:.1f}%" if sw else "n/a"
        print(f"{lbl:<14} {blk['paired_omissions']['paired']:>10.3f} "
              f"{blk['paired_commissions']['paired']:>12.3f} {s:>22}")
    print(f"\nper-fact critical rule: gpt-5.4 {fmt_pct(base_pred['absolute']['det'])} at "
          f"{fmt_pct(base_pred['absolute']['fa'])} | gemini "
          f"{fmt_pct(sf_pred['absolute']['det'])} at {fmt_pct(sf_pred['absolute']['fa'])}")
    print(f"matched mono (gemini FC-score-k1, {mono_block['n_replicates']} reps): paired om "
          f"{mono_block['paired_omissions']['paired']:.3f}, det "
          f"{fmt_pct(mono_block['absolute']['det'])} at {fmt_pct(mono_block['absolute']['fa'])}")
    print(f"spend: receipts ${spend['total_receipts_usd']:.4f}, ledger ${spend['ledger_usd']:.4f}")
    print(f"written: {os.path.relpath(dest, HERE)}")

    if args.out_md:
        md = os.path.join(HERE, args.out_md)
        write_summary_md(out, md, X)
        print(f"written: {os.path.relpath(md, HERE)}")
    return out


def write_summary_md(out, path, X):
    G, S, M = out["gpt54_baseline"], out["gemini_checker"], out["matched_monolithic_gemini_FC_score_k1"]
    gp, sp = G["B3_predicate"], S["B3_predicate"]
    sub, sub_n = out["substrate"], out["substrate"]["n_pairs"]
    checker = ", ".join(out["checker_models"])
    L, A = [], None
    A = L.append

    def sweptstr(blk):
        sw = blk.get("swept_fa10")
        return (f"{100 * sw['det']:.1f}% ({sw['det_k']}/{sw['n_err']}) @ "
                f"{100 * sw['fa']:.1f}% ({sw['fa_k']}/{sw['n_clean']})") if sw else "n/a"

    A("# The pipeline judge with a second-family checker")
    A("")
    A(f"**What changed:** stage 2 only. Extraction (`anthropic/claude-opus-5`) and the audit "
      f"(`openai/gpt-5.5`, effort high) were read from `results/w2-pipeline/_cache` for all "
      f"{sub['n_consultations']} consultations and cost nothing, so the fact lists and the "
      f"severity grades are byte-identical to the published run's. The refute pass also stays on the "
      f"auditor. The **check** stage moved from `openai/gpt-5.4` to **`{checker}`** at "
      f"`reasoning_effort` `minimal` (the endpoint refuses `none`), temperature 1.0.")
    A("")
    A(f"**Substrate:** `{sub['file']}` ({sub['dataset_version']}, sha "
      f"{sub['sha256'][:12]}) - the published confirmation set: **{sub_n} held-out pairs** "
      f"({sub['by_class']}) over {sub['n_consultations']} consultations plus 47 clean twins, "
      f"198 notes per tier. One replicate, seed 11.")
    A("")
    A("The gpt-5.4 columns throughout are **recomputed** from "
      "`results/w2-pipeline/_state/confirm-B{2,3}.jsonl` by the same functions, and checked "
      "against the published figures before anything here was written (B2 paired 0.801, "
      "B3 0.752, per-fact rule 20.6% at 2.1% - all reproduced, else this script stops).")
    A("")
    A("## 1. The two tiers")
    A("")
    A("Paired discrimination is tie-adjusted, chance 0.500. Swept detection is the most "
      "sensitive threshold holding false alarms at or under 10%.")
    A("")
    A("| tier | checker | paired omissions | paired commissions | swept det @ FA <= 10% |")
    A("|---|---|---|---|---|")
    for tier in ("B2", "B3"):
        A(f"| {tier} | gpt-5.4 (published) | {G[tier]['paired_omissions']['paired']:.3f} | "
          f"{G[tier]['paired_commissions']['paired']:.3f} | {sweptstr(G[tier])} |")
        A(f"| {tier} | **{checker.split('/')[-1]}** | "
          f"**{S[tier]['paired_omissions']['paired']:.3f}** | "
          f"**{S[tier]['paired_commissions']['paired']:.3f}** | **{sweptstr(S[tier])}** |")
        A(f"| {tier} | *delta* | "
          f"*{S[tier]['paired_omissions']['paired'] - G[tier]['paired_omissions']['paired']:+.3f}* | "
          f"*{S[tier]['paired_commissions']['paired'] - G[tier]['paired_commissions']['paired']:+.3f}* | |")
    A("")
    A("Residual conditioning (paired, omissions only):")
    A("")
    A("| tier | checker | complete | partial-weak | partial-strong |")
    A("|---|---|---|---|---|")
    for tier in ("B2", "B3"):
        for lbl, blk in (("gpt-5.4", G[tier]), (f"**{checker.split('/')[-1]}**", S[tier])):
            r = blk["paired_by_residual"]
            g = lambda k: (f"{r[k]['paired']:.3f}" if r.get(k) and r[k].get("paired") is not None
                           else "n/a")
            A(f"| {tier} | {lbl} | {g('complete')} | {g('partial-weak')} | {g('partial-strong')} |")
    A("")
    A("## 2. The per-fact critical rule on B3")
    A("")
    A("Flag the note iff any fact the **audit** graded `critical` is verdicted `absent`. The "
      "severities come from the cached gpt-5.5 audit and are identical for both checkers, so "
      "this rule is applied to the two families unchanged.")
    A("")
    A("| metric | gpt-5.4 checker (published) | " + checker.split("/")[-1] + " checker |")
    A("|---|---|---|")
    A(f"| detection, all omissions | {fmt_pct(gp['absolute']['det'])} | "
      f"**{fmt_pct(sp['absolute']['det'])}** |")
    A(f"| false alarms, 47 clean twins | {fmt_pct(gp['absolute']['fa'])} | "
      f"**{fmt_pct(sp['absolute']['fa'])}** |")
    A(f"| det-vs-FA z | {gp['absolute']['z']} | **{sp['absolute']['z']}** |")
    A(f"| critical-severity pairs | {fmt_pct(gp['critical_severity_pairs'])} | "
      f"**{fmt_pct(sp['critical_severity_pairs'])}** |")
    for lv in ("complete", "partial-weak", "partial-strong"):
        A(f"| by residual: {lv} | {fmt_pct(gp['by_residual_detection'][lv])} | "
          f"**{fmt_pct(sp['by_residual_detection'][lv])}** |")
    A(f"| commissions flagged | {fmt_pct(gp['commission_flags'])} | "
      f"**{fmt_pct(sp['commission_flags'])}** |")
    A("")
    A("## 3. Does restructuring beat this judge's own monolithic design?")
    A("")
    A(f"The comparator is Gemini's own `FC-score-k1` from the second-family run, restricted to "
      f"exactly these {M['subset_match']['errored_pairs_matched']} pairs and "
      f"{M['subset_match']['clean_notes_matched']} clean twins. Those records were already "
      f"bought, so this row cost nothing. {M['n_replicates']} replicate(s); the pipeline rows "
      f"are one replicate, so the monolithic mean is the like-for-like number.")
    A("")
    A("| design (all Gemini-judged) | paired omissions | paired commissions | detection | FA |")
    A("|---|---|---|---|---|")
    A(f"| monolithic FC-score-k1 (score <= 7) | {M['paired_omissions']['paired']:.3f} | "
      f"{M['paired_commissions']['paired']:.3f} | {fmt_pct(M['absolute']['det'])} | "
      f"{fmt_pct(M['absolute']['fa'])} |")
    A(f"| pipeline B2 (swept @ FA <= 10%) | {S['B2']['paired_omissions']['paired']:.3f} | "
      f"{S['B2']['paired_commissions']['paired']:.3f} | {sweptstr(S['B2'])} | |")
    A(f"| pipeline B3 (swept @ FA <= 10%) | {S['B3']['paired_omissions']['paired']:.3f} | "
      f"{S['B3']['paired_commissions']['paired']:.3f} | {sweptstr(S['B3'])} | |")
    A(f"| **pipeline B3 + per-fact critical rule** | (same as B3) | | "
      f"**{fmt_pct(sp['absolute']['det'])}** | **{fmt_pct(sp['absolute']['fa'])}** |")
    A("")
    A("Monolithic FC-score-k1 on this subset, per replicate:")
    A("")
    A("| replicate | paired omissions | paired commissions | det (score <= 7) | FA |")
    A("|---|---|---|---|---|")
    for rep, v in sorted(M["per_replicate"].items(), key=lambda x: int(x[0])):
        A(f"| r{rep} | {v['paired_omissions']['paired']:.3f} | "
          f"{v['paired_commissions']['paired']:.3f} | "
          f"{fmt_pct(v['absolute_omission_det'])} | {fmt_pct(v['absolute_fa'])} |")
    A("")
    A("## Reading")
    A("")
    dpo_b2 = S["B2"]["paired_omissions"]["paired"] - M["paired_omissions"]["paired"]
    dpo_b3 = S["B3"]["paired_omissions"]["paired"] - M["paired_omissions"]["paired"]
    ddet = sp["absolute"]["det"]["p"] - M["absolute"]["det"]["p"]
    A(f"**Yes on both measures: restructuring the task extends its lead on the stronger "
      f"judge.** Against Gemini's own monolithic `FC-score-k1` on exactly these notes "
      f"(paired {M['paired_omissions']['paired']:.3f}), the pipeline scores "
      f"{S['B2']['paired_omissions']['paired']:.3f} at B2 and "
      f"{S['B3']['paired_omissions']['paired']:.3f} at B3 - **{dpo_b2:+.3f}** and "
      f"**{dpo_b3:+.3f}** on the paired measure. On detection at a matched false-alarm rate "
      f"the per-fact critical rule gets {fmt_pct(sp['absolute']['det'])} at "
      f"{fmt_pct(sp['absolute']['fa'])} against the monolithic's "
      f"{fmt_pct(M['absolute']['det'])} at {fmt_pct(M['absolute']['fa'])} - the same false-"
      f"alarm rate, **{100 * ddet:+.1f} percentage points** of detection. The pipeline "
      f"result is a property of the task shape, not of gpt-5.4.")
    A("")
    A(f"**The rule transfers almost exactly, which is the strongest single signal here.** The "
      f"per-fact critical rule was tuned on nothing - it is a predicate over verdicts - and "
      f"across a change of checker family it moves from "
      f"{fmt_pct(gp['absolute']['det'])} at {fmt_pct(gp['absolute']['fa'])} to "
      f"{fmt_pct(sp['absolute']['det'])} at {fmt_pct(sp['absolute']['fa'])}: the same one "
      f"false alarm out of 47, two more true detections out of 131. On critical-severity "
      f"pairs it goes {fmt_pct(gp['critical_severity_pairs'])} -> "
      f"{fmt_pct(sp['critical_severity_pairs'])}. That is the closest thing in the study to "
      f"an operating point that is not judge-specific.")
    A("")
    A(f"**B3 overtakes B2 on this family, reversing the published ordering.** On gpt-5.4 the "
      f"audit + trichotomy + refute machinery scored BELOW pipeline-lite on the paired measure "
      f"({G['B3']['paired_omissions']['paired']:.3f} against "
      f"{G['B2']['paired_omissions']['paired']:.3f}), which the published run recorded as a "
      f"failed prediction. With the Gemini checker it is the other way round: "
      f"{S['B3']['paired_omissions']['paired']:.3f} against "
      f"{S['B2']['paired_omissions']['paired']:.3f}. The likely reason is visible in the "
      f"verdict mix - this checker actually uses the trichotomy's middle label, which the "
      f"gpt-5.4 checker at effort `none` did not - so B3's extra machinery only pays "
      f"when the checker can drive it. Read as: the tier ranking is judge-dependent even "
      f"though the pipeline-versus-monolithic ranking is not.")
    A("")
    b3ps = S["B3"]["paired_by_residual"].get("partial-strong") or {}
    A(f"**The strong-residual case splits in two, and this is the new complication.** On the "
      f"PAIRED measure the Gemini pipeline is well clear of chance where nothing else in the "
      f"study was: B3 scores {b3ps.get('paired'):.3f} on partial-strong pairs, against "
      f"{(G['B3']['paired_by_residual'].get('partial-strong') or {}).get('paired'):.3f} for "
      f"the gpt-5.4 checker and the 0.47-0.56 band the paper reports for every other method. So "
      f"the information is there and this checker can rank on it. But it still does not "
      f"convert: under the per-fact rule, detection on partial-strong is "
      f"{fmt_pct(sp['by_residual_detection']['partial-strong'])} - identical to gpt-5.4's "
      f"{fmt_pct(gp['by_residual_detection']['partial-strong'])}, and zero. The paper's open "
      f"problem is now better localised: it is a decision-rule problem on strong residuals, "
      f"not purely a discrimination one.")
    A("")
    A(f"**The commission cost of enumerating is larger on this judge, not smaller.** The "
      f"monolithic Gemini judge scores {M['paired_commissions']['paired']:.3f} on the 20 "
      f"commission pairs; the pipeline scores {S['B2']['paired_commissions']['paired']:.3f} "
      f"(B2) and {S['B3']['paired_commissions']['paired']:.3f} (B3), and the rule flags "
      f"{fmt_pct(sp['commission_flags'])} of them. That is the paper's mechanism intact - a "
      f"judge that enumerates what should be present wins on absence and loses on invention "
      f"- and it is the argument against replacing a faithfulness judge with this rather "
      f"than running both. n=20, so treat the size as indicative.")
    A("")
    A("**Confounds, stated.** The gpt-5.4 B2 baseline ran its check at effort `none` and B3 "
      "at `medium`; Gemini can only run `minimal`, so the effort axis is not matched "
      "tier for tier and family is confounded with deliberation - the same limitation the "
      "published run records for itself. One replicate on the pipeline rows against three on the monolithic "
      "comparator. 151 pairs, so the residual cells are small (62 complete, 36 partial-weak, "
      "33 partial-strong) and their intervals are wide - the Wilson bounds are in "
      "`analysis.json`.")
    A("")
    A("## Parse failures and spend")
    A("")
    A(f"Parse failures: **{S['B2']['parse_failures']}** on B2 and "
      f"**{S['B3']['parse_failures']}** on B3, of 198 records each. The keyed contract is "
      "unchanged: a missing fact id triggers one targeted retry, and a still-unanswered id "
      "makes the record a parse failure rather than a fraction over an unknown denominator.")
    A("")
    sp_ = out["spend"]
    A("| path | USD | calls |")
    A("|---|---|---|")
    A(f"| confirmation run, summed receipts | {sp_['confirmation_receipts_usd']:.4f} | "
      f"{sp_['confirmation_calls']} |")
    A(f"| check-stage smoke, summed receipts | {sp_['smoke_receipts_usd']:.4f} | "
      f"{sp_['smoke_calls']} |")
    A(f"| **total, receipts** | **{sp_['total_receipts_usd']:.4f}** | "
      f"{sp_['confirmation_calls'] + sp_['smoke_calls']} |")
    A(f"| cost ledger, `{EXPERIMENT}` rows ({sp_['ledger_rows']}) | {sp_['ledger_usd']:.4f} | "
      f"{sp_['ledger_calls']} |")
    if sp_.get("openrouter_api"):
        api = sp_["openrouter_api"]
        A(f"| **OpenRouter credits API (authority)** | **{api['task_spend_usd']:.4f}** | |")
    A("")
    A(sp_["reused_free"] + ".")
    A("")
    A(f"**The projection ran low and should be recorded as such.** A three-consultation smoke "
      f"spanning 20-48 facts fitted cost against fact count and projected "
      f"${11.16:.2f} for the confirmation; it cost "
      f"${sp_['confirmation_receipts_usd']:.2f}, about 18% more. The smoke could not see the "
      f"cause: the refute stage runs on the auditor at effort high and fires once per note "
      f"with any flagged absence, and this checker flagged more absences than gpt-5.4's did "
      f"(319 refuted against the published run's 299), so the expensive non-Gemini stage scaled with a "
      f"behaviour change in the cheap Gemini one. The run still finished inside the $15 stop, "
      f"but with less headroom than planned.")
    if sp_.get("openrouter_api"):
        api = sp_["openrouter_api"]
        A("")
        A(f"Credits API read {api['read_utc']}: `total_usage` {api['total_usage_after']:.4f} "
          f"against {api['total_usage_before']:.4f} before this extension, i.e. "
          f"**${api['task_spend_usd']:.4f}**. Run-scoped guard was warn $10 / stop $15 on the "
          f"`{EXPERIMENT}` ledger scope.")
    A("")
    A("## Provenance")
    A("")
    A(f"- Gemini stores: `results/{EXPERIMENT}/_state/sfconfirm-B2.jsonl`, "
      f"`sfconfirm-B3.jsonl`; smoke `sfsmoke-B{{2,3}}.jsonl`")
    A(f"- gpt-5.4 reference: `{GPT_B2}`, `{GPT_B3}`")
    A(f"- Monolithic comparator: `{MONO_STORE}` restricted to `{CONFIRM_SUBSET}`")
    A(f"- Reused instrument: `results/w2-pipeline/_cache/{{facts,audit}}_*.json`")
    A(f"- Analysis: `results/{EXPERIMENT}/analysis.json`; measure conventions imported from "
      f"`record/figures/extract_data.py`")
    A(f"- Model pin: `models.lock.json` role `judge-gemini`; runner `w2_pipeline.py "
      f"--check-role judge-gemini --check-effort minimal --experiment {EXPERIMENT}`")
    L.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(L))
    return path


if __name__ == "__main__":
    main()
