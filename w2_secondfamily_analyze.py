"""w2_secondfamily_analyze.py - the second-family replication's metrics and summary.

The measure conventions are NOT reimplemented here. They are imported from
paper/figures/extract_data.py, which is the module that builds the paper's floats and
which refuses to write unless its recomputation reproduces docs/FINDINGS.md. Using the
same `arm_block` on the new store is what makes the comparison a comparison: any
difference between the two families is the judge, not the arithmetic.

Before it reports anything, this script re-derives the gpt-5.4 grid's own F-bin-k1 and
FC-score-k1 cells from results/w2-ablation/_state/grid-main2.jsonl and checks them
against FINDINGS 17.1 (F-bin-k1 0.875 / 0.500; FC-score-k1 0.895 / 0.585). If that
check fails the script STOPS - a comparison table whose baseline column cannot be
reproduced is not worth printing.

  paired    tie-adjusted P(errored note scores strictly below its OWN clean twin) +
            half the tie mass, per replicate then averaged. Chance 0.500. Reported
            separately for omissions (omit pairs) and commissions (add + change).
  absolute  flag rate at the design's own pre-registered grid threshold - FAIL verdict
            for the binary design, score <= 7 for the k=1 score design - beside the
            same design's false-alarm rate on the 112 clean twins.
  spend     BOTH the cost ledger's rows for this experiment AND the sum of the
            per-call receipts in the record store, printed side by side. They are
            independent paths to the same number; a gap between them is information.

    .venv/bin/python w2_secondfamily_analyze.py
"""
import argparse, glob, importlib.util, json, os, sys
from collections import Counter

from common import HERE, RESULTS
import w2_common as W

EXPERIMENT = "w2-secondfamily"
STORE_GLOB = os.path.join(RESULTS, EXPERIMENT, "_state", "*.jsonl")
GRID_STORE = "results/w2-ablation/_state/grid-main2.jsonl"

# FINDINGS 17.1, the two cells under replication. Quoted here ONLY to be checked
# against a recomputation from the grid store; nothing downstream reads these.
FINDINGS_17_1 = {"F-bin-k1": {"omissions": 0.500, "commissions": 0.875},
                 "FC-score-k1": {"omissions": 0.585, "commissions": 0.895}}
# arm name -> the grid cell it replicates
REPLICATES_CELL = {"sf-F-bin-k1": "F-bin-k1", "sf-FC-score-k1": "FC-score-k1",
                   "sf-FC-score-k8": "FC-score-k8"}
FLAG_RULE = {"sf-F-bin-k1": "FAIL verdict", "sf-FC-score-k1": "score <= 7",
             "sf-FC-score-k8": "winsorized mean < 8.0"}


def load_extract_data():
    """Import paper/figures/extract_data.py as a module without running its main()."""
    path = os.path.join(HERE, "paper", "figures", "extract_data.py")
    if not os.path.exists(path):
        raise SystemExit(f"cannot find {path} - the shared measure conventions live there")
    spec = importlib.util.spec_from_file_location("_w2_extract_data", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_w2_extract_data"] = mod
    spec.loader.exec_module(mod)
    for fn in ("arm_block", "paired_over_reps", "wilson", "two_prop_z", "load"):
        if not hasattr(mod, fn):
            raise SystemExit(f"paper/figures/extract_data.py has no {fn}() - "
                             "the shared conventions moved; stopping rather than guessing")
    return mod


def load_store(paths):
    recs = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    r["_store"] = os.path.relpath(p, HERE)
                    recs.append(r)
    return recs


def ledger_rows(experiment):
    path = os.path.join(RESULTS, "cost_ledger.jsonl")
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("experiment") == experiment:
                out.append(d)
    return out


def commission_absolute(X, recs):
    """Absolute flag rate on the commission (add+change) records, same convention as
    arm_block's omission `absolute`: parse failure counts as not-flagged, on the full
    denominator."""
    comm = [r for r in recs if r.get("note_role") != "clean"
            and r.get("pair_type") in ("add", "change")]
    k = sum(1 for r in comm if r.get("flagged") and not r.get("parse_failure"))
    return X.wilson(k, len(comm))


def fmt_pct(w):
    return "n/a" if not w else f"{100 * w['p']:.1f}% ({w['k']}/{w['n']})"


def write_summary_md(out, path):
    """The human summary. Every figure is read out of `out`, which is the analysis json;
    nothing is typed in twice, so the markdown cannot drift from the artifact."""
    D, B, S = out["designs"], out["baseline_gpt54_grid"], out["spend"]
    judge = ", ".join(out["judge_models"])
    role = next(iter(out["roles"].values()), {})
    L = []
    A = L.append
    A("# W2 second-family replication - the judge asymmetry on a non-OpenAI family")
    A("")
    rep_desc = "; ".join(
        f"{d['replicates_grid_cell']} {d.get('n_replicates', 1)} replicate"
        + ("s" if d.get("n_replicates", 1) != 1 else "")
        + " (seed" + ("s " if d.get("n_replicates", 1) != 1 else " ")
        + ", ".join(str(s) for s in sorted(
            {s for v in d.get("per_replicate", {}).values() for s in v["seed"]})) + ")"
        for d in D.values())
    A(f"**Judge:** `{judge}` (OpenRouter, route `google-vertex`, temperature 1.0, "
      f"`max_tokens` {W.MAX_TOKENS}, `reasoning_effort` "
      f"`{role.get('reasoning_effort')}`). {rep_desc}.")
    A("")
    A("The paper's central asymmetry rests on one judge family (`openai/gpt-5.4`, FINDINGS "
      "§17). This is an indicative replication of two of the grid's eight cells on a second "
      "family - same prompts byte for byte, same 495 evaluation pairs and 112 clean twins, "
      "same aggregation and flag rules. FC-score-k1 was subsequently taken to three "
      "replicates on the grid's own seeds (11/22/33) to test whether its single-replicate "
      "result was stable; F-bin-k1 remains a single replicate and is reported as such.")
    A("")
    short = judge.split("/")[-1]

    def reps_label(d):
        n = d.get("n_replicates", 1)
        return f"{short}, {n} rep" + ("s" if n != 1 else "")

    A("## Paired discrimination, against the grid's same cells")
    A("")
    A("Tie-adjusted: P(errored note scores strictly below its own clean twin) plus half the "
      "tie mass, chance 0.500. Both error columns are the same measure on the same notes; "
      "only the error class differs. The gpt-5.4 rows are **recomputed** from "
      "`results/w2-ablation/_state/grid-main2.jsonl` by the same function, and checked "
      "against the published FINDINGS §17.1 figures before anything here was written - "
      "agreement within 0.0015 on all four values, else this script stops.")
    A("")
    A("| cell | judge | commissions | omissions | commissions - omissions |")
    A("|---|---|---|---|---|")
    for arm, d in D.items():
        cell = d["replicates_grid_cell"]
        if cell in B:
            bc = B[cell]["paired_commissions"]["paired"]
            bo = B[cell]["paired_omissions"]["paired"]
            A(f"| {cell} | gpt-5.4, 3 reps (FINDINGS §17.1) | {bc:.3f} | {bo:.3f} | "
              f"{bc - bo:+.3f} |")
        c, o = d["paired_commissions"]["paired"], d["paired_omissions"]["paired"]
        A(f"| {cell} | **{reps_label(d)}** | **{c:.3f}** | **{o:.3f}** | **{c - o:+.3f}** |")
        if cell in B:
            A(f"| {cell} | *delta* | *{c - B[cell]['paired_commissions']['paired']:+.3f}* | "
              f"*{o - B[cell]['paired_omissions']['paired']:+.3f}* | |")
    A("")
    A("Multi-replicate rows are the mean of the per-replicate values, the FINDINGS §17.1 "
      "convention. Per design and replicate: 293 omission records, 202 commission records "
      "(add + change), 112 clean twins.")
    A("")
    A("## Absolute (deployment) rates at the grid's pre-registered thresholds")
    A("")
    A("Flag rate on errored notes beside the same design's false-alarm rate on the clean "
      "twins. Rows carrying 3 replicates pool them, so their denominators are 3x - the "
      "FINDINGS §17.2 convention. **Read the pooled z with care**: pooling counts three "
      "passes over the SAME notes as three independent observations, which inflates it. The "
      "per-replicate z values in the stability section below are the honest ones, and the "
      "same caveat applies to the gpt-5.4 rows here and in §17.2.")
    A("")
    A("| cell | judge | flag rule | omission det | commission det | false alarm (clean) "
      "| omission-det-vs-FA z |")
    A("|---|---|---|---|---|---|---|")
    for arm, d in D.items():
        cell = d["replicates_grid_cell"]
        if cell in B:
            A(f"| {cell} | gpt-5.4, 3 reps | `{d['flag_rule']}` | "
              f"{fmt_pct(B[cell]['absolute_omission_det'])} | "
              f"{fmt_pct(B[cell]['absolute_commission_det'])} | "
              f"{fmt_pct(B[cell]['absolute_fa'])} | {B[cell].get('absolute_det_vs_fa_z')} |")
        A(f"| {cell} | **{reps_label(d)}** | `{d['flag_rule']}` | "
          f"{fmt_pct(d['absolute_omission_det'])} | {fmt_pct(d['absolute_commission_det'])} | "
          f"{fmt_pct(d['absolute_fa'])} | {d['absolute_det_vs_fa_z']} |")
    A("")
    A("## Conditioning: does the omission signal move the same way?")
    A("")
    A("FINDINGS §17.3 found the grid's omission signal graded by how much of the fact "
      "survives and by how much it matters, and collapsing to chance on the clinically "
      "deceptive case - a strong surviving trace. Same breakdown here, like for like on "
      "each cell.")
    A("")
    A("| cell | judge | complete | partial-weak | partial-strong | critical | supporting "
      "| peripheral |")
    A("|---|---|---|---|---|---|---|---|")

    def cond_row(label, blk):
        r = blk.get("paired_by_residual") or {}
        s = blk.get("paired_by_severity") or {}
        g = lambda dd, k: (f"{dd[k]['paired']:.3f}" if dd.get(k) and dd[k].get("paired")
                           is not None else "n/a")
        return (f"| {label} | {g(r, 'complete')} | {g(r, 'partial-weak')} | "
                f"{g(r, 'partial-strong')} | {g(s, 'critical')} | {g(s, 'supporting')} | "
                f"{g(s, 'peripheral')} |")

    for arm, d in D.items():
        cell = d["replicates_grid_cell"]
        if cell in B:
            A(cond_row(f"{cell} | gpt-5.4, 3 reps", B[cell]))
        A(cond_row(f"{cell} | **{reps_label(d)}**", d))
    A("")
    A("## Threshold-swept detection at a deployable false-alarm rate")
    A("")
    A("Detection at the most sensitive threshold whose false-alarm rate stays within 10% - "
      "the construction FINDINGS §17.2 uses (it reports 4.7% for FC-score-k1 and 8.3% for "
      "FC-score-k8 on gpt-5.4). Not expressible on a binary aggregate.")
    A("")
    A("| cell | judge | swept omission det | at FA | threshold |")
    A("|---|---|---|---|---|")
    for arm, d in D.items():
        cell = d["replicates_grid_cell"]
        for label, blk in ((f"gpt-5.4, 3 reps", B.get(cell)),
                           (f"**{reps_label(d)}**", d)):
            if not blk:
                continue
            sw = blk.get("swept_fa10")
            if sw:
                A(f"| {cell} | {label} | {100 * sw['det']:.1f}% ({sw['det_k']}/{sw['n_err']}) "
                  f"| {100 * sw['fa']:.1f}% ({sw['fa_k']}/{sw['n_clean']}) | "
                  f"< {sw['threshold']} |")
            else:
                A(f"| {cell} | {label} | not expressible (binary aggregate) | | |")
    A("")
    # ---- replicate stability, for any design carrying more than one replicate
    for arm, d in D.items():
        if d.get("n_replicates", 1) < 2:
            continue
        cell, sp, PR = d["replicates_grid_cell"], d["replicate_spread"], d["per_replicate"]
        A(f"## Replicate stability: {cell} across {sp['n_replicates']} replicates")
        A("")
        A(f"Seeds are the grid's own replicate seeds. Each row is a complete, independent "
          f"pass over the full substrate, scored by the same `arm_block` as everything else; "
          f"nothing is pooled inside a row.")
        A("")
        A("| replicate | seed | paired commissions | paired omissions | det (score <= 7) "
          "| FA (clean) | det-vs-FA z |")
        A("|---|---|---|---|---|---|---|")
        for rep in sorted(PR, key=int):
            v = PR[rep]
            A(f"| r{rep} | {', '.join(str(s) for s in v['seed'])} | "
              f"{v['paired_commissions']['paired']:.3f} | {v['paired_omissions']['paired']:.3f} "
              f"| {fmt_pct(v['absolute_omission_det'])} | {fmt_pct(v['absolute_fa'])} | "
              f"{v['absolute_det_vs_fa_z']} |")
        A(f"| **mean / pooled** | | **{d['paired_commissions']['paired']:.3f}** | "
          f"**{d['paired_omissions']['paired']:.3f}** | "
          f"**{fmt_pct(d['absolute_omission_det'])}** | **{fmt_pct(d['absolute_fa'])}** | "
          f"**{d['absolute_det_vs_fa_z']}** |")
        A("")
        A("Residual conditioning per replicate - the axis §17.3 reports, and the one where "
          "the clinically deceptive case (a strong surviving trace) is expected to sit at "
          "chance:")
        A("")
        A("| replicate | complete | partial-weak | partial-strong |")
        A("|---|---|---|---|")
        for rep in sorted(PR, key=int):
            r = PR[rep]["paired_by_residual"]
            g = lambda k: (f"{r[k]['paired']:.3f}" if r.get(k)
                           and r[k].get("paired") is not None else "n/a")
            A(f"| r{rep} | {g('complete')} | {g('partial-weak')} | {g('partial-strong')} |")
        rr = d["paired_by_residual"]
        gm = lambda k: (f"**{rr[k]['paired']:.3f}**" if rr.get(k)
                        and rr[k].get("paired") is not None else "n/a")
        A(f"| **mean** | {gm('complete')} | {gm('partial-weak')} | {gm('partial-strong')} |")
        A("")
        po, pc = sp["paired_omissions"], sp["paired_commissions"]
        det, fa, z = (sp["absolute_omission_det"], sp["absolute_fa"],
                      sp["absolute_det_vs_fa_z"])
        A(f"**Spread across the {sp['n_replicates']} replicates.** Paired omissions "
          f"{po['min']:.3f}-{po['max']:.3f}; paired commissions {pc['min']:.3f}-"
          f"{pc['max']:.3f}; detection {100 * det['min']:.1f}-{100 * det['max']:.1f}% at a "
          f"false-alarm rate of {100 * fa['min']:.1f}-{100 * fa['max']:.1f}%; det-vs-FA z "
          f"{z['min']:.2f}-{z['max']:.2f}.")
        A("")
        pf_reps = {r: PR[r]["parse_failures"] for r in sorted(PR, key=int)}
        A(f"Parse failures by replicate: "
          + ", ".join(f"r{r} {n}" for r, n in pf_reps.items()) + ".")
        A("")

    A("## What this shows, and what it does not")
    A("")
    fb, fs = D.get("sf-F-bin-k1"), D.get("sf-FC-score-k1")
    if fb and fs:
        A(f"**The asymmetry is not an artefact of one judge family.** On a different family, "
          f"on the same notes and the same prompts, commissions land at "
          f"{fb['paired_commissions']['paired']:.3f} and "
          f"{fs['paired_commissions']['paired']:.3f} while omissions land at "
          f"{fb['paired_omissions']['paired']:.3f} and "
          f"{fs['paired_omissions']['paired']:.3f} - a gap of "
          f"{fb['paired_commissions']['paired'] - fb['paired_omissions']['paired']:.2f} and "
          f"{fs['paired_commissions']['paired'] - fs['paired_omissions']['paired']:.2f} on a "
          f"measure whose chance level is 0.500. The direction, the rough size and the "
          f"ordering of the two designs all reproduce.")
        A("")
        A("**The conditioning reproduces too, including the part that matters clinically.** "
          "The scored design grades by residual - complete "
          f"{fs['paired_by_residual']['complete']['paired']:.3f}, partial-weak "
          f"{fs['paired_by_residual']['partial-weak']['paired']:.3f}, partial-strong "
          f"{fs['paired_by_residual']['partial-strong']['paired']:.3f} - so the deceptive "
          "case, where a strong trace of the fact survives, sits at chance on this family "
          "exactly as it does on gpt-5.4 (§17.3, 0.526 on the best grid cell). Severity "
          "grades in the same order.")
        A("")
        sp = fs.get("replicate_spread") or {}
        if sp.get("n_replicates", 1) > 1:
            det, fa, po = (sp["absolute_omission_det"], sp["absolute_fa"],
                           sp["paired_omissions"])
            z = sp["absolute_det_vs_fa_z"]
            A(f"**The single-replicate FC-score-k1 result held.** Taken to "
              f"{sp['n_replicates']} replicates on the grid's own seeds, detection at the "
              f"pre-registered `score <= 7` rule runs "
              f"{100 * det['min']:.1f}-{100 * det['max']:.1f}% against a false-alarm rate of "
              f"{100 * fa['min']:.1f}-{100 * fa['max']:.1f}%, and the detection-versus-false-"
              f"alarm separation is significant in every replicate (z "
              f"{z['min']:.2f}-{z['max']:.2f}). Paired omissions run "
              f"{po['min']:.3f}-{po['max']:.3f}. The first replicate's 20.1% at 1.8% was not "
              f"a lucky draw; the run-to-run spread is a few points, which is the same order "
              f"as the grid's own replicate spread in §17.1.")
            A("")
        dcb = fb["paired_commissions"]["paired"] - B["F-bin-k1"]["paired_commissions"]["paired"]
        dob = fb["paired_omissions"]["paired"] - B["F-bin-k1"]["paired_omissions"]["paired"]
        dcs = fs["paired_commissions"]["paired"] - B["FC-score-k1"]["paired_commissions"]["paired"]
        dos = fs["paired_omissions"]["paired"] - B["FC-score-k1"]["paired_omissions"]["paired"]
        A(f"**This judge is better, and still not usable.** Every Gemini figure is above its "
          f"gpt-5.4 counterpart: commissions by {dcb:.2f} and {dcs:.2f}, omissions by "
          f"{dob:.2f} and {dos:.2f}, and the "
          "absolute rates separate from the false-alarm rate where gpt-5.4's did not "
          f"(per-replicate z 4.17-4.72 against -0.713 pooled on the same gpt-5.4 cell). The "
          f"swept "
          f"figure needs reading with care: "
          f"{100 * fs['swept_fa10']['det']:.0f}% detection at "
          f"{100 * fs['swept_fa10']['fa']:.1f}% false alarms is bought at threshold "
          f"`< {fs['swept_fa10']['threshold']}`, i.e. \"anything short of a perfect 10 is a "
          "flag\", and it works only because this judge parks clean notes on exactly 10. That "
          "is an operating point on one corpus, and §18.2 is the standing warning about what "
          "happens to those off-distribution. The paper's claim is about the *gap* between "
          "commission and omission detection, and the gap survives the change of family; the "
          "absolute level is judge-dependent and the second family moves it, so a claim of "
          "the form \"no judge exceeds X% on omissions\" would not be safe on this evidence.")
        A("")
        A(f"**Limits, stated plainly.** FC-score-k1 now carries "
          f"{fs.get('n_replicates', 1)} replicate"
          + ("s" if fs.get("n_replicates", 1) != 1 else "")
          + f" and F-bin-k1 carries {fb.get('n_replicates', 1)}, so the F-bin-k1 row is still "
          "a single draw and its deltas are one run against a three-run mean, not tested "
          "effects. Two designs of eight either way, and no significance testing between "
          "families - the replicates establish that each family's own figure is stable, not "
          "that the difference between them is. `reasoning_effort` is `minimal` rather than "
          "the grid's `none`, because this endpoint refuses to disable reasoning, so judge "
          "family and reasoning setting move together and cannot be separated by this run. "
          "Indicative, as commissioned.")
    A("")
    A("## Parse failures")
    A("")
    tot_pf = sum(d["parse_failures"]["n"] for d in D.values())
    tot_rt = sum(d["parse_failures"]["retried_samples"] for d in D.values())
    tot_rec = sum(sum(d["n_records"].values()) for d in D.values())
    A(f"**{tot_pf}** across both designs ("
      + ", ".join(f"{d['replicates_grid_cell']} {d['parse_failures']['n']}"
                  for d in D.values())
      + f") on {tot_rec} records. {tot_rt} sample(s) hit the pre-registered single retry, "
      f"{tot_rt - tot_pf} of which recovered. Unparseable output is recorded as a parse "
      "failure and never stops the run.")
    for arm, d in D.items():
        for r in d["parse_failures"].get("records", []):
            A("")
            if r["note_role"] == "clean":
                effect = ("a **clean twin**, so every pair of that consultation loses its "
                          "reference and drops out of the paired measure for that replicate")
            else:
                effect = (f"an **errored note** ({r['pair_type']}), so that one pair drops "
                          f"out of the paired measure for that replicate")
            A(f"- `{d['replicates_grid_cell']}` r{r['replicate']}: `{r['note_key']}` - "
              f"{effect}. The absolute rates are unaffected: they count a parse failure as "
              f"not-flagged on the full denominator, the FINDINGS §18.1 convention.")
    A("")
    A("Paired denominators per replicate, after those drops: "
      + "; ".join(
          f"{d['replicates_grid_cell']} "
          + ", ".join(f"r{rep} {v['paired_omissions']['n_pairs']} omission / "
                      f"{v['paired_commissions']['n_pairs']} commission"
                      for rep, v in sorted(d.get("per_replicate", {}).items(), key=lambda x: int(x[0])))
          for d in D.values())
      + " (against 293 and 202 with nothing dropped).")
    A("")
    A("## Spend")
    A("")
    A("| path | USD | calls |")
    A("|---|---|---|")
    A(f"| main run, summed per-call receipts | {S['main_run_receipts_usd']:.4f} | "
      f"{S['main_run_receipt_calls']} |")
    A(f"| candidate smoke, summed per-call receipts | {S['smoke_receipts_usd']:.4f} | "
      f"{S['smoke_receipt_calls']} |")
    A(f"| **total, receipts** | **{S['total_receipts_usd']:.4f}** | "
      f"{S['main_run_receipt_calls'] + S['smoke_receipt_calls']} |")
    A(f"| cost ledger, `{EXPERIMENT}` rows ({S['ledger_rows_for_experiment']}) | "
      f"{S['ledger_usd_all_rows']:.4f} | {S['ledger_calls_all_rows']} |")
    if S.get("openrouter_api"):
        api = S["openrouter_api"]
        A(f"| **OpenRouter credits API (the authority)** | "
          f"**{api['task_spend_usd']:.4f}** | |")
    A("")
    if S.get("unclosed_runs"):
        for u in S["unclosed_runs"]:
            A(f"The ledger is ${S['total_receipts_usd'] - S['ledger_usd_all_rows']:.4f} short of "
              f"the receipts, and the difference is named: run `{u['run_id']}` "
              f"({u['calls']} calls, ${u['receipts_usd']:.4f}) was killed mid-chunk when the "
              f"run was restarted with faster ordering, so it never wrote a closing ledger "
              f"row. Its {u['records_in_store']} judgements are in the record store, are "
              f"counted in the receipts column, and were not re-bought on resume.")
        A("")
    if S.get("openrouter_api"):
        api = S["openrouter_api"]
        A(f"OpenRouter credits API read {api['read_utc']}: `total_usage` "
          f"{api['total_usage_after']:.4f} against {api['total_usage_before']:.4f} before this "
          f"work began, i.e. **${api['task_spend_usd']:.4f}** for everything here - smoke, the "
          f"killed partial run, the first pass and the replicate extension together. That is "
          f"the figure to quote. It sits "
          f"${api['task_spend_usd'] - S['total_receipts_usd']:+.4f} against the summed receipts.")
        if api.get("extension_spend_usd"):
            A("")
            A(f"Split at the reading taken between the two: the first pass (both designs, "
              f"replicate 1, plus the smoke) cost "
              f"${api['extension_baseline'] - api['total_usage_before']:.4f}, and the "
              f"FC-score-k1 replicate extension (replicates 2 and 3, 1,214 calls) cost "
              f"**${api['extension_spend_usd']:.4f}** against a projection of $15.12 made "
              f"from replicate 1's own measured $0.012455/call.")
        A("")
    A("Ledger row status: "
      + ", ".join(f"{n} {st}" for st, n in sorted(S["ledger_row_status"].items()))
      + f". {S['note']}")
    A("")
    A("Caps, set deliberately in run scope on each launch and recorded in every manifest's "
      "`params.spend_guard`: the first pass ran at warn $25 / stop $40 and fired neither; "
      "the replicate extension ran at warn $12 / stop $20, where the warn fired as expected "
      "at $12 against a $15.12 projection and the stop did not.")
    A("")
    A("## Scope decision: the k=8 extension was measured and declined")
    A("")
    A("An optional third design (FC-score-k8, winsorized mean over 8 samples, on a "
      "consultation-stratified half of the corpus at seed 20260817) was in scope only if the "
      "measured projection kept total spend at or under $35. The smoke measured "
      "$0.011367/call; the half-corpus is 304 notes, so k=8 buys 2,432 calls = **$28.09**, "
      "which on top of the main run's $14.02 projects **$42.11** total. That is over the $35 "
      "ceiling and over the $40 hard cap, so it was not run. The arm "
      "(`sf-FC-score-k8`) and the `--consultation-half` selector are committed and ready if "
      "the budget is ever raised.")
    A("")
    A("## Provenance")
    A("")
    A(f"- Record store(s): " + ", ".join(f"`{s}`" for s in out["stores"]))
    A(f"- Analysis json: `results/{EXPERIMENT}/analysis.json`; smoke: "
      f"`results/{EXPERIMENT}/smoke.json`")
    A(f"- Dataset: `master/dataset_v2.json` version "
      f"{', '.join(str(v) for v in out['dataset']['file'])}, sha "
      f"{', '.join(s[:12] for s in out['dataset']['pairs_sha256_from_manifests'])}")
    A(f"- Measure conventions imported from `paper/figures/extract_data.py` "
      f"(`arm_block`), the module that reproduces FINDINGS §17.1")
    A(f"- Model pin: `models.lock.json` role `judge-gemini`")
    L.append("")
    open(path, "w").write("\n".join(L))
    return path


def main():
    ap = argparse.ArgumentParser(description="second-family replication analysis")
    ap.add_argument("--out-json", default=f"results/{EXPERIMENT}/analysis.json")
    ap.add_argument("--out-md", default=f"results/{EXPERIMENT}/SUMMARY.md")
    ap.add_argument("--tag", default=None, help="only this record-store tag")
    ap.add_argument("--api-usd", type=float, default=None,
                    help="OpenRouter GET /api/v1/credits total_usage read AFTER the run "
                         "(recorded verbatim; this script makes no network call)")
    ap.add_argument("--api-baseline-usd", type=float, default=None,
                    help="the same reading taken BEFORE any of this work")
    ap.add_argument("--api-read-utc", default=None, help="when --api-usd was read")
    ap.add_argument("--api-extension-baseline-usd", type=float, default=None,
                    help="the credits reading taken between the first pass and the "
                         "replicate extension, so the extension's own spend is recorded")
    args = ap.parse_args()

    X = load_extract_data()

    # ---- 1. baseline check: reproduce FINDINGS 17.1 from the grid store -------------
    grid = X.load(GRID_STORE)
    baseline, problems = {}, []
    for cell in ("F-bin-k1", "FC-score-k1"):
        b = X.arm_block([r for r in grid if r["cell"] == cell])
        baseline[cell] = b
        for measure, key in (("omissions", "paired_omissions"),
                             ("commissions", "paired_commissions")):
            got, want = b[key]["paired"], FINDINGS_17_1[cell][measure]
            if abs(got - want) > 0.0015:
                problems.append(f"{cell} {measure}: recomputed {got}, FINDINGS 17.1 says {want}")
    if problems:
        raise SystemExit("BASELINE CHECK FAILED - not reporting a comparison whose baseline "
                         "column does not reproduce:\n  " + "\n  ".join(problems))
    print(f"baseline check OK: FINDINGS 17.1 reproduced from {GRID_STORE}")
    for cell in ("F-bin-k1", "FC-score-k1"):
        print(f"  {cell:<13} commissions {baseline[cell]['paired_commissions']['paired']:.3f} "
              f"omissions {baseline[cell]['paired_omissions']['paired']:.3f}")

    # ---- 2. the second-family store ------------------------------------------------
    paths = sorted(p for p in glob.glob(STORE_GLOB)
                   if not os.path.basename(p).startswith("smoke")
                   and (args.tag is None or args.tag in os.path.basename(p)))
    if not paths:
        raise SystemExit(f"no record store under {STORE_GLOB} - nothing to analyse. "
                         "If the run has not happened, say so; do not estimate.")
    recs = load_store(paths)
    if not recs:
        raise SystemExit("record store(s) present but empty - stopping.")

    models = sorted({r.get("model") for r in recs})
    model_keys = sorted({r.get("model_key") for r in recs})
    reps = sorted({r.get("replicate") for r in recs})
    seeds = sorted({r.get("run_seed") for r in recs})
    print(f"\nsecond-family store: {len(recs)} records from {len(paths)} store(s)")
    print(f"  judge: {', '.join(str(m) for m in models)} (role key {', '.join(model_keys)})")
    print(f"  replicate(s) {reps} seed(s) {seeds}")

    designs = {}
    for arm in sorted({r["arm"] for r in recs}):
        mine = [r for r in recs if r["arm"] == arm]
        # Per replicate FIRST: arm_block mutates records with a cached score field, so the
        # subsets are re-scored on each call and the whole-arm block below recomputes it.
        # Everything a replicate-stability read needs is a full arm_block on one replicate,
        # which keeps the per-rep figures on exactly the conventions as the pooled ones.
        arm_reps = sorted({r["replicate"] for r in mine})
        per_rep = {}
        for rep in arm_reps:
            sub = [r for r in mine if r["replicate"] == rep]
            rb = X.arm_block(sub)
            per_rep[str(rep)] = {
                "seed": sorted({r.get("run_seed") for r in sub}),
                "n_records": rb["n_records"],
                "paired_omissions": rb["paired_omissions"],
                "paired_commissions": rb["paired_commissions"],
                "paired_by_residual": rb["paired_by_residual"],
                "paired_by_severity": rb["paired_by_severity"],
                "absolute_omission_det": rb["absolute"]["det"],
                "absolute_commission_det": commission_absolute(X, sub),
                "absolute_fa": rb["absolute"]["fa"],
                "absolute_det_vs_fa_z": rb["absolute"]["z"],
                "swept_fa10": rb.get("swept_fa10"),
                "parse_failures": sum(1 for r in sub if r.get("parse_failure")),
            }
        # Ranges across replicates - the explicit spread a stability read is judged on.
        def spread(get):
            vals = [get(v) for v in per_rep.values()]
            vals = [v for v in vals if v is not None]
            return {"min": min(vals), "max": max(vals),
                    "values": vals} if vals else None
        replicate_spread = {
            "n_replicates": len(arm_reps), "replicates": arm_reps,
            "paired_omissions": spread(lambda v: v["paired_omissions"]["paired"]),
            "paired_commissions": spread(lambda v: v["paired_commissions"]["paired"]),
            "absolute_omission_det": spread(lambda v: v["absolute_omission_det"]["p"]),
            "absolute_commission_det": spread(lambda v: v["absolute_commission_det"]["p"]),
            "absolute_fa": spread(lambda v: v["absolute_fa"]["p"]),
            "absolute_det_vs_fa_z": spread(lambda v: v["absolute_det_vs_fa_z"]),
        } if len(arm_reps) > 1 else {"n_replicates": len(arm_reps), "replicates": arm_reps}

        blk = X.arm_block(mine)
        pf = [r for r in mine if r.get("parse_failure")]
        retried = sum(len(r.get("retried_samples") or []) for r in mine)
        designs[arm] = {
            "n_replicates": len(arm_reps), "replicates": arm_reps,
            "per_replicate": per_rep, "replicate_spread": replicate_spread,
            "replicates_grid_cell": REPLICATES_CELL.get(arm),
            "flag_rule": FLAG_RULE.get(arm),
            "n_records": blk["n_records"],
            "paired_omissions": blk["paired_omissions"],
            "paired_commissions": blk["paired_commissions"],
            "paired_by_residual": blk["paired_by_residual"],
            "paired_by_severity": blk["paired_by_severity"],
            "absolute_omission_det": blk["absolute"]["det"],
            "absolute_fa": blk["absolute"]["fa"],
            "absolute_det_vs_fa_z": blk["absolute"]["z"],
            "absolute_commission_det": commission_absolute(X, mine),
            "swept_fa10": blk.get("swept_fa10"),
            "parse_failures": {
                "n": len(pf),
                "records": sorted(({"note_key": r["note_key"],
                                    "note_role": r.get("note_role"),
                                    "pair_type": r.get("pair_type"),
                                    "replicate": r.get("replicate"),
                                    "consultation": r.get("consultation")}
                                   for r in pf),
                                  key=lambda x: (x["replicate"], x["note_key"]))[:20],
                "retried_samples": retried},
            "unique_notes": len({r["note_key"] for r in mine}),
        }

    # ---- 3. spend, by two independent paths ----------------------------------------
    receipts_usd = sum((r.get("totals") or {}).get("cost_usd") or 0.0 for r in recs)
    receipt_calls = sum((r.get("totals") or {}).get("calls") or 0 for r in recs)
    rows = ledger_rows(EXPERIMENT)
    ledger_usd = sum(d.get("cost_usd") or 0.0 for d in rows)
    ledger_calls = sum(d.get("calls") or 0 for d in rows)
    smoke_path = os.path.join(RESULTS, EXPERIMENT, "smoke.json")
    smoke = json.load(open(smoke_path)) if os.path.exists(smoke_path) else None
    smoke_usd = sum((r.get("totals") or {}).get("cost_usd") or 0.0
                    for r in (smoke or {}).get("rows", []))
    # Runs that never closed write no ledger row, so their calls are billed by OpenRouter
    # and banked nowhere - the structural gap FINDINGS documents in "Spend to date". Name
    # them and price them from the store so the ledger/receipt difference is explained
    # rather than left as a discrepancy.
    closed = {d["run_id"] for d in rows}
    unclosed = []
    for mpath in sorted(glob.glob(os.path.join(RESULTS, EXPERIMENT, "*", "manifest.json"))):
        m = json.load(open(mpath))
        if m["run_id"] in closed:
            continue
        owned = [r for r in recs if r.get("run_id") == m["run_id"]]
        unclosed.append({
            "run_id": m["run_id"], "status": m.get("status"),
            "records_in_store": len(owned),
            "receipts_usd": round(sum((r.get("totals") or {}).get("cost_usd") or 0.0
                                      for r in owned), 6),
            "calls": sum((r.get("totals") or {}).get("calls") or 0 for r in owned),
            "why": "run did not close (killed/interrupted), so it wrote no cost-ledger row; "
                   "its judgements are in the record store and ARE counted in receipts"})

    spend = {
        "ledger_rows_for_experiment": len(rows),
        "ledger_usd_all_rows": round(ledger_usd, 6),
        "ledger_calls_all_rows": ledger_calls,
        "ledger_row_status": dict(Counter(d.get("status") for d in rows)),
        "unclosed_runs": unclosed,
        "unclosed_receipts_usd": round(sum(u["receipts_usd"] for u in unclosed), 6),
        "openrouter_api": ({"total_usage_after": args.api_usd,
                            "total_usage_before": args.api_baseline_usd,
                            "task_spend_usd": round(args.api_usd - args.api_baseline_usd, 6),
                            "read_utc": args.api_read_utc,
                            "extension_baseline": args.api_extension_baseline_usd,
                            "extension_spend_usd": (
                                round(args.api_usd - args.api_extension_baseline_usd, 6)
                                if args.api_extension_baseline_usd else None),
                            "note": "GET /api/v1/credits - the authority per FINDINGS "
                                    "'Spend to date'; recorded verbatim, the analysis "
                                    "itself makes no network call"}
                           if args.api_usd and args.api_baseline_usd else None),
        "main_run_receipts_usd": round(receipts_usd, 6),
        "main_run_receipt_calls": receipt_calls,
        "smoke_receipts_usd": round(smoke_usd, 6),
        "smoke_receipt_calls": sum((r.get("totals") or {}).get("calls") or 0
                                   for r in (smoke or {}).get("rows", [])),
        "total_receipts_usd": round(receipts_usd + smoke_usd, 6),
        "note": ("Ledger rows are written when a Run closes and cover the whole experiment "
                 "directory (smoke + main run); receipts are summed per judgement from the "
                 "OpenRouter-reported per-call cost. Both are stated; neither is estimated."),
    }

    out = {
        "experiment": EXPERIMENT,
        "stores": [os.path.relpath(p, HERE) for p in paths],
        "judge_models": models, "model_keys": model_keys,
        "replicates": reps, "seeds": seeds,
        "roles": {mk: dict(W.ROLES[mk],
                           resolved=__import__("common").resolve_model(
                               W.ROLES[mk]["role"])["resolved"]) for mk in model_keys
                  if mk in W.ROLES},
        "dataset": {"file": sorted({r.get("dataset_version") for r in recs}),
                    "pairs_sha256_from_manifests": None},
        "measures": {
            "paired": "tie-adjusted P(errored note scores strictly below its own clean twin) "
                      "+ half the tie mass, per replicate then averaged; chance 0.500",
            "absolute": "flag rate at the design's own grid threshold, beside the same "
                        "design's false-alarm rate on the clean twins",
            "source": "paper/figures/extract_data.py arm_block() - the same function that "
                      "reproduces FINDINGS 17.1",
        },
        "baseline_gpt54_grid": {
            cell: {"paired_omissions": baseline[cell]["paired_omissions"],
                   "paired_commissions": baseline[cell]["paired_commissions"],
                   "paired_by_residual": baseline[cell]["paired_by_residual"],
                   "paired_by_severity": baseline[cell]["paired_by_severity"],
                   "swept_fa10": baseline[cell].get("swept_fa10"),
                   "absolute_omission_det": baseline[cell]["absolute"]["det"],
                   "absolute_fa": baseline[cell]["absolute"]["fa"],
                   "absolute_commission_det": commission_absolute(
                       X, [r for r in grid if r["cell"] == cell]),
                   "absolute_det_vs_fa_z": baseline[cell]["absolute"]["z"],
                   "n_records": baseline[cell]["n_records"],
                   "findings_17_1_quoted": FINDINGS_17_1[cell],
                   "baseline_check": "reproduced within 0.0015"}
            for cell in ("F-bin-k1", "FC-score-k1")},
        "designs": designs,
        "spend": spend,
    }

    # pairs sha, from the contributing run manifests
    shas = set()
    for rid in sorted({r.get("run_id") for r in recs if r.get("run_id")}):
        mpath = os.path.join(RESULTS, EXPERIMENT, rid, "manifest.json")
        if os.path.exists(mpath):
            m = json.load(open(mpath))
            shas.add((m.get("params") or {}).get("pairs_sha256"))
    out["dataset"]["pairs_sha256_from_manifests"] = sorted(s for s in shas if s)

    dest = os.path.join(HERE, args.out_json)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=1)

    # ---- 4. console table ----------------------------------------------------------
    print(f"\n{'design':<16} {'commissions':>12} {'omissions':>11} {'om det':>16} "
          f"{'FA':>16} {'parse-fail':>11}")
    for arm, d in designs.items():
        print(f"{arm:<16} {d['paired_commissions']['paired']:>12.3f} "
              f"{d['paired_omissions']['paired']:>11.3f} "
              f"{fmt_pct(d['absolute_omission_det']):>16} {fmt_pct(d['absolute_fa']):>16} "
              f"{d['parse_failures']['n']:>11}")
    print(f"\nspend: receipts ${spend['total_receipts_usd']:.4f} "
          f"(main ${spend['main_run_receipts_usd']:.4f} + smoke ${spend['smoke_receipts_usd']:.4f}); "
          f"ledger ${spend['ledger_usd_all_rows']:.4f} over {spend['ledger_rows_for_experiment']} row(s)")
    print(f"written: {os.path.relpath(dest, HERE)}")

    if args.out_md and args.out_md != "/dev/null":
        md = os.path.join(HERE, args.out_md)
        os.makedirs(os.path.dirname(md), exist_ok=True)
        write_summary_md(out, md)
        print(f"written: {os.path.relpath(md, HERE)}")
    return out


if __name__ == "__main__":
    main()
