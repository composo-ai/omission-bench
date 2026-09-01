#!/usr/bin/env python3
r"""Read out the MEDEC external anchor. No API calls; recomputable from results/ alone.

Conventions are the judge paper's, because the whole point is that these numbers sit beside those:

  ABSOLUTE (primary here)  flag rate at the arm's own pre-registered rule, on one text at a time,
                           always quoted beside the same arm's false-alarm rate on error-free
                           texts. The deployment frame.
  PAIRED (secondary)       tie-adjusted P(errored text scores strictly below its OWN clean twin)
                           + half the tie mass; chance = 0.500. This is the measure the paper
                           reports for commissions, so it is the apples-to-apples column.
  SWEPT                    detection at the best threshold holding false alarms <= 10% (and 5%),
                           read off the record-level ROC - the same construction as the swept
                           detection numbers the judge paper reports.

Parse failures follow a pre-registered conservative rule: in flag-mode metrics a parse failure takes
the outcome least favourable to the judge (a miss on an error-present text, a false alarm on an
error-free one); in paired metrics it counts as non-discrimination (a tie). Score-based metrics
drop it and report the count.

    python3 anchors/wb_medec_analyze.py [--tag medec-main]
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import argparse, json, math, os
from collections import Counter, defaultdict

from scipy import stats

from common import HERE, RESULTS
import w2_analyze as Z          # wilson / auc_lower_is_positive / mcnemar_exact - one impl only

EXPERIMENT = "wb-medec"
OUT = os.path.join(HERE, "wb_medec_results.json")

# Published MEDEC reference points. VERIFIED 2026-08-14 by two independent WebFetch reads of the
# primary paper's own Table 3 (arXiv:2412.19260 HTML, v2), which agreed on every figure. The
# same fetch reproduced three counts that our ingest derived independently from the CSV - 597 MS
# test rows, 925 total test rows, 328 UW rows - which is what makes it a read of the real table
# rather than a summary of one. Table 3's caption: "Accuracy and error correction scores on each
# subset: MS & UW Test Sets." The metric is the paper's own "Error Flag" accuracy for subtask A
# (the paper says only "We relied on Accuracy for Error Flag Prediction (subtask A)"; no formula
# is given, so we compute plain accuracy over all texts and say so).
LEADERBOARD = {
    "source": "Ben Abacha et al., MEDEC (arXiv:2412.19260), Table 3, MS (M#1) subset rows",
    "verified": "2026-08-14, two independent WebFetch reads of arxiv.org/html/2412.19260v2 agreed",
    "metric": "Error Flag accuracy, subtask A, on the full 597-text MS test split",
    "rows": {"Claude 3.5 Sonnet (2024-10-22)": 0.6750, "o1-preview (2024-09-12)": 0.7286,
             "Medical Doctor #1": 0.8125, "Medical Doctor #2": 0.6890},
    "caveat": ("published figures are on all 597 MS test texts; ours are on the pre-registered "
               "stratified 300 of them, and the official MEDIQA-CORR test set also contains the "
               "access-restricted UW half. Reference points, never a leaderboard entry."),
}


def two_prop_z(k1, n1, k2, n2):
    """Pooled two-proportion z: detection rate against false-alarm rate, the `det_vs_fa` column."""
    if not n1 or not n2:
        return None
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return {"z": None, "p": None}
    z = (p1 - p2) / se
    return {"z": round(z, 4), "p": round(2 * stats.norm.sf(abs(z)), 8)}


def flagged_conservative(r):
    """Flag outcome with the conservative parse-failure rule applied."""
    if r["parse_failure"]:
        return not r["error_flag"]      # miss on error-present, false alarm on error-free
    return bool(r["flagged"])


def roc(pos, neg):
    """Sweep every threshold. `pos` = errored scores, `neg` = error-free scores; a note is
    flagged iff its score <= t, so lower scores mean 'worse note' throughout."""
    if not pos or not neg:
        return []
    out = []
    for t in sorted(set(pos) | set(neg)):
        det = sum(1 for v in pos if v <= t) / len(pos)
        fa = sum(1 for v in neg if v <= t) / len(neg)
        out.append({"t": t, "det": round(det, 6), "fa": round(fa, 6),
                    "youden_j": round(det - fa, 6)})
    return out


def swept(curve, cap):
    best = None
    for p in curve:
        if p["fa"] <= cap and (best is None or p["det"] > best["det"]):
            best = p
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="medec-main")
    args = ap.parse_args()

    path = os.path.join(RESULTS, EXPERIMENT, "_state", f"{args.tag}.jsonl")
    records = [json.loads(l) for l in open(path) if l.strip()]
    by_arm = defaultdict(list)
    for r in records:
        by_arm[r["arm"]].append(r)

    items = json.load(open(os.path.join(HERE, "wb_medec_items.json")))["items"]
    sample = sorted(it["text_id"] for it in items if it["in_sample"])

    out = {
        "generated": "2026-08-14", "script": "wb_medec_analyze.py", "design":
            "external anchor: this study's judges on a third-party labelled corpus, two arms",
        "store": os.path.relpath(path, HERE), "n_records": len(records),
        "dataset": {
            "name": "MEDEC MS test split", "license": "CC BY 4.0",
            "citation": ("Ben Abacha, Yim, Fu, Sun, Yetisgen, Xia, Lin. MEDEC: A Benchmark for "
                         "Medical Error Detection and Correction in Clinical Notes. "
                         "Findings of ACL 2025, 22539-22550."),
            "labels": "physician-written and physician-validated; commission errors only, "
                      "one per errored text, erroneous sentence identified; no omissions",
            "sample": "pre-registered stratified n=300, seed 20260728, frozen by "
                      "wb_medec_ingest.py before any judging",
            "text_ids": sample},
        "arms": {}, "leaderboard": LEADERBOARD,
    }

    flags, label_of = {}, {}
    for arm, rs in sorted(by_arm.items()):
        by_role = defaultdict(list)
        for r in rs:
            by_role[r["note_role"]].append(r)
        err, free, twin = by_role["errored"], by_role["error_free"], by_role["clean_twin"]

        det_k = sum(flagged_conservative(r) for r in err)
        fa_k = sum(flagged_conservative(r) for r in free)
        fat_k = sum(flagged_conservative(r) for r in twin)
        det, fa, fat = (Z.wilson(det_k, len(err)), Z.wilson(fa_k, len(free)),
                        Z.wilson(fat_k, len(twin)))
        acc_k = det_k + (len(free) - fa_k)
        flags[arm] = {r["text_id"]: flagged_conservative(r) for r in err + free}
        label_of.update({r["text_id"]: r["error_flag"] for r in err + free})

        pos = [r["aggregate"] for r in err if r["aggregate"] is not None]
        neg = [r["aggregate"] for r in free if r["aggregate"] is not None]
        negt = [r["aggregate"] for r in twin if r["aggregate"] is not None]
        curve = roc(pos, neg)
        youden = max(curve, key=lambda p: p["youden_j"]) if curve else None

        # paired: each errored text against its OWN reconstructed clean twin
        twin_by = {r["pair_key"]: r for r in twin}
        wins = losses = ties = 0
        for r in err:
            t = twin_by.get(r["pair_key"])
            if t is None:
                continue
            if r["aggregate"] is None or t["aggregate"] is None:
                ties += 1                     # a parse failure counts as non-discrimination
            elif r["aggregate"] < t["aggregate"]:
                wins += 1
            elif r["aggregate"] > t["aggregate"]:
                losses += 1
            else:
                ties += 1
        npair = wins + losses + ties
        paired = round((wins + 0.5 * ties) / npair, 6) if npair else None
        sign_p = (float(stats.binomtest(wins, wins + losses, 0.5).pvalue)
                  if wins + losses else 1.0)

        cap10 = swept(curve, 0.10)
        cap05 = swept(curve, 0.05)

        # Best error-flag accuracy reachable at ANY single threshold. This is an ORACLE number -
        # the cut is chosen on these same 300 texts - so it is an upper bound, not a result. It
        # is here because our arms' flag rules were pre-registered for scribe-note evaluation and
        # never tuned on MEDEC, so quoting only the own-rule accuracy against systems built for
        # this task would confound the operating point with separability.
        best_acc = None
        for p in curve:
            k = round(p["det"] * len(pos)) + (len(neg) - round(p["fa"] * len(neg)))
            if best_acc is None or k > best_acc["correct"]:
                best_acc = {"t": p["t"], "correct": k, "accuracy": round(k / (len(pos) + len(neg)), 6),
                            "det": p["det"], "fa": p["fa"]}

        by_type = {}
        for r in err:
            t = r["error_type"] or "(none)"
            d = by_type.setdefault(t, {"n": 0, "flagged": 0, "swept10": 0})
            d["n"] += 1
            d["flagged"] += flagged_conservative(r)
            if cap10 and r["aggregate"] is not None and r["aggregate"] <= cap10["t"]:
                d["swept10"] += 1
        for t, d in by_type.items():
            d["det_own_rule"] = round(d["flagged"] / d["n"], 4)
            d["det_swept10"] = round(d["swept10"] / d["n"], 4)

        entry = {
            "k": rs[0]["k"], "flag_rule": rs[0]["flag_rule"], "model": rs[0]["model"],
            "n": {"errored": len(err), "error_free": len(free), "clean_twin": len(twin)},
            "parse_failures": sum(r["parse_failure"] for r in rs),
            "absolute": {
                "detection": det, "false_alarm_error_free": fa, "false_alarm_clean_twin": fat,
                "det_vs_fa": two_prop_z(det_k, len(err), fa_k, len(free)),
                "error_flag_accuracy": Z.wilson(acc_k, len(err) + len(free)),
                "note": "accuracy is the paper's subtask-A frame: correct binary call over all "
                        "300 texts, our arms at their OWN pre-registered flag rule"},
            "threshold_free": {
                "auc_vs_error_free": Z.auc_lower_is_positive(pos, neg),
                "auc_vs_clean_twin": Z.auc_lower_is_positive(pos, negt),
                "youden": youden,
                "best_accuracy_oracle_threshold": best_acc,
                "swept_det_at_fa_10": cap10, "swept_det_at_fa_05": cap05,
                "n_scored": {"errored": len(pos), "error_free": len(neg),
                             "clean_twin": len(negt)}},
            "paired": {"tie_adjusted": paired, "n_pairs": npair, "wins": wins,
                       "losses": losses, "ties": ties, "sign_test_p": round(sign_p, 8),
                       "note": "errored text vs its own clean twin; comparable to the "
                               "paper's commission columns (chance 0.500)"},
            "by_error_type": by_type,
            "scores": {role: {"mean": round(sum(v) / len(v), 4), "n": len(v)}
                       for role, v in (("errored", pos), ("error_free", neg),
                                       ("clean_twin", negt)) if v},
        }

        # the engineered arm's own structured list is a decision rule in its own right:
        # flag iff it returns an entry it graded critical.
        if any("omissions_reported" in r for r in rs):
            def crit(r):
                return any((o or {}).get("severity") == "critical"
                           for o in r.get("omissions_reported") or [])
            ck, cn = sum(crit(r) for r in err), len(err)
            fk, fn = sum(crit(r) for r in free), len(free)
            entry["critical_list_rule"] = {
                "detection": Z.wilson(ck, cn), "false_alarm": Z.wilson(fk, fn),
                "det_vs_fa": two_prop_z(ck, cn, fk, fn),
                "note": "flag iff the arm's returned list contains a critical entry - the rule "
                        "that beat this arm's own score threshold on the study's benchmark"}
        out["arms"][arm] = entry

    # ---- pairwise contrasts, split by label, because the two halves answer different questions:
    # on error-free texts the delta IS the false-alarm cost of one arm over another; on
    # errored texts it is the detection difference.
    names = sorted(flags)
    contrasts = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            entry = {}
            for label, keep in (("all_300", None), ("error_free", False), ("errored", True)):
                fa = {k: v for k, v in flags[a].items()
                      if keep is None or label_of[k] == keep}
                fb = {k: v for k, v in flags[b].items()
                      if keep is None or label_of[k] == keep}
                mc = Z.mcnemar_exact(fa, fb)
                n = mc["n_items"]
                bb, cc = mc["discordant_a_only"], mc["discordant_b_only"]
                delta = (bb - cc) / n if n else None
                se = (math.sqrt(max(bb + cc - (bb - cc) ** 2 / n, 0)) / n) if n else None
                mc["delta_a_minus_b"] = round(delta, 6) if delta is not None else None
                if se is not None:
                    mc["delta_ci95"] = [round(delta - 1.96 * se, 6), round(delta + 1.96 * se, 6)]
                entry[label] = mc
            contrasts[f"{a} vs {b}"] = entry
    out["arm_contrasts"] = {
        "pairs": contrasts,
        "note": ("exact McNemar on the binary flag, same texts both arms; delta = a's rate minus "
                 "b's, with a Wald 95% CI on the paired difference. The error_free row is what "
                 "one criterion costs the other in false alarms.")}

    tot = {"calls": 0, "cost_usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
    for r in records:
        for f in tot:
            tot[f] += (r.get("totals") or {}).get(f, 0) or 0
    tot["cost_usd"] = round(tot["cost_usd"], 4)
    out["spend"] = {**tot, "run_ids": sorted({r["run_id"] for r in records}),
                    "note": "wb-medec own receipts only, summed over the record store"}

    json.dump(out, open(OUT, "w"), indent=1)

    print(f"{'arm':<26} {'det':>14} {'FA(free)':>14} {'FA(twin)':>10} {'z':>6} {'acc':>7} "
          f"{'AUC':>6} {'paired':>7} {'swept@10%':>12}")
    for arm, e in out["arms"].items():
        ab, tf = e["absolute"], e["threshold_free"]
        s10 = tf["swept_det_at_fa_10"]
        print(f"{arm:<26} {ab['detection']['p']:>6.3f} ({ab['detection']['k']:>3}/"
              f"{ab['detection']['n']:<3}) {ab['false_alarm_error_free']['p']:>6.3f} "
              f"({ab['false_alarm_error_free']['k']:>3}/{ab['false_alarm_error_free']['n']:<3})"
              f" {ab['false_alarm_clean_twin']['p']:>10.3f}"
              f" {(ab['det_vs_fa'] or {}).get('z') or 0:>6.2f}"
              f" {ab['error_flag_accuracy']['p']:>7.3f} {tf['auc_vs_error_free']:>6.3f}"
              f" {e['paired']['tie_adjusted']:>7.3f}"
              + (f" {s10['det']:>6.3f}@{s10['fa']:.2f}" if s10 else f" {'-':>12}"))
    print(f"\npublished MS-subset error-flag accuracy (Table 3): "
          + ", ".join(f"{k} {v:.4f}" for k, v in LEADERBOARD["rows"].items()))
    print(f"spend ${tot['cost_usd']:.2f} over {tot['calls']} calls -> "
          f"{os.path.relpath(OUT, HERE)}")


if __name__ == "__main__":
    main()
