"""w2_pipeline_analyze.py - read-out for Arm B, the pipeline judge (w2_pipeline.py).

Everything here is computed from results/w2-pipeline/ alone, as the study's reproducibility
rules require: the tier record stores, the stage-cost log, and the cost ledger. Nothing is
re-judged.

What it reports, per tier, and why each measure is here:

  (a) the detection / false-alarm CURVE, swept over the score threshold. The tiers write
      `flagged: null` on purpose - no threshold is pre-registered for them - so the operating
      point is chosen here, subject to the deployment constraint: the headline is DETECTION AT
      THE BEST THRESHOLD WITH FA <= 10% (and <= 5%), which is the measure the papers report
      for the grid (4.7-8.3%).
  (b) the same split by pair_class (omit-complete vs omit-partial) and by residual level
      (partial-weak vs partial-strong) - partials are the point of the trichotomy, and the
      grid's own numbers show its signal dying exactly there (0.526 at a strong residual).
  (c) PAIRED tie-adjusted discrimination, P(errored scores below its own clean twin) + half the
      tie mass, for comparability with the grid's paired table (best cell 0.634). A relative
      measure that needs the clean twin: reported next to (a), never instead of it.
  (d) commissions (add / change) as calibration - the grid is near-ceiling there (0.939), so a
      pipeline that cannot see a fabrication is broken rather than conservative.
  (e) measured cost per note, marginal and fully loaded, from the receipts.

Small-sample warning it prints for itself: the smoke subset has 32 clean notes, so the FA axis
moves in steps of 3.1pp and "FA <= 5%" means at most one false alarm. Treat the operating points
as indicative of scale, not as estimates with two significant figures.

    python3 judges/w2_pipeline_analyze.py --tag smoke
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import argparse, glob, json, math, os
from collections import Counter, defaultdict

from common import HERE, RESULTS
import w2_common as W

EXPERIMENT = "w2-pipeline"
STATE = os.path.join(RESULTS, EXPERIMENT, "_state")

# The published figures for the eight-cell ablation grid this arm is measured against
# (495 pairs x 3 replicates, gpt-5.4).
GRID = {
    "det_at_fa10": {"FC-score-k8": 0.083, "FC-score-k1": 0.047},
    "paired_omissions": {"FC-score-k8": 0.634, "FC-score-k1": 0.585, "F-bin-k1": 0.500},
    "paired_by_residual": {"complete": 0.690, "partial-weak": 0.607, "partial-strong": 0.526},
    "paired_by_severity": {"critical": 0.683, "supporting": 0.586, "peripheral": 0.568},
    "paired_commissions": 0.939,
    "pooled_auc": [0.503, 0.575],
}
RESIDUALS = ["complete", "partial-weak", "partial-strong", "partial"]
CLASSES = ["omit-complete", "omit-partial"]
SEVERITIES = ["critical", "supporting", "peripheral"]


def wilson(k, n, z=1.959963984540054):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": k, "n": n, "p": round(p, 4), "lo": round(max(0.0, c - h), 4),
            "hi": round(min(1.0, c + h), 4)}


def residual_level_of(rec):
    """Same normalisation as w2_analyze: recover the graded level from the surviving
    mention's strength when the record only carries the coarse 'partial'."""
    lvl = (rec.get("residual_level") or "").strip().lower()
    if lvl == "partial":
        return W.STRENGTH_TO_LEVEL.get((rec.get("residual_strength") or "").strip().lower(),
                                       "partial")
    return lvl or ("complete" if rec.get("pair_type") == "omit" else rec.get("pair_type"))


def load_tier(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


# ---------------------------------------------------------------- the measures
def sweep(err_scores, clean_scores):
    """Every distinct operating point of `flag iff aggregate < t`, coarse to fine."""
    cand = sorted({round(v, 9) for v in list(err_scores) + list(clean_scores)})
    out = []
    for t in cand + [(cand[-1] + 1e-6) if cand else 1.0]:
        det_k = sum(1 for v in err_scores if v < t)
        fa_k = sum(1 for v in clean_scores if v < t)
        out.append({"threshold": round(t, 6),
                    "det": det_k / len(err_scores) if err_scores else None,
                    "det_k": det_k, "n_err": len(err_scores),
                    "fa": fa_k / len(clean_scores) if clean_scores else None,
                    "fa_k": fa_k, "n_clean": len(clean_scores)})
    return out


def best_at_fa(curve, cap):
    """The highest-detection operating point whose false-alarm rate stays within `cap`."""
    ok = [p for p in curve if p["fa"] is not None and p["fa"] <= cap + 1e-12]
    if not ok:
        return None
    best = max(ok, key=lambda p: (p["det"], -p["fa"], p["threshold"]))
    return dict(best, cap=cap, det_ci=wilson(best["det_k"], best["n_err"]),
                fa_ci=wilson(best["fa_k"], best["n_clean"]))


def auc_lower_is_positive(pos, neg):
    """P(errored < clean) + 0.5 P(tie), pooled (not paired)."""
    if not pos or not neg:
        return None
    less = sum(1 for a in pos for b in neg if a < b)
    ties = sum(1 for a in pos for b in neg if a == b)
    return round((less + 0.5 * ties) / (len(pos) * len(neg)), 4)


def sign_test(p):
    """Two-sided exact sign test on the strict wins/losses behind a paired figure (ties are
    uninformative and are dropped, which is what makes it a test of the ordering and not of
    the tie mass)."""
    if not p:
        return None
    n = p["strict_wins"] + p["losses"]
    if not n:
        return {"n_informative": 0, "p": 1.0}
    from scipy import stats
    return {"n_informative": n,
            "p": round(float(stats.binomtest(p["strict_wins"], n, 0.5).pvalue), 8)}


def two_prop_z(k1, n1, k2, n2):
    """det vs FA at one operating point - the comparison the grid reports, on 32-note pools."""
    if not n1 or not n2:
        return None
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if not se:
        return None
    from scipy import stats
    z = (k1 / n1 - k2 / n2) / se
    return {"z": round(z, 3), "p": round(float(2 * (1 - stats.norm.cdf(abs(z)))), 5)}


def paired(rows):
    """Tie-adjusted paired discrimination over (err, clean) aggregate pairs."""
    rows = [(e, c) for e, c in rows if e is not None and c is not None]
    if not rows:
        return None
    s = sum(1.0 if e < c else 0.5 if e == c else 0.0 for e, c in rows)
    return {"paired": round(s / len(rows), 4), "n": len(rows),
            "strict_wins": sum(1 for e, c in rows if e < c),
            "ties": sum(1 for e, c in rows if e == c),
            "losses": sum(1 for e, c in rows if e > c)}


def rescore(rec, mode):
    """Re-score ONE purchased record under a different aggregation rule. No new calls: every
    variant reads the per-fact verdict map that was already bought.

      unweighted   severity weights off (all facts weight 1)
      strict       "partial" scores 0 - captured fully or not at all
      lenient      "partial" scores 1 - the B2 question asked of B3's verdicts
      norefute     the refute pass undone (flipped ids restored to absent)

    These are DIAGNOSTICS for attribution, not a menu to pick a winner from: choosing the
    variant that scores best on this smoke's own omission pairs would be tuning on the test.
    """
    d = rec.get("detail") or {}
    v = dict(d.get("verdicts") or {})
    if not v:
        return None
    if mode == "norefute":
        for fl in d.get("refute_flip_detail") or []:
            v[fl["id"]] = "absent"
    sev = {m["id"]: m.get("severity") for m in (d.get("missing_facts") or [])}
    credit = {"full": 1.0, "present": 1.0, "absent": 0.0,
              "partial": 0.0 if mode == "strict" else 1.0 if mode == "lenient" else 0.5}
    weights = {"critical": 4.0, "supporting": 2.0, "peripheral": 1.0}
    num = den = 0.0
    for fid, lab in v.items():
        w = 1.0 if mode == "unweighted" else weights.get(sev.get(fid), 2.0)
        num += w * credit.get(lab, 0.0)
        den += w
    return round(num / den, 6) if den else None


def variant_report(recs, mode):
    """The two headline numbers - paired discrimination and det@FA<=10% - under a variant."""
    scores = {r["note_key"]: rescore(r, mode) for r in recs}
    cleans = {r["note_key"]: r for r in recs if r["note_role"] == "clean"}
    omis = [r for r in recs if r["note_role"] == "errored" and r["pair_type"] == "omit"
            and scores.get(r["note_key"]) is not None]
    clean_scores = [s for k, s in scores.items() if k in cleans and s is not None]
    if not omis or not clean_scores:
        return None
    rows = [(scores[r["note_key"]], scores.get(r["clean_key"])) for r in omis]
    p = paired([(e, c) for e, c in rows if c is not None])
    b = best_at_fa(sweep([scores[r["note_key"]] for r in omis], clean_scores), 0.10)
    return {"mode": mode, "paired": p["paired"] if p else None,
            "det_at_fa10": b["det"] if b else None, "fa": b["fa"] if b else None,
            "threshold": b["threshold"] if b else None}


def tier_report(recs, label):
    cleans = {r["note_key"]: r for r in recs if r["note_role"] == "clean"}
    errs = [r for r in recs if r["note_role"] == "errored"]
    pf = [r for r in recs if r.get("parse_failure")]
    clean_scores = [r["aggregate"] for r in cleans.values() if r["aggregate"] is not None]

    omis = [r for r in errs if r["pair_type"] == "omit" and r["aggregate"] is not None]
    comm = [r for r in errs if r["pair_type"] in ("add", "change") and r["aggregate"] is not None]
    curve = sweep([r["aggregate"] for r in omis], clean_scores)

    def sub_det(rows, thr):
        k = sum(1 for r in rows if r["aggregate"] < thr)
        return wilson(k, len(rows))

    ops = {}
    for cap, name in ((0.10, "fa10"), (0.05, "fa05"), (0.20, "fa20")):
        b = best_at_fa(curve, cap)
        if not b:
            continue
        thr = b["threshold"]
        b["by_class"] = {c: sub_det([r for r in omis if r["pair_class"] == c], thr)
                         for c in CLASSES}
        b["by_residual"] = {lv: sub_det([r for r in omis if residual_level_of(r) == lv], thr)
                            for lv in RESIDUALS}
        b["by_severity"] = {s: sub_det([r for r in omis if r["severity"] == s], thr)
                            for s in SEVERITIES}
        b["commissions_det"] = sub_det(comm, thr)
        b["det_vs_fa"] = two_prop_z(b["det_k"], b["n_err"], b["fa_k"], b["n_clean"])
        ops[name] = b

    def pair_rows(rows):
        return [(r["aggregate"], (cleans.get(r["clean_key"]) or {}).get("aggregate"))
                for r in rows]

    pr = {"omissions": paired(pair_rows(omis)), "commissions": paired(pair_rows(comm)),
          "by_commission_type": {
              t: paired(pair_rows([r for r in comm if r["pair_type"] == t]))
              for t in ("add", "change")},
          "by_class": {c: paired(pair_rows([r for r in omis if r["pair_class"] == c]))
                       for c in CLASSES},
          "by_residual": {lv: paired(pair_rows([r for r in omis if residual_level_of(r) == lv]))
                          for lv in RESIDUALS},
          "by_severity": {s: paired(pair_rows([r for r in omis if r["severity"] == s]))
                          for s in SEVERITIES}}
    pr["omissions_sign_test"] = sign_test(pr["omissions"])

    det_scores = [r["aggregate"] for r in omis]
    facts = [r.get("n_facts") for r in recs if r.get("n_facts")]
    flips = sum(r.get("refute_flips") or 0 for r in recs)
    pre_abs = sum((r.get("detail") or {}).get("n_absent_pre_refute") or 0 for r in recs)
    unver = sum((r.get("detail") or {}).get("refute_unverified") or 0 for r in recs)
    return {
        "tier": label, "n_records": len(recs), "n_clean": len(cleans), "n_omission": len(omis),
        "n_commission": len(comm), "n_parse_failures": len(pf),
        "parse_failure_keys": [r["key"] for r in pf][:10],
        "n_facts": {"mean": round(sum(facts) / len(facts), 1) if facts else None,
                    "min": min(facts) if facts else None, "max": max(facts) if facts else None,
                    "total_judged": sum(facts)},
        "verdict_mix": {k: sum(r.get(f"n_{k}") or 0 for r in recs)
                        for k in ("full", "partial", "absent")},
        "refute": {"absences_flagged": pre_abs, "flips": flips,
                   "flip_rate": round(flips / pre_abs, 4) if pre_abs else None,
                   "unverified_quotes_rejected": unver},
        "score_spread": {"clean_mean": round(sum(clean_scores) / len(clean_scores), 4)
                         if clean_scores else None,
                         "err_mean": round(sum(det_scores) / len(det_scores), 4)
                         if det_scores else None,
                         "n_distinct_scores": len({round(v, 6) for v in clean_scores + det_scores})},
        "pooled_auc": auc_lower_is_positive(det_scores, clean_scores),
        "operating_points": ops, "paired": pr, "curve": curve,
        "variants": {m: variant_report(recs, m)
                     for m in ("unweighted", "strict", "lenient", "norefute")},
    }


# ---------------------------------------------------------------- cost
def cost_report(recs_by_tier, stage_path):
    stages = []
    if os.path.exists(stage_path):
        with open(stage_path) as f:
            for line in f:
                try:
                    stages.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    ex = [s for s in stages if s["stage"] == "extract"]
    au = [s for s in stages if s["stage"] == "audit"]
    ex_cost = sum(s["totals"]["cost_usd"] for s in ex)
    au_cost = sum(s["totals"]["cost_usd"] for s in au)
    out = {"extraction": {"calls": len(ex), "usd": round(ex_cost, 4),
                          "per_consultation": round(ex_cost / len(ex), 4) if ex else None},
           "audit": {"calls": len(au), "usd": round(au_cost, 4),
                     "per_consultation": round(au_cost / len(au), 4) if au else None},
           "tiers": {}}
    for tier, recs in recs_by_tier.items():
        judged = [r for r in recs if r["aggregate"] is not None or r.get("parse_failure")]
        n = len(judged) or 1
        chk = sum(r["totals"]["cost_usd"] for r in recs)
        notes_per_consult = len(recs) / max(1, len({r["consultation"] for r in recs}))
        shared = (ex_cost / len(ex) if ex else 0.0) + ((au_cost / len(au)) if (au and tier == "B3")
                                                       else 0.0)
        out["tiers"][tier] = {
            "n_notes": len(recs),
            "check_usd_total": round(chk, 4),
            "marginal_usd_per_note": round(chk / n, 4),
            "shared_usd_per_consultation": round(shared, 4),
            "notes_per_consultation": round(notes_per_consult, 2),
            "loaded_usd_per_note": round(chk / n + shared / notes_per_consult, 4),
            "calls_per_note": round(sum(len(r.get("calls") or []) for r in recs) / n, 2)}
    return out


def ledger_total(experiment):
    path = os.path.join(RESULTS, "cost_ledger.jsonl")
    tot, rows = 0.0, 0
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("experiment") == experiment:
                    tot += d.get("cost_usd") or 0.0
                    rows += 1
    return round(tot, 4), rows


# ---------------------------------------------------------------- printing
def pct(x):
    return "  -  " if x is None else f"{100 * x:5.1f}%"


def num(x, w=5):
    return " " * w if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser(description="Arm B read-out (B2 / B3 pipeline judge)")
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--out", default="w2_pipeline_results.json")
    args = ap.parse_args()

    recs_by_tier = {}
    for path in sorted(glob.glob(os.path.join(STATE, f"{args.tag}-*.jsonl"))):
        stem = os.path.basename(path)[:-6]
        tier = stem.split("-")[-1]
        if tier in ("B2", "B3"):
            recs_by_tier[tier] = load_tier(path)
    if not recs_by_tier:
        raise SystemExit(f"no tier stores matching {args.tag}-*.jsonl in {STATE}")

    reports = {t: tier_report(r, t) for t, r in sorted(recs_by_tier.items())}
    costs = cost_report(recs_by_tier, os.path.join(STATE, f"{args.tag}-stages.jsonl"))
    led, rows = ledger_total(EXPERIMENT)

    res = {"tag": args.tag, "experiment": EXPERIMENT, "grid_reference": GRID,
           "tiers": reports, "cost": costs,
           "ledger_total_usd": led, "ledger_rows": rows,
           "caveat": ("38-pair smoke: 32 clean notes, so the FA axis steps in 3.1pp units and "
                      "FA<=5% means at most one false alarm. Operating points are indicative of "
                      "scale, not two-significant-figure estimates.")}
    json.dump(res, open(os.path.join(HERE, args.out), "w"), indent=1)

    print(f"\nARM B - the pipeline judge | tag={args.tag} | ledger {EXPERIMENT} = ${led} "
          f"over {rows} runs\n")
    for t, r in reports.items():
        f = r["n_facts"]
        v = r["verdict_mix"]
        tot = max(1, sum(v.values()))
        print(f"{t}: {r['n_records']} notes ({r['n_clean']} clean, {r['n_omission']} omission, "
              f"{r['n_commission']} commission), {r['n_parse_failures']} parse failures")
        print(f"    facts/note {f['mean']} (range {f['min']}-{f['max']}), "
              f"{f['total_judged']} per-fact verdicts | mix full {100*v['full']/tot:.1f}% "
              f"partial {100*v['partial']/tot:.1f}% absent {100*v['absent']/tot:.1f}%")
        print(f"    score mean: clean {r['score_spread']['clean_mean']}, "
              f"errored {r['score_spread']['err_mean']}, "
              f"{r['score_spread']['n_distinct_scores']} distinct values | pooled AUC "
              f"{r['pooled_auc']}")
        rf = r["refute"]
        if rf["absences_flagged"]:
            print(f"    refute: {rf['flips']}/{rf['absences_flagged']} absences rescued "
                  f"({pct(rf['flip_rate'])}), {rf['unverified_quotes_rejected']} rescues "
                  f"rejected for an unverifiable quote")

    print("\n(a) ABSOLUTE detection at matched false-alarm rate  [grid best: 8.3% at FA<=10%]")
    print(f"  {'tier':<6} {'cap':<6} {'thr':>6} {'det':>7} {'det 95% CI':>16} {'FA':>7} "
          f"{'flags':>12}")
    for t, r in reports.items():
        for name, cap in (("fa10", "<=10%"), ("fa05", "<=5%"), ("fa20", "<=20%")):
            b = r["operating_points"].get(name)
            if not b:
                continue
            ci, zz = b["det_ci"], b.get("det_vs_fa") or {}
            print(f"  {t:<6} {cap:<6} {b['threshold']:>6.3f} {pct(b['det']):>7} "
                  f"[{ci['lo']:.3f},{ci['hi']:.3f}]".ljust(48)
                  + f"{pct(b['fa']):>7} {b['det_k']}/{b['n_err']} err, "
                    f"{b['fa_k']}/{b['n_clean']} clean"
                  + (f"  det-vs-FA z={zz['z']} p={zz['p']}" if zz else ""))

    print("\n(b) detection by class and residual level, at each tier's FA<=10% threshold")
    print(f"  {'tier':<6} {'omit-complete':>14} {'omit-partial':>14} {'partial-weak':>14} "
          f"{'partial-strong':>15} {'critical':>10}")
    for t, r in reports.items():
        b = r["operating_points"].get("fa10")
        if not b:
            continue
        g = lambda d, k: (f"{pct(d[k]['p'])} ({d[k]['k']}/{d[k]['n']})" if d.get(k) else "-")
        print(f"  {t:<6} {g(b['by_class'],'omit-complete'):>14} "
              f"{g(b['by_class'],'omit-partial'):>14} {g(b['by_residual'],'partial-weak'):>14} "
              f"{g(b['by_residual'],'partial-strong'):>15} "
              f"{g(b['by_severity'],'critical'):>10}")

    print("\n(c) PAIRED tie-adjusted discrimination (chance 0.500)  [grid best: 0.634]")
    print(f"  {'tier':<6} {'omissions':>10} {'complete':>10} {'partial':>10} {'p-weak':>10} "
          f"{'p-strong':>10} {'critical':>10} {'commissions':>12}")
    for t, r in reports.items():
        p = r["paired"]
        g = lambda d: ("   -  " if not d else f"{d['paired']:.3f} ")
        print(f"  {t:<6} {g(p['omissions']):>10} {g(p['by_residual'].get('complete')):>10} "
              f"{g(p['by_class'].get('omit-partial')):>10} "
              f"{g(p['by_residual'].get('partial-weak')):>10} "
              f"{g(p['by_residual'].get('partial-strong')):>10} "
              f"{g(p['by_severity'].get('critical')):>10} {g(p['commissions']):>12}")
    for t, r in reports.items():
        st, p = r["paired"]["omissions_sign_test"], r["paired"]["omissions"]
        print(f"    {t}: {p['strict_wins']} wins / {p['ties']} ties / {p['losses']} losses, "
              f"exact sign test on the {st['n_informative']} informative pairs p={st['p']:.2e}")
    gr = GRID["paired_by_residual"]
    print(f"  {'grid':<6} {GRID['paired_omissions']['FC-score-k8']:>10.3f} "
          f"{gr['complete']:>10.3f} {'-':>10} {gr['partial-weak']:>10.3f} "
          f"{gr['partial-strong']:>10.3f} "
          f"{GRID['paired_by_severity']['critical']:>10.3f} "
          f"{GRID['paired_commissions']:>12.3f}")

    print("\n(d) commissions - the calibration side. A coverage score cannot see a fabrication:")
    print(f"  {'tier':<6} {'add (paired)':>14} {'change (paired)':>16} {'both':>8}   "
          f"[grid 0.939 on both]")
    for t, r in reports.items():
        p = r["paired"]["by_commission_type"]
        g = lambda d: ("  -  " if not d else f"{d['paired']:.3f} (n={d['n']})")
        print(f"  {t:<6} {g(p.get('add')):>14} {g(p.get('change')):>16} "
              f"{r['paired']['commissions']['paired'] if r['paired']['commissions'] else '-':>8}")

    print("\n    post-hoc re-scorings of the SAME purchased verdicts (attribution diagnostics,")
    print("    not a menu - picking the best here would be tuning on the smoke):")
    print(f"    {'tier':<6} {'variant':<12} {'paired':>8} {'det@FA<=10%':>12} {'FA':>7}")
    for t, r in reports.items():
        for m, vv in r["variants"].items():
            if vv:
                print(f"    {t:<6} {m:<12} {num(vv['paired']):>8} {pct(vv['det_at_fa10']):>12} "
                      f"{pct(vv['fa']):>7}")

    print("\n(e) measured cost")
    c = costs
    print(f"  extraction {c['extraction']['calls']} calls, ${c['extraction']['usd']} "
          f"(${c['extraction']['per_consultation']}/consultation, shared by both tiers)")
    print(f"  audit      {c['audit']['calls']} calls, ${c['audit']['usd']} "
          f"(${c['audit']['per_consultation']}/consultation, B3 only)")
    for t, d in c["tiers"].items():
        print(f"  {t}: ${d['marginal_usd_per_note']}/note marginal, "
              f"${d['loaded_usd_per_note']}/note fully loaded "
              f"({d['calls_per_note']} calls/note, {d['notes_per_consultation']} notes per "
              f"consultation in this subset)")
    print(f"\n  full artifact -> {args.out}")


if __name__ == "__main__":
    main()
