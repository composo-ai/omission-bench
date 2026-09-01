#!/usr/bin/env python3
"""extract_data.py - build every number the paper's five floats display, from the record
stores, and refuse to write anything unless the recomputation reproduces the figures the
two papers report.

Nothing here is hand-typed. Every value in figures/*.json comes out of:

  results/w2-ablation/_state/grid-main2.jsonl      the 2x2x2 grid
  results/w2-strong/_state/arms-main.jsonl         engineered + GEPA winner
  results/w2-v14/_state/arms-main.jsonl            the deployed judge's three wordings
  results/w2-baselines/_state/arms-main.jsonl      RAGAS, G-Eval, checklist
  results/w2-pipeline/_state/confirm-B{2,3}.jsonl  the pipeline confirmation
  results/w2-pipeline/_cache/{facts,audit}_*.json  the extractor + audit stage
  master/dataset_v2.json, master/arms_confirm_subset.json, master/fact_sites.json
  master/sitting_results.json                      the clinician sitting
  authored_scenarios.json                          the authored transcripts

Conventions:
  paired      tie-adjusted P(errored note scores strictly below its OWN clean twin) + half
              the tie mass, computed per replicate then averaged. Chance 0.500.
  absolute    flag rate at the arm's own flag rule, pooled over replicates, only ever
              reported beside the same arm's false-alarm rate on clean notes.
  swept       detection at the most sensitive threshold whose pooled record-level
              false-alarm rate stays within the cap.

What this file is doing in this release. The stores listed above are the study's own
run and construction artifacts and are not part of it: the released judge records are the
dataset repository's `judgements/judges/`, and the released pair set is its
`pairs/dataset_v2.json`. `main()` therefore cannot be run from a clone, and nothing in this
repository calls it. What five analysis modules here do use is everything above `main()` -
`wilson`, `two_prop_z`, `paired_over_reps`, `sweep`, `residual_level_of`, `arm_block` and
the two error-class predicates - which they load from this file by path so that the study
has one definition of each measure and it cannot drift between analyses.

Usage: python3 record/figures/extract_data.py
"""
import json, math, os, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # the repository root: this file sits in record/figures/
OUT = HERE

STRENGTH_TO_LEVEL = {"explicit": "partial-strong", "paraphrase": "partial-strong",
                     "partial": "partial-weak", "fragment": "partial-weak"}


# ---------------------------------------------------------------- primitives
def wilson(k, n, z=1.959963984540054):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": k, "n": n, "p": round(p, 6),
            "lo": round(max(0.0, c - h), 6), "hi": round(min(1.0, c + h), 6)}


def two_prop_z(k1, n1, k2, n2):
    """Detection versus the same arm's false alarms at one operating point."""
    if not n1 or not n2:
        return None
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if not se:
        return None
    z = (k1 / n1 - k2 / n2) / se
    return round(z, 3)


def paired_once(rows):
    """One replicate's tie-adjusted paired discrimination over (err, clean) score pairs."""
    rows = [(e, c) for e, c in rows if e is not None and c is not None]
    if not rows:
        return None
    s = sum(1.0 if e < c else 0.5 if e == c else 0.0 for e, c in rows)
    return s / len(rows), len(rows)


def paired_over_reps(recs, clean_by, sel, score):
    """Paired per replicate then averaged - the study's paired-discrimination convention.

    Parse failures: a pair is dropped when either side is unparseable. This reproduces
    every reference arm exactly; RAGAS, the only arm with parse failures, lands on 0.817
    on its completeness component, which is the figure the papers print, 0.005 above the
    0.812 an earlier read-out of the same component recorded."""
    per_rep, n_total = [], 0
    for rep in sorted({r["replicate"] for r in recs}):
        rows = []
        for r in recs:
            if r["replicate"] != rep or r.get("note_role") == "clean" or not sel(r):
                continue
            c = clean_by.get((rep, r.get("clean_key")))
            rows.append((score(r), c))
        got = paired_once(rows)
        if got:
            per_rep.append(got[0])
            n_total = max(n_total, got[1])
    if not per_rep:
        return None
    return {"paired": round(sum(per_rep) / len(per_rep), 4),
            "per_replicate": [round(v, 4) for v in per_rep], "n_pairs": n_total}


def sweep(err_scores, clean_scores):
    """Every operating point of `flag iff aggregate < t`, over all real t.

    For each observed value v two operating points exist: t = v (records AT v not
    flagged) and t just above v (records AT v flagged). The original implementation
    evaluated only the strict points plus one inclusive point past the maximum, so any
    best cut that needed to include records sitting exactly on an interior threshold
    was invisible - that is how the checklist arm's swept cell read 15.4% while
    the published 16.0% (94/586 at the same 19/224 false alarms) was the true best
    point. Fixed on 24 August 2026: both points are evaluated per
    value; on every other arm the strict point was already optimal, so only the
    checklist cell moves (verified against the regenerated _recomputed.json)."""
    cand = sorted({round(v, 9) for v in list(err_scores) + list(clean_scores)})
    thresholds = []
    for v in cand:
        thresholds += [v, v + 1e-6]
    out = []
    for t in thresholds or [1.0]:
        dk = sum(1 for v in err_scores if v < t)
        fk = sum(1 for v in clean_scores if v < t)
        out.append({"threshold": round(t, 6), "det": dk / len(err_scores) if err_scores else None,
                    "det_k": dk, "n_err": len(err_scores),
                    "fa": fk / len(clean_scores) if clean_scores else None,
                    "fa_k": fk, "n_clean": len(clean_scores)})
    return out


def best_at_fa(curve, cap):
    ok = [p for p in curve if p["fa"] is not None and p["fa"] <= cap + 1e-12]
    if not ok:
        return None
    return max(ok, key=lambda p: (p["det"], -p["fa"], p["threshold"]))


def residual_level_of(rec):
    lvl = (rec.get("residual_level") or "").strip().lower()
    if lvl == "partial":
        return STRENGTH_TO_LEVEL.get((rec.get("residual_strength") or "").strip().lower(), "partial")
    return lvl or ("complete" if rec.get("pair_type") == "omit" else rec.get("pair_type"))


def load(path):
    recs = []
    with open(os.path.join(ROOT, path)) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def clean_index(recs, score=lambda r: r.get("aggregate")):
    return {(r["replicate"], r["note_key"]): (None if r.get("parse_failure") else score(r))
            for r in recs if r.get("note_role") == "clean"}


IS_OMIT = lambda r: r.get("pair_type") == "omit"
IS_COMM = lambda r: r.get("pair_type") in ("add", "change")


def arm_block(recs, score=lambda r: r.get("aggregate"), flagged=lambda r: r.get("flagged"),
              sel=lambda r: True):
    """paired (omissions, commissions, by residual, by severity), absolute det/FA, swept."""
    recs = [r for r in recs if sel(r)]
    sc = lambda r: (None if r.get("parse_failure") else score(r))
    for r in recs:
        r["_s"] = sc(r)
    ci = {(r["replicate"], r["note_key"]): r["_s"] for r in recs if r.get("note_role") == "clean"}
    pr = lambda f: paired_over_reps(recs, ci, f, sc)
    err = [r for r in recs if r.get("note_role") != "clean"]
    cln = [r for r in recs if r.get("note_role") == "clean"]
    omis = [r for r in err if IS_OMIT(r)]
    comm = [r for r in err if IS_COMM(r)]

    out = {"paired_omissions": pr(IS_OMIT), "paired_commissions": pr(IS_COMM),
           "paired_by_residual": {lv: pr(lambda r, lv=lv: IS_OMIT(r) and residual_level_of(r) == lv)
                                  for lv in ("complete", "partial-weak", "partial-strong")},
           "paired_by_severity": {s: pr(lambda r, s=s: IS_OMIT(r) and r.get("severity") == s)
                                  for s in ("critical", "supporting", "peripheral")},
           "n_records": {"omission": len(omis), "commission": len(comm), "clean": len(cln)}}

    if flagged is not None:
        # parse failure = not flagged, on the full denominator (580/586, 221/224)
        dk = sum(1 for r in omis if flagged(r) and not r.get("parse_failure"))
        fk = sum(1 for r in cln if flagged(r) and not r.get("parse_failure"))
        out["absolute"] = {"det": wilson(dk, len(omis)), "fa": wilson(fk, len(cln)),
                           "z": two_prop_z(dk, len(omis), fk, len(cln))}
    es = [r["_s"] for r in omis if r["_s"] is not None]
    cs = [r["_s"] for r in cln if r["_s"] is not None]
    if es and cs and len({round(v, 9) for v in es + cs}) > 2:
        b = best_at_fa(sweep(es, cs), 0.10)
        out["swept_fa10"] = ({"det": round(b["det"], 4), "fa": round(b["fa"], 4),
                              "det_k": b["det_k"], "n_err": b["n_err"],
                              "fa_k": b["fa_k"], "n_clean": b["n_clean"],
                              "threshold": b["threshold"]} if b else None)
    else:
        out["swept_fa10"] = None   # binary / 5-point aggregates: not expressible
    return out


def check(label, got, want, tol, problems):
    if got is None or abs(got - want) > tol:
        problems.append(f"{label}: recomputed {got!r}, published value {want} (tol {tol})")
    return got


# ---------------------------------------------------------------- the arms
def main():
    problems = []
    data = {}

    # ---- the 2x2x2 grid, 3 replicates, 495 eval pairs --------------------------
    grid = load("results/w2-ablation/_state/grid-main2.jsonl")
    grid_cells = {}
    for cell in sorted({r["cell"] for r in grid}):
        grid_cells[cell] = arm_block([r for r in grid if r["cell"] == cell])
    data["grid"] = grid_cells

    for cell, (o, c) in {"F-bin-k1": (0.500, 0.875), "F-bin-k8": (0.509, 0.904),
                         "FC-bin-k1": (0.539, 0.792), "FC-bin-k8": (0.591, 0.869),
                         "F-score-k1": (0.549, 0.911), "F-score-k8": (0.582, 0.944),
                         "FC-score-k1": (0.585, 0.895), "FC-score-k8": (0.634, 0.939)}.items():
        check(f"grid {cell} omissions", grid_cells[cell]["paired_omissions"]["paired"], o, 0.0015, problems)
        check(f"grid {cell} commissions", grid_cells[cell]["paired_commissions"]["paired"], c, 0.0015, problems)
    best = grid_cells["FC-score-k8"]
    for lv, want in (("complete", 0.690), ("partial-weak", 0.607), ("partial-strong", 0.526)):
        check(f"grid FC-score-k8 by residual {lv}",
              best["paired_by_residual"][lv]["paired"], want, 0.0015, problems)
    for sv, want in (("critical", 0.683), ("supporting", 0.586), ("peripheral", 0.568)):
        check(f"grid FC-score-k8 by severity {sv}",
              best["paired_by_severity"][sv]["paired"], want, 0.0015, problems)
    check("grid FC-score-k8 det", best["absolute"]["det"]["p"], 0.083, 0.001, problems)
    check("grid FC-score-k8 FA", best["absolute"]["fa"]["p"], 0.065, 0.001, problems)

    # ---- the reference arms, 2 replicates, same substrate ----------------------
    arms = (load("results/w2-strong/_state/arms-main.jsonl")
            + load("results/w2-v14/_state/arms-main.jsonl")
            + load("results/w2-baselines/_state/arms-main.jsonl"))
    arm_blocks = {}
    for a in sorted({r["arm"] for r in arms}):
        arm_blocks[a] = arm_block([r for r in arms if r["arm"] == a])
    # RAGAS's two components, scored separately
    rag = [r for r in arms if r["arm"] == "ragas"]
    for comp in ("completeness", "faithfulness"):
        arm_blocks[f"ragas-{comp}"] = arm_block(
            rag, score=lambda r, c=comp: (r.get("detail") or {}).get(c), flagged=None)
    data["arms"] = arm_blocks

    for a, (o, c) in {"engineered-completeness": (0.639, 0.754), "gepa-optimized": (0.549, 0.861),
                      "v14-asis": (0.518, 0.886), "v14-noexcl": (0.505, 0.890),
                      "v14-incl": (0.556, 0.887), "geval": (0.568, 0.890),
                      "checklist": (0.647, 0.744)}.items():
        check(f"arm {a} omissions", arm_blocks[a]["paired_omissions"]["paired"], o, 0.0015, problems)
        check(f"arm {a} commissions", arm_blocks[a]["paired_commissions"]["paired"], c, 0.0015, problems)
    # RAGAS is the only arm carrying parse failures; on the drop convention its completeness
    # component reproduces the published 0.817 rather than the 0.812 of the earlier read-out.
    check("ragas completeness paired", arm_blocks["ragas-completeness"]["paired_omissions"]["paired"],
          0.817, 0.0015, problems)
    check("ragas faithfulness paired", arm_blocks["ragas-faithfulness"]["paired_omissions"]["paired"],
          0.549, 0.0015, problems)
    check("ragas aggregate omissions", arm_blocks["ragas"]["paired_omissions"]["paired"],
          0.809, 0.0015, problems)
    check("ragas aggregate commissions", arm_blocks["ragas"]["paired_commissions"]["paired"],
          0.712, 0.0015, problems)
    check("ragas absolute det", arm_blocks["ragas"]["absolute"]["det"]["p"], 0.990, 0.002, problems)
    check("ragas absolute FA", arm_blocks["ragas"]["absolute"]["fa"]["p"], 0.987, 0.002, problems)
    check("ragas swept det", arm_blocks["ragas"]["swept_fa10"]["det"], 0.209, 0.0015, problems)
    # The published 16.0% @ 8.5% is 94/586 at 19/224 - a cut INCLUSIVE of the four
    # errored records sitting exactly on the threshold. The strict-< sweep this script
    # originally ran could not express that point and read 15.4%, which both papers'
    # T6s then hand-corrected; the sweep is inclusive-aware since 2026-08-24 and the
    # store now reproduces the published figure at the source.
    check("checklist swept det", arm_blocks["checklist"]["swept_fa10"]["det"], 0.1604, 0.0015, problems)
    check("engineered det", arm_blocks["engineered-completeness"]["absolute"]["det"]["p"],
          0.0836, 0.001, problems)
    check("engineered FA", arm_blocks["engineered-completeness"]["absolute"]["fa"]["p"],
          0.0268, 0.001, problems)

    # the engineered arm's own free-text critical-omission rule
    eng = [r for r in arms if r["arm"] == "engineered-completeness"]
    crit_free = lambda r: any((o or {}).get("severity") == "critical"
                              for o in (r.get("omissions_reported") or []))
    data["arms"]["engineered-freetext-critical"] = arm_block(eng, flagged=crit_free)
    check("free-text critical det",
          data["arms"]["engineered-freetext-critical"]["absolute"]["det"]["p"], 0.0956, 0.002, problems)
    check("free-text critical FA",
          data["arms"]["engineered-freetext-critical"]["absolute"]["fa"]["p"], 0.0089, 0.002, problems)

    # ---- the pipeline confirmation, 151 held-out pairs -------------------------
    # Since the 2026-08-24 evaluation-set extension confirm-B3.jsonl holds the FULL
    # 495-pair evaluation set (607 notes); the confirmation figures are the 151-pair
    # held-out subset, so B3 is restricted to it here and the full-set records feed the
    # separate `pipeline_evalset` block below. B2 was never extended (198 records).
    b2 = load("results/w2-pipeline/_state/confirm-B2.jsonl")
    b3_full = load("results/w2-pipeline/_state/confirm-B3.jsonl")
    confirm = json.load(open(os.path.join(ROOT, "master/arms_confirm_subset.json")))
    conf_pairs = {p["pair_id"] for p in confirm["pairs"]}
    conf_cons = {(p["stratum"], p["id"]) for p in confirm["pairs"]}
    conf_clean = {r["clean_key"] for r in b2 if r.get("clean_key")}
    sel_conf = lambda r: (r.get("pair_id") in conf_pairs) or (r["note_key"] in conf_clean)
    b3 = [r for r in b3_full if sel_conf(r)]
    data["pipeline"] = {"B2": arm_block(b2, flagged=None), "B3": arm_block(b3, flagged=None)}
    check("pipeline B2 paired", data["pipeline"]["B2"]["paired_omissions"]["paired"], 0.801, 0.0015, problems)
    check("pipeline B3 paired", data["pipeline"]["B3"]["paired_omissions"]["paired"], 0.752, 0.0015, problems)
    check("pipeline B2 complete", data["pipeline"]["B2"]["paired_by_residual"]["complete"]["paired"],
          0.936, 0.002, problems)
    check("pipeline B2 swept det", data["pipeline"]["B2"]["swept_fa10"]["det"], 0.160, 0.002, problems)
    check("pipeline B3 swept det", data["pipeline"]["B3"]["swept_fa10"]["det"], 0.183, 0.002, problems)

    # the per-fact predicate: flag iff any audit-critical fact is verdicted absent
    def predicate(r):
        return any(m.get("severity") == "critical" and m.get("verdict") == "absent"
                   for m in ((r.get("detail") or {}).get("missing_facts") or []))
    pred = arm_block(b3, flagged=predicate)
    omis3 = [r for r in b3 if r.get("note_role") != "clean" and IS_OMIT(r)]
    pred["by_residual_detection"] = {}
    for lv in ("complete", "partial-weak", "partial-strong"):
        sub = [r for r in omis3 if residual_level_of(r) == lv]
        pred["by_residual_detection"][lv] = wilson(sum(1 for r in sub if predicate(r)), len(sub))
    crit = [r for r in omis3 if r.get("severity") == "critical"]
    pred["critical_severity_pairs"] = wilson(sum(1 for r in crit if predicate(r)), len(crit))
    data["pipeline"]["B3-predicate"] = pred
    check("predicate det", pred["absolute"]["det"]["p"], 0.206, 0.002, problems)
    check("predicate FA", pred["absolute"]["fa"]["p"], 0.0213, 0.002, problems)
    check("predicate complete", pred["by_residual_detection"]["complete"]["p"], 0.339, 0.002, problems)
    check("predicate partial-strong", pred["by_residual_detection"]["partial-strong"]["p"],
          0.0, 1e-9, problems)
    check("predicate on critical-severity pairs", pred["critical_severity_pairs"]["p"], 0.328, 0.002, problems)

    # the aggregate-score rival at the predicate's own 2.1% false-alarm level
    cln3 = [r["aggregate"] for r in b3 if r.get("note_role") == "clean"]
    es3 = [r["aggregate"] for r in omis3]
    b = best_at_fa(sweep(es3, cln3), pred["absolute"]["fa"]["p"] + 1e-9)
    data["pipeline"]["B3-score-at-predicate-FA"] = {
        "det": round(b["det"], 4), "det_k": b["det_k"], "n_err": b["n_err"],
        "fa": round(b["fa"], 4), "fa_k": b["fa_k"], "n_clean": b["n_clean"],
        "threshold": b["threshold"],
        "ratio_predicate_over_score": round(pred["absolute"]["det"]["p"] / b["det"], 3) if b["det"] else None}
    # An earlier draft said 6.2% here (hence "3.3x"); the store's best aggregate-score
    # threshold at the predicate's own 1/47 false alarms detects 10/131 = 7.6%, so the
    # multiple is 2.7x.
    check("score rival at 2.1% FA", data["pipeline"]["B3-score-at-predicate-FA"]["det"],
          0.0763, 0.002, problems)

    # ---- the pipeline replicates (2026-08-25) ----------------------------------
    # Replicates 2 and 3 live in their own stores (confirmr{2,3}-B{2,3}.jsonl); the
    # banked read-out is results/w2-pipeline-replicates/analysis.json. This block
    # reads that artifact and refuses to write unless it reproduces the published
    # replicate means and rule rates, so the floats' three-run means cannot drift
    # from the banked values.
    reps = json.load(open(os.path.join(ROOT, "results/w2-pipeline-replicates/analysis.json")))
    rp = reps["paired_discrimination"]
    data["pipeline_replicates"] = {
        "B2": {"omissions": rp["B2"]["omissions"], "commissions": rp["B2"]["commissions"]},
        "B3": {"omissions": rp["B3"]["omissions"], "commissions": rp["B3"]["commissions"]},
        "critical_rule": reps["critical_rule"]}
    check("replicates B2 omissions mean", rp["B2"]["omissions"]["mean"], 0.786, 0.0015, problems)
    check("replicates B3 omissions mean", rp["B3"]["omissions"]["mean"], 0.762, 0.0015, problems)
    check("replicates B2 omissions rep1", rp["B2"]["omissions"]["per_replicate"][0], 0.8015, 0.0015, problems)
    check("replicates B2 omissions range", rp["B2"]["omissions"]["range"], 0.0305, 0.0015, problems)
    check("replicates B2 commissions mean", rp["B2"]["commissions"]["mean"], 0.650, 0.0015, problems)
    check("replicates B3 commissions mean", rp["B3"]["commissions"]["mean"], 0.617, 0.0015, problems)
    for i, want in enumerate((0.206, 0.229, 0.214)):
        check(f"critical rule det rep{i + 1}",
              reps["critical_rule"]["per_replicate"][str(i + 1)]["detection"]["p"], want, 0.002, problems)
        check(f"critical rule FA rep{i + 1}",
              reps["critical_rule"]["per_replicate"][str(i + 1)]["false_alarms"]["p"], 0.0213, 0.002, problems)
    check("critical rule majority det", reps["critical_rule"]["majority_of_three"]["detection"]["p"],
          0.214, 0.002, problems)

    # ---- the same tier and rule on the FULL 495-pair evaluation set -----------------
    # (single replicate at the confirmation settings; the 2026-08-24 extension)
    ev = {"B3": arm_block(b3_full, flagged=None),
          "B3-predicate": arm_block(b3_full, flagged=predicate)}
    omis_ev = [r for r in b3_full if r.get("note_role") != "clean" and IS_OMIT(r)]
    ev["B3-predicate"]["by_residual_detection"] = {}
    for lv in ("complete", "partial-weak", "partial-strong"):
        sub = [r for r in omis_ev if residual_level_of(r) == lv]
        ev["B3-predicate"]["by_residual_detection"][lv] = wilson(
            sum(1 for r in sub if predicate(r)), len(sub))
    crit_ev = [r for r in omis_ev if r.get("severity") == "critical"]
    ev["B3-predicate"]["critical_severity_pairs"] = wilson(
        sum(1 for r in crit_ev if predicate(r)), len(crit_ev))
    data["pipeline_evalset"] = ev
    check("predicate det, eval set", ev["B3-predicate"]["absolute"]["det"]["p"],
          0.2457, 0.002, problems)
    check("predicate FA, eval set", ev["B3-predicate"]["absolute"]["fa"]["p"],
          0.0268, 0.002, problems)
    check("predicate critical-severity catch, eval set", ev["B3-predicate"]["critical_severity_pairs"]["p"],
          0.3974, 0.002, problems)

    # ---- matched substrate, the 151 confirmation pairs -------------------------
    # (conf_pairs / conf_clean / sel_conf defined with the pipeline load above)
    data["matched_substrate_151"] = {
        "ragas_completeness": arm_block(rag, score=lambda r: (r.get("detail") or {}).get("completeness"),
                                        flagged=None, sel=sel_conf)["paired_omissions"],
        "ragas_faithfulness": arm_block(rag, score=lambda r: (r.get("detail") or {}).get("faithfulness"),
                                        flagged=None, sel=sel_conf)["paired_omissions"],
        "pipeline_B2": data["pipeline"]["B2"]["paired_omissions"],
        "grid_FC_score_k8_per_replicate": None}
    gsub = [r for r in grid if r["cell"] == "FC-score-k8" and sel_conf(r)]
    gci = clean_index(gsub)
    per_rep = []
    for rep in sorted({r["replicate"] for r in gsub}):
        rows = [(r.get("aggregate"), gci.get((rep, r.get("clean_key"))))
                for r in gsub if r["replicate"] == rep and r.get("note_role") != "clean" and IS_OMIT(r)]
        got = paired_once(rows)
        if got:
            per_rep.append(round(got[0], 4))
    data["matched_substrate_151"]["grid_FC_score_k8_per_replicate"] = per_rep
    check("ragas completeness on 151",
          data["matched_substrate_151"]["ragas_completeness"]["paired"], 0.801, 0.002, problems)
    check("ragas faithfulness on 151",
          data["matched_substrate_151"]["ragas_faithfulness"]["paired"], 0.547, 0.002, problems)
    check("grid span low on 151", min(per_rep), 0.531, 0.002, problems)
    check("grid span high on 151", max(per_rep), 0.649, 0.002, problems)

    # ---- measured cost per note, straight off the receipts ---------------------
    def per_note_cost(recs, keyf):
        agg = collections.defaultdict(lambda: [0.0, 0, 0])
        for r in recs:
            a = agg[keyf(r)]
            t = r.get("totals") or {}
            a[0] += t.get("cost_usd", 0.0)
            a[1] += t.get("calls", 0)
            a[2] += 1
        return {k: {"usd_per_note": round(v[0] / v[2], 4), "calls_per_note": round(v[1] / v[2], 2),
                    "n_note_judgements": v[2]} for k, v in agg.items()}

    # Stage receipts restricted to the 47 confirmation consultations: the 2026-08-24
    # evaluation-set extension added 43 audit rows for consultations outside the subset,
    # and the published per-consultation prices are the confirmation run's (37 rows each).
    stages = collections.defaultdict(lambda: [0.0, 0])
    for r in load("results/w2-pipeline/_state/confirm-stages.jsonl"):
        if (r.get("stratum"), r.get("consultation")) not in conf_cons:
            continue
        stages[r["stage"]][0] += (r.get("totals") or {}).get("cost_usd", 0.0)
        stages[r["stage"]][1] += 1
    cost = {"grid": per_note_cost(grid, lambda r: r["cell"]),
            "arms": per_note_cost(arms, lambda r: r["arm"]),
            "pipeline_marginal": {"B2": per_note_cost(b2, lambda r: "B2")["B2"],
                                  "B3": per_note_cost(b3, lambda r: "B3")["B3"]},
            "pipeline_stages_per_consultation": {
                k: {"usd": round(v[0] / v[1], 4), "n_consultations": v[1]} for k, v in stages.items()}}
    # production framing: one note per consultation, so a per-consultation stage is not amortised
    ex = cost["pipeline_stages_per_consultation"]["extract"]["usd"]
    au = cost["pipeline_stages_per_consultation"]["audit"]["usd"]
    cost["production_usd_per_note"] = {
        "B2": round(ex + cost["pipeline_marginal"]["B2"]["usd_per_note"], 4),
        "B3": round(ex + au + cost["pipeline_marginal"]["B3"]["usd_per_note"], 4)}
    data["cost"] = cost
    check("cost: extract per consultation", ex, 0.0868, 0.0002, problems)
    check("cost: audit per consultation", au, 0.3194, 0.0002, problems)
    check("cost: B2 marginal", cost["pipeline_marginal"]["B2"]["usd_per_note"], 0.0077, 0.0002, problems)
    check("cost: B3 marginal", cost["pipeline_marginal"]["B3"]["usd_per_note"], 0.0440, 0.0002, problems)
    check("cost: RAGAS per note", cost["arms"]["ragas"]["usd_per_note"], 0.0305, 0.0002, problems)
    check("cost ladder: B2 production", cost["production_usd_per_note"]["B2"], 0.094, 0.001, problems)
    check("cost ladder: B3 production", cost["production_usd_per_note"]["B3"], 0.45, 0.001, problems)
    check("cost ladder: k=8 judge", cost["grid"]["FC-score-k8"]["usd_per_note"], 0.036, 0.001, problems)

    # ---- the monolithic judge on the pipeline's own substrate (F4 panel B) -----
    gconf = [r for r in grid if r["cell"] == "FC-score-k8" and sel_conf(r)]
    own = []
    for rep in sorted({r["replicate"] for r in gconf}):
        sub = [r for r in gconf if r["replicate"] == rep]
        om = [r for r in sub if r.get("note_role") != "clean" and IS_OMIT(r)]
        cl = [r for r in sub if r.get("note_role") == "clean"]
        own.append({"replicate": rep,
                    "det": wilson(sum(1 for r in om if r.get("flagged")), len(om)),
                    "fa": wilson(sum(1 for r in cl if r.get("flagged")), len(cl))})
    data["grid_best_on_confirmation_substrate"] = own

    if problems:
        print("RECOMPUTATION DOES NOT REPRODUCE THE PUBLISHED FIGURES - refusing to write:", file=sys.stderr)
        for p in problems:
            print("  -", p, file=sys.stderr)
        sys.exit(1)

    with open(os.path.join(OUT, "_recomputed.json"), "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    print("OK - every checked figure reproduces the published values. wrote record/figures/_recomputed.json")
    return data


if __name__ == "__main__":
    main()
