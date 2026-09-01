#!/usr/bin/env python3
"""Read-out for the extended evaluation set.

Joins the pre-existing stores with the extension records - nothing is re-bought, nothing
overwritten - and answers the four questions the runs were commissioned for:

  1. Head-to-head at the consultation level, now at 112 consultations: the three-stage
     pipeline's per-fact critical rule vs the powered GEPA winner's "flag below 10"
     majority rule, McNemar on discordant consultations - the consultation is the unit.
  2. Calibration: do clean notes still park at exactly 10 for the powered call on 112
     twins (per replicate + majority), and what the false-alarm intervals become.
  3. Detection/paired for both methods on the full 495-pair evaluation set.
  4. Family transfer of the single call: the same powered prompt on the Gemini judge
     (151-pair subset), parse health, calibration, detection at its own rule.

Stores read (never written):
  results/w2-pipeline/_state/confirm-B3.jsonl   B3 records, 495 pairs + 112 twins
  results/w2-power/_state/power.jsonl           power-gepa-high r1-r3, same substrate
  results/w2-power/_state/power-gemini.jsonl    power-gepa-high on judge-gemini, r1-r3
  master/arms_confirm_subset.json               the 151-pair held-out subset (for splits)

Output: results/w2-evalfull/analysis.json (+ SUMMARY.md), written fresh each run.
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path

import collections
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repository root
RESULTS = os.path.join(HERE, "results")
OUT_DIR = os.path.join(RESULTS, "w2-evalfull")

FLAG_BELOW = 10.0        # the powered single call's rule: flag anything below 10
SEED = 20260824


def read_store(path, pred=lambda r: True):
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if pred(r):
                out.append(r)
    return out


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, (c - h) / d, (c + h) / d)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_disc(errored, clean_score_by_key, score_of):
    wins = ties = tot = 0
    for r in errored:
        c = clean_score_by_key.get(r["clean_key"])
        if c is None:
            continue
        tot += 1
        s = score_of(r)
        if s < c:
            wins += 1
        elif s == c:
            ties += 1
    return (wins + 0.5 * ties) / tot if tot else float("nan"), tot


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sp = os.path.join(HERE, "master/arms_confirm_subset.json")
    subset_pairs = {p["pair_id"] for p in json.load(open(sp))["pairs"]} \
        if os.path.exists(sp) else set()

    # ---------------- B3: per-fact critical rule per note --------------------------
    b3 = read_store(os.path.join(RESULTS, "w2-pipeline/_state/confirm-B3.jsonl"),
                    lambda r: r.get("tier") == "B3" or r.get("cell") == "B3" or True)
    # rule: note flagged iff any fact graded critical is verdicted absent
    # (per-fact rows live in detail.missing_facts - same read as w2_pipeline_analyze:180)
    def b3_flag(r):
        for f in (r.get("detail") or {}).get("missing_facts") or []:
            if f.get("severity") == "critical" and f.get("verdict") == "absent":
                return True
        return False

    b3_by_note = {}
    for r in b3:
        if r.get("note_key"):
            b3_by_note[r["note_key"]] = r     # one replicate; last write wins

    # ---------------- powered GEPA: majority-of-3 flag per note --------------------
    gp = read_store(os.path.join(RESULTS, "w2-power/_state/power.jsonl"),
                    lambda r: r.get("arm") == "power-gepa-high")
    gp_scores = collections.defaultdict(dict)      # note_key -> rep -> score
    gp_meta = {}
    for r in gp:
        gp_scores[r["note_key"]][r["replicate"]] = r.get("aggregate")
        gp_meta[r["note_key"]] = r

    def gp_flag_majority(nk):
        reps = gp_scores.get(nk, {})
        votes = [1 if (s is not None and s < FLAG_BELOW) else 0 for s in reps.values()]
        return sum(votes) >= 2, len(votes)

    # ---------------- note roster from B3 (role, class, consultation) --------------
    # the study's residual convention throughout: `partial` splits by strength into
    # partial-weak / partial-strong (same map as record/figures/extract_data.py)
    strength_to_level = {"explicit": "partial-strong", "paraphrase": "partial-strong",
                         "partial": "partial-weak", "fragment": "partial-weak"}

    def residual_of(r):
        lvl = (r.get("residual_level") or "").strip().lower()
        if lvl == "partial":
            return strength_to_level.get(
                (r.get("residual_strength") or "").strip().lower(), "partial")
        return lvl or None

    roster = {}
    for r in b3:
        nk = r.get("note_key")
        if nk:
            roster[nk] = {
                "role": r.get("note_role"), "pair_class": r.get("pair_class"),
                "consultation": f"{r.get('stratum')}|{r.get('consultation')}",
                "pair_id": r.get("pair_id"), "clean_key": r.get("clean_key"),
                "severity": r.get("severity"), "residual": residual_of(r),
            }

    omission_notes = [nk for nk, m in roster.items()
                      if m["role"] == "errored" and str(m["pair_class"]).startswith("omit")]
    clean_notes = [nk for nk, m in roster.items() if m["role"] == "clean"]
    out = {"seed": SEED, "counts": {
        "b3_records": len(b3), "gepa_records": len(gp),
        "omission_notes": len(omission_notes), "clean_notes": len(clean_notes)}}

    # ---------------- 1+3: detection and FA, both methods, full set ---------------
    for name, flag_of in (("b3_rule", lambda nk: b3_flag(b3_by_note[nk])),
                          ("gepa_majority", lambda nk: gp_flag_majority(nk)[0])):
        det_k = sum(1 for nk in omission_notes if flag_of(nk))
        fa_k = sum(1 for nk in clean_notes if flag_of(nk))
        p, lo, hi = wilson(det_k, len(omission_notes))
        fp, flo, fhi = wilson(fa_k, len(clean_notes))
        crit = [nk for nk in omission_notes if roster[nk].get("severity") == "critical"]
        crit_k = sum(1 for nk in crit if flag_of(nk))
        out[name] = {
            "det": {"k": det_k, "n": len(omission_notes), "p": round(p, 4),
                    "wilson": [round(lo, 4), round(hi, 4)]},
            "fa": {"k": fa_k, "n": len(clean_notes), "p": round(fp, 4),
                   "wilson": [round(flo, 4), round(fhi, 4)]},
            "critical": {"k": crit_k, "n": len(crit)},
        }

    # ---------------- consultation-level head-to-head ------------------------------
    by_cons = collections.defaultdict(list)
    for nk in omission_notes:
        by_cons[roster[nk]["consultation"]].append(nk)
    both = g_only = r_only = neither = 0
    for cons, nks in sorted(by_cons.items()):
        g = any(gp_flag_majority(nk)[0] for nk in nks)
        b = any(b3_flag(b3_by_note[nk]) for nk in nks)
        both += g and b; g_only += g and not b; r_only += b and not g
        neither += (not g) and (not b)
    out["head_to_head_consultation"] = {
        "n_consultations": len(by_cons), "both": both, "gepa_only": g_only,
        "rule_only": r_only, "neither": neither,
        "union": both + g_only + r_only,
        "mcnemar_p": round(mcnemar_exact(g_only, r_only), 6)}

    # ---------------- per-residual and per-severity detection, both rules ----------
    # the published figures break detection down by how much of the fact survives
    def det_split(flag_of, keyf):
        split = {}
        for nk in omission_notes:
            split.setdefault(keyf(roster[nk]), [0, 0])
            split[keyf(roster[nk])][1] += 1
            if flag_of(nk):
                split[keyf(roster[nk])][0] += 1
        return {k: {"k": v[0], "n": v[1], "p": round(v[0] / v[1], 4),
                    "wilson": [round(x, 4) for x in wilson(v[0], v[1])[1:]]}
                for k, v in sorted(split.items()) if k}
    out["b3_rule"]["det_by_residual"] = det_split(
        lambda nk: b3_flag(b3_by_note[nk]), lambda m: m["residual"])
    out["gepa_majority"]["det_by_residual"] = det_split(
        lambda nk: gp_flag_majority(nk)[0], lambda m: m["residual"])

    # ---------------- 2: calibration on the clean twins ----------------------------
    cal = {}
    for rep in (1, 2, 3):
        at10 = n = 0
        for nk in clean_notes:
            s = gp_scores.get(nk, {}).get(rep)
            if s is None:
                continue
            n += 1
            if s == 10:
                at10 += 1
        cal[f"rep{rep}"] = {"at_10": at10, "n": n}
    out["gepa_clean_calibration"] = cal

    # ---------------- paired discrimination on omissions, both methods -------------
    gp_clean_mean = {nk: (sum(v for v in reps.values() if v is not None) /
                          max(1, sum(1 for v in reps.values() if v is not None)))
                     for nk, reps in gp_scores.items()}
    err = [dict(roster[nk], note_key=nk) for nk in omission_notes]
    for r in err:
        r["clean_key"] = roster[r["note_key"]]["clean_key"]
    pd_g, n_g = paired_disc(
        [r for r in err if r["note_key"] in gp_clean_mean and r["clean_key"] in gp_clean_mean],
        {nk: gp_clean_mean[nk] for nk in clean_notes if nk in gp_clean_mean},
        lambda r: gp_clean_mean[r["note_key"]])
    out["gepa_paired_omissions_evalset"] = {"paired": round(pd_g, 4), "n_pairs": n_g}

    # ---------------- paired discrimination, the study's usual convention ----------
    # per replicate then averaged - the convention every other paired figure in the
    # study uses - for both error classes and per residual level, so the published
    # figures can carry the powered arm beside the older arms.
    def paired_per_rep(scores_by_note, note_sel):
        reps_seen = sorted({rep for v in scores_by_note.values() for rep in v})
        per_rep = []
        n_max = 0
        for rep in reps_seen:
            rows = []
            for nk in omission_notes if note_sel is None else note_sel:
                m = roster[nk]
                s = scores_by_note.get(nk, {}).get(rep)
                c = scores_by_note.get(m["clean_key"], {}).get(rep)
                if s is None or c is None:
                    continue
                rows.append((s, c))
            if rows:
                w = sum(1.0 if s < c else 0.5 if s == c else 0.0 for s, c in rows)
                per_rep.append(w / len(rows))
                n_max = max(n_max, len(rows))
        if not per_rep:
            return None
        return {"paired": round(sum(per_rep) / len(per_rep), 4),
                "per_replicate": [round(v, 4) for v in per_rep], "n_pairs": n_max}

    commission_notes = [nk for nk, m in roster.items()
                        if m["role"] == "errored" and m["pair_class"] in ("add", "change")]
    gp_block = {"paired_omissions": paired_per_rep(gp_scores, omission_notes),
                "paired_commissions": paired_per_rep(gp_scores, commission_notes),
                "paired_by_residual": {
                    lv: paired_per_rep(gp_scores, [nk for nk in omission_notes
                                                   if roster[nk]["residual"] == lv])
                    for lv in ("complete", "partial-weak", "partial-strong")},
                "cost_usd_per_note": round(sum((r.get("totals") or {}).get("cost_usd", 0.0)
                                               for r in gp) / max(1, len(gp)), 4)}
    out["gepa_evalset_paired_per_replicate"] = gp_block

    b3_scores = {nk: {1: r.get("aggregate")} for nk, r in b3_by_note.items()}
    out["b3_evalset_paired"] = {
        "omissions": paired_per_rep(b3_scores, omission_notes),
        "commissions": paired_per_rep(b3_scores, commission_notes)}

    # held-out-subset vs remainder splits for detection (labels only, no re-testing)
    def split(name, flag_of):
        for lab, keep in (("heldout_subset", lambda m: m["pair_id"] in subset_pairs),
                          ("remainder", lambda m: m["pair_id"] not in subset_pairs)):
            nks = [nk for nk in omission_notes if keep(roster[nk])]
            k = sum(1 for nk in nks if flag_of(nk))
            out[name][lab] = {"k": k, "n": len(nks),
                              "p": round(k / len(nks), 4) if nks else None}
    split("b3_rule", lambda nk: b3_flag(b3_by_note[nk]))
    split("gepa_majority", lambda nk: gp_flag_majority(nk)[0])

    # ---------------- 4: the Gemini transfer ---------------------------------------
    gem_path = os.path.join(RESULTS, "w2-power/_state/power-gemini.jsonl")
    if os.path.exists(gem_path):
        gm = read_store(gem_path)
        parse_fail = sum(1 for r in gm if r.get("parse_failure"))
        by_note = collections.defaultdict(dict)
        for r in gm:
            by_note[r["note_key"]][r["replicate"]] = r.get("aggregate")
        reps_seen = sorted({r["replicate"] for r in gm})
        gm_cal, gm_det = {}, {}
        for rep in reps_seen:
            cl = [nk for nk in by_note if roster.get(nk, {}).get("role") == "clean"
                  and rep in by_note[nk]]
            at10 = sum(1 for nk in cl if by_note[nk][rep] == 10)
            om = [nk for nk in by_note if roster.get(nk, {}).get("role") == "errored"
                  and str(roster[nk]["pair_class"]).startswith("omit") and rep in by_note[nk]]
            det = sum(1 for nk in om if by_note[nk][rep] is not None
                      and by_note[nk][rep] < FLAG_BELOW)
            fa = sum(1 for nk in cl if by_note[nk][rep] is not None
                     and by_note[nk][rep] < FLAG_BELOW)
            gm_cal[f"rep{rep}"] = {"clean_at_10": at10, "n_clean": len(cl)}
            gm_det[f"rep{rep}"] = {"det_k": det, "det_n": len(om),
                                   "fa_k": fa, "fa_n": len(cl)}
        out["gemini_transfer"] = {"records": len(gm), "parse_failures": parse_fail,
                                  "replicates": reps_seen,
                                  "calibration": gm_cal, "detection_own_rule": gm_det}

        # majority-of-3 collapse + paired + receipts, mirroring the primary family's
        # blocks so the published figures can print the two families side by side
        def gm_flag_majority(nk):
            votes = [1 if (s is not None and s < FLAG_BELOW) else 0
                     for s in by_note.get(nk, {}).values()]
            return sum(votes) >= 2
        om_g = [nk for nk in omission_notes if nk in by_note]
        cl_g = [nk for nk in clean_notes if nk in by_note]
        co_g = [nk for nk in commission_notes if nk in by_note]
        det_k = sum(1 for nk in om_g if gm_flag_majority(nk))
        fa_k = sum(1 for nk in cl_g if gm_flag_majority(nk))
        p, lo, hi = wilson(det_k, len(om_g))
        fp, flo, fhi = wilson(fa_k, len(cl_g))
        crit_g = [nk for nk in om_g if roster[nk].get("severity") == "critical"]
        out["gemini_transfer"]["majority_rule"] = {
            "det": {"k": det_k, "n": len(om_g), "p": round(p, 4),
                    "wilson": [round(lo, 4), round(hi, 4)]},
            "fa": {"k": fa_k, "n": len(cl_g), "p": round(fp, 4),
                   "wilson": [round(flo, 4), round(fhi, 4)]},
            "critical": {"k": sum(1 for nk in crit_g if gm_flag_majority(nk)),
                         "n": len(crit_g)}}
        gm_scores = {nk: reps for nk, reps in by_note.items()}
        out["gemini_transfer"]["paired_per_replicate"] = {
            "paired_omissions": paired_per_rep(gm_scores, om_g),
            "paired_commissions": paired_per_rep(gm_scores, co_g),
            "paired_by_residual": {
                lv: paired_per_rep(gm_scores, [nk for nk in om_g
                                               if roster[nk]["residual"] == lv])
                for lv in ("complete", "partial-weak", "partial-strong")},
            "cost_usd_per_note": round(sum((r.get("totals") or {}).get("cost_usd", 0.0)
                                           for r in gm) / max(1, len(gm)), 4)}

    json.dump(out, open(os.path.join(OUT_DIR, "analysis.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
