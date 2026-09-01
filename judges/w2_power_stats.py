"""w2_power_stats.py - the statistical treatment the judge-power numbers need before any
paper sentence. No model calls; stored verdicts only.

By convention the analysis unit is the CONSULTATION wherever a claim spans pairs (131
omission pairs over 47 consultations are not 131 independent observations):

  1. Powered-vs-control paired-omission deltas with consultation-clustered bootstrap CIs.
  2. The power-gepa-high deployable operating point (flag iff score < 10): detection vs its
     own false alarms (two-proportion z + exact Fisher), a clustered bootstrap CI on
     detection, and the comparison against B3's per-fact critical rule - pair-level McNemar
     (reported but labelled pseudo-replicated) AND consultation-level any-catch McNemar.
  3. Severity- and residual-stratified detection for the powered arms' "<10" rule and the
     per-fact rule, plus the catch-overlap table between them.
  4. The sitting's ten blind-adjudicated disagreement cases re-read under the powered arms
     (matched to note keys via the B3 fact text the sitting builder drew from).

Bootstrap seed 20260818, 10,000 resamples. Reads the same stores as w2_power_analyze.py;
also results/w2-power/_state/power-eng.jsonl when it exists.

    python3 judges/w2_power_stats.py
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import json, math, os, random
from collections import defaultdict

from common import HERE
import w2_common as W

STORES = ["results/w2-power/_state/power.jsonl", "results/w2-power/_state/power-eng.jsonl"]
GRID = "results/w2-ablation/_state/grid-main2.jsonl"
FACTLIST_STORE = "results/w2-factlist/_state/factlist.jsonl"
STRONG_STORE = "results/w2-strong/_state/arms-main.jsonl"
B3_STORE = "results/w2-pipeline/_state/confirm-B3.jsonl"
SUBSET = "master/arms_confirm_subset.json"
SITTING = "master/sitting_results.json"
OUT = "results/w2-power/stats.json"

BOOT_SEED, BOOT_N = 20260818, 10000

CONTROLS = {"power-FC-med": ("grid", "FC-score-k1"),
            "power-FC-high": ("grid", "FC-score-k1"),
            "power-factlist-high": ("factlist", None),
            "power-gepa-high": ("strong", "gepa-optimized"),
            "power-engineered-high": ("strong", "engineered-completeness")}


def load(path):
    full = os.path.join(HERE, path)
    if not os.path.exists(full):
        return []
    return [json.loads(l) for l in open(full) if l.strip()]


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"p": round(p, 4), "k": k, "n": n, "lo": round(c - h, 4), "hi": round(c + h, 4)}


def binom_two_sided(k, n):
    """Exact two-sided binomial test against 0.5 (the McNemar exact test on n discordant)."""
    if n == 0:
        return None
    pk = lambda i: math.comb(n, i) * 0.5 ** n
    ref = pk(k)
    return round(min(1.0, sum(pk(i) for i in range(n + 1) if pk(i) <= ref + 1e-15)), 6)


def fisher_right(a, b, c, d):
    """P(X >= a) for the 2x2 [[a,b],[c,d]] under the hypergeometric null (one-sided)."""
    row1, col1, n = a + b, a + c, a + b + c + d
    lo, hi = max(0, row1 + col1 - n), min(row1, col1)
    denom = math.comb(n, col1)
    return round(sum(math.comb(row1, x) * math.comb(n - row1, col1 - x)
                     for x in range(a, hi + 1)) / denom, 8)


def paired_outcome(err, clean):
    return 1.0 if err < clean else 0.5 if err == clean else 0.0


def per_pair_outcomes(recs):
    """pair_id -> mean paired outcome across replicates (omissions only), plus consultation."""
    by_rep = defaultdict(dict)   # (rep) -> note_key -> rec
    for r in recs:
        by_rep[r["replicate"]][r["note_key"]] = r
    out, cons = defaultdict(list), {}
    for rep, notes in by_rep.items():
        clean = {(r["stratum"], r["consultation"]): r for r in notes.values()
                 if r["note_role"] == "clean"}
        for r in notes.values():
            if not str(r.get("pair_class") or "").startswith("omit"):
                continue
            c = clean.get((r["stratum"], r["consultation"]))
            if c is None or r.get("parse_failure") or c.get("parse_failure"):
                continue
            out[r["pair_id"]].append(paired_outcome(r["aggregate"], c["aggregate"]))
            cons[r["pair_id"]] = f"{r['stratum']}|{r['consultation']}"
    return {p: sum(v) / len(v) for p, v in out.items()}, cons


def cluster_boot(values_by_cons, stat, n=BOOT_N, seed=BOOT_SEED):
    """Bootstrap CI resampling consultations with replacement. values_by_cons:
    cons -> list of per-pair values; stat: flat list -> float."""
    rng = random.Random(seed)
    cons = sorted(values_by_cons)
    draws = []
    for _ in range(n):
        sample = [v for c in (rng.choice(cons) for _ in cons) for v in values_by_cons[c]]
        if sample:
            draws.append(stat(sample))
    draws.sort()
    return {"lo": round(draws[int(0.025 * len(draws))], 4),
            "hi": round(draws[int(0.975 * len(draws))], 4)}


def main():
    subset = [json.loads(json.dumps(p)) for p in W.load_dataset(path=SUBSET)[0]]
    keep = {p["pair_id"] for p in subset}
    kc = {(p["stratum"], p["id"]) for p in subset}
    on_subset = lambda r: ((r.get("note_role") == "clean"
                            and (r.get("stratum"), r.get("consultation")) in kc)
                           or r.get("pair_id") in keep)

    # power.jsonl also holds power-gepa-high over the full 495-pair evaluation set
    # since 2026-08-24; this read-out is the held-out dose-response, so
    # powered records are restricted to the confirmation subset - without this the
    # gepa arm fails the 198-multiple completeness gate below and silently drops.
    power = [r for s in STORES for r in load(s) if on_subset(r)]
    grid = load(GRID)
    strong = load(STRONG_STORE)
    factlist = load(FACTLIST_STORE)
    b3 = load(B3_STORE)

    sources = {"factlist": factlist,
               "grid": [r for r in grid if on_subset(r)],
               "strong": [r for r in strong if on_subset(r)]}

    out = {"seed": BOOT_SEED, "boot_n": BOOT_N, "deltas": {}, "gepa_detection": {},
           "strata": {}, "overlap": {}, "sitting_F": {}}

    # ---- 1. clustered deltas -----------------------------------------------------
    # An arm is only reportable when its run is complete (198 notes per replicate);
    # a mid-flight store would print a delta computed on a fragment.
    counts = defaultdict(set)
    for r in power:
        counts[r["arm"]].add((r["replicate"], r["note_key"]))
    arms_present = sorted(a for a, ks in counts.items()
                          if len(ks) % 198 == 0 and len(ks) > 0)
    partial = sorted(set(counts) - set(arms_present))
    if partial:
        print(f"(skipping mid-flight arms: {partial})")
    for an in arms_present:
        src, cell = CONTROLS[an]
        crecs = sources[src] if cell is None else \
            [r for r in sources[src] if (r.get("cell") or r.get("arm")) == cell]
        precs = [r for r in power if r["arm"] == an]
        po, cons = per_pair_outcomes(precs)
        co, _ = per_pair_outcomes(crecs)
        common = sorted(set(po) & set(co))
        diffs_by_cons = defaultdict(list)
        for p in common:
            diffs_by_cons[cons[p]].append(po[p] - co[p])
        mean = lambda v: sum(v) / len(v)
        delta = mean([po[p] - co[p] for p in common])
        ci = cluster_boot(diffs_by_cons, mean)
        n_reps = len({r["replicate"] for r in precs})
        out["deltas"][an] = {
            "n_pairs": len(common), "n_consultations": len(diffs_by_cons),
            "n_replicates_powered": n_reps,
            "powered_paired": round(mean([po[p] for p in common]), 4),
            "control_paired": round(mean([co[p] for p in common]), 4),
            "delta": round(delta, 4), "delta_ci95_cluster": ci,
            "excludes_zero": ci["lo"] > 0 or ci["hi"] < 0}

    # ---- 2. the gepa operating point ---------------------------------------------
    gp = [r for r in power if r["arm"] == "power-gepa-high"]
    # only complete replicates (198 notes each) - a mid-flight replicate would skew the
    # majority-flag rule and the FA denominator
    by_rep_n = defaultdict(set)
    for r in gp:
        by_rep_n[r["replicate"]].add(r["note_key"])
    gp = [r for r in gp if len(by_rep_n[r["replicate"]]) == 198]
    if gp:
        flags = defaultdict(list)     # note_key -> [flag per replicate]
        for r in gp:
            if not r.get("parse_failure"):
                flags[r["note_key"]].append(r["aggregate"] < 10)
        nflag = {k: (sum(v) / len(v) >= 0.5) for k, v in flags.items()}  # majority
        om = [r for r in gp if str(r.get("pair_class") or "").startswith("omit")
              and r["replicate"] == min(r2["replicate"] for r2 in gp)]
        cl = [r for r in gp if r["note_role"] == "clean"
              and r["replicate"] == min(r2["replicate"] for r2 in gp)]
        det = wilson(sum(1 for r in om if nflag[r["note_key"]]), len(om))
        fa = wilson(sum(1 for r in cl if nflag[r["note_key"]]), len(cl))
        z = ((det["p"] - fa["p"]) /
             math.sqrt(sum(1 for r in om + cl if nflag[r["note_key"]]) / (len(om) + len(cl))
                       * (1 - sum(1 for r in om + cl if nflag[r["note_key"]])
                          / (len(om) + len(cl))) * (1 / len(om) + 1 / len(cl))))
        det_by_cons = defaultdict(list)
        for r in om:
            det_by_cons[f"{r['stratum']}|{r['consultation']}"].append(
                1.0 if nflag[r["note_key"]] else 0.0)
        # per-fact rule catches from B3
        rule_catch = {}
        for r in b3:
            if str(r.get("pair_class") or "").startswith("omit"):
                rule_catch[r["pair_id"]] = any(
                    m.get("severity") == "critical" and m.get("verdict") == "absent"
                    for m in ((r.get("detail") or {}).get("missing_facts") or []))
        gp_catch = {r["pair_id"]: nflag[r["note_key"]] for r in om}
        both = sorted(set(gp_catch) & set(rule_catch))
        b_g = sum(1 for p in both if gp_catch[p] and not rule_catch[p])
        b_r = sum(1 for p in both if rule_catch[p] and not gp_catch[p])
        cons_any = defaultdict(lambda: [False, False])
        for p in both:
            c = [pp for pp in subset if pp["pair_id"] == p][0]
            key = f"{c['stratum']}|{c['id']}"
            cons_any[key][0] |= gp_catch[p]
            cons_any[key][1] |= rule_catch[p]
        c_g = sum(1 for g, r in cons_any.values() if g and not r)
        c_r = sum(1 for g, r in cons_any.values() if r and not g)
        out["gepa_detection"] = {
            "rule": "flag iff score < 10 (majority across available replicates)",
            "replicates_used": sorted({r["replicate"] for r in gp}),
            "det": det, "fa": fa, "z_vs_own_noise": round(z, 2),
            "fisher_det_vs_fa_p": fisher_right(
                det["k"], det["n"] - det["k"], fa["k"], fa["n"] - fa["k"]),
            "det_ci95_cluster": cluster_boot(det_by_cons, lambda v: sum(v) / len(v)),
            "vs_per_fact_rule": {
                "n_pairs": len(both),
                "gepa_catches": sum(gp_catch[p] for p in both),
                "rule_catches": sum(rule_catch[p] for p in both),
                "both": sum(1 for p in both if gp_catch[p] and rule_catch[p]),
                "gepa_only": b_g, "rule_only": b_r,
                "mcnemar_pair_level_p_PSEUDOREPLICATED": binom_two_sided(b_g, b_g + b_r),
                "consultations_gepa_only": c_g, "consultations_rule_only": c_r,
                "mcnemar_consultation_level_p": binom_two_sided(c_g, c_g + c_r)}}

        # ---- 3. strata + overlap -------------------------------------------------
        sev = {p["pair_id"]: p.get("severity") for p in subset}
        res = {p["pair_id"]: p.get("residual_level") for p in subset}
        for name, catch in (("power-gepa-high(<10)", gp_catch), ("per-fact-rule", rule_catch)):
            by = {}
            for dim, m in (("severity", sev), ("residual", res)):
                agg = defaultdict(lambda: [0, 0])
                for p, c in catch.items():
                    if p in keep:
                        agg[m.get(p) or "none"][0] += int(c)
                        agg[m.get(p) or "none"][1] += 1
                by[dim] = {k: wilson(v[0], v[1]) for k, v in sorted(agg.items())}
            out["strata"][name] = by
        out["overlap"] = {"note": "omission pairs where each rule fires",
                          "both": out["gepa_detection"]["vs_per_fact_rule"]["both"],
                          "gepa_only": b_g, "rule_only": b_r,
                          "neither": len(both) - b_g - b_r
                                     - out["gepa_detection"]["vs_per_fact_rule"]["both"]}

    # ---- 4. the sitting's F cases under the powered arms -------------------------
    sit = json.load(open(os.path.join(HERE, SITTING)))
    rows = sit["sections"]["F"]["adjudication_table"]
    fact_to_note = defaultdict(set)
    for r in b3:
        for m in ((r.get("detail") or {}).get("missing_facts") or []):
            fact_to_note[(m.get("fact") or "").strip().lower()].add(r["note_key"])
    by_note = defaultdict(dict)
    for r in power:
        by_note[r["note_key"]][(r["arm"], r["replicate"])] = r
    fout = []
    for row in rows:
        key = (row.get("item_fact") or "").strip().lower()
        cands = fact_to_note.get(key, set())
        entry = {"id": row["id"], "case": row["case"], "clinician": row["clinician"],
                 "matched_note": sorted(cands)[0] if len(cands) == 1 else None,
                 "ambiguous": len(cands) != 1}
        if len(cands) == 1:
            nk = sorted(cands)[0]
            entry["powered_scores"] = {
                f"{a}(r{rep})": by_note[nk][(a, rep)]["aggregate"]
                for (a, rep) in sorted(by_note.get(nk, {}))}
        fout.append(entry)
    out["sitting_F"] = {"note": "the sitting's ten blind-adjudicated disagreement cases; "
                                "matched by B3 fact text; scores are the powered arms' "
                                "aggregates on the same note",
                        "cases": fout,
                        "matched": sum(1 for e in fout if e["matched_note"])}

    dest = os.path.join(HERE, OUT)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=1)

    print(f"deltas (consultation-clustered bootstrap, seed {BOOT_SEED}):")
    for an, d in out["deltas"].items():
        print(f"  {an:<24} {d['control_paired']:.3f} -> {d['powered_paired']:.3f} "
              f"delta {d['delta']:+.3f} CI [{d['delta_ci95_cluster']['lo']:+.3f}, "
              f"{d['delta_ci95_cluster']['hi']:+.3f}]"
              f"{'  *' if d['excludes_zero'] else '  (includes 0)'}")
    g = out.get("gepa_detection")
    if g:
        print(f"\npower-gepa-high '<10' rule (reps {g['replicates_used']}): "
              f"det {100*g['det']['p']:.1f}% [{100*g['det']['lo']:.1f}, {100*g['det']['hi']:.1f}] "
              f"@ FA {100*g['fa']['p']:.1f}%; z={g['z_vs_own_noise']}, "
              f"Fisher p={g['fisher_det_vs_fa_p']:.2e}; "
              f"clustered det CI [{100*g['det_ci95_cluster']['lo']:.1f}, "
              f"{100*g['det_ci95_cluster']['hi']:.1f}]%")
        v = g["vs_per_fact_rule"]
        print(f"  vs per-fact rule: both {v['both']}, gepa-only {v['gepa_only']}, "
              f"rule-only {v['rule_only']}; McNemar pair-level p={v['mcnemar_pair_level_p_PSEUDOREPLICATED']} "
              f"(pseudo-replicated), consultation-level {v['consultations_gepa_only']}-"
              f"{v['consultations_rule_only']} p={v['mcnemar_consultation_level_p']}")
    for name, s in out["strata"].items():
        sv = s["severity"]
        print(f"\n{name} detection by severity: " + ", ".join(
            f"{k} {100*v['p']:.0f}% ({v['k']}/{v['n']})" for k, v in sv.items() if v))
    sf = out["sitting_F"]
    print(f"\nsitting F cases matched: {sf['matched']}/10 (detail in stats.json)")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
