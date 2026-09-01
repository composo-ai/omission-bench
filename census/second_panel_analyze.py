"""second_panel_analyze.py - read-out for the second-panel arm.

Recomputes everything from `results/second-panel/_state/<tag>.jsonl` plus the shipped verified
master, asserts the joins against the census's published totals, and refuses to print a number
it could not reproduce. Nothing here is hand-typed: every published figure for this arm comes
out of `results/second-panel/analysis.json`.

Four standards over ONE candidate set (see second_panel.py's header for why four and not two):

  A  constructor, STRICT refute instruction, single call     - already bought by the census
  B  auditor (gpt-5.5), STRICT refute instruction, single    - already bought by the census
  C  the shipped two-family panel + tiebreak                 - already bought by the census
  D  constructor, LENIENT instruction, single call           - bought by this run

A vs D is the controlled contrast: same model, same evidence, same settings, same candidates,
and the instruction is the only difference. A vs C prices the panel machinery at a fixed
instruction. C vs D is the headline pair, and it moves both at once - so it is reported
as the two-standard bound it is, with the decomposition beside it.

    python3 census/second_panel_analyze.py --tag lenient-r1
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
from w2_analyze import wilson, cluster_bootstrap, mcnemar_exact

EXPERIMENT = "second-panel"
VERIFIED = "master/findings_verified_master.json"
BOOT_DRAWS = 10000
BOOT_SEED = 20260824

# The census's published totals. The joins are asserted against these; a mismatch is a refusal
# rather than a warning: every rate below is only meaningful if it sits on the same candidate set.
CENSUS = {"n_candidates": 5898, "n_verified": 618,
          "constructor_refute_rate": 0.9060, "auditor_refute_rate": 0.8130,
          "notes": 565, "notes_with_verified": 177}

ARMS = [("constructor_strict", "constructor, strict refute, single call (A)"),
        ("auditor_strict", "auditor gpt-5.5, strict refute, single call (B)"),
        ("panel", "two-family panel + tiebreak, shipped (C)"),
        ("lenient", "constructor, lenient standard, single call (D)")]


def load(tag):
    rows = []
    path = os.path.join(RESULTS, EXPERIMENT, "_state", f"{tag}.jsonl")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    by_id = {r["finding_id"]: r for r in rows}      # last write wins; keys are unique per call
    return by_id, path


def verdicts(row):
    """The four standards' keep/cut decisions for one candidate."""
    return {"constructor_strict": bool(row["constructor_strict_keep"]),
            "auditor_strict": bool(row["auditor_strict_keep"]),
            "panel": bool(row["panel_is_real"]),
            "lenient": bool(row["value"])}


def assert_joins(by_id, verified, subset):
    """Refuse to report if the substrate is not the one the census published."""
    issues = {f["finding_id"]: f for f in verified["all_issues"]}
    problems = []
    if len(issues) != CENSUS["n_candidates"]:
        problems.append(f"verified master holds {len(issues)} candidates, the census says "
                        f"{CENSUS['n_candidates']}")
    nv = sum(1 for f in issues.values() if f["verdict"]["is_real"])
    if nv != CENSUS["n_verified"]:
        problems.append(f"verified master holds {nv} verified, the census says "
                        f"{CENSUS['n_verified']}")
    for role, want in (("constructor", CENSUS["constructor_refute_rate"]),
                       ("auditor", CENSUS["auditor_refute_rate"])):
        got = sum(1 for f in issues.values()
                  if f["verdict"]["votes"].get(role) == "refute") / len(issues)
        if abs(got - want) > 0.0005:
            problems.append(f"{role} census refute rate {got:.4f} != the published {want:.4f}")
    # every scored candidate must be one of the drawn ones, and must carry the panel's own
    # verdict unchanged from the artifact
    drawn = set(subset["finding_ids"])
    stray = sorted(set(by_id) - drawn)
    if stray:
        problems.append(f"{len(stray)} scored candidates are not in the declared draw: {stray[:3]}")
    drift = [fid for fid, r in by_id.items()
             if fid in issues and bool(issues[fid]["verdict"]["is_real"]) != bool(r["panel_is_real"])]
    if drift:
        problems.append(f"{len(drift)} records disagree with the artifact's panel verdict")
    if problems:
        raise SystemExit("JOIN CHECK FAILED - refusing to report:\n  - " + "\n  - ".join(problems))
    return {"census_candidates": len(issues), "census_verified": nv,
            "drawn": len(drawn), "scored": len(by_id),
            "coverage_of_draw": round(len(by_id) / len(drawn), 6)}


def rate_block(rows, pick):
    k = sum(1 for r in rows if pick(r))
    return wilson(k, len(rows))


def by_group(rows, group, arms):
    out = {}
    buckets = defaultdict(list)
    for r in rows:
        buckets[group(r)].append(r)
    for g, rs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        out[str(g)] = {"n": len(rs),
                       **{a: rate_block(rs, lambda r, a=a: verdicts(r)[a]) for a, _ in arms}}
    return out


def note_level(rows, subset, arms):
    """Share of NOTES carrying at least one verified finding, under each standard.

    The denominator is every drawn note, including the ones the panel found no candidate on -
    which is why the draw is note-level: this figure is unbiased here and directly comparable to
    the census's published note-level rate of 31.3% [27.6, 35.3], where a candidate-level draw
    would undercount both arms.
    """
    per_note = defaultdict(list)
    for r in rows:
        per_note[r["note_key"]].append(r)
    denom = subset["note_keys"]
    out = {"n_notes": len(denom),
           "n_notes_with_candidates": len(per_note),
           "denominator": "every drawn note, including notes with no panel candidate"}
    for a, _ in arms:
        k = sum(1 for nk in denom if any(verdicts(r)[a] for r in per_note.get(nk, [])))
        out[a] = wilson(k, len(denom))
    out["per_note_counts"] = {
        a: {"mean": round(sum(sum(1 for r in per_note.get(nk, []) if verdicts(r)[a])
                              for nk in denom) / len(denom), 4)} for a, _ in arms}
    return out


def clustered(rows, arms):
    """Consultation-clustered bootstrap on every pairwise ratio and difference.

    Candidates cluster within consultations, so the Wilson intervals on each rate are optimistic
    about between-consultation variation and the ratio needs a clustered interval of its own.
    """
    clusters = defaultdict(list)
    for r in rows:
        clusters[r["consultation"]].append(r)

    def rate_of(arm):
        def f(sample):
            flat = [r for c in sample for r in c]
            return (sum(1 for r in flat if verdicts(r)[arm]) / len(flat)) if flat else None
        return f

    def ratio_of(a, b):
        def f(sample):
            flat = [r for c in sample for r in c]
            if not flat:
                return None
            ka = sum(1 for r in flat if verdicts(r)[a])
            kb = sum(1 for r in flat if verdicts(r)[b])
            return (ka / kb) if kb else None
        return f

    def diff_of(a, b):
        def f(sample):
            flat = [r for c in sample for r in c]
            if not flat:
                return None
            return (sum(1 for r in flat if verdicts(r)[a])
                    - sum(1 for r in flat if verdicts(r)[b])) / len(flat)
        return f

    out = {"n_clusters": len(clusters), "draws": BOOT_DRAWS, "seed": BOOT_SEED,
           "unit": "consultation",
           "rates": {a: cluster_bootstrap(clusters, rate_of(a), BOOT_DRAWS, BOOT_SEED)
                     for a, _ in arms}}
    pairs = [("lenient", "constructor_strict"), ("lenient", "panel"),
             ("auditor_strict", "constructor_strict"), ("panel", "constructor_strict")]
    out["ratios"] = {f"{a}_over_{b}": cluster_bootstrap(clusters, ratio_of(a, b),
                                                        BOOT_DRAWS, BOOT_SEED)
                     for a, b in pairs}
    out["differences"] = {f"{a}_minus_{b}": cluster_bootstrap(clusters, diff_of(a, b),
                                                              BOOT_DRAWS, BOOT_SEED)
                          for a, b in pairs}
    return out


def point_ratio(rows, a, b):
    ka = sum(1 for r in rows if verdicts(r)[a])
    kb = sum(1 for r in rows if verdicts(r)[b])
    return {"a": a, "b": b, "k_a": ka, "k_b": kb, "n": len(rows),
            "ratio": round(ka / kb, 4) if kb else None,
            "difference_pp": round(100 * (ka - kb) / len(rows), 3) if rows else None}


def gradient(rows, arms):
    """Does the checkability gradient survive a lenient standard?

    The published checkability finding is an ORDERING over tier-2 categories, not a set of
    levels, so the test is whether the ordering holds - Spearman between each arm's per-category
    rate and the panel's.
    """
    cats = defaultdict(list)
    for r in rows:
        cats[r["frame_tier2"]].append(r)
    keep = {c: rs for c, rs in cats.items() if len(rs) >= 20}     # rates on n<20 are noise
    base = {c: sum(1 for r in rs if verdicts(r)["panel"]) / len(rs) for c, rs in keep.items()}

    def spearman(x, y):
        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            rk = [0.0] * len(v)
            i = 0
            while i < len(order):                                  # average ties
                j = i
                while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                    j += 1
                for t in range(i, j + 1):
                    rk[order[t]] = (i + j) / 2 + 1
                i = j + 1
            return rk
        rx, ry = rank(x), rank(y)
        mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
        return round(num / den, 4) if den else None

    cs = sorted(keep)
    out = {"categories_used": cs, "min_n": 20,
           "categories_excluded_small": sorted(set(cats) - set(keep)),
           "spearman_vs_panel": {}}
    for a, _ in arms:
        r = {c: sum(1 for x in keep[c] if verdicts(x)[a]) / len(keep[c]) for c in cs}
        out["spearman_vs_panel"][a] = spearman([base[c] for c in cs], [r[c] for c in cs])
        out[f"rates_{a}"] = {c: round(r[c], 4) for c in cs}
    out["reading"] = ("Spearman of each arm's per-category verification rate against the shipped "
                      "panel's, over tier-2 categories with n>=20 in the subset. Near 1 means the "
                      "checkability ordering is a property of the material, not of adversarial "
                      "review; near 0 means the ordering is the panel's.")
    return out


def mcnemar(rows, a, b):
    """Exact McNemar, with the p-value kept at full precision.

    `w2_analyze.mcnemar_exact` rounds p to 8dp, which prints an overwhelming result as a
    literal 0. The rounded field is kept for comparability with every other read-out; `p_exact_
    scientific` carries the number that should actually be quoted.
    """
    out = mcnemar_exact({r["finding_id"]: verdicts(r)[a] for r in rows},
                        {r["finding_id"]: verdicts(r)[b] for r in rows})
    n = out["discordant_a_only"] + out["discordant_b_only"]
    p = 1.0 if n == 0 else float(stats.binomtest(out["discordant_a_only"], n, 0.5).pvalue)
    out["p_exact_scientific"] = f"{p:.3e}"
    out["arms"] = {"a": a, "b": b}
    return out


def spend(rows, tag):
    cost = sum(r.get("cost_usd") or 0 for r in rows)
    usage = defaultdict(int)
    for r in rows:
        for f, v in (r.get("usage") or {}).items():
            usage[f] += v or 0
    return {"tag": tag, "n_calls": len(rows), "cost_usd": round(cost, 4),
            "mean_cost_per_call": round(cost / len(rows), 6) if rows else None,
            "usage": dict(usage),
            "errors": sum(1 for r in rows if r.get("error")),
            "defaulted_verdicts": sum(1 for r in rows if r.get("defaulted")),
            "parsed_by": dict(Counter(r.get("parsed_by") or "json(pre-fix)" for r in rows))}


def main():
    ap = argparse.ArgumentParser(description="read-out for the second-panel arm")
    ap.add_argument("--tag", default="lenient-r1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    by_id, store_path = load(args.tag)
    verified = json.load(open(os.path.join(HERE, VERIFIED)))
    subset = json.load(open(os.path.join(RESULTS, EXPERIMENT, f"subset_{args.tag}.json")))
    joins = assert_joins(by_id, verified, subset)
    rows = [by_id[f] for f in sorted(by_id)]
    if joins["coverage_of_draw"] < 1.0:
        print(f"! partial: {joins['scored']} of {joins['drawn']} drawn candidates scored "
              f"({joins['coverage_of_draw']:.1%}) - every figure below is on the scored subset")

    arms = ARMS
    res = {
        "experiment": EXPERIMENT, "tag": args.tag,
        "generated_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     __import__("time").gmtime()),
        "store": os.path.relpath(store_path, HERE),
        "subset_artifact": f"results/{EXPERIMENT}/subset_{args.tag}.json",
        "draw": subset["draw"], "marginal_fit": subset["marginal_fit"],
        "joins": joins,
        "arms": {a: {"label": lab} for a, lab in arms},
        "n_candidates": len(rows),
        "n_consultations": len({r["consultation"] for r in rows}),
        "n_notes": len({r["note_key"] for r in rows}),
        "primary": {
            "rates_wilson": {a: rate_block(rows, lambda r, a=a: verdicts(r)[a]) for a, _ in arms},
            "controlled_contrast": point_ratio(rows, "lenient", "constructor_strict"),
            "headline_pair": point_ratio(rows, "lenient", "panel"),
            "panel_machinery": point_ratio(rows, "panel", "constructor_strict"),
            "mcnemar_lenient_vs_constructor_strict": mcnemar(rows, "lenient", "constructor_strict"),
            "mcnemar_lenient_vs_panel": mcnemar(rows, "lenient", "panel"),
            "mcnemar_panel_vs_constructor_strict": mcnemar(rows, "panel", "constructor_strict"),
        },
        "clustered": clustered(rows, arms),
        "by_frame_tier2": by_group(rows, lambda r: r["frame_tier2"], arms),
        "by_salience": by_group(rows, lambda r: r["salience"], arms),
        "by_scribe": by_group(rows, lambda r: r["scribe"], arms),
        "by_frame_tier1": by_group(rows, lambda r: r.get("frame_tier1") or "(open)", arms),
        "gradient": gradient(rows, arms),
        "note_level": note_level(rows, subset, arms),
        "agreement": {
            "lenient_kept_panel_cut": sum(1 for r in rows
                                          if verdicts(r)["lenient"] and not verdicts(r)["panel"]),
            "panel_kept_lenient_cut": sum(1 for r in rows
                                          if verdicts(r)["panel"] and not verdicts(r)["lenient"]),
            "both_kept": sum(1 for r in rows if verdicts(r)["lenient"] and verdicts(r)["panel"]),
            "both_cut": sum(1 for r in rows
                            if not verdicts(r)["lenient"] and not verdicts(r)["panel"]),
            "lenient_recall_of_panel_survivors": None,
        },
        "spend": spend(rows, args.tag),
    }
    a = res["agreement"]
    a["lenient_recall_of_panel_survivors"] = round(
        a["both_kept"] / (a["both_kept"] + a["panel_kept_lenient_cut"]), 4) \
        if (a["both_kept"] + a["panel_kept_lenient_cut"]) else None

    out = args.out or os.path.join(RESULTS, EXPERIMENT, "analysis.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)

    # ------------------------------------------------------------ read-out
    p = res["primary"]
    print(f"\n=== second-panel arm | {res['n_candidates']} candidates, {res['n_notes']} notes, "
          f"{res['n_consultations']} consultations ===")
    print(f"draw seed {res['draw']['seed']}, worst marginal gap "
          f"{res['marginal_fit']['max_abs_share_gap_pp']}pp; panel reproduces its census rate on "
          f"the subset ({p['rates_wilson']['panel']['p']:.4f} v 0.1048 over 5,898)\n")
    print(f"{'standard':52} {'verified':>14}  {'rate':>7}  {'Wilson 95%':>18}  {'cluster 95%':>18}")
    for k, lab in arms:
        w = p["rates_wilson"][k]
        c = res["clustered"]["rates"][k]
        print(f"  {lab:50} {w['k']:5}/{w['n']:<6} {w['p']:7.2%}  "
              f"[{w['lo']:6.2%},{w['hi']:6.2%}]  [{c['lo']:6.2%},{c['hi']:6.2%}]")
    for name, key, cl in (("CONTROLLED (instruction only, D/A)", "controlled_contrast",
                           "lenient_over_constructor_strict"),
                          ("headline pair (D/C)", "headline_pair", "lenient_over_panel"),
                          ("panel machinery (C/A)", "panel_machinery",
                           "panel_over_constructor_strict")):
        r, cb = p[key], res["clustered"]["ratios"][cl]
        print(f"\n{name}: {r['k_a']}/{r['n']} v {r['k_b']}/{r['n']} = "
              f"ratio {r['ratio']}x, {r['difference_pp']:+.1f}pp; "
              f"consultation-clustered 95% CI on the ratio [{cb['lo']}, {cb['hi']}] "
              f"({cb['n_clusters']} clusters)")
    for lab, key in (("D v A", "mcnemar_lenient_vs_constructor_strict"),
                     ("C v A", "mcnemar_panel_vs_constructor_strict")):
        m = p[key]
        print(f"  exact McNemar ({lab}), paired on the same candidates: "
              f"{m['discordant_a_only']} / {m['discordant_b_only']} discordant, "
              f"p = {m['p_exact_scientific']}")

    print("\nby tier-2 category (n >= 20), verification rate:")
    print(f"  {'category':24} {'n':>5}  {'panel':>7} {'C-strict':>9} {'lenient':>8}")
    for c, b in res["by_frame_tier2"].items():
        if b["n"] < 20:
            continue
        print(f"  {c:24} {b['n']:5}  {b['panel']['p']:7.1%} {b['constructor_strict']['p']:9.1%} "
              f"{b['lenient']['p']:8.1%}")
    g = res["gradient"]
    print("  Spearman of per-category rate against the panel's: " +
          ", ".join(f"{k} {v}" for k, v in g["spearman_vs_panel"].items() if k != "panel"))

    print("\nby panel salience:")
    for s, b in res["by_salience"].items():
        print(f"  {s:8} n={b['n']:5}  panel {b['panel']['p']:6.1%}  "
              f"C-strict {b['constructor_strict']['p']:6.1%}  lenient {b['lenient']['p']:6.1%}")

    nl = res["note_level"]
    print(f"\nnotes carrying at least one verified finding (n = {nl['n_notes']} drawn notes):")
    for k, lab in arms:
        w = nl[k]
        print(f"  {lab:50} {w['k']:4}/{w['n']:<4} = {w['p']:6.1%} [{w['lo']:.1%}, {w['hi']:.1%}]")
    print(f"  census anchor: 177/565 = 31.3% [27.6%, 35.3%] under the shipped panel")

    print(f"\nagreement: both keep {a['both_kept']}, lenient-only {a['lenient_kept_panel_cut']}, "
          f"panel-only {a['panel_kept_lenient_cut']}, both cut {a['both_cut']}; the lenient "
          f"standard keeps {a['lenient_recall_of_panel_survivors']:.1%} of the panel's survivors")
    s = res["spend"]
    print(f"\nspend: ${s['cost_usd']:.2f} over {s['n_calls']} calls "
          f"(${s['mean_cost_per_call']:.5f}/call), {s['errors']} transport errors, "
          f"{s['defaulted_verdicts']} verdicts still unreadable; replies read as {s['parsed_by']}")
    print(f"saved -> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
