"""second_panel_classmix.py - the class mix of what each review standard verifies.

A review of the GenAI4Health version of the census paper asked for exactly one table cell: the tier-1 class mix
of the 1,295-candidate sample under the lenient standard. This script computes it - and the
full composition of every standard's survivor set - from stored artifacts alone. No model
calls: the placement of open-pass findings uses the same deterministic June family matcher
(regex over the discovery model's mode label + description) that placed the census's own 618,
imported from the pilot script exactly as taxonomy_analyze.py imports it, so the two mixes
are computed by one code path and cannot diverge by construction.

What it tests: the census paper's contribution (iii) says the published audits' divergent class mixes
(omission 54-86% of errors there, 23.1% here) are consistent with instrument differences of
the size we measure. The published four-standard comparison measured the instrument's effect on the COUNT; this addendum
measures its effect on the MIX - what share of the verified findings each class contributes,
under each of the four standards on the same candidates.

Reported whatever it shows: if the mix does not move, that bounds the claim and goes in with
equal prominence.

Asserts against the published four-standard totals (1,295 candidates; A 121, B 230, C 134, D 1,023) and
against the published census class mix (618 placed as 207/181/143/46/41) before writing
results/second-panel/classmix.json.

    python3 census/second_panel_classmix.py
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import importlib.util, json, os, random, sys
from collections import Counter, defaultdict

from common import HERE, RESULTS
import taxonomy_common as T

# The June family matcher, imported from the pilot script itself (same as taxonomy_analyze).
_spec = importlib.util.spec_from_file_location(
    "cross_scribe_matches", os.path.join(HERE, "pilot", "scripts", "cross_scribe_matches.py"))
_xs = importlib.util.module_from_spec(_spec)
sys.path.insert(0, HERE)
_spec.loader.exec_module(_xs)
family, FAMILIES = _xs.family, _xs.FAMILIES
T.assert_family_keys_known(FAMILIES)

VERIFIED = "master/findings_verified_master.json"
STATE = os.path.join(RESULTS, "second-panel", "_state", "lenient-r1.jsonl")
OUT = os.path.join(RESULTS, "second-panel", "classmix.json")
BOOT_DRAWS = 10000
BOOT_SEED = 20260825

# Published four-standard and census totals - a mismatch is a refusal, not a warning.
EXPECT = {"n": 1295, "A": 121, "B": 230, "C": 134, "D": 1023,
          "census_tier1_placed": {"wrong_output": 207, "addition": 181, "omission": 143,
                                  "misplaced_or_irrelevant": 46, "unmapped": 41}}

TIER1_ORDER = ["omission", "addition", "wrong_output", "misplaced_or_irrelevant", "unmapped"]


def placed_tier1(f):
    t2, t1, how = T.frame_place(f, family_fn=family)
    return t1 or "unmapped"


def tob(x):
    return x if isinstance(x, bool) else str(x) == "True"


def load_rows():
    by_id = {}
    with open(STATE) as fh:
        for line in fh:
            r = json.loads(line)
            by_id[r["finding_id"]] = r          # last write wins (the 4 re-buys)
    return by_id


def mix(fids, findings):
    t1 = Counter(placed_tier1(findings[f]) for f in fids)
    t2 = Counter((findings[f].get("frame_tier2") or findings[f].get("pass")) for f in fids)
    n = len(fids)
    return {"n": n,
            "tier1_placed": {k: {"n": t1.get(k, 0), "share": round(t1.get(k, 0) / n, 4)}
                             for k in TIER1_ORDER},
            "tier2_pass": {k: {"n": v, "share": round(v / n, 4)}
                           for k, v in t2.most_common()}}


def cluster_boot_share(fids, findings, cls, by_cons, draws=BOOT_DRAWS, seed=BOOT_SEED):
    """Consultation-clustered percentile bootstrap on the share of a survivor set placed
    under `cls`. Resamples whole consultations from the consultations that hold at least
    one survivor in the set (the survivor set is the denominator of a mix share)."""
    keep = set(fids)
    cons = sorted({findings[f]["consultation"] for f in fids})
    per = {c: [f for f in by_cons[c] if f in keep] for c in cons}
    rng = random.Random(seed)
    stats = []
    for _ in range(draws):
        draw = [c for _ in cons for c in [cons[rng.randrange(len(cons))]]]
        num = den = 0
        for c in draw:
            for f in per[c]:
                den += 1
                if placed_tier1(findings[f]) == cls:
                    num += 1
        stats.append(num / den if den else 0.0)
    stats.sort()
    lo, hi = stats[int(0.025 * draws)], stats[int(0.975 * draws) - 1]
    return [round(lo, 4), round(hi, 4)]


def main():
    verified = json.load(open(os.path.join(HERE, VERIFIED)))
    findings = {f["finding_id"]: f for f in verified["all_issues"]}
    if len(findings) != 5898:
        raise SystemExit(f"master holds {len(findings)} candidates, expected 5,898")

    # Reproduce the published census class mix through this script's own placement path first.
    census_survivors = [f for f in verified["all_issues"] if (f.get("verdict") or {}).get("is_real")]
    census_t1 = Counter(placed_tier1(f) for f in census_survivors)
    if dict(census_t1) != EXPECT["census_tier1_placed"]:
        raise SystemExit(f"census tier-1 placement mismatch: {dict(census_t1)} "
                         f"vs published {EXPECT['census_tier1_placed']}")

    by_id = load_rows()
    if len(by_id) != EXPECT["n"]:
        raise SystemExit(f"{len(by_id)} scored candidates, expected {EXPECT['n']}")
    for fid in by_id:
        if fid not in findings:
            raise SystemExit(f"scored candidate {fid} not in the master")

    sets = {
        "A_constructor_strict": [f for f, r in by_id.items() if tob(r["constructor_strict_keep"])],
        "B_auditor_strict": [f for f, r in by_id.items() if tob(r["auditor_strict_keep"])],
        "C_panel": [f for f, r in by_id.items() if tob(r["panel_is_real"])],
        "D_lenient": [f for f, r in by_id.items() if tob(r["value"])],
    }
    for key, exp in (("A_constructor_strict", EXPECT["A"]), ("B_auditor_strict", EXPECT["B"]),
                     ("C_panel", EXPECT["C"]), ("D_lenient", EXPECT["D"])):
        if len(sets[key]) != exp:
            raise SystemExit(f"{key}: {len(sets[key])} survivors, expected {exp}")

    by_cons = defaultdict(list)
    for fid in by_id:
        by_cons[findings[fid]["consultation"]].append(fid)

    out = {"generated_utc": T.utc_now() if hasattr(T, "utc_now") else None,
           "in": {"state": STATE, "verified": VERIFIED},
           "method": "placed tier-1 via taxonomy_common.frame_place with the June family "
                     "matcher (deterministic regex; the census's own placement path, one "
                     "import), tier-2 as the pass that found the candidate. No model calls.",
           "expect": EXPECT,
           "pool_mix": mix(sorted(by_id), findings),
           "survivor_mix": {k: mix(sorted(v), findings) for k, v in sets.items()},
           "omission_share_ci": {}, "misplaced_share_ci": {}}

    # Clustered CIs on the shares the write-up leans on - all four standards for omission,
    # panel vs lenient for misplaced - plus the paired B-minus-A omission-share difference,
    # since "the family moves the mix" must not be asserted without a test.
    for cls, slot in (("omission", "omission_share_ci"), ("misplaced_or_irrelevant", "misplaced_share_ci")):
        arms = sets if cls == "omission" else {k: sets[k] for k in ("C_panel", "D_lenient")}
        for k in arms:
            out[slot][k] = cluster_boot_share(sorted(sets[k]), findings, cls, by_cons)

    # B minus A omission-share difference, consultation-clustered: resample consultations,
    # recompute both survivor mixes on the drawn set, difference of shares.
    rngd = random.Random(BOOT_SEED + 1)
    consL = sorted(by_cons)
    inA, inB = set(sets["A_constructor_strict"]), set(sets["B_auditor_strict"])
    diffs = []
    for _ in range(BOOT_DRAWS):
        drawn = [consL[rngd.randrange(len(consL))] for _ in consL]
        na = oa = nb = ob = 0
        for c in drawn:
            for fid in by_cons[c]:
                t1 = placed_tier1(findings[fid])
                if fid in inA:
                    na += 1
                    oa += t1 == "omission"
                if fid in inB:
                    nb += 1
                    ob += t1 == "omission"
        if na and nb:
            diffs.append(ob / nb - oa / na)
    diffs.sort()
    out["omission_share_B_minus_A"] = {
        "observed": round(78 / 230 - 27 / 121, 4),
        "ci95_clustered": [round(diffs[int(0.025 * len(diffs))], 4),
                           round(diffs[int(0.975 * len(diffs)) - 1], 4)],
        "excludes_zero": bool(diffs[int(0.025 * len(diffs))] > 0
                              or diffs[int(0.975 * len(diffs)) - 1] < 0)}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {OUT}")

    for k in ("pool",):
        print("\npool:", json.dumps(out["pool_mix"]["tier1_placed"]))
    for k, m in out["survivor_mix"].items():
        print(f"\n{k} (n={m['n']}):")
        for c in TIER1_ORDER:
            e = m["tier1_placed"][c]
            print(f"  {c:26s} {e['n']:5d}  {100*e['share']:5.1f}%")
    print("\nomission share CIs:", json.dumps(out["omission_share_ci"]))
    print("misplaced share CIs:", json.dumps(out["misplaced_share_ci"]))


if __name__ == "__main__":
    main()
