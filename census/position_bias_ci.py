"""position_bias_ci.py - uncertainty for A.5's verification position-bias comparison.

The census paper's Appendix A.5 reports that panel survivors sit systematically earlier than
the candidate pool they came from (transcript position 0.477 vs 0.527; note position 0.465 vs
0.525). This attaches an interval to that difference, offline, from
the stored quote spans, with no model calls.

The estimand is the difference in mean normalised start position, verified findings minus the
full candidate pool, for transcript spans and note spans separately. Interval: percentile
bootstrap resampling whole consultations (10,000 draws, fixed seed), the same clustering
convention as every headline interval in the census paper, because candidates from one consultation share a
transcript. The verified set is a subset of the pool; each bootstrap draw recomputes both
means on the drawn consultations, so the dependence is carried, not assumed away.

Positions are recomputed here through taxonomy_analyze.locate - the same graded quote-location
function that produced the published means - and the artifact asserts the recomputed means
match the published figures to 3 decimal places before writing.

    python3 census/position_bias_ci.py
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import json, os, random

from common import HERE, RESULTS
import taxonomy_common as T
from taxonomy_analyze import locate

MASTER = "master/findings_master.json"
VERIFIED = "master/findings_verified_master.json"
OUT = os.path.join(RESULTS, "position-bias-ci", "position_ci.json")
DRAWS = 10000
SEED = 20260825

# The published means - the recompute must land on these (3dp) or the script refuses.
EXPECT = {"tx_all": 0.527, "tx_verified": 0.477, "note_all": 0.525, "note_verified": 0.465}


def main():
    m = json.load(open(os.path.join(HERE, MASTER)))
    d = json.load(open(os.path.join(HERE, VERIFIED)))
    candidates = m["all_issues"]
    if len(candidates) != 13678:
        raise SystemExit(f"{len(candidates)} discovery candidates, expected 13,678")
    vids = {f["finding_id"] for f in d["all_issues"] if (f.get("verdict") or {}).get("is_real")}
    if len(vids) != 618:
        raise SystemExit(f"{len(vids)} verified, expected 618")

    units, _ = T.note_units()
    notes = {u["note_key"]: u for u in units}

    rows = []
    for f in candidates:
        u = notes.get(f["note_key"])
        if not u:
            continue
        tpos, _ = locate(f.get("source_quote"), u["transcript"][:40000])
        npos, _ = locate(f.get("note_quote"), u["note"])
        rows.append({"cons": f["id"], "tx": tpos, "note": npos,
                     "verified": f["finding_id"] in vids})

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    obs = {"tx_all": mean([r["tx"] for r in rows]),
           "tx_verified": mean([r["tx"] for r in rows if r["verified"]]),
           "note_all": mean([r["note"] for r in rows]),
           "note_verified": mean([r["note"] for r in rows if r["verified"]])}
    for k, v in EXPECT.items():
        if abs(obs[k] - v) > 5e-4:
            raise SystemExit(f"recomputed {k}={obs[k]:.4f} does not match published {v}")

    by_cons = {}
    for r in rows:
        by_cons.setdefault(r["cons"], []).append(r)
    cons = sorted(by_cons)
    rng = random.Random(SEED)

    diffs = {"tx": [], "note": []}
    for _ in range(DRAWS):
        drawn = [cons[rng.randrange(len(cons))] for _ in cons]
        pool = [r for c in drawn for r in by_cons[c]]
        for field in ("tx", "note"):
            a = mean([r[field] for r in pool])
            v = mean([r[field] for r in pool if r["verified"]])
            if a is not None and v is not None:
                diffs[field].append(v - a)

    out = {"draws": DRAWS, "seed": SEED, "n_clusters": len(cons),
           "n_candidates_located": {"tx": sum(r["tx"] is not None for r in rows),
                                    "note": sum(r["note"] is not None for r in rows)},
           "n_verified_located": {"tx": sum(r["tx"] is not None for r in rows if r["verified"]),
                                  "note": sum(r["note"] is not None for r in rows if r["verified"])},
           "observed_means": {k: round(v, 4) for k, v in obs.items()}}
    for field, a_key, v_key in (("tx", "tx_all", "tx_verified"), ("note", "note_all", "note_verified")):
        ds = sorted(diffs[field])
        lo, hi = ds[int(0.025 * len(ds))], ds[int(0.975 * len(ds)) - 1]
        out[f"{field}_diff"] = {"observed": round(obs[v_key] - obs[a_key], 4),
                                "ci95_clustered": [round(lo, 4), round(hi, 4)],
                                "excludes_zero": bool(hi < 0 or lo > 0)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
