"""precision_sitting_analyze.py - unblind and score the census precision sitting.

Takes the physician's answers (one line per item: `<item>: genuine|not-genuine|cannot-judge
[critical|supporting|peripheral]`), joins them to results/precision-sitting/key.json, and
computes the pre-stated read-outs:

  - precision on the 30 sampled verified findings: x of 30 judged genuine, Wilson 95%,
    plus a consultation-clustered bootstrap interval, and the abstention-excluded read
  - the foil agreement rate: of the 15 panel-refused foils, how many he judged not genuine
  - severity agreement on verified items judged genuine, against the rubric grades

Answers file: results/precision-sitting/answers.txt (or --answers PATH), free-form lines,
`#` comments ignored. Writes results/precision-sitting/sitting_results.json.

    python3 sittings/precision_sitting_analyze.py
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import argparse, json, os, random, re

from common import HERE, RESULTS
from w2_analyze import wilson

OUTDIR = os.path.join(RESULTS, "precision-sitting")
DRAWS = 10000
SEED = 20260825


def parse_answers(path):
    ans = {}
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line:
            continue
        m = re.match(r"^(\d+)\s*[:.]\s*(genuine|not-genuine|not genuine|cannot-judge|cannot judge|g|n|-)"
                     r"(?:\s*[,;]?\s*(critical|supporting|peripheral|c|s|p))?\s*$", line, re.I)
        if not m:
            raise SystemExit(f"cannot parse answer line: {line!r}")
        item = int(m.group(1))
        verdict = {"g": "genuine", "n": "not-genuine", "-": "cannot-judge"}.get(
            m.group(2).lower(), m.group(2).lower().replace(" ", "-"))
        sev = {"c": "critical", "s": "supporting", "p": "peripheral"}.get(
            (m.group(3) or "").lower(), (m.group(3) or "").lower() or None)
        if item in ans:
            raise SystemExit(f"duplicate answer for item {item}")
        ans[item] = {"verdict": verdict, "severity": sev}
    return ans


def cluster_boot(items, hit, draws=DRAWS, seed=SEED):
    """Percentile bootstrap over consultations on the share of items with hit(item)."""
    by_cons = {}
    for it in items:
        by_cons.setdefault(it["consultation"], []).append(it)
    cons = sorted(by_cons)
    rng = random.Random(seed)
    stats = []
    for _ in range(draws):
        drawn = [cons[rng.randrange(len(cons))] for _ in cons]
        num = den = 0
        for c in drawn:
            for it in by_cons[c]:
                den += 1
                num += bool(hit(it))
        if den:
            stats.append(num / den)
    stats.sort()
    return [round(stats[int(0.025 * len(stats))], 4), round(stats[int(0.975 * len(stats)) - 1], 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default=os.path.join(OUTDIR, "answers.txt"))
    args = ap.parse_args()

    key = json.load(open(os.path.join(OUTDIR, "key.json")))
    ans = parse_answers(args.answers)
    missing = sorted(k["item"] for k in key if k["item"] not in ans)
    n_reached = max(ans) if ans else 0
    if missing and missing != list(range(n_reached + 1, len(key) + 1)):
        raise SystemExit(f"answers must be a prefix of the shuffled order; gaps: {missing[:5]}")
    stray = [i for i in ans if i not in {k['item'] for k in key}]
    if stray:
        raise SystemExit(f"answers for unknown items: {stray}")

    key = [k for k in key if k["item"] <= n_reached]
    for k in key:
        k.update(ans[k["item"]])
    ver = [k for k in key if k["status"] == "verified"]
    foil = [k for k in key if k["status"] == "refused_foil"]
    nv, nf = len(ver), len(foil)

    g = sum(1 for k in ver if k["verdict"] == "genuine")
    ng = sum(1 for k in ver if k["verdict"] == "not-genuine")
    cj = sum(1 for k in ver if k["verdict"] == "cannot-judge")
    w_all = wilson(g, nv)
    lo_all, hi_all = w_all["lo"], w_all["hi"]
    answered = g + ng
    if answered:
        w_ans = wilson(g, answered)
        lo_ans, hi_ans = w_ans["lo"], w_ans["hi"]
    else:
        lo_ans = hi_ans = None

    fg = sum(1 for k in foil if k["verdict"] == "genuine")
    fng = sum(1 for k in foil if k["verdict"] == "not-genuine")
    fcj = sum(1 for k in foil if k["verdict"] == "cannot-judge")

    sev_pairs = [(k["severity_rubric"], k["severity"]) for k in ver
                 if k["verdict"] == "genuine" and k["severity"] and k["severity_rubric"]]
    order = {"peripheral": 0, "supporting": 1, "critical": 2}
    exact = sum(1 for a, b in sev_pairs if a == b)
    adjacent = sum(1 for a, b in sev_pairs if abs(order[a] - order[b]) == 1)
    downgrades = sum(1 for a, b in sev_pairs if order[b] < order[a])
    upgrades = sum(1 for a, b in sev_pairs if order[b] > order[a])

    out = {
        "n_reached": n_reached, "n_verified": nv, "n_foils": nf,
        "stopping": "pre-declared prefix of the shuffled order",
        "precision": {
            "genuine": g, "not_genuine": ng, "cannot_judge": cj,
            "point_all_denominator": round(g / nv, 4),
            "wilson95_all": [round(lo_all, 4), round(hi_all, 4)],
            "clustered95_all": cluster_boot(ver, lambda k: k["verdict"] == "genuine"),
            "point_answered_only": round(g / answered, 4) if answered else None,
            "wilson95_answered_only": [round(lo_ans, 4), round(hi_ans, 4)] if answered else None,
            "n_consultations": len({k["consultation"] for k in ver}),
        },
        "foils": {"judged_genuine": fg, "judged_not_genuine": fng, "cannot_judge": fcj,
                  "agreement_with_refusal": round(fng / nf, 4) if nf else None},
        "severity": {"n_graded_pairs": len(sev_pairs), "exact": exact, "adjacent": adjacent,
                     "beyond_adjacent": len(sev_pairs) - exact - adjacent,
                     "physician_lower": downgrades, "physician_higher": upgrades,
                     "pairs": sev_pairs},
        "disagreements": [
            {"item": k["item"], "finding_id": k["finding_id"], "status": k["status"],
             "verdict": k["verdict"], "panel_decided_by": k["panel_decided_by"],
             "frame_tier2": k["frame_tier2"]}
            for k in key
            if (k["status"] == "verified" and k["verdict"] != "genuine")
            or (k["status"] == "refused_foil" and k["verdict"] != "not-genuine")],
    }
    with open(os.path.join(OUTDIR, "sitting_results.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    print(f"\nwrote {OUTDIR}/sitting_results.json")


if __name__ == "__main__":
    main()
