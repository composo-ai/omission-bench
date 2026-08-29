"""gepa_report.py - read gepa/results.json + lineage.jsonl and print what happened.

    .venv/bin/python gepa/gepa_report.py            # the table the README quotes
    .venv/bin/python gepa/gepa_report.py --misses   # which examples the winner still fails
    .venv/bin/python gepa/gepa_report.py --tree     # the search tree, parent -> child

Every number in README.md comes out of here, so a reviewer can regenerate the table
rather than trust the prose.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results.json")
LINEAGE = os.path.join(HERE, "lineage.jsonl")

COLS = [("od_omit_complete", "omit-complete", 14), ("od_omit_partial", "omit-partial", 5),
        ("od_add", "add", 16), ("od_change", "change", 22), ("fa", "FA(clean)", 22),
        ("od_omit_severity_weighted", "omit sev-wt", 19)]


def pct(x):
    return "   -  " if x is None else f"{100 * x:5.1f}%"


def table(res, only=None):
    rows = res["candidates"]
    if only == "key":
        keep = {c["id"] for c in rows if c["kind"] == "seed"} | {res["winner"]["id"]}
        rows = [c for c in rows if c["id"] in keep]
    print(f"{'#':>3}  {'candidate':<18} {'from':<6} "
          + "  ".join(f"{lab:>12}" for _k, lab, _n in COLS) + "     J   feasible")
    print(f"{'':>3}  {'':<18} {'':<6} " + "  ".join(f"{'(n=%d)' % n:>12}" for _k, _l, n in COLS))
    for c in sorted(rows, key=lambda c: c["id"]):
        v = c["vec"]
        src = "seed" if c["parent"] is None else f"#{c['parent']}"
        star = " <- WINNER" if c["id"] == res["winner"]["id"] else ""
        print(f"{c['id']:>3}  {c['name']:<18} {src:<6} "
              + "  ".join(f"{pct(v[k]):>12}" for k, _l, _n in COLS)
              + f"  {v['J']:.3f}  {'yes' if v['feasible'] else 'NO ':<3}{star}")


def deltas(res):
    """Winner minus each seed, on the axes that matter."""
    w = res["winner"]["vec"]
    print("\nwinner vs each seed (percentage points):")
    for c in sorted(res["candidates"], key=lambda c: c["id"]):
        if c["kind"] != "seed":
            continue
        v = c["vec"]
        parts = []
        for k, lab, _n in COLS:
            if v[k] is None or w[k] is None:
                continue
            d = 100 * (w[k] - v[k])
            parts.append(f"{lab} {d:+.1f}")
        print(f"  vs {c['name']:<18} " + "  ".join(parts))


def misses(res):
    """Which dev examples the winner still gets wrong, and whether the seeds got them."""
    pe = res.get("per_example")
    if not pe:
        print("(no per-example detail in this results.json)")
        return
    wid, seeds = str(res["winner"]["id"]), [c for c in res["candidates"] if c["kind"] == "seed"]
    wsc = pe["scores"][wid]
    print(f"\nwinner (#{wid}) errors on the dev pool - score, then each seed's score:")
    for e in pe["examples"]:
        s = wsc.get(e["key"])
        if s is None:
            continue
        flagged, gold_err = s <= 7, e["kind"] != "clean"
        if flagged == gold_err:
            continue
        seed_s = "  ".join(f"{c['name'].replace('seed_', '')}={pe['scores'][str(c['id'])].get(e['key'])}"
                           for c in seeds)
        kind = "FALSE ALARM" if not gold_err else "MISS"
        print(f"  {kind:<11} {e['key']:<44} {e['kind']:<14} sev={str(e['severity']):<10} "
              f"winner={s}  {seed_s}")


def tree(res):
    print("\nsearch tree (accepted candidates only; rejects are in lineage.jsonl):")
    kids = {}
    for c in res["candidates"]:
        kids.setdefault(c["parent"], []).append(c)

    def walk(pid, depth):
        for c in sorted(kids.get(pid, []), key=lambda c: c["id"]):
            v = c["vec"]
            print("  " + "  " * depth + f"#{c['id']} {c['name']} (it{c['iter']}, {c['chars']}ch) "
                  f"omit-sw {pct(v['od_omit_severity_weighted'])} FA {pct(v['fa'])} J {v['J']:.3f}")
            walk(c["id"], depth + 1)
    walk(None, 0)


def lineage_stats():
    if not os.path.exists(LINEAGE):
        return
    ev = {}
    for line in open(LINEAGE):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev[r["event"]] = ev.get(r["event"], 0) + 1
    print("\nlineage: " + ", ".join(f"{k} {v}" for k, v in sorted(ev.items())))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="every candidate, not just seeds + winner")
    ap.add_argument("--misses", action="store_true")
    ap.add_argument("--tree", action="store_true")
    args = ap.parse_args()
    if not os.path.exists(RESULTS):
        sys.exit("no gepa/results.json - run gepa_optimize.py first")
    res = json.load(open(RESULTS))
    print(f"run {res['run_id']} | {res['n_candidates']} candidates ({res['n_feasible']} feasible) "
          f"| ${res['spend_usd']:.2f} | {res['wall_s']}s | stopped: {res['stopped_because']}")
    print(f"  student {res['student_model']} | reflection {res['reflection_model']} | "
          f"dev pool {res['provenance']['n_examples']} judgements / "
          f"{res['provenance']['n_consultations']} consultations | "
          f"eval overlap {res['provenance']['eval_overlap']}\n")
    table(res, None if args.all else "key")
    deltas(res)
    if args.misses:
        misses(res)
    if args.tree:
        tree(res)
    lineage_stats()


if __name__ == "__main__":
    main()
