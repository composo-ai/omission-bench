#!/usr/bin/env python3
"""arms_audit.py - decision-grade read on the judge arms from a stratified smoke.

Written 2026-08-12 to answer the lead author's question directly: not just what the full arms run
would cost, but whether each arm is WORKING - so the scale-up call is obvious rather
than a guess. Four questions per arm:

  1. Does it produce usable output at all?  (RAGAS silently returned misaligned verdicts
     before the keyed rewrite; a baseline that cannot parse is not a baseline)
  2. Does it discriminate?                  flag rate on errored notes minus clean notes
  3. Is it useful where it matters?         omission detection, split by how much of the
     fact survives and how much it matters
  4. What does the full run cost?           measured, then projected

Reads the smoke stores read-only. Usage: python3 arms_audit.py [--tag armsmoke]
"""
import argparse, collections, glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
STORES = ["results/w2-strong/_state", "results/w2-v14/_state", "results/w2-baselines/_state"]
FULL_PAIRS, SMOKE_PAIRS = 495, 38


def pct(n, d):
    return f"{100*n/d:5.1f}%" if d else "    -"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="armsmoke")
    ap.add_argument("--replicates", type=int, default=3, help="replicates the full run would use")
    a = ap.parse_args()

    rows = []
    for d in STORES:
        for path in glob.glob(os.path.join(HERE, d, f"*{a.tag}*.jsonl")):
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no records tagged {a.tag!r} - are the smokes still running?")

    arms = sorted({r.get("arm") or r.get("cell") for r in rows})
    print(f"{len(rows)} judgements over {len(arms)} arms, {SMOKE_PAIRS} stratified pairs\n")

    print("=== 1-2. HEALTH AND DISCRIMINATION ===")
    print(f"  {'arm':28s} {'usable':>8s} {'cleanFA':>8s} {'errored':>8s} {'gap':>8s}  verdict")
    summary = {}
    for arm in arms:
        sub = [r for r in rows if (r.get("arm") or r.get("cell")) == arm]
        usable = sum(1 for r in sub if not r.get("parse_failure")
                     and (r.get("flagged") is not None or r.get("aggregate") is not None))
        cl = [r for r in sub if r.get("note_role") == "clean"]
        er = [r for r in sub if r.get("note_role") != "clean"]
        cf = sum(1 for r in cl if r.get("flagged"))
        ef = sum(1 for r in er if r.get("flagged"))
        gap = (100*ef/len(er) - 100*cf/len(cl)) if (cl and er) else 0
        if usable < len(sub):
            v = "BROKEN - parse failures"
        elif gap < 5:
            v = "no discrimination"
        elif cl and 100*cf/len(cl) > 40:
            v = "discriminates but very noisy"
        else:
            v = "working"
        summary[arm] = (gap, v)
        print(f"  {arm:28s} {pct(usable, len(sub)):>8s} {pct(cf, len(cl)):>8s} "
              f"{pct(ef, len(er)):>8s} {gap:7.1f}pp  {v}")

    print("\n=== 3. WHERE IT MATTERS: omission detection ===")
    cells = ["omit-complete", "omit-partial"]
    sevs = ["critical", "supporting", "peripheral"]
    print(f"  {'arm':28s} " + " ".join(f"{c.split('-')[1][:4]}/{s[:4]:5s}" for c in cells for s in sevs))
    for arm in arms:
        sub = [r for r in rows if (r.get("arm") or r.get("cell")) == arm
               and r.get("note_role") != "clean"]
        line = []
        for c in cells:
            for s in sevs:
                g = [r for r in sub if r.get("pair_class") == c and r.get("severity") == s]
                line.append(f"{pct(sum(1 for r in g if r.get('flagged')), len(g)):>10s}")
        print(f"  {arm:28s} " + " ".join(line))

    print("\n=== 4. COST ===")
    led = os.path.join(HERE, "results/cost_ledger.jsonl")
    spend = collections.Counter()
    calls = collections.Counter()
    if os.path.exists(led):
        for line in open(led):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            exp = d.get("experiment") or ""
            if exp in ("w2-strong", "w2-v14", "w2-baselines") and (d.get("run_id") or "") >= "20260812-22":
                spend[exp] += d.get("cost_usd") or 0
                calls[exp] += d.get("calls") or 0
    total = sum(spend.values())
    scale = (FULL_PAIRS / SMOKE_PAIRS) * a.replicates
    for exp in sorted(spend):
        print(f"  {exp:16s} ${spend[exp]:6.2f} over {calls[exp]:5d} calls")
    print(f"  {'SMOKE TOTAL':16s} ${total:6.2f}")
    print(f"\n  full arms run = smoke x {FULL_PAIRS}/{SMOKE_PAIRS} pairs x {a.replicates} replicates "
          f"= x{scale:.1f}  ->  PROJECTED ${total*scale:.0f}")
    print("  (caching improves with scale - repeated notes across arms - so this is an upper bound)")

    print("\n=== RECOMMENDATION INPUTS ===")
    broken = [a_ for a_, (g, v) in summary.items() if v.startswith("BROKEN")]
    flat = [a_ for a_, (g, v) in summary.items() if v == "no discrimination"]
    print(f"  arms working: {len(summary)-len(broken)-len(flat)}/{len(summary)}")
    if broken:
        print(f"  BROKEN (fix before scaling): {', '.join(broken)}")
    if flat:
        print(f"  no discrimination (keep as an honest negative, or drop): {', '.join(flat)}")


if __name__ == "__main__":
    main()
