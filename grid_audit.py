#!/usr/bin/env python3
"""grid_audit.py - interim health + signal check on a running (or finished) W2 grid.

Written 2026-08-12 for the hourly audit the lead author asked for while the main grid runs.
Read-only: touches nothing the runner owns. Answers four questions in order of
how likely they are to justify killing the run:

  1. Is it mechanically sound?   parse failures, cell balance, retries, judged clean twins
  2. What is it costing?         measured $/judgement, projection to the full run, the rail
  3. Is the signal coherent?     flag rates on clean vs errored, per class and severity
  4. Is anything drifting?       per-replicate comparison once replicate 2 starts

Usage: python3 grid_audit.py [--store results/w2-ablation/_state/grid-main2.jsonl]
                             [--target 14568] [--rail-usd 355]
"""
import argparse, collections, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(store):
    rows = []
    with open(store) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(n, d):
    return f"{100*n/d:5.1f}%" if d else "    -"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="results/w2-ablation/_state/grid-main2.jsonl")
    ap.add_argument("--target", type=int, default=14568, help="judgements in the full run")
    ap.add_argument("--rail-usd", type=float, default=355.0,
                    help="the lead author's rail: one full run + 15%%; exceeding it needs his sign-off")
    a = ap.parse_args()

    store = a.store if os.path.isabs(a.store) else os.path.join(HERE, a.store)
    if not os.path.exists(store):
        sys.exit(f"no store at {store}")
    rows = load(store)
    if not rows:
        sys.exit("store is empty")

    print(f"=== 1. MECHANICS ===  {len(rows)}/{a.target} judgements ({pct(len(rows), a.target)})")
    pf = sum(1 for r in rows if r.get("parse_failure"))
    unusable = sum(1 for r in rows if r.get("flagged") is None and r.get("aggregate") is None)
    retried = sum(len(r.get("retried_samples") or []) for r in rows)
    print(f"  parse failures: {pf}   unusable records: {unusable}   retried samples: {retried}")
    cells = collections.Counter(r.get("cell") for r in rows)
    spread = (max(cells.values()) - min(cells.values())) if cells else 0
    print(f"  cells: {len(cells)} active, spread {spread} (a large spread means one cell is stalling)")
    roles = collections.Counter(r.get("note_role") for r in rows)
    print(f"  note roles judged: {dict(roles)}  <- 'clean' must be present or false alarms are unmeasurable")
    reps = collections.Counter(r.get("replicate") for r in rows)
    print(f"  replicates: {dict(sorted(reps.items()))}")

    print("\n=== 2. COST ===")
    tok = collections.Counter()
    for r in rows:
        t = r.get("totals") or {}
        for key in ("calls", "prompt_tokens", "cached_tokens", "completion_tokens"):
            tok[key] += t.get(key) or 0
    # Per-record cost is not always populated by the transport; fall back to the ledger,
    # filtered to this experiment, which is the number the spend guard itself reads.
    led = os.path.join(HERE, "results/cost_ledger.jsonl")
    grid_usd, grid_calls = 0.0, 0
    if os.path.exists(led):
        for line in open(led):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("experiment") == "w2-ablation" and (d.get("run_id") or "") >= "20260812-2213":
                grid_usd += d.get("cost_usd") or 0
                grid_calls += d.get("calls") or 0
    per_call = (grid_usd / grid_calls) if grid_calls else 0.00470  # smoke-measured fallback
    calls_per_judgement = (tok["calls"] / len(rows)) if rows else 4.5
    projected = per_call * calls_per_judgement * a.target
    print(f"  ledger (this grid): ${grid_usd:.2f} over {grid_calls} calls"
          + ("" if grid_calls else "   [no chunk banked yet - using smoke rate]"))
    print(f"  cache: {tok['cached_tokens']:,}/{tok['prompt_tokens']:,} prompt tokens = "
          f"{pct(tok['cached_tokens'], tok['prompt_tokens'])} (90% cheaper per cached token)")
    print(f"  ${per_call:.5f}/call x {calls_per_judgement:.1f} calls/judgement x {a.target} = "
          f"PROJECTED ${projected:.0f}")
    verdict = "OK" if projected <= a.rail_usd else "OVER RAIL - stop and ask the lead author"
    print(f"  rail ${a.rail_usd:.0f} (one full run + 15%): {verdict}")

    print("\n=== 3. SIGNAL ===  flag rate, clean vs errored (discrimination = errored - clean)")
    sig = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for r in rows:
        role = "clean" if r.get("note_role") == "clean" else "errored"
        s = sig[r.get("cell")][role]
        s[0] += 1 if r.get("flagged") else 0
        s[1] += 1
    print(f"  {'cell':13s} {'clean FA':>10s} {'errored':>10s} {'gap':>8s}")
    for cell in sorted(sig):
        cf, cn = sig[cell]["clean"]
        ef, en = sig[cell]["errored"]
        gap = (100*ef/en - 100*cf/cn) if (cn and en) else 0
        print(f"  {cell:13s} {pct(cf, cn):>10s} {pct(ef, en):>10s} {gap:7.1f}pp")

    # The central axis. Records written before the 2026-08-12 classifier fix carry the
    # coarse level "partial", so derive the fine grain from the surviving mention's
    # strength - the identical rule the dataset used, verified to reproduce all 148
    # graded partials. Without this the surface collapses to two levels.
    STRENGTH_TO_LEVEL = {"explicit": "partial-strong", "paraphrase": "partial-strong",
                         "partial": "partial-weak"}

    def level_of(r):
        lvl = r.get("residual_level")
        if lvl == "partial":
            return STRENGTH_TO_LEVEL.get(r.get("residual_strength"), "partial")
        return lvl

    surf = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("note_role") == "clean" or not r.get("pair_class"):
            continue
        s = surf[(level_of(r), r.get("severity"))]
        s[0] += 1 if r.get("flagged") else 0
        s[1] += 1
    levels = [l for l in ("complete", "partial-strong", "partial-weak") if any(k[0] == l for k in surf)]
    if levels:
        print("\n  THE SURFACE - detection by how much of the fact survives x how much it matters:")
        print(f"    {'':16s}" + "".join(f"{s:>14s}" for s in ("critical", "supporting", "peripheral")))
        for lvl in levels:
            cells = []
            for sev in ("critical", "supporting", "peripheral"):
                f, n = surf.get((lvl, sev), [0, 0])
                cells.append(f"{pct(f, n)} ({n:4d})" if n else f"{'-':>12s}")
            print(f"    {lvl:16s}" + "".join(f"{c:>14s}" for c in cells))

    byclass = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("note_role") == "clean":
            continue
        k = (r.get("pair_class"), r.get("severity"))
        byclass[k][0] += 1 if r.get("flagged") else 0
        byclass[k][1] += 1
    known = {k: v for k, v in byclass.items() if k[0]}
    if known:
        print("\n  detection by class x severity (all cells pooled):")
        for k in sorted(known, key=str):
            f, n = known[k]
            print(f"    {str(k[0]):14s} {str(k[1]):11s} {f:4d}/{n:4d} = {pct(f, n)}")
    else:
        print("\n  (class/severity detail appears once errored pairs are judged - "
              "clean twins are judged first within each note block)")

    if len(reps) > 1:
        print("\n=== 4. DRIFT ===  same cells across replicates should agree within noise")
        for cell in sorted(cells):
            line = []
            for rep in sorted(reps):
                sub = [r for r in rows if r.get("cell") == cell and r.get("replicate") == rep
                       and r.get("note_role") != "clean"]
                if sub:
                    line.append(f"r{rep} {pct(sum(1 for r in sub if r.get('flagged')), len(sub))}")
            if len(line) > 1:
                print(f"  {cell:13s} " + "   ".join(line))
    else:
        print("\n=== 4. DRIFT ===  single replicate so far - drift check starts at replicate 2")


if __name__ == "__main__":
    main()
