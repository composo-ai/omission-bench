"""w2_grid.py - the 8-cell 2x2x2 ablation runner (specs/w2-ablation-grid.md section 5.3).

Criterion (F | FC) x output format (bin | score) x ensemble (k=1 | k=8), all on one
pinned judge, 3 replicates (seeds 11/22/33). Iterates PAIR-MAJOR inside
per-consultation blocks so every call sharing a transcript prefix runs adjacently
(the cache-harder mandate - see w2_common's ordering contract).

Idempotent per (cell, replicate, note) with chunked resume: completed judgements
land in results/w2-ablation/_state/<tag>.jsonl and are skipped on restart, so a
killed process resumes without re-buying a single call. Each chunk of blocks is
its own Run() - manifests and per-call logs stay honest about what each one bought.

    # smoke (2 pairs, all 8 cells, 3 replicates, real calls, ~$1-3)
    .venv/bin/python w2_grid.py --smoke --allow-unfrozen -y

    # the full grid, once pairs_master_frozen.json exists
    .venv/bin/python w2_grid.py --model gpt54 --runs 3 -y
"""
import argparse, json, os, time

from common import HERE, RESULTS, Run
import w2_common as W


def cell_prompt_name(cell):
    crit, fmt, _ = W.parse_cell(cell.split("@")[0])
    return f"grid_{crit}_{fmt}"


def pick_smoke_pairs(pairs, n=2):
    """Deterministic, representative smoke selection: n pairs from n different
    consultations, spread across error types and (where present) strata, so the
    measured token/cost numbers are not all from one transcript length."""
    strata = sorted({p["stratum"] for p in pairs})
    types = ["omit", "add", "change"]
    chosen, used = [], set()
    for i in range(n):
        want_t, want_s = types[i % len(types)], strata[i % len(strata)]
        cand = sorted((p for p in pairs
                       if p["type"] == want_t and p["stratum"] == want_s
                       and (p["stratum"], p["id"]) not in used),
                      key=lambda p: p["pair_id"])
        if not cand:
            cand = sorted((p for p in pairs if (p["stratum"], p["id"]) not in used),
                          key=lambda p: p["pair_id"])
        if not cand:
            break
        chosen.append(cand[0])
        used.add((cand[0]["stratum"], cand[0]["id"]))
    return chosen


def main():
    ap = argparse.ArgumentParser(description="W2 2x2x2 ablation grid runner")
    ap.add_argument("--model", default="gpt54", choices=sorted(W.ROLES))
    ap.add_argument("--cells", default="all",
                    help="'all', 'generality' (4 k=1 + FC-score-k8), or a comma list")
    ap.add_argument("--runs", type=int, default=3, help="number of replicates to run")
    ap.add_argument("--run", type=int, default=None, help="run ONLY this replicate")
    ap.add_argument("--limit", type=int, default=None, help="first N pairs only (smoke)")
    ap.add_argument("--smoke", action="store_true",
                    help="2 representative pairs x all cells x --runs replicates")
    ap.add_argument("--temp0-supplement", action="store_true",
                    help="ALSO run the 4 k=1 criterion x format combos at temp 0 (appendix)")
    ap.add_argument("--dataset", default=None,
                    help=f"dataset file (default: first of {', '.join(W.DATASET_CANDIDATES)})")
    ap.add_argument("--allow-unfrozen", action="store_true",
                    help="smoke only: read the pre-audit construction pair file")
    ap.add_argument("--workers", type=int, default=12,
                    help="concurrent JUDGEMENTS; k=8 judgements fan out up to 8 calls each")
    ap.add_argument("--chunk", type=int, default=25, help="consultation blocks per Run()")
    ap.add_argument("--experiment", default="w2-ablation",
                    help="results/<experiment>/ for records and manifests. The default is the paper's grid; point a non-benchmark arm somewhere else, because w2_analyze.load_records globs every _state/*.jsonl under w2-ablation and would otherwise pool foreign records into the grid")
    ap.add_argument("--subset-spec", default=None,
                    help="opt-in JSON: judge only the listed note_keys, and merge in any transcripts the benchmark index lacks (see w2_common.load_subset_spec). Used by the real-error arm to point these judges at census notes; off by default")
    ap.add_argument("--order", default="note-major", choices=["note-major", "block-major"],
                    help="prefix-sharing unit: note-major caches best (see w2_common.sub_blocks)")
    ap.add_argument("--tag", default=None, help="override the record-store tag")
    ap.add_argument("--warn-usd", type=float, default=700.0)
    ap.add_argument("--stop-usd", type=float, default=900.0)
    ap.add_argument("--authorize-spend", action="store_true",
                    help="proceed past the hard spend tripwire (fresh authorization)")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    if args.allow_unfrozen and not (args.smoke or args.limit):
        raise SystemExit("--allow-unfrozen is a smoke-only gate: pass --smoke or --limit too")

    # ---- substrate
    pairs, info = W.load_dataset(path=args.dataset, allow_unfrozen=args.allow_unfrozen)
    if args.smoke:
        pairs = pick_smoke_pairs(pairs, n=2)
    elif args.limit:
        pairs = sorted(pairs, key=lambda p: p["pair_id"])[:args.limit]
    info["n_pairs_selected"] = len(pairs)
    transcripts, tprov = W.transcript_index()
    subset = W.load_subset_spec(args.subset_spec)
    if subset:
        transcripts.update(subset["transcripts"])
    blocks, notes = W.note_units(pairs, transcripts)
    blocks, notes = W.apply_subset_spec(subset, blocks, notes)

    # ---- cells + prompts (frozen, loaded from disk, sha-verified)
    if args.cells == "all":
        cells = list(W.CELLS)
    elif args.cells == "generality":
        cells = list(W.GENERALITY_CELLS)
    else:
        cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    if args.model != "gpt54" and args.cells == "all":
        cells = list(W.GENERALITY_CELLS)  # W2 section 2: generality rows are k=1 + the recipe cell
    if args.temp0_supplement:
        cells += [f"{c}@t0" for c in ("F-bin-k1", "F-score-k1", "FC-bin-k1", "FC-score-k1")]
    for c in cells:
        W.parse_cell(c.split("@")[0])
    prompts = W.load_prompts(sorted({cell_prompt_name(c) for c in cells}))

    role_cfg = W.ROLES[args.model]
    replicates = [args.run] if args.run else list(range(1, args.runs + 1))
    tag = args.tag or (f"grid-{args.model}-{info['sha256'][:8]}"
                       + ("-smoke" if (args.smoke or args.limit) else ""))
    store = W.RecordStore(args.experiment, tag)

    # ---- plan + spend
    jobs_total, calls_total = 0, 0
    for rep in replicates:
        for b in blocks:
            for c in cells:
                for n in b["notes"]:
                    jobs_total += 1
                    calls_total += W.parse_cell(c.split("@")[0])[2]
    done_now = sum(1 for rep in replicates for b in blocks for c in cells for n in b["notes"]
                   if store.has(f"{c}|r{rep}|{n['note_key']}"))
    print(f"w2 grid | model={args.model} ({role_cfg['role']}) | dataset={info['pairs_file']} "
          f"version={info['dataset_version']} sha {info['sha256'][:12]}")
    print(f"  classes: {info['by_class']} | severity: {info['by_severity']}")
    print(f"  {len(pairs)} pairs -> {len(blocks)} consultation blocks, {len(notes)} unique notes "
          f"({sum(1 for n in notes if n['note_role'] == 'clean')} clean)")
    print(f"  {len(cells)} cells x {len(replicates)} replicates = {jobs_total} judgements "
          f"({calls_total} calls); {done_now} already done, {jobs_total - done_now} to run")
    print(f"  ordering: {args.order} sub-blocks, warm-up call then {args.workers} workers "
          f"(up to {args.workers * 8} requests in flight on k=8 cells)")
    if not info["frozen"]:
        print("  ! PRE-AUDIT SUBSTRATE - smoke only, these numbers are not paper-bound")
    if jobs_total == done_now:
        print("nothing to do - every judgement is already in the record store")
        return
    W.confirm(f"proceed with ~{calls_total - done_now * 1} calls of OpenRouter credits?", args.yes)

    guard = W.SpendGuard(warn=args.warn_usd, stop=args.stop_usd, authorized=args.authorize_spend)
    print(f"  credits spent to date: ${guard.baseline:.2f} "
          f"(warn ${args.warn_usd:.0f} / stop ${args.stop_usd:.0f})")

    t0 = time.time()
    made = 0
    for rep in replicates:
        seed = W.RUN_SEEDS.get(rep, 11 * rep)
        pending_blocks = [b for b in blocks
                          if any(not store.has(f"{c}|r{rep}|{n['note_key']}")
                                 for c in cells for n in b["notes"])]
        if not pending_blocks:
            print(f"replicate {rep}: already complete")
            continue
        for ci, chunk in enumerate(W.chunked(pending_blocks, args.chunk), 1):
            params = W.run_params(
                {"cells": cells, "model_key": args.model, "role": role_cfg["role"],
                 "reasoning_effort": role_cfg["reasoning_effort"], "max_tokens": W.MAX_TOKENS,
                 "workers": args.workers, "ordering": f"{args.order}-warmup",
                 "chunk": ci, "chunk_blocks": [b["consultation"] for b in chunk],
                 "record_store": os.path.relpath(store.path, HERE)}, info)
            inputs = [info["pairs_file"]] + [f for f in ("authored_scenarios.json",)
                                             if os.path.exists(os.path.join(HERE, f))]
            W.assert_dataset_stable(info)
            with Run(args.experiment, params=params, replicate=rep, seed=seed, inputs=inputs,
                     spec=W.SPEC, allow_dirty=True) as run:
                run.register_prompts(prompts)
                chunk_records = []

                def worker(job):
                    cell, note, block = job
                    base = cell.split("@")[0]
                    crit, fmt, k = W.parse_cell(base)
                    temp = 0.0 if cell.endswith("@t0") else 1.0
                    prompt = W.render(prompts[cell_prompt_name(cell)],
                                      transcript=block["transcript"], note=note["text"])
                    base_seed = seed * 100000 + note["item_index"] * 10
                    parser = W.parse_score if fmt == "score" else W.parse_verdict
                    vals, texts, metas, retried = W.judge(
                        prompt, role_cfg["role"], k, base_seed, temperature=temp,
                        reasoning_effort=role_cfg["reasoning_effort"], parser=parser)
                    agg, verdict, flagged = W.aggregate_cell(fmt, k, vals)
                    rcs = W.receipts(metas)
                    tot = W.receipt_totals(rcs)
                    rec = {
                        "key": f"{cell}|r{rep}|{note['note_key']}", "cell": cell,
                        "criterion": crit, "format": fmt, "k": k, "temperature": temp,
                        "replicate": rep, "run_seed": seed, "base_seed": base_seed,
                        "model_key": args.model, "role": role_cfg["role"],
                        "model": metas[0]["model"], "k_impl": metas[0]["k_impl"],
                        **W.note_fields(note),
                        "samples": vals, "aggregate": agg, "verdict": verdict,
                        "flagged": flagged, "parse_failure": agg is None,
                        "retried_samples": retried, "receipts": rcs, "totals": tot,
                        "prompt_chars": len(prompt), "note_chars": len(note["text"]),
                        "transcript_chars": len(block["transcript"]),
                        "run_id": run.run_id, "t": round(time.time(), 3),
                        "substrate_frozen": info["frozen"],
                        "dataset_version": info["dataset_version"],
                    }
                    store.put(rec)
                    chunk_records.append(rec)
                    guard.add(tot["cost_usd"])

                for b in chunk:
                    # pair-major within the block; k=1 jobs first so the warm-up call
                    # that primes the shared prefix is a single request.
                    n_block = 0
                    for sub in W.sub_blocks(b, cells, args.order,
                                            lambda c, n: store.has(f"{c}|r{rep}|{n['note_key']}")):
                        n_block += W.run_block(sub, worker, args.workers)
                    made += n_block
                    print(f"  r{rep} chunk{ci} {b['stratum']}/{b['consultation']}: "
                          f"{n_block} judgements | credits ${guard.total:.2f} "
                          f"| {time.time() - t0:.0f}s", flush=True)

                run.save(f"w2_grid_{args.model}.json",
                         {"tag": tag, "cells": cells, "replicate": rep, "seed": seed,
                          "params": params, "transcript_sources": tprov,
                          "n_records": len(chunk_records), "records": chunk_records})

    store.close()
    print(f"\ndone: {made} judgements this session, {len(store.records)} in the store")
    print(f"credits: ${guard.spent_here:.2f} this session, ${guard.total:.2f} cumulative")
    print(f"record store: {os.path.relpath(store.path, HERE)}")


if __name__ == "__main__":
    main()
