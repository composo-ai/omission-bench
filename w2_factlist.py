"""w2_factlist.py - the fact-list-in-prompt ablation: the missing cell of the mechanism triad.

Three designs now sit on the same substrate, and between them they separate two things that
§19 confounded:

  monolithic (grid FC-score-k1)   no fact list, one open 0-10 score
  THIS ARM                        the AUDITED fact list in-prompt, one open 0-10 score
  pipeline B2/B3                  the fact list AND a closed verdict per fact id

If having the list is what matters, this arm should climb toward the pipeline's ~0.75-0.80
paired omissions. If it stays in the monolithic band (~0.6), then the active ingredient is
the CLOSED per-fact output structure rather than the information, and the mechanism claim
closes.

Everything except the prompt is held fixed at the grid's settings: judge `openai/gpt-5.4`
(role judge-primary) at reasoning_effort none, temperature 1.0, max_tokens 1024, seed 11,
one replicate, and the grid's own k=1 score flag rule (score <= 7).

THE INSTRUMENT IS NOT REGENERATED. The fact list is read from the audit artifacts the
pipeline already bought (`results/w2-pipeline/_cache/audit_*.json`, the `facts` key = the
kept, severity-graded list B3's checker saw) and rendered by `w2_pipeline.facts_block`, the
same function the pipeline's own prompts use - so the block is byte-identical to what the
checker read, minus nothing. Severity is deliberately not shown, exactly as there.

PROMPT DIFF, stated plainly (`w2_prompts/grid_FC_score_factlist.txt` against
`grid_FC_score.txt`): two hunks. (1) a delimited `REFERENCE FACT LIST` block inserted
after the transcript, carrying `{facts_block}`; (2) one sentence appended to the opening
line telling the judge what that block is. The evaluation criterion and the entire answer
format - the `Score: <integer between 0 and 10>` contract and the 9-10 reservation - are
byte-identical to the grid's, verified in the manifest.

    .venv/bin/python w2_factlist.py --plan
    .venv/bin/python w2_factlist.py --limit 3 -y      # smoke
    .venv/bin/python w2_factlist.py -y
"""
import argparse, json, os, time

from common import HERE, RESULTS, Run
import w2_common as W
import w2_pipeline as P

EXPERIMENT = "w2-factlist"
ARM = "factlist-FC-score-k1"
PROMPT_NAME = "grid_FC_score_factlist"
GRID_PROMPT_NAME = "grid_FC_score"
DATASET = "master/arms_confirm_subset.json"
ROLE_KEY = "gpt54"           # the grid's judge, at the grid's settings
FLAG_RULE = "score <= 7"     # the grid's own k=1 score rule


def audited_facts(block):
    """The kept, severity-graded fact list for one consultation, from the pipeline's cache.

    Read-only by construction: if the artifact is absent we STOP rather than regenerate,
    because a regenerated list would be a different instrument and the comparison against
    B2/B3 would silently stop being like-for-like.
    """
    cache = P.StageCache("audit")
    key = P.cache_key(block)
    rec = cache.get(key)
    if not rec or not rec.get("facts"):
        raise SystemExit(
            f"no cached audit for {block['stratum']}/{block['consultation']} "
            f"({cache.path(key)}). This arm reuses the pipeline's audited lists and must "
            "never regenerate them - stopping.")
    return rec["facts"], rec


def main():
    ap = argparse.ArgumentParser(description="fact-list-in-prompt ablation")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--limit", type=int, default=None,
                    help="first N notes only (smoke); records go to a -smoke tag")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=25, help="consultation blocks per Run()")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--warn-usd", type=float, default=6.0)
    ap.add_argument("--stop-usd", type=float, default=10.0)
    ap.add_argument("--authorize-spend", action="store_true")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    prompts = W.load_prompts([PROMPT_NAME, GRID_PROMPT_NAME])
    # The ablation is only interpretable if the scoring contract is untouched.
    a, b = prompts[GRID_PROMPT_NAME], prompts[PROMPT_NAME]
    tail = "Evaluation criterion:"
    if a[a.index(tail):] != b[b.index(tail):]:
        raise SystemExit("the factlist prompt's criterion/answer-format tail differs from "
                         "grid_FC_score.txt - that would confound the ablation. Stopping.")

    pairs, info = W.load_dataset(path=args.dataset)
    transcripts, tprov = W.transcript_index()
    blocks, notes = W.note_units(pairs, transcripts)

    # Attach the audited list per consultation up-front, so a missing artifact stops the
    # run before any credit is spent rather than half way through.
    fact_meta = {}
    for blk in blocks:
        facts, rec = audited_facts(blk)
        blk["_facts"] = facts
        fact_meta[f"{blk['stratum']}|{blk['consultation']}"] = {
            "n_facts": len(facts), "audit_prompt_sha256": rec.get("prompt_sha256"),
            "audit_role": rec.get("role"), "audit_effort": rec.get("effort"),
            "by_severity": rec.get("by_severity")}

    if args.limit:
        flat = [n for blk in blocks for n in blk["notes"]]
        keep = {n["note_key"] for n in sorted(flat, key=lambda n: n["note_key"])[:args.limit]}
        blocks = [dict(blk, notes=[n for n in blk["notes"] if n["note_key"] in keep])
                  for blk in blocks]
        blocks = [blk for blk in blocks if blk["notes"]]
        notes = [n for blk in blocks for n in blk["notes"]]

    role_cfg = W.ROLES[ROLE_KEY]
    rep, seed = 1, W.RUN_SEEDS[1]
    tag = args.tag or ("factlist" + ("-smoke" if args.limit else ""))
    store = W.RecordStore(EXPERIMENT, tag)
    todo = [(blk, n) for blk in blocks for n in blk["notes"]
            if not store.has(f"{ARM}|r{rep}|{n['note_key']}")]

    print(f"w2 fact-list ablation | arm {ARM} | judge {role_cfg['role']} "
          f"effort={role_cfg['reasoning_effort']!r} temp=1.0 max_tokens={W.MAX_TOKENS} "
          f"seed={seed} rep={rep}")
    print(f"  dataset {info['pairs_file']} version={info['dataset_version']} "
          f"sha {info['sha256'][:12]} | {info['n_pairs']} pairs, by class {info['by_class']}")
    print(f"  {len(blocks)} consultations, {len(notes)} notes "
          f"({sum(1 for n in notes if n['note_role'] == 'clean')} clean); {len(todo)} to run")
    print(f"  prompt {PROMPT_NAME}.txt sha {W.sha256_text(b)[:12]} "
          f"(grid {GRID_PROMPT_NAME} sha {W.sha256_text(a)[:12]}); "
          f"criterion + answer format byte-identical to the grid's")
    nf = [m["n_facts"] for m in fact_meta.values()]
    print(f"  audited fact lists: {len(fact_meta)} consultations, "
          f"{sum(nf) / max(1, len(nf)):.1f} facts each (min {min(nf)}, max {max(nf)}), "
          f"read from {os.path.relpath(P.CACHE_DIR, HERE)} - never regenerated")
    print(f"  store: {os.path.relpath(store.path, HERE)}")
    if args.plan:
        print(f"  would buy ~{len(todo)} calls")
        return
    if not todo:
        print("nothing to do - every judgement is already in the record store")
        return
    W.confirm(f"proceed with ~{len(todo)} calls of OpenRouter credits?", args.yes)

    guard = W.SpendGuard(args.warn_usd, args.stop_usd, args.authorize_spend, scope="run")
    print(f"  spend guard: run scope, warn ${args.warn_usd:.0f} / stop ${args.stop_usd:.0f} "
          f"(ledger to date ${guard.ledger_at_start:.2f})")

    t0, made = time.time(), 0
    pending = [blk for blk in blocks
               if any(not store.has(f"{ARM}|r{rep}|{n['note_key']}") for n in blk["notes"])]
    for ci, chunk in enumerate(W.chunked(pending, args.chunk), 1):
        params = W.run_params(
            {"arm": ARM, "family": "factlist-ablation", "model_key": ROLE_KEY,
             "role": role_cfg["role"], "reasoning_effort": role_cfg["reasoning_effort"],
             "temperature": 1.0, "max_tokens": W.MAX_TOKENS, "k": 1,
             "replicate": rep, "seed": seed, "chunk": ci, "workers": args.workers,
             "prompt": PROMPT_NAME,
             "prompt_sha256": {n: W.sha256_text(p) for n, p in prompts.items()},
             "prompt_diff_vs_grid": (
                 "two hunks against grid_FC_score.txt: (1) a delimited REFERENCE FACT LIST "
                 "block carrying {facts_block} inserted after the transcript; (2) one "
                 "sentence appended to the opening line describing that block. The "
                 "evaluation criterion and the whole answer format are byte-identical, "
                 "asserted at launch."),
             "fact_list_source": (
                 "results/w2-pipeline/_cache/audit_*.json `facts` (the kept, severity-graded "
                 "list B3's checker saw), rendered by w2_pipeline.facts_block; severity is "
                 "NOT shown, exactly as in the pipeline's own prompts. Read-only: the runner "
                 "stops if an artifact is missing rather than regenerate it."),
             "fact_lists": fact_meta,
             "flag_rule": FLAG_RULE,
             "spend_guard": {"scope": "run", "warn_usd": args.warn_usd,
                             "stop_usd": args.stop_usd},
             "record_store": os.path.relpath(store.path, HERE)}, info)
        W.assert_dataset_stable(info)
        with Run(EXPERIMENT, params=params, replicate=rep, seed=seed,
                 inputs=[info["pairs_file"]], spec=W.SPEC, allow_dirty=True) as run:
            run.register_prompts({PROMPT_NAME: b})
            chunk_records = []

            def worker(job):
                blk, note = job
                prompt = W.render(prompts[PROMPT_NAME], transcript=blk["transcript"],
                                  note=note["text"],
                                  facts_block=P.facts_block(blk["_facts"]))
                base_seed = seed * 100000 + note["item_index"] * 10
                vals, texts, metas, retried = W.judge(
                    prompt, role_cfg["role"], 1, base_seed, temperature=1.0,
                    reasoning_effort=role_cfg["reasoning_effort"], parser=W.parse_score)
                agg, verdict, flagged = W.aggregate_cell("score", 1, vals)
                rcs = W.receipts(metas)
                tot = W.receipt_totals(rcs)
                rec = {"key": f"{ARM}|r{rep}|{note['note_key']}", "cell": ARM, "arm": ARM,
                       "arm_family": "factlist-ablation", "format": "score", "k": 1,
                       "temperature": 1.0, "replicate": rep, "run_seed": seed,
                       "base_seed": base_seed, "model_key": ROLE_KEY,
                       "role": role_cfg["role"], "model": metas[0]["model"],
                       "k_impl": metas[0]["k_impl"],
                       **W.note_fields(note),
                       "n_facts_in_prompt": len(blk["_facts"]),
                       "samples": vals, "aggregate": agg, "verdict": verdict,
                       "flagged": flagged, "parse_failure": agg is None,
                       "retried_samples": retried, "receipts": rcs, "totals": tot,
                       "prompt_chars": len(prompt), "note_chars": len(note["text"]),
                       "transcript_chars": len(blk["transcript"]),
                       "run_id": run.run_id, "t": round(time.time(), 3),
                       "substrate_frozen": info["frozen"],
                       "dataset_version": info["dataset_version"]}
                store.put(rec)
                chunk_records.append(rec)
                guard.add(tot["cost_usd"])

            for blk in chunk:
                jobs = [(blk, n) for n in sorted(
                    blk["notes"], key=lambda n: (n["note_role"] != "clean", n["note_key"]))
                    if not store.has(f"{ARM}|r{rep}|{n['note_key']}")]
                n_block = W.run_block(jobs, worker, args.workers)
                made += n_block
                print(f"  chunk{ci} {blk['stratum']}/{blk['consultation']}: {n_block} "
                      f"judgements | credits ${guard.spent_here:.2f} | "
                      f"{time.time() - t0:.0f}s", flush=True)

            run.save("w2_factlist.json",
                     {"tag": tag, "arm": ARM, "replicate": rep, "seed": seed,
                      "params": params, "transcript_sources": tprov,
                      "n_records": len(chunk_records), "records": chunk_records})

    store.close()
    pf = sum(1 for r in store.records.values() if r.get("parse_failure"))
    print(f"\ndone: {made} judgements this session, {len(store.records)} in the store "
          f"({pf} parse failures)")
    print(f"credits: ${guard.spent_here:.2f} this session")
    print(f"record store: {os.path.relpath(store.path, HERE)}")


if __name__ == "__main__":
    main()
