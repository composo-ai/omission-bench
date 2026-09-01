"""w2_power.py - the judge-power test: does test-time compute close the monolithic gap?

The one objection the study had not
tested: every grid cell ran at `reasoning_effort="none"` and `max_tokens=1024`, while the
arms that won - RAGAS-style decomposition and the pipeline - spend more tokens across more
calls. If deliberation is the active ingredient, the paper's task-structure claim is partly
a compute artifact. The same cap applied INSIDE the GEPA searches: all three campaigns
executed candidate prompts at none/1024 (only the reflection model got high/16k), so a
candidate that said "enumerate from the transcript first" was never given the tokens to do
it. This runner buys the dose-response the study is missing, on the same substrate every
mechanism number already lives on.

Arms (all `openai/gpt-5.4` via role judge-primary, temp 1.0, k=1, seed 11, one replicate,
the grid's own `score <= 7` flag rule, the confirmation substrate - 151 held-out pairs
+ 47 clean twins, 198 notes):

  power-FC-med       grid_FC_score, effort medium, 12k cap    <- any deliberation at all
  power-FC-high      grid_FC_score, effort high, 12k cap      <- the dose-response endpoint
  power-factlist-high  grid_FC_score_factlist, effort high    <- the strongest case for it: the
                       (the audited list in-prompt)              list AND room to think
  power-gepa-high    gepa/judge_prompt.txt (v1 winner), high  <- was GEPA's "enumerate
                                                                 first" instruction merely
                                                                 token-starved?

Matched effort-none controls all exist already and are NEVER re-run: FC-score-k1
(grid-main2, 3 reps), the fact-list arm (w2-factlist), gepa-optimized (w2-strong
arms-main, subset-restricted) - the ONLY thing that moves per comparison is reasoning
effort + token budget. Prompts are byte-identical to their controls, asserted at launch
by sha. Fact lists come from the pipeline's own audit cache via w2_factlist.audited_facts
(read-only; a missing artifact stops the run).

    python3 judges/w2_power.py --plan
    python3 judges/w2_power.py --limit 3 -y          # smoke, records to power-smoke
    python3 judges/w2_power.py -y
    python3 judges/w2_power.py --arms power-FC-high -y
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import argparse, json, os, time

from common import HERE, Run
import w2_common as W
import w2_pipeline as P
from w2_arms import parse_score_json
from w2_factlist import audited_facts

EXPERIMENT = "w2-power"
DATASET = "master/arms_confirm_subset.json"
ROLE_KEY = "gpt54"
FLAG_RULE = "score <= 7"
POWER_MAX_TOKENS = 12000
GEPA_PROMPT_PATH = os.path.join("gepa", "judge_prompt.txt")

# name -> (prompt source, effort, needs fact list, parser, control it is the powered twin of)
ARMS = {
    "power-FC-med": {
        "prompt_name": "grid_FC_score", "effort": "medium", "facts": False,
        "parser": W.parse_score,
        "control": "FC-score-k1 @ none/1024 (grid-main2, 3 reps)"},
    "power-FC-high": {
        "prompt_name": "grid_FC_score", "effort": "high", "facts": False,
        "parser": W.parse_score,
        "control": "FC-score-k1 @ none/1024 (grid-main2, 3 reps)"},
    "power-factlist-high": {
        "prompt_name": "grid_FC_score_factlist", "effort": "high", "facts": True,
        "parser": W.parse_score,
        "control": "factlist-FC-score-k1 @ none/1024 (w2-factlist, 1 rep)"},
    "power-gepa-high": {
        "prompt_name": GEPA_PROMPT_PATH, "prompt_kind": "path", "effort": "high",
        "facts": False, "parser": parse_score_json,
        "control": "gepa-optimized @ none/1024 (w2-strong arms-main, subset-restricted)"},
    # The attribution control for power-gepa-high: gepa-04 is a mutation of seed_engineered,
    # so the hand-written parent at the same effort separates "the optimiser's mutations
    # become executable with tokens" from "reasoning does this for any strong prompt".
    "power-engineered-high": {
        "prompt_name": "engineered_completeness", "effort": "high", "facts": False,
        "parser": parse_score_json,
        "control": "engineered-completeness @ none/1024 (w2-strong arms-main, "
                   "subset-restricted)"},
}


def load_prompt(arm):
    if arm.get("prompt_kind") == "path":
        return open(os.path.join(HERE, arm["prompt_name"])).read()
    return W.load_prompts([arm["prompt_name"]])[arm["prompt_name"]]


def main():
    ap = argparse.ArgumentParser(description="judge-power test (reasoning effort + token cap)")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--experiment", default=EXPERIMENT,
                    help="results/<experiment>/ for records and manifests; keeps a non-benchmark arm out of the paper's power run")
    ap.add_argument("--subset-spec", default=None,
                    help="opt-in JSON: judge only the listed note_keys, and merge in any transcripts the benchmark index lacks (see w2_common.load_subset_spec). Used by the real-error arm to point these judges at census notes; off by default")
    ap.add_argument("--arms", default="all", help="comma-separated arm names, or 'all'")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N notes only (smoke); records go to a -smoke tag")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=25, help="consultation blocks per Run()")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--replicate", type=int, default=1, choices=sorted(W.RUN_SEEDS),
                    help="replicate number; sets the run seed from the fixed per-replicate seed table")
    ap.add_argument("--role", default=ROLE_KEY, choices=sorted(W.ROLES),
                    help="models.lock role key for the judge (default gpt54; 'gemini' runs "
                         "the same powered arm on the second family - reasoning effort still "
                         "comes from the arm, so 'high' is sent, which that endpoint accepts)")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--warn-usd", type=float, default=50.0)
    ap.add_argument("--stop-usd", type=float, default=80.0)
    ap.add_argument("--authorize-spend", action="store_true")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    arm_names = list(ARMS) if args.arms == "all" else \
        [a.strip() for a in args.arms.split(",") if a.strip()]
    for n in arm_names:
        if n not in ARMS:
            raise SystemExit(f"unknown arm {n!r}; known: {', '.join(ARMS)}")

    prompts = {}
    for n in arm_names:
        prompts[n] = load_prompt(ARMS[n])

    # The comparison is only interpretable if each powered prompt is byte-identical to its
    # effort-none control's prompt. grid + factlist prompts are loaded from the same files
    # the control runs read; the GEPA prompt is the file its own control ran. Assert
    # the factlist prompt still shares the grid's criterion/answer tail (as w2_factlist did).
    if "power-factlist-high" in arm_names:
        a = W.load_prompts(["grid_FC_score"])["grid_FC_score"]
        b = prompts["power-factlist-high"]
        tail = "Evaluation criterion:"
        if a[a.index(tail):] != b[b.index(tail):]:
            raise SystemExit("factlist prompt tail drifted from grid_FC_score.txt - stopping.")

    pairs, info = W.load_dataset(path=args.dataset)
    transcripts, tprov = W.transcript_index()
    subset = W.load_subset_spec(args.subset_spec)
    if subset:
        transcripts.update(subset["transcripts"])
    blocks, notes = W.note_units(pairs, transcripts)
    blocks, notes = W.apply_subset_spec(subset, blocks, notes)

    need_facts = any(ARMS[n]["facts"] for n in arm_names)
    fact_meta = {}
    if need_facts:
        for blk in blocks:
            facts, rec = audited_facts(blk)
            blk["_facts"] = facts
            fact_meta[f"{blk['stratum']}|{blk['consultation']}"] = {
                "n_facts": len(facts), "audit_prompt_sha256": rec.get("prompt_sha256")}

    if args.limit:
        flat = [n for blk in blocks for n in blk["notes"]]
        keep = {n["note_key"] for n in sorted(flat, key=lambda n: n["note_key"])[:args.limit]}
        blocks = [dict(blk, notes=[n for n in blk["notes"] if n["note_key"] in keep])
                  for blk in blocks]
        blocks = [blk for blk in blocks if blk["notes"]]
        notes = [n for blk in blocks for n in blk["notes"]]

    role_cfg = W.ROLES[args.role]
    rep = args.replicate
    seed = W.RUN_SEEDS[rep]
    tag = args.tag or ("power" + ("-smoke" if args.limit else ""))
    store = W.RecordStore(args.experiment, tag)
    todo = [(an, blk, n) for an in arm_names for blk in blocks for n in blk["notes"]
            if not store.has(f"{an}|r{rep}|{n['note_key']}")]

    print(f"judge-power test | arms {arm_names} | judge {role_cfg['role']} "
          f"temp=1.0 max_tokens={POWER_MAX_TOKENS} seed={seed} rep={rep}")
    print(f"  dataset {info['pairs_file']} version={info['dataset_version']} "
          f"sha {info['sha256'][:12]} | {info['n_pairs']} pairs, by class {info['by_class']}")
    print(f"  {len(blocks)} consultations, {len(notes)} notes "
          f"({sum(1 for n in notes if n['note_role'] == 'clean')} clean); "
          f"{len(todo)} judgements to buy")
    for n in arm_names:
        print(f"  {n:<22} effort={ARMS[n]['effort']:<7} prompt {ARMS[n]['prompt_name']} "
              f"sha {W.sha256_text(prompts[n])[:12]} | control: {ARMS[n]['control']}")
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

    t0 = time.time()
    made = 0
    for an in arm_names:
        arm, aprompt = ARMS[an], prompts[an]
        pending = [blk for blk in blocks
                   if any(not store.has(f"{an}|r{rep}|{n['note_key']}") for n in blk["notes"])]
        if not pending:
            print(f"\n{an}: complete in store, skipping")
            continue
        print(f"\n=== arm {an} (effort {arm['effort']}, {len(pending)} consultations) ===")
        for ci, chunk in enumerate(W.chunked(pending, args.chunk), 1):
            params = W.run_params(
                {"arm": an, "family": "judge-power", "model_key": args.role,
                 "role": role_cfg["role"],
                 "reasoning_effort": arm["effort"], "temperature": 1.0,
                 "max_tokens": POWER_MAX_TOKENS, "k": 1,
                 "replicate": rep, "seed": seed, "chunk": ci, "workers": args.workers,
                 "prompt": arm["prompt_name"],
                 "prompt_sha256": {arm["prompt_name"]: W.sha256_text(aprompt)},
                 "powered_twin_of": arm["control"],
                 "what_moved": "reasoning_effort none->" + arm["effort"] +
                               f", max_tokens 1024->{POWER_MAX_TOKENS}; prompt, model, "
                               "temperature, seed, substrate, flag rule all held",
                 "fact_lists": fact_meta if arm["facts"] else None,
                 "flag_rule": FLAG_RULE,
                 "spend_guard": {"scope": "run", "warn_usd": args.warn_usd,
                                 "stop_usd": args.stop_usd},
                 "record_store": os.path.relpath(store.path, HERE)}, info)
            W.assert_dataset_stable(info)
            with Run(args.experiment, params=params, replicate=rep, seed=seed,
                     inputs=[info["pairs_file"]], spec=W.SPEC, allow_dirty=True) as run:
                run.register_prompts({arm["prompt_name"]: aprompt})
                chunk_records = []

                def worker(job):
                    blk, note = job
                    slots = {"transcript": blk["transcript"], "note": note["text"]}
                    if arm["facts"]:
                        slots["facts_block"] = P.facts_block(blk["_facts"])
                    prompt = W.render(aprompt, **slots)
                    base_seed = seed * 100000 + note["item_index"] * 10
                    vals, texts, metas, retried = W.judge(
                        prompt, role_cfg["role"], 1, base_seed, temperature=1.0,
                        reasoning_effort=arm["effort"], parser=arm["parser"],
                        max_tokens=POWER_MAX_TOKENS)
                    agg, verdict, flagged = W.aggregate_cell("score", 1, vals)
                    rcs = W.receipts(metas)
                    tot = W.receipt_totals(rcs)
                    rec = {"key": f"{an}|r{rep}|{note['note_key']}", "cell": an, "arm": an,
                           "arm_family": "judge-power", "format": "score", "k": 1,
                           "temperature": 1.0, "replicate": rep, "run_seed": seed,
                           "base_seed": base_seed, "model_key": args.role,
                           "role": role_cfg["role"], "model": metas[0]["model"],
                           "k_impl": metas[0]["k_impl"],
                           "reasoning_effort": arm["effort"],
                           "max_tokens": POWER_MAX_TOKENS,
                           **W.note_fields(note),
                           "n_facts_in_prompt": len(blk["_facts"]) if arm["facts"] else None,
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

                # Fan out across the whole chunk, not per consultation: a consultation
                # holds only ~2-6 notes, so a per-consultation pool left most workers
                # idle. Records and seeds are per note, so this does not change them.
                jobs = [(blk, n) for blk in chunk
                        for n in sorted(blk["notes"],
                                        key=lambda n: (n["note_role"] != "clean",
                                                       n["note_key"]))
                        if not store.has(f"{an}|r{rep}|{n['note_key']}")]
                n_block = W.run_block(jobs, worker, args.workers)
                made += n_block
                rts = [r.get("totals", {}).get("reasoning_tokens", 0)
                       for r in chunk_records]
                print(f"  {an} chunk{ci} ({len(chunk)} consultations): "
                      f"{n_block} judgements | reasoning tokens/call "
                      f"~{sum(rts) / max(1, len(rts)):.0f} | "
                      f"credits ${guard.spent_here:.2f} | {time.time() - t0:.0f}s",
                      flush=True)

                run.save("w2_power.json",
                         {"tag": tag, "arm": an, "replicate": rep, "seed": seed,
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
