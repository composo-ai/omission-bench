"""w2_v14_arms.py - the three v14 auxiliary arms of the judge benchmark.

Three arms, same items, same metrics, same caching and resume as the grid:

  v14-asis    the deployed hand-tuned faithfulness judge verbatim (w2_prompts/v14_as_shipped.txt)
  v14-noexcl  identical bar the two omission-exclusion hunks (w2_prompts/v14_noexcl.txt)
  v14-incl    v14-noexcl plus one affirmative omission bullet (w2_prompts/v14_incl.txt)

Together they separate "the explicit exclusion causes the blindness" from "v14's
note-centric structure causes it". Verdict parses from the `Verdict: YES|NO` line;
YES = flagged, so a pair discriminates iff errored=YES and clean=NO.

v14's prompt puts its ~4.9k-token instruction block BEFORE the transcript, so the
cacheable shared prefix here is global rather than per-consultation - block
ordering still helps (the transcript extends the prefix) but the floor is higher.

    python3 judges/w2_v14_arms.py --smoke --allow-unfrozen -y
    python3 judges/w2_v14_arms.py --runs 3 -y
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import argparse, os, time

from common import HERE, Run
import w2_common as W

#: The deployed judge's prompt as shipped. The study read it from the internal
#: project it belongs to; this copy is byte-identical and hashed in PROMPTS.sha256.
V14_SOURCE = os.path.join(HERE, "w2_prompts", "v14_as_shipped.txt")
ARMS = ["v14-asis", "v14-noexcl", "v14-incl"]


def load_arm_prompts(arms):
    """All three wordings are frozen w2_prompts files. The study read v14-asis from
    the deployed judge's own source rather than from w2_prompts/, so the runs hashed it
    into their manifests instead of checking it against PROMPTS.sha256; the released copy
    is byte-identical and is in PROMPTS.sha256 as well."""
    frozen = W.load_prompts()
    out = {}
    for a in arms:
        if a == "v14-asis":
            out[a] = open(V14_SOURCE).read()
        else:
            out[a] = frozen[a.replace("-", "_")]
        for slot in ("{transcript}", "{summary}"):
            assert slot in out[a], f"{a}: prompt lost its {slot} slot"
    return out


def main():
    ap = argparse.ArgumentParser(description="Judge-benchmark v14 auxiliary arms")
    ap.add_argument("--model", default="gpt54", choices=sorted(W.ROLES))
    ap.add_argument("--arms", default="all", help="'all' or a comma list of " + ",".join(ARMS))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--run", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dataset", default=None,
                    help=f"dataset file (default: first of {', '.join(W.DATASET_CANDIDATES)})")
    ap.add_argument("--allow-unfrozen", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--warn-usd", type=float, default=700.0)
    ap.add_argument("--stop-usd", type=float, default=900.0)
    ap.add_argument("--authorize-spend", action="store_true")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    if args.allow_unfrozen and not (args.smoke or args.limit):
        raise SystemExit("--allow-unfrozen is a smoke-only gate: pass --smoke or --limit too")

    arms = ARMS if args.arms == "all" else [a.strip() for a in args.arms.split(",") if a.strip()]
    prompts = load_arm_prompts(arms)

    pairs, info = W.load_dataset(path=args.dataset, allow_unfrozen=args.allow_unfrozen)
    if args.smoke:
        from w2_grid import pick_smoke_pairs
        pairs = pick_smoke_pairs(pairs, n=2)
    elif args.limit:
        pairs = sorted(pairs, key=lambda p: p["pair_id"])[:args.limit]
    info["n_pairs_selected"] = len(pairs)
    transcripts, tprov = W.transcript_index()
    blocks, notes = W.note_units(pairs, transcripts)

    role_cfg = W.ROLES[args.model]
    replicates = [args.run] if args.run else list(range(1, args.runs + 1))
    tag = args.tag or (f"v14-{args.model}-{info['sha256'][:8]}"
                       + ("-smoke" if (args.smoke or args.limit) else ""))
    store = W.RecordStore("w2-v14", tag)

    todo = [(rep, b, a, n) for rep in replicates for b in blocks for a in arms
            for n in b["notes"] if not store.has(f"{a}|r{rep}|{n['note_key']}")]
    print(f"v14 arms | {len(arms)} arms x {len(replicates)} replicates x {len(notes)} notes "
          f"= {len(arms) * len(replicates) * len(notes)} judgements; {len(todo)} to run")
    if not info["frozen"]:
        print("  ! PRE-AUDIT SUBSTRATE - smoke only")
    if not todo:
        print("nothing to do")
        return
    W.confirm(f"proceed with ~{len(todo)} calls?", args.yes)
    guard = W.SpendGuard(args.warn_usd, args.stop_usd, args.authorize_spend)

    t0 = time.time()
    for rep in replicates:
        seed = W.RUN_SEEDS.get(rep, 11 * rep)
        pending = [b for b in blocks if any(not store.has(f"{a}|r{rep}|{n['note_key']}")
                                            for a in arms for n in b["notes"])]
        for ci, chunk in enumerate(W.chunked(pending, args.chunk), 1):
            params = W.run_params(
                {"arms": arms, "model_key": args.model, "role": role_cfg["role"],
                 "reasoning_effort": role_cfg["reasoning_effort"], "chunk": ci,
                 "v14_source": os.path.relpath(V14_SOURCE, HERE),
                 "prompt_sha256": {a: W.sha256_text(p) for a, p in prompts.items()},
                 "record_store": os.path.relpath(store.path, HERE)}, info)
            W.assert_dataset_stable(info)
            with Run("w2-v14", params=params, replicate=rep, seed=seed,
                     inputs=[info["pairs_file"]], spec=W.SPEC, allow_dirty=True) as run:
                run.register_prompts(prompts)
                chunk_records = []

                def worker(job):
                    arm, note, block = job
                    prompt = (prompts[arm].replace("{transcript}", block["transcript"][:40000])
                              .replace("{summary}", note["text"]))
                    base_seed = seed * 100000 + note["item_index"] * 10
                    vals, texts, metas, retried = W.judge(
                        prompt, role_cfg["role"], 1, base_seed, temperature=1.0,
                        reasoning_effort=role_cfg["reasoning_effort"], parser=W.parse_yesno)
                    flagged = None if vals[0] is None else (vals[0] == "YES")
                    rcs = W.receipts(metas)
                    tot = W.receipt_totals(rcs)
                    rec = {"key": f"{arm}|r{rep}|{note['note_key']}", "cell": arm, "arm": arm,
                           "format": "bin", "k": 1, "temperature": 1.0, "replicate": rep,
                           "run_seed": seed, "base_seed": base_seed, "model_key": args.model,
                           "role": role_cfg["role"], "model": metas[0]["model"],
                           **W.note_fields(note),
                           "samples": vals, "aggregate": None if flagged is None else (0.0 if flagged else 10.0),
                           "verdict": None if flagged is None else ("FAIL" if flagged else "PASS"),
                           "flagged": flagged, "parse_failure": vals[0] is None,
                           "retried_samples": retried, "receipts": rcs, "totals": tot,
                           "prompt_chars": len(prompt), "run_id": run.run_id,
                           "t": round(time.time(), 3), "substrate_frozen": info["frozen"],
                           "dataset_version": info["dataset_version"]}
                    store.put(rec)
                    chunk_records.append(rec)
                    guard.add(tot["cost_usd"])

                for b in chunk:
                    jobs = [(a, n, b) for n in b["notes"] for a in arms
                            if not store.has(f"{a}|r{rep}|{n['note_key']}")]
                    jobs.sort(key=lambda j: (j[1]["note_role"] != "clean", j[0]))
                    W.run_block(jobs, worker, args.workers)
                    print(f"  r{rep} chunk{ci} {b['stratum']}/{b['consultation']}: {len(jobs)} "
                          f"judgements | credits ${guard.total:.2f} | {time.time() - t0:.0f}s",
                          flush=True)
                run.save("w2_v14.json", {"tag": tag, "arms": arms, "replicate": rep,
                                         "seed": seed, "params": params,
                                         "transcript_sources": tprov,
                                         "n_records": len(chunk_records),
                                         "records": chunk_records})
    store.close()
    print(f"\ndone: {len(store.records)} records; credits ${guard.spent_here:.2f} this session")


if __name__ == "__main__":
    main()
