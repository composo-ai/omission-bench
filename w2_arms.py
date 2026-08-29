"""w2_arms.py - THE registry of every judge design W2 compares, plus the runner for
the single-prompt ones (specs/w2-ablation-grid.md; post-review amendment 2026-08-12).

The grid's 8 cells are minimal pairs: they exist to separate criterion from format
from ensembling, and each differs from its neighbour by one thing. That makes them a
clean ablation and a WEAK field - nobody ships a one-line judge. So the comparison
also carries STRONG SYSTEMS, reported alongside rather than inside the ablation:
v14 (a hand-tuned production judge), three published baselines, an engineered
completeness judge with salience few-shots, and a slot for a GEPA-optimised prompt.

An arm is a small record:

    {name, prompt (a w2_prompts stem, a path, or a callable), role, k, format}

plus the bookkeeping a runner needs (parser, flag rule, which experiment store it
writes, how its prompt slots are named). Single-prompt arms - the grid cells, v14,
engineered, gepa - run through THIS file. Multi-call arms (G-Eval, RAGAS, checklist)
carry a `callable` pointing at their implementation in w2_baselines.py, which is the
runner that executes them; the registry still owns their identity so there is exactly
one list of what W2 compares.

    .venv/bin/python w2_arms.py --list                     # the registry, dry-run
    .venv/bin/python w2_arms.py --arms strong --plan       # what a run would buy
    .venv/bin/python w2_arms.py --arms strong --runs 3 -y  # run the strong arms
"""
import argparse, json, os, time

from common import HERE, Run
import w2_common as W

STRONG_EXPERIMENT = "w2-strong"
# Optional, may not exist yet: W-F publishes the GEPA-optimised judge prompt here.
GEPA_PROMPT = "gepa/judge_prompt.txt"
V14_SOURCE = os.path.join(
    HERE, "../../experiments/clinical-hallucination-detection/methods/v14_v9-none/prompt.txt")


# ---------------------------------------------------------------- parsers
def parse_score_json(text):
    """{"score": n, "omissions": [...]} -> n; falls back to a bare `Score: n` line."""
    j = W.parse_json_blob(text)
    if isinstance(j, dict) and j.get("score") is not None:
        try:
            v = int(round(float(j["score"])))
        except (TypeError, ValueError):
            return None
        return v if 0 <= v <= 10 else None
    return W.parse_score(text)


def parse_verdict_json(text):
    """{"verdict": "PASS"|"FAIL"} -> the verdict; falls back to a `Verdict:` line."""
    j = W.parse_json_blob(text)
    if isinstance(j, dict) and isinstance(j.get("verdict"), str):
        v = j["verdict"].strip().upper()
        if v in ("PASS", "FAIL"):
            return v
    return W.parse_verdict(text)


PARSERS = {"bin": W.parse_verdict, "score": W.parse_score, "yesno": W.parse_yesno,
           "score_json": parse_score_json, "bin_json": parse_verdict_json}


# ---------------------------------------------------------------- the registry
def _arm(name, family, prompt, fmt, k=1, role="gpt54", parser=None, experiment=None,
         runner=None, flag_rule=None, note=None, prompt_kind="w2_prompts",
         max_transcript_chars=None, optional=False, in_default=True):
    return {"name": name, "family": family, "prompt": prompt, "prompt_kind": prompt_kind,
            "callable": None, "format": fmt, "k": k, "role": role,
            "parser": parser or PARSERS[fmt], "experiment": experiment or STRONG_EXPERIMENT,
            "runner": runner or "w2_arms.py", "flag_rule": flag_rule, "note": note,
            "max_transcript_chars": max_transcript_chars, "optional": optional,
            "in_default": in_default}


def _registry():
    arms = {}

    # -- the ablation itself: 8 minimal-pair cells (run by w2_grid.py)
    for cell in W.CELLS:
        crit, fmt, k = W.parse_cell(cell)
        arms[cell] = _arm(
            cell, "grid", f"grid_{crit}_{fmt}", fmt, k=k, experiment="w2-ablation",
            runner="w2_grid.py",
            flag_rule=("FAIL verdict" if fmt == "bin" else
                       "score <= 7" if k == 1 else "winsorized mean < 8.0"),
            note=f"minimal pair: {'faithfulness-only' if crit == 'F' else 'faithful AND complete'} "
                 f"criterion, {'binary verdict' if fmt == 'bin' else '0-10 score'}, k={k}")

    # -- v14: a hand-tuned production judge and its two edits (run by w2_v14_arms.py)
    for name, prompt, kind, why in (
            ("v14-asis", V14_SOURCE, "path", "a colleague's hand-tuned faithfulness judge verbatim"),
            ("v14-noexcl", "v14_noexcl", "w2_prompts", "v14 minus the two omission-exclusion hunks"),
            ("v14-incl", "v14_incl", "w2_prompts", "v14-noexcl plus one affirmative omission bullet")):
        arms[name] = _arm(name, "v14", prompt, "bin", parser=W.parse_yesno,
                          experiment="w2-v14", runner="w2_v14_arms.py", prompt_kind=kind,
                          flag_rule="Verdict: YES", note=why, max_transcript_chars=40000)

    # -- published baselines: multi-call, implemented in w2_baselines.py
    for name, fmt, why in (
            ("geval", "score", "G-Eval CoT scoring, consistency + completeness, note = min"),
            ("ragas", "score", "RAGAS decomposition; coverage by KEYED per-fact verdicts"),
            ("checklist", "score", "instance checklist generated once per consultation, "
                                   "answered per note")):
        a = _arm(name, "baseline", None, fmt, experiment="w2-baselines",
                 runner="w2_baselines.py", prompt_kind="callable",
                 flag_rule="below full marks (runner convention)", note=why)
        a["callable"] = f"w2_baselines:arm_{name}"
        arms[name] = a

    # -- strong systems: the field the grid is measured against
    arms["engineered-completeness"] = _arm(
        "engineered-completeness", "strong", "engineered_completeness", "score",
        k=1, parser=parse_score_json, flag_rule="score <= 7",
        note="W3's judge design built out: shared completeness criterion + the severity "
             "decision procedure + 8 hand-authored salience few-shots (4 flag / 4 no-flag, "
             "including the partial-residual case) + a structured omissions list. Deliberately "
             "the strongest prompt-only judge we can write - an upper reference, not a "
             "minimal pair.")
    arms["engineered-completeness-k8"] = _arm(
        "engineered-completeness-k8", "strong", "engineered_completeness", "score",
        k=8, parser=parse_score_json, flag_rule="winsorized mean < 8.0", in_default=False,
        note="the engineered judge with the grid's k=8 winsorized ensemble on top; off by "
             "default (8x the calls), enable when the k=1 result justifies it")
    arms["gepa-optimized"] = _arm(
        "gepa-optimized", "strong", GEPA_PROMPT, "score", k=1, parser=parse_score_json,
        prompt_kind="path", flag_rule="score <= 7", optional=True,
        note=f"W-F's GEPA-optimised judge prompt ({GEPA_PROMPT}); skipped with a note while "
             "that file does not exist. Optimised on the GEPA dev pool only - never on "
             "evaluation consultations.")
    # -- second-family replication (2026-08-17). The grid's central asymmetry rests on
    # one judge family (openai/gpt-5.4). These arms re-run two of its cells on a
    # different family, on the SAME prompts (grid_F_bin / grid_FC_score, byte-identical -
    # the prompt stem is the same string the grid cell uses), the SAME substrate and the
    # SAME aggregation + flag rules, so the only moving part is the judge. Indicative,
    # not a gate: one replicate, seed 11. Their own experiment directory keeps them out
    # of the grid's stores and out of w2_analyze.py's default sweep - a different judge
    # in the ablation's store would read as more of the ablation.
    for name, cell in (("sf-F-bin-k1", "F-bin-k1"),
                       ("sf-FC-score-k1", "FC-score-k1"),
                       ("sf-FC-score-k8", "FC-score-k8")):
        crit, fmt, k = W.parse_cell(cell)
        arms[name] = _arm(
            name, "secondfamily", f"grid_{crit}_{fmt}", fmt, k=k,
            experiment="w2-secondfamily", runner="w2_arms.py", in_default=False,
            flag_rule=("FAIL verdict" if fmt == "bin" else
                       "score <= 7" if k == 1 else "winsorized mean < 8.0"),
            note=f"second-family replication of grid cell {cell}: identical prompt, "
                 f"substrate, aggregation and flag rule; judge family is the variable. "
                 f"Run with --model gemini (or --model opus). Off by default.")
    return arms


ARMS = _registry()
FAMILIES = ["grid", "v14", "baseline", "strong", "secondfamily"]


def prompt_path(arm):
    if arm["prompt_kind"] == "w2_prompts":
        return os.path.join(W.PROMPT_DIR, arm["prompt"] + ".txt")
    if arm["prompt_kind"] == "path":
        p = arm["prompt"]
        return p if os.path.isabs(p) else os.path.join(HERE, p)
    return None


def available(arm):
    """(bool, reason). An optional arm whose prompt file is absent is skipped, not fatal."""
    if arm["prompt_kind"] == "callable":
        return True, f"callable {arm['callable']} (run by {arm['runner']})"
    path = prompt_path(arm)
    if path and os.path.exists(path):
        return True, os.path.relpath(path, HERE)
    return False, f"prompt file missing: {os.path.relpath(path, HERE) if path else arm['prompt']}"


def load_arm_prompt(arm):
    ok, why = available(arm)
    if not ok:
        raise FileNotFoundError(why)
    return open(prompt_path(arm)).read()


def render_arm(text, transcript, note, max_transcript_chars=None):
    """Fill whichever of {transcript} / {note} / {summary} the arm's prompt uses.

    Literal replacement, never str.format - several prompts end with a literal JSON
    schema that str.format would read as a replacement field (w2_common.render).
    """
    if max_transcript_chars:
        transcript = transcript[:max_transcript_chars]
    out = text
    for token, value in (("{transcript}", transcript), ("{note}", note), ("{summary}", note)):
        if token in out:
            out = out.replace(token, value)
    for token in ("{transcript}", "{note}", "{summary}"):
        assert token not in out, f"unfilled slot {token}"
    return out


def select(spec, include_optional=False):
    """'all' | a family name | 'default' | a comma list of arm names -> [arm, ...]."""
    if spec in FAMILIES:
        chosen = [a for a in ARMS.values() if a["family"] == spec]
    elif spec == "all":
        chosen = list(ARMS.values())
    elif spec == "default":
        chosen = [a for a in ARMS.values() if a["in_default"]]
    else:
        names = [s.strip() for s in spec.split(",") if s.strip()]
        unknown = [n for n in names if n not in ARMS]
        if unknown:
            raise SystemExit(f"unknown arm(s) {unknown}. Known: {', '.join(sorted(ARMS))}")
        return [ARMS[n] for n in names]
    if not include_optional:
        chosen = [a for a in chosen if a["in_default"]]
    return sorted(chosen, key=lambda a: (FAMILIES.index(a["family"]), a["name"]))


def registry_table():
    rows = []
    for a in sorted(ARMS.values(), key=lambda a: (FAMILIES.index(a["family"]), a["name"])):
        ok, why = available(a)
        rows.append({"name": a["name"], "family": a["family"], "role": a["role"],
                     "k": a["k"], "format": a["format"], "runner": a["runner"],
                     "experiment": a["experiment"],
                     "prompt": a["callable"] or a["prompt"], "available": ok, "resolves_to": why,
                     "in_default_set": a["in_default"], "flag_rule": a["flag_rule"],
                     "note": a["note"]})
    return rows


# ---------------------------------------------------------------- runner
def main():
    ap = argparse.ArgumentParser(description="W2 arm registry + strong-arm runner")
    ap.add_argument("--list", action="store_true", help="print the registry and exit")
    ap.add_argument("--json", action="store_true", help="with --list: emit JSON")
    ap.add_argument("--plan", action="store_true", help="plan the run (no calls) and exit")
    ap.add_argument("--arms", default="strong",
                    help="'strong' | 'all' | 'default' | a family | a comma list of arm names")
    ap.add_argument("--include-optional", action="store_true",
                    help="also run arms that are off by default (e.g. the k=8 engineered arm)")
    ap.add_argument("--model", default="gpt54", choices=sorted(W.ROLES))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--run", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dataset", default=None,
                    help=f"dataset file (default: first of {', '.join(W.DATASET_CANDIDATES)})")
    ap.add_argument("--allow-unfrozen", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--order", default="note-major", choices=["note-major", "block-major"],
                    help="prefix-sharing unit (w2_common.sub_blocks' contract). note-major "
                         "(default, unchanged) holds one note's prefix warm across arms and "
                         "pays a serial warm-up call per note; block-major warms once per "
                         "consultation and fans the rest out - much faster when the judge "
                         "route caches nothing, which is measurable in the receipts")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--warn-usd", type=float, default=700.0)
    ap.add_argument("--stop-usd", type=float, default=900.0)
    ap.add_argument("--guard-scope", default="cumulative", choices=["cumulative", "run"],
                    help="what --warn-usd/--stop-usd measure: total project credits "
                         "(default, the original behaviour) or THIS run's credits")
    ap.add_argument("--consultation-half", type=int, default=None,
                    help="run a stratum-balanced HALF of the consultations, chosen by this "
                         "seed (deterministic); for k=8 extensions that cannot afford the "
                         "whole corpus")
    ap.add_argument("--authorize-spend", action="store_true")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    if args.list:
        rows = registry_table()
        if args.json:
            print(json.dumps(rows, indent=1))
        else:
            print(f"{len(rows)} arms registered\n")
            print(f"{'arm':<28} {'family':<9} {'k':>2} {'fmt':<6} {'runner':<18} avail  prompt")
            for r in rows:
                print(f"{r['name']:<28} {r['family']:<9} {r['k']:>2} {r['format']:<6} "
                      f"{r['runner']:<18} {'yes' if r['available'] else 'NO ':<5}  {r['prompt']}")
            miss = [r for r in rows if not r["available"]]
            for r in miss:
                print(f"\n  ! {r['name']}: {r['resolves_to']} - skipped with a note")
        return

    arms = select(args.arms, include_optional=args.include_optional)
    skipped = []
    runnable = []
    for a in arms:
        ok, why = available(a)
        if not ok:
            skipped.append((a, why))
        elif a["callable"]:
            skipped.append((a, f"multi-call arm - run it with {a['runner']}"))
        else:
            runnable.append(a)
    for a, why in skipped:
        print(f"  skipping {a['name']}: {why}")
    if not runnable:
        print("nothing runnable in this selection")
        return

    prompts = {a["name"]: load_arm_prompt(a) for a in runnable}
    W.load_prompts(quiet=True)  # populate the change-detector for the manifest

    pairs, info = W.load_dataset(path=args.dataset, allow_unfrozen=args.allow_unfrozen)
    if args.smoke:
        from w2_grid import pick_smoke_pairs
        pairs = pick_smoke_pairs(pairs, n=2)
    elif args.limit:
        pairs = sorted(pairs, key=lambda p: p["pair_id"])[:args.limit]
    info["n_pairs_selected"] = len(pairs)
    transcripts, tprov = W.transcript_index()
    blocks, notes = W.note_units(pairs, transcripts)

    if args.consultation_half is not None:
        # Stratified by `stratum`, halved WITHIN each stratum so the subset keeps the
        # corpus's source mix. Selection is on consultations, not pairs, so every note
        # of a kept consultation - including its clean twin - travels together and the
        # paired measure still has its own reference. item_index is left untouched
        # (it is derived from the full note set) so seeds match the full-corpus run.
        import random as _random
        keep = []
        for st in sorted({b["stratum"] for b in blocks}):
            ids = sorted(b["consultation"] for b in blocks if b["stratum"] == st)
            rng = _random.Random(f"{args.consultation_half}|{st}")
            rng.shuffle(ids)
            keep += [(st, c) for c in ids[: len(ids) // 2 + len(ids) % 2]]
        keepset = set(keep)
        blocks = [b for b in blocks if (b["stratum"], b["consultation"]) in keepset]
        notes = [n for b in blocks for n in b["notes"]]
        info["consultation_half"] = {
            "seed": args.consultation_half, "n_consultations": len(blocks),
            "n_notes": len(notes),
            "consultations": sorted(f"{s}|{c}" for s, c in keepset)}
        print(f"  consultation-half seed {args.consultation_half}: {len(blocks)} "
              f"consultations, {len(notes)} notes")

    role_cfg = W.ROLES[args.model]
    replicates = [args.run] if args.run else list(range(1, args.runs + 1))
    experiments = sorted({a["experiment"] for a in runnable})
    if len(experiments) > 1:
        # One Run writes one manifest into one experiment directory, and every record
        # traces to its manifest by (experiment, run_id). A selection spanning
        # experiments would leave records pointing at a manifest that is not under
        # their own experiment, which w2_analyze.py reads as a missing manifest.
        raise SystemExit(
            f"selection spans {len(experiments)} experiments ({', '.join(experiments)}).\n"
            "  Run one family at a time - the grid cells with w2_grid.py, the v14 arms with "
            "w2_v14_arms.py, the strong systems with `--arms strong`.")
    tag = args.tag or (f"strong-{args.model}-{info['sha256'][:8]}"
                       + ("-smoke" if (args.smoke or args.limit) else ""))
    stores = {e: W.RecordStore(e, tag) for e in experiments}

    def done(arm, rep, note):
        return stores[arm["experiment"]].has(f"{arm['name']}|r{rep}|{note['note_key']}")

    calls = sum(a["k"] for rep in replicates for b in blocks for a in runnable for n in b["notes"]
                if not done(a, rep, n))
    todo = sum(1 for rep in replicates for b in blocks for a in runnable for n in b["notes"]
               if not done(a, rep, n))
    print(f"w2 arms | {len(runnable)} arm(s): {', '.join(a['name'] for a in runnable)}")
    print(f"  dataset: {info['pairs_file']} version={info['dataset_version']} "
          f"sha {info['sha256'][:12]} | {len(pairs)} pairs, {len(notes)} unique notes")
    print(f"  {todo} judgements to run ({calls} calls) over {len(replicates)} replicate(s)")
    if args.plan:
        for a in runnable:
            print(f"    {a['name']:<28} k={a['k']} fmt={a['format']} -> "
                  f"results/{a['experiment']}/_state/{tag}.jsonl")
        return
    if not todo:
        print("nothing to do - every judgement is already in the record store")
        return
    W.confirm(f"proceed with ~{calls} calls of OpenRouter credits?", args.yes)
    guard = W.SpendGuard(args.warn_usd, args.stop_usd, args.authorize_spend,
                         scope=args.guard_scope)
    print(f"  spend guard: {args.guard_scope} scope, warn ${args.warn_usd:.0f} / "
          f"stop ${args.stop_usd:.0f} (ledger to date ${guard.ledger_at_start:.2f})")

    t0, made = time.time(), 0
    for rep in replicates:
        seed = W.RUN_SEEDS.get(rep, 11 * rep)
        pending = [b for b in blocks
                   if any(not done(a, rep, n) for a in runnable for n in b["notes"])]
        for ci, chunk in enumerate(W.chunked(pending, args.chunk), 1):
            params = W.run_params(
                {"arms": [a["name"] for a in runnable], "model_key": args.model,
                 "role": role_cfg["role"], "reasoning_effort": role_cfg["reasoning_effort"],
                 "max_tokens": W.MAX_TOKENS, "workers": args.workers, "chunk": ci,
                 "ordering": f"{args.order}-warmup",
                 "spend_guard": {"scope": args.guard_scope, "warn_usd": args.warn_usd,
                                 "stop_usd": args.stop_usd},
                 "arm_specs": [{k: v for k, v in a.items() if k != "parser"} for a in runnable],
                 "prompt_sha256": {n: W.sha256_text(p) for n, p in prompts.items()},
                 "record_stores": {e: os.path.relpath(s.path, HERE) for e, s in stores.items()}},
                info)
            W.assert_dataset_stable(info)
            with Run(experiments[0], params=params, replicate=rep, seed=seed,
                     inputs=[info["pairs_file"]], spec=W.SPEC, allow_dirty=True) as run:
                run.register_prompts(prompts)
                chunk_records = []

                def worker(job):
                    arm, note, block = job
                    prompt = render_arm(prompts[arm["name"]], block["transcript"], note["text"],
                                        arm["max_transcript_chars"])
                    base_seed = seed * 100000 + note["item_index"] * 10
                    vals, texts, metas, retried = W.judge(
                        prompt, role_cfg["role"], arm["k"], base_seed, temperature=1.0,
                        reasoning_effort=role_cfg["reasoning_effort"], parser=arm["parser"])
                    fmt = "bin" if arm["format"] == "bin" else "score"
                    agg, verdict, flagged = W.aggregate_cell(fmt, arm["k"], vals)
                    rcs = W.receipts(metas)
                    tot = W.receipt_totals(rcs)
                    rec = {"key": f"{arm['name']}|r{rep}|{note['note_key']}",
                           "cell": arm["name"], "arm": arm["name"], "arm_family": arm["family"],
                           "format": fmt, "k": arm["k"], "temperature": 1.0, "replicate": rep,
                           "run_seed": seed, "base_seed": base_seed, "model_key": args.model,
                           "role": role_cfg["role"], "model": metas[0]["model"],
                           "k_impl": metas[0]["k_impl"],
                           **W.note_fields(note),
                           "samples": vals, "aggregate": agg, "verdict": verdict,
                           "flagged": flagged, "parse_failure": agg is None,
                           "retried_samples": retried, "receipts": rcs, "totals": tot,
                           "prompt_chars": len(prompt), "note_chars": len(note["text"]),
                           "transcript_chars": len(block["transcript"]),
                           "run_id": run.run_id, "t": round(time.time(), 3),
                           "substrate_frozen": info["frozen"],
                           "dataset_version": info["dataset_version"]}
                    if arm["format"] == "score":  # keep the structured omissions list
                        j = W.parse_json_blob(texts[0])
                        if isinstance(j, dict) and isinstance(j.get("omissions"), list):
                            rec["omissions_reported"] = j["omissions"][:20]
                    stores[arm["experiment"]].put(rec)
                    chunk_records.append(rec)
                    guard.add(tot["cost_usd"])

                for b in chunk:
                    n_block = 0
                    ordered_notes = sorted(
                        b["notes"], key=lambda n: (n["note_role"] != "clean", n["note_key"]))
                    if args.order == "block-major":
                        # The whole consultation as ONE unit: warm up on the clean note,
                        # then fan every remaining (arm, note) job out across `workers`.
                        # Only worth choosing when the judge is not actually caching -
                        # note-major exists to hold a long shared prefix warm, and with
                        # few arms per note it costs a serial call per note to do it.
                        jobs = [(a, n, b) for n in ordered_notes for a in runnable
                                if not done(a, rep, n)]
                        jobs.sort(key=lambda j: (j[0]["k"], j[1]["note_role"] != "clean",
                                                 j[1]["note_key"], j[0]["name"]))
                        n_block += W.run_block(jobs, worker, args.workers)
                    else:
                        # note-major sub-blocks with a serial warm-up, exactly as the grid:
                        # every call for one note shares transcript + note + preamble.
                        for note in ordered_notes:
                            jobs = [(a, note, b) for a in runnable if not done(a, rep, note)]
                            jobs.sort(key=lambda j: (j[0]["k"], j[0]["name"]))
                            n_block += W.run_block(jobs, worker, args.workers)
                    made += n_block
                    print(f"  r{rep} chunk{ci} {b['stratum']}/{b['consultation']}: {n_block} "
                          f"judgements | credits ${guard.total:.2f} | {time.time() - t0:.0f}s",
                          flush=True)
                run.save("w2_arms.json", {"tag": tag, "arms": [a["name"] for a in runnable],
                                          "replicate": rep, "seed": seed, "params": params,
                                          "transcript_sources": tprov,
                                          "n_records": len(chunk_records),
                                          "records": chunk_records})
    for s in stores.values():
        s.close()
    print(f"\ndone: {made} judgements this session; credits ${guard.spent_here:.2f}")
    for e, s in stores.items():
        print(f"  {os.path.relpath(s.path, HERE)}: {len(s.records)} records")


if __name__ == "__main__":
    main()
