"""w2_secondfamily_smoke.py - the candidate-judge smoke for the second-family replication.

The paper's central asymmetry (commissions ~0.79-0.94 paired, omissions ~0.50-0.63)
rests on ONE judge family: openai/gpt-5.4. This smoke picks the family that will carry
the indicative replication, on the only two questions that can sink it before a paid run:

  1. Does the candidate PARSE? Both grid output contracts - `Verdict: PASS|FAIL` and
     `Score: 0-10` - through the grid's own parsers (W.parse_verdict / W.parse_score),
     on real transcripts and real notes, with the grid's own prompts unchanged.
  2. What will the full run COST? Measured prompt/completion/reasoning tokens per call
     from real judgements, projected onto the exact call counts the full run buys.

Five notes per candidate, deliberately spread across consultations and error types so a
single short transcript cannot flatter the token projection. Both designs on every note,
so the smoke is 10 judgements = 10 calls per candidate.

Everything paper-bound goes through the shared machinery: W.judge (which carries the
pre-registered single retry of an unparseable sample), W.aggregate_cell, and a Run()
context so the smoke writes its own manifest and lands in results/cost_ledger.jsonl
like every other run. Nothing here is a finding - it chooses an instrument.

    python3 judges/w2_secondfamily_smoke.py --models gemini,opus -y
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

EXPERIMENT = "w2-secondfamily"
# The two designs under replication, as (arm name, prompt stem, format, k). These mirror
# w2_arms.py's sf-* registry entries exactly; the smoke states them locally so it stays
# readable as a standalone instrument check.
DESIGNS = [("sf-F-bin-k1", "grid_F_bin", "bin", 1),
           ("sf-FC-score-k1", "grid_FC_score", "score", 1)]
PARSERS = {"bin": W.parse_verdict, "score": W.parse_score}

# What the full run buys, for the projection. Both designs are k=1 over every unique
# note, one replicate: 607 notes x 2 designs = 1,214 calls.
FULL_DESIGNS = 2
FULL_REPLICATES = 1


def pick_smoke_notes(blocks, n=5):
    """n notes from n DIFFERENT consultations, spread across STRATA and note roles.

    Cost per call is dominated by the transcript, and the four strata differ by ~50% in
    mean transcript length (aci ~6.4k chars, primock ~9.7k), so a selection drawn from
    one stratum projects a cost that is not the corpus's. Two rules keep the projection
    honest: rotate the strata, and inside a stratum take the consultation whose
    transcript is closest to that stratum's MEDIAN rather than the first by name.
    Deterministic - sorted candidates, fixed want-order, no RNG.
    """
    want = ["clean", "omit", "add", "change", "omit"]
    strata = sorted({b["stratum"] for b in blocks})
    med = {}
    for st in strata:
        lens = sorted(len(b["transcript"]) for b in blocks if b["stratum"] == st)
        med[st] = lens[len(lens) // 2]

    chosen, used = [], set()
    for i, role in enumerate(want[:n]):
        st = strata[i % len(strata)]
        cand = []
        for b in sorted(blocks, key=lambda b: b["consultation"]):
            if b["stratum"] != st or b["consultation"] in used:
                continue
            for note in sorted(b["notes"], key=lambda x: x["note_key"]):
                is_clean = note["note_role"] == "clean"
                if (role == "clean" and is_clean) or (not is_clean and note["pair_type"] == role):
                    cand.append((abs(len(b["transcript"]) - med[st]), b, note))
                    break
        if not cand:  # this stratum has no note of that role left - fall back to any stratum
            for b in sorted(blocks, key=lambda b: b["consultation"]):
                if b["consultation"] in used:
                    continue
                for note in sorted(b["notes"], key=lambda x: x["note_key"]):
                    is_clean = note["note_role"] == "clean"
                    if (role == "clean" and is_clean) or (not is_clean
                                                          and note["pair_type"] == role):
                        cand.append((abs(len(b["transcript"]) - med[b["stratum"]]), b, note))
                        break
        if not cand:
            continue
        _, b, note = min(cand, key=lambda c: (c[0], c[1]["consultation"]))
        chosen.append({"block": b, "note": note, "wanted": role})
        used.add(b["consultation"])
    return chosen[:n]


def smoke_one(model_key, notes, prompts, run):
    """Every design x every note on one candidate. Returns per-call rows."""
    role_cfg = W.ROLES[model_key]
    rows = []
    for arm, stem, fmt, k in DESIGNS:
        for i, sel in enumerate(notes):
            note, block = sel["note"], sel["block"]
            prompt = W.render(prompts[stem], transcript=block["transcript"], note=note["text"])
            base_seed = W.RUN_SEEDS[1] * 100000 + note["item_index"] * 10
            t0 = time.time()
            try:
                vals, texts, metas, retried = W.judge(
                    prompt, role_cfg["role"], k, base_seed, temperature=1.0,
                    reasoning_effort=role_cfg["reasoning_effort"], parser=PARSERS[fmt])
                err = None
            except Exception as ex:                      # a candidate that cannot be called
                rows.append({"model_key": model_key, "arm": arm, "note_key": note["note_key"],
                             "error": f"{type(ex).__name__}: {ex}"[:400], "parsed": False})
                print(f"  {model_key:<7} {arm:<15} {note['note_key'][:44]:<44} CALL FAILED: "
                      f"{type(ex).__name__}: {str(ex)[:120]}", flush=True)
                continue
            agg, verdict, flagged = W.aggregate_cell(fmt, k, vals)
            rcs = W.receipts(metas)
            tot = W.receipt_totals(rcs)
            row = {"model_key": model_key, "role": role_cfg["role"],
                   "model": metas[0]["model"], "arm": arm, "format": fmt, "k": k,
                   "note_key": note["note_key"], "note_role": note["note_role"],
                   "pair_type": note["pair_type"], "consultation": note["consultation"],
                   "wanted_role": sel["wanted"], "samples": vals, "aggregate": agg,
                   "verdict": verdict, "flagged": flagged, "parsed": agg is not None,
                   "retried_samples": retried, "totals": tot, "receipts": rcs,
                   "prompt_chars": len(prompt), "transcript_chars": len(block["transcript"]),
                   "raw_head": (texts[0] or "")[:400], "raw_len": len(texts[0] or ""),
                   "wall_s": round(time.time() - t0, 2), "error": None}
            rows.append(row)
            print(f"  {model_key:<7} {arm:<15} {note['note_key'][:44]:<44} "
                  f"{'OK ' if row['parsed'] else 'PARSE-FAIL'} agg={agg} "
                  f"pt={tot['prompt_tokens']} ct={tot['completion_tokens']} "
                  f"rt={tot['reasoning_tokens']} cached={tot['cached_tokens']} "
                  f"${tot['cost_usd']:.5f} {row['wall_s']}s", flush=True)
    return rows


def summarise(rows, n_notes_full, half_notes=None, length_factor=1.0):
    """Parse rate + measured cost per call, projected onto the full run's call counts.

    `length_factor` = (note-weighted corpus mean transcript chars) / (the smoke's mean).
    Even a stratum-rotated five-note draw cannot land exactly on the corpus mean, and
    transcript length is what buys prompt tokens, so every projection is reported twice:
    as measured, and scaled by this factor. The scaled figure is the one to plan against.
    """
    out = {}
    for model_key in sorted({r["model_key"] for r in rows}):
        mine = [r for r in rows if r["model_key"] == model_key]
        ok = [r for r in mine if r.get("parsed")]
        by_design = {}
        for arm, _, fmt, _ in DESIGNS:
            d = [r for r in mine if r["arm"] == arm]
            by_design[arm] = {
                "n": len(d), "parsed": sum(1 for r in d if r.get("parsed")),
                "call_failures": sum(1 for r in d if r.get("error")),
                "retried": sum(len(r.get("retried_samples") or []) for r in d if not r.get("error")),
                "aggregates": [r.get("aggregate") for r in d],
                "raw_lens": [r.get("raw_len") for r in d]}
        calls = sum(r["totals"]["calls"] for r in ok) or 0
        cost = sum(r["totals"]["cost_usd"] for r in ok)
        tok = {k: sum(r["totals"][k] for r in ok) for k in
               ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens")}
        per_call = (cost / calls) if calls else None
        # The full run is k=1 x 2 designs x every unique note. The smoke's per-call cost
        # is the honest projector: same prompts, same substrate, same settings. Prefix
        # caching can only make the real run cheaper, never dearer, so this is an upper
        # estimate on a warm cache and a fair one on a cold cache.
        full_calls = n_notes_full * FULL_DESIGNS * FULL_REPLICATES
        scaled = (per_call * length_factor) if per_call else None
        out[model_key] = {
            "smoke_calls": calls, "smoke_cost_usd": round(cost, 6),
            "cost_per_call_usd": round(per_call, 6) if per_call else None,
            "length_factor": round(length_factor, 4),
            "cost_per_call_usd_scaled": round(scaled, 6) if scaled else None,
            "mean_tokens_per_call": {k: round(v / calls, 1) for k, v in tok.items()} if calls else None,
            "parse_rate": round(len(ok) / len(mine), 4) if mine else None,
            "by_design": by_design,
            "projected_full_run": {
                "calls": full_calls,
                "usd": round(per_call * full_calls, 2) if per_call else None,
                "usd_scaled": round(scaled * full_calls, 2) if scaled else None},
        }
        if half_notes:
            # Optional extension: FC-score-k8 on a consultation-stratified half.
            k8_calls = half_notes * 8
            out[model_key]["projected_k8_half"] = {
                "notes": half_notes, "calls": k8_calls,
                "usd": round(per_call * k8_calls, 2) if per_call else None,
                "usd_scaled": round(scaled * k8_calls, 2) if scaled else None}
            out[model_key]["projected_total_usd"] = (
                round(per_call * (full_calls + k8_calls), 2) if per_call else None)
            out[model_key]["projected_total_usd_scaled"] = (
                round(scaled * (full_calls + k8_calls), 2) if scaled else None)
    return out


def main():
    ap = argparse.ArgumentParser(description="second-family candidate smoke")
    ap.add_argument("--models", default="gemini,opus",
                    help="comma list of w2_common.ROLES keys to smoke, in preference order")
    ap.add_argument("--notes", type=int, default=5, help="notes per candidate")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default="results/w2-secondfamily/smoke.json")
    ap.add_argument("--warn-usd", type=float, default=1.0)
    ap.add_argument("--stop-usd", type=float, default=2.0)
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in model_keys if m not in W.ROLES]
    if unknown:
        raise SystemExit(f"unknown model key(s) {unknown}. Known: {', '.join(sorted(W.ROLES))}")

    pairs, info = W.load_dataset(path=args.dataset)
    transcripts, tprov = W.transcript_index()
    blocks, notes_all = W.note_units(pairs, transcripts)
    picked = pick_smoke_notes(blocks, n=args.notes)
    prompts = W.load_prompts(sorted({stem for _, stem, _, _ in DESIGNS}))

    print(f"second-family smoke | candidates: {', '.join(model_keys)}")
    print(f"  dataset {info['pairs_file']} version={info['dataset_version']} "
          f"sha {info['sha256'][:12]} | {info['n_pairs']} eval pairs, {len(notes_all)} unique notes")
    for name, stem, fmt, k in DESIGNS:
        print(f"  design {name:<15} prompt {stem}.txt sha {W.sha256_text(prompts[stem])[:12]} "
              f"fmt={fmt} k={k}")
    print(f"  {len(picked)} notes: " + ", ".join(
        f"{p['note']['consultation']}/{p['note']['note_role']}"
        f"{'/' + p['note']['pair_type'] if p['note']['pair_type'] else ''}" for p in picked))
    n_calls = len(picked) * len(DESIGNS) * len(model_keys)
    print(f"  {n_calls} calls total ({len(picked) * len(DESIGNS)} per candidate)")
    W.confirm(f"proceed with ~{n_calls} smoke calls of OpenRouter credits?", args.yes)

    # Run-scoped guard: this smoke is capped at its own spend, not the project ledger's.
    guard = W.SpendGuard(args.warn_usd, args.stop_usd, scope="run")
    print(f"  spend guard: run scope, warn ${args.warn_usd:.2f} / stop ${args.stop_usd:.2f}")

    rows = []
    params = W.run_params(
        {"purpose": "second-family candidate smoke: parseability + cost projection",
         "candidates": model_keys,
         "roles": {m: W.ROLES[m] for m in model_keys},
         "designs": [{"arm": a, "prompt": s, "format": f, "k": k} for a, s, f, k in DESIGNS],
         "prompt_sha256": {s: W.sha256_text(p) for s, p in prompts.items()},
         "n_notes": len(picked), "max_tokens": W.MAX_TOKENS, "temperature": 1.0,
         "spend_guard": {"scope": "run", "warn_usd": args.warn_usd, "stop_usd": args.stop_usd},
         "notes_selected": [p["note"]["note_key"] for p in picked]}, info)
    with Run(EXPERIMENT, params=params, replicate=1, seed=W.RUN_SEEDS[1],
             inputs=[info["pairs_file"]], spec=W.SPEC, allow_dirty=True) as run:
        run.register_prompts(prompts)
        for model_key in model_keys:
            print(f"\n-- {model_key} ({W.ROLES[model_key]['role']} -> "
                  f"{__import__('common').resolve_model(W.ROLES[model_key]['role'])['resolved']}, "
                  f"reasoning_effort={W.ROLES[model_key]['reasoning_effort']!r})")
            got = smoke_one(model_key, picked, prompts, run)
            rows += got
            for r in got:
                guard.add((r.get("totals") or {}).get("cost_usd") or 0.0)

        half_notes = None
        # Stratum-balanced half of the consultations, the same construction the runner's
        # --consultation-half uses, so the k=8 projection counts the notes it would buy.
        import random as _random
        keep = []
        for st in sorted({b["stratum"] for b in blocks}):
            ids = sorted(b["consultation"] for b in blocks if b["stratum"] == st)
            rng = _random.Random(f"20260817|{st}")
            rng.shuffle(ids)
            keep += [(st, c) for c in ids[: len(ids) // 2 + len(ids) % 2]]
        keepset = set(keep)
        half_notes = sum(len(b["notes"]) for b in blocks
                         if (b["stratum"], b["consultation"]) in keepset)

        corpus_mean_tchars = sum(len(b["transcript"]) * len(b["notes"]) for b in blocks) \
            / max(1, len(notes_all))
        smoke_mean_tchars = sum(len(p["block"]["transcript"]) for p in picked) / len(picked)
        length_factor = corpus_mean_tchars / smoke_mean_tchars if smoke_mean_tchars else 1.0
        summary = summarise(rows, n_notes_full=len(notes_all), half_notes=half_notes,
                            length_factor=length_factor)
        out = {"dataset": {k: info[k] for k in ("pairs_file", "sha256", "dataset_version",
                                                "n_pairs")},
               "n_unique_notes": len(notes_all), "k8_half_notes": half_notes,
               "transcript_chars": {"corpus_note_weighted_mean": round(corpus_mean_tchars),
                                    "smoke_mean": round(smoke_mean_tchars),
                                    "length_factor": round(length_factor, 4)},
               "designs": [{"arm": a, "prompt": s, "sha256": W.sha256_text(prompts[s]),
                            "format": f, "k": k} for a, s, f, k in DESIGNS],
               "roles": {m: dict(W.ROLES[m],
                                 resolved=__import__('common').resolve_model(
                                     W.ROLES[m]["role"])["resolved"]) for m in model_keys},
               "notes": [{"note_key": p["note"]["note_key"], "role": p["note"]["note_role"],
                          "pair_type": p["note"]["pair_type"],
                          "consultation": p["note"]["consultation"],
                          "transcript_chars": len(p["block"]["transcript"])} for p in picked],
               "rows": rows, "summary": summary, "run_id": run.run_id}
        run.save("smoke.json", out)

    dest = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=1)

    print("\n" + "=" * 100)
    print(f"corpus note-weighted mean transcript {out['transcript_chars']['corpus_note_weighted_mean']} "
          f"chars vs smoke {out['transcript_chars']['smoke_mean']} chars -> "
          f"length factor {out['transcript_chars']['length_factor']}")
    print(f"{'candidate':<10} {'parse':<8} {'$/call':<10} {'mean pt/ct/rt':<22} "
          f"{'full run $':<12} {'+k8 half $':<12} total $   (scaled)")
    for model_key, s in summary.items():
        t = s["mean_tokens_per_call"] or {}
        per_call = f"{s['cost_per_call_usd']:.5f}" if s["cost_per_call_usd"] else "n/a"
        toks = (f"{t['prompt_tokens']:.0f}/{t['completion_tokens']:.0f}"
                f"/{t['reasoning_tokens']:.0f}") if t else "n/a"
        k8 = (s.get("projected_k8_half") or {}).get("usd_scaled")
        print(f"{model_key:<10} {str(s['parse_rate']):<8} {per_call:<10} {toks:<22} "
              f"{str(s['projected_full_run']['usd_scaled']):<12} {str(k8):<12} "
              f"{s.get('projected_total_usd_scaled')}")
        for arm, d in s["by_design"].items():
            print(f"    {arm:<16} parsed {d['parsed']}/{d['n']}  "
                  f"call-failures {d['call_failures']}  retried-samples {d['retried']}  "
                  f"aggregates {d['aggregates']}")
    print(f"\nsmoke spend this session: ${guard.spent_here:.4f}")
    print(f"written: {os.path.relpath(dest, HERE)}")


if __name__ == "__main__":
    main()
