#!/usr/bin/env python3
"""W-E step 8 (spec Section 5) - two-layer verification of the RoSE injected pairs.

Pre-registered method: specs/w-e-cross-domain-generalization.md
  Section 2.2 - "(1) algorithmic single-edit check, W-A's exact acceptance constants
                 (MAX_UNITS=2, MAX_REGIONS=2, MAX_CHANGED_WORDS=30) applied to sentence-unit
                 diffs of the summary text, with an explicit recalibration checkpoint after the
                 first 10 pairs ...; (2) an LLM semantic check (WE_EDIT_CHECK), adapted
                 field-for-field from W-A's EDIT_CHECK, majority-of-3 seeded runs"
  Section 3.5 - regeneration reuses W-A's REGEN_SUFFIX pattern, max 3 attempts, then the
                deterministic fallback W-A uses for stubborn omit pairs (programmatic deletion of
                the unit carrying the target ACU) and hand-written edits for stubborn add/change

W-A's helpers are IMPORTED, not reimplemented (spec 2.4): units_of, word_diff_stats,
unit_diff_text, algo_check_v1 (W-A's frozen v1 rule = the constants W-E pre-registers) and
algo_check (W-A's post-2026-08-10 v2 rule). BOTH are computed for every pair; --algo-rule selects
which one gates. Default v1 = the text W-E actually pre-registers. See we_common's module
docstring for the divergence and why it is a decision for the review, not for this script.

Seeds: spec section 4 pins "seeds 71/72/73 (majority of 3)" for WE_EDIT_CHECK on the Claude plan
path. `claude -p` exposes no seed parameter, so majority-of-3 is implemented as 3 independent
calls - the same substitution W-B Amendment A3 makes for k=8 where a route cannot pass n through.
Recorded per pair as seed_impl.

Claude plan path only - NO judge-arm call anywhere in this file.

Usage
  python3 we_verify_pairs.py [--limit N] [--workers 4] [--dry-run] [--algo-rule v1|v2]
                             [--max-regen 3] [--out-suffix _shapecheck]
"""

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import we_common as W
from we_injections import inject_one, parse_inject_instr, regen_extra

MAX_REGEN = 3                    # spec 3.5
N_SEM_CALLS = 3                  # spec section 4: majority of 3
SEM_SEEDS = (71, 72, 73)         # spec section 4 (recorded; the plan path exposes no seed param)
LOAD_BEARING = ("single_edit", "type_match", "is_error", "truly_absent")


def sem_prompt(prompts, pair, doc):
    return W.render(
        prompts["we_edit_check"],
        typ=pair["type"],
        change=pair.get("change") or "",
        what=pair.get("what") or "",
        diff=W.unit_diff_text(pair["clean"], pair["errored"]),
        errored=pair["errored"],
        acu_list=W.acu_list_block(doc["reference_acus"]),
        article=doc["source"],
    )


def sem_majority(results, typ):
    """Majority over the 3 calls on each load-bearing field; fields that are null-by-type
    (is_error for omit, truly_absent for add/change) are skipped, matching W-A's rule."""
    verdict, detail = True, {}
    for field in LOAD_BEARING:
        if typ == "omit" and field == "is_error":
            continue
        if typ != "omit" and field == "truly_absent":
            continue
        votes = [bool(r.get(field)) for r in results if isinstance(r, dict) and field in r
                 and r.get(field) is not None]
        if len(votes) < 2:
            detail[field] = {"votes": votes, "passed": False, "reason": "fewer than 2 usable votes"}
            verdict = False
            continue
        passed = sum(votes) > len(votes) / 2
        detail[field] = {"votes": votes, "passed": passed}
        verdict = verdict and passed
    acu_votes = [r.get("acu_idx") for r in results
                 if isinstance(r, dict) and isinstance(r.get("acu_idx"), int) and r["acu_idx"] >= 0]
    detail["acu_idx"] = Counter(acu_votes).most_common(1)[0][0] if acu_votes else None
    detail["reasons"] = [r.get("reason") for r in results if isinstance(r, dict) and r.get("reason")]
    return verdict, detail


def programmatic_omit(clean, target_acu):
    """W-A's deterministic stubborn-omit fallback: drop the sentence unit that best carries the
    target ACU (highest word overlap), add nothing in its place."""
    units = W.units_of(clean)
    if not units or not target_acu:
        return None
    tgt = set(target_acu.lower().split())
    best, best_score = None, 0.0
    for i, u in enumerate(units):
        uw = set(u.lower().split())
        score = len(tgt & uw) / max(len(tgt), 1)
        if score > best_score:
            best, best_score = i, score
    if best is None or best_score == 0:
        return None
    return "\n".join(u for i, u in enumerate(units) if i != best)


def verify_one(prompts, instr, docs, pair, args):
    doc = docs[pair["doc_id"]]
    rec = {"pair_id": pair["pair_id"], "doc_id": pair["doc_id"], "corpus": pair["corpus"],
           "type": pair["type"], "attempts": [], "seed_impl": (
               f"{N_SEM_CALLS} independent plan-path calls; spec seeds {SEM_SEEDS} recorded but "
               "`claude -p` exposes no seed parameter")}
    current = dict(pair)
    for attempt in range(1, args.max_regen + 2):
        if not current.get("errored"):
            rec["attempts"].append({"attempt": attempt, "algo": None, "sem": None,
                                    "reason": "injector returned no errored summary"})
            reasons = ["the injector returned no edited summary"]
        else:
            v1 = W.algo_check_v1(current["clean"], current["errored"], current["type"])
            v2 = W.algo_check(current["clean"], current["errored"], current["type"])
            algo_pass = v1 if args.algo_rule == "v1" else v2["pass"]
            sem_pass, sem_detail = None, None
            if algo_pass:
                if args.dry_run:
                    sem_pass, sem_detail = None, {"dry_run": True}
                else:
                    outs = [W.construct(sem_prompt(prompts, current, doc))
                            for _ in range(N_SEM_CALLS)]
                    sem_pass, sem_detail = sem_majority(outs, current["type"])
            rec["attempts"].append({"attempt": attempt, "algo_v1": v1, "algo_v2": v2,
                                    "algo_rule": args.algo_rule, "algo_pass": bool(algo_pass),
                                    "sem_pass": sem_pass, "sem_detail": sem_detail})
            if algo_pass and (sem_pass or args.dry_run):
                removed = (sem_detail or {}).get("acu_idx")
                rec.update({"verified": True, "errored": current["errored"],
                            "change": current.get("change"), "what": current.get("what"),
                            "acu_idx": removed,
                            "target_acu_idx": pair.get("target_acu_idx"),
                            # RECORDED, NOT GATED. WE_EDIT_CHECK item 5 returns the index of the
                            # ACU the edit actually removed; the injection targeted
                            # target_acu_idx. Nothing in the pre-registration requires those to
                            # agree, but OD_cell is defined on CRITICAL-importance omit pairs, so
                            # an omit pair that drops a different (possibly peripheral) ACU while
                            # carrying a `critical` target label would silently mis-stratify the
                            # headline metric. Surfaced here; adding it as a gate is a decision
                            # for the review.
                            "target_acu_removed": (None if current["type"] != "omit"
                                                   or removed is None
                                                   or pair.get("target_acu_idx") is None
                                                   else removed == pair["target_acu_idx"]),
                            "resolution": "accepted", "n_attempts": attempt})
                return rec
            reasons = ([v2.get("reason")] if not algo_pass and v2.get("reason") else []) or []
            reasons += (sem_detail or {}).get("reasons", [])[:2]
            reasons = [r for r in reasons if r] or ["failed the single-edit check"]

        if attempt > args.max_regen or args.dry_run:
            break
        current = {**current, **inject_one(prompts, instr, doc, {"must_not_contain": []},
                                           current["type"], current.get("target_acu"),
                                           extra=regen_extra(reasons), dry=args.dry_run)}

    # deterministic fallback (spec 3.5)
    if current["type"] == "omit" and not args.dry_run:
        fb = programmatic_omit(current["clean"], current.get("target_acu"))
        if fb:
            v1 = W.algo_check_v1(current["clean"], fb, "omit")
            v2 = W.algo_check(current["clean"], fb, "omit")
            ok = v1 if args.algo_rule == "v1" else v2["pass"]
            rec["attempts"].append({"attempt": "fallback", "algo_v1": v1, "algo_v2": v2,
                                    "algo_pass": bool(ok), "method": "programmatic_omit"})
            if ok:
                rec.update({"verified": True, "errored": fb,
                            "change": "programmatic deletion of the unit carrying the target ACU",
                            "what": current.get("target_acu"), "resolution": "programmatic_omit",
                            "n_attempts": args.max_regen + 1})
                return rec
    rec.update({"verified": False, "errored": current.get("errored"),
                "resolution": ("queued_for_hand_edit" if current["type"] in ("add", "change")
                               else "fallback_failed"),
                "n_attempts": len(rec["attempts"])})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="render + algorithmic layer only")
    ap.add_argument("--algo-rule", default="v1", choices=("v1", "v2"),
                    help="v1 = the constants W-E pre-registers (W-A's frozen rule); "
                         "v2 = W-A's 2026-08-10 revision. Both are always recorded.")
    ap.add_argument("--max-regen", type=int, default=MAX_REGEN)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    prompts = W.load_prompts()
    instr = parse_inject_instr(prompts["we_inject_instr"])
    docs = {d["doc_id"]: d for d in W.load_sample()}
    pairs_path = os.path.splitext(W.PAIRS)[0] + args.out_suffix + ".json"
    if not os.path.exists(pairs_path):
        raise SystemExit(f"missing {pairs_path} - run we_injections.py first")
    pairs = json.load(open(pairs_path))["pairs"]
    if args.limit:
        pairs = pairs[:args.limit]

    print(f"verifying {len(pairs)} pairs (algo rule {args.algo_rule})"
          + (" - dry-run: algorithmic layer only" if args.dry_run else ""))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(lambda p: verify_one(prompts, instr, docs, p, args), pairs))

    by_type = Counter((r["type"], bool(r.get("verified"))) for r in results)
    algo_v1 = sum(1 for r in results if r["attempts"] and r["attempts"][0].get("algo_v1"))
    algo_v2 = sum(1 for r in results
                  if r["attempts"] and (r["attempts"][0].get("algo_v2") or {}).get("pass"))
    out = {
        "generated": date.today().isoformat(),
        "script": "we_verify_pairs.py",
        "spec": "w-e-cross-domain-generalization.md sections 2.2, 3.5, 5 step 8",
        "algo_rule_gating": args.algo_rule,
        "algo_rule_note": ("W-E pre-registers MAX_UNITS=2 / MAX_REGIONS=2 / MAX_CHANGED_WORDS=30 "
                           "(W-A's v1). W-A replaced those on 2026-08-10 with its section-6 single "
                           "revision (v2). Both pass-rates are reported on the same pairs so the "
                           "review can choose; nothing here decides it."),
        "first_attempt_pass_counts": {"algo_v1": algo_v1, "algo_v2": algo_v2, "n": len(results)},
        "recalibration_checkpoint": ("spec 2.2 requires an explicit look at the constants after "
                                     "the first 10 pairs - the per-pair algo_v1/algo_v2 records "
                                     "below are that checkpoint's evidence"),
        "verified_by_type": {f"{t}_{'ok' if ok else 'fail'}": n for (t, ok), n in by_type.items()},
        "omit_target_acu_check": {
            "note": ("RECORDED, NOT GATED - whether the ACU the semantic check says was removed "
                     "is the ACU the injection targeted. OD_cell is defined on critical-importance "
                     "omit pairs, so a mismatch mis-stratifies the headline metric."),
            "matched": sum(1 for r in results if r.get("target_acu_removed") is True),
            "mismatched": sum(1 for r in results if r.get("target_acu_removed") is False),
            "unknown": sum(1 for r in results
                           if r["type"] == "omit" and r.get("target_acu_removed") is None),
            "mismatched_pair_ids": [r["pair_id"] for r in results
                                    if r.get("target_acu_removed") is False],
        },
        "n_verified": sum(1 for r in results if r.get("verified")),
        "n": len(results),
        "results": results,
    }
    path = os.path.splitext(W.PAIR_VERIFICATION)[0] + args.out_suffix + ".json"
    W.save_json(path, out)
    print(f"verified {out['n_verified']}/{len(results)}  ->  {path}")
    print(f"first-attempt algorithmic pass: v1 {algo_v1}/{len(results)}  "
          f"v2 {algo_v2}/{len(results)}")


if __name__ == "__main__":
    main()
