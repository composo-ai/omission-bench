#!/usr/bin/env python3
"""W-E step 7 (spec Section 5) - build the 450 injected pairs over the RoSE sample.

Pre-registered method: specs/w-e-cross-domain-generalization.md
  Section 2.2 - 3 pairs per document (add / change / omit), exactly one injected edit each;
                "omit: delete the clause or sentence carrying one targeted ACU, add nothing in
                its place. Targeted ACU is drawn from the `critical` importance tier by default
                (Section 2.3, Section 10 Q10 flags this as a design choice, not a given)"
  Section 3.5 - the WE_INJECT prompt and the per-type {instr} table, both loaded verbatim from
                we_prompts/ (hash-gated), plus W-A's REGEN_SUFFIX pattern for regeneration

Claude plan path only (spec section 4) - NO judge-arm call anywhere in this file.

Target selection, stated because the spec pre-registers only part of it:
  omit    - default `critical` tier, per section 2.2. --omit-target-tier changes it (Q10).
  change  - the spec says "alter exactly one existing ACU-bearing claim" without naming a tier.
            Default here is a seeded draw over ALL graded ACUs; --change-target-tier restricts it.
            FLAGGED: the tier for `change` is genuinely unspecified in the pre-registration.
  add     - no ACU target; the injector picks one of the document's audited must_not_contain
            claims (the prompt's UNSUPPORTED_CLAIMS slot carries the whole audited pool, exactly
            as section 3.5 writes it, and the model's own `what` field records which it used).

Usage
  python3 we_injections.py [--limit N] [--workers 4] [--dry-run] [--out-suffix _shapecheck]
"""

import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import we_common as W
from w_a_master import REGEN_SUFFIX

STATE = os.path.join(W.ROSE_MASTER, "injection_state.json")
INSTR_LINE = re.compile(r"^(add|change|omit):\s*\"(.*)$")


def parse_inject_instr(text):
    """we_prompts/we_inject_instr.txt is the spec's per-type table: `add:  "<instruction ...`
    with indented continuation lines inside the quotes. The instruction is the quoted string with
    the table's line wrapping collapsed to single spaces - a layout artefact, not prompt content.
    """
    out, cur = {}, None
    for line in text.split("\n"):
        m = INSTR_LINE.match(line)
        if m:
            cur = m.group(1)
            out[cur] = m.group(2)
        elif cur and line.strip():
            out[cur] += " " + line.strip()
    for k in list(out):
        out[k] = re.sub(r"\s+", " ", out[k]).strip().rstrip('"').strip()
    missing = [t for t in W.TYPES if t not in out]
    if missing:
        raise RuntimeError(f"we_inject_instr.txt missing types {missing}")
    return out


def pick_target(doc, sheet, typ, rng, omit_tier, change_tier):
    """Return (target_acu_text, acu_idx, tier) or (None, None, None) for add."""
    if typ == "add":
        return None, None, None
    imp = sheet.get("acu_importance") or {}
    tier = omit_tier if typ == "omit" else change_tier
    pool = [(int(i), doc["reference_acus"][int(i)]) for i, v in sorted(imp.items(), key=lambda kv: int(kv[0]))
            if int(i) < len(doc["reference_acus"]) and (tier == "any" or v == tier)]
    if not pool:
        return None, None, tier
    idx, acu = pool[rng.randrange(len(pool))]
    return acu, idx, imp.get(str(idx))


def build_prompt(prompts, instr, doc, sheet, typ, target_acu, extra=""):
    claims = sheet.get("must_not_contain") or []
    return W.render(
        prompts["we_inject"],
        article=doc["source"],
        summary=doc["reference"],
        instr=instr[typ] + extra,
        target_acu=target_acu or "(not applicable for this edit type)",
        unsupported_claims=("\n".join(f"- {c}" for c in claims)
                            if claims else "(none available for this document)"),
    )


def regen_extra(reasons, max_units=2):
    """W-A's REGEN_SUFFIX pattern verbatim, with its one domain word swapped (note -> summary)."""
    return (REGEN_SUFFIX.replace("{failure_reasons}", "\n".join(f"- {r}" for r in reasons))
            .replace("{max_units}", str(max_units))
            .replace("from the note", "from the summary"))


def inject_one(prompts, instr, doc, sheet, typ, target_acu, extra="", dry=False):
    prompt = build_prompt(prompts, instr, doc, sheet, typ, target_acu, extra)
    if dry:
        return {"rendered_chars": len(prompt), "preview": prompt[:400]}
    out = W.construct(prompt) or {}
    return {"errored": (out.get("summary") or "").strip() or None,
            "change": out.get("change"), "what": out.get("what"), "raw_ok": bool(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="render only, zero LLM calls")
    ap.add_argument("--omit-target-tier", default="critical",
                    choices=("critical", "supporting", "peripheral", "any"))
    ap.add_argument("--change-target-tier", default="any",
                    choices=("critical", "supporting", "peripheral", "any"))
    ap.add_argument("--fact-sheets", default=None, help="override the fact-sheet file")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    prompts = W.load_prompts()
    instr = parse_inject_instr(prompts["we_inject_instr"])
    docs = {d["doc_id"]: d for d in W.load_sample()}

    fs_path = args.fact_sheets or (os.path.splitext(W.FACT_SHEETS)[0] + args.out_suffix + ".json")
    if not os.path.exists(fs_path):
        raise SystemExit(f"missing {fs_path} - run we_construct.py --stage all first")
    sheets = {s["doc_id"]: s for s in json.load(open(fs_path))["fact_sheets"]}

    doc_ids = [d for d in sheets if d in docs]
    doc_ids.sort()
    if args.limit:
        doc_ids = doc_ids[:args.limit]

    state = W.load_state(STATE, {})
    jobs = []
    for did in doc_ids:
        rng = random.Random(f"{W.SEED_INJECT}:{did}")     # per-document, order-independent
        for typ in W.TYPES:
            pid = W.pair_id(did, typ)
            if pid in state and not args.dry_run:
                continue
            acu, idx, tier = pick_target(docs[did], sheets[did], typ, rng,
                                         args.omit_target_tier, args.change_target_tier)
            jobs.append((pid, did, typ, acu, idx, tier))

    print(f"{len(jobs)} pairs to build over {len(doc_ids)} documents"
          + (" (dry-run: no LLM calls)" if args.dry_run else ""))

    def work(job):
        pid, did, typ, acu, idx, tier = job
        res = inject_one(prompts, instr, docs[did], sheets[did], typ, acu, dry=args.dry_run)
        return pid, {"pair_id": pid, "doc_id": did, "corpus": docs[did]["corpus"], "type": typ,
                     "clean": docs[did]["reference"], "target_acu": acu, "target_acu_idx": idx,
                     "target_tier": tier, "attempts": 1, **res}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for pid, rec in ex.map(work, jobs):
            state[pid] = rec
            print(f"  {pid}: " + json.dumps({k: v for k, v in rec.items()
                                             if k in ("target_tier", "rendered_chars", "raw_ok")}))

    if args.dry_run:
        print("dry-run complete - prompts rendered, no state written, no calls made")
        return

    W.save_json(STATE, state)
    pairs = [state[W.pair_id(d, t)] for d in doc_ids for t in W.TYPES
             if W.pair_id(d, t) in state]
    out = os.path.splitext(W.PAIRS)[0] + args.out_suffix + ".json"
    W.save_json(out, {
        "generated": date.today().isoformat(),
        "script": "we_injections.py",
        "spec": "w-e-cross-domain-generalization.md sections 2.2, 3.5, 5 step 7",
        "seed": W.SEED_INJECT,
        "target_tiers": {"omit": args.omit_target_tier, "change": args.change_target_tier,
                         "note": ("the spec pre-registers the tier for omit only (2.2 / Q10); the "
                                  "change tier is unspecified and defaults to `any` here")},
        "prompt_sha256": W.prompt_hashes(),
        "n": len(pairs),
        "pairs": pairs,
    })
    built = sum(1 for p in pairs if p.get("errored"))
    print(f"pairs written: {len(pairs)} ({built} with an errored summary)  ->  {out}")


if __name__ == "__main__":
    main()
