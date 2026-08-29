#!/usr/bin/env python3
r"""W-B P3 - MEDEC external anchor: two judge arms on physician-labelled clinical texts.

Why this run exists. Every judge number in the study so far is scored against ground truth WE
built (authored pairs, constructed injections, LLM-verified). MEDEC is the opposite: physicians
wrote and validated the labels, each text is error-free or carries exactly ONE error with the
erroneous sentence identified, and the errors are all COMMISSIONS - no omissions by construction.
So it anchors the commission side of the study's asymmetry claim to ground truth we had no hand in.

Design (specs/w-b-external-anchors.md section 2(b), narrowed to two arms):
  items   the pre-registered stratified n=300 sample of the MEDEC MS test split, frozen with its
          seed by wb_medec_ingest.py (156 errored / 144 error-free). Sample membership is read
          from wb_medec_items.json - this runner never re-draws it.
  arms    FC-score-k8            the grid's best design (specs/w2-ablation-grid.md; FINDINGS 17)
          engineered-completeness the strongest prompt-only judge in the study (FINDINGS 18)
          Both come from w2_arms.ARMS - identity, prompt, k and flag rule are the registry's,
          not this file's.
  modes   FLAG (primary)   judge each of the 300 texts alone -> detection on the 156 errored,
                           false alarm on the 144 error-free. The deployment frame.
          PAIRED (secondary) judge each errored text's CLEAN TWIN -> tie-adjusted P(errored
                           scores below its own twin), the measure FINDINGS 17.1/18.1 quote.
  1 replicate (r=1, seed 11 - w2_common.RUN_SEEDS[1], the grid's own run-1 seed).

Adapter contract. wb_adapters.medec_flag is the PRE-REGISTERED adapter (spec 3.2) and it is
arm-agnostic - item -> (transcript_slot, note_slot) - so both arms use it verbatim, with
MEDEC_NO_SOURCE occupying the transcript slot. No new adapter was written. Two disclosures the
results must carry:
  1. The engineered arm's prompt is explicitly omission-hunting ("You are looking above all for
     CONTENT THAT IS MISSING"). MEDEC has no source document and no omissions, so that arm is
     being asked a question its prompt is not aimed at. Reported, not corrected: changing the
     prompt would stop it being the same arm.
  2. Clean-twin rendering. wb_medec_ingest_report.json flags the choice as OPEN and freezes two
     renderings (newline-joined / space-joined), neither of which matches the dataset's own
     `Text` byte for byte: `Text` keeps CRLF line breaks inside lab panels, and 25 of the 156
     errored texts differ from a space-joined sentence list by exactly that whitespace. Judging
     an errored text in one rendering against its twin in another would vary FORMATTING across
     the pair as well as content. This runner therefore builds the twin by substituting the
     annotator's corrected sentence INTO the dataset's own `Text` - each error sentence occurs
     there exactly once as an exact substring (verified for all 156 before the run), so the twin
     is byte-identical to the errored text apart from the one substituted sentence. Content is
     identical to the pre-registered twin; only whitespace outside the substitution differs.

Cost. Own-run tripwire only (LocalGuard): this run's OWN receipts, resumed from the record store,
warn at \$35 and stop at \$45. w2_common.SpendGuard is deliberately NOT used - it trips on the
project-wide ledger baseline (~\$1.4K), which says nothing about this run.

Run:
    .venv/bin/python wb_run_medec.py --plan
    .venv/bin/python wb_run_medec.py --limit 6 --tag medec-pilot -y      # cost probe
    .venv/bin/python wb_run_medec.py -y                                  # the anchor
"""
import argparse, json, os, re, time

from common import HERE, Run
import w2_common as W
import w2_arms as A
import wb_adapters as AD

EXPERIMENT = "wb-medec"
SPEC = "specs/w-b-external-anchors.md"
ITEMS_FILE = "wb_medec_items.json"
ARM_NAMES = ["FC-score-k8", "engineered-completeness"]
REPLICATE = 1


# ---------------------------------------------------------------- notes
def clean_twin_inplace(item):
    """The errored text with the annotator's corrected sentence substituted in place.

    Exact-substring substitution into the dataset's own `Text`, so the twin differs from the
    errored text by the corrected sentence and nothing else (see the module docstring). Raises
    rather than guessing if the error sentence is not present exactly once - a silent fallback
    would put an asymmetric pair into the paired metric.
    """
    text, err, corr = item["text"], item["error_sentence"], item["corrected_sentence"]
    if not (err and corr):
        raise ValueError(f"{item['item_id']}: errored item without both sentences")
    n = text.count(err)
    if n != 1:
        raise ValueError(f"{item['item_id']}: error sentence occurs {n}x in Text, expected 1")
    return text.replace(err, corr)


def build_notes(items, paired=True):
    """One note per judgement this run buys.

    role  errored     the dataset's errored text        -> detection numerator
          error_free  the dataset's error-free text     -> false-alarm numerator (deployment)
          clean_twin  the reconstructed corrected text  -> paired measure + a second FA read

    item_index is assigned over the sorted note keys, exactly as w2_common.note_units does, so
    the seed formula is stable across restarts and an item's flag note and twin note never
    collide on a seed.
    """
    notes = []
    for it in items:
        notes.append({"note_key": f"{it['item_id']}|flag",
                      "note_role": "errored" if it["error_flag"] else "error_free",
                      "text": it["text"], "item": it, "pair_key": it["item_id"]})
        if paired and it["error_flag"]:
            notes.append({"note_key": f"{it['item_id']}|twin", "note_role": "clean_twin",
                          "text": clean_twin_inplace(it), "item": it,
                          "pair_key": it["item_id"]})
    for i, n in enumerate(sorted(notes, key=lambda n: n["note_key"])):
        n["item_index"] = i
    return notes


def note_fields(note):
    """The identity block copied onto every record, so the analysis reads results/ alone."""
    it = note["item"]
    return {"note_key": note["note_key"], "note_role": note["note_role"],
            "item_id": it["item_id"], "text_id": it["text_id"], "pair_key": note["pair_key"],
            "error_flag": it["error_flag"], "error_type": it["error_type"],
            "error_sentence_id": it["error_sentence_id"], "n_sentences": it["n_sentences"],
            "item_index": note["item_index"], "split": it["split"]}


# ---------------------------------------------------------------- own-run spend tripwire
class LocalGuard:
    """Tripwire on THIS run's own spend, resumed from the record store.

    The project ledger is shared with every other workstream, so a global baseline cannot say
    whether this run is over budget. Every record carries its own receipts; summing them over
    the store gives the wb-medec total across restarts, which is the number the cap applies to.
    """

    def __init__(self, prior, warn=35.0, stop=45.0):
        self.prior, self.warn, self.stop, self.spent_here, self._warned = prior, warn, stop, 0.0, False

    @property
    def total(self):
        return self.prior + self.spent_here

    def add(self, usd):
        self.spent_here += usd or 0.0
        if self.total >= self.warn and not self._warned:
            self._warned = True
            print(f"  ! SPEND WARNING: wb-medec at ${self.total:.2f} (warn ${self.warn:.0f})",
                  flush=True)
        if self.total >= self.stop:
            raise SystemExit(
                f"\nSPEND STOP: wb-medec own spend ${self.total:.2f} >= ${self.stop:.0f}.\n"
                "Stopping cleanly - every completed judgement is already in the record store, "
                "so a re-launch resumes rather than re-buys.")


def store_spend(store):
    return round(sum((r.get("totals") or {}).get("cost_usd") or 0.0
                     for r in store.records.values()), 6)


# ---------------------------------------------------------------- runner
def main():
    ap = argparse.ArgumentParser(description="W-B MEDEC external anchor")
    ap.add_argument("--arms", default=",".join(ARM_NAMES))
    ap.add_argument("--tag", default="medec-main")
    ap.add_argument("--limit", type=int, default=None, help="first N sample items (pilot only)")
    ap.add_argument("--no-paired", action="store_true", help="flag mode only")
    ap.add_argument("--all-items", action="store_true",
                    help="ignore the frozen n=300 sample and judge all 597 MS-test items")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=60, help="notes per Run/manifest")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="stop after N chunks this invocation (resume by re-running)")
    ap.add_argument("--time-budget", type=float, default=None,
                    help="seconds: do not START another chunk past this (resume by re-running)")
    ap.add_argument("--warn-usd", type=float, default=35.0)
    ap.add_argument("--stop-usd", type=float, default=45.0)
    ap.add_argument("--model", default="gpt54", choices=sorted(W.ROLES))
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    arms = A.select(args.arms)
    for a in arms:
        ok, why = A.available(a)
        if not ok or a["callable"]:
            raise SystemExit(f"{a['name']} is not runnable here: {why}")
    prompts = {a["name"]: A.load_arm_prompt(a) for a in arms}
    W.load_prompts(quiet=True)   # populate the drift detector for the manifest

    items = AD.load_medec_items(sample_only=not args.all_items)
    items.sort(key=lambda it: it["item_id"])
    if args.limit:
        items = items[:args.limit]
    notes = build_notes(items, paired=not args.no_paired)

    blob = json.load(open(os.path.join(HERE, ITEMS_FILE)))
    info = {"pairs_file": ITEMS_FILE, "sha256": W.sha256_file(os.path.join(HERE, ITEMS_FILE)),
            "dataset_version": f"medec-ms-test@{blob['generated']}", "dataset_kind": "external_medec",
            "n_pairs": len(items), "frozen": True}
    role_cfg = W.ROLES[args.model]
    store = W.RecordStore(EXPERIMENT, args.tag)
    prior = store_spend(store)

    def done(arm, note):
        return store.has(f"{arm['name']}|r{REPLICATE}|{note['note_key']}")

    pend = [(a, n) for n in notes for a in arms if not done(a, n)]
    calls = sum(a["k"] for a, _ in pend)
    roles = {"errored": 0, "error_free": 0, "clean_twin": 0}
    for n in notes:
        roles[n["note_role"]] += 1
    print(f"W-B MEDEC anchor | arms: {', '.join(a['name'] for a in arms)}")
    print(f"  items {len(items)} ({roles['errored']} errored / {roles['error_free']} error-free)"
          f" | notes {len(notes)} (+{roles['clean_twin']} clean twins)")
    print(f"  {len(pend)} judgements to buy ({calls} calls); {len(store.records)} already in "
          f"{os.path.relpath(store.path, HERE)} (${prior:.2f} spent)")
    if args.plan:
        for a in arms:
            print(f"    {a['name']:<26} k={a['k']} fmt={a['format']} rule={a['flag_rule']}")
        return
    if not pend:
        print("nothing to do")
        return
    W.confirm(f"proceed with ~{calls} calls (cap ${args.stop_usd:.0f})?", args.yes)

    guard = LocalGuard(prior, args.warn_usd, args.stop_usd)
    seed = W.RUN_SEEDS[REPLICATE]
    t0, made, run_ids = time.time(), 0, []

    ran = 0
    for ci, chunk in enumerate(W.chunked(notes, args.chunk), 1):
        jobs = [(a, n) for n in chunk for a in arms if not done(a, n)]
        if not jobs:
            continue
        if args.max_chunks and ran >= args.max_chunks:
            print(f"  stopping at --max-chunks {args.max_chunks}; re-run to resume", flush=True)
            break
        if args.time_budget and time.time() - t0 >= args.time_budget:
            print(f"  stopping at --time-budget {args.time_budget:.0f}s; re-run to resume",
                  flush=True)
            break
        ran += 1
        # k=1 first: run_block's serial warm-up should be a single request, and the engineered
        # arm's 6.8K-char preamble is the only prefix long enough for the provider to cache.
        jobs.sort(key=lambda j: (j[0]["k"], j[0]["name"], j[1]["note_key"]))
        params = W.run_params(
            {"arms": [a["name"] for a in arms], "model_key": args.model, "role": role_cfg["role"],
             "reasoning_effort": role_cfg["reasoning_effort"], "max_tokens": W.MAX_TOKENS,
             "workers": args.workers, "chunk": ci, "mode": "flag+paired" if not args.no_paired
             else "flag", "sample": "all_597" if args.all_items else "frozen_n300",
             "adapter": "wb_adapters.medec_flag (pre-registered, spec 3.2) for both arms; "
                        "clean twin = corrected sentence substituted into the dataset Text "
                        "in place (formatting held constant across the pair)",
             "arm_specs": [{k: v for k, v in a.items() if k != "parser"} for a in arms],
             "prompt_sha256": {n: W.sha256_text(p) for n, p in prompts.items()},
             "record_store": os.path.relpath(store.path, HERE),
             "n_errored": roles["errored"], "n_error_free": roles["error_free"],
             "n_clean_twins": roles["clean_twin"]},
            info)
        W.assert_dataset_stable(info)
        with Run(EXPERIMENT, params=params, replicate=REPLICATE, seed=seed, inputs=[ITEMS_FILE],
                 spec=SPEC, allow_dirty=True) as run:
            run.register_prompts(prompts)
            run_ids.append(run.run_id)
            chunk_records = []

            def worker(job):
                arm, note = job
                transcript, note_text = AD.medec_flag({"text": note["text"]})
                prompt = A.render_arm(prompts[arm["name"]], transcript, note_text,
                                      arm["max_transcript_chars"])
                base_seed = seed * 100000 + note["item_index"] * 10
                vals, texts, metas, retried = W.judge(
                    prompt, role_cfg["role"], arm["k"], base_seed, temperature=1.0,
                    reasoning_effort=role_cfg["reasoning_effort"], parser=arm["parser"])
                fmt = "bin" if arm["format"] == "bin" else "score"
                agg, verdict, flagged = W.aggregate_cell(fmt, arm["k"], vals)
                rcs = W.receipts(metas)
                tot = W.receipt_totals(rcs)
                rec = {"key": f"{arm['name']}|r{REPLICATE}|{note['note_key']}",
                       "cell": arm["name"], "arm": arm["name"], "arm_family": arm["family"],
                       "format": fmt, "k": arm["k"], "temperature": 1.0, "replicate": REPLICATE,
                       "run_seed": seed, "base_seed": base_seed, "model_key": args.model,
                       "role": role_cfg["role"], "model": metas[0]["model"],
                       "k_impl": metas[0]["k_impl"], "flag_rule": arm["flag_rule"],
                       **note_fields(note),
                       "samples": vals, "aggregate": agg, "verdict": verdict, "flagged": flagged,
                       "parse_failure": agg is None, "retried_samples": retried,
                       "receipts": rcs, "totals": tot, "prompt_chars": len(prompt),
                       "note_chars": len(note_text), "run_id": run.run_id,
                       "t": round(time.time(), 3), "dataset_version": info["dataset_version"]}
                if arm["format"] == "score":   # the engineered arm's structured error list
                    j = W.parse_json_blob(texts[0])
                    if isinstance(j, dict) and isinstance(j.get("omissions"), list):
                        rec["omissions_reported"] = j["omissions"][:20]
                store.put(rec)
                chunk_records.append(rec)
                guard.add(tot["cost_usd"])

            made += W.run_block(jobs, worker, args.workers)
            run.save("wb_medec_run.json", {"tag": args.tag, "chunk": ci, "params": params,
                                           "n_records": len(chunk_records),
                                           "records": chunk_records})
        print(f"  chunk {ci}: {len(jobs)} judgements | ${guard.total:.2f} | "
              f"{time.time() - t0:.0f}s", flush=True)

    store.close()
    print(f"\ndone: {made} judgements this session; ${guard.spent_here:.2f} this session, "
          f"${guard.total:.2f} for wb-medec total")
    print(f"  store: {os.path.relpath(store.path, HERE)} ({len(store.records)} records)")
    print(f"  run_ids: {', '.join(run_ids)}")


if __name__ == "__main__":
    main()
