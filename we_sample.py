#!/usr/bin/env python3
"""W-E step 3 (spec Section 5) - the seeded stratified RoSE sample.

Pre-registered method: specs/w-e-cross-domain-generalization.md
  Section 2.2 - "150 documents: 75 sampled from RoSE's CNN/DailyMail test split (500 available,
                 all ACU-annotated) + 75 from RoSE's XSum split (500 available), seeded RNG,
                 seed 20260730, stratified by source-article word-length tercile within each
                 corpus (mirroring W-D's ACI-Bench stratification approach) so the sample isn't
                 skewed toward unusually short or long articles."
  Section 5.3 - "seeded stratified sample -> rose_master/sample.json + a sample_manifest.json
                 recording strata counts (mirrors W-D's subsample_manifest.json pattern)."

Terciles are computed within each corpus over all 500 annotated documents by source-article word
count, split at the 33.3rd/66.7th percentiles, and 25 documents are drawn from each tercile
(75 = 3 x 25 exactly, so no rounding rule is needed). Ties at a tercile boundary go to the lower
tercile - a stable sort on (source_words, doc_id) makes the partition deterministic.

No exclusions are applied: spec 2.2 pre-registers none, and every RoSE document carries a source
article, a reference summary and a non-empty ACU list (verified by we_fetch_rose.py). Documents
whose reference summary carries a single ACU are FLAGGED in the manifest, not dropped - see
single_acu_documents there.

Deterministic, stdlib-only, NO LLM calls.

Inputs   rose_master/rose_raw.json          (we_fetch_rose.py)
Outputs  rose_master/sample.json            the 150 sampled documents, full text (committed)
         rose_master/sample_manifest.json   strata counts + provenance (committed)

Run: python3 we_sample.py [--per-corpus 75] [--force]
"""

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import date

HARNESS = os.path.dirname(os.path.abspath(__file__))
ROSE_MASTER = os.path.join(HARNESS, "rose_master")
RAW = os.path.join(ROSE_MASTER, "rose_raw.json")
OUT_SAMPLE = os.path.join(ROSE_MASTER, "sample.json")
OUT_MANIFEST = os.path.join(ROSE_MASTER, "sample_manifest.json")

SEED = 20260730          # spec section 2.2
PER_CORPUS = 75          # spec section 2.2 (75 CNN/DailyMail + 75 XSum = 150)
N_TERCILES = 3


def terciles(docs):
    """Partition docs into 3 word-length strata within a corpus, low to high."""
    ordered = sorted(docs, key=lambda d: (d["source_words"], d["doc_id"]))
    n = len(ordered)
    cuts = [round(n * i / N_TERCILES) for i in range(N_TERCILES + 1)]
    return [ordered[cuts[i]:cuts[i + 1]] for i in range(N_TERCILES)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-corpus", type=int, default=PER_CORPUS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(RAW):
        sys.exit(f"missing {RAW} - run we_fetch_rose.py first")
    if os.path.exists(OUT_SAMPLE) and not args.force:
        sys.exit(f"{OUT_SAMPLE} exists - re-run with --force only before construction starts")
    if args.per_corpus % N_TERCILES:
        sys.exit(f"--per-corpus must divide by {N_TERCILES} for an even tercile draw")

    raw = json.load(open(RAW))
    records = raw["records"]
    by_corpus = {}
    for r in records:
        by_corpus.setdefault(r["corpus"], []).append(r)

    per_tercile = args.per_corpus // N_TERCILES
    rng = random.Random(SEED)
    sample, strata_report = [], {}
    for corpus in sorted(by_corpus):
        docs = by_corpus[corpus]
        strata = terciles(docs)
        strata_report[corpus] = {"pool": len(docs), "drawn": 0, "terciles": []}
        for i, stratum in enumerate(strata):
            members = sorted(stratum, key=lambda d: d["doc_id"])
            rng.shuffle(members)
            drawn = members[:per_tercile]
            sample.extend(drawn)
            words = [d["source_words"] for d in stratum]
            strata_report[corpus]["terciles"].append({
                "tercile": i + 1,
                "available": len(stratum),
                "drawn": len(drawn),
                "source_words_range": [min(words), max(words)] if words else None,
                "drawn_source_words_mean": (round(sum(d["source_words"] for d in drawn)
                                                  / len(drawn), 1) if drawn else None),
            })
            strata_report[corpus]["drawn"] += len(drawn)

    sample.sort(key=lambda d: (d["corpus"], d["doc_id"]))
    ids = [d["doc_id"] for d in sample]

    checks = {
        "sample_size": len(sample) == args.per_corpus * len(by_corpus),
        "per_corpus_balanced": all(v["drawn"] == args.per_corpus for v in strata_report.values()),
        "doc_ids_unique": len(set(ids)) == len(ids),
        "every_doc_has_source_reference_and_acus":
            all(d["source"] and d["reference"] and d["reference_acus"] for d in sample),
        "sample_is_subset_of_raw": set(ids) <= {r["doc_id"] for r in records},
    }
    failed = [k for k, v in checks.items() if not v]

    n_acus = sum(d["n_acus"] for d in sample)
    manifest = {
        "generated": date.today().isoformat(),
        "script": "we_sample.py",
        "spec": "w-e-cross-domain-generalization.md sections 2.2, 5 step 3",
        "seed": SEED,
        "n_target_per_corpus": args.per_corpus,
        "n_sampled": len(sample),
        "stratification": ("source-article word-length terciles within each corpus, computed over "
                           "all 500 annotated documents; equal draw per tercile"),
        "strata": strata_report,
        "acu_totals": {
            "acus_in_sample": n_acus,
            "by_corpus": {c: sum(d["n_acus"] for d in sample if d["corpus"] == c)
                          for c in sorted(by_corpus)},
            "mean_acus_per_doc": round(n_acus / len(sample), 2) if sample else None,
        },
        "planned_pairs": {
            "rule": "spec 2.2 - 3 pairs per document (add/change/omit), one injected error each",
            "pairs": 3 * len(sample),
            "unique_summary_judgements_per_cell_per_run": len(sample) + 3 * len(sample),
        },
        "single_acu_documents": {
            "n": sum(1 for d in sample if d["n_acus"] == 1),
            "doc_ids": [d["doc_id"] for d in sample if d["n_acus"] == 1],
            "flag": ("a document whose reference summary carries exactly one ACU cannot yield an "
                     "omit pair that deletes a targeted ACU and still leaves a summary - the "
                     "'errored' summary would be empty. Spec 2.2 pre-registers no exclusion, so "
                     "none is applied here; how construction handles these is a decision for the "
                     "review (drop the doc, allow a partial-clause deletion, or substitute a "
                     "supporting-tier target)."),
        },
        "inputs": {"file": "rose_master/rose_raw.json",
                   "sha256": hashlib.sha256(open(RAW, "rb").read()).hexdigest(),
                   "records": len(records)},
        "license_note": ("sample.json carries full source-article and reference-summary text for "
                         "both strata, which spec 3.6 clears for RUNNING the study. The public "
                         "release package must still apply 3.6's two-tier split (CNN/DailyMail "
                         "text open, XSum text pointer-only) - open question Q1, not decided here."),
        "doc_ids": ids,
        "validation": {"checks": checks, "failed": failed},
    }

    json.dump({"generated": date.today().isoformat(), "script": "we_sample.py", "seed": SEED,
               "spec": "w-e-cross-domain-generalization.md sections 2.2, 5 step 3",
               "n": len(sample), "documents": sample},
              open(OUT_SAMPLE, "w", encoding="utf-8"), indent=1)
    json.dump(manifest, open(OUT_MANIFEST, "w", encoding="utf-8"), indent=1)

    print(f"sampled {len(sample)} documents ({n_acus} ACUs)  ->  {OUT_SAMPLE}")
    for c, v in strata_report.items():
        rng_txt = " | ".join(f"T{t['tercile']} {t['source_words_range'][0]}-"
                             f"{t['source_words_range'][1]}w x{t['drawn']}" for t in v["terciles"])
        print(f"  {c:7s} {v['drawn']}/{v['pool']}   {rng_txt}")
    print(f"planned pairs: {3 * len(sample)}")
    print("validation:", "PASS" if not failed else f"FAIL {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
