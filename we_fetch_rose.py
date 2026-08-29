#!/usr/bin/env python3
"""W-E step 2 (spec Section 5) - fetch the RoSE ACU benchmark and normalise it.

Pre-registered method: specs/w-e-cross-domain-generalization.md
  Section 3.7 - "RoSE ACU annotations: HuggingFace Salesforce/rose, splits cnndm_test and xsum -
                 fetched once, cached to rose_master/rose_raw.json. Exact field names (does the
                 release ship source article text inline, or document IDs only requiring a join
                 against abisee/cnn_dailymail / EdinburghNLP/xsum?) need verification at
                 construction start - flagged, not assumed (Section 10 Q2)."
  Section 5.2 - "pulls Salesforce/rose (cnndm_test + xsum splits) and joins to
                 abisee/cnn_dailymail / EdinburghNLP/xsum source text by document id ->
                 rose_master/rose_raw.json. Verifies the join is complete (every RoSE-annotated
                 document resolves to source text) and aborts loudly if not."

Q2 RESOLVED AT FETCH (2026-08-10): no join is needed. The Salesforce/rose release ships the full
source article inline in a `source` field alongside `reference` (the human reference summary) and
`reference_acus` (the ACU list), one record per document. Every one of the 500 + 500 annotated
documents therefore resolves to source text by construction, and abisee/cnn_dailymail /
EdinburghNLP/xsum are never contacted. The completeness check below is retained and asserted
anyway, so a future release that drops the inline text still aborts loudly.

Q3 CHECKED AT FETCH: the release carries NO per-ACU importance / weight / uniformity field.
`reference_acus` is a flat list of ACU strings; `annotations` holds per-system binary acu_labels
(which ACUs each system's summary covered) and the derived acu / normalized_acu scores. So the
Section 3.2 WE_SALIENCE grading pass is NOT redundant. (Section 10 Q3 asks a colleague to also check the
ACL 2023 PDF and demo.ipynb - this only settles what the released annotations contain.)

License verification (Section 3.6, re-verified 2026-08-10 against the primary sources):
  - Salesforce/rose dataset card, Ethical Considerations, verbatim: "This release is for research
    purposes only in support of an academic paper." No SPDX license tag in the card YAML.
  - The release ships LICENSE.txt = BSD 3-Clause (Salesforce.com, Inc.), covering the repo's code.
  - Constraint is on REDISTRIBUTION, not on running the experiment (Section 3.6's verdict). The
    XSum stratum's source text is BBC-copyrighted; Section 3.6/Q1's two-tier public-release split
    is unchanged by the inline-text finding, except that the "ships document IDs resolving through
    the standard EdinburghNLP/xsum loader" mechanism now points at RoSE's own release instead.

Deterministic, stdlib-only, NO LLM calls.

Outputs
  external/rose/                     raw download (gitignored): rose_data.tar.gz, LICENSE.txt,
                                     README.md, extracted rose_data/*.jsonl
  rose_master/rose_raw.json          normalised records (gitignored - regenerable raw cache;
                                     carries full CNN/DailyMail + XSum article text)
  rose_master/rose_fetch_report.json counts, license findings, Q2/Q3 resolutions (committed)

Run: python3 we_fetch_rose.py [--force]
"""

import argparse
import hashlib
import json
import os
import sys
import tarfile
import urllib.request
from datetime import date

HARNESS = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HARNESS, "external", "rose")
ROSE_MASTER = os.path.join(HARNESS, "rose_master")
OUT_RAW = os.path.join(ROSE_MASTER, "rose_raw.json")
OUT_REPORT = os.path.join(ROSE_MASTER, "rose_fetch_report.json")

HF_BASE = "https://huggingface.co/datasets/Salesforce/rose/resolve/main/"
FILES = ["rose_data.tar.gz", "LICENSE.txt", "README.md", "dataset_infos.json"]

# (HF split name per the dataset card table, file inside rose_data/, our corpus label)
SPLITS = [
    ("cnndm_test", "cnndm.test.acus.aggregated.jsonl", "cnndm"),
    ("xsum", "xsum.test.acus.aggregated.jsonl", "xsum"),
]

REQUIRED_FIELDS = ["source", "reference", "reference_acus", "example_id", "count_id"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(force=False):
    os.makedirs(RAW_DIR, exist_ok=True)
    fetched = []
    for name in FILES:
        path = os.path.join(RAW_DIR, name)
        if os.path.exists(path) and not force:
            continue
        urllib.request.urlretrieve(HF_BASE + name, path)
        fetched.append(name)
    tarball = os.path.join(RAW_DIR, "rose_data.tar.gz")
    if not os.path.exists(os.path.join(RAW_DIR, "rose_data")) or force:
        with tarfile.open(tarball) as t:
            t.extractall(RAW_DIR)
    return fetched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download and re-extract")
    args = ap.parse_args()

    os.makedirs(ROSE_MASTER, exist_ok=True)
    fetched = download(args.force)

    records, per_split, problems = [], {}, []
    for hf_split, fname, corpus in SPLITS:
        path = os.path.join(RAW_DIR, "rose_data", fname)
        if not os.path.exists(path):
            sys.exit(f"missing {path} - the RoSE tarball layout changed; re-check the release")
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        for r in rows:
            missing = [f for f in REQUIRED_FIELDS if f not in r]
            if missing:
                problems.append({"corpus": corpus, "example_id": r.get("example_id"),
                                 "issue": f"missing fields {missing}"})
                continue
            source, reference = (r["source"] or "").strip(), (r["reference"] or "").strip()
            acus = [a.strip() for a in (r["reference_acus"] or []) if a and a.strip()]
            # Section 5.2's abort condition, kept even though the join is now trivial.
            if not source:
                problems.append({"corpus": corpus, "example_id": r["example_id"],
                                 "issue": "no source article text - would need an external join"})
                continue
            if not reference or not acus:
                problems.append({"corpus": corpus, "example_id": r["example_id"],
                                 "issue": "no reference summary or empty ACU list"})
                continue
            records.append({
                "doc_id": f"{corpus}_{r['example_id']}",
                "corpus": corpus,
                "hf_split": hf_split,
                "example_id": r["example_id"],
                "count_id": r["count_id"],
                "source": source,
                "reference": reference,
                "reference_acus": acus,
                "n_acus": len(acus),
                "source_words": len(source.split()),
                "reference_words": len(reference.split()),
                "systems_annotated": sorted((r.get("annotations") or {}).keys()),
            })
        got = [x for x in records if x["corpus"] == corpus]
        per_split[corpus] = {
            "hf_split": hf_split,
            "file": fname,
            "rows_in_file": len(rows),
            "records_kept": len(got),
            "card_expected_docs": 500,
            "acus_per_doc_min": min((x["n_acus"] for x in got), default=None),
            "acus_per_doc_max": max((x["n_acus"] for x in got), default=None),
            "acus_total": sum(x["n_acus"] for x in got),
            "source_words_median": sorted(x["source_words"] for x in got)[len(got) // 2] if got else None,
            "reference_words_median": sorted(x["reference_words"] for x in got)[len(got) // 2] if got else None,
            "single_acu_docs": sum(1 for x in got if x["n_acus"] == 1),
        }

    checks = {
        "every_annotated_document_resolves_to_source_text": not problems,
        "doc_ids_unique": len({r["doc_id"] for r in records}) == len(records),
        "both_splits_present": all(per_split[c]["records_kept"] > 0 for _, _, c in SPLITS),
        "counts_match_dataset_card": all(per_split[c]["records_kept"] == 500 for _, _, c in SPLITS),
    }
    failed = [k for k, v in checks.items() if not v]

    report = {
        "generated": date.today().isoformat(),
        "script": "we_fetch_rose.py",
        "spec": "w-e-cross-domain-generalization.md sections 3.6, 3.7, 5 step 2",
        "source": {
            "dataset": "Salesforce/rose (RoSE benchmark)",
            "url": "https://huggingface.co/datasets/Salesforce/rose",
            "paper": "Liu, Fabbri et al., 'Revisiting the Gold Standard: Grounding Summarization "
                     "Evaluation with Robust Human Evaluation', ACL 2023 (arXiv:2212.07981)",
            "code_repo": "https://github.com/Yale-LILY/ROSE",
            "files_downloaded": FILES,
            "files_sha256": {n: sha256_file(os.path.join(RAW_DIR, n)) for n in FILES
                             if os.path.exists(os.path.join(RAW_DIR, n))},
            "newly_fetched_this_run": fetched,
        },
        "license_verification": {
            "checked": "2026-08-10, against the primary sources (re-verification of spec 3.6)",
            "dataset_card": ("no SPDX license tag in the card YAML; Ethical Considerations states "
                             "verbatim: 'This release is for research purposes only in support of "
                             "an academic paper.'"),
            "license_file_in_release": "BSD 3-Clause, Copyright (c) 2023 Salesforce.com, Inc.",
            "verdict_for_running_the_experiment": (
                "PERMITTED - spec 3.6's verdict stands unchanged: a pre-registered academic study "
                "constructing evaluation material and running judge experiments is exactly "
                "'research purposes only in support of an academic paper'."),
            "redistribution_constraint": (
                "unchanged and still open (spec 3.6 / Section 10 Q1): CNN/DailyMail source text is "
                "Apache-2.0 and freely redistributable; XSum source text is BBC-copyrighted with "
                "an 'unknown' license on its own HF card. Because RoSE ships both corpora's text "
                "INLINE (see q2_resolution), the two-tier release's pointer mechanism would point "
                "at RoSE's own release rather than the EdinburghNLP/xsum loader spec 3.6 names. "
                "No decision taken here."),
        },
        "q2_resolution": {
            "question": "does the RoSE release embed source article text, or only document IDs?",
            "answer": ("EMBEDS IT. Each record carries `source` (full article), `reference` (human "
                       "reference summary), `reference_acus` (the ACU list), `example_id`, "
                       "`count_id`, `annotations` (per-system binary acu_labels + acu / "
                       "normalized_acu scores) and `system_outputs`. No join against "
                       "abisee/cnn_dailymail or EdinburghNLP/xsum is required, and neither was "
                       "contacted."),
            "consequence": ("spec 5 step 2's join and its abort condition are vacuous but retained "
                            "as an assertion; spec 3.7's 'joined by document id' lines are moot"),
        },
        "q3_finding": {
            "question": "does RoSE ship a native per-ACU importance / weight / 'Uniformity' field?",
            "answer": ("NOT in the released annotations. reference_acus is a flat list of strings; "
                       "the only per-ACU signal is `annotations[system].acu_labels`, a binary "
                       "did-this-system's-summary-contain-the-ACU vector used for ACU-recall "
                       "scoring. No importance, weight or uniformity key appears anywhere in the "
                       "records."),
            "consequence": ("the Section 3.2 WE_SALIENCE grading pass is NOT redundant. Section 10 "
                            "Q3 additionally asks a colleague to check the ACL 2023 PDF and the repo's "
                            "demo.ipynb - this settles only what the released data contains."),
            "opportunity_not_taken": (
                "acu_labels could give a data-driven difficulty proxy (an ACU that most systems "
                "miss is a hard one). That is NOT the same construct as the pre-registered "
                "reader-importance grade and is not used - flagged as a possible validation "
                "signal for the review, not adopted."),
        },
        "counts": {"records": len(records), "by_corpus": per_split},
        "problems": problems,
        "validation": {"checks": checks, "failed": failed},
        "outputs": ["rose_master/rose_raw.json (gitignored raw cache)",
                    "rose_master/rose_fetch_report.json"],
    }

    json.dump({"generated": date.today().isoformat(), "script": "we_fetch_rose.py",
               "license": "Salesforce/rose - research use only (dataset card); code BSD-3-Clause",
               "n": len(records), "records": records},
              open(OUT_RAW, "w", encoding="utf-8"), indent=1)
    json.dump(report, open(OUT_REPORT, "w", encoding="utf-8"), indent=1)

    print(f"records: {len(records)}  ->  {OUT_RAW}")
    for c, v in per_split.items():
        print(f"  {c:7s} {v['records_kept']:4d} docs, {v['acus_total']:5d} ACUs "
              f"({v['acus_per_doc_min']}-{v['acus_per_doc_max']}/doc), "
              f"source median {v['source_words_median']}w, "
              f"reference median {v['reference_words_median']}w, "
              f"single-ACU docs {v['single_acu_docs']}")
    print("validation:", "PASS" if not failed else f"FAIL {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
