#!/usr/bin/env python3
"""W-B P0.4 - MEDEC MS-subset ingest, structural verification, clean-twin build, n=300 sample.

Pre-registered method: specs/w-b-external-anchors.md
  section 2(b)   - flag mode (primary) + paired mode (secondary); stratified random sample n=300
                   from the MS TEST split, seed 20260728, stratified on error-flag
  section 3.2    - the medec_flag / medec_clean_twin adapters; field names PIN-AT-INGEST
  section 5 P0.4 - "pull MS-subset, pin field names, verify one-error-or-none structure and the
                   erroneous-sentence annotations, reconstruct clean twins, draw the stratified
                   n=300 sample (seed 20260728), freeze wb_medec_items.json"

Deterministic, stdlib-only, NO LLM calls.

Source: github.com/abachaa/MEDEC, directory MEDEC-MS, file
        MEDEC-MS-TestSet-with-GroundTruth-and-ErrorType.csv
        (the split MEDIQA-CORR 2024 systems are ranked on).
License: CC BY 4.0 (repo README "License" section) - text may be redistributed with attribution,
        so unlike the MedVAL distillate this item file carries the clinical text inline.
Citation: Ben Abacha et al., "MEDEC: A Benchmark for Medical Error Detection and Correction in
        Clinical Notes", Findings of ACL 2025.

Inputs
  external/medec/MEDEC-MS-TestSet-with-GroundTruth-and-ErrorType.csv   (gitignored raw cache;
        --fetch downloads it)

Outputs (harness root, committed)
  wb_medec_items.json          frozen items: text, sentence list, error annotations, clean twins,
                               the n=300 sample membership flag, structural-verification flags
  wb_medec_ingest_report.json  field pins, structural verification, sample strata, caveats

Run: python3 wb_medec_ingest.py [--fetch] [--n 300] [--force]
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import urllib.request
from datetime import date

HARNESS = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HARNESS, "external", "medec")
RAW_CSV = os.path.join(RAW_DIR, "MEDEC-MS-TestSet-with-GroundTruth-and-ErrorType.csv")
RAW_URL = ("https://raw.githubusercontent.com/abachaa/MEDEC/main/MEDEC-MS/"
           "MEDEC-MS-TestSet-with-GroundTruth-and-ErrorType.csv")

OUT_ITEMS = os.path.join(HARNESS, "wb_medec_items.json")
OUT_REPORT = os.path.join(HARNESS, "wb_medec_ingest_report.json")

SEED = 20260728          # spec section 5, base seed
N_SAMPLE_DEFAULT = 300   # spec section 2(b); the second degrade lever is n=300 -> n=200 (A4)

FIELD_PINS = {
    "text_id": "Text ID",
    "text": "Text",
    "sentences": "Sentences",
    "error_flag": "Error Flag",
    "error_type": "Error Type",
    "error_sentence_id": "Error Sentence ID",
    "error_sentence": "Error Sentence",
    "corrected_sentence": "Corrected Sentence",
    "corrected_text": "Corrected Text",
}

NUMBERED = re.compile(r"^\s*(\d+)\s+(.*)$")


def ws(s):
    return re.sub(r"\s+", " ", s or "").strip()


def alnum(s):
    return re.sub(r"\s+", "", s or "")


def parse_sentences(blob):
    """MEDEC ships `Sentences` as '<id> <sentence>' lines.

    The repo README warns the raw source carries "unintended line breaks": 8 of the 597 MS-test
    rows split a lab-value line mid-sentence, producing an unnumbered continuation line. Those are
    folded back into the preceding numbered sentence (lossless; recorded per item as
    continuation_lines_folded). Sentence ids are preserved as the dataset assigns them and are
    never renumbered - one row (ms-test-529) skips an id, and Error Sentence ID indexes the
    dataset's numbering, not our position.
    """
    out, folded = [], 0
    for line in (blob or "").replace("\r\n", "\n").split("\n"):
        if not line.strip():
            continue
        m = NUMBERED.match(line)
        if m:
            out.append([int(m.group(1)), m.group(2)])
        elif out:
            out[-1][1] = out[-1][1].rstrip() + " " + line.strip()
            folded += 1
        else:
            out.append([-1, line])
    return [(i, t) for i, t in out], folded


def fetch(force=False):
    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(RAW_CSV) and not force:
        return False
    urllib.request.urlretrieve(RAW_URL, RAW_CSV)
    urllib.request.urlretrieve("https://raw.githubusercontent.com/abachaa/MEDEC/main/README.md",
                               os.path.join(RAW_DIR, "README.md"))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download the MS test CSV if missing")
    ap.add_argument("--n", type=int, default=N_SAMPLE_DEFAULT,
                    help="sample size (spec 2(b) n=300; A4's second degrade lever is n=200)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing frozen item file")
    args = ap.parse_args()

    if args.fetch:
        fetch()
    if not os.path.exists(RAW_CSV):
        sys.exit(f"missing {RAW_CSV} - re-run with --fetch")
    if os.path.exists(OUT_ITEMS) and not args.force:
        sys.exit(f"{OUT_ITEMS} already frozen - re-run with --force only before judging starts")

    csv.field_size_limit(10 ** 8)
    with open(RAW_CSV, newline="", encoding="utf-8-sig") as f:
        raw = list(csv.DictReader(f))

    # The shipped CSV pads to 925 rows: 597 MS-test records plus 328 entirely-empty rows, one per
    # access-restricted UW-collection text (spec 2(b) already notes the UW half is unavailable, so
    # exact MEDIQA-CORR leaderboard comparability is impossible either way).
    blank = [r for r in raw if not any((v or "").strip() for v in r.values())]
    rows = [r for r in raw if (r.get(FIELD_PINS["text_id"]) or "").strip()]
    partial = len(raw) - len(rows) - len(blank)

    items, problems = [], []
    for r in rows:
        tid = r[FIELD_PINS["text_id"]].strip()
        flag_raw = (r[FIELD_PINS["error_flag"]] or "").strip()
        if flag_raw not in ("0", "1"):
            problems.append({"text_id": tid, "issue": f"error_flag not 0/1: {flag_raw!r}"})
            continue
        error_flag = flag_raw == "1"
        sents, folded = parse_sentences(r[FIELD_PINS["sentences"]])
        sid_map = {i: t for i, t in sents}
        ids = [i for i, _ in sents]
        try:
            esid = int((r[FIELD_PINS["error_sentence_id"]] or "").strip())
        except ValueError:
            esid = None
        err_sent = (r[FIELD_PINS["error_sentence"]] or "").strip()
        corr_sent = (r[FIELD_PINS["corrected_sentence"]] or "").strip()
        corr_text_shipped = (r[FIELD_PINS["corrected_text"]] or "").strip()

        # ---- one-error-or-none structure (spec 5 P0.4) --------------------------------
        if error_flag:
            struct_ok = esid is not None and esid >= 0 and esid in sid_map
            sentence_matches = struct_ok and ws(sid_map[esid]) == ws(err_sent)
            has_correction = bool(corr_sent) and corr_sent.upper() != "NA"
        else:
            struct_ok = esid == -1
            sentence_matches = None
            has_correction = None
        if not struct_ok:
            problems.append({"text_id": tid, "error_flag": error_flag,
                             "issue": f"error_sentence_id {esid!r} does not resolve"})
        if error_flag and not sentence_matches:
            problems.append({"text_id": tid, "issue": "Sentences[error_sentence_id] != "
                                                      "Error Sentence"})
        if error_flag and not has_correction:
            problems.append({"text_id": tid, "issue": "errored row with no corrected sentence"})

        # ---- clean twin (spec 3.2 medec_clean_twin) -----------------------------------
        # The pre-registered adapter joins the substituted sentence list with "\n"; the dataset's
        # own `Text` is space-joined. Both renderings are frozen so paired mode can be run with
        # clean and errored formatted IDENTICALLY (see the report's formatting_symmetry flag).
        if error_flag and struct_ok:
            subst = [(corr_sent if i == esid else t) for i, t in sents]
            clean_nl = "\n".join(subst)
            clean_sp = " ".join(subst)
            errored_nl = "\n".join(t for _, t in sents)
            errored_sp = " ".join(t for _, t in sents)
            matches_shipped = ws(clean_sp) == ws(corr_text_shipped)
            alnum_identical = alnum(clean_sp) == alnum(corr_text_shipped)
            # order-insensitive: same characters, possibly a relocated sentence
            content_identical = sorted(alnum(clean_sp)) == sorted(alnum(corr_text_shipped))
        else:
            clean_nl = clean_sp = errored_nl = errored_sp = None
            matches_shipped = alnum_identical = content_identical = None

        items.append({
            "item_id": f"medec_ms_test_{tid}",
            "text_id": tid,
            "split": "ms_test",
            "text": r[FIELD_PINS["text"]].strip(),
            "error_flag": error_flag,
            "error_type": (r[FIELD_PINS["error_type"]] or "").strip() or None,
            "error_sentence_id": esid,
            "error_sentence": err_sent or None,
            "corrected_sentence": corr_sent if (corr_sent and corr_sent.upper() != "NA") else None,
            "sentences": [[i, t] for i, t in sents],
            "n_sentences": len(sents),
            "clean_twin": clean_nl,
            "clean_twin_spacejoin": clean_sp,
            "errored_sentence_join": errored_nl,
            "errored_spacejoin": errored_sp,
            "corrected_text_shipped": corr_text_shipped or None,
            "clean_twin_matches_shipped": matches_shipped,
            "clean_twin_alnum_identical": alnum_identical,
            "clean_twin_content_identical": content_identical,
            "sentence_ids_contiguous": ids == list(range(len(ids))),
            "continuation_lines_folded": folded,
            "structure_ok": bool(struct_ok),
            "in_sample": False,
        })

    # ---- stratified n=300 sample on the error flag (spec 2(b)) -------------------------
    strata = {"errored": [it for it in items if it["error_flag"] and it["structure_ok"]],
              "error_free": [it for it in items if not it["error_flag"] and it["structure_ok"]]}
    total = sum(len(v) for v in strata.values())
    n_target = min(args.n, total)
    exact = {k: n_target * len(v) / total for k, v in strata.items()}
    alloc = {k: int(v) for k, v in exact.items()}
    for k in sorted(strata, key=lambda k: -(exact[k] - alloc[k]))[:n_target - sum(alloc.values())]:
        alloc[k] += 1
    rng = random.Random(SEED)
    sample = []
    for k in sorted(strata):
        members = sorted(strata[k], key=lambda it: it["text_id"])
        rng.shuffle(members)
        sample.extend(members[:alloc[k]])
    sample_ids = {it["item_id"] for it in sample}
    for it in items:
        it["in_sample"] = it["item_id"] in sample_ids

    # ---- validation --------------------------------------------------------------------
    errored = [it for it in items if it["error_flag"]]
    checks = {
        "every_row_has_error_flag_0_or_1": len(items) == len(rows),
        "every_errored_row_resolves_its_error_sentence":
            all(it["structure_ok"] for it in errored),
        "every_errored_row_sentence_matches_annotation":
            all(ws(dict(it["sentences"]).get(it["error_sentence_id"], "")) == ws(it["error_sentence"])
                for it in errored),
        "every_errored_row_has_a_clean_twin": all(it["clean_twin"] for it in errored),
        "clean_twin_content_identical_to_shipped_correction":
            all(it["clean_twin_content_identical"] for it in errored),
        "error_free_rows_have_no_twin":
            all(it["clean_twin"] is None for it in items if not it["error_flag"]),
        "sample_size": len(sample) == n_target,
        "sample_strata_proportional":
            abs(sum(1 for it in sample if it["error_flag"]) / n_target
                - len(strata["errored"]) / total) < 0.01,
        "item_ids_unique": len({it["item_id"] for it in items}) == len(items),
    }
    failed = [k for k, v in checks.items() if not v]

    by_type = {}
    for it in items:
        by_type[it["error_type"] or "(none)"] = by_type.get(it["error_type"] or "(none)", 0) + 1
    sample_by_type = {}
    for it in sample:
        k = it["error_type"] or "(none)"
        sample_by_type[k] = sample_by_type.get(k, 0) + 1

    n_ws_only = sum(1 for it in errored if not it["clean_twin_matches_shipped"])
    report = {
        "generated": date.today().isoformat(),
        "script": "wb_medec_ingest.py",
        "spec": "w-b-external-anchors.md sections 2(b), 3.2, 5 P0.4",
        "seed": SEED,
        "source": {
            "repo": "https://github.com/abachaa/MEDEC",
            "file": "MEDEC-MS/MEDEC-MS-TestSet-with-GroundTruth-and-ErrorType.csv",
            "license": "CC BY 4.0 (repo README, License section) - verified 2026-08-10",
            "citation": ("Ben Abacha, Yim, Fu, Sun, Yetisgen, Xia, Lin. MEDEC: A Benchmark for "
                         "Medical Error Detection and Correction in Clinical Notes. Findings of "
                         "ACL 2025, 22539-22550."),
            "uw_collection": ("the UW half of the official test set is access-restricted (DUA via "
                              "UW) and is NOT ingested; the shipped CSV pads it out as "
                              f"{len(blank)} entirely-empty rows, dropped here. Spec 2(b) already "
                              "states exact MEDIQA-CORR leaderboard comparability is impossible "
                              "for this reason - published numbers are reference points only."),
        },
        "field_pins": FIELD_PINS,
        "counts": {
            "csv_rows": len(raw),
            "blank_uw_placeholder_rows_dropped": len(blank),
            "partially_populated_rows": partial,
            "ms_test_rows": len(rows),
            "items_frozen": len(items),
            "errored": len(errored),
            "error_free": len(items) - len(errored),
            "by_error_type": by_type,
        },
        "structure_verification": {
            "one_error_or_none": ("every MS-test row carries exactly one error sentence or none: "
                                  "error-free rows all have Error Sentence ID -1 and 'NA' in both "
                                  "correction fields"),
            "rows_with_folded_continuation_lines":
                sum(1 for it in items if it["continuation_lines_folded"]),
            "rows_with_non_contiguous_sentence_ids":
                [it["text_id"] for it in items if not it["sentence_ids_contiguous"]],
            "problems": problems,
        },
        "clean_twin": {
            "method": ("spec 3.2 medec_clean_twin - substitute the annotator's corrected sentence "
                       "for the flagged sentence in the sentence list"),
            "cross_check": ("compared against the dataset's own shipped Corrected Text column, "
                            "which the spec's adapter does not use"),
            "exact_after_whitespace_normalisation": len(errored) - n_ws_only,
            "whitespace_only_differences": n_ws_only,
            "sentence_position_differences":
                [it["text_id"] for it in errored
                 if it["clean_twin_content_identical"] and not it["clean_twin_alnum_identical"]],
            "content_differences":
                [it["text_id"] for it in errored if not it["clean_twin_content_identical"]],
            "note": ("the whitespace-only differences are all inside lab-value runs ('4,500 /mm3' "
                     "vs '4,500/mm3') where MEDEC's own sentence splitter moved a space. One row "
                     "(ms-test-273) additionally differs by POSITION: MEDEC's shipped Corrected "
                     "Text relocates the corrected sentence, while the pre-registered in-place "
                     "substitution keeps it at the flagged sentence's index - same characters, "
                     "different order. The spec's method governs; no row differs in content."),
        },
        "formatting_symmetry_FLAG": (
            "spec 3.2's medec_clean_twin joins the substituted sentence list with '\\n', while "
            "flag mode feeds the dataset's space-joined `Text`. Running paired mode with a "
            "newline-joined clean twin against a space-joined errored text would vary formatting "
            "across the pair as well as content. Both renderings are therefore frozen per item "
            "(clean_twin / clean_twin_spacejoin and errored_sentence_join / errored_spacejoin) so "
            "the runner can hold formatting constant. Which rendering paired mode uses is a "
            "judgement call left OPEN for review - nothing here decides it."),
        "sample": {
            "n_requested": args.n,
            "n_drawn": len(sample),
            "stratified_on": "error flag (spec 2(b))",
            "strata": {k: {"available": len(v), "drawn": alloc[k]} for k, v in strata.items()},
            "by_error_type": sample_by_type,
            "flag_mode_items": len(sample),
            "paired_mode_items": sum(1 for it in sample if it["error_flag"]),
        },
        "validation": {"checks": checks, "failed": failed},
        "outputs": ["wb_medec_items.json", "wb_medec_ingest_report.json"],
    }

    frozen = {
        "generated": date.today().isoformat(),
        "script": "wb_medec_ingest.py",
        "spec": "w-b-external-anchors.md sections 2(b), 3.2, 5 P0.4 (frozen BEFORE any judging)",
        "seed": SEED,
        "license": "CC BY 4.0 - Ben Abacha et al., Findings of ACL 2025 (github.com/abachaa/MEDEC)",
        "field_pins": FIELD_PINS,
        "n": len(items),
        "n_in_sample": len(sample),
        "items": items,
    }
    json.dump(frozen, open(OUT_ITEMS, "w", encoding="utf-8"), indent=1)
    json.dump(report, open(OUT_REPORT, "w", encoding="utf-8"), indent=1)

    print(f"items frozen: {len(items)} ({len(errored)} errored / {len(items) - len(errored)} "
          f"error-free)  ->  {OUT_ITEMS}")
    print(f"sample n={len(sample)}: " + ", ".join(f"{k} {alloc[k]}" for k in sorted(alloc)))
    print(f"clean twins: {len(errored)} built, "
          f"{len(report['clean_twin']['content_differences'])} content differences vs shipped "
          f"({len(report['clean_twin']['sentence_position_differences'])} position-only)")
    print("validation:", "PASS" if not failed else f"FAIL {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
