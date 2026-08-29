#!/usr/bin/env python3
"""W-B P0.3 - ACI-Bench / MedVAL-Bench dialogue2note overlap check.

Pre-registered method (specs/w-b-external-anchors.md section 5, step P0.3):
  normalise (lowercase, strip whitespace/punctuation) every MedVAL dialogue2note
  SOURCE dialogue and every ACI-Bench dialogue; flag exact matches and
  near-duplicates (word 8-gram Jaccard > 0.5). Emit wb_overlap_report.json with
  the overlap fraction and matched ID pairs.

Handling rule (specs/w-d-master-dataset.md section 9): flagged ACI-Bench
encounters are TAGGED medval_overlap and excluded from analyses that also use
MedVAL - they are NOT removed from the benchmark.

Inputs:
  master/aci_raw/aci_all.json        - all 207 ACI-Bench encounters (id, split, transcript)
  master/aci_subsample.json          - the drawn 48-encounter master stratum
  external/medval_hf/{train,test}.csv - HF open release of stanfordmimi/MedVAL-Bench
                                        (gitignored raw cache; re-download via curl from
                                        huggingface.co/datasets/stanfordmimi/MedVAL-Bench)
  external/medval_physionet/**/medval_bench.csv
                                     - credentialed PhysioNet release (840 physician-annotated
                                       rows, single CSV, all six tasks). PRIMARY wherever both
                                       releases carry a row (spec 5 P0.1). Optional: absent ->
                                       HF-only check, recorded as such in the report.

The d2n source universe is reconciled BEFORE matching (PhysioNet primary, HF rows kept only if
their content is absent from PhysioNet) so flagged_pairs stays 1:1 with MedVAL items and no
dialogue is counted twice. The report's headline addition is `medval_d2n_rows_non_aci`: the number
of dialogue2note rows that match NO ACI-Bench encounter, i.e. the size of the d2n analysis that
Amendment A1's exclusion would leave standing.

Output: wb_overlap_report.json at the harness root (committed). Flagged pairs carry the SHA-256 of
the matched MedVAL source dialogue (not the dialogue) so wb_medval_ingest.py can join overlap tags
release-agnostically - the `#` column is NOT unique across HF splits (train re-uses 1-2000), so a
row-number join is unsafe.

Deterministic, stdlib-only, no LLM calls. Run: python3 wb_overlap_check.py
"""

import csv
import glob
import hashlib
import json
import os
import re
import sys
from datetime import date

HARNESS = os.path.dirname(os.path.abspath(__file__))
ACI_ALL = os.path.join(HARNESS, "master", "aci_raw", "aci_all.json")
ACI_SUB = os.path.join(HARNESS, "master", "aci_subsample.json")
MEDVAL_DIR = os.path.join(HARNESS, "external", "medval_hf")
PHYSIONET_DIR = os.path.join(HARNESS, "external", "medval_physionet")
OUT = os.path.join(HARNESS, "wb_overlap_report.json")
# Internal v14/v23b tuning splits (second-order caveat, W-B spec P0.3 + W-F spec):
INTERNAL_SPLITS = os.path.join(HARNESS, "..", "..", "experiments",
                               "clinical-hallucination-detection", "splits.json")

NGRAM_N = 8
JACCARD_THRESHOLD = 0.5

_nonalnum = re.compile(r"[^a-z0-9]+")


def normalize_tokens(text):
    """Spec normalisation: lowercase, strip punctuation, collapse whitespace -> token list."""
    return [t for t in _nonalnum.sub(" ", text.lower()).split() if t]


def ngram_set(tokens, n=NGRAM_N):
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def sha256(s):
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def _d2n_rows(all_rows, release, split):
    return [{"release": release, "split": split, "row": r["#"], "task_id": r["id"],
             "input": r["input"], "output": r.get("output") or ""}
            for r in all_rows if r.get("task") == "dialogue2note"]


def load_medval_d2n(medval_dir):
    """All dialogue2note rows from the HF open release, both splits."""
    csv.field_size_limit(10 ** 8)
    rows = []
    split_totals = {}
    for split in ("train", "test"):
        path = os.path.join(medval_dir, f"{split}.csv")
        if not os.path.exists(path):
            sys.exit(f"missing {path} - download the HF release first (see docstring)")
        with open(path, newline="", encoding="utf-8") as f:
            all_rows = list(csv.DictReader(f))
        split_totals[split] = len(all_rows)
        rows.extend(_d2n_rows(all_rows, "hf", split))
    return rows, split_totals


def load_physionet_d2n(directory):
    """dialogue2note rows from the credentialed PhysioNet release.

    Layout differs from HF: ONE `medval_bench.csv` carrying all 840 physician-annotated rows and
    all six tasks, no train/test split, nested inside a version-named directory from the zip. The
    glob absorbs the version directory so a re-download of a later version still resolves.
    """
    csv.field_size_limit(10 ** 8)
    hits = sorted(glob.glob(os.path.join(directory, "**", "medval_bench.csv"), recursive=True))
    if not hits:
        return [], {}, None
    path = hits[0]
    with open(path, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    rel = os.path.relpath(path, HARNESS)
    return _d2n_rows(all_rows, "physionet", "all"), {"all": len(all_rows)}, rel


def reconcile_d2n(pn_rows, hf_rows):
    """PhysioNet primary (spec 5 P0.1): keep every PhysioNet d2n row, and an HF d2n row only when
    its (input, output) content is absent from PhysioNet. Returns (rows, provenance)."""
    if not pn_rows:
        return list(hf_rows), {"basis": "hf_only", "physionet_rows": 0,
                              "hf_rows_kept": len(hf_rows), "hf_rows_dropped_as_duplicate": 0}
    pn_content = {(sha256(r["input"]), sha256(r["output"])) for r in pn_rows}
    hf_new = [r for r in hf_rows if (sha256(r["input"]), sha256(r["output"])) not in pn_content]
    return pn_rows + hf_new, {
        "basis": "physionet_primary",
        "physionet_rows": len(pn_rows),
        "hf_rows_seen": len(hf_rows),
        "hf_rows_kept": len(hf_new),
        "hf_rows_dropped_as_duplicate": len(hf_rows) - len(hf_new),
        "note": ("PhysioNet is primary wherever both releases carry a dialogue; HF d2n rows whose "
                 "(input, output) content already appears in the PhysioNet release are dropped so "
                 "no dialogue is matched or counted twice"),
    }


def internal_tuning_overlap(aci_flagged_ids, sub_affected):
    """Second-order caveat (W-B spec P0.3 note + W-F spec contamination guard):
    v14 (kappa 0.774) and v23b were tuned on the internal splits.json docs, so
    ACI-derived MedVAL items whose encounter sits in that file are additionally
    not contamination-free for W3-adjacent / v14-derived arms."""
    path = os.path.normpath(INTERNAL_SPLITS)
    if not os.path.exists(path):
        return {"available": False, "note": f"{path} not found; cross-reference skipped"}
    splits = json.load(open(path))
    by_split = {}
    for key, split in splits.items():
        by_split.setdefault(split, set()).add(key.split("__")[0])
    flagged = set(aci_flagged_ids)
    out = {"available": True,
           "source": "experiments/clinical-hallucination-detection/splits.json",
           "note": ("flagged ACI ids that also appear in the internal v14/v23b tuning "
                    "splits - these MedVAL d2n items are not contamination-free for "
                    "W3-adjacent arms even beyond the master-substrate double-count")}
    for split, ids in sorted(by_split.items()):
        hit = sorted(flagged & ids)
        out[f"flagged_in_internal_{split}"] = {"n": len(hit), "ids": hit}
    out["subsample_affected_in_internal_any"] = sorted(
        set(sub_affected) & set().union(*by_split.values()))
    return out


def main():
    aci = json.load(open(ACI_ALL))
    sub_ids = [e["id"] for e in json.load(open(ACI_SUB))]
    hf_d2n, split_totals = load_medval_d2n(MEDVAL_DIR)
    pn_d2n, pn_totals, pn_path = load_physionet_d2n(PHYSIONET_DIR)
    medval, provenance = reconcile_d2n(pn_d2n, hf_d2n)

    # Precompute normalized forms
    aci_norm = {}
    for e in aci:
        toks = normalize_tokens(e["transcript"])
        aci_norm[e["id"]] = {
            "split": e["split"],
            "joined": " ".join(toks),
            "ngrams": ngram_set(toks),
            "n_tokens": len(toks),
        }

    # Distinct MedVAL d2n sources (multiple perturbed outputs can share a dialogue)
    distinct_sources = {}
    for m in medval:
        toks = normalize_tokens(m["input"])
        m["joined"] = " ".join(toks)
        m["ngrams"] = ngram_set(toks)
        m["n_tokens"] = len(toks)
        distinct_sources.setdefault(m["joined"], []).append(m)

    flagged_pairs = []
    medval_best = {}   # medval row key -> (best aci id, best jaccard)
    aci_flagged = set()
    medval_matched_rows = set()

    for m in medval:
        key = f"{m['release']}:{m['split']}#{m['row']}"
        best_id, best_j = None, -1.0
        for aid, a in aci_norm.items():
            j = jaccard(m["ngrams"], a["ngrams"])
            if j > best_j:
                best_id, best_j = aid, j
            if j > JACCARD_THRESHOLD:
                flagged_pairs.append({
                    "aci_id": aid,
                    "aci_split": a["split"],
                    "medval_release": m["release"],
                    "medval_split": m["split"],
                    "medval_row": m["row"],
                    "medval_task_id": m["task_id"],
                    # release-agnostic join key for wb_medval_ingest.py: the `#` column is not
                    # unique across HF splits, the source-dialogue hash is
                    "medval_input_sha256": sha256(m["input"]),
                    "jaccard_8gram": round(j, 4),
                    "exact_match_normalized": m["joined"] == a["joined"],
                })
                aci_flagged.add(aid)
                medval_matched_rows.add(key)
        medval_best[key] = {"best_aci_id": best_id, "best_jaccard": round(best_j, 4),
                            "medval_input_sha256": sha256(m["input"])}

    flagged_pairs.sort(key=lambda p: (p["aci_id"], p["medval_release"], p["medval_split"],
                                      int(p["medval_row"])))
    aci_flagged_ids = sorted(aci_flagged)
    sub_affected = sorted(set(sub_ids) & aci_flagged)

    n_exact = sum(1 for p in flagged_pairs if p["exact_match_normalized"])
    best_js = sorted(v["best_jaccard"] for v in medval_best.values())
    unmatched = {k: v for k, v in medval_best.items()
                 if k not in medval_matched_rows}

    # Derivation probe: is MedVAL d2n ACI-Bench-derived?
    n_rows = len(medval)
    n_matched = len(medval_matched_rows)
    release_label = ("the full credentialed PhysioNet release" if pn_d2n
                     else "the HF open release")
    if n_rows and n_matched == n_rows:
        verdict = (f"CONFIRMED: every MedVAL dialogue2note source dialogue in {release_label} "
                   "matches an ACI-Bench encounter above the 0.5 near-duplicate "
                   "threshold - MedVAL d2n is ACI-Bench-derived, as the HF dataset card "
                   "itself states ('dialogue2note: doctor-patient dialogue -> note "
                   "(ACI-Bench dataset)'). There is NO non-ACI d2n remainder to build a "
                   "contamination-free primary d2n analysis on.")
    elif n_rows and n_matched >= 0.8 * n_rows:
        verdict = (f"LARGELY CONFIRMED: {n_matched}/{n_rows} MedVAL d2n sources match an "
                   "ACI-Bench encounter above threshold; the remainder are listed in "
                   "medval_rows_unmatched for manual review.")
    else:
        verdict = (f"NOT CONFIRMED on the HF open release: only {n_matched}/{n_rows} "
                   "d2n sources match above threshold.")

    report = {
        "generated": date.today().isoformat(),
        "script": "wb_overlap_check.py",
        "spec": ("w-b-external-anchors.md section 5 P0.3 (method) + Amendment A1 "
                 "(blocking input to master pair build); w-d-master-dataset.md "
                 "section 9 (handling rule: tag medval_overlap, exclude from "
                 "MedVAL-co-analyses, do not remove from benchmark)"),
        "method": {
            "normalization": "lowercase; non-alphanumeric -> space; whitespace collapse; word tokens",
            "exact_match": "equality of full normalized token sequences",
            "near_match": f"word {NGRAM_N}-gram Jaccard > {JACCARD_THRESHOLD}",
            "comparison": "all ACI encounters x all MedVAL dialogue2note source dialogues (both HF splits)",
        },
        "inputs": {
            "aci_file": "master/aci_raw/aci_all.json",
            "aci_encounters": len(aci),
            "aci_subsample_file": "master/aci_subsample.json",
            "aci_subsample_n": len(sub_ids),
            "medval_source": (
                "PhysioNet credentialed release (PRIMARY) + HF stanfordmimi/MedVAL-Bench open "
                "release" if pn_d2n else
                "HF stanfordmimi/MedVAL-Bench (open release; test.csv + train.csv)"),
            "medval_releases": {
                "hf_open": {"split_row_totals": split_totals, "d2n_rows": len(hf_d2n),
                            "d2n_rows_by_split": {
                                s: sum(1 for m in hf_d2n if m["split"] == s)
                                for s in ("train", "test")}},
                "physionet": {"present": bool(pn_d2n), "file": pn_path,
                              "row_totals": pn_totals, "d2n_rows": len(pn_d2n)},
            },
            "d2n_reconciliation": provenance,
            "medval_d2n_rows": n_rows,
            "medval_d2n_distinct_sources": len(distinct_sources),
        },
        "summary": {
            "aci_encounters_flagged": len(aci_flagged_ids),
            "aci_overlap_fraction": round(len(aci_flagged_ids) / len(aci), 4) if aci else 0.0,
            "medval_d2n_rows_matched": n_matched,
            "medval_d2n_match_fraction": round(n_matched / n_rows, 4) if n_rows else 0.0,
            # THE number Amendment A1's primary d2n analysis would be built on
            "medval_d2n_rows_non_aci": n_rows - n_matched,
            "medval_d2n_distinct_sources_non_aci": len(
                {m["joined"] for m in medval
                 if f"{m['release']}:{m['split']}#{m['row']}" not in medval_matched_rows}),
            "flagged_pairs": len(flagged_pairs),
            "exact_match_pairs": n_exact,
            "near_match_only_pairs": len(flagged_pairs) - n_exact,
            "best_jaccard_per_medval_row_min": best_js[0] if best_js else None,
            "best_jaccard_per_medval_row_max": best_js[-1] if best_js else None,
        },
        "subsample_impact": {
            "description": ("ids from master/aci_subsample.json (the drawn 48 master-stratum "
                            "encounters) that match a MedVAL d2n source - the number W-D's "
                            "substrate decision needs; per the pre-registered rule these get "
                            "TAGGED medval_overlap and excluded from analyses that also use "
                            "MedVAL, not removed from the benchmark"),
            "n_affected": len(sub_affected),
            "affected_ids": sub_affected,
        },
        "aci_flagged_ids_all": aci_flagged_ids,
        "flagged_pairs_detail": flagged_pairs,
        "medval_rows_unmatched": unmatched,
        "internal_tuning_overlap": internal_tuning_overlap(aci_flagged_ids, sub_affected),
        "derivation_probe": {
            "suspicion": ("W-F spec context: MedVAL d2n suspected ACI-Bench-derived; "
                          "v14 (kappa 0.774) and v23b were tuned on ACI-Bench, so ACI-derived "
                          "MedVAL items are not contamination-free for W3-adjacent arms"),
            "verdict": verdict,
        },
        "coverage_caveat": (
            "RESOLVED (PhysioNet release ingested). The open HF test split carries 530 of the "
            "paper's 840 physician-annotated rows; the credentialed PhysioNet release carries all "
            "840. The 310-row difference is NOT extra dialogue2note data - it is two tasks the "
            "open release omits entirely (impression2simplified 190, bhc2spanish 120). "
            "dialogue2note is 85 rows in BOTH releases, the same 85 dialogues, so the full "
            "release adds no d2n coverage whatever. The earlier hope that the withheld PhysioNet "
            "rows would supply non-ACI d2n sources is closed: they do not exist."
            if pn_d2n else
            "The HF open release test split has 530 physician-annotated rows, not the "
            "paper's 840; its 85 dialogue2note rows are an open subset. The full "
            "physician-labeled set on PhysioNet may contain additional d2n sources not "
            "present here; re-run this script after the PhysioNet download (P0.1) to "
            "extend coverage. The HF train split (2000 rows) contains zero dialogue2note "
            "items, so the open-release d2n universe is exactly the 85 test rows."
        ),
        "dua": ("PhysioNet raw text is read from the gitignored external/medval_physionet/ cache "
                "at run time and never written here: this report carries row numbers, task ids, "
                "ACI ids, SHA-256 hashes and Jaccard scores only."),
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {OUT}")
    print(f"ACI flagged: {len(aci_flagged_ids)}/{len(aci)} "
          f"({report['summary']['aci_overlap_fraction']:.1%})")
    print(f"subsample-48 affected: {len(sub_affected)} -> {sub_affected}")
    print(f"d2n universe: {provenance['basis']} "
          f"(physionet {len(pn_d2n)}, hf kept {provenance['hf_rows_kept']}) -> {n_rows} rows")
    print(f"MedVAL d2n rows matched: {n_matched}/{n_rows} "
          f"(exact pairs {n_exact}, near-only {len(flagged_pairs) - n_exact})")
    print(f"MedVAL d2n rows NON-ACI: {n_rows - n_matched}")
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
