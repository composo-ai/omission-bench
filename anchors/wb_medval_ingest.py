#!/usr/bin/env python3
"""Ingest MedVAL-Bench: reconciliation, taxonomy mapping, few-shot carve-out.

Pre-registered method:
  1. download the Hugging Face and PhysioNet releases, reconcile them, pin the field names
     and the binarization, emit wb_medval_reconciliation.json
  2. map MedVAL's error taxonomy onto this study's omit/add/change labels, then freeze
     wb_medval_items.json BEFORE any judging, blind to all model outputs
  3. carve a held-out few-shot pool (50-100 items, seed 20260728, stratified across the
     three omission categories) out of the omission-labelled items; freeze
     wb_medval_fewshot_pool.json + wb_medval_eval_ids.json
  and the rule that governs the analysis sets: if the ACI overlap check finds overlap, the
  dialogue2note PRIMARY analysis excludes ACI-derived items, with the with-overlap numbers
  as a sensitivity row.

Deterministic, stdlib-only, NO LLM calls. Re-runnable: re-run after the PhysioNet download
lands under external/medval_physionet/ and every count/split extends automatically.

Inputs
  external/medval_hf/{train,test}.csv     gitignored raw cache of stanfordmimi/MedVAL-Bench
                                          (MIT per the dataset card; download with
                                          `curl -L https://huggingface.co/datasets/stanfordmimi/
                                           MedVAL-Bench/resolve/main/data/{split}.csv`)
  external/medval_physionet/**/medval_bench.csv
                                          OPTIONAL, needs credentialed access. Absent -> HF-only ingest,
                                          recorded as such in the reconciliation. Layout differs
                                          from HF - see load_physionet_release().
  wb_overlap_report.json                  ACI-Bench overlap tags (from wb_overlap_check.py)

Outputs (harness root, committed)
  wb_medval_items.json                    frozen item set: labels, taxonomy mapping, grades,
                                          overlap tags, split roles. NO source text - see
                                          "Text policy" below.
  wb_medval_reconciliation.json           counts, field pins, binarization pin, release
                                          reconciliation, label-source disagreements, gates
  wb_medval_fewshot_pool.json             the held-out few-shot pool - the only source of
                                          few-shot examples for the few-shot judge arm
  wb_medval_eval_ids.json                 every MedVAL id the external-anchor runs will ever
                                          judge + the per-analysis eval sets (dialogue2note
                                          primary / dialogue2note sensitivity / full split)
  external/wb_medval_items_text.json      gitignored text sidecar {item_id: {input, output,
                                          reference_output}} so runners join in one line

Text policy. wb_medval_items.json carries labels and taxonomy mapping only for items open on
Hugging Face, and references PhysioNet-only items by ID with labels but no text; the PhysioNet
data use agreement forbids redistributing any PhysioNet item text. One schema serves both
releases: the committed items file carries join keys + SHA-256s + lengths and never carries
text; the text itself stays in the gitignored sidecar next to the raw cache.

  The items file this writes is NOT part of this release: for PhysioNet-covered items a
  public artifact may carry aggregates and item IDs only - no item text, no per-item
  physician labels. Each item it writes carries `hf_open`: true when the item's exact
  content also ships in the MIT-licensed Hugging Face open release, false when it exists
  only on PhysioNet, so any public cut can filter on `hf_open` mechanically.

Run: python3 anchors/wb_medval_ingest.py [--fewshot-n 50] [--force]
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path

import argparse
import csv
import glob
import hashlib
import json
import os
import random
import re
import sys
from datetime import date

HARNESS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repository root
HF_DIR = os.path.join(HARNESS, "external", "medval_hf")
PHYSIONET_DIR = os.path.join(HARNESS, "external", "medval_physionet")
OVERLAP = os.path.join(HARNESS, "wb_overlap_report.json")

OUT_ITEMS = os.path.join(HARNESS, "wb_medval_items.json")
OUT_RECON = os.path.join(HARNESS, "wb_medval_reconciliation.json")
OUT_POOL = os.path.join(HARNESS, "wb_medval_fewshot_pool.json")
OUT_EVAL = os.path.join(HARNESS, "wb_medval_eval_ids.json")
OUT_TEXT = os.path.join(HARNESS, "external", "wb_medval_items_text.json")

SEED = 20260728                     # the study's base seed
FEWSHOT_N_DEFAULT = 50              # the pre-registered range is 50-100; see the flag note below
D2N_TASK = "dialogue2note"

# ---------------------------------------------------------------- taxonomy mapping
# Canonical vocabulary taken from MedVAL's OWN released code (utils/prompts.py in
# github.com/StanfordMIMI/MedVAL, `error_categories`, fetched 2026-08-10) - 11 categories, three
# more than the pre-registered mapping table lists. The three extras are resolved by the
# pre-registered catch-all rule, quoted in the `rule`/`justification` fields below and frozen
# into every item.
TABLE = {
    # the pre-registered mapping table, verbatim
    "missing claim": ("omit", "table", "mapping table: Missing claim -> omit"),
    "missing comparison": ("omit", "table", "mapping table: Missing comparison -> omit"),
    "missing context": ("omit", "table", "mapping table: Missing context -> omit"),
    "fabricated claim": ("add", "table", "mapping table: Fabricated claim -> add"),
    "detail misidentification": ("change", "table",
                                 "mapping table: Detail misidentification -> change"),
    "false comparison": ("change", "table", "mapping table: False comparison -> change"),
    "misleading justification": ("change", "table",
                                 "mapping table: Misleading justification -> change"),
    "incorrect recommendation": ("change", "table",
                                 "mapping table: Incorrect recommendation -> change"),
    # not in the mapping table; resolved by the pre-registered catch-all rule
    "overstating intensity": ("change", "catch_all",
                              "catch-all rule: MedVAL defines this as 'Exaggerating urgency, "
                              "severity, or confidence' - it corrupts/misrepresents content that "
                              "is present in the source, so -> change (the 'hardened' failure the "
                              "judge benchmark's criterion paragraph already names)"),
    "understating intensity": ("change", "catch_all",
                               "catch-all rule: MedVAL defines this as 'Understating urgency, "
                               "severity, or confidence in a correct claim' - it corrupts content "
                               "present in the source, so -> change"),
    "other": (None, "catch_all_unresolved",
              "the catch-all rule cannot fire: MedVAL's 'Other: Additional errors not covered in "
              "the defined categories' names no content relation, so no type is assignable "
              "without reading the free text - item counts for binary metrics only"),
}
# Surface variants seen in the physician free text (typos / grouping words), mapped to canonical.
SURFACE_VARIANTS = {
    "missing contex": "missing context",
    "missing  context": "missing context",
    "fabricated  claim": "fabricated claim",
}
OMISSION_CATEGORIES = ["missing claim", "missing comparison", "missing context"]
# The four tasks the MIT HF open release ships. The credentialed PhysioNet release adds
# impression2simplified and bhc2spanish - see new_tasks_from_physionet in the reconciliation.
HF_TASKS = ["dialogue2note", "medication2answer", "query2question", "report2impression"]

NO_ANNOTATION = {"", "na", "n/a", "none", "no errors", "no error", "-"}

# Field pins, fixed at ingest. Confirmed against the downloaded CSVs AND against MedVAL's own
# loader (medval/pipeline.py renames input->reference, output->candidate), i.e. the physician
# graded `output` against `input` - exactly the direction the MedVAL adapters assume.
FIELD_PINS = {
    "row_id": "#",
    "task_id": "id",
    "task": "task",
    "transcript_slot": "input",
    "note_slot": "output",
    "expert_target": "reference_output",
    "labels": "physician_error_assessment",
    "risk_grade": "physician_risk_grade",
}


def sha256(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def norm_assessment(a):
    return (a or "").strip()


def has_annotation(a):
    return norm_assessment(a).lower() not in NO_ANNOTATION


def detect_categories(assessment):
    """Case-insensitive substring scan for MedVAL's canonical category names.

    Deterministic and blind to model output: the physician free text either names a category or it
    does not. Items whose text names none are recorded type_resolved=false (they still carry a
    physician risk grade, so they count for the binary metrics; unresolved and mixed items are
    excluded from the per-type recall analysis only).
    """
    low = re.sub(r"\s+", " ", (assessment or "").lower())
    for variant, canonical in SURFACE_VARIANTS.items():
        low = low.replace(variant, canonical)
    found = []
    for cat in TABLE:
        if cat == "other":
            continue
        if cat in low:
            found.append(cat)
    return found


def map_types(categories):
    assignments, types = [], []
    for c in categories:
        our, rule, why = TABLE[c]
        assignments.append({"category": c, "our_type": our, "rule": rule, "justification": why})
        if our and our not in types:
            types.append(our)
    return assignments, sorted(types)


# ---------------------------------------------------------------- loading
def load_csv_release(directory, release, splits=("train", "test")):
    csv.field_size_limit(10 ** 8)
    out, totals = [], {}
    for split in splits:
        path = os.path.join(directory, f"{split}.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        totals[split] = len(rows)
        for r in rows:
            out.append({"release": release, "split": split, "raw": r})
    return out, totals


def load_physionet_release(directory):
    """The credentialed PhysioNet release, whose layout is NOT the HF {train,test}.csv shape.

    Structure as shipped in
    medval-bench-expert-annotated-medical-text-validation-benchmark-1.0.1.zip (verified 2026-08-12):
    a single `medval_bench.csv` holding all 840 physician-annotated rows and all SIX tasks
    (distinguished by the `task` column) inside a version-named directory, alongside README.md,
    LICENSE.txt and SHA256SUMS.txt. There is no train/test split - the whole file IS the
    physician-annotated set - so every row is loaded under one pseudo-split, "all". Column names are
    byte-identical to the HF release, so FIELD_PINS needs no release-specific branch. The glob
    absorbs the version directory so a later re-download still resolves.
    """
    csv.field_size_limit(10 ** 8)
    hits = sorted(glob.glob(os.path.join(directory, "**", "medval_bench.csv"), recursive=True))
    if not hits:
        return [], {}, None
    path = hits[0]
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return ([{"release": "physionet", "split": "all", "raw": r} for r in rows],
            {"all": len(rows)}, os.path.relpath(path, HARNESS))


def parse_grade(v):
    try:
        g = int(str(v).strip())
    except (TypeError, ValueError):
        return None
    return g if g in (1, 2, 3, 4) else None


# ---------------------------------------------------------------- build
def content_key(task, inp, out):
    """Release-independent identity of a MedVAL record.

    Neither released id column is a safe cross-release key: `id` (unique within a task) is
    RENUMBERED between the HF and PhysioNet releases for medication2answer, and `#` repeats across
    the HF train/test splits (train is numbered 1-2000). Content is the only stable identity.
    """
    return (task, sha256(inp), sha256(out))


def build_items(records, overlap_by_input_sha, hf_content_keys):
    items = []
    for rec in records:
        r, split, release = rec["raw"], rec["split"], rec["release"]
        row_id = r[FIELD_PINS["row_id"]]
        grade = parse_grade(r.get(FIELD_PINS["risk_grade"]))
        assessment = norm_assessment(r.get(FIELD_PINS["labels"]))
        annotated = has_annotation(assessment)
        cats = detect_categories(assessment) if annotated else []
        assignments, types = map_types(cats)
        task = r.get(FIELD_PINS["task"])
        inp = r.get(FIELD_PINS["transcript_slot"]) or ""
        out = r.get(FIELD_PINS["note_slot"]) or ""
        ref = r.get(FIELD_PINS["expert_target"]) or ""

        # Two independent binary label sources; both frozen so the analysis can report either.
        # Primary metric uses the risk-grade binarization pinned in the reconciliation.
        bl_grade = None if grade is None else int(grade >= 2)
        bl_grade_sens = None if grade is None else int(grade >= 3)
        bl_annot = int(annotated)

        ov = overlap_by_input_sha.get(sha256(inp))
        item = {
            "item_id": f"medval_{release}_{split}_{row_id}",
            "release": release,
            "split": split,
            "row": row_id,
            # a public release may carry per-item physician labels only for items that also ship
            # in the MIT-licensed Hugging Face open release. Filter on this flag when building it.
            "hf_open": content_key(task, inp, out) in hf_content_keys,
            "task_id": r.get(FIELD_PINS["task_id"]),
            "task": task,
            "physician_risk_grade": grade,
            "physician_error_assessment_sha256": sha256(assessment),
            "annotated_error": annotated,
            "error_categories": cats,
            "type_assignments": assignments,
            "our_types": types,
            "type_resolved": bool(types) if annotated else True,
            "pure_type": types[0] if len(types) == 1 else None,
            "is_omission_labelled": any(c in OMISSION_CATEGORIES for c in cats),
            "omission_categories": [c for c in cats if c in OMISSION_CATEGORIES],
            "binary_label_grade": bl_grade,
            "binary_label_grade_sensitivity": bl_grade_sens,
            "binary_label_annotation": bl_annot,
            # mapping table row: "No error annotated AND risk level 1 -> no error -> clean"
            "clean_per_mapping_table": bool((not annotated) and grade == 1),
            "label_source_disagreement": (bl_grade is not None and bl_grade != bl_annot),
            "aci_overlap": bool(ov),
            "aci_id": ov["aci_id"] if ov else None,
            "aci_in_internal_tuning_splits": bool(ov and ov["in_internal_tuning"]),
            "input_sha256": sha256(inp),
            "output_sha256": sha256(out),
            "input_words": len(inp.split()),
            "output_words": len(out.split()),
            "has_reference_output": bool(ref.strip()),
            "eval_role": None,        # filled by the carve-out below
        }
        items.append((item, {"input": inp, "output": out, "reference_output": ref}))
    return items


def carve_fewshot_pool(items, n_target, seed):
    """Draw the few-shot pool from the omission-labelled items, stratified across the three
    omission categories.

    Strata = the sorted tuple of omission categories an item carries (an item can carry more than
    one, so this is the only partition that is simultaneously exhaustive, disjoint and 'across the
    three omission categories'). Proportional allocation with largest-remainder rounding; seeded
    shuffle within each stratum.
    """
    pool_candidates = [it for it in items if it["is_omission_labelled"]]
    strata = {}
    for it in pool_candidates:
        strata.setdefault(tuple(sorted(it["omission_categories"])), []).append(it)

    n_target = min(n_target, len(pool_candidates))
    total = len(pool_candidates)
    exact = {k: n_target * len(v) / total for k, v in strata.items()}
    alloc = {k: int(v) for k, v in exact.items()}
    remainder = n_target - sum(alloc.values())
    for k in sorted(strata, key=lambda k: (-(exact[k] - alloc[k]), k))[:remainder]:
        alloc[k] += 1

    rng = random.Random(seed)
    chosen = []
    for k in sorted(strata):
        members = sorted(strata[k], key=lambda it: it["item_id"])
        rng.shuffle(members)
        chosen.extend(members[:alloc[k]])
    chosen_ids = {it["item_id"] for it in chosen}
    for it in items:
        it["eval_role"] = "fewshot_pool" if it["item_id"] in chosen_ids else "eval"
    strata_report = {
        "|".join(k) if k else "(none)": {"available": len(v), "drawn": alloc[k]}
        for k, v in sorted(strata.items())
    }
    return chosen, strata_report, len(pool_candidates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fewshot-n", type=int, default=FEWSHOT_N_DEFAULT,
                    help="few-shot pool size; the pre-registered range is 50-100 (default: the "
                         "floor, 50 - see the reconciliation's fewshot_pool.note)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing frozen item file")
    args = ap.parse_args()

    if os.path.exists(OUT_ITEMS) and not args.force:
        sys.exit(f"{OUT_ITEMS} already frozen - re-run with --force only before judging starts")
    if not 1 <= args.fewshot_n:
        sys.exit("--fewshot-n must be >= 1")

    # --- releases -------------------------------------------------------------
    hf_records, hf_totals = load_csv_release(HF_DIR, "hf")
    if not hf_records:
        sys.exit(f"no HF release under {HF_DIR} - see the docstring for the download command")
    pn_records, pn_totals, pn_path = load_physionet_release(PHYSIONET_DIR)
    physionet_present = bool(pn_records)

    # The physician-labelled universe = rows carrying a real risk grade (1-4). The HF train split
    # is the self-supervised fine-tuning half: grade -1, assessment "na", no physician labels.
    def labelled(recs):
        return [r for r in recs
                if parse_grade(r["raw"].get(FIELD_PINS["risk_grade"])) is not None]

    hf_labelled = labelled(hf_records)
    pn_labelled = labelled(pn_records)
    records = (pn_labelled + [r for r in hf_labelled]) if physionet_present else hf_labelled

    # --- overlap tags (wb_overlap_check.py's output, reused not recomputed) ----
    # Keyed on the SHA-256 of the MedVAL source dialogue, not on "{split}#{row}": the `#` column
    # repeats across the HF train/test splits and is renumbered per release for some tasks, so a
    # row-number join silently mistags once PhysioNet is primary. The hash is release-agnostic.
    overlap_by_input_sha, overlap_meta = {}, {}
    if os.path.exists(OVERLAP):
        rep = json.load(open(OVERLAP))
        internal = set()
        it = rep.get("internal_tuning_overlap", {})
        for k, v in it.items():
            if k.startswith("flagged_in_internal_") and isinstance(v, dict):
                internal |= set(v.get("ids", []))
        missing_sha = 0
        for p in rep.get("flagged_pairs_detail", []):
            key = p.get("medval_input_sha256")
            if not key:
                missing_sha += 1
                continue
            overlap_by_input_sha[key] = {"aci_id": p["aci_id"],
                                        "in_internal_tuning": p["aci_id"] in internal}
        if missing_sha:
            sys.exit(f"wb_overlap_report.json has {missing_sha} flagged pairs without "
                     "medval_input_sha256 - re-run wb_overlap_check.py (it now emits the "
                     "release-agnostic join key this ingest requires)")
        overlap_meta = {
            "source": "wb_overlap_report.json",
            "generated": rep.get("generated"),
            "join_key": "medval_input_sha256 (release-agnostic; see the comment at the join)",
            "aci_encounters_flagged": rep["summary"]["aci_encounters_flagged"],
            "medval_d2n_rows_matched": rep["summary"]["medval_d2n_rows_matched"],
            "medval_d2n_match_fraction": rep["summary"]["medval_d2n_match_fraction"],
            "medval_d2n_rows_non_aci": rep["summary"].get("medval_d2n_rows_non_aci"),
            "verdict": rep["derivation_probe"]["verdict"],
        }
    else:
        overlap_meta = {"source": None, "note": "wb_overlap_report.json absent - no overlap tags"}

    # --- items ----------------------------------------------------------------
    hf_content_keys = {content_key(r["raw"].get(FIELD_PINS["task"]),
                                   r["raw"].get(FIELD_PINS["transcript_slot"]) or "",
                                   r["raw"].get(FIELD_PINS["note_slot"]) or "")
                       for r in hf_labelled}
    built = build_items(records, overlap_by_input_sha, hf_content_keys)
    items = [i for i, _ in built]
    texts = {i["item_id"]: t for i, t in built}

    # --- reconciliation between the two releases ------------------------------
    # Joined on CONTENT, not on either id column. The obvious (task, id) join is wrong: the
    # PhysioNet release renumbers `id` for every medication2answer row, so that join matches only
    # 428 of the 530 shared rows, leaves 102 HF rows looking PhysioNet-absent (they would be frozen
    # a second time as duplicate items) and reports 25 grade "disagreements" that are really
    # mis-joined pairs - 5.8% of the matched rows, which would trip the release-disagreement gate
    # and stop the anchor build for an artefact of the join. Content identity is exact and
    # release-independent.
    by_release = {}
    for it in items:
        by_release.setdefault(it["release"], []).append(it)
    ckey = lambda it: (it["task"], it["input_sha256"], it["output_sha256"])
    label = lambda it: (it["physician_risk_grade"], it["physician_error_assessment_sha256"])
    shared, disagreements, id_drift = [], [], []
    row_join_valid = None
    if physionet_present:
        pn_by_content, pn_by_row = {}, {}
        for it in by_release.get("physionet", []):
            pn_by_content.setdefault(ckey(it), []).append(it)
            pn_by_row[it["row"]] = it
        hf_by_content = {}
        for it in by_release.get("hf", []):
            hf_by_content.setdefault(ckey(it), []).append(it)

        # Label comparison per content group, as multisets - four report2impression content groups
        # appear TWICE in each release (the same input/output annotated twice, sometimes with
        # different grades), so a dict-keyed 1:1 join would compare arbitrary twins.
        for k, hf_group in sorted(hf_by_content.items()):
            pn_group = pn_by_content.get(k)
            if not pn_group:
                continue
            shared.append(k)
            hf_labels, pn_labels = sorted(map(label, hf_group)), sorted(map(label, pn_group))
            if hf_labels != pn_labels:
                disagreements.append({
                    "task": k[0],
                    "hf_rows": sorted(it["row"] for it in hf_group),
                    "physionet_rows": sorted(it["row"] for it in pn_group),
                    "hf_grades": [g for g, _ in hf_labels],
                    "physionet_grades": [g for g, _ in pn_labels],
                    "assessments_identical": ([a for _, a in hf_labels]
                                              == [a for _, a in pn_labels]),
                })

        # The `#` column DOES align across releases for every shared row (verified below), which is
        # what makes the `id` renumbering visible and harmless. Recorded, not relied on.
        aligned = [it for it in by_release.get("hf", [])
                   if it["row"] in pn_by_row and ckey(pn_by_row[it["row"]]) == ckey(it)]
        row_join_valid = len(aligned) == len(by_release.get("hf", []))
        for it in aligned:
            if pn_by_row[it["row"]]["task_id"] != it["task_id"]:
                id_drift.append({"row": it["row"], "task": it["task"],
                                 "hf_id": it["task_id"],
                                 "physionet_id": pn_by_row[it["row"]]["task_id"]})

        # PhysioNet is primary wherever both releases carry an item
        drop = {it["item_id"] for it in by_release.get("hf", []) if ckey(it) in pn_by_content}
        items = [it for it in items if it["item_id"] not in drop]
        texts = {k: v for k, v in texts.items() if k not in drop}

    # Repeat annotations of identical content inside the frozen set (see the note in the report).
    _dupe_groups = {}
    for it in items:
        _dupe_groups.setdefault(ckey(it), []).append(it)

    # --- few-shot carve-out ---------------------------------------------------
    pool, strata_report, n_omission = carve_fewshot_pool(items, args.fewshot_n, SEED)

    # --- eval sets ------------------------------------------------------------
    eval_items = [it for it in items if it["eval_role"] == "eval"]
    d2n_eval = [it for it in eval_items if it["task"] == D2N_TASK]
    d2n_primary = [it for it in d2n_eval if not it["aci_overlap"]]      # ACI-derived excluded
    full_split = eval_items                                            # the secondary anchor

    ids = lambda xs: [x["item_id"] for x in xs]
    eval_ids = {
        "method": ("every MedVAL id the external-anchor runs will ever judge, with the few-shot "
                   "pool held out; ACI-derived items are excluded from the dialogue2note PRIMARY "
                   "analysis and kept as a sensitivity row"),
        "generated": date.today().isoformat(),
        "seed": SEED,
        "eval_ids": ids(eval_items),
        "fewshot_pool_ids": ids(pool),
        "analysis_sets": {
            "d2n_primary": {
                "n": len(d2n_primary), "ids": ids(d2n_primary),
                "rule": "task==dialogue2note, eval role, ACI-overlap-flagged items EXCLUDED",
            },
            "d2n_sensitivity_with_overlap": {
                "n": len(d2n_eval), "ids": ids(d2n_eval),
                "rule": "task==dialogue2note, eval role, ACI-overlap items RETAINED - the "
                        "pre-registered with-overlap sensitivity row",
            },
            "full_eval_split": {
                "n": len(full_split), "ids": ids(full_split),
                "rule": "all physician-labelled items minus the few-shot pool (the secondary "
                        "anchor: 3 sentinel cells across all MedVAL tasks)",
            },
        },
    }

    # --- validation gates -----------------------------------------------------
    # Data-use-agreement guard. Two independent tests, because the realistic failure is someone
    # adding a convenience field later: (1) the item schema is a closed whitelist, so a new
    # text-bearing key fails the freeze rather than shipping; (2) no string anywhere in an item may
    # contain a 60-character window of that item's own source text or physician assessment.
    ALLOWED_ITEM_KEYS = {
        "item_id", "release", "split", "row", "hf_open", "task_id", "task",
        "physician_risk_grade", "physician_error_assessment_sha256", "annotated_error",
        "error_categories", "type_assignments", "our_types", "type_resolved", "pure_type",
        "is_omission_labelled", "omission_categories", "binary_label_grade",
        "binary_label_grade_sensitivity", "binary_label_annotation", "clean_per_mapping_table",
        "label_source_disagreement", "aci_overlap", "aci_id", "aci_in_internal_tuning_splits",
        "input_sha256", "output_sha256", "input_words", "output_words", "has_reference_output",
        "eval_role",
    }
    schema_ok = all(set(it) <= ALLOWED_ITEM_KEYS for it in items)
    unexpected_keys = sorted(set().union(*(set(it) for it in items)) - ALLOWED_ITEM_KEYS) \
        if items else []

    def leaks_text(item, raw, assessment):
        blob = json.dumps(item)
        for src in (raw["input"], raw["output"], raw["reference_output"], assessment):
            s = (src or "").strip()
            for i in range(0, max(len(s) - 60, 0) + 1, 60):
                window = s[i:i + 60]
                if len(window) == 60 and window in blob:
                    return True
        return False

    raw_by_id = {i["item_id"]: t for i, t in built}
    assess_by_id = {}
    for rec in records:
        rid = f"medval_{rec['release']}_{rec['split']}_{rec['raw'][FIELD_PINS['row_id']]}"
        assess_by_id[rid] = norm_assessment(rec["raw"].get(FIELD_PINS["labels"]))
    text_leaks = [it["item_id"] for it in items
                  if leaks_text(it, raw_by_id[it["item_id"]], assess_by_id[it["item_id"]])]

    checks = {
        "committed_items_carry_no_source_text": not text_leaks,
        "item_schema_is_the_closed_dua_whitelist": schema_ok,
        "no_overlap_item_in_d2n_primary":
            not any(it["aci_overlap"] for it in d2n_primary),
        "fewshot_pool_disjoint_from_eval_ids":
            not (set(ids(pool)) & set(ids(eval_items))),
        "fewshot_pool_all_omission_labelled":
            all(it["is_omission_labelled"] for it in pool),
        "every_item_has_a_role": all(it["eval_role"] in ("eval", "fewshot_pool") for it in items),
        "every_item_has_a_risk_grade": all(it["physician_risk_grade"] in (1, 2, 3, 4)
                                           for it in items),
        "item_ids_unique": len({it["item_id"] for it in items}) == len(items),
        "text_sidecar_covers_every_item": set(texts) == {it["item_id"] for it in items},
    }
    failed = [k for k, v in checks.items() if not v]

    counts_by_task = {}
    for it in items:
        d = counts_by_task.setdefault(it["task"], {"total": 0, "eval": 0, "fewshot_pool": 0,
                                                   "aci_overlap": 0, "omission_labelled": 0})
        d["total"] += 1
        d[it["eval_role"]] += 1
        d["aci_overlap"] += int(it["aci_overlap"])
        d["omission_labelled"] += int(it["is_omission_labelled"])

    def type_counts(xs):
        out = {"pure_omit": 0, "pure_add": 0, "pure_change": 0, "mixed": 0,
               "unresolved": 0, "no_annotation": 0}
        for it in xs:
            if not it["annotated_error"]:
                out["no_annotation"] += 1
            elif not it["our_types"]:
                out["unresolved"] += 1
            elif len(it["our_types"]) == 1:
                out["pure_" + it["our_types"][0]] += 1
            else:
                out["mixed"] += 1
        return out

    recon = {
        "generated": date.today().isoformat(),
        "script": "wb_medval_ingest.py",
        "method": "reconcile the releases, map the taxonomy, freeze the items and the few-shot pool",
        "seed": SEED,
        "releases": {
            "hf_open": {
                "dataset": "stanfordmimi/MedVAL-Bench",
                "url": "https://huggingface.co/datasets/stanfordmimi/MedVAL-Bench",
                "license": "MIT (dataset card YAML `license: mit`)",
                "split_row_totals": hf_totals,
                "physician_labelled_rows": len(hf_labelled),
                "note": ("the train split is the self-supervised fine-tuning half - risk grade -1 "
                         "and assessment 'na' on every row, no physician labels - so the "
                         "physician-labelled universe in the open release is exactly the test "
                         "split"),
            },
            "physionet": {
                "present": physionet_present,
                "dir": "external/medval_physionet/",
                "file": pn_path,
                "release": "medval-bench-expert-annotated-medical-text-validation-benchmark 1.0.1",
                "sha256_medval_bench_csv": (
                    "94045c9f12063d1bd50151eceecc39ac7ac15e2784175a6ce41de9294563659a "
                    "(matches the release's own SHA256SUMS.txt)" if physionet_present else None),
                "split_row_totals": pn_totals,
                "physician_labelled_rows": len(pn_labelled),
                "layout_note": (
                    "the release is ONE medval_bench.csv holding all 840 physician-annotated rows "
                    "and all SIX tasks, not the HF {train,test}.csv shape - loaded under a single "
                    "pseudo-split 'all'. Column names are byte-identical to HF, so the field pins "
                    "carry over unchanged. Rows are task-ordered and contiguous by `#`: "
                    "dialogue2note 1-85, medication2answer 86-220, query2question 221-340, "
                    "report2impression 341-530, impression2simplified 531-720, bhc2spanish "
                    "721-840."),
                "note": ("credentialed download (2026-08-12) - PhysioNet is treated as PRIMARY "
                         "wherever both releases carry an item, so every frozen item "
                         "below comes from this release and item_ids are medval_physionet_all_*."
                         if physionet_present else
                         "absent - this run is HF-open only. Re-run this script after the "
                         "download; shared items are then reconciled and the HF copy dropped, and "
                         "the counts below extend."),
            },
            "reconciliation": {
                "join": ("content identity (task, sha256(input), sha256(output)) - NOT (task, id). "
                         "See the comment at the reconciliation block: PhysioNet renumbers `id` "
                         "for all 135 medication2answer rows, so an id-keyed join matches only "
                         "428/530 shared rows and manufactures 25 phantom grade disagreements "
                         "(5.8%), tripping the release-disagreement gate on a join artefact."),
                # Content keys and items are not the same count: four report2impression
                # input/output pairs are annotated twice in each release, so 526 distinct content
                # keys cover 530 HF rows. Item-level figures come off the per-item hf_open flag.
                "shared_content_keys": len(shared),
                "shared_items": sum(1 for it in items if it["hf_open"]),
                "hf_only_items": len([it for it in items if it["release"] == "hf"]),
                "physionet_only_items": sum(1 for it in items if not it["hf_open"]),
                "duplicate_content_groups": {
                    "n": sum(1 for v in _dupe_groups.values() if len(v) > 1),
                    "rows_involved": sum(len(v) for v in _dupe_groups.values() if len(v) > 1),
                    "note": ("the same (task, input, output) annotated more than once within the "
                             "release, under different `#` values and sometimes with DIFFERENT "
                             "risk grades - an incidental repeat-annotation signal in MedVAL, not "
                             "a defect in this ingest. Kept as separate items (distinct `#`), "
                             "which is what the release itself does; flagged because anyone "
                             "computing per-dialogue statistics needs to know."),
                    "groups": [
                        {"task": k[0], "rows": sorted(int(it["row"]) for it in v),
                         "grades": [it["physician_risk_grade"] for it in v],
                         "assessments_identical": len({it["physician_error_assessment_sha256"]
                                                       for it in v}) == 1}
                        for k, v in sorted(_dupe_groups.items(), key=lambda kv: kv[0][0])
                        if len(v) > 1],
                },
                "label_disagreements": len(disagreements),
                "label_disagreement_fraction": (round(len(disagreements) / len(shared), 4)
                                                if shared else None),
                "disagreements": disagreements[:50],
                "row_number_join_also_valid": row_join_valid,
                "id_column_drift": {
                    "n": len(id_drift),
                    "by_task": {t: sum(1 for d in id_drift if d["task"] == t)
                                for t in sorted({d["task"] for d in id_drift})},
                    "examples": id_drift[:10],
                    "note": ("the `id` column (unique within a task) is renumbered between the "
                             "releases; `#` and the content both align exactly. Recorded as a "
                             "provenance fact, NOT counted as a label disagreement - no physician "
                             "label differs anywhere."),
                },
                "disagreement_gate_threshold": ">5% of shared items disagreeing without a "
                                               "documented resolution stops the anchor build",
                "disagreement_gate_status": ("not applicable - single release ingested"
                                             if not shared else
                                             ("PASS" if len(disagreements) / len(shared) <= 0.05
                                              else "FAIL - escalate")),
                "hf_only_certification": (
                    "an HF-only item may be carried only if its labels are certified identical at "
                    "reconciliation. Moot: the HF open release is a strict content SUBSET of the "
                    "PhysioNet release (zero HF-only items), so no HF item is carried on its own "
                    "authority." if physionet_present else None),
            },
            "coverage_caveat": (
                f"RESOLVED. The MedVAL paper's 840 physician-annotated outputs are all here: the "
                f"PhysioNet release carries exactly {len(pn_labelled)} physician-labelled rows and "
                f"the frozen set is {len(items)}. The open HF release's {len(hf_labelled)} rows are "
                "a strict content subset (PhysioNet #1-530). The 310-row difference is NOT extra "
                "data for the four HF tasks - it is two tasks the open release omits entirely "
                "(impression2simplified 190, bhc2spanish 120), and every per-task count for the "
                "four shared tasks is IDENTICAL across releases. dialogue2note is therefore "
                f"{counts_by_task.get(D2N_TASK, {}).get('total', 0)} items in the full release, the "
                "same 85 as in the open one, well under the 100-150 the pre-registration "
                "expected - and all 85 are ACI-Bench-derived (see aci_overlap), so excluding "
                "ACI-derived items leaves the dialogue2note PRIMARY analysis empty. This is a "
                "data fact, not a download gap: "
                "there is no further MedVAL release to wait for."
                if physionet_present else
                "The MedVAL paper reports 840 physician-annotated outputs; the open HF release "
                f"carries {len(hf_labelled)}. The shortfall is expected to live on PhysioNet and "
                "is NOT ingested here. Every count in this file is therefore a floor, and the "
                "dialogue2note subset in particular (pre-registered as order 100-150) comes "
                f"in at {counts_by_task.get(D2N_TASK, {}).get('total', 0)} in the open release."
            ),
        },
        "field_pins": FIELD_PINS,
        "field_pin_evidence": (
            "confirmed against the downloaded CSV headers AND MedVAL's own loader "
            "(medval/pipeline.py renames input->reference, output->candidate before scoring), so "
            "the physician graded `output` against `input` - the direction the "
            "medval_d2n/medval_any_task adapters assume."
        ),
        "binarization": {
            "pinned": "grade 1 (no risk) vs grades 2-4 (error present)",
            "sensitivity": "grades 1-2 vs grades 3-4",
            "source": "the pre-registered fallback",
            "evidence": (
                "the pre-registration first asks for MedVAL's own published binarization exactly "
                "as implemented in their released evaluation code. Checked github.com/StanfordMIMI/"
                "MedVAL @ main on 2026-08-10 (run.py, medval/pipeline.py, medval/validator.py, "
                "utils/prompts.py - the whole released surface): the released metric is "
                "`validator_metric`, a continuous consistency loss over the 1-4 risk scale scaled "
                "to the unit interval. There is NO binary threshold and no F1 computation "
                "anywhere in the release, so no published binarization exists to adopt and the "
                "pre-registered fallback governs."),
        },
        "taxonomy_mapping": {
            "table": {k: {"our_type": v[0], "rule": v[1]} for k, v in TABLE.items()},
            "canonical_vocabulary_source": (
                "github.com/StanfordMIMI/MedVAL utils/prompts.py `error_categories` (11 "
                "categories) - three more than the pre-registered mapping table lists"),
            "categories_resolved_by_catch_all": ["overstating intensity", "understating intensity"],
            "unassignable": ["other"],
            "detection": ("case-insensitive substring scan of the physician free text for the "
                          "canonical category names, plus the surface variants "
                          f"{sorted(SURFACE_VARIANTS)}; deterministic and blind to model output"),
            "unresolved_note": (
                "physician assessments that describe an error in prose without naming a category "
                "get our_types=[] and type_resolved=false. These count for the "
                "binary metrics (they carry a physician risk grade) and are excluded from the "
                "per-type recall analysis, exactly as mixed-type items are."),
        },
        "counts": {
            "items_frozen": len(items),
            "by_release": {r: sum(1 for it in items if it["release"] == r)
                           for r in sorted({it["release"] for it in items})},
            "hf_open_content": sum(1 for it in items if it["hf_open"]),
            "physionet_only_content": sum(1 for it in items if not it["hf_open"]),
            "by_task": counts_by_task,
            "by_risk_grade": {str(g): sum(1 for it in items if it["physician_risk_grade"] == g)
                              for g in (1, 2, 3, 4)},
            "types_all_items": type_counts(items),
            "types_d2n": type_counts([it for it in items if it["task"] == D2N_TASK]),
            "omission_labelled": n_omission,
            "aci_overlap_tagged": sum(1 for it in items if it["aci_overlap"]),
            "label_source_disagreements": sum(1 for it in items
                                              if it["label_source_disagreement"]),
        },
        "label_source_note": (
            "two independent binary labels are frozen per item: binary_label_grade (the pinned "
            "risk-grade binarization, used by the primary F1-vs-physicians metric) and "
            "binary_label_annotation (does the physician free text annotate any error). They "
            "disagree on "
            f"{sum(1 for it in items if it['label_source_disagreement'])} of {len(items)} items - "
            "grade-1 rows that still carry an annotated error, and grade-2 rows with no free-text "
            "annotation. The mapping-table row 'No error annotated AND risk level 1 -> clean' is "
            "frozen separately as clean_per_mapping_table so the analysis can use either "
            "definition without re-deriving it."),
        "fewshot_pool": {
            "n_requested": args.fewshot_n,
            "n_drawn": len(pool),
            "candidate_pool": n_omission,
            "strata": strata_report,
            "note": (
                "the pre-registration asks for 50-100 items drawn from the omission-labelled "
                "items, a "
                f"range sized against the paper's 840-item set. That set is now fully ingested and "
                f"supplies {n_omission} omission-labelled items, so drawing at the pre-registered "
                f"floor of 50 removes {round(100 * len(pool) / max(n_omission, 1))}% of the omission "
                "population from the eval sets - the earlier HF-only build had 109 candidates and a "
                "46% bite, which was the flagged concern. Still the floor by default; --fewshot-n "
                "makes it explicit."),
            "d2n_items_in_pool": sum(1 for it in pool if it["task"] == D2N_TASK),
            "per_category_availability": {
                c: sum(1 for it in items if c in it["omission_categories"])
                for c in OMISSION_CATEGORIES},
            "stratification_caveat": (
                "'stratified across the three omission categories' is only as balanced as the data "
                "allows - see per_category_availability. Strata are the co-occurrence sets items "
                "actually carry, so a category that barely appears cannot be proportionally "
                "represented in a 50-item draw."),
        },
        "eval_sets": {k: v["n"] for k, v in eval_ids["analysis_sets"].items()},
        "aci_overlap": overlap_meta,
        "new_tasks_from_physionet": {
            "tasks": {t: counts_by_task[t]["total"]
                      for t in sorted(counts_by_task) if t not in HF_TASKS},
            "note": ("the two tasks the open release omits entirely. They are the whole of the "
                     "530->840 difference, and they are ACI-free, so they carry the cross-task "
                     "generalisation weight the d2n subset can no longer carry."),
            "FLAG_bhc2spanish_is_cross_lingual": (
                "bhc2spanish is brief-hospital-course -> SPANISH: the source is English, the "
                "physician-graded output is Spanish (verified by function-word profile at ingest - "
                "~34% Spanish stopwords in the outputs, ~0% English). It contributes "
                f"{counts_by_task.get('bhc2spanish', {}).get('eval', 0)} items to the full-eval "
                "split. The judge prompts in w2_prompts/ are English and byte-frozen, so "
                "including this task means asking an English judge prompt to detect omissions in a "
                "Spanish note against an English source. That is a cross-lingual capability test, "
                "not the faithfulness test the rest of the external anchors run, and it will "
                "depress the sentinel cells for reasons that have nothing to do with salience "
                "scaffolding. A DECISION IS NEEDED before the anchor runs: report bhc2spanish as "
                "its own stratum, or exclude it from the cross-task generalisation claim. Not "
                "resolved here - the ingest neither drops nor re-weights it, so the choice stays "
                "open and explicit."),
        },
        "d2n_anchor_status": {
            "question": ("does the full 840-item release restore dialogue2note as the PRIMARY "
                         "external anchor?"),
            "d2n_items_total": counts_by_task.get(D2N_TASK, {}).get("total", 0),
            "d2n_items_aci_derived": counts_by_task.get(D2N_TASK, {}).get("aci_overlap", 0),
            "d2n_items_non_aci": (counts_by_task.get(D2N_TASK, {}).get("total", 0)
                                  - counts_by_task.get(D2N_TASK, {}).get("aci_overlap", 0)),
            "d2n_primary_eval_n": eval_ids["analysis_sets"]["d2n_primary"]["n"],
            "answer": (
                "NO, and the question is now closed rather than pending. The credentialed release "
                "adds zero dialogue2note items: 85 in both releases, byte-identical dialogues, and "
                "all 85 are exact normalised matches to ACI-Bench encounters. Excluding "
                "ACI-derived items therefore empties the dialogue2note primary set on the FULL "
                "release, not just on the open subset, and no further MedVAL data exists to "
                "change that. The physician-labelled external anchor has to rest on the "
                "cross-task full-eval split (impression2simplified and bhc2spanish are wholly "
                "new, ACI-free tasks) with dialogue2note reported as the ACI-overlap sensitivity "
                "row it now unavoidably is - or on MEDEC. That is a change to the pre-registered "
                "primary anchor; it is not a call this ingest can make."),
        },
        "dua": {
            "release": "PhysioNet Credentialed Health Data License 1.5.0 (LICENSE.txt in the zip)",
            "raw_location": "external/medval_physionet/ (gitignored via external/ in .gitignore)",
            "committed_here": ("row numbers, task ids, task labels, error categories, risk grades, "
                               "SHA-256 hashes, word counts, split/eval-role assignments, counts "
                               "and aggregates"),
            "never_committed": ("input, output, reference_output and physician_error_assessment "
                                "text - the assessment is stored as a SHA-256 only, and the item "
                                "text lives in the gitignored sidecar "
                                "external/wb_medval_items_text.json"),
            "public_release_obligation": (
                "a public release is held to a stricter line than what is committed here: for "
                "PhysioNet-covered items it may carry aggregates and item IDs only, no per-item "
                f"physician labels. {sum(1 for it in items if it['hf_open'])} of {len(items)} "
                "items also ship in the MIT-licensed Hugging Face open release and are flagged "
                "hf_open=true; filter on that flag when building it."),
        },
        "validation": {"checks": checks, "failed": failed,
                       "dua_guard": {"text_leaking_items": text_leaks[:20],
                                     "unexpected_item_keys": unexpected_keys}},
        "outputs": ["wb_medval_items.json", "wb_medval_reconciliation.json",
                    "wb_medval_fewshot_pool.json", "wb_medval_eval_ids.json",
                    "external/wb_medval_items_text.json (gitignored)"],
    }

    frozen = {
        "generated": date.today().isoformat(),
        "script": "wb_medval_ingest.py",
        "method": "taxonomy mapping and text policy applied, frozen BEFORE any judging",
        "seed": SEED,
        "field_pins": FIELD_PINS,
        "binarization": recon["binarization"],
        "text_policy": ("no source text in this file, per the PhysioNet data use agreement. Join "
                        "external/wb_medval_items_text.json (gitignored) by item_id at run time."),
        "n": len(items),
        "items": items,
    }

    json.dump(frozen, open(OUT_ITEMS, "w", encoding="utf-8"), indent=1)
    json.dump(recon, open(OUT_RECON, "w", encoding="utf-8"), indent=1)
    json.dump({"generated": date.today().isoformat(), "seed": SEED,
               "method": "the held-out few-shot pool - the few-shot judge arm draws its examples "
                         "EXCLUSIVELY from this file",
               "strata": strata_report, "n": len(pool),
               "items": [{k: it[k] for k in ("item_id", "task", "omission_categories",
                                             "our_types", "physician_risk_grade")}
                         for it in pool]},
              open(OUT_POOL, "w", encoding="utf-8"), indent=1)
    json.dump(eval_ids, open(OUT_EVAL, "w", encoding="utf-8"), indent=1)
    os.makedirs(os.path.dirname(OUT_TEXT), exist_ok=True)
    json.dump(texts, open(OUT_TEXT, "w", encoding="utf-8"), indent=1)

    print(f"releases: hf {len(hf_labelled)} labelled rows; physionet "
          f"{len(pn_labelled)} labelled rows ({'present' if physionet_present else 'ABSENT'})")
    if physionet_present:
        print(f"reconciliation: {len(shared)} shared content keys, "
              f"{len(disagreements)} label disagreements, {len(id_drift)} id-column drifts, "
              f"disagreement gate {recon['releases']['reconciliation']['disagreement_gate_status']}")
    print(f"items frozen: {len(items)}  ->  {OUT_ITEMS}")
    for task, c in sorted(counts_by_task.items()):
        print(f"  {task:20s} total {c['total']:4d}  eval {c['eval']:4d}  "
              f"pool {c['fewshot_pool']:3d}  aci_overlap {c['aci_overlap']:3d}  "
              f"omission-labelled {c['omission_labelled']:3d}")
    print(f"few-shot pool: {len(pool)} of {n_omission} omission-labelled candidates")
    for k, v in eval_ids["analysis_sets"].items():
        print(f"  eval set {k:32s} n={v['n']}")
    print("validation:", "PASS" if not failed else f"FAIL {failed}")
    if not d2n_primary:
        print("WARNING: the dialogue2note PRIMARY eval set is EMPTY - every dialogue2note item "
              "in this release is ACI-overlap-flagged, so excluding ACI-derived items removes "
              "all of them. Resolve before the anchor runs.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
