"""Feeds a third-party corpus to this study's judges. Judge prompts are loaded from the w2_prompts/ txt files (hash-checked
against PROMPTS.sha256) and are NEVER modified here.
An adapter maps a dataset item to (transcript_slot, note_slot) strings. Field names marked
PIN-AT-INGEST are confirmed against the downloaded data by wb_medval_ingest.py / wb_medec_ingest.py
and recorded in the ingest manifest before any judging runs.

The four adapter bodies are the pre-registered text, unchanged; the loaders below them are
plumbing added at ingest time so the adapters receive dicts carrying exactly the field names
the pre-registration fixed.

PIN CONFIRMATION (2026-08-10, recorded in wb_medval_reconciliation.json / wb_medec_ingest_report.json):
  MedVAL  input / output / task            - confirmed against stanfordmimi/MedVAL-Bench columns
                                             AND MedVAL's own loader, which renames
                                             input->reference and output->candidate before the
                                             physician comparison, i.e. `output` was graded
                                             against `input`. RE-CONFIRMED 2026-08-12 against the
                                             credentialed PhysioNet release (medval_bench.csv):
                                             byte-identical column names, so the pins hold for
                                             both releases and no release-specific branch exists.

MedVAL release note (2026-08-12). The frozen items are now the full 840-item PhysioNet set, which
covers SIX tasks - the four in the HF open release plus impression2simplified and bhc2spanish.
medval_any_task already handles arbitrary tasks by labelling the source slot, so both new tasks
route through it unchanged. medval_d2n is unaffected: dialogue2note is the same 85 items it was.
  MEDEC   text / error_flag / sentences /  - confirmed against
          corrected_sentence /               MEDEC-MS-TestSet-with-GroundTruth-and-ErrorType.csv
          error_sentence_id                  (columns Text / Error Flag / Sentences /
                                             Corrected Sentence / Error Sentence ID)
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path

import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repository root

TRUNC = 40000  # same source-side truncation convention as the shared client (common.py)


def medval_d2n(item):
    """MedVAL-Bench dialogue2note item -> judge input. The source dialogue is the transcript,
    the physician-graded LM note is the note. No added instructions."""
    return item["input"].strip()[:TRUNC], item["output"].strip()   # field names PIN-AT-INGEST


def medval_any_task(item):
    """MedVAL-Bench item from any task (full-840 secondary run). The source is not always a
    dialogue, so the slot is prefixed with a one-line content label INSIDE the input slot -
    the prompt template around it is untouched."""
    src = f"[Source document for this item (task: {item['task']}); the note below must be faithful to it.]\n" \
          + item["input"].strip()
    return src[:TRUNC], item["output"].strip()


MEDEC_NO_SOURCE = (
    "[No source transcript exists for this item. The clinical text below is the entire record. "
    "Judge it against internal consistency and established clinical knowledge: treat any statement "
    "that is clinically impossible, contradicted elsewhere in the text, or a mismatch of diagnosis, "
    "management, or pharmacotherapy with the case described as a difference that would change the "
    "clinician's understanding or the patient's care.]"
)


def medec_flag(item):
    """MEDEC flag mode: the placeholder occupies the transcript slot; the clinical text is the note."""
    return MEDEC_NO_SOURCE, item["text"].strip()   # field name PIN-AT-INGEST


def medec_clean_twin(item):
    """MEDEC paired mode: rebuild the corrected text by substituting the annotator's corrected
    sentence for the flagged sentence. Returns None for error-free items."""
    if not item["error_flag"]:
        return None
    return "\n".join(
        item["corrected_sentence"] if sid == item["error_sentence_id"] else s
        for sid, s in item["sentences"]            # (sentence_id, text) pairs PIN-AT-INGEST
    )


# ------------------------------------------------------------------ loaders (plumbing)
def load_medval_items(with_text=True):
    """Frozen MedVAL items, merged with the gitignored text sidecar so each dict carries the
    `input` / `output` / `task` keys the adapters above expect.

    wb_medval_items.json is committed WITHOUT source text (PhysioNet's data use agreement
    forbids redistributing item text, and one schema serves both releases); the text lives
    beside the raw cache in external/wb_medval_items_text.json.
    """
    items = json.load(open(os.path.join(HERE, "wb_medval_items.json")))["items"]
    if not with_text:
        return items
    path = os.path.join(HERE, "external", "wb_medval_items_text.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing - re-run wb_medval_ingest.py to rebuild the text sidecar "
            "(it is gitignored, so a fresh clone will not have it)")
    text = json.load(open(path))
    merged = []
    for it in items:
        t = text[it["item_id"]]
        merged.append({**it, "input": t["input"], "output": t["output"],
                       "reference_output": t["reference_output"]})
    return merged


def load_medval_eval_sets():
    """{set_name: [item_id, ...]} for the analysis sets frozen by the ingest, plus the ids of
    the held-out pool the few-shot judge arm draws its examples from, which the leakage
    check tests against."""
    d = json.load(open(os.path.join(HERE, "wb_medval_eval_ids.json")))
    sets = {k: v["ids"] for k, v in d["analysis_sets"].items()}
    sets["eval_ids"] = d["eval_ids"]
    sets["fewshot_pool_ids"] = d["fewshot_pool_ids"]
    return sets


def load_medec_items(sample_only=True):
    """Frozen MEDEC items. `sentences` is stored as a list of [id, text] lists in JSON; the
    adapter unpacks two-element sequences, so no conversion is needed.

    sample_only=True returns the pre-registered stratified n=300 sample; False returns
    all 597 MS-test items.
    """
    items = json.load(open(os.path.join(HERE, "wb_medec_items.json")))["items"]
    return [it for it in items if it["in_sample"]] if sample_only else items
