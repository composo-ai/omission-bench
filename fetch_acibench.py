"""W-D step 1 (spec Section 5) - fetch the official ACI-Bench release and normalise every
encounter to {id, split, transcript, ref_note}.

Real repo structure (checked directly, 2026-07-29, github.com/wyim/aci-bench @ main): the data
ships as 5 CSVs under data/challenge_data/, one per official split from the MEDIQA-CHAT/MEDIQA-SUM
2023 challenges (not the train/valid/test/test2/test3 txt-file layout the spec guessed at):

    train.csv (67), valid.csv (20), clinicalnlp_taskB_test1.csv (40),
    clinicalnlp_taskC_test2.csv (40), clef_taskC_test3.csv (40)  ->  207 unique encounters

Each row has encounter_id (e.g. "D2N001"), dialogue (the transcript) and note (the clinician
reference note), plus a `dataset` tag (aci / virtassist / virtscribe - the encounter's original
recording/generation method). encounter_id ranges are contiguous and non-overlapping across the
5 files (D2N001-D2N207), so "unique source encounters" = the union of all 5 files with no dedup
needed. Full ground-truth notes are present in every split (this is the post-challenge release,
not the blind test set) - CC BY 4.0, Yim et al. 2023, Nature Scientific Data.

Pulled via raw.githubusercontent.com rather than a full git clone: the repo also carries
baselines/predictions + results/ (unneeded model outputs, adds nothing here).

Usage: python fetch_acibench.py [--ref main]
Output: master/aci_raw/aci_all.json - list of {id, split, transcript, ref_note, source_dataset}
"""
import csv
import io
import json
import os
import sys

import requests

from common import HERE

RAW_URL = "https://raw.githubusercontent.com/wyim/aci-bench/{ref}/data/challenge_data/{fname}.csv"
# (repo filename, our normalised split label) - label matches the spec's expected 5-way split shape.
SPLIT_FILES = [
    ("train", "train"),
    ("valid", "valid"),
    ("clinicalnlp_taskB_test1", "test1"),
    ("clinicalnlp_taskC_test2", "test2"),
    ("clef_taskC_test3", "test3"),
]
OUT_DIR = os.path.join(HERE, "master", "aci_raw")
OUT_PATH = os.path.join(OUT_DIR, "aci_all.json")


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


REF = _arg("--ref", "main")


def fetch_split(fname, split_label):
    url = RAW_URL.format(ref=REF, fname=fname)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(r.text)))
    out = []
    for row in rows:
        eid = (row.get("encounter_id") or "").strip()
        dialogue = (row.get("dialogue") or "").strip()
        note = (row.get("note") or "").strip()
        if not eid or not dialogue or not note:
            raise ValueError(
                f"{fname}.csv row {eid!r} missing id/dialogue/note - refusing to write a "
                "partial encounter (real ACI-Bench rows always carry all three)")
        out.append({
            "id": eid,
            "split": split_label,
            "transcript": dialogue,
            "ref_note": note,
            "source_dataset": (row.get("dataset") or "").strip(),
        })
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_records, seen = [], set()
    for fname, split_label in SPLIT_FILES:
        recs = fetch_split(fname, split_label)
        for r in recs:
            if r["id"] in seen:
                raise ValueError(f"duplicate encounter id across ACI-Bench splits: {r['id']}")
            seen.add(r["id"])
        all_records.extend(recs)
        print(f"{split_label:6s} ({fname}.csv): {len(recs)} encounters")

    all_records.sort(key=lambda r: r["id"])
    json.dump(all_records, open(OUT_PATH, "w"), indent=1)
    print(f"\ntotal: {len(all_records)} unique encounters -> {OUT_PATH}")


if __name__ == "__main__":
    main()
