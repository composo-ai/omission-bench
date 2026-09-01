"""build_census_realerror.py - turn the census subset into something the P1 judges can read.

The real-error arm points three existing judge designs at the census's REAL vendor notes.
Everything P1 measures is on constructed absence, and the cold reader's closing objection is
that every omission in the benchmark "was put where a model knew it was putting it". The
census is the only material that answers it: 565 real vendor notes with 618 panel-verified
failures mapped on.

This script does the adapting and NOTHING else - it makes no model call and buys nothing. Two
files come out:

  master/census_realerror_pairs.json   a dataset in the shape w2_common.load_dataset expects
  master/census_realerror_spec.json    the --subset-spec: which notes to judge, plus the
                                       transcripts the benchmark's index does not carry

Why a "pairs" file at all, when the census has no pairs. Every P1 runner reaches its notes
through `load_dataset -> transcript_index -> note_units`, and note_units is built around
(clean, errored) twins. Rather than fork three runners - which would put the real-error arm on
a different instrument from the benchmark it is meant to be compared against, defeating the
point - each census note enters as the `errored` side of a degenerate pair. The `clean` slot is
a sentinel that is never judged: note_units mints one note from it per consultation, and the
subset allowlist removes every one before a single call is bought. That is what makes the arm
cost 261 note-judgements rather than roughly 400.

The five missing transcripts. The benchmark's transcript index covers 137 of the census's 142
consultations and note_units RAISES on a gap rather than skipping it, so three of the 87
verified-omission notes would take the run down. `taxonomy_common`'s own index covers all 142,
and on the 137 they share the two return byte-identical text (asserted below, not assumed), so
the spec carries the five as an override rather than the arm quietly losing them.

    python3 census/build_census_realerror.py
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
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repository root
sys.path.insert(0, HERE)

import taxonomy_common as tc                      # noqa: E402  the census's canonical loader
import w2_common as W                             # noqa: E402  the benchmark's index

SUBSET = os.path.join(HERE, "master", "census_realerror_261.json")
PAIRS_OUT = os.path.join(HERE, "master", "census_realerror_pairs.json")
SPEC_OUT = os.path.join(HERE, "master", "census_realerror_spec.json")
DATASET_VERSION = "census-realerror-261-v1"

# Never judged. It exists because note_units requires a `clean` side, and every note minted
# from it is removed by the allowlist below before any call is bought. Kept identical across
# a consultation so exactly one is minted per consultation rather than one per note.
SENTINEL = ("[not judged] The real-error arm scores single census notes against the census's "
            "own verified findings, not against a clean twin. This slot exists only because "
            "note_units requires a clean side; the subset allowlist removes every note minted "
            "from it before any call is bought.")


def main():
    if not os.path.exists(SUBSET):
        raise SystemExit(f"{SUBSET} not found - build the 261-note subset first")
    sub = json.load(open(SUBSET))
    notes = sub["notes"]
    print(f"subset: {len(notes)} notes, seed {sub.get('seed')}")

    by_arm = {}
    for n in notes:
        by_arm[n["arm"]] = by_arm.get(n["arm"], 0) + 1
    print(f"  by arm: {by_arm}")
    if len(notes) != 261:
        raise SystemExit(f"expected 261 notes, subset holds {len(notes)} - refusing to build")

    # ---- transcripts: the benchmark index, plus the five the census has and it does not ----
    bench, _prov = W.transcript_index()
    census_tx = {}
    for n in notes:
        census_tx[(n["source"], n["id"])] = n["transcript"]

    shared = [k for k in census_tx if k in bench]
    differing = [k for k in shared if census_tx[k] != bench[k]]
    if differing:
        raise SystemExit(
            f"{len(differing)} consultation(s) where the census transcript differs from the "
            f"benchmark's, e.g. {differing[:2]} - the arm would be judging a different source "
            "text from the benchmark and is not comparable. Stopping.")
    print(f"  transcripts: {len(shared)} shared with the benchmark index, all byte-identical")

    override = {f"{s}|{i}": census_tx[(s, i)] for (s, i) in census_tx if (s, i) not in bench}
    print(f"  transcripts supplied by the spec (absent from the benchmark index): "
          f"{len(override)}")
    for k in sorted(override):
        print(f"      {k}")

    # ---- the dataset ----------------------------------------------------------------------
    pairs, seen = [], set()
    for n in notes:
        pid = n["note_key"]                       # census note_key, unique by construction
        if pid in seen:
            raise SystemExit(f"duplicate note_key {pid} - pair_ids must be unique")
        seen.add(pid)
        pairs.append({
            "pair_id": pid,
            "id": n["id"],
            "stratum": n["source"],
            "type": "omit",     # a label the arm never scores on; ground truth is the census's
            "clean": SENTINEL,
            "errored": n["note_text"],
            # carried for provenance; the runners ignore unknown keys
            "census_arm": n["arm"],
            "census_scribe_letter": n["product_letter"],
            "census_template": n["template"],
        })

    blob = {"dataset_version": DATASET_VERSION,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "source_subset": os.path.relpath(SUBSET, HERE),
            "seed": sub.get("seed"),
            "note": ("Each census note is the errored side of a degenerate pair; the clean "
                     "side is a sentinel that the subset allowlist removes before any call. "
                     "'type' is a required field, not a claim about the note."),
            "pairs": pairs}
    with open(PAIRS_OUT, "w") as fh:
        json.dump(blob, fh, indent=1)

    # ---- the spec: allowlist + transcript overrides ----------------------------------------
    # note_units keys an errored note as f"{pair_id}|err"; the allowlist must use that form.
    note_keys = sorted(f"{p['pair_id']}|err" for p in pairs)
    with open(SPEC_OUT, "w") as fh:
        json.dump({"generated_utc": blob["generated_utc"],
                   "dataset_version": DATASET_VERSION,
                   "note": ("note_keys: exactly the 261 census notes to judge, so the clean "
                            "sentinel notes are never bought. transcripts: the consultations "
                            "the benchmark's index lacks, verified byte-identical to it on "
                            "the ones both carry."),
                   "note_keys": note_keys,
                   "transcripts": override}, fh, indent=1)

    # ---- prove it round-trips through the real loaders --------------------------------------
    loaded, info = W.load_dataset(path=os.path.relpath(PAIRS_OUT, HERE))
    tx, _ = W.transcript_index()
    spec = W.load_subset_spec(os.path.relpath(SPEC_OUT, HERE))
    tx.update(spec["transcripts"])
    blocks, judged = W.note_units(loaded, tx)
    minted = len(judged)
    blocks, judged = W.apply_subset_spec(spec, blocks, judged)

    print(f"\nwrote {os.path.relpath(PAIRS_OUT, HERE)} ({len(pairs)} pairs)")
    print(f"wrote {os.path.relpath(SPEC_OUT, HERE)} ({len(note_keys)} note_keys, "
          f"{len(override)} transcript overrides)")
    print(f"\nround-trip through load_dataset + note_units + apply_subset_spec:")
    print(f"  dataset_version {info['dataset_version']} sha {info['sha256'][:12]} "
          f"kind={info['dataset_kind']} frozen={info['frozen']}")
    print(f"  note_units minted {minted} notes; allowlist keeps {len(judged)} "
          f"over {len(blocks)} consultations")
    print(f"  sentinel notes never judged: {minted - len(judged)}")
    roles = {}
    for n in judged:
        roles[n["note_role"]] = roles.get(n["note_role"], 0) + 1
    print(f"  by note_role: {roles}")
    if len(judged) != 261:
        raise SystemExit(f"allowlist kept {len(judged)}, expected 261 - stopping")
    print("\nOK - 261 notes, and nothing else, will be judged.")


if __name__ == "__main__":
    main()
