"""Draw the seeded ACI-Bench subsample, excluding encounters a prior study had used.

The exclusion rule was corrected during the corpus build: the original version excluded only the
21-doc `test` split of splits.json, and mislabelled it "20-doc" throughout. Corrected rule:

  1. Pool ALL unique source encounters from master/aci_raw/aci_all.json (all 5 official splits).
  2. EXCLUDE any encounter whose id appears in EITHER the 46-doc `train` split OR the 21-doc `test`
     split of the prior judge's `splits.json` (an earlier internal project; not shipped here)
     (67 ids total - that file was used to optimise a prior GEPA judge run on this repo's
     ACI-Bench access, so both its splits are excluded defensively, not just its test half).
  3. Stratify the remainder by (official ACI-Bench split) x (transcript word-length tercile,
     computed over the remainder).
  4. Sample proportionally to stratum size to N=48, seeded RNG seed 20260728, ids sorted before
     sampling so the draw is reproducible byte-for-byte.

splits.json cross-check (done here, not assumed): its keys look like "D2N001__9" - a "__9"
fold-tag suffix on top of what turns out to be ACI-Bench's own encounter_id. Stripping the suffix
and checking against aci_all.json's real ids confirms they match exactly, AND that all 67 of
splits.json's ids fall inside ACI-Bench's own 67-row official `train` split (same D2N001-D2N067
range) - so the corrected rule excludes ACI-Bench's entire official train split from the pool.

Usage: python corpus/subsample_acibench.py [--n 48] [--seed 20260728]
Output: master/aci_subsample.json (N sampled encounters, same schema as aci_all.json)
        master/subsample_manifest.json (pool/exclusion/stratum/sample accounting)
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
from collections import defaultdict
from random import Random

from common import HERE

RAW = os.path.join(HERE, "master", "aci_raw", "aci_all.json")
# The prior judge's split file is not part of this release; supply your own copy to re-run.
SPLITS_JSON = os.environ.get("PRIOR_JUDGE_SPLITS", os.path.join(HERE, "master", "splits.json"))
OUT_SAMPLE = os.path.join(HERE, "master", "aci_subsample.json")
OUT_MANIFEST = os.path.join(HERE, "master", "subsample_manifest.json")


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


N = int(_arg("--n", "48"))
SEED = int(_arg("--seed", "20260728"))


def _strip_fold_tag(k):
    """'D2N001__9' -> 'D2N001'. splits.json ids carry an internal fold-tag suffix that ACI-Bench's
    own encounter_id does not."""
    return k.split("__")[0]


def tercile_labels(pool):
    """Rank-based terciles (near-equal group sizes despite tied word counts), computed over `pool`
    only - the post-exclusion remainder, not the full corpus. Returns {id: 'T1'|'T2'|'T3'}."""
    ranked = sorted(pool, key=lambda r: (len(r["transcript"].split()), r["id"]))
    n = len(ranked)
    return {r["id"]: f"T{(i * 3 // n) + 1}" for i, r in enumerate(ranked)}


def apportion(sizes, total):
    """Largest-remainder (Hare quota) proportional allocation of `total` across strata keyed by
    `sizes` (stratum -> pool count). Deterministic tie-break: quota remainder desc, then stratum
    key asc. Caps each stratum at its own pool size and hands any overflow to the strata with the
    most spare room, same deterministic ordering."""
    pool_n = sum(sizes.values())
    quotas = {k: total * v / pool_n for k, v in sizes.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    remainder_slots = total - sum(alloc.values())
    order = sorted(sizes, key=lambda k: (-(quotas[k] - alloc[k]), k))
    for k in order[:remainder_slots]:
        alloc[k] += 1
    overflow = 0
    for k in list(alloc):
        if alloc[k] > sizes[k]:
            overflow += alloc[k] - sizes[k]
            alloc[k] = sizes[k]
    if overflow:
        room_order = sorted(sizes, key=lambda k: (-(sizes[k] - alloc[k]), k))
        for k in room_order:
            room = sizes[k] - alloc[k]
            if room <= 0:
                continue
            take = min(room, overflow)
            alloc[k] += take
            overflow -= take
            if overflow == 0:
                break
    assert overflow == 0, "apportionment could not place all slots - N exceeds pool size"
    return alloc


def main():
    pool = json.load(open(RAW))
    pool.sort(key=lambda r: r["id"])
    pool_ids = {r["id"] for r in pool}
    if len(pool_ids) != len(pool):
        raise ValueError("duplicate ids in aci_all.json - fetch_acibench.py output is broken")

    splits = json.load(open(SPLITS_JSON))
    train_ids = {_strip_fold_tag(k) for k, v in splits.items() if v == "train"}
    test_ids = {_strip_fold_tag(k) for k, v in splits.items() if v == "test"}
    if train_ids & test_ids:
        raise ValueError("splits.json train/test overlap after stripping fold tags - investigate before sampling")
    print(f"splits.json: {len(train_ids)} train ids, {len(test_ids)} test ids "
          f"({len(train_ids) + len(test_ids)} total, matches the expected 67 ids total: "
          f"{len(train_ids) + len(test_ids) == 67})")

    train_excl = sorted(train_ids & pool_ids)
    test_excl = sorted(test_ids & pool_ids)
    unmatched = sorted((train_ids | test_ids) - pool_ids)
    if unmatched:
        print(f"WARNING: {len(unmatched)} splits.json ids do not appear in aci_all.json at all: {unmatched}")
    excluded = sorted(set(train_excl) | set(test_excl))

    remainder = [r for r in pool if r["id"] not in set(excluded)]
    remainder.sort(key=lambda r: r["id"])

    labels = tercile_labels(remainder)
    for r in remainder:
        r["_tercile"] = labels[r["id"]]
        r["_word_count"] = len(r["transcript"].split())

    strata_ids = defaultdict(list)
    for r in remainder:
        strata_ids[(r["split"], r["_tercile"])].append(r["id"])
    for k in strata_ids:
        strata_ids[k].sort()

    sizes = {k: len(v) for k, v in strata_ids.items()}
    alloc = apportion(sizes, N)

    rng = Random(SEED)
    sampled_ids = []
    for k in sorted(alloc):
        quota = alloc[k]
        if quota <= 0:
            continue
        sampled_ids.extend(rng.sample(strata_ids[k], quota))

    if len(sampled_ids) != N:
        raise ValueError(f"apportionment produced {len(sampled_ids)} ids, expected {N}")
    if len(set(sampled_ids)) != len(sampled_ids):
        raise ValueError("sampled ids are not unique - stratification bug")

    by_id = {r["id"]: r for r in remainder}
    sample = [
        {"id": r["id"], "split": r["split"], "transcript": r["transcript"],
         "ref_note": r["ref_note"], "source_dataset": r.get("source_dataset", "")}
        for r in (by_id[eid] for eid in sorted(sampled_ids))
    ]

    os.makedirs(os.path.dirname(OUT_SAMPLE), exist_ok=True)
    json.dump(sample, open(OUT_SAMPLE, "w"), indent=1)

    manifest = {
        "seed": SEED,
        "n_target": N,
        "n_sampled": len(sample),
        "pool_size_total": len(pool),
        "exclusion": {
            "rule": "exclude ids in EITHER splits.json's train OR test split (corrects an "
                    "earlier version of the rule, which excluded only the 21-doc test "
                    "split)",
            "source_file": SPLITS_JSON,
            "splits_json_train_ids": len(train_ids),
            "splits_json_test_ids": len(test_ids),
            "train_split_exclusions_matched_in_pool": len(train_excl),
            "test_split_exclusions_matched_in_pool": len(test_excl),
            "total_excluded": len(excluded),
            "splits_json_ids_not_found_in_pool": unmatched,
            "excluded_ids": excluded,
        },
        "pool_size_after_exclusion": len(remainder),
        "stratum_definition": "(official ACI-Bench split) x (transcript word-length tercile, "
                               "rank-based, computed over the post-exclusion remainder)",
        "stratum_pool_counts": {f"{k[0]}/{k[1]}": v for k, v in sorted(sizes.items())},
        "stratum_sample_counts": {f"{k[0]}/{k[1]}": alloc.get(k, 0) for k in sorted(sizes)},
        "sample_by_stratum": {
            f"{k[0]}/{k[1]}": sorted(i for i in sampled_ids if i in strata_ids[k])
            for k in sorted(sizes) if alloc.get(k, 0) > 0
        },
        "sample_ids": sorted(sampled_ids),
    }
    json.dump(manifest, open(OUT_MANIFEST, "w"), indent=1)

    print(f"\npool: {len(pool)} total encounters")
    print(f"excluded: {len(excluded)} (train-split {len(train_excl)}, test-split {len(test_excl)})")
    print(f"remainder pool: {len(remainder)}")
    print(f"sampled: {len(sample)} across {sum(1 for v in alloc.values() if v > 0)} strata")
    print(f"saved -> {OUT_SAMPLE}")
    print(f"saved -> {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
