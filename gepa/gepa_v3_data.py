"""gepa_v3_data.py - the v3 prompt-optimisation run's data PARTITION, and the example objects both legs score.

WHY v3 HAS A PARTITION AT ALL. v1 and v2 shared one flaw: a single small pool served as
BOTH the reflection source (what the mutator was shown) and the acceptance validator (what
decided whether a mutation survived). That is the overspecialisation the DSPy GEPA docs warn
about in as many words, and it cashed out - v1's winner scored 0.685-equivalent on its dev
pool and 0.549 paired on the evaluation set. v3 fixes the inner split at
real scale:

    TEST     master/arms_confirm_subset.json - 151 pairs / 47 consultations, and NOTHING
             touches it during the search. Baselines on it already exist and are free:
             pipeline B2 from results/w2-pipeline/_state/confirm-B2.jsonl, the grid's
             FC-score cells from results/w2-ablation/_state/grid-main2.jsonl. A winner is
             run on it ONCE.
    TRAIN    every eval_set pair of master/dataset_v2.json whose CONSULTATION is not one of
             the test's 47. Split again at consultation level:
    REFLECT  ~70% of TRAIN's consultations. Minibatches are drawn here and the reflector
             only ever reads traces from here.
    VALID    ~30% of TRAIN's consultations. Acceptance, per-instance Pareto rewards and the
             winner choice are computed here. The reflector never sees a VALID note.

Consultation-level everywhere, because two pairs from one consultation share a transcript,
a clean twin and a fact list - splitting at pair level would leak all three.

The manifest (gepa/v3_partition.json) is written once, committed before any optimisation,
and asserted on every load: disjointness of the three splits, disjointness from the TEST
consultations, and disjointness from gepa/eval_exclusions.json (everything the v1 and v2
optimizers ever trained on, at both pair and consultation level).

    python3 gepa/gepa_v3_data.py --build     # write the manifest
    python3 gepa/gepa_v3_data.py             # print it + verify
"""
import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path

import w2_common as W  # noqa: E402

MASTER = os.path.join(ROOT, "master")
TRAIN_FILE = "master/dataset_v2.json"
TEST_FILE = "master/arms_confirm_subset.json"
EXCLUSIONS = os.path.join(HERE, "eval_exclusions.json")
PARTITION = os.path.join(HERE, "v3_partition.json")

SPLIT_SEED = 20260814
VALID_FRACTION = 0.30

OMIT_KINDS = ("omit-complete", "omit-partial")
ERR_KINDS = ("omit-complete", "omit-partial", "add", "change")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---------------------------------------------------------------- the raw material
def _pairs(rel):
    pairs, info = W.load_dataset(path=rel)
    return pairs, info


def _consultations(pairs):
    return {(p["stratum"], p["id"]) for p in pairs}


def _exclusions():
    """Every pair id and consultation the earlier prompt-optimisation runs trained on."""
    d = json.load(open(EXCLUSIONS))
    pair_ids = set(d.get("trained_pair_ids") or []) | set(d.get("trained_partial_pair_ids") or [])
    pair_ids |= set(d.get("dev_pool_pair_ids_v2") or [])
    cons = set()
    for stratum, ids in (d.get("consultation_ids") or {}).items():
        for cid in ids:
            cons.add((stratum, cid))
    return pair_ids, cons


# ---------------------------------------------------------------- example objects
def _kind_of(pair):
    fac = pair.get("_fac") or W.factorial_of(pair)
    return fac["class"], fac


def build_examples(pairs, keep_consultations=None):
    """Note units -> the example objects the legs score.

    One example per NOTE (a consultation's clean twin is one example shared by all its
    errored notes, exactly as w2_pipeline judges it), carrying everything a rich reflection
    trace needs: the gold removed fact, the surviving mention for partials, the severity and
    residual grades, plus the transcript and the item_index the seed formula reads.

    item_index comes from W.note_units over the pairs passed in, so a leg that loads the
    same file with the same filter gets the same seeds as any earlier run of that file -
    which is what makes the TEST comparison against confirm-B2.jsonl a matched one.
    """
    transcripts, tprov = W.transcript_index()
    blocks, notes = W.note_units(pairs, transcripts)
    by_pair = {p["pair_id"]: p for p in pairs}
    out, kept_blocks = [], []
    for b in blocks:
        if keep_consultations is not None and (b["stratum"], b["consultation"]) not in keep_consultations:
            continue
        kept_blocks.append(b)
        for n in b["notes"]:
            ex = {"key": n["note_key"], "clean_key": n["clean_key"], "note": n["text"],
                  "stratum": n["stratum"], "consultation": n["consultation"],
                  "transcript": b["transcript"], "item_index": n["item_index"],
                  "pair_id": n["pair_id"], "pair_type": n["pair_type"],
                  "note_role": n["note_role"]}
            if n["note_role"] == "clean":
                ex.update({"kind": "clean", "severity": None, "residual_level": None,
                           "residual_strength": None, "fact": None, "residual": None,
                           "severity_rationale": None})
            else:
                p = by_pair[n["pair_id"]]
                cls, fac = _kind_of(p)
                res = p.get("residual") if isinstance(p.get("residual"), dict) else {}
                sites = (res.get("verified_sites") or res.get("sites") or [])[:3]
                ex.update({
                    "kind": cls, "severity": fac["severity"],
                    "residual_level": fac["residual_level"],
                    "residual_strength": fac["residual_strength"],
                    "fact": (p.get("fact") or p.get("what") or p.get("change") or "")[:400],
                    "severity_rationale": (p.get("severity_rationale") or "")[:400] or None,
                    "residual": [{"quote": (s.get("quote") or "")[:200],
                                  "section": s.get("section"),
                                  "strength": s.get("strength")} for s in sites if isinstance(s, dict)]
                                or None})
            out.append(ex)
    out.sort(key=lambda e: e["key"])
    return out, kept_blocks, tprov


# ---------------------------------------------------------------- the split
def _split_train_consultations(pairs, seed=SPLIT_SEED, valid_fraction=VALID_FRACTION):
    """Consultation-level REFLECT / VALID draw, stratified by corpus stratum and by whether
    the consultation carries a partial omission - the class every measure in this study
    collapses on, and the one a 20-consultation VALID could otherwise miss entirely."""
    by_c = defaultdict(list)
    for p in pairs:
        by_c[(p["stratum"], p["id"])].append(p)
    rows = []
    for key, ps in sorted(by_c.items()):
        kinds = Counter(_kind_of(p)[0] for p in ps)
        rows.append({"key": key, "stratum": key[0],
                     "has_partial": kinds.get("omit-partial", 0) > 0,
                     "n_pairs": len(ps), "kinds": dict(kinds)})
    rng = random.Random(seed)
    valid = set()
    for bucket in sorted({(r["stratum"], r["has_partial"]) for r in rows}):
        members = [r for r in rows if (r["stratum"], r["has_partial"]) == bucket]
        rng.shuffle(members)
        n = int(round(valid_fraction * len(members)))
        n = max(1, n) if len(members) >= 3 else n
        valid |= {m["key"] for m in members[:n]}
    reflect = {r["key"] for r in rows} - valid
    return reflect, valid, rows


def build_partition(seed=SPLIT_SEED, valid_fraction=VALID_FRACTION, write=True):
    test_pairs, test_info = _pairs(TEST_FILE)
    all_pairs, train_info = _pairs(TRAIN_FILE)
    test_cons = _consultations(test_pairs)
    train_pairs = [p for p in all_pairs if (p["stratum"], p["id"]) not in test_cons]
    train_cons = _consultations(train_pairs)

    reflect_cons, valid_cons, rows = _split_train_consultations(train_pairs, seed, valid_fraction)
    excl_pairs, excl_cons = _exclusions()

    # ---- the assertions this file exists for
    assert not (test_cons & train_cons), "TEST and TRAIN share consultations"
    assert not (reflect_cons & valid_cons), "REFLECT and VALID share consultations"
    assert reflect_cons | valid_cons == train_cons, "the TRAIN split does not cover TRAIN"
    test_ids = {p["pair_id"] for p in test_pairs}
    train_ids = {p["pair_id"] for p in train_pairs}
    assert not (test_ids & train_ids), "TEST and TRAIN share pair ids"
    assert not (train_ids & excl_pairs), "TRAIN contains a pair id an earlier optimisation run trained on"
    assert not (train_cons & excl_cons), "TRAIN contains a consultation an earlier optimisation run trained on"
    assert not (test_ids & excl_pairs) and not (test_cons & excl_cons), "TEST is contaminated"
    assert all(p.get("split") != "gepa_dev" for p in train_pairs + test_pairs)

    def summarise(pairs, cons):
        kinds = Counter(_kind_of(p)[0] for p in pairs)
        return {"n_consultations": len(cons), "n_pairs": len(pairs),
                "n_notes": len(pairs) + len(cons),   # +1 clean twin per consultation
                "by_class": dict(kinds),
                "by_residual_level": dict(Counter(_kind_of(p)[1]["residual_level"]
                                                  for p in pairs if p["type"] == "omit")),
                "by_severity": dict(Counter(_kind_of(p)[1]["severity"] for p in pairs)),
                "by_stratum": dict(Counter(p["stratum"] for p in pairs))}

    reflect_pairs = [p for p in train_pairs if (p["stratum"], p["id"]) in reflect_cons]
    valid_pairs = [p for p in train_pairs if (p["stratum"], p["id"]) in valid_cons]

    man = {
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "what": "v3 prompt-optimisation partition: TEST is held out entirely; TRAIN splits into REFLECT "
                "(minibatches + reflection traces) and VALID (acceptance, Pareto rewards, "
                "winner choice). Consultation-level throughout.",
        "spec": "gepa/V2-DESIGN.md (v3 section)",
        "split_seed": seed, "valid_fraction": valid_fraction,
        "files": {"train_source": TRAIN_FILE, "train_sha256": train_info["sha256"],
                  "train_dataset_version": train_info["dataset_version"],
                  "test_source": TEST_FILE, "test_sha256": test_info["sha256"],
                  "test_dataset_version": test_info["dataset_version"],
                  "exclusions": "gepa/eval_exclusions.json",
                  "exclusions_sha256": _file_sha(EXCLUSIONS)},
        "test": summarise(test_pairs, test_cons),
        "train": summarise(train_pairs, train_cons),
        "reflect": summarise(reflect_pairs, reflect_cons),
        "valid": summarise(valid_pairs, valid_cons),
        "assertions": {
            "test_train_consultation_overlap": 0, "test_train_pair_overlap": 0,
            "reflect_valid_consultation_overlap": 0,
            "train_x_wf_trained_pair_ids": 0, "train_x_wf_trained_consultations": 0,
            "test_x_wf_trained_pair_ids": 0, "test_x_wf_trained_consultations": 0,
            "gepa_dev_pairs_present": 0,
            "checked_against": {"n_wf_trained_pair_ids": len(excl_pairs),
                                "n_wf_trained_consultations": len(excl_cons)}},
        "consultations": {
            "test": sorted(f"{s}|{c}" for s, c in test_cons),
            "reflect": sorted(f"{s}|{c}" for s, c in reflect_cons),
            "valid": sorted(f"{s}|{c}" for s, c in valid_cons)},
        "pair_ids": {"valid": sorted(p["pair_id"] for p in valid_pairs)},
    }
    if write:
        json.dump(man, open(PARTITION, "w"), indent=1)
    return man


# ---------------------------------------------------------------- loading a split
_CACHE = {}


def load_split(split):
    """(examples, blocks, meta) for 'reflect' | 'valid' | 'train' | 'test'.

    Re-asserts the manifest's disjointness claims against the files on disk every time, so a
    dataset edited between invocations cannot silently move a consultation across the wall.
    """
    if split in _CACHE:
        return _CACHE[split]
    if not os.path.exists(PARTITION):
        raise SystemExit("gepa/v3_partition.json missing - run "
                         "python3 gepa/gepa_v3_data.py --build first")
    man = json.load(open(PARTITION))
    if split == "test":
        pairs, info = _pairs(TEST_FILE)
        if info["sha256"] != man["files"]["test_sha256"]:
            raise SystemExit("TEST file changed since the partition was built")
        keep = None
    else:
        pairs, info = _pairs(TRAIN_FILE)
        if info["sha256"] != man["files"]["train_sha256"]:
            raise SystemExit("TRAIN file changed since the partition was built")
        test_cons = {tuple(k.split("|", 1)) for k in man["consultations"]["test"]}
        pairs = [p for p in pairs if (p["stratum"], p["id"]) not in test_cons]
        if split == "train":
            keep = None
        else:
            keep = {tuple(k.split("|", 1)) for k in man["consultations"][split]}
    examples, blocks, tprov = build_examples(pairs, keep)
    meta = {"split": split, "source": info["pairs_file"], "sha256": info["sha256"],
            "dataset_version": info["dataset_version"], "n_examples": len(examples),
            "n_consultations": len(blocks), "transcript_sources": tprov,
            "by_kind": dict(Counter(e["kind"] for e in examples)),
            "partition_generated_utc": man["generated_utc"]}
    _CACHE[split] = (examples, blocks, meta)
    return _CACHE[split]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--valid-fraction", type=float, default=VALID_FRACTION)
    args = ap.parse_args()
    if args.build:
        man = build_partition(args.seed, args.valid_fraction)
        print(f"wrote {os.path.relpath(PARTITION, ROOT)}")
    else:
        man = json.load(open(PARTITION))
    for name in ("test", "train", "reflect", "valid"):
        s = man[name]
        print(f"{name:>8}: {s['n_consultations']:>3} consultations, {s['n_pairs']:>3} pairs, "
              f"{s['n_notes']:>3} notes | {s['by_class']}")
        print(f"          residual {s['by_residual_level']} | severity {s['by_severity']}")
    print(f"assertions: {man['assertions']}")
    for split in ("reflect", "valid", "test"):
        ex, blocks, meta = load_split(split)
        print(f"  loaded {split}: {len(ex)} examples over {len(blocks)} consultations "
              f"{meta['by_kind']}")


if __name__ == "__main__":
    main()
