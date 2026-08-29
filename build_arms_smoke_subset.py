#!/usr/bin/env python3
"""build_arms_smoke_subset.py - a purposive slice of dataset_v2 for the arms smoke.

Written 2026-08-12. the lead author's ask: the arms smoke must tell us more than the price - it has
to show whether each arm is WORKING (parses, discriminates, and does something useful on
omissions) so the scale-up call is easy.

`--limit N` cannot do that: it takes the first N pairs, which in dataset_v2 order means a
near-pure block of one stratum and one class. This builds a stratified subset instead -
every class x severity cell the full run contains, sampled deterministically, so each arm
gets measured on the same spread of difficulty the real run will present.

Deterministic: sorted pair_ids, fixed seed, no clock. Re-running gives the identical file.
"""
import argparse, collections, hashlib, json, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "master/dataset_v2.json")
OUT = os.path.join(HERE, "master/arms_smoke_subset.json")

# Per class x severity. Omissions are the paper's subject so they get the depth; add and
# change are controls that only need to show the arms are not simply broken.
QUOTA = {
    ("omit-complete", "critical"): 6, ("omit-complete", "supporting"): 6,
    ("omit-complete", "peripheral"): 4,
    ("omit-partial", "critical"): 6, ("omit-partial", "supporting"): 6,
    ("omit-partial", "peripheral"): 4,
    ("add", "critical"): 3, ("change", "critical"): 3,
}

# --profile confirm (added 2026-08-14 for the Arm B confirmation batch). Three changes from
# the smoke profile, each for a stated reason:
#   * the partial cells split by RESIDUAL LEVEL, because partial-strong is the cell the grid
#     dies in (0.526) and the smoke drew only 5 of them - too few to say anything;
#   * ~4x the depth on omissions and ~3x on commissions (the fabrication-blindness reading
#     rested on n=6, which is not a finding, it is an anecdote);
#   * the draw is CONSULTATION-FIRST (--consultation-first), taking whole dense consultations
#     in turn rather than sampling pairs independently. Extraction is bought once per
#     consultation and cached, so it is the fixed cost that dominates: the smoke's 1.19
#     pairs/consultation made extraction ~55% of B2's bill, and a dense draw makes the same
#     150 pairs cost roughly half as much to judge.
CONFIRM_QUOTA = {
    ("omit-complete", "critical"): 26, ("omit-complete", "supporting"): 22,
    ("omit-complete", "peripheral"): 14,
    ("omit-partial-weak", "critical"): 16, ("omit-partial-weak", "supporting"): 16,
    ("omit-partial-weak", "peripheral"): 4,
    ("omit-partial-strong", "critical"): 16, ("omit-partial-strong", "supporting"): 16,
    ("omit-partial-strong", "peripheral"): 1,
    ("add", "critical"): 10, ("change", "critical"): 10,
}
PROFILES = {"smoke": QUOTA, "confirm": CONFIRM_QUOTA}


def cell_of(pair, split_residual):
    """The quota cell a pair belongs to. `split_residual` grades the partials by how much of
    the fact survives, which is the axis the confirmation batch is powered on."""
    cls = pair.get("class")
    if split_residual and cls == "omit-partial":
        return (f"omit-{pair.get('residual_level')}", pair.get("severity"))
    return (cls, pair.get("severity"))


def draw_consultation_first(buckets, quota, rng):
    """Fill the quota by taking whole consultations, densest first.

    Ranking is by how many still-wanted pairs a consultation can contribute, recomputed as
    quotas fill, with the pair_id as the tie-break - so it is deterministic and seed-free by
    construction. Within a chosen consultation every pair whose cell still has room is taken,
    which is what concentrates the notes and amortises the cached extraction call.
    """
    remaining = dict(quota)
    by_consult = collections.defaultdict(list)
    for cell, ps in buckets.items():
        for p in ps:
            by_consult[(p.get("stratum"), p.get("id"))].append((cell, p))
    picked = []
    while any(v > 0 for v in remaining.values()):
        def yield_of(key):
            got, room = 0, dict(remaining)
            for cell, _ in sorted(by_consult[key], key=lambda cp: cp[1]["pair_id"]):
                if room.get(cell, 0) > 0:
                    room[cell] -= 1
                    got += 1
            return got
        best = max(sorted(by_consult), key=lambda k: (yield_of(k), -len(by_consult[k])))
        if yield_of(best) == 0:
            break
        for cell, p in sorted(by_consult.pop(best), key=lambda cp: cp[1]["pair_id"]):
            if remaining.get(cell, 0) > 0:
                remaining[cell] -= 1
                picked.append(p)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--profile", default="smoke", choices=sorted(PROFILES))
    ap.add_argument("--exclude", default=None,
                    help="a subset file whose pair_ids are held OUT of this draw, so the "
                         "result is a genuine confirmation set and not a re-measure")
    ap.add_argument("--consultation-first", action="store_true",
                    help="take whole dense consultations in turn instead of sampling pairs "
                         "independently (amortises the cached per-consultation extraction)")
    a = ap.parse_args()

    blob = json.load(open(SRC))
    pairs = [p for p in blob["pairs"] if p.get("eval_set") is not False]
    excluded = set()
    if a.exclude:
        path = a.exclude if os.path.isabs(a.exclude) else os.path.join(HERE, a.exclude)
        excluded = {p["pair_id"] for p in json.load(open(path))["pairs"]}
        pairs = [p for p in pairs if p["pair_id"] not in excluded]

    quota = PROFILES[a.profile]
    split_residual = any("partial-" in c for c, _ in quota)
    buckets = collections.defaultdict(list)
    for p in pairs:
        buckets[cell_of(p, split_residual)].append(p)

    rng = random.Random(a.seed)
    if a.consultation_first:
        picked = draw_consultation_first(buckets, quota, rng)
        took = collections.Counter(cell_of(p, split_residual) for p in picked)
        report = [(cell, want, len(buckets.get(cell, [])), took.get(cell, 0))
                  for cell, want in sorted(quota.items(), key=str)]
    else:
        picked, report = [], []
        for cell, want in sorted(quota.items(), key=str):
            have = sorted(buckets.get(cell, []), key=lambda p: p["pair_id"])
            take = have if len(have) <= want else rng.sample(have, want)
            picked.extend(take)
            report.append((cell, want, len(have), len(take)))

    picked.sort(key=lambda p: p["pair_id"])
    picked.sort(key=lambda p: p["pair_id"])
    out = {k: v for k, v in blob.items() if k != "pairs"}
    out["pairs"] = picked
    out["what"] = ("Stratified arms-smoke subset of dataset_v2 - decision-grade sample for "
                   "the go/no-go on scaling the judge arms. NOT a paper substrate."
                   if a.profile == "smoke" else
                   "Stratified CONFIRMATION subset of dataset_v2 for Arm B (the pipeline "
                   "judge), disjoint from the smoke subset it confirms. NOT a paper "
                   "substrate.")
    out["subset_of"] = {"file": "master/dataset_v2.json",
                        "dataset_version": blob.get("dataset_version"),
                        "seed": a.seed, "profile": a.profile,
                        "draw": "consultation-first" if a.consultation_first else "per-cell",
                        "excluded_file": a.exclude, "n_excluded_pair_ids": len(excluded),
                        "quota": {f"{c}|{s}": n for (c, s), n in quota.items()}}
    out["dataset_version"] = (f"{blob.get('dataset_version')}-armsmoke" if a.profile == "smoke"
                              else f"{blob.get('dataset_version')}-armconfirm")
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)

    print(f"{'cell':32s} {'want':>5s} {'avail':>6s} {'took':>5s}")
    for cell, want, have, took in report:
        flag = "" if took == want else ("  <- corpus-limited" if have < want else "")
        print(f"  {str(cell):30s} {want:5d} {have:6d} {took:5d}{flag}")
    print(f"\n{len(picked)} pairs over {len({p['consultation'] if 'consultation' in p else p['id'] for p in picked})} consultations"
          f" -> {a.out}")
    print("sha256:", hashlib.sha256(open(a.out, 'rb').read()).hexdigest()[:12])


if __name__ == "__main__":
    main()
