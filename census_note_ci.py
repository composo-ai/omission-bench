#!/usr/bin/env python3
"""Consultation-clustered intervals for the census's note-level rates (P2).

Notes from one consultation share a transcript, so intervals that treat the 565
notes as independent understate the design's uncertainty. This script computes,
from the master artifacts alone:

  - share of notes with >=1 verified finding: pooled and per product;
  - share of notes with >=1 verified finding placed under tier-1 omission
    (the Taylor comparison in P2 section 4.4);
  - the candidate-level panel survival rate, for the record.

Each row carries the Wilson interval (notes independent) and a consultation-
clustered bootstrap percentile interval (whole consultations resampled with
replacement, 10,000 draws, a fixed per-row seed so results do not depend on
computation order). Every count is asserted against FINDINGS 16.5/16.6; the
script refuses to write on any mismatch.

Motivated by an independent pre-submission review of P2 (2026-08-25). Writes
master/findings_note_ci_clustered.json; P2's float extractor reads that artifact.
"""

import json
import math
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(HERE, "master")
SEED = 20260825
DRAWS = 10000


def die(msg):
    sys.stderr.write("MISMATCH: %s\n" % msg)
    sys.exit(1)


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def cluster_boot(clusters, seed_tag):
    """Percentile CI for the pooled rate under resampling of whole clusters."""
    rng = random.Random("%d:%s" % (SEED, seed_tag))
    k = len(clusters)
    draws = []
    for _ in range(DRAWS):
        num = den = 0
        for _ in range(k):
            c = clusters[rng.randrange(k)]
            num += sum(c)
            den += len(c)
        draws.append(num / den)
    draws.sort()
    return draws[int(DRAWS * 0.025) - 1], draws[int(DRAWS * 0.975)]


m = json.load(open(os.path.join(MASTER, "findings_master.json")))
vm = json.load(open(os.path.join(MASTER, "findings_verified_master.json")))
roster = m["per_note"]
if len(roster) != 565:
    die("roster: expected 565 notes, got %d" % len(roster))
if len({n["consultation"] for n in roster}) != 142:
    die("roster: expected 142 consultations")

# Placement of verified findings into published tier-1 classes: the same code
# path the analysis stage and extract_p2_floats.py use, so this cannot drift.
import importlib.util
sys.path.insert(0, HERE)
import taxonomy_common as _tc
_spec = importlib.util.spec_from_file_location(
    "cross_scribe_matches", os.path.join(HERE, "pilot", "scripts", "cross_scribe_matches.py"))
_xs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_xs)
_tc.assert_family_keys_known(_xs.FAMILIES)

verified_notes = set()
omission_notes = set()
n_verified = 0
for i in vm["all_issues"]:
    if not i.get("verdict", {}).get("is_real"):
        continue
    n_verified += 1
    verified_notes.add(i["note_key"])
    if _tc.frame_place(i, family_fn=_xs.family)[1] == "omission":
        omission_notes.add(i["note_key"])
if len(vm["all_issues"]) != 5898 or n_verified != 618:
    die("verified master: %d candidates, %d verified" % (len(vm["all_issues"]), n_verified))

has_any = {n["note_key"]: n["note_key"] in verified_notes for n in roster}
has_om = {n["note_key"]: n["note_key"] in omission_notes for n in roster}

EXPECT = {  # FINDINGS 16.5 / 16.6
    "pooled": (177, 565), "scribe_A": (60, 282), "scribe_B": (57, 141), "scribe_C": (60, 142),
    "omission_notes": (87, 565),
}

rows = {}


def add_row(name, flags, notes):
    k = sum(flags[n["note_key"]] for n in notes)
    n = len(notes)
    if EXPECT.get(name) and (k, n) != EXPECT[name]:
        die("%s: recomputed %d/%d, FINDINGS says %d/%d" % (name, k, n, *EXPECT[name]))
    clusters = defaultdict(list)
    for note in notes:
        clusters[note["consultation"]].append(flags[note["note_key"]])
    lo, hi = cluster_boot(list(clusters.values()), name)
    wlo, whi = wilson(k, n)
    rows[name] = {
        "k": k, "n": n, "p": round(k / n, 6),
        "wilson_95": [round(wlo, 6), round(whi, 6)],
        "clustered_95": [round(lo, 6), round(hi, 6)],
        "n_clusters": len(clusters),
    }


add_row("pooled", has_any, roster)
for v in ("scribe_A", "scribe_B", "scribe_C"):
    add_row(v, has_any, [n for n in roster if n["scribe"] == v])
add_row("omission_notes", has_om, roster)

# Candidate-level panel survival, clustered by consultation, for the record.
cand_clusters = defaultdict(list)
for i in vm["all_issues"]:
    cand_clusters[i["consultation"]].append(bool(i.get("verdict", {}).get("is_real")))
lo, hi = cluster_boot(list(cand_clusters.values()), "survival")
wlo, whi = wilson(618, 5898)
rows["panel_survival_candidates"] = {
    "k": 618, "n": 5898, "p": round(618 / 5898, 6),
    "wilson_95": [round(wlo, 6), round(whi, 6)],
    "clustered_95": [round(lo, 6), round(hi, 6)],
    "n_clusters": len(cand_clusters),
}

out = {
    "generated_by": "census_note_ci.py",
    "seed": SEED, "draws": DRAWS, "unit": "consultation",
    "method": "percentile bootstrap resampling whole consultations with replacement; "
              "per-row rng seeded '%d:<row>' so results are order-independent" % SEED,
    "sources": ["master/findings_master.json", "master/findings_verified_master.json"],
    "rows": rows,
}
path = os.path.join(MASTER, "findings_note_ci_clustered.json")
json.dump(out, open(path, "w"), indent=1)
print("wrote %s" % path)
for name, r in rows.items():
    print("%-26s %4d/%4d = %6.2f%%  wilson [%.1f, %.1f]  clustered [%.1f, %.1f]  (%d clusters)" % (
        name, r["k"], r["n"], 100 * r["p"],
        100 * r["wilson_95"][0], 100 * r["wilson_95"][1],
        100 * r["clustered_95"][0], 100 * r["clustered_95"][1], r["n_clusters"]))
