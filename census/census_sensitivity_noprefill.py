#!/usr/bin/env python3
"""Census note-level rate with the no-patient-record classes removed (sensitivity analysis).

Every product in the census was handed a transcript and nothing else: no
demographics header, no problem list, no encounter date. Two of the five largest
verified-finding clusters are what a product with those fields produces when it
has nothing to populate them from:

  - "Invented patient identity: name and sex"            93 findings
  - "Relative timing converted to invented calendar dates" 44 findings

That is 137 of the 618, 22% of the census. The manuscript says the design
inflates the count for these two classes and does not say by how much. This
script says by how much.

Method. Findings are removed, notes are not: a note keeps its place in the
denominator and loses only the excluded findings, so it leaves the numerator
only if it had nothing else wrong with it. Each rung carries a Wilson interval
(notes independent) and a consultation-clustered percentile bootstrap by exactly
census_note_ci.py's method and conventions - whole consultations resampled with
replacement, 10,000 draws, a per-row rng seeded "<SEED>:<row>" so results do not
depend on computation order. Rung A reuses that script's row tags, so its
intervals reproduce the published ones bit for bit and act as the join check.

The ladder is deliberately not collapsed into one number. A deployed scribe
legitimately knows the encounter date and may resolve "yesterday" against it, so
the date class is a weaker artefact claim than the identity class. Present the
ladder; let the reader pick their own exclusion.

The 55 unassigned findings. The clustering step covers 563 of the 618 and leaves
55 as HDBSCAN noise, so they belong to no cluster and are RETAINED at every rung
by default. They are not left to fall silently into one side: NOISE_DATE_LIKE
below names, with a reason each, the ones an analyst reading all 55 would put in
the date class if forced to, and rung D removes them as an upper bound. Zero of
the 55 concern an invented name or sex, which is asserted rather than asserted-
of-me: see the assertion in `check_noise_identity`.

Every count is asserted against the census paper's published figures before anything
is written; the script refuses to write on any mismatch.

Writes results/census-sensitivity/census_sensitivity_noprefill.json.
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
import math
import os
import random
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repository root
MASTER = os.path.join(HERE, "master")
OUTDIR = os.path.join(HERE, "results", "census-sensitivity")

# Same seed and draw count as census_note_ci.py. Rung A's row tags are that
# script's tags, so rung A's clustered intervals are the published ones.
SEED = 20260825
DRAWS = 10000

IDENTITY_LABEL = "Invented patient identity: name and sex"
DATES_LABEL = "Relative timing converted to invented calendar dates"

# Published figures this script refuses to proceed without reproducing.
EXPECT_CLUSTER_SIZES = {IDENTITY_LABEL: 93, DATES_LABEL: 44}    # published cluster sizes
EXPECT_N_CLUSTERS, EXPECT_N_NOISE = 17, 55                      # published clusters / noise
EXPECT_ROWS = {                                                 # published note-level rows
    "pooled": (177, 565), "scribe_A": (60, 282), "scribe_B": (57, 141), "scribe_C": (60, 142),
}
# The clustered intervals as published, to 1dp.
EXPECT_CLUSTERED_1DP = {
    "pooled": (27.0, 35.6), "scribe_A": (16.0, 27.0), "scribe_B": (32.6, 48.9), "scribe_C": (33.8, 50.0),
}

# The 55 unassigned findings, read individually. These are the ones a reader
# could reasonably assign to the date class: each converts vague or relative
# timing into a specific date or duration the transcript never supplied, which
# is the same failure the 44-member cluster describes. The remaining 48 are not
# date findings (the thyroxine ATTRIBUTION items below are excluded on purpose -
# their failure is law-firm-versus-doctor, which a patient record does not fix).
# This list is an analyst judgement, not an output of the clustering, which is
# why it drives its own clearly-labelled rung and nothing else.
NOISE_DATE_LIKE = {
    "scribe_B|authored|gastro_temporal_dose|audio#temporal#0":
        "relative 'yesterday' rendered as the calendar day 'Monday'",
    "scribe_C|primock|day1_consultation10|audio#temporal#0":
        "'a couple of years ago' invented for a reaction the transcript never dated",
    "scribe_B|primock|day3_consultation03|audio#dose_value#2":
        "3.5-month cessation duration invented; patient could not date it",
    "scribe_C|primock|day3_consultation03|audio#temporal#1":
        "same 3.5-month duration invented, other product",
    "scribe_B|primock|day3_consultation03|audio#fabrication#5":
        "assigns the 3.5-month timeframe (also the one item the precision sitting rejected)",
    "scribe_B|primock|day3_consultation03|audio#open#2":
        "invents the 3.5-month timeframe on the open pass",
    "scribe_B|primock|day3_consultation03|audio#misplaced_text#0":
        "attaches cessation to a 3.5-month timeline the consultation did not establish",
}

# Words that would betray an invented name or sex among the 55. Used only to
# assert the absence, never to select anything.
IDENTITY_WORDS = re.compile(
    r"\b(name|names|named|forename|surname|gender|sex|male|female|man|woman|men|women|"
    r"mr|mrs|ms|miss|he|she|his|her|him|hers|identity|demographics?|patient's name)\b", re.I)


def die(msg):
    sys.stderr.write("MISMATCH: %s\n" % msg)
    sys.exit(1)


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def cluster_boot(clusters, seed_tag):
    """Percentile CI under resampling of whole consultations. census_note_ci.py's."""
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


# ---------------------------------------------------------------- load + assert

def _store(name):
    """One of the census record stores, which are the study's own construction artifacts
    and not part of this release. The released findings are the dataset repository's
    `judgements/findings/`, and this analysis's own output is released beside them as
    `judgements/findings/census_sensitivity_noprefill.json`."""
    path = os.path.join(MASTER, name)
    if not os.path.exists(path):
        raise SystemExit("cannot find %s - the census record stores are not part of this "
                         "release; the released findings are the dataset repository's "
                         "judgements/findings/" % path)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


m = _store("findings_master.json")
vm = _store("findings_verified_master.json")
cl = _store("findings_clusters.json")

roster = m["per_note"]
if len(roster) != 565:
    die("roster: expected 565 notes, got %d" % len(roster))
if len({n["consultation"] for n in roster}) != 142:
    die("roster: expected 142 consultations")

verified = [i for i in vm["all_issues"] if i.get("verdict", {}).get("is_real")]
if len(vm["all_issues"]) != 5898 or len(verified) != 618:
    die("verified master: %d candidates, %d verified (want 5898 / 618)"
        % (len(vm["all_issues"]), len(verified)))
by_id = {i["finding_id"]: i for i in verified}
if len(by_id) != 618:
    die("finding_id is not unique across the 618")

if cl.get("n_clusters") != EXPECT_N_CLUSTERS or cl.get("n_noise") != EXPECT_N_NOISE:
    die("clusters: %s clusters / %s noise, the published figures say %d / %d"
        % (cl.get("n_clusters"), cl.get("n_noise"), EXPECT_N_CLUSTERS, EXPECT_N_NOISE))
if cl.get("n_findings") != 618:
    die("clusters: n_findings %s" % cl.get("n_findings"))

scaffold = {s["label"]: s for s in cl["scaffold"]}
members = {}
for label, want in EXPECT_CLUSTER_SIZES.items():
    if label not in scaffold:
        die("cluster %r absent from the scaffold" % label)
    ids = scaffold[label]["member_finding_ids"]
    if scaffold[label]["size"] != want or len(ids) != want:
        die("cluster %r: size %d / %d members, the published figure is %d"
            % (label, scaffold[label]["size"], len(ids), want))
    missing = [x for x in ids if x not in by_id]
    if missing:
        die("cluster %r: %d members are not verified findings" % (label, len(missing)))
    members[label] = set(ids)
if members[IDENTITY_LABEL] & members[DATES_LABEL]:
    die("the two excluded clusters overlap")

noise_ids = {k for k, v in cl["assignments"].items() if v == -1}
if len(noise_ids) != EXPECT_N_NOISE or any(x not in by_id for x in noise_ids):
    die("noise bucket: %d ids, not %d verified findings" % (len(noise_ids), EXPECT_N_NOISE))
if not set(NOISE_DATE_LIKE) <= noise_ids:
    die("NOISE_DATE_LIKE names ids that are not in the noise bucket")


def check_noise_identity():
    """Assert that no unassigned finding is an invented-name-or-sex finding.

    A keyword sweep over all 55 descriptions; every hit is printed for a human
    read rather than silently dismissed, and the script stops if any hit is not
    on the reviewed-and-cleared list below.
    """
    cleared = {
        # Read individually 2026-08-26: each hit is a pronoun or an ordinary
        # clinical word, not an invented identity. Under the current pattern the
        # live hits are the day1_consultation07 throat-self-examination family
        # ("he"/"his") and day1_consultation14 ("her hypertension"). The
        # day3_consultation03 thyroxine entries below were hits under an earlier,
        # looser pattern; they were read then and are kept cleared so a future
        # widening of the pattern does not re-trip on material already reviewed.
        "scribe_B|primock|day1_consultation07|audio#attribution#0",
        "scribe_B|primock|day1_consultation07|audio#exam_provenance#0",
        "scribe_B|primock|day1_consultation07|audio#fabrication#0",
        "scribe_B|primock|day1_consultation07|audio#misplaced_text#1",
        "scribe_B|primock|day1_consultation07|audio#modality_hardening#0",
        "scribe_B|primock|day1_consultation07|audio#negation#0",
        "scribe_B|primock|day1_consultation07|audio#open#0",
        "scribe_C|primock|day1_consultation14|audio#omission#0",
        "scribe_B|primock|day3_consultation03|audio#attribution#0",
        "scribe_B|primock|day3_consultation03|audio#dose_value#2",
        "scribe_B|primock|day3_consultation03|audio#fabrication#5",
        "scribe_B|primock|day3_consultation03|audio#misplaced_text#0",
        "scribe_B|primock|day3_consultation03|audio#open#2",
        "scribe_C|primock|day1_consultation10|audio#temporal#0",
        "scribe_C|primock|day3_consultation03|audio#attribution#1",
        "scribe_C|primock|day3_consultation03|audio#misplaced_text#0",
        "scribe_C|primock|day3_consultation03|audio#temporal#1",
        "scribe_B|authored|gastro_temporal_dose|audio#temporal#0",
    }
    hits = sorted(x for x in noise_ids if IDENTITY_WORDS.search(by_id[x]["description"]))
    unreviewed = [x for x in hits if x not in cleared]
    if unreviewed:
        sys.stderr.write("UNREVIEWED identity-word hits in the noise bucket:\n")
        for x in unreviewed:
            sys.stderr.write("  %s\n    %s\n" % (x, by_id[x]["description"]))
        die("the noise bucket has %d unreviewed identity-word hits; read them, then "
            "either clear them here or add them to an identity bound" % len(unreviewed))
    return hits


identity_hits = check_noise_identity()

# ------------------------------------------------------------------- the ladder

RUNGS = [
    ("A_all", "All 618 verified findings (published census)", set()),
    ("B_no_identity", "Excluding invented patient identity (93)", members[IDENTITY_LABEL]),
    ("C_no_identity_no_dates", "Excluding invented identity and invented calendar dates (137)",
     members[IDENTITY_LABEL] | members[DATES_LABEL]),
    ("D_bound_incl_unassigned", "Upper bound: C plus the %d date-like unassigned findings (%d)"
     % (len(NOISE_DATE_LIKE), 137 + len(NOISE_DATE_LIKE)),
     members[IDENTITY_LABEL] | members[DATES_LABEL] | set(NOISE_DATE_LIKE)),
]

notes_by_vendor = {v: [n for n in roster if n["scribe"] == v] for v in ("scribe_A", "scribe_B", "scribe_C")}
results = {}

for rung_key, rung_desc, excluded in RUNGS:
    kept = [i for i in verified if i["finding_id"] not in excluded]
    flagged_notes = {i["note_key"] for i in kept}
    flags = {n["note_key"]: n["note_key"] in flagged_notes for n in roster}
    rows = {}

    def add_row(name, notes, published_tag=None):
        k = sum(flags[n["note_key"]] for n in notes)
        n = len(notes)
        clusters = defaultdict(list)
        for note in notes:
            clusters[note["consultation"]].append(flags[note["note_key"]])
        # Rung A reuses census_note_ci.py's row tags so its draws are identical.
        tag = published_tag if published_tag else "%s:%s" % (rung_key, name)
        lo, hi = cluster_boot(list(clusters.values()), tag)
        wlo, whi = wilson(k, n)
        rows[name] = {
            "k": k, "n": n, "p": round(k / n, 6),
            "wilson_95": [round(wlo, 6), round(whi, 6)],
            "clustered_95": [round(lo, 6), round(hi, 6)],
            "n_clusters": len(clusters), "boot_tag": tag,
        }

    published = rung_key == "A_all"
    add_row("pooled", roster, "pooled" if published else None)
    for v in ("scribe_A", "scribe_B", "scribe_C"):
        add_row(v, notes_by_vendor[v], v if published else None)

    if published:
        for name, want in EXPECT_ROWS.items():
            got = (rows[name]["k"], rows[name]["n"])
            if got != want:
                die("join check %s: recomputed %d/%d, the published figures say %d/%d"
                    % (name, got[0], got[1], want[0], want[1]))
        for name, (wlo, whi) in EXPECT_CLUSTERED_1DP.items():
            got = (round(100 * rows[name]["clustered_95"][0], 1),
                   round(100 * rows[name]["clustered_95"][1], 1))
            if got != (wlo, whi):
                die("join check %s clustered: recomputed [%.1f, %.1f], the published "
                    "interval is [%.1f, %.1f]" % (name, got[0], got[1], wlo, whi))

    results[rung_key] = {
        "description": rung_desc,
        "n_findings_excluded": len(excluded),
        "n_findings_kept": len(kept),
        "excluded_by_vendor": {v: sum(1 for x in excluded if by_id[x]["scribe"] == v)
                               for v in ("scribe_A", "scribe_B", "scribe_C")},
        "rows": rows,
    }

# Notes that leave the numerator only because they had nothing else wrong.
base_notes = {i["note_key"] for i in verified}
for rung_key, _, excluded in RUNGS[1:]:
    kept_notes = {i["note_key"] for i in verified if i["finding_id"] not in excluded}
    lost = sorted(base_notes - kept_notes)
    results[rung_key]["notes_cleared"] = len(lost)
    results[rung_key]["notes_cleared_by_vendor"] = {
        v: sum(1 for nk in lost if nk.split("|")[0] == v) for v in ("scribe_A", "scribe_B", "scribe_C")}

# ------------------------------------------------------------------------ write

out = {
    "generated_by": "census_sensitivity_noprefill.py",
    "question": "How much of the census's note-level rate is attributable to the two "
                "failure classes that a patient record would have prefilled?",
    "seed": SEED, "draws": DRAWS, "unit": "consultation",
    "method": "findings removed, notes retained in the denominator; Wilson plus a "
              "consultation-clustered percentile bootstrap by census_note_ci.py's "
              "method, per-row rng seeded '%d:<tag>' so results are order-independent. "
              "Rung A reuses that script's row tags and reproduces the published "
              "intervals exactly, as the join check." % SEED,
    "sources": ["master/findings_master.json", "master/findings_verified_master.json",
                "master/findings_clusters.json"],
    "asserted_against": "the census paper's published cluster sizes, note-level rows and "
                        "clustered intervals",
    "unassigned_handling": {
        "n_unassigned": EXPECT_N_NOISE,
        "default": "retained at every rung - they belong to no cluster",
        "identity_class": "none of the 55 concerns an invented name or sex; %d descriptions "
                          "contain an identity word and all %d were read and cleared"
                          % (len(identity_hits), len(identity_hits)),
        "date_class_bound": NOISE_DATE_LIKE,
        "bound_rung": "D_bound_incl_unassigned",
    },
    "rungs": results,
}

os.makedirs(OUTDIR, exist_ok=True)
path = os.path.join(OUTDIR, "census_sensitivity_noprefill.json")
json.dump(out, open(path, "w"), indent=1)
print("wrote %s\n" % path)

for rung_key, rung_desc, _ in RUNGS:
    r = results[rung_key]
    print("%s - %s" % (rung_key, rung_desc))
    print("  %d findings excluded, %d kept%s" % (
        r["n_findings_excluded"], r["n_findings_kept"],
        "" if rung_key == "A_all" else "; %d notes cleared (%s)" % (
            r["notes_cleared"], ", ".join("%s %d" % (v, c) for v, c
                                          in r["notes_cleared_by_vendor"].items()))))
    for name in ("pooled", "scribe_A", "scribe_B", "scribe_C"):
        row = r["rows"][name]
        print("  %-8s %4d/%4d = %5.1f%%  wilson [%.1f, %.1f]  clustered [%.1f, %.1f]" % (
            name, row["k"], row["n"], 100 * row["p"],
            100 * row["wilson_95"][0], 100 * row["wilson_95"][1],
            100 * row["clustered_95"][0], 100 * row["clustered_95"][1]))
    print()
