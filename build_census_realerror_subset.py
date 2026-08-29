#!/usr/bin/env python3
"""Build master/census_realerror_261.json - the real-error scoring subset.

Design (fixed, seed 20260825):
  * all 87 census notes carrying at least one tier-1 OMISSION verified finding;
  * 174 notes with NO verified finding, drawn from the 388-note clean pool by a
    seeded stratified sample (product x substrate, proportional, largest
    remainder), mirroring second_panel.draw();
  * the 90 notes whose only verified findings are non-omission classes are
    EXCLUDED, including the 3 whose verified findings are all tier-1 unmapped.

Every published count is reproduced and asserted before a byte is written.
Tier-1 placement is RECOMPUTED via taxonomy_common.frame_place, never read off
the record's frame_tier1 field.

Read-only against everything except the one output file.
"""
import ast, hashlib, importlib.util, json, os, random, sys, time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

OUT = os.path.join(HERE, "master", "census_realerror_261.json")
SEED = 20260825
N_CLEAN_TARGET = 174
CACHE_DIR = os.path.join(HERE, "results", "w2-pipeline", "_cache")
PRODUCT_LETTER = {"scribe_A": "Scribe A", "scribe_B": "Scribe B", "scribe_C": "Scribe C"}

import taxonomy_common as T
_spec = importlib.util.spec_from_file_location(
    "cross_scribe_matches", os.path.join(HERE, "pilot", "scripts", "cross_scribe_matches.py"))
_xs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_xs)
T.assert_family_keys_known(_xs.FAMILIES)


def die(msg):
    sys.stderr.write("MISMATCH: %s\nREFUSING TO WRITE.\n" % msg)
    sys.exit(1)


# ---------------------------------------------------------------- integrity constants
def ci_constants():
    """census_note_ci.py's EXPECT dict, lifted by AST parse.

    Importing that module would execute it and rewrite one of its own artifacts,
    so the constants are read out of the source instead.
    """
    src = open(os.path.join(HERE, "census_note_ci.py")).read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "EXPECT":
                    return ast.literal_eval(node.value), src
    die("census_note_ci.py no longer defines EXPECT")


EXPECT, CI_SRC = ci_constants()
# the four totals census_note_ci.py guards inline with die() rather than in EXPECT
INLINE_GUARDS = {"n_notes": 565, "n_consultations": 142,
                 "n_candidates": 5898, "n_verified": 618}
for _v in INLINE_GUARDS.values():
    if str(_v) not in CI_SRC:
        die("census_note_ci.py no longer carries the inline guard %d" % _v)
# tier-1 distribution of the 618, from docs/FINDINGS.md section 16
EXPECT_TIER1 = {"wrong_output": 207, "addition": 181, "omission": 143,
                "misplaced_or_irrelevant": 46, "unmapped": 41}

integrity = []


def assert_eq(label, got, want):
    ok = (got == want)
    integrity.append({"check": label, "got": got, "expected": want, "ok": ok})
    print("%-48s got %-14s want %-14s %s"
          % (label, got, want, "OK" if ok else "*** MISMATCH ***"))
    if not ok:
        die("%s: got %r, expected %r" % (label, got, want))


# ---------------------------------------------------------------- reproduce
m = json.load(open(os.path.join(HERE, "master/findings_master.json")))
vm = json.load(open(os.path.join(HERE, "master/findings_verified_master.json")))
roster = m["per_note"]
issues = vm["all_issues"]

print("=== step 1: reproduce the census ===")
assert_eq("total notes (roster)", len(roster), INLINE_GUARDS["n_notes"])
assert_eq("total consultations", len({n["consultation"] for n in roster}),
          INLINE_GUARDS["n_consultations"])
assert_eq("total candidates", len(issues), INLINE_GUARDS["n_candidates"])

verified_notes, omission_notes = set(), set()
tier1_counts = Counter()
verified_by_note = defaultdict(list)      # note_key -> [(issue, tier1)]
n_verified = 0
for i in issues:
    if not i.get("verdict", {}).get("is_real"):
        continue
    n_verified += 1
    _t2, t1, _how = T.frame_place(i, family_fn=_xs.family)
    tier1_counts[t1 if t1 is not None else "unmapped"] += 1
    verified_notes.add(i["note_key"])
    verified_by_note[i["note_key"]].append((i, t1))
    if t1 == "omission":
        omission_notes.add(i["note_key"])

assert_eq("verified findings", n_verified, INLINE_GUARDS["n_verified"])
assert_eq("notes with >=1 verified finding", len(verified_notes), EXPECT["pooled"][0])
assert_eq("omission notes (tier-1 recomputed)", len(omission_notes), EXPECT["omission_notes"][0])

roster_keys = [n["note_key"] for n in roster]
roster_by_key = {n["note_key"]: n for n in roster}
no_finding = [k for k in roster_keys if k not in verified_notes]
excluded = [k for k in roster_keys if k in verified_notes and k not in omission_notes]
assert_eq("no-finding notes", len(no_finding), 388)
assert_eq("excluded (verified, no tier-1 omission)", len(excluded), 90)
assert_eq("omission + no-finding + excluded", len(omission_notes) + len(no_finding) + len(excluded),
          len(roster))

for k in sorted(set(tier1_counts) | set(EXPECT_TIER1)):
    assert_eq("tier-1 of the 618: %s" % k, tier1_counts.get(k, 0), EXPECT_TIER1.get(k, 0))
assert_eq("tier-1 total", sum(tier1_counts.values()), n_verified)

assert_eq("EXPECT pooled", list(EXPECT["pooled"]), [len(verified_notes), len(roster)])
for v in ("scribe_A", "scribe_B", "scribe_C"):
    sub = [n for n in roster if n["scribe"] == v]
    assert_eq("EXPECT %s (%s)" % (PRODUCT_LETTER[v], v),
              [sum(1 for n in sub if n["note_key"] in verified_notes), len(sub)],
              list(EXPECT[v]))
assert_eq("EXPECT omission_notes", list(EXPECT["omission_notes"]),
          [len(omission_notes), len(roster)])

# ---------------------------------------------------------------- step 2: unmapped-only
unmapped_only = sorted(k for k in excluded
                       if {t1 for _i, t1 in verified_by_note[k]} == {None})
unmapped_only_by_product = Counter(roster_by_key[k]["scribe"] for k in unmapped_only)
excluded_with_any_unmapped = sum(1 for k in excluded
                                 if None in {t1 for _i, t1 in verified_by_note[k]})
print("\n=== step 2: the unmapped-only edge case ===")
print("excluded notes whose verified findings are ALL tier-1 unmapped: %d" % len(unmapped_only))
print("  by product: %s" % {PRODUCT_LETTER[k]: v for k, v in sorted(unmapped_only_by_product.items())})
print("  decision: kept EXCLUDED (an unplaced finding gives no omission ground truth)")
print("  for context, excluded notes carrying >=1 unmapped finding: %d" % excluded_with_any_unmapped)

# ---------------------------------------------------------------- step 3: the draw
def draw_clean(pool_keys, seed=SEED, target=N_CLEAN_TARGET):
    """Stratified sample of whole notes, product x substrate, proportional, largest
    remainder, fixed seed - the note-level analogue of second_panel.draw()."""
    strata = defaultdict(list)
    for k in pool_keys:
        n = roster_by_key[k]
        strata[(n["scribe"], n["source"])].append(k)

    frac = target / len(pool_keys)
    alloc, remainders, base = {}, [], 0
    for k in sorted(strata):
        exact = frac * len(strata[k])
        alloc[k] = int(exact)
        remainders.append((exact - alloc[k], k))
        base += alloc[k]
    for _r, k in sorted(remainders, reverse=True)[: round(frac * len(pool_keys)) - base]:
        alloc[k] += 1

    rng = random.Random(seed)
    picked = []
    for k in sorted(strata):
        picked += rng.sample(sorted(strata[k]), alloc[k])
    picked.sort()
    return picked, strata, alloc, frac


clean_picked, clean_strata, clean_alloc, clean_frac = draw_clean(no_finding)
print("\n=== step 3: the draw ===")
print("pool %d clean notes, sampling fraction %.6f, drawn %d"
      % (len(no_finding), clean_frac, len(clean_picked)))
assert_eq("clean notes drawn", len(clean_picked), N_CLEAN_TARGET)
assert_eq("clean draw is a subset of the clean pool",
          set(clean_picked) <= set(no_finding), True)
assert_eq("clean draw has no duplicates", len(set(clean_picked)), len(clean_picked))
assert_eq("clean draw disjoint from omission arm",
          bool(set(clean_picked) & omission_notes), False)

print("\n%-12s %-11s %8s %8s %9s" % ("product", "substrate", "pool", "drawn", "exact"))
strata_out = {}
for k in sorted(clean_strata):
    scribe, src = k
    exact = clean_frac * len(clean_strata[k])
    realised = sum(1 for nk in clean_picked
                   if roster_by_key[nk]["scribe"] == scribe and roster_by_key[nk]["source"] == src)
    if realised != clean_alloc[k]:
        die("stratum %s: allocated %d, realised %d" % (k, clean_alloc[k], realised))
    print("%-12s %-11s %8d %8d %9.3f"
          % (PRODUCT_LETTER[scribe], src, len(clean_strata[k]), realised, exact))
    strata_out["%s|%s" % (scribe, src)] = {
        "product_letter": PRODUCT_LETTER[scribe], "scribe": scribe, "source": src,
        "clean_pool_notes": len(clean_strata[k]), "exact_allocation": round(exact, 6),
        "allocated": clean_alloc[k], "realised": realised,
    }

# marginal fit of the draw against the 388-note clean pool
def marginal_fit(sample_keys, pool_keys):
    out = {}
    for name, keyfn in (("product", lambda n: PRODUCT_LETTER[n["scribe"]]),
                        ("substrate", lambda n: n["source"]),
                        ("product_x_substrate",
                         lambda n: "%s|%s" % (PRODUCT_LETTER[n["scribe"]], n["source"]))):
        pop = Counter(keyfn(roster_by_key[k]) for k in pool_keys)
        smp = Counter(keyfn(roster_by_key[k]) for k in sample_keys)
        out[name] = {str(k): {
            "pool": v, "pool_share": round(v / len(pool_keys), 6),
            "drawn": smp.get(k, 0), "drawn_share": round(smp.get(k, 0) / len(sample_keys), 6),
            "share_gap_pp": round(100 * (smp.get(k, 0) / len(sample_keys) - v / len(pool_keys)), 2),
        } for k, v in pop.most_common()}
    out["max_abs_share_gap_pp"] = round(
        max(abs(r["share_gap_pp"]) for t in ("product", "substrate", "product_x_substrate")
            for r in out[t].values()), 2)
    return out


fit = marginal_fit(clean_picked, no_finding)
print("\nmarginal fit vs the %d-note clean pool:" % len(no_finding))
for t in ("product", "substrate", "product_x_substrate"):
    for k, r in fit[t].items():
        print("  %-22s %-24s pool %5.2f%%  drawn %5.2f%%  gap %+5.2fpp"
              % (t, k, 100 * r["pool_share"], 100 * r["drawn_share"], r["share_gap_pp"]))
print("  worst marginal gap: %.2fpp" % fit["max_abs_share_gap_pp"])

# ---------------------------------------------------------------- step 4: emit
scored = sorted(omission_notes) + clean_picked
scored.sort()
assert_eq("scored subset size", len(scored), 261)
assert_eq("scored subset has no duplicates", len(set(scored)), 261)
assert_eq("scored subset is a subset of the roster", set(scored) <= set(roster_keys), True)

units, skipped = T.note_units()
assert_eq("note_units() unit count", len(units), len(roster))
assert_eq("note_units() keys match the roster", {u["note_key"] for u in units} == set(roster_keys),
          True)
unit_by_key = {u["note_key"]: u for u in units}

records = []
for nk in scored:
    n = roster_by_key[nk]
    u = unit_by_key[nk]
    arm = "omission" if nk in omission_notes else "no_finding"
    oms = []
    for i, t1 in verified_by_note.get(nk, []):
        if t1 != "omission":
            continue
        oms.append({
            "finding_id": i["finding_id"],
            "description": i.get("description"),
            "source_quote": i.get("source_quote"),
            "note_quote": i.get("note_quote"),
            "severity": i.get("severity"),
            "salience": i.get("salience"),
        })
    oms.sort(key=lambda d: d["finding_id"])
    if arm == "omission" and not oms:
        die("%s is in the omission arm with no verified omissions" % nk)
    if arm == "no_finding" and (oms or verified_by_note.get(nk)):
        die("%s is in the no-finding arm but carries verified findings" % nk)
    records.append({
        "note_key": nk, "scribe": n["scribe"], "product_letter": PRODUCT_LETTER[n["scribe"]],
        "source": n["source"], "id": n["id"], "consultation": n["consultation"],
        "template": n["template"], "arm": arm,
        "note_text": u["note"], "transcript": u["transcript"],
        "verified_omissions": oms,
        "n_verified_all_classes": len(verified_by_note.get(nk, [])),
    })

assert_eq("records emitted", len(records), 261)
assert_eq("omission-arm records", sum(1 for r in records if r["arm"] == "omission"), 87)
assert_eq("no-finding-arm records", sum(1 for r in records if r["arm"] == "no_finding"), 174)
assert_eq("verified omissions carried",
          sum(len(r["verified_omissions"]) for r in records), tier1_counts["omission"])
assert_eq("every note has text", all(r["note_text"] and r["transcript"] for r in records), True)

by_product = Counter(r["product_letter"] for r in records)
by_arm_product = Counter((r["product_letter"], r["arm"]) for r in records)
by_substrate = Counter(r["source"] for r in records)
consults = Counter(r["consultation"] for r in records)
per_consult_hist = Counter(consults.values())

cache_missing = sorted({(r["source"], r["id"]) for r in records
                        if not os.path.exists(os.path.join(
                            CACHE_DIR, "facts_%s__%s.json" % (r["source"], r["id"])))})

out = {
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "seed": SEED,
    "design": ("real-error scoring subset of the P2 census: all 87 notes carrying a tier-1 "
               "verified omission plus 174 notes with no verified finding, drawn from the "
               "388-note clean pool by a product x substrate proportional largest-remainder "
               "sample at seed 20260825; the 90 notes whose only verified findings are "
               "non-omission classes are excluded"),
    "sources": ["master/findings_master.json", "master/findings_verified_master.json",
                "taxonomy_common.note_units()",
                "pilot/scripts/cross_scribe_matches.py (family matcher)"],
    "method": {
        "tier1_placement": ("taxonomy_common.frame_place(issue, "
                            "family_fn=cross_scribe_matches.family)[1] - recomputed, never read "
                            "off the record's frame_tier1 field"),
        "draw": ("whole notes, strata = product x substrate, proportional to stratum size, "
                 "largest remainder, random.Random(20260825).sample over each stratum's "
                 "sorted note_keys taken in sorted stratum order - the note-level analogue of "
                 "second_panel.draw()"),
        "reproduce": ("python build_realerror_261.py - deterministic in seed 20260825 alone; "
                      "verify with note_keys_sha256 below"),
    },
    "counts": {
        "census_total_notes": len(roster),
        "census_total_consultations": len({n["consultation"] for n in roster}),
        "census_total_candidates": len(issues),
        "census_verified_findings": n_verified,
        "census_notes_with_any_verified_finding": len(verified_notes),
        "census_omission_notes": len(omission_notes),
        "census_no_finding_notes": len(no_finding),
        "census_excluded_notes": len(excluded),
        "census_tier1_distribution_of_verified": dict(sorted(tier1_counts.items())),
        "census_notes_by_product": {PRODUCT_LETTER[k]: v for k, v in
                                    sorted(Counter(n["scribe"] for n in roster).items())},
        "excluded_unmapped_only_notes": len(unmapped_only),
        "excluded_unmapped_only_by_product": {PRODUCT_LETTER[k]: v for k, v in
                                              sorted(unmapped_only_by_product.items())},
        "excluded_unmapped_only_note_keys": unmapped_only,
        "excluded_notes_with_any_unmapped_finding": excluded_with_any_unmapped,
        "unmapped_decision": ("kept EXCLUDED - ground truth here is the tier-1 placement, and a "
                              "finding the frame cannot place gives no omission ground truth "
                              "either way; disclosed rather than silent"),
        "scored_notes": len(records),
        "scored_omission_arm": 87,
        "scored_no_finding_arm": 174,
        "clean_pool_notes": len(no_finding),
        "clean_sampling_fraction": round(clean_frac, 6),
        "scored_by_product": dict(sorted(by_product.items())),
        "scored_by_product_arm": {"%s|%s" % (p, a): v for (p, a), v in sorted(by_arm_product.items())},
        "scored_by_substrate": dict(sorted(by_substrate.items())),
        "scored_verified_omissions_total": sum(len(r["verified_omissions"]) for r in records),
        "scored_verified_all_classes_total": sum(r["n_verified_all_classes"] for r in records),
        "scored_distinct_consultations": len(consults),
        "scored_notes_per_consultation_hist": {str(k): v for k, v in sorted(per_consult_hist.items())},
        "scored_consultations_with_exactly_one_note": per_consult_hist.get(1, 0),
        "extraction_cache_dir": "results/w2-pipeline/_cache",
        "extraction_cache_missing_consultations": len(cache_missing),
        "extraction_cache_missing_list": ["%s/%s" % (s, i) for s, i in cache_missing],
        "note_keys_sha256": hashlib.sha256("\n".join(scored).encode()).hexdigest(),
        "clean_draw_note_keys_sha256": hashlib.sha256("\n".join(clean_picked).encode()).hexdigest(),
    },
    "strata": {
        "definition": "product x substrate over the 388-note clean pool",
        "allocation": strata_out,
        "marginal_fit_vs_clean_pool": fit,
        "worst_marginal_gap_pp": fit["max_abs_share_gap_pp"],
    },
    "integrity": {
        "harness": ("census_note_ci.py EXPECT dict + its inline die() guards, read by AST parse "
                    "(importing that module would execute it and rewrite one of its artifacts); "
                    "tier-1 distribution from docs/FINDINGS.md section 16"),
        "expect_dict": {k: list(v) for k, v in EXPECT.items()},
        "inline_guards": INLINE_GUARDS,
        "expected_tier1_distribution": EXPECT_TIER1,
        "checks": integrity,
        "n_checks": len(integrity),
        "n_mismatches": sum(1 for c in integrity if not c["ok"]),
    },
    "notes": records,
}

json.dump(out, open(OUT, "w"), indent=1)
print("\n=== step 4: emitted ===")
print("wrote %s (%.1f MB)" % (OUT, os.path.getsize(OUT) / 1e6))
print("  261 notes: %s" % dict(sorted(by_product.items())))
print("  by arm x product: %s" % {"%s|%s" % k: v for k, v in sorted(by_arm_product.items())})
print("  by substrate: %s" % dict(sorted(by_substrate.items())))
print("  distinct consultations: %d" % len(consults))
print("  notes per consultation: %s" % {str(k): v for k, v in sorted(per_consult_hist.items())})
print("  consultations carrying exactly one scored note: %d" % per_consult_hist.get(1, 0))
print("  verified omissions carried: %d" % sum(len(r["verified_omissions"]) for r in records))
print("  consultations with no extraction cache: %d %s"
      % (len(cache_missing), ["%s/%s" % (s, i) for s, i in cache_missing][:10]))
print("  note_keys sha256: %s" % out["counts"]["note_keys_sha256"])
print("  %d integrity checks, %d mismatches"
      % (len(integrity), sum(1 for c in integrity if not c["ok"])))
