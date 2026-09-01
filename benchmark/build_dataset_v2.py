"""build_dataset_v2.py - one file holding every pair the paper evaluates on.

The assembly step of the omission pair build. Four sources, one schema, nothing regenerated:

  frozen_add_change   the 202 add/change pairs, exactly as frozen. These are the CONTROLS -
                      they establish the judges are not simply broken, and no part of the
                      factorial touches them.
  frozen_omit         the 79 frozen omission pairs. Every one was verified `truly_absent`,
                      which is what the factorial calls a COMPLETE omission, so they join the
                      complete row of the grid at their existing severity. Their construction
                      differs from the new pairs (a v1 model rewrite rather than a span
                      deletion off the site map) and every row says so in `construction`, so
                      an analysis can use the v2-only cells, the pooled cells, or both.
  partial_seed        the 34 pairs the ground-truth gate excluded for failing `truly_absent`.
                      They are partial omissions with a surviving residual, not failures, and
                      their verified residual strength puts them in the strong or weak row.
  factorial           the new severity x residual pairs, built to the allocation and verified
                      on the cross-family panel.

Every row carries: type, class, residual_level, severity (+ its source), the cell it sits in,
residual metadata where a residual exists, provenance, and `dataset_version`.

Usage:
  python3 benchmark/build_dataset_v2.py                 # writes master/dataset_v2.json
  python3 benchmark/build_dataset_v2.py --include-unverified
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import argparse, json, os, sys
from collections import Counter, defaultdict

from common import HERE
import omission_sites as OS

MASTER = OS.MASTER
OUT_FILE = os.path.join(MASTER, "dataset_v2.json")
FROZEN = os.path.join(MASTER, "pairs_master_frozen.json")
SEED_SET = os.path.join(MASTER, "partial_omission_seed_set.json")
PAIRS_V2 = os.path.join(MASTER, "omission_pairs_v2.json")
VERIFY_V2 = os.path.join(MASTER, "omission_verification_v2.json")
VERIFY_SEED = os.path.join(MASTER, "omission_verification_v2_seed_set.json")
ALLOC = os.path.join(MASTER, "factorial_allocation.json")
GEPA_EXCL = os.path.join(HERE, "gepa", "eval_exclusions.json")

DATASET_VERSION = "factorial-v1"
STRONG = ("explicit", "paraphrase")
LEVELS = ("complete", "partial-strong", "partial-weak")
SEVERITIES = ("critical", "supporting", "peripheral")


def cell(level, severity):
    return f"{level}|{severity}" if level and severity else None


def from_frozen():
    fz = json.load(open(FROZEN))
    out = []
    for p in fz["pairs"]:
        omit = p["type"] == "omit"
        lvl = "complete" if omit else None
        out.append({
            "pair_id": p["pair_id"], "source": "frozen_omit" if omit else "frozen_add_change",
            "type": p["type"], "class": "omit-complete" if omit else p["type"],
            "residual_level": lvl, "severity": p.get("severity"),
            "severity_source": p.get("severity_source"),
            "severity_rationale": p.get("severity_rationale"),
            "cell": cell(lvl, p.get("severity")),
            "stratum": p["stratum"], "id": p["id"], "key": f"{p['stratum']}|{p['id']}",
            "split": "eval", "clean": p["clean"], "errored": p["errored"],
            "change": p.get("change"), "what": p.get("what"), "fact": p.get("what"),
            "target": p.get("target"), "residual": None,
            "verified": bool(p.get("verified")),
            "verification": {"instrument": "w_a_master (frozen 281)",
                             "truly_absent": True if omit else None},
            "construction": {"method": "v1_model_rewrite",
                             "note": "preserved verbatim from the frozen artefact"},
            "provenance": p.get("provenance"), "dataset_version": DATASET_VERSION})
    return out


def from_seed_set():
    if not os.path.exists(SEED_SET):
        return []
    doc = json.load(open(SEED_SET))
    ver = {}
    if os.path.exists(VERIFY_SEED):
        ver = {r["pair_id"]: r for r in json.load(open(VERIFY_SEED))["pairs"]}
    out = []
    for p in doc["pairs"]:
        res = p.get("residual") or {}
        strongest = res.get("strongest") or res.get("max_strength")
        lvl = ("partial-strong" if strongest in STRONG else
               "partial-weak" if strongest == "partial" else None)
        v = ver.get(p["pair_id"], {})
        # these 34 predate the site map, so the algorithmic gate designed for site-map
        # construction is deliberately off for them (verify_omissions_v2 algo_checks). Their
        # receipt is the class-appropriate PARTIAL_CHECK: all four fields true.
        rvv = ((p.get("residual_verification") or {}).get("sem_verdict")
               or (v.get("sem_verdict") or {}))
        ok = bool(rvv) and all(rvv.get(f) is True for f in
                               ("primary_site_removed", "residual_present",
                                "no_collateral_loss", "reads_naturally"))
        out.append({
            "pair_id": p["pair_id"], "source": "partial_seed", "type": "omit",
            "class": "omit-partial", "residual_level": lvl,
            "severity": p.get("severity"), "severity_source": p.get("severity_source"),
            "severity_rationale": p.get("severity_rationale"),
            "cell": cell(lvl, p.get("severity")),
            "stratum": p["stratum"], "id": p["id"], "key": p.get("key"),
            "split": p.get("split", "eval"), "clean": p["clean"], "errored": p["errored"],
            "change": p.get("change"), "what": p.get("what"), "fact": p.get("fact"),
            "target": p.get("target"), "residual": res,
            "verified": ok,
            "verification": {"instrument": "w_a_master receipts + verify_omissions_v2 "
                                           "--seed-set PARTIAL_CHECK pass",
                             "sem_verdict": rvv,
                             "algo_gated": False,
                             "residual_verified": v.get("residual_verified")
                             or p.get("residual_verification")},
            "construction": {"method": "v1_model_rewrite",
                             "note": "built as a v1 single-span omission; relabelled partial "
                                     "after the corrected truly_absent check found a residual"},
            "provenance": p.get("provenance"), "dataset_version": DATASET_VERSION})
    return out


def from_factorial(include_unverified=False):
    if not os.path.exists(PAIRS_V2):
        return [], {}
    pairs = json.load(open(PAIRS_V2))["pairs"]
    ver = {}
    if os.path.exists(VERIFY_V2):
        ver = {r["pair_id"]: r for r in json.load(open(VERIFY_V2))["pairs"]}
    out, dropped = [], {}
    for p in pairs:
        v = ver.get(p["pair_id"])
        ok = bool(v and v["verified"])
        if not ok and not include_unverified:
            dropped[p["pair_id"]] = (v or {}).get("verified_reason") or "not verified"
            continue
        rv = (v or {}).get("residual_verified") or {}
        res = dict(p.get("residual") or {})
        if rv:
            res["verified_strongest"] = rv.get("strongest")
            res["verified_n"] = rv.get("n_verified")
            res["verified_sites"] = rv.get("sites")
        out.append({
            "pair_id": p["pair_id"], "source": "factorial", "type": "omit",
            "class": p["class"], "residual_level": p.get("residual_level"),
            "severity": p.get("severity"), "severity_source": p.get("severity_source"),
            "severity_flagged": p.get("severity_flagged"),
            "severity_arms": p.get("severity_arms"),
            "severity_rationale": p.get("severity_rationale"),
            "cell": p.get("cell") or cell(p.get("residual_level"), p.get("severity")),
            "stratum": p["stratum"], "id": p["id"], "key": p["key"], "split": p["split"],
            "clean": p["clean"], "errored": p["errored"], "change": p.get("change"),
            "what": p.get("what"), "fact": p.get("fact"), "target": p.get("target"),
            "fact_uid": p.get("fact_uid"), "fact_kind": p.get("fact_kind"),
            "bucket": p.get("bucket"),
            "n_sites_total": p.get("n_sites_total"),
            "removed_site_ids": p.get("removed_site_ids"),
            "kept_site_ids": p.get("kept_site_ids"),
            "primary_site_id": p.get("primary_site_id"),
            "matched_pair_ids": p.get("matched_pair_ids"),
            "residual": res, "verified": ok,
            "verification": {"instrument": "verify_omissions_v2 cross-family panel",
                             "sem_verdict": (v or {}).get("sem_verdict"),
                             "unanimous": (v or {}).get("unanimous"),
                             "cross_family_split": (v or {}).get("cross_family_split"),
                             "algo_pass": ((v or {}).get("algo") or {}).get("pass"),
                             "sem_fails": (v or {}).get("sem_fails")},
            "construction": p.get("construction"), "provenance": p.get("provenance"),
            "dataset_version": DATASET_VERSION})
    return out, dropped


def trained_pair_ids():
    """The pairs the prompt optimizer trained on - the authoritative list, from its own record.

    A pair the optimizer trained on cannot sit in the evaluation set: it is contaminated for
    the gepa-optimized arm, and dropping it for that arm ALONE would leave the arms scoring
    different item sets, which is worse than the contamination. So these are held out of the
    evaluation set globally and every arm sees the same items. They stay in the file, tagged,
    because they are still real verified pairs and the optimizer needs them.
    """
    if not os.path.exists(GEPA_EXCL):
        return set(), None
    d = json.load(open(GEPA_EXCL))
    ids = (set(d.get("trained_pair_ids") or [])
           | set(d.get("trained_partial_pair_ids") or [])
           | set(d.get("dev_pool_pair_ids_v2") or []))   # dev-pool members are off-limits too
    keys = {f"{st}|{cid}" for st, lst in (d.get("consultation_ids") or {}).items()
            for cid in lst}
    return ids, keys, d.get("generated_utc")


def mark_eval_set(rows, trained):
    """Stamp every row with `eval_set`, so a consumer cannot accidentally evaluate on a
    held-out pair by reading `pairs` and ignoring `split`."""
    for r in rows:
        why = None
        if r["pair_id"] in trained:
            why = "trained_by_prompt_optimizer"
        elif r.get("split") == "gepa_dev":
            why = "gepa_dev_split"
        r["eval_set"] = why is None
        r["held_out_reason"] = why
    return rows


def summarise(rows):
    ev = [r for r in rows if r.get("eval_set")]
    om = [r for r in ev if r["type"] == "omit"]
    grid = {f"{l}|{g}": 0 for l in LEVELS for g in SEVERITIES}
    grid_v2 = dict(grid)
    for r in om:
        if r.get("cell") in grid:
            grid[r["cell"]] += 1
            if r["source"] == "factorial":
                grid_v2[r["cell"]] += 1
    return {
        "pairs_total": len(rows), "pairs_eval_set": len(ev),
        "pairs_held_out": len(rows) - len(ev),
        "held_out_reasons": dict(Counter(r.get("held_out_reason") for r in rows
                                         if not r.get("eval_set"))),
        "by_source": dict(Counter(r["source"] for r in rows)),
        "by_type": dict(Counter(r["type"] for r in rows)),
        "by_class": dict(Counter(r["class"] for r in rows)),
        "by_severity": dict(Counter(r.get("severity") for r in rows)),
        "by_severity_source": dict(Counter(r.get("severity_source") for r in rows)),
        "by_stratum": dict(Counter(r["stratum"] for r in rows)),
        "by_split": dict(Counter(r.get("split") for r in rows)),
        "omission_grid_all_sources": grid,
        "omission_grid_factorial_only": grid_v2,
        "omission_by_residual_level": dict(Counter(r.get("residual_level") for r in om)),
        "severity_by_type": {t: dict(Counter(r.get("severity") for r in ev
                                             if r["type"] == t))
                             for t in sorted({r["type"] for r in ev})},
        "consultations": len({r.get("key") for r in rows}),
        "matched_sets": sum(1 for r in rows if r.get("matched_pair_ids")),
    }


DEV_PAIRS = os.path.join(MASTER, "omission_pairs_dev.json")
DEV_VERIFY = os.path.join(MASTER, "omission_verification_dev.json")
DEV_OUT = os.path.join(MASTER, "gepa_dev_factorial.json")


def write_dev_pool():
    """The densified GEPA dev pairs, in the shape gepa_data.load_examples wants.

    Built ONLY on the 22 gepa_dev consultations, which are already burned for evaluation, so
    this costs the eval set no power at all. Verification here is training grade by design
    (gepa/V2-DESIGN.md): one seed for the complete class, a cross-family pair of calls for
    omit-partial, because omit-partial is the class the paper's claim rests on and the class
    v1's pool had five of.
    """
    pairs = json.load(open(DEV_PAIRS))["pairs"]
    ver = {r["pair_id"]: r for r in json.load(open(DEV_VERIFY))["pairs"]}
    out = []
    for p in pairs:
        v = ver.get(p["pair_id"])
        if not (v and v["verified"]):
            continue
        rv = (v.get("residual_verified") or {})
        out.append({
            "pair_id": p["pair_id"], "id": p["id"], "stratum": p["stratum"],
            "key": p["key"], "split": "gepa_dev", "type": "omit", "class": p["class"],
            "residual_level": p.get("residual_level"), "cell": p.get("cell"),
            "severity": p.get("severity"), "severity_source": p.get("severity_source"),
            "severity_rationale": p.get("severity_rationale"),
            "severity_flagged": p.get("severity_flagged"),
            "clean": p["clean"], "errored": p["errored"], "change": p.get("change"),
            "what": p.get("what"), "fact": p.get("fact"), "fact_uid": p.get("fact_uid"),
            "residual": {**(p.get("residual") or {}),
                         "verified_strongest": rv.get("strongest"),
                         "verified_n": rv.get("n_verified"),
                         "verified_sites": rv.get("sites")},
            "verification": {"instrument": "verify_omissions_v2 (training grade)",
                             "panel": v.get("panel"), "n_calls": v.get("n_seeds"),
                             "sem_verdict": v.get("sem_verdict"),
                             "unanimous": v.get("unanimous")},
            "construction": p.get("construction"), "provenance": p.get("provenance"),
            "dataset_version": DATASET_VERSION})
    doc = {"generated_utc": OS.now_utc(), "spec": "gepa/V2-DESIGN.md",
           "dataset_version": DATASET_VERSION,
           "what": "densified GEPA dev pairs on the 22 gepa_dev consultations - the "
                   "severity x residual factorial applied to the training split, so the "
                   "optimizer sees omit-partial in quantity instead of five times. Never "
                   "part of any evaluation set.",
           "split": "gepa_dev",
           "counts": {"pairs": len(out),
                      "by_class": dict(Counter(r["class"] for r in out)),
                      "by_cell": dict(Counter(r["cell"] for r in out)),
                      "by_severity": dict(Counter(r["severity"] for r in out)),
                      "by_residual_strength": dict(Counter(
                          (r["residual"] or {}).get("verified_strongest") for r in out
                          if r["class"] == "omit-partial")),
                      "consultations": len({r["key"] for r in out})},
           "pairs": out}
    tmp = DEV_OUT + ".tmp"
    json.dump(doc, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, DEV_OUT)
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dev-pool", action="store_true",
                    help="write master/gepa_dev_factorial.json instead (the optimizer's training pool)")
    ap.add_argument("--include-unverified", action="store_true",
                    help="keep factorial pairs that failed verification (tagged)")
    args = ap.parse_args()
    if args.dev_pool:
        d = write_dev_pool()
        c = d["counts"]
        print(f"master/gepa_dev_factorial.json - {c['pairs']} verified dev pairs over "
              f"{c['consultations']} consultations")
        print(f"  classes {c['by_class']} | severity {c['by_severity']}")
        print(f"  cells {c['by_cell']}")
        print(f"  partial residual strength {c['by_residual_strength']}")
        return 0

    rows = from_frozen() + from_seed_set()
    fac, dropped = from_factorial(args.include_unverified)
    rows += fac
    rows.sort(key=lambda r: (r["source"], r["pair_id"]))
    trained, trained_keys, excl_utc = trained_pair_ids()
    mark_eval_set(rows, trained)
    eval_ids = {r["pair_id"] for r in rows if r["eval_set"]}
    leak = sorted(eval_ids & trained)
    assert not leak, f"CONTAMINATION: {len(leak)} optimizer-trained pairs in the eval set: {leak[:5]}"
    # the optimizer also SAW the clean notes of its 22 training consultations, so no eval-set
    # pair may come from one of them either - a different edit on a memorised note is still a
    # note the gepa arm has seen
    key_leak = sorted({r["key"] for r in rows if r["eval_set"] and r.get("key") in trained_keys})
    assert not key_leak, f"CONTAMINATION: eval-set pairs from optimizer training consultations: {key_leak[:5]}"

    alloc = json.load(open(ALLOC)) if os.path.exists(ALLOC) else {}
    out = {
        "generated_utc": OS.now_utc(), "spec": OS.SPEC, "dataset_version": DATASET_VERSION,
        "what": "every pair the paper evaluates on: the frozen add/change controls, the "
                "frozen complete omissions, the relabelled partial seed set, and the "
                "severity x residual factorial. One schema, nothing regenerated.",
        "sources": {
            "frozen_add_change": "master/pairs_master_frozen.json (add + change)",
            "frozen_omit": "master/pairs_master_frozen.json (omit; complete by construction)",
            "partial_seed": "master/partial_omission_seed_set.json",
            "factorial": "master/omission_pairs_v2.json + omission_verification_v2.json"},
        "design": {
            "residual_levels": list(LEVELS), "severities": list(SEVERITIES),
            "residual_rule": "complete removes every mapped site of the fact; the partial "
                             "levels remove every site but ONE, chosen for its strength, so "
                             "residual strength is manipulated and residual count held at 1",
            "severity_rule": "graded per fact on the published severity rubric; traps inherit "
                             "their consensus importance, must_contain facts get two "
                             "cross-family rubric arms with the conservative tie-break",
            "allocation_target_per_cell": (alloc.get("design") or {}).get("target_per_cell")},
        "eval_set_rule": "eval_set = true unless the pair is in gepa/eval_exclusions.json "
                         "(trained by the prompt optimizer) or carries split gepa_dev. Held-out "
                         "pairs stay in the file, tagged, and are excluded for EVERY arm so "
                         "all arms score an identical item set.",
        "integrity": {
            "gepa_exclusions_file": "gepa/eval_exclusions.json",
            "gepa_exclusions_generated_utc": excl_utc,
            "trained_pair_ids_seen": len(trained),
            "held_out_of_eval_set": sorted(r["pair_id"] for r in rows if not r["eval_set"]),
            "eval_set_intersect_trained": len(leak),
            "trained_consultation_keys": len(trained_keys),
            "eval_set_intersect_trained_consultations": len(key_leak),
            "assertion": "the evaluation set shares no pair id and no consultation key "
                         "with the pairs the prompt optimizer trained on; both are "
                         "checked at build and the build fails rather than emitting "
                         "a contaminated set"},
        "counts": summarise(rows),
        "factorial_pairs_dropped_unverified": dropped,
        "pairs": rows,
    }
    tmp = OUT_FILE + ".tmp"
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, OUT_FILE)
    c = out["counts"]
    print(f"master/dataset_v2.json - {c['pairs_total']} pairs ({c['pairs_eval_set']} in the "
          f"eval set, {c['pairs_held_out']} held out {c['held_out_reasons']}) | "
          f"sources {c['by_source']}")
    print(f"  integrity: eval_set n trained = {out['integrity']['eval_set_intersect_trained']} "
          f"(over {out['integrity']['trained_pair_ids_seen']} trained ids)")
    print(f"  types {c['by_type']} | severity {c['by_severity']}")
    print("  omission grid (all sources / factorial only):")
    for l in LEVELS:
        print("   " + " ".join(
            f"{l}x{g}: {c['omission_grid_all_sources'][f'{l}|{g}']}"
            f"/{c['omission_grid_factorial_only'][f'{l}|{g}']}" for g in SEVERITIES))
    if dropped:
        print(f"  factorial pairs excluded as unverified: {len(dropped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
