#!/usr/bin/env python3
"""W-B / W-D overlap-handling check - both sides of the medval_overlap rule, in one place.

The pre-registered handling rule (specs/w-d-master-dataset.md section 9, restated in
wb_overlap_report.json): ACI-Bench encounters that match a MedVAL dialogue2note source are TAGGED
`medval_overlap` and excluded from analyses that ALSO use MedVAL - they are NOT removed from the
benchmark. specs/w-b-external-anchors.md Amendment A1 states the same rule from the MedVAL side:
W-B's d2n PRIMARY analysis excludes ACI-derived items, with the with-overlap numbers as a
sensitivity row.

This script asserts both directions and emits the exclusion list a MedVAL co-analysis has to apply:

  (a) MedVAL side - no ACI-overlap-tagged item sits in W-B's d2n primary eval set
      (wb_medval_eval_ids.json).
  (b) Master side - which pair_ids in master/pairs_master_frozen.json come from an
      overlap-flagged ACI encounter, i.e. must be dropped from any MedVAL co-analysis.

It is READ-ONLY with respect to the frozen master file: pairs_master_frozen.json is W-D's frozen
artifact and is never rewritten here. The exclusion list is emitted as a derived artifact so a
co-analysis can apply it without the frozen file having to carry the tag.

Deterministic, stdlib-only, no LLM calls.
Output: wb_medval_coanalysis_exclusions.json (harness root, committed)
Run: python3 wb_medval_coanalysis_check.py
"""

import json
import os
import sys
from datetime import date

HARNESS = os.path.dirname(os.path.abspath(__file__))
OVERLAP = os.path.join(HARNESS, "wb_overlap_report.json")
EVAL_IDS = os.path.join(HARNESS, "wb_medval_eval_ids.json")
ITEMS = os.path.join(HARNESS, "wb_medval_items.json")
PAIRS = os.path.join(HARNESS, "master", "pairs_master_frozen.json")
OUT = os.path.join(HARNESS, "wb_medval_coanalysis_exclusions.json")


def main():
    for p in (OVERLAP, EVAL_IDS, ITEMS):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
    rep = json.load(open(OVERLAP))
    eval_ids = json.load(open(EVAL_IDS))
    items = {it["item_id"]: it for it in json.load(open(ITEMS))["items"]}
    flagged_aci = set(rep["aci_flagged_ids_all"])

    # (a) MedVAL side ------------------------------------------------------------------
    sets = eval_ids["analysis_sets"]
    medval_side = {}
    for name, s in sets.items():
        tagged = [i for i in s["ids"] if items[i]["aci_overlap"]]
        medval_side[name] = {"n": s["n"], "overlap_tagged": len(tagged),
                             "overlap_tagged_ids": tagged[:20]}

    # (b) master side ------------------------------------------------------------------
    master_side = {"available": os.path.exists(PAIRS)}
    if master_side["available"]:
        m = json.load(open(PAIRS))
        pairs = m["pairs"]
        aci_pairs = [p for p in pairs if p.get("stratum") == "aci"]
        excl = [p for p in aci_pairs if p.get("id") in flagged_aci]
        by_type = {}
        for p in excl:
            by_type[p["type"]] = by_type.get(p["type"], 0) + 1
        remaining = {}
        for p in pairs:
            if p.get("stratum") == "aci" and p.get("id") in flagged_aci:
                continue
            remaining[p["type"]] = remaining.get(p["type"], 0) + 1
        master_side.update({
            "file": "master/pairs_master_frozen.json",
            "frozen_utc": m.get("frozen_utc"),
            "pairs_total": len(pairs),
            "aci_pairs": len(aci_pairs),
            "aci_encounters_in_frozen_set": len({p["id"] for p in aci_pairs}),
            "overlap_flagged_encounters_present": sorted({p["id"] for p in excl}),
            "pairs_to_exclude_from_medval_coanalysis": len(excl),
            "excluded_by_type": by_type,
            "pair_ids_to_exclude": sorted(p["pair_id"] for p in excl),
            "remaining_pairs_by_type_after_exclusion": remaining,
            "tag_present_in_frozen_file": any("medval_overlap" in json.dumps(p) for p in aci_pairs),
        })

    checks = {
        "no_overlap_tagged_item_in_d2n_primary":
            medval_side["d2n_primary"]["overlap_tagged"] == 0,
        "d2n_sensitivity_set_retains_overlap_items_by_design":
            medval_side["d2n_sensitivity_with_overlap"]["overlap_tagged"] > 0
            or medval_side["d2n_sensitivity_with_overlap"]["n"] == 0,
        "master_exclusion_list_emitted": master_side.get("available", False),
    }
    failed = [k for k, v in checks.items() if not v]

    out = {
        "generated": date.today().isoformat(),
        "script": "wb_medval_coanalysis_check.py",
        "spec": ("w-d-master-dataset.md section 9 (tag medval_overlap, exclude from "
                 "MedVAL-co-analyses, do not remove from the benchmark); "
                 "w-b-external-anchors.md Amendment A1 (d2n primary excludes ACI-derived items, "
                 "with-overlap numbers as a sensitivity row)"),
        "overlap_source": {"file": "wb_overlap_report.json",
                           "generated": rep.get("generated"),
                           "aci_encounters_flagged": len(flagged_aci)},
        "medval_side": medval_side,
        "master_side": master_side,
        "validation": {"checks": checks, "failed": failed},
        "flags": [],
    }
    if master_side.get("available") and not master_side.get("tag_present_in_frozen_file"):
        out["flags"].append(
            "master/pairs_master_frozen.json carries NO medval_overlap tag on its aci-stratum "
            "pairs, although the pre-registered handling rule says flagged encounters are tagged. "
            "The frozen file is not rewritten here; this exclusion list is the substitute a "
            "MedVAL co-analysis must apply. Worth a W-D decision on whether the tag should be "
            "added at the next freeze.")
    if medval_side["d2n_primary"]["n"] == 0:
        ingested = json.load(open(ITEMS)).get("items", [])
        releases = sorted({it.get("release") for it in ingested})
        full = "physionet" in releases
        out["flags"].append(
            "W-B's d2n PRIMARY eval set is EMPTY: every dialogue2note item is ACI-derived, so "
            "Amendment A1's default exclusion removes all of them. The primary MedVAL analysis as "
            "specified cannot run." + (
                " This now holds on the FULL credentialed 840-item release, not just the HF open "
                "subset - the credentialed release adds zero dialogue2note items (85 in both, the "
                "same dialogues), so there is no further data that could revive it. Needs a spec "
                "decision on W-B's primary anchor, not another download."
                if full else
                " Ingested from the HF open release only; re-check after the PhysioNet download."))

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"wrote {OUT}")
    for name, v in medval_side.items():
        print(f"  {name:32s} n={v['n']:4d}  overlap-tagged {v['overlap_tagged']}")
    if master_side.get("available"):
        print(f"  master pairs to exclude from MedVAL co-analysis: "
              f"{master_side['pairs_to_exclude_from_medval_coanalysis']} of "
              f"{master_side['pairs_total']} "
              f"(from {len(master_side['overlap_flagged_encounters_present'])} encounters)")
    for f in out["flags"]:
        print("  FLAG:", f)
    print("validation:", "PASS" if not failed else f"FAIL {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
