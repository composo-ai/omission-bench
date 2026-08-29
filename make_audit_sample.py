"""W-D gate 1 prep - the seeded human-audit draw over the critiqued extracted fact sheets.

Spec section 6 gate 1: the lead author audits 15 extracted fact sheets - 5 per source (primock/aci/trapblind),
seeded draw 20260728 - checking every must_contain item's evidence quote against the transcript.
This script only PRODUCES the sample list; the audit itself is the lead author's.

`authored_extracted` (Amendment 2026-07-30 part N) is drawn too, on the same rule, and flagged
`beyond_pre_registered_gate1: true` - gate 1 as written names three sources and the amendment did
not reopen it, so those 5 are an OPTIONAL fourth draw. Auditing them is what would let WD-R4's
recovery number rest on a human-checked reference rather than a model-checked one; skipping them
leaves gate 1 exactly as pre-registered. the lead author's call, made explicit rather than silently folded in.

Deterministic: per source, ids of the kept (post-critique) sheets are sorted, then drawn with an
independent random.Random(20260728) so each source's draw is reproducible on its own.

Usage: python make_audit_sample.py
Output: master/human_audit_sample.json
"""
import json, os, random
from common import HERE

SEED = 20260728
PER_SOURCE = 5
SOURCES = ("primock", "aci", "trapblind")
EXTRA = ("authored_extracted",)
OUT = os.path.join(HERE, "master", "human_audit_sample.json")


def main():
    sample = {"seed": SEED, "per_source": PER_SOURCE,
              "drawn_from": "master/fact_sheets_<source>.json (kept sheets after critique)",
              "gate": "spec section 6 gate 1: >=95% of audited must_contain items supported by "
                      "their evidence quote (Wilson CI reported)",
              "pre_registered_sources": list(SOURCES),
              "additional_sources": {s: "beyond the pre-registered gate-1 15 (Amendment "
                                        "2026-07-30 part N artifact); optional for the lead author"
                                     for s in EXTRA},
              "samples": {}, "counts": {}}
    for src in SOURCES + EXTRA:
        path = os.path.join(HERE, "master", f"fact_sheets_{src}.json")
        rows = json.load(open(path))
        ids = sorted(r["id"] for r in rows)
        drawn = sorted(random.Random(SEED).sample(ids, min(PER_SOURCE, len(ids))))
        sample["samples"][src] = drawn
        sample["counts"][src] = {
            "kept_sheets": len(ids), "drawn": len(drawn),
            "must_contain_items_to_check": sum(len(r["fact_sheet"]["must_contain"])
                                               for r in rows if r["id"] in set(drawn)),
            "beyond_pre_registered_gate1": src in EXTRA}
        print(f"{src:20} {len(drawn)} of {len(ids)} kept | "
              f"{sample['counts'][src]['must_contain_items_to_check']} must_contain items to check"
              + ("  (BEYOND pre-registered gate 1)" if src in EXTRA else ""))
        print(f"  {drawn}")
    sample["total_must_contain_items_pre_registered_15"] = sum(
        sample["counts"][s]["must_contain_items_to_check"] for s in SOURCES)
    json.dump(sample, open(OUT, "w"), indent=1)
    print(f"-> {os.path.relpath(OUT, HERE)}")


if __name__ == "__main__":
    main()
