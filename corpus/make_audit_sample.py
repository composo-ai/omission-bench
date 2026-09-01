"""The seeded draw for the human audit of the critiqued fact sheets.

The pre-registered acceptance check on the extracted fact sheets: a human auditor reads 15
sheets - 5 per source (primock/aci/trapblind), seeded draw 20260728 - checking every
must_contain item's evidence quote against the transcript. This script only PRODUCES the
sample list; the audit itself is done by hand.

`authored_extracted` is drawn too, on the same rule, and flagged
`beyond_pre_registered_sample: true`: the check as pre-registered names three sources only, so
those 5 are an OPTIONAL fourth draw. Auditing them is what would let the blind-extraction
recovery figure rest on a human-checked reference rather than a model-checked one; skipping
them leaves the check exactly as pre-registered. The optional draw is flagged rather than
silently folded in.

Deterministic: per source, ids of the kept (post-critique) sheets are sorted, then drawn with an
independent random.Random(20260728) so each source's draw is reproducible on its own.

Usage: python corpus/make_audit_sample.py
Output: master/human_audit_sample.json
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
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
              "gate": "acceptance rule: >=95% of audited must_contain items supported by "
                      "their evidence quote (Wilson CI reported)",
              "pre_registered_sources": list(SOURCES),
              "additional_sources": {s: "beyond the pre-registered 15; an optional extra draw"
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
            "beyond_pre_registered_sample": src in EXTRA}
        print(f"{src:20} {len(drawn)} of {len(ids)} kept | "
              f"{sample['counts'][src]['must_contain_items_to_check']} must_contain items to check"
              + ("  (BEYOND the pre-registered sample)" if src in EXTRA else ""))
        print(f"  {drawn}")
    sample["total_must_contain_items_pre_registered_15"] = sum(
        sample["counts"][s]["must_contain_items_to_check"] for s in SOURCES)
    json.dump(sample, open(OUT, "w"), indent=1)
    print(f"-> {os.path.relpath(OUT, HERE)}")


if __name__ == "__main__":
    main()
