"""Build a notes_corpus-format file from the scribe_B/scribe_C scraped .txt notes, so discover.py can
run the same broad mechanism on them (cross-scribe failures).

Each scribe note file is named <source>__<id>.txt in scribe_B_notes/ or scribe_C_notes/. We look up the
transcript (+ ref_note for primock / fact_sheet for authored) by id and emit one record per note,
tagged scribe + template='audio'. Output: scribe_notes_corpus.json.

Usage: python capture/ingest_scribe_notes.py
Then:  python census/discover.py --corpus scribe_notes_corpus.json --source primock,authored
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import json, os, glob
from common import HERE, load_primock

def corpora():
    """The two transcript corpora this script looks a captured note up against.

    Neither is carried here. PriMock57's parsed transcripts come from PriMock57's own
    release (see PRIMOCK_PARSED in common.py); the authored consultations are released as
    the dataset repository's transcripts/authored/ and fact_sheets/.
    """
    authored_file = os.path.join(HERE, "authored_scenarios.json")
    if not os.path.exists(authored_file):
        raise SystemExit(
            "cannot find %s - the authored consultations are released as the dataset "
            "repository's transcripts/authored/ and fact_sheets/" % authored_file)
    with open(authored_file, encoding="utf-8") as fh:
        return ({r["id"]: r for r in load_primock()},
                {s["id"]: s for s in json.load(fh)})


def lookup(source, cid, primock, authored):
    if source == "primock" and cid in primock:
        r = primock[cid]
        return {"transcript": r["transcript"], "ref_note": r.get("summary")}
    if source == "authored" and cid in authored:
        s = authored[cid]
        return {"transcript": s["transcript"], "fact_sheet": s["fact_sheet"]}
    return None


def main():
    primock, authored = corpora()
    out = []
    for scribe, d in (("scribe_B", "scribe_B_notes"), ("scribe_C", "scribe_C_notes")):
        for fp in glob.glob(os.path.join(HERE, d, "*.txt")):
            base = os.path.basename(fp)[:-4]
            if "__" not in base:
                continue
            source, cid = base.split("__", 1)
            note = open(fp).read().strip()
            if len(note) < 120:
                continue
            ctx = lookup(source, cid, primock, authored)
            if not ctx:
                print(f"  [skip] no transcript for {source}/{cid}")
                continue
            rec = {"source": source, "id": cid, "scribe": scribe, "template": "audio",
                   "note": note, **ctx}
            out.append(rec)
    json.dump(out, open(os.path.join(HERE, "scribe_notes_corpus.json"), "w"), indent=1)
    from collections import Counter
    print(f"ingested {len(out)} scribe notes: {dict(Counter((r['scribe']) for r in out))}")
    print("saved -> scribe_notes_corpus.json")


if __name__ == "__main__":
    main()
