"""Build a notes_corpus-format file from the scribe_B/scribe_C scraped .txt notes, so discover.py can
run the same broad mechanism on them (cross-scribe failures).

Each scribe note file is named <source>__<id>.txt in scribe_B_notes/ or scribe_C_notes/. We look up the
transcript (+ ref_note for primock / fact_sheet for authored) by id and emit one record per note,
tagged scribe + template='audio'. Output: scribe_notes_corpus.json.

Usage: python ingest_scribe_notes.py
Then:  python discover.py --corpus scribe_notes_corpus.json --source primock,authored
"""
import json, os, glob
from common import HERE, load_primock

PRIMOCK = {r["id"]: r for r in load_primock()}
AUTHORED = {s["id"]: s for s in json.load(open(os.path.join(HERE, "authored_scenarios.json")))}


def lookup(source, cid):
    if source == "primock" and cid in PRIMOCK:
        r = PRIMOCK[cid]
        return {"transcript": r["transcript"], "ref_note": r.get("summary")}
    if source == "authored" and cid in AUTHORED:
        s = AUTHORED[cid]
        return {"transcript": s["transcript"], "fact_sheet": s["fact_sheet"]}
    return None


def main():
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
            ctx = lookup(source, cid)
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
