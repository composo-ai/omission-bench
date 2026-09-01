"""Generate Scribe A's notes through its documented API, across every stratum.

June pilot instrument, unchanged: common.scribe_AClient (REST, OAuth client-creds with 300s token
auto-refresh, trailing-slash endpoints), templates short + detailed, text-transcript-in note-out
(isolates note-generation from transcription error).

Sources (each record tagged):
  - authored  : 30 contamination-free scenarios (authored_scenarios.json) - carries `fact_sheet`.
  - primock   : PriMock57 via common.load_primock - carries the clinician `ref_note`.
  - aci       : ACI-Bench kept consults - ids from master/fact_sheets_aci_core.json, transcript +
                `ref_note` (+ split) from master/aci_subsample.json.
  - trapblind : trap-blind authored kept consults - ids from master/fact_sheets_trapblind_core.json,
                transcript + presenting_complaint from master/trapblind_scenarios_critiqued.json.

Master run (builds the master corpus at master/notes_corpus_master.json):
    python scribe_A_generate.py --sources authored,primock,aci,trapblind \
        --out master/notes_corpus_master.json --carry notes_corpus.json [--limit 70]

  - Idempotent by (source, id, template): records already in --out with a note are never
    re-generated - re-run the same command to resume. --limit caps NEW API calls per invocation
    (chunking for foreground runs); records that previously errored are retried.
  - --carry notes_corpus.json: notes captured by the June run are carried into the master corpus
    VERBATIM - the instrument is unchanged, so they were not regenerated - and tagged
    run="june_run". Fresh generations are tagged run="master_run" + generated_utc.
  - 429s honour Retry-After (else exponential backoff); 5xx/transport errors back off and retry;
    other 4xx fail fast and are recorded per-record (retried on the next invocation).
  - Per-call capture log appends to master/scribe_A_master_capture_log.jsonl.

Legacy June usage (unchanged default output; the script REFUSES to overwrite an existing
notes_corpus.json to protect the June artifact - that run is complete):
    python scribe_A_generate.py [--sources authored,primock] [--templates short,detailed]
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import httpx
from common import scribe_AClient, load_primock, scribe_A_TEMPLATES, HERE


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


SOURCES = (_arg("--sources") or "authored,primock").split(",")
TEMPLATES = (_arg("--templates") or "short,detailed").split(",")
PRIMOCK_N = int(_arg("--primock-n", "0")) or None   # default: all 57
LIMIT = int(_arg("--limit")) if "--limit" in sys.argv else None  # cap on NEW API calls this run
OUT = os.path.join(HERE, _arg("--out", "notes_corpus.json"))
CARRY = _arg("--carry")                              # e.g. notes_corpus.json (June notes)
WORKERS = int(_arg("--workers", "5"))
CAPTURE_LOG = os.path.join(HERE, "master", "scribe_A_master_capture_log.jsonl")

scribe_A = scribe_AClient()
_IO_LOCK = threading.Lock()


# ---------------------------------------------------------------- loaders (one per source)
def load_authored():
    rows = json.load(open(os.path.join(HERE, "authored_scenarios.json")))
    return [{"source": "authored", "id": r["id"], "transcript": r["transcript"],
             "fact_sheet": r["fact_sheet"], "presenting_complaint": r.get("presenting_complaint")}
            for r in rows]


def load_primock_rows():
    return [{"source": "primock", "id": r["id"], "transcript": r["transcript"],
             "ref_note": r.get("summary")} for r in load_primock(PRIMOCK_N)]


def load_aci():
    kept = json.load(open(os.path.join(HERE, "master", "fact_sheets_aci_core.json")))
    sub = {r["id"]: r for r in json.load(open(os.path.join(HERE, "master", "aci_subsample.json")))}
    return [{"source": "aci", "id": r["id"], "split": sub[r["id"]].get("split"),
             "transcript": sub[r["id"]]["transcript"], "ref_note": sub[r["id"]].get("ref_note")}
            for r in kept]


def load_trapblind():
    kept = json.load(open(os.path.join(HERE, "master", "fact_sheets_trapblind_core.json")))
    crit = {r["id"]: r for r in
            json.load(open(os.path.join(HERE, "master", "trapblind_scenarios_critiqued.json")))}
    return [{"source": "trapblind", "id": r["id"], "transcript": crit[r["id"]]["transcript"],
             "presenting_complaint": crit[r["id"]].get("presenting_complaint")} for r in kept]


LOADERS = {"authored": load_authored, "primock": load_primock_rows,
           "aci": load_aci, "trapblind": load_trapblind}


# ---------------------------------------------------------------- generation with backoff
def generate_with_retry(transcript, tname, max_attempts=5):
    """Returns (note, sections, meta). meta: attempts / e429 / e5xx / transport / error."""
    meta = {"attempts": 0, "e429": 0, "e5xx": 0, "transport": 0, "error": None}
    delay = 5
    for _ in range(max_attempts):
        meta["attempts"] += 1
        try:
            note, sd = scribe_A.generate(transcript, tname)
            if note.strip():
                meta["error"] = None
                return note, sd, meta
            meta["error"] = "empty note"
        except httpx.HTTPStatusError as e:
            st = e.response.status_code
            meta["error"] = f"HTTP {st}: {e.response.text[:150]}"
            if st == 429:
                meta["e429"] += 1
                ra = e.response.headers.get("Retry-After")
                time.sleep(min(int(ra), 120) if ra and ra.isdigit() else delay)
                delay = min(delay * 2, 60)
                continue
            if 500 <= st < 600:
                meta["e5xx"] += 1
            else:
                return None, None, meta          # other 4xx: fail fast, retried next invocation
        except Exception as e:
            meta["transport"] += 1
            meta["error"] = f"{type(e).__name__}: {e}"[:200]
        time.sleep(delay)
        delay = min(delay * 2, 60)
    return None, None, meta


def _log_call(entry):
    os.makedirs(os.path.dirname(CAPTURE_LOG), exist_ok=True)
    with _IO_LOCK:
        with open(CAPTURE_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")


def _save(index, jobs, path):
    """Atomic write, deterministic order: jobs order first, then any pre-existing extras."""
    job_keys = [(r["source"], r["id"], t) for r, t in jobs]
    ordered = [index[k] for k in job_keys if k in index]
    ordered += [v for k, v in index.items() if k not in set(job_keys)]
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ordered, f, indent=1)
    os.replace(tmp, path)


def main():
    base = []
    for s in SOURCES:
        base += LOADERS[s]()
    jobs = [(row, t) for row in base for t in TEMPLATES]
    per_src = {s: sum(1 for r in base if r["source"] == s) for s in SOURCES}
    print(f"scribe_A: {len(jobs)} target notes = {len(base)} consults x {len(TEMPLATES)} "
          f"templates {TEMPLATES} (consults per source: {per_src})")

    if os.path.abspath(OUT) == os.path.join(HERE, "notes_corpus.json") and os.path.exists(OUT) \
            and "--force" not in sys.argv:
        sys.exit("refusing to overwrite notes_corpus.json (June artifact) - pass --out (or --force)")

    # resume index from the output file; records with a note are done, errored ones retry
    index = {}
    if os.path.exists(OUT):
        for r in json.load(open(OUT)):
            index[(r["source"], r["id"], r["template"])] = r
    done = {k for k, r in index.items() if r.get("note")}
    print(f"resume: {len(done)} already in {os.path.relpath(OUT, HERE)}")

    # carry June-instrument notes verbatim for any target key not already done
    if CARRY:
        carried = 0
        for r in json.load(open(os.path.join(HERE, CARRY))):
            k = (r["source"], r["id"], r["template"])
            if r.get("note") and k not in done:
                rec = dict(r)
                rec["run"] = "june_run"
                index[k] = rec
                done.add(k)
                carried += 1
        if carried:
            _save(index, jobs, OUT)
        print(f"carry: {carried} June-instrument notes tagged run=june_run from {CARRY}")

    todo = [(row, t) for row, t in jobs if (row["source"], row["id"], t) not in done]
    capped = todo[:LIMIT] if LIMIT else todo
    print(f"todo: {len(todo)} API calls ({len(capped)} this invocation, workers={WORKERS})")

    def one(job):
        row, tname = job
        rec = {k: row[k] for k in
               ("source", "id", "transcript", "fact_sheet", "ref_note", "presenting_complaint",
                "split") if k in row and row[k] is not None}
        rec["scribe"], rec["template"], rec["run"] = "scribe_A", tname, "master_run"
        t0 = time.time()
        note, sd, meta = generate_with_retry(row["transcript"], tname)
        if note:
            rec["note"], rec["sections"] = note, sd
            rec["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        else:
            rec["error"] = meta["error"]
        _log_call({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "source": row["source"], "id": row["id"], "template": tname,
                   "ok": bool(note), "elapsed_s": round(time.time() - t0, 1), **meta})
        return (row["source"], row["id"], tname), rec

    n_ok = n_err = 0
    if capped:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = [ex.submit(one, j) for j in capped]
            for i, fut in enumerate(as_completed(futs), 1):
                k, rec = fut.result()
                with _IO_LOCK:
                    index[k] = rec
                n_ok += bool(rec.get("note"))
                n_err += "error" in rec
                if i % 10 == 0 or i == len(capped):
                    _save(index, jobs, OUT)
                    print(f"  [{i}/{len(capped)}] ok={n_ok} err={n_err}", flush=True)
    _save(index, jobs, OUT)

    # verified-from-file summary
    final = json.load(open(OUT))
    from collections import Counter
    counts = Counter((r["source"], r["template"], r.get("run", "june_run"))
                     for r in final if r.get("note"))
    errs = [r for r in final if not r.get("note")]
    print(f"\nfile: {os.path.relpath(OUT, HERE)} | records with note: "
          f"{sum(counts.values())} | error records: {len(errs)}")
    for (src, t, run), n in sorted(counts.items()):
        print(f"  {src:10s} {t:9s} {run:10s} {n}")
    remaining = len(todo) - len(capped) + n_err
    print(f"remaining {remaining}")


if __name__ == "__main__":
    main()
