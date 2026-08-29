"""Build master/scribe_C_queue.json - the ordered capture queue for the scribe_C overnight runs.

Target set (W-D spec section 5 step 9 + Amendment part S): 30 authored + 10 trap-blind +
57 PriMock + 48 ACI = 145 consults, minus the 15 June-pilot captures already in
scribe_C_notes/ (10 authored + 5 PriMock day1_01-04,08) = 130 canonical captures, plus 4
speed-test variant runs (see audio/speed_test/README.md) = 134 queue entries.

Order: speed-test A/B block first (night 1 decides whether later nights run 1.15x),
then remaining authored, trap-blind, remaining PriMock, ACI.

Note filenames follow the June convention <source>__<id>.txt; variant runs append
__v1p15 / __vsilence so they never collide with the canonical note.

Done-ness is dynamic: scribe_C_overnight.sh checks scribe_C_notes/ on every run; the
"already_captured" list here is just the build-time snapshot for the record.

Usage: python3 build_scribe_C_queue.py   (re-run any time; safe - queue is derived state)
"""
import json, os, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(HERE, "audio")
NOTES = os.path.join(HERE, "scribe_C_notes")

SPEED_TEST = [  # (id, source, variant) - pairs adjacent so each A/B lands in one night
    ("chest_pain_onset", "authored", None),
    ("chest_pain_onset", "authored", "1p15"),
    ("day1_consultation05", "primock", None),
    ("day1_consultation05", "primock", "1p15"),
    ("day1_consultation06", "primock", None),
    ("day1_consultation06", "primock", "silence"),
    ("day1_consultation07", "primock", None),
    ("day1_consultation07", "primock", "silence"),
]


def wav_dur(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return round(float(r.stdout.strip()), 1)
    except ValueError:
        return None


def source_ids():
    authored = [s["id"] for s in json.load(open(os.path.join(HERE, "authored_scenarios.json")))]
    trapblind = [s["id"] for s in json.load(open(os.path.join(HERE, "master/trapblind_scenarios_critiqued.json")))]
    primock = [s["id"] for s in json.load(open(os.path.join(
        HERE, "../../experiments/ai_scribe/dataset/primock57_parsed.json")))]
    aci = [s["id"] for s in json.load(open(os.path.join(HERE, "master/aci_subsample.json")))]
    return {"authored": authored, "trapblind": trapblind, "primock": primock, "aci": aci}


def note_name(source, cid, variant):
    stem = f"{source}__{cid}" + (f"__v{variant}" if variant else "")
    return f"{stem}.txt"


def entry(seq, cid, source, variant):
    wav = (os.path.join("audio", "speed_test", f"{cid}__{variant}.wav") if variant
           else os.path.join("audio", f"{cid}.wav"))
    return {"seq": seq, "id": cid, "source": source, "variant": variant,
            "wav": wav, "note": os.path.join("scribe_C_notes", note_name(source, cid, variant)),
            "dur_s": wav_dur(os.path.join(HERE, wav))}


def main():
    ids = source_ids()
    captured = {f for f in os.listdir(NOTES) if f.endswith(".txt")
                and os.path.getsize(os.path.join(NOTES, f)) > 150}
    is_done = lambda src, cid: note_name(src, cid, None) in captured

    entries, seq = [], 1
    in_speed_block = {(cid, v) for cid, _, v in SPEED_TEST}
    for cid, src, variant in SPEED_TEST:
        entries.append(entry(seq, cid, src, variant)); seq += 1
    for src in ("authored", "trapblind", "primock", "aci"):
        for cid in ids[src]:
            if is_done(src, cid) or (cid, None) in in_speed_block:
                continue
            entries.append(entry(seq, cid, src, None)); seq += 1

    per_source = {}
    for src, lst in ids.items():
        done = sum(1 for c in lst if is_done(src, c))
        per_source[src] = {"total": len(lst), "already_captured": done, "queued": len(lst) - done}
    q = {"built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "target": "145 consults (30 authored + 10 trapblind + 57 primock + 48 aci)",
         "per_source": per_source,
         "already_captured": sorted(captured),
         "speed_test": "first 8 entries; see audio/speed_test/README.md for the A/B design",
         "entries": entries}
    outp = os.path.join(HERE, "master", "scribe_C_queue.json")
    json.dump(q, open(outp, "w"), indent=1)
    n_missing = sum(1 for e in entries if e["dur_s"] is None)
    total_h = sum(e["dur_s"] or 0 for e in entries) / 3600
    print(f"queue: {len(entries)} entries ({len(entries)-8} canonical + 8 speed-test runs, "
          f"4 of them variants)")
    print(f"per-source: " + ", ".join(f"{s}:{v['queued']}/{v['total']}" for s, v in per_source.items()))
    print(f"audio present for {len(entries)-n_missing}/{len(entries)}; missing {n_missing}")
    print(f"pure playback time queued: {total_h:.1f} h (excl. per-consult overhead)")
    print(f"-> {outp}")


if __name__ == "__main__":
    main()
