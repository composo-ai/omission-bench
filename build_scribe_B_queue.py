"""Build master/scribe_B_queue.json - the ordered upload queue for the scribe_B batch runs.

Target set (mirrors the scribe_C queue's sources but scribe_B-specific keeps):
  authored  : all 30 minus June-pilot captures already in scribe_B_notes/ (10) -> 20
  trapblind : the 9 KEPT scenarios (ids in master/fact_sheets_trapblind_core.json;
              tb_af_anticoag_review was dropped at critique)
  aci       : the 45 KEPT consults (ids in master/fact_sheets_aci_core.json)
  primock   : all 57 minus June-pilot captures (day1_consultation01-12) -> 45
Order: authored, trapblind, aci, primock (per the 2026-08-10 batch spec).

Note filenames follow the June convention scribe_B_notes/<source>__<id>.txt.
Done-ness is dynamic: scribe_B_overnight.py re-checks scribe_B_notes/ on every run; the
"already_captured" list here is just the build-time snapshot for the record.
dur_s comes from master/audio_manifest.json (ffprobe fallback).

Usage: python3 build_scribe_B_queue.py   (re-run any time; safe - queue is derived state)
"""
import json, os, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
NOTES = os.path.join(HERE, "scribe_B_notes")


def wav_dur(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return round(float(r.stdout.strip()), 1)
    except ValueError:
        return None


def source_ids():
    authored = [s["id"] for s in json.load(open(os.path.join(HERE, "authored_scenarios.json")))]
    tb_core = json.load(open(os.path.join(HERE, "master/fact_sheets_trapblind_core.json")))
    aci_core = json.load(open(os.path.join(HERE, "master/fact_sheets_aci_core.json")))

    def core_ids(x):
        if isinstance(x, dict):
            for k in ("sheets", "entries", "fact_sheets", "items"):
                if k in x:
                    x = x[k]
                    break
        if isinstance(x, dict):
            return sorted(x.keys())
        return sorted({e.get("id") or e.get("consult_id") for e in x})

    # keep original scenario order for trapblind, filtered to the kept set
    tb_keep = set(core_ids(tb_core))
    trapblind = [s["id"] for s in json.load(open(os.path.join(
        HERE, "master/trapblind_scenarios_critiqued.json"))) if s["id"] in tb_keep]
    primock = [s["id"] for s in json.load(open(os.path.join(
        HERE, "../../experiments/ai_scribe/dataset/primock57_parsed.json")))]
    aci_keep = set(core_ids(aci_core))
    aci = [s["id"] for s in json.load(open(os.path.join(HERE, "master/aci_subsample.json")))
           if s["id"] in aci_keep]
    return {"authored": authored, "trapblind": trapblind, "aci": aci, "primock": primock}


def main():
    ids = source_ids()
    man = {f["id"]: f["dur_s"] for f in json.load(
        open(os.path.join(HERE, "master/audio_manifest.json")))["files"]}
    captured = {f for f in os.listdir(NOTES) if f.endswith(".txt")
                and os.path.getsize(os.path.join(NOTES, f)) > 150}
    is_done = lambda src, cid: f"{src}__{cid}.txt" in captured

    entries, seq = [], 1
    for src in ("authored", "trapblind", "aci", "primock"):
        for cid in ids[src]:
            if is_done(src, cid):
                continue
            wav = os.path.join("audio", f"{cid}.wav")
            entries.append({"seq": seq, "id": cid, "source": src, "variant": None,
                            "wav": wav,
                            "note": os.path.join("scribe_B_notes", f"{src}__{cid}.txt"),
                            "dur_s": man.get(cid) or wav_dur(os.path.join(HERE, wav))})
            seq += 1

    per_source = {}
    for src, lst in ids.items():
        done = sum(1 for c in lst if is_done(src, c))
        per_source[src] = {"kept": len(lst), "already_captured": done, "queued": len(lst) - done}
    q = {"built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "target": ("119 consults at build time (20 authored + 9 trapblind kept + 45 aci kept "
                    "+ 45 primock), i.e. sources minus June-pilot scribe_B_notes/ captures"),
         "per_source": per_source,
         "already_captured": sorted(captured),
         "entries": entries}
    outp = os.path.join(HERE, "master", "scribe_B_queue.json")
    json.dump(q, open(outp, "w"), indent=1)
    n_missing = sum(1 for e in entries if not os.path.exists(os.path.join(HERE, e["wav"])))
    total_h = sum(e["dur_s"] or 0 for e in entries) / 3600
    print(f"queue: {len(entries)} entries")
    print("per-source: " + ", ".join(f"{s}:{v['queued']}/{v['kept']}" for s, v in per_source.items()))
    print(f"wav missing on disk: {n_missing}")
    print(f"audio time queued: {total_h:.1f} h (uploads run faster than real time)")
    print(f"-> {outp}")


if __name__ == "__main__":
    main()
