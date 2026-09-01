"""Build master/audio_manifest.json - the committed record of every capture wav
(the audio itself is not redistributed: it is regenerable from the released
transcripts via audio/tts_batch.py and tts_aci_adapter.py, so this manifest is the
durable inventory + verification artifact).

Per file: source stratum, provenance kind (real PriMock recording / OpenAI TTS /
speed-test variant), sample rate, channels, duration, size, plus a scribe_C-readiness
verdict (16kHz mono, duration >30s). Non-capture strays (smoke tests, *_48k retests)
are listed under "extras" so nothing in audio/ is unaccounted for.

Usage: python3 capture/build_audio_manifest.py   (re-run after any audio change)
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import json, os, subprocess, time

from common import PRIMOCK

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repository root
AUDIO = os.path.join(HERE, "audio")


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration:stream=sample_rate,channels", "-of", "json", path],
                       capture_output=True, text=True)
    try:
        j = json.loads(r.stdout)
        return {"dur_s": round(float(j["format"]["duration"]), 1),
                "sr": int(j["streams"][0]["sample_rate"]),
                "ch": int(j["streams"][0]["channels"])}
    except (KeyError, ValueError, IndexError, json.JSONDecodeError):
        return None


def main():
    ids = {
        "authored": [s["id"] for s in json.load(open(os.path.join(HERE, "authored_scenarios.json")))],
        "trapblind": [s["id"] for s in json.load(open(os.path.join(HERE, "master/trapblind_scenarios_critiqued.json")))],
        "primock": [s["id"] for s in json.load(open(os.path.join(
            PRIMOCK)))],
        "aci": [s["id"] for s in json.load(open(os.path.join(HERE, "master/aci_subsample.json")))],
    }
    by_id = {c: src for src, lst in ids.items() for c in lst}
    files, extras = [], []
    wavs = sorted(f for f in os.listdir(AUDIO) if f.endswith(".wav"))
    for f in wavs:
        cid = f[:-4]
        p = probe(os.path.join(AUDIO, f))
        row = {"file": f"audio/{f}", **(p or {"dur_s": None, "sr": None, "ch": None})}
        row["size_b"] = os.path.getsize(os.path.join(AUDIO, f))
        if cid in by_id:
            src = by_id[cid]
            row.update({"id": cid, "source": src,
                        "kind": "real_recording" if src == "primock" else "openai_tts",
                        "scribe_C_ready": bool(p and p["sr"] == 16000 and p["ch"] == 1 and p["dur_s"] > 30)})
            files.append(row)
        else:
            extras.append(row)
    st_dir = os.path.join(AUDIO, "speed_test")
    variants = []
    for f in sorted(os.listdir(st_dir)) if os.path.isdir(st_dir) else []:
        if not f.endswith(".wav"):
            continue
        cid, variant = f[:-4].rsplit("__", 1)
        p = probe(os.path.join(st_dir, f))
        variants.append({"file": f"audio/speed_test/{f}", "id": cid, "variant": variant,
                         "source": by_id.get(cid), "kind": "speed_test_variant",
                         **(p or {}), "size_b": os.path.getsize(os.path.join(st_dir, f)),
                         "scribe_C_ready": bool(p and p["sr"] == 16000 and p["ch"] == 1 and p["dur_s"] > 30)})

    per_source = {}
    for src, lst in ids.items():
        have = [r for r in files if r["source"] == src and r["scribe_C_ready"]]
        per_source[src] = {"target": len(lst), "scribe_C_ready_wavs": len(have),
                           "missing": sorted(set(lst) - {r["id"] for r in have}),
                           "playback_h": round(sum(r["dur_s"] for r in have) / 3600, 2)}
    man = {"built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "note": "audio/ is gitignored; this manifest is the committed inventory. "
                   "TTS via tts_batch.py (tts_consult.py / tts_aci_adapter.py); PriMock via "
                   "fetch_audio.sh (real 2-channel recordings amix-ed to mono 16kHz). "
                   "speed_test variants: capture-throughput A/B only, never dataset audio - "
                   "see audio/speed_test/README.md for the disclosure note.",
           "per_source": per_source,
           "files": files, "speed_test_variants": variants, "extras": extras}
    outp = os.path.join(HERE, "master", "audio_manifest.json")
    json.dump(man, open(outp, "w"), indent=1)
    tot = sum(v["scribe_C_ready_wavs"] for v in per_source.values())
    print(f"manifest: {tot}/145 scribe_C-ready wavs; per-source: " +
          ", ".join(f"{s}:{v['scribe_C_ready_wavs']}/{v['target']}" for s, v in per_source.items()))
    for s, v in per_source.items():
        if v["missing"]:
            print(f"  missing {s}: {v['missing']}")
    print(f"-> {outp}")


if __name__ == "__main__":
    main()
