"""TTS batch driver for the scribe_C capture gap - synthesises every consult that has no
real or June-pilot audio: the 20 authored scenarios beyond June's 10, the 10 trap-blind,
and the 48 ACI-Bench subsample consults (78 total). PriMock is NEVER synthesised - it
has real recorded audio (fetch_audio.sh).

Per consult it shells out to the pinned single-consult instruments:
  authored/trapblind -> tts_consult.py  (voices onyx/nova/shimmer, 0.4s gaps)
  aci                -> tts_aci_adapter.py  (same pattern; handles [patient_guest])

Resume-safe: a consult whose audio/<id>.wav already exists with duration >30s is
skipped, so re-running after a crash or partial batch only does the remainder
(tts_consult.py writes the wav last, so a crashed consult leaves no wav behind).
Parallel: N workers (default 6), one retry per failed consult, 20-min per-consult
timeout. Writes master/tts_batch_report.json (per-consult status, chars, duration,
cost estimate at published gpt-4o-mini-tts rates).

Usage: python3 audio/tts_batch.py [--workers 6] [--limit N] [--sources authored,trapblind,aci]
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repository root
PY = sys.executable      # the interpreter this driver is running under
AUDIO = os.path.join(HERE, "audio")

SOURCES = {  # source -> (script, json path relative to HERE)
    "authored": ("tts_consult.py", "authored_scenarios.json"),
    "trapblind": ("tts_consult.py", "master/trapblind_scenarios_critiqued.json"),
    "aci": ("tts_aci_adapter.py", "master/aci_subsample.json"),
}
# published gpt-4o-mini-tts rates (openai.com/api/pricing): ~$0.60/1M input text tokens
# (~4 chars/token) + ~$12/1M audio output tokens ~= OpenAI's own "~$0.015 per minute" figure.
COST_PER_MIN = 0.015
COST_PER_MCHAR_TEXT = 0.60 / 4  # $/1M chars


def wav_ok(path, min_s=30):
    if not os.path.exists(path):
        return False
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return float(r.stdout.strip()) > min_s
    except ValueError:
        return False


def wav_dur(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def build_jobs(wanted_sources):
    jobs = []
    for src_name, (script, jpath) in SOURCES.items():
        if src_name not in wanted_sources:
            continue
        for s in json.load(open(os.path.join(HERE, jpath))):
            wav = os.path.join(AUDIO, f"{s['id']}.wav")
            if wav_ok(wav):
                continue  # June pilot / already synthesised
            jobs.append({"id": s["id"], "source": src_name, "script": script,
                         "from": jpath, "chars": len(s["transcript"])})
    return jobs


def run_job(job):
    t0 = time.time()
    cmd = [PY, os.path.join(AUDIO, job["script"]), job["id"], "--from", job["from"]]
    for attempt in (1, 2):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=HERE)
        except subprocess.TimeoutExpired:
            job["status"] = "timeout"; continue
        wav = os.path.join(AUDIO, f"{job['id']}.wav")
        if r.returncode == 0 and wav_ok(wav):
            job["status"] = "ok"; job["dur_s"] = round(wav_dur(wav), 1)
            job["attempts"] = attempt; job["wall_s"] = round(time.time() - t0)
            return job
        job["status"] = f"fail(rc={r.returncode})"
        job["stderr_tail"] = (r.stderr or "")[-300:]
    job["attempts"] = 2; job["wall_s"] = round(time.time() - t0)
    return job


def main():
    argv = sys.argv[1:]
    workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 6
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    wanted = (argv[argv.index("--sources") + 1].split(",") if "--sources" in argv
              else list(SOURCES))
    jobs = build_jobs(wanted)
    if limit:
        jobs = jobs[:limit]
    total_chars = sum(j["chars"] for j in jobs)
    print(f"TTS batch: {len(jobs)} consults, ~{total_chars/1000:.0f}K transcript chars, "
          f"{workers} workers", flush=True)
    if not jobs:
        print("nothing to do"); return

    done, results = 0, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_job, dict(j)): j["id"] for j in jobs}
        for fut in as_completed(futs):
            r = fut.result(); done += 1; results.append(r)
            print(f"[{done}/{len(jobs)}] {r['source']:9} {r['id']:28} {r['status']:8} "
                  f"{r.get('dur_s','-'):>7}s  (wall {r.get('wall_s','?')}s)", flush=True)

    ok = [r for r in results if r["status"] == "ok"]
    fails = [r for r in results if r["status"] != "ok"]
    audio_min = sum(r["dur_s"] for r in ok) / 60
    ok_chars = sum(r["chars"] for r in ok)
    est = audio_min * COST_PER_MIN + ok_chars / 1e6 * COST_PER_MCHAR_TEXT
    report = {"run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "jobs": len(jobs), "ok": len(ok), "failed": [r["id"] for r in fails],
              "audio_minutes": round(audio_min, 1), "chars_synthesised": ok_chars,
              "est_cost_usd": round(est, 2),
              "rates": {"per_audio_min": COST_PER_MIN, "per_1M_text_chars": COST_PER_MCHAR_TEXT},
              "results": sorted(results, key=lambda r: r["id"])}
    outp = os.path.join(HERE, "master", "tts_batch_report.json")
    json.dump(report, open(outp, "w"), indent=1)
    print(f"\nTTS BATCH DONE ok={len(ok)} fail={len(fails)}  "
          f"audio={audio_min:.0f}min  est_cost=${est:.2f}  -> {outp}", flush=True)
    if fails:
        for r in fails:
            print(f"  FAILED {r['id']}: {r['status']}  {r.get('stderr_tail','')[:120]}")


if __name__ == "__main__":
    main()
