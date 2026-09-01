"""ACI-Bench TTS adapter - mirrors tts_consult.py (same voices-per-turn + ffmpeg-concat
pattern, same outputs: mp3 for scribe_B upload + 16kHz mono wav for scribe_C playback), adapted
to the ACI subsample's schema, which tts_consult.py cannot read directly:

- speaker tags are [doctor]/[patient]/[patient_guest] (no [relative]); tts_consult.py's
  TURN_RE would silently absorb [patient_guest] turns into the preceding speaker's text
  and speak the literal tag aloud - hence this adapter, not a --from flag.
- transcripts come from master/aci_subsample.json (keys: id, transcript, ref_note,
  source_dataset, split).
- delivery instruction says US clinic visit, not UK GP (ACI-Bench is US primary
  care/specialty). Everything else - model, voices, gap length, MAXCHARS turn splitting,
  output formats - is kept identical to tts_consult.py for instrument consistency.

ACI transcript quirks kept VERBATIM (data fidelity - production scribes face the same):
ASR-style token spacing (" , ", " ?"), lowercase, and the doctor's "hey dragon"
dictation asides.

Usage:
  python audio/tts_aci_adapter.py <aci_id> [--from master/aci_subsample.json] [--out-dir audio-for-scribe_B]
  python audio/tts_aci_adapter.py --smoke
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import json, os, re, subprocess, sys, tempfile
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repository root
load_dotenv(os.path.join(HERE, "secrets.env"))
from openai import OpenAI

if "OPENAI_API_KEY" not in os.environ:
    raise SystemExit("OPENAI_API_KEY is not set. The text-to-speech scripts read it "
                     "from the environment - see the README's Audio section.")
CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
TTS_MODEL = "gpt-4o-mini-tts"
VOICES = {"doctor": "onyx", "patient": "nova", "patient_guest": "shimmer"}
INSTRUCTIONS = ("Speak naturally and conversationally, as in a real US clinic visit - unhurried, "
                "clear, with natural pacing. Not theatrical.")
TURN_RE = re.compile(r"\[(doctor|patient|patient_guest)\]\s*(.*?)(?=\n\[(?:doctor|patient|patient_guest)\]|\Z)", re.S)
MAXCHARS = 3800  # stay under the TTS per-request limit; split long turns


def parse_turns(transcript):
    turns = [(spk, txt.strip().replace("\n", " ")) for spk, txt in TURN_RE.findall(transcript) if txt.strip()]
    out = []
    for spk, txt in turns:
        while len(txt) > MAXCHARS:
            cut = txt.rfind(". ", 0, MAXCHARS) + 1 or MAXCHARS
            out.append((spk, txt[:cut].strip())); txt = txt[cut:].strip()
        out.append((spk, txt))
    return out


def synth_turn(spk, text, path):
    voice = VOICES.get(spk, "echo")
    with CLIENT.audio.speech.with_streaming_response.create(
            model=TTS_MODEL, voice=voice, input=text, instructions=INSTRUCTIONS,
            response_format="mp3") as resp:
        resp.stream_to_file(path)


def make_silence(path, secs=0.4):
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                    f"anullsrc=r=24000:cl=mono", "-t", str(secs), path],
                   check=True, capture_output=True)


def build(turns, out_mp3, out_wav):
    with tempfile.TemporaryDirectory() as td:
        sil = os.path.join(td, "sil.mp3"); make_silence(sil)
        parts = []
        for i, (spk, text) in enumerate(turns):
            p = os.path.join(td, f"t{i:03d}.mp3")
            synth_turn(spk, text, p)
            parts.append(p); parts.append(sil)
            print(f"  [{i+1}/{len(turns)}] {spk:13} {len(text):4d} chars", flush=True)
        listf = os.path.join(td, "list.txt")
        open(listf, "w").write("\n".join(f"file '{p}'" for p in parts))
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listf,
                        "-c", "copy", out_mp3], check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", out_mp3, "-ar", "16000", "-ac", "1", out_wav],
                       check=True, capture_output=True)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", out_wav], capture_output=True, text=True).stdout.strip()
    print(f"  -> {out_mp3}  +  {out_wav}  ({float(dur):.0f}s)")


def main():
    if "--smoke" in sys.argv:
        turns = [("doctor", "hi , ms. brooks . i'm dr. baker . how are you today ?"),
                 ("patient_guest", "she's been a bit under the weather , doctor , i drove her in .")]
        build(turns, os.path.join(HERE, "tts_aci_smoke.mp3"), os.path.join(HERE, "tts_aci_smoke.wav"))
        return
    sid = sys.argv[1]
    src = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else "master/aci_subsample.json"
    outdir = sys.argv[sys.argv.index("--out-dir") + 1] if "--out-dir" in sys.argv else "audio-for-scribe_B"
    scns = json.load(open(os.path.join(HERE, src)))
    scn = next(s for s in scns if s["id"] == sid)
    turns = parse_turns(scn["transcript"])
    print(f"{sid}: {len(turns)} turns")
    os.makedirs(os.path.join(HERE, outdir), exist_ok=True)
    os.makedirs(os.path.join(HERE, "audio"), exist_ok=True)
    build(turns, os.path.join(HERE, outdir, f"{sid}.mp3"), os.path.join(HERE, "audio", f"{sid}.wav"))


if __name__ == "__main__":
    main()
