"""Turn an authored scenario transcript into a two-voice consultation audio file,
so the verified-clean authored consults can be run through scribe_B (audio upload) and scribe_C (real-time
BlackHole playback), not just scribe_A (text-in).

Per turn ([doctor]/[patient]/[relative]) we synthesise with a distinct OpenAI voice, then ffmpeg-concat
with short gaps. Outputs an mp3 (scribe_B upload) + a 16kHz mono wav (scribe_C via scribe_C_play.sh / afplay).

Usage:
  python audio/tts_consult.py <scenario_id> [--from critiqued_scenarios.json] [--out-dir audio-for-scribe_B]
  python audio/tts_consult.py --smoke      # 2-line synth to validate the OpenAI call + ffmpeg concat
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
VOICES = {"doctor": "onyx", "patient": "nova", "relative": "shimmer"}
INSTRUCTIONS = ("Speak naturally and conversationally, as in a real UK GP consultation - unhurried, "
                "clear, with natural pacing. Not theatrical.")
TURN_RE = re.compile(r"\[(doctor|patient|relative)\]\s*(.*?)(?=\n\[(?:doctor|patient|relative)\]|\Z)", re.S)
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
            print(f"  [{i+1}/{len(turns)}] {spk:8} {len(text):4d} chars", flush=True)
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
        turns = [("doctor", "Morning, come and sit down. What's brought you in today?"),
                 ("patient", "It's my throat, doctor. It's been really sore since the weekend.")]
        build(turns, os.path.join(HERE, "tts_smoke.mp3"), os.path.join(HERE, "tts_smoke.wav"))
        return
    sid = sys.argv[1]
    src = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else "critiqued_scenarios.json"
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
