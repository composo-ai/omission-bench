"""Silence-compressed variant builder for the scribe_C speed test (real-audio arm).

Rule (as pre-agreed for the A/B): every silence LONGER than 1.2s is collapsed to 0.6s
(0.3s kept on each side, preserving the natural decay/attack); silences <=1.2s are
untouched; speech is never touched.

Implementation: silencedetect (noise=-35dB, d=1.2) -> keep-segments between the cut
windows (silence_start+0.3, silence_end-0.3) -> atrim + concat filter_complex ->
16kHz mono wav. Two rejected routes, measured on day1_06/07: ffmpeg's silenceremove
keeps ~stop_duration+stop_silence of each gap (removed only 10s/3s vs ~49s/26s of
actual >1.2s dead air), and aselect is inert for dropping audio frames in ffmpeg
8.0.1 (aselect=0 returned the input byte-identical) - atrim+concat is exact.

Usage: python3 make_silence_variant.py <in.wav> <out.wav>   (prints before/after)
"""
import re, subprocess, sys, tempfile

MIN_SIL, KEEP_EDGE = 1.2, 0.3   # collapse silences > MIN_SIL to 2*KEEP_EDGE
NOISE = "-35dB"


def detect(path):
    r = subprocess.run(["ffmpeg", "-i", path, "-af",
                        f"silencedetect=noise={NOISE}:d={MIN_SIL}", "-f", "null", "-"],
                       capture_output=True, text=True)
    log = r.stderr
    starts = [float(m) for m in re.findall(r"silence_start: ([\d.]+)", log)]
    ends = [float(m) for m in re.findall(r"silence_end: ([\d.]+)", log)]
    return list(zip(starts, ends))


def main():
    src, dst = sys.argv[1], sys.argv[2]
    sils = detect(src)
    cuts = [(s + KEEP_EDGE, e - KEEP_EDGE) for s, e in sils if e - s > MIN_SIL]
    if not cuts:
        sys.exit(f"no silences >{MIN_SIL}s found in {src} - nothing to compress")
    # keep-segments between the cuts (0 -> cut1.start, cut1.end -> cut2.start, ..., last -> EOF)
    keeps, pos = [], 0.0
    for a, b in cuts:
        keeps.append((pos, a)); pos = b
    keeps.append((pos, None))
    parts, labels = [], []
    for i, (a, b) in enumerate(keeps):
        rng = f"start={a:.3f}" + (f":end={b:.3f}" if b is not None else "")
        parts.append(f"[0:a]atrim={rng},asetpts=PTS-STARTPTS[k{i}]")
        labels.append(f"[k{i}]")
    graph = ";".join(parts) + f";{''.join(labels)}concat=n={len(keeps)}:v=0:a=1[out]"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(graph); script = f.name
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                    "-filter_complex_script", script, "-map", "[out]",
                    "-ar", "16000", "-ac", "1", dst], check=True)
    dur = lambda p: float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
        capture_output=True, text=True).stdout.strip())
    d0, d1 = dur(src), dur(dst)
    print(f"{src}: {d0:.1f}s -> {d1:.1f}s  (-{d0-d1:.1f}s, {100*(d0-d1)/d0:.1f}%)  "
          f"[{len(cuts)} silences >{MIN_SIL}s collapsed to {2*KEEP_EDGE}s]")


if __name__ == "__main__":
    main()
