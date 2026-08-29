"""Content-based attribution for scraped scribe_B notes (the June lesson: never trust
upload order or the visible session - match by content).

Pure-python TF-IDF cosine over word vectors: a scraped note is attributed to consult
C only if sim(note, transcript_C) >= ABS_MIN and C is the argmax over ALL 145 consult
transcripts (authored 30 + trapblind 10 + primock 57 + aci 48). Threshold calibrated
on the 22 known-good June scribe_B_notes pairs - see calibrate() (run this file directly).

June's matcher (match_scribe_B_authored.py) used Cohere embeddings at 0.55; this inline
verifier is dependency- and network-free so the overnight loop can't be stalled by an
API. ABS_MIN is set from calibration, not copied from the embedding threshold.
"""
import json, math, os, re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))

STOP = set("""a an and are as at be been but by for from had has have he her his i if in is it its
me my no nor not of on or our she so that the their them then there they this to was we were what
when which who will with would you your dr doctor patient ok okay yeah yes um uh mm hmm right just
know like really going go get got well see think thats its dont im ive youre hes shes thanks thank
bye hello hi good morning afternoon today sure course come came bit lot little quite very much any
some all can could should shall do does did done also back still now then here take taken make made
one two three four five six seven eight nine ten""".split())

_token_rx = re.compile(r"[a-z]+")


def tokens(text):
    return [w for w in _token_rx.findall(text.lower()) if len(w) > 2 and w not in STOP]


def load_transcripts():
    """id -> transcript for every consult in the study (all four sources)."""
    out = {}
    for s in json.load(open(os.path.join(HERE, "authored_scenarios.json"))):
        out[s["id"]] = s["transcript"]
    for s in json.load(open(os.path.join(HERE, "master/trapblind_scenarios_critiqued.json"))):
        out[s["id"]] = s["transcript"]
    for s in json.load(open(os.path.join(
            HERE, "../../experiments/ai_scribe/dataset/primock57_parsed.json"))):
        out[s["id"]] = s["transcript"]
    for s in json.load(open(os.path.join(HERE, "master/aci_subsample.json"))):
        out[s["id"]] = s["transcript"]
    return out


class Matcher:
    def __init__(self, abs_min=0.20):
        self.transcripts = load_transcripts()
        self.abs_min = abs_min
        docs = {cid: Counter(tokens(t)) for cid, t in self.transcripts.items()}
        n = len(docs)
        df = Counter()
        for c in docs.values():
            df.update(c.keys())
        self.idf = {w: math.log(n / (1 + k)) + 1 for w, k in df.items()}
        self.default_idf = math.log(n) + 1  # unseen-in-corpus words
        self.vecs = {cid: self._vec(c) for cid, c in docs.items()}
        self.tr_tokens = {cid: set(c.keys()) for cid, c in docs.items()}

    def _vec(self, counts):
        v = {w: f * self.idf.get(w, self.default_idf) for w, f in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    def sims(self, note_text):
        """cosine of the note against every consult transcript -> {cid: sim}"""
        nv = self._vec(Counter(tokens(note_text)))
        out = {}
        for cid, tv in self.vecs.items():
            small, big = (nv, tv) if len(nv) < len(tv) else (tv, nv)
            out[cid] = sum(x * big.get(w, 0.0) for w, x in small.items())
        return out

    def verify(self, note_text, intended_id):
        """-> dict(ok, sim, best_id, best_sim, second_sim). ok iff intended is the
        argmax over ALL consults AND sim >= abs_min."""
        s = self.sims(note_text)
        ranked = sorted(s.items(), key=lambda kv: -kv[1])
        best_id, best_sim = ranked[0]
        sim = s.get(intended_id, 0.0)
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        return {"ok": best_id == intended_id and sim >= self.abs_min,
                "sim": round(sim, 4), "best_id": best_id, "best_sim": round(best_sim, 4),
                "second_sim": round(second, 4)}

    # ---- containment gate (the one the batch runner uses) ----------------------
    # Calibrated 2026-08-10 on the 116 known-good June pairs (22 scribe_B + 94 scribe_C):
    # IDF-weighted containment of the note's terms in the transcript is length-robust
    # (PriMock transcripts are noisy with <UNSURE>/filler, which sinks cosine: cosine
    # argmax was wrong on 6/116 known-GOOD pairs; containment argmax 116/116).
    # Correct pairs: min 0.289, p5 0.348. Simulated wrong pairs: max 0.257, p95 0.205.
    # Gate: containment >= 0.28 AND intended is argmax over all 145 (tie eps 0.005).
    CONT_MIN = 0.28
    TIE_EPS = 0.005

    def containment(self, note_text, cid):
        nt = Counter(tokens(note_text))
        tot = sum(self.idf.get(w, self.default_idf) * c for w, c in nt.items())
        if not tot:
            return 0.0
        ts = self.tr_tokens[cid]
        hit = sum(self.idf.get(w, self.default_idf) * c for w, c in nt.items() if w in ts)
        return hit / tot

    def verify_containment(self, note_text, intended_id):
        """The batch gate. -> dict(ok, cont, best_id, best_cont, cosine_sim)."""
        scores = {c: self.containment(note_text, c) for c in self.tr_tokens}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best_id, best_cont = ranked[0]
        cont = scores.get(intended_id, 0.0)
        cos = self.sims(note_text).get(intended_id, 0.0)
        ok = cont >= self.CONT_MIN and (best_id == intended_id
                                        or best_cont - cont <= self.TIE_EPS)
        return {"ok": ok, "cont": round(cont, 4), "best_id": best_id,
                "best_cont": round(best_cont, 4), "cosine_sim": round(cos, 4)}


def calibrate():
    """Known-good pairs = the June scribe_B_notes (10 authored + 12 primock) and, as extra
    mass, all existing scribe_C_notes. Prints per-pair sim, rank, and margin."""
    m = Matcher()
    rows = []
    for d, tag in ((os.path.join(HERE, "scribe_B_notes"), "scribe_B"),
                   (os.path.join(HERE, "scribe_C_notes"), "scribe_C")):
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".txt") or "__v" in fn:
                continue
            src, cid = fn[:-4].split("__", 1)
            if cid not in m.transcripts:
                continue
            note = open(os.path.join(d, fn)).read()
            v = m.verify(note, cid)
            rank = 1 if v["best_id"] == cid else None
            rows.append((tag, fn, v["sim"], v["best_sim"], v["second_sim"], rank))
    ok = [r for r in rows if r[5] == 1]
    bad = [r for r in rows if r[5] != 1]
    sims = sorted(r[2] for r in ok)
    margins = sorted(r[2] - r[4] for r in ok)
    print(f"{len(rows)} known-good pairs ({sum(1 for r in rows if r[0]=='scribe_B')} scribe_B, "
          f"{sum(1 for r in rows if r[0]=='scribe_C')} scribe_C)")
    print(f"rank-1 (argmax correct): {len(ok)}/{len(rows)}")
    if bad:
        for r in bad:
            print(f"  NOT rank-1: {r[1]} sim={r[2]:.3f} but best={r[3]:.3f}")
    if sims:
        print(f"correct-pair sim: min={sims[0]:.3f} p5={sims[len(sims)//20]:.3f} "
              f"median={sims[len(sims)//2]:.3f} max={sims[-1]:.3f}")
        print(f"margin over runner-up: min={margins[0]:.3f} median={margins[len(margins)//2]:.3f}")
    # wrong-pair sims: what does a MISattribution look like?
    wrong = []
    for tag, fn, sim, best, second, rank in rows[:40]:
        wrong.append(second)  # runner-up sim ~ the best wrong consult
    wrong.sort()
    if wrong:
        print(f"best-WRONG-consult sim (runner-up): min={wrong[0]:.3f} "
              f"median={wrong[len(wrong)//2]:.3f} max={wrong[-1]:.3f}")


if __name__ == "__main__":
    calibrate()
