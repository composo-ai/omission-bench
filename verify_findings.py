"""Phase C - adversarial verification layer (runs BEFORE the lead author sees findings).

On-plan (claude -p) panels that harden discover.py's wide-cast findings:
  1. REFUTE panel - N independent skeptics per finding, each seeing the FULL note + source and
     trying hard to refute it (paraphrase / synonymy / normal convention / justified inference /
     misquote / immaterial -> refuted; default REFUTED if uncertain). Majority-refute => cut.
  2. SALIENCE rater - independent high/medium/low (overrides the differ's guess).
  3. COMPLETENESS critic (optional, --complete) - per note, hunt discrepancies the differ MISSED;
     new candidates go through the same refute panel. (loop-until-dry: --rounds R)
Every finding ends with a verdict {is_real, votes, salience}; nothing is silently dropped - cut
findings are kept in the file with is_real=false and a reason, and counts are logged.

Usage: python verify_findings.py [--in discover_findings.json] [--panel 3] [--complete] [--rounds 1]
Output: discover_verified.json
"""
import json, os, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from common import claude_json, HERE


def _arg(flag, d=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else d


IN = _arg("--in", "discover_findings.json")
PANEL = int(_arg("--panel", "3"))
DO_COMPLETE = "--complete" in sys.argv
ROUNDS = int(_arg("--rounds", "1"))
WORKERS = int(_arg("--workers", "6"))   # keep modest: shares the lead author's plan rate limit
MODEL = _arg("--model", "claude-opus-4-8")
EFFORT = _arg("--effort", "medium")

# load BOTH the scribe_A corpus and the cross-scribe (scribe_B/scribe_C) corpus; key includes scribe so the
# scribe_B & scribe_C versions of the same consult (both template="audio") don't collide.
CORPUS = {}
for _cf in ("notes_corpus.json", "scribe_notes_corpus.json"):
    _p = os.path.join(HERE, _cf)
    if os.path.exists(_p):
        for r in json.load(open(_p)):
            if r.get("note"):
                CORPUS[(r["source"], r["id"], r["template"], r.get("scribe", "scribe_A"))] = r


def _src(rec):
    fs = ""
    base = rec["transcript"]
    return base, fs


REFUTE = """A discovery tool flagged a possible discrepancy in an AI-generated clinical NOTE versus its SOURCE. Your job is to REFUTE it if at all defensible - be a hard skeptic. It is NOT a real, reportable discrepancy if it is any of: paraphrase, clinical synonymy, a standard abbreviation, a normal documentation convention, justified clinical inference, a quote that does not actually appear, a mischaracterisation by the tool, or clinically immaterial. Default to REFUTED if you are uncertain.

FLAGGED MODE: {mode}
claim: {description}
note says: {note_quote}
source says: {source_quote}

FULL NOTE:
{note}

FULL SOURCE ({skind}):
{source}

Is this a REAL, correctly-characterised discrepancy of the stated kind? Return ONLY JSON {{"refuted": true|false, "reason": "<short>"}}  (refuted=true means it is NOT a genuine issue)."""

SALIENCE = """A confirmed discrepancy in an AI-generated clinical note. Rate how much it matters CLINICALLY - would it change a clinician's understanding of the patient or the patient's care/safety?
mode {mode}: {description}
note: {note_quote}  | source: {source_quote}
Return ONLY JSON {{"salience":"high"|"medium"|"low","reason":"<short>"}}."""

COMPLETE = """Re-read this AI-generated clinical NOTE against its SOURCE. A first pass already found these discrepancies:
{known}
Find ADDITIONAL clinically-meaningful discrepancies the first pass MISSED - ANY kind of failure, do NOT restrict to a predefined list; name the mode yourself with a short label. Prioritise ones that matter (high salience). Quote precisely. If the first pass looks complete, return an empty list.
Return ONLY JSON {{"missed":[{{"mode","note_quote","source_quote","description","severity","salience"}}]}}

FULL NOTE:
{note}

FULL SOURCE ({skind}):
{source}"""


def rec_for(f):
    return CORPUS.get((f["source"], f["id"], f["template"], f.get("scribe", "scribe_A")))


def refute_one(f, k):
    rec = rec_for(f)
    src, _ = _src(rec)
    skind = "transcript" if f.get("check") != "fact_sheet" else "transcript (ground-truth fact-sheet check)"
    r = claude_json(REFUTE.format(mode=f.get("mode"), description=f.get("description"),
                                  note_quote=f.get("note_quote", "-"), source_quote=f.get("source_quote", "-"),
                                  note=rec["note"], source=src[:40000], skind=skind), model=MODEL, effort=EFFORT)
    return bool(r.get("refuted")) if isinstance(r, dict) else True  # unparseable -> treat as refuted (conservative)


def salience_one(f):
    rec = rec_for(f)
    r = claude_json(SALIENCE.format(mode=f.get("mode"), description=f.get("description"),
                                    note_quote=f.get("note_quote", "-"), source_quote=f.get("source_quote", "-")), model=MODEL, effort=EFFORT)
    return r.get("salience") if isinstance(r, dict) else None


def verify_panel(findings):
    """Attach verdict to each finding via PANEL refuters + 1 salience rater."""
    jobs = [("ref", fi, k) for fi in range(len(findings)) for k in range(PANEL)] + [("sal", fi, 0) for fi in range(len(findings))]

    def run(job):
        kind, fi, k = job
        return (kind, fi, refute_one(findings[fi], k) if kind == "ref" else salience_one(findings[fi]))

    refuted = {fi: [] for fi in range(len(findings))}
    sal = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for kind, fi, val in ex.map(run, jobs):
            if kind == "ref":
                refuted[fi].append(val)
            else:
                sal[fi] = val
    for fi, f in enumerate(findings):
        nref = sum(refuted[fi])
        f["verdict"] = {"is_real": nref < (PANEL // 2 + 1), "votes": f"{PANEL-nref}/{PANEL} keep", "salience": sal.get(fi)}
        if sal.get(fi):
            f["salience"] = sal[fi]  # panel salience overrides differ guess
    return findings


def completeness(existing):
    """Per note, find missed discrepancies; return new candidate findings (un-verified)."""
    by_note = {}
    for f in existing:
        by_note.setdefault((f["source"], f["id"], f["template"]), []).append(f)
    keys = list(CORPUS)

    def run(key):
        rec = CORPUS[key]
        known = "; ".join(f"{f['mode']}: {f.get('description','')}" for f in by_note.get(key, [])) or "(none)"
        src, _ = _src(rec)
        skind = "transcript"
        r = claude_json(COMPLETE.format(known=known[:6000], note=rec["note"], source=src[:40000], skind=skind), model=MODEL, effort=EFFORT)
        miss = (r or {}).get("missed", []) if isinstance(r, dict) else []
        out = []
        for m in miss:
            out.append({"id": rec["id"], "source": rec["source"], "template": rec["template"],
                        "check": "completeness", "discovered_by": "completeness", **m})
        return out

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        nested = list(ex.map(run, keys))
    return [x for sub in nested for x in sub]


def main():
    data = json.load(open(os.path.join(HERE, IN)))
    findings = data["all_issues"]
    sal_filter = _arg("--salience")          # e.g. "high,medium" - verify only those (skip low to save cost)
    if sal_filter:
        keep = set(sal_filter.split(","))
        before = len(findings)
        findings = [f for f in findings if f.get("salience") in keep]
        print(f"salience filter {keep}: {before} -> {len(findings)} candidates")
    print(f"verify: {len(findings)} findings, panel={PANEL}, complete={DO_COMPLETE}, rounds={ROUNDS} "
          f"[provider={'openai/'+os.environ.get('BATCH_OPENAI_MODEL','gpt-5.5') if os.environ.get('BATCH_LLM')=='openai' else 'claude-plan'}]")

    findings = verify_panel(findings)
    if DO_COMPLETE:
        for rnd in range(ROUNDS):
            real_so_far = [f for f in findings if f["verdict"]["is_real"]]
            new = completeness(real_so_far)
            print(f"  completeness round {rnd+1}: {len(new)} new candidates")
            if not new:
                break
            new = verify_panel(new)
            findings += new
            if not any(f["verdict"]["is_real"] for f in new):
                break

    real = [f for f in findings if f["verdict"]["is_real"]]
    json.dump({"per_note": data.get("per_note", []), "all_issues": findings},
              open(os.path.join(HERE, "discover_verified.json"), "w"), indent=1)

    print(f"\n=== verified: {len(real)}/{len(findings)} findings survived the refute panel ===")
    print("survived by mode:    ", dict(Counter(f["mode"] for f in real)))
    print("survived by salience:", dict(Counter(f.get("salience") for f in real)))
    cut = [f for f in findings if not f["verdict"]["is_real"]]
    print(f"cut: {len(cut)} (kept in file with is_real=false)")
    print("cut by mode:         ", dict(Counter(f["mode"] for f in cut)))
    from common import _OAI_USAGE
    if _OAI_USAGE["calls"]:
        print(f"OpenAI usage: {_OAI_USAGE['calls']} calls | {_OAI_USAGE['in']:,} in + {_OAI_USAGE['out']:,} out tokens")
    print("saved -> discover_verified.json")


if __name__ == "__main__":
    main()
