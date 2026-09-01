"""The broad discovery pass: one wide-cast read of a note against its transcript.

Two complementary checks per note, both cast WIDE - find everything, cap nothing. What
survives is decided afterwards, by the adversarial refutation panel in
`verify_findings.py`, not here:

  1. multi-pass-by-mode-family differ (ALL notes): one focused pass per failure mode vs the
     TRANSCRIPT, so rare modes (laterality, attribution) aren't crowded out by omissions.
  2. fact-sheet check (AUTHORED notes only): the note vs the verified ground-truth
     must_contain / must_not_contain, which is the strongest signal available: for those
     consultations the answer key was written before the note existed.

Each finding separates the verifiable call - is it in the source? - from the salience
call, does it matter? The two are graded independently and reported separately.

Usage: python census/discover.py [--source authored,primock] [--limit N] [--modes omission,negation,...]
Output: discover_findings.json  (+ printed summary)
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import json, os, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from common import claude_json, HERE


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


# Each mode = one focused pass. Keep them disjoint so a pass hunts ONE shape exhaustively.
MODE_PASSES = {
    "omission":            "clinically-significant information clearly present in the TRANSCRIPT but MISSING from the NOTE (a stated symptom, finding, medication, allergy, plan element, safety-net instruction, or red flag a clinician would expect documented). NOT trivial/normal omissions - only ones that matter clinically.",
    "negation":            "polarity/negation flips or distortions: the NOTE asserts the opposite presence/absence vs the TRANSCRIPT (e.g. 'no chest pain'->'chest pain', 'denies'->'reports', normal->abnormal).",
    "dose_value":          "a medication dose / frequency / route, or a numeric value (BP, count, duration number, lab) in the NOTE that differs from or is unsupported by the TRANSCRIPT.",
    "laterality":          "a left / right / bilateral / site / anatomy in the NOTE that differs from the TRANSCRIPT.",
    "attribution":         "a finding the NOTE attributes to the wrong subject vs the TRANSCRIPT (patient's own symptom vs family history; who said or did what).",
    "modality_hardening":  "tentativeness in the TRANSCRIPT hardened in the NOTE into something more definite: a suspicion/'?'/'possible'/'considered'/'wait-and-see' recorded as a firm diagnosis or decision or 'started'; a planned/ordered test recorded as already done or resulted.",
    "temporal":            "onset / duration / timing / tense in the NOTE that differs from or OVER-SPECIFIES the TRANSCRIPT (e.g. a vague 'maybe a year' written as 'for 1 year'; 'today'->'yesterday').",
    "fabrication":         "a clinically-meaningful fact / finding / diagnosis / detail ASSERTED in the NOTE that is NOT supported anywhere in the TRANSCRIPT.",
    "internal_inconsistency": "the NOTE contradicting ITSELF - one section/statement inconsistent with another within the same note.",
}

MODES = (_arg("--modes") or ",".join(list(MODE_PASSES) + ["open"])).split(",")  # 9 targeted + 1 open emergent pass
SOURCES = (_arg("--source") or "authored,primock").split(",")
LIMIT = int(_arg("--limit")) if "--limit" in sys.argv else None
MODEL = _arg("--model", "claude-opus-4-8")     # opus: tier is ~free on latency, best recall on subtle misses
EFFORT = _arg("--effort", "medium")            # medium = ~28s/call vs 104s at max, full detection
WORKERS = int(_arg("--workers", "6"))   # keep modest: shares one Claude subscription's rate limit

OPEN_DIFFER = """You are checking an AI-generated clinical NOTE against the consultation TRANSCRIPT it was produced from.
Find EVERY clinically-meaningful discrepancy of ANY kind - do NOT restrict to a predefined list. Anything a clinician
reviewing the note against the transcript would flag as wrong, missing, misleading, fabricated, distorted, or unsafe.
Ignore pure paraphrase, clinical synonymy, standard abbreviation, and normal documentation conventions.
For each: mode (a SHORT label YOU choose for the kind of failure, e.g. 'diagnosis omitted', 'invented finding',
'hardened plan', 'wrong subject', 'exam not done but documented'), note_quote (or "-" for an omission),
source_quote (or "-" if genuinely absent), description (one line), severity (critical|moderate|trivial),
salience (high|medium|low = would THIS difference change a clinician's understanding or the patient's care?).
Return ONLY JSON: {{"issues":[{{"mode","note_quote","source_quote","description","severity","salience"}}]}}  (empty list if none).

TRANSCRIPT:
{transcript}

NOTE:
{note}"""

DIFFER = """You are checking an AI-generated clinical NOTE against the consultation TRANSCRIPT it was produced from.
Focus ONLY on this failure mode - {mode}: {instr}
List EVERY instance of THIS mode you can find (do NOT cap the list; do NOT report other modes). Ignore pure paraphrase,
clinical synonymy, standard abbreviation, and normal documentation conventions.
For each instance: note_quote (short quote from the note, or "-" for an omission), source_quote (the relevant transcript
quote, or "-" if genuinely absent), description (one line), severity (critical|moderate|trivial = clinical seriousness of
the fact), salience (high|medium|low = would THIS difference change a clinician's understanding or the patient's care?).
Return ONLY JSON: {{"issues":[{{"note_quote","source_quote","description","severity","salience"}}]}}  (empty list if none).

TRANSCRIPT:
{transcript}

NOTE:
{note}"""

FS_CHECK = """You are checking an AI-generated clinical NOTE against a VERIFIED ground-truth fact-sheet for the same consultation.
MUST_CONTAIN - a correct note should reflect each (omitting or materially distorting one is an error):
{must_contain}
MUST_NOT_CONTAIN - asserting any of these is an error (unsupported / over-reach / hardened):
{must_not_contain}
Produce an issue for each MUST_CONTAIN item the NOTE omits or materially distorts, and each MUST_NOT_CONTAIN item the NOTE asserts.
Return ONLY JSON: {{"issues":[{{"kind":"omission"|"forbidden_asserted","item","note_quote","description","severity","salience"}}]}}

NOTE:
{note}"""


def differ_pass(note_rec, mode):
    if mode == "open":
        res = claude_json(OPEN_DIFFER.format(transcript=note_rec["transcript"][:40000], note=note_rec["note"]),
                          model=MODEL, effort=EFFORT)
        issues = (res or {}).get("issues", []) if isinstance(res, dict) else []
        for i in issues:
            i["check"] = "open"          # keep the model's emergent mode label
            i.setdefault("mode", "open")
        return issues
    res = claude_json(DIFFER.format(mode=mode, instr=MODE_PASSES[mode],
                                    transcript=note_rec["transcript"][:40000], note=note_rec["note"]),
                      model=MODEL, effort=EFFORT)
    issues = (res or {}).get("issues", []) if isinstance(res, dict) else []
    for i in issues:
        i["mode"], i["check"] = mode, "transcript"
    return issues


def factsheet_pass(note_rec):
    fs = note_rec["fact_sheet"]
    res = claude_json(FS_CHECK.format(
        must_contain="\n".join("- " + x for x in fs["must_contain"]),
        must_not_contain="\n".join("- " + x for x in fs["must_not_contain"]),
        note=note_rec["note"]), model=MODEL, effort=EFFORT)
    issues = (res or {}).get("issues", []) if isinstance(res, dict) else []
    for i in issues:
        i["mode"] = "omission" if i.get("kind") == "omission" else "fabrication"
        i["check"] = "fact_sheet"
    return issues


def main():
    corpus_file = os.path.join(HERE, _arg("--corpus", "notes_corpus.json"))
    if "--help" in sys.argv or "-h" in sys.argv or not os.path.exists(corpus_file):
        raise SystemExit(
            (__doc__ or "").strip() + "\n\n"
            "  --corpus FILE   the notes to read (default: notes_corpus.json). A JSON list\n"
            "                  of {id, source, scribe, template, note, transcript} records;\n"
            "                  `ingest_scribe_notes.py` builds one from captured notes.\n"
            "  --source LIST   which strata to read (default: authored,primock)\n"
            "  --modes LIST    which passes to run (default: every targeted pass, plus open)\n"
            "  --limit N       first N consultations only\n\n"
            + ("" if os.path.exists(corpus_file) else "No corpus at %s.\n" % corpus_file))
    corpus = [r for r in json.load(open(corpus_file))
              if r.get("note") and r["source"] in SOURCES]
    if LIMIT:
        # limit by consult id, keep both templates
        keep_ids = []
        for r in corpus:
            if r["id"] not in keep_ids:
                keep_ids.append(r["id"])
            if len(keep_ids) >= LIMIT:
                break
        corpus = [r for r in corpus if r["id"] in set(keep_ids)]
    print(f"discover: {len(corpus)} notes x {len(MODES)} mode-passes"
          f"{' + fact_sheet check (authored)' if any(r['source']=='authored' for r in corpus) else ''}")

    # build jobs: (note_idx, mode) for differ + (note_idx, '__fs__') for authored fact-sheet
    jobs = [(idx, m) for idx in range(len(corpus)) for m in MODES]
    jobs += [(idx, "__fs__") for idx, r in enumerate(corpus) if r["source"] == "authored" and r.get("fact_sheet")]

    def run(job):
        idx, m = job
        rec = corpus[idx]
        try:
            issues = factsheet_pass(rec) if m == "__fs__" else differ_pass(rec, m)
        except Exception as e:
            print(f"  [warn] {rec['source']}/{rec['id']}/{rec['template']} {m}: {str(e)[:80]}")
            issues = []
        tag = {"id": rec["id"], "source": rec["source"], "template": rec["template"],
               "scribe": rec.get("scribe", "scribe_A")}
        return [{**tag, **i} for i in issues]

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        nested = list(ex.map(run, jobs))
    all_issues = [i for sub in nested for i in sub]

    # per-note rollup
    per_note = {}
    for r in corpus:
        k = (r["id"], r["template"], r.get("scribe", "scribe_A"))
        per_note.setdefault(k, {"id": r["id"], "source": r["source"], "template": r["template"],
                                "scribe": r.get("scribe", "scribe_A"), "n_issues": 0})
    for i in all_issues:
        per_note[(i["id"], i["template"], i.get("scribe", "scribe_A"))]["n_issues"] += 1

    json.dump({"per_note": list(per_note.values()), "all_issues": all_issues},
              open(os.path.join(HERE, "discover_findings.json"), "w"), indent=1)

    with_issues = [v for v in per_note.values() if v["n_issues"] > 0]
    print(f"\n=== discover: {len(all_issues)} candidate findings across {len(corpus)} notes ===")
    print(f"notes with >=1 finding: {len(with_issues)}/{len(corpus)}")
    print("by mode:    ", dict(Counter(i["mode"] for i in all_issues)))
    print("by severity:", dict(Counter(i.get("severity") for i in all_issues)))
    print("by salience:", dict(Counter(i.get("salience") for i in all_issues)))
    print("by check:   ", dict(Counter(i.get("check") for i in all_issues)))
    print("by source:  ", dict(Counter(i["source"] for i in all_issues)))
    print("\n-- sample high-salience (up to 15) --")
    hi = [i for i in all_issues if i.get("salience") == "high"]
    for i in hi[:15]:
        print(f"[{i['source']}/{i['id']}/{i['template']}] {i['mode']}/{i.get('severity')}: {i.get('description')}")
    print(f"\n({len(hi)} high-salience total)  saved -> discover_findings.json")


if __name__ == "__main__":
    main()
