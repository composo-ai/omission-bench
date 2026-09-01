"""Blind fact-sheet extraction from a transcript, with no note in view.

One `claude-opus-5` call per consultation on the pre-registered extraction prompt, used
VERBATIM, with the `importance` grade appended to the salience_traps item spec. The call
goes through common.claude - the `claude` binary, billed to a Claude subscription - as all
construction work does. The model was changed mid-study from claude-opus-4-8, and the
opus-4-8 sheets were discarded to master/discarded_opus48/ so the extracted corpus is
consistent on one model.

Sources (--source):
  primock   57 consults from the parsed PriMock57 transcripts (common.load_primock)
  aci       48 consults from master/aci_subsample.json
  trapblind 10 consults from master/trapblind_scenarios_critiqued.json
  authored  30 consults from authored_scenarios.json - the strictly ADDITIVE
            re-extraction of the authored stratum. Only id/presenting_complaint/transcript
            are loaded: the authored fact_sheet is NEVER shown to the extractor (that blindness is
            the whole point), and authored_scenarios.json is only ever READ by this script.

ROUTE: the extraction layer runs WHOLLY on one transport, `claude -p` via common.claude, 12
workers. This is deliberate and not a cost decision: extraction produces the ground-truth
annotation artifact, and every other artifact in this corpus (the June authored fact sheets, the
trap-blind transcripts, the severity backfill) came through `claude -p`, which wraps its own
system prompt and so is not the same instrument as a bare API call. Keeping the whole extraction
layer on one transport means the blind-recovery reading of fact-sheet provenance measures
authored-vs-extracted, not transport-vs-transport.
The critique layer (critique_extracted.py --route) may use a different transport - it audits the
artifact rather than producing it.

PROMPT VERSIONS (--prompt-version, default v1). The PriMock stratum
re-runs on a v2 prompt derived from v1 by fixing exactly the four root causes documented in
master/critique_extracted_primock_v1.md - (1) ASR-uncertainty rule (garbled/ambiguous/contradicted
turns never become confident facts; must_contain rests only on clean evidence), (2) equivalence
rule (must_not_contain only for renderings clinically wrong under any standard reading; correct
unit/dose/frequency conversions protected), (3) note-bearing-only rule (no closure/boilerplate/
patient-education padding in must_contain), (4) transcript-only traps (no scribe-meta framing;
this also rewrites v1's own "that notes commonly drop" tail, which seeded the meta framing the
leakage critic then flagged) - plus the rubric-anchored importance instruction (condensed
from the study's severity rubric, replacing the earlier one-line wording). JSON schema identical.
Other strata stay on v1 (their gates passed; the v2 changes are ASR-specific). Both prompts are
published verbatim: prompts/extraction_prompt_v1.txt and prompts/extraction_prompt_v2.txt.

INPUT VIEW (--input-view full|transcript-only), the "v2.1" instrument: the
primock57_parsed.json `presenting_complaint` header is the actor's case-card prompt, not a fact
about the consultation (actors improvised and deviated - day4_consultation10's header claims
diarrhoea the transcript's patient denies twice). It was never legitimate ground truth, so for
PriMock the extraction RECORD carries id + transcript ONLY - the header is struck from the saved
record and therefore from every downstream critic/revise view (critique_extracted.py builds its
views from this record). transcript-only is FORCED for --source primock; other sources default
to the full record view, unchanged. Instrument label: v2 prompt text + transcript-only view =
"v2.1". The extraction LLM prompt itself always consumed {transcript} only - the header
contamination entered via the saved record into the critique layer's views.

Usage: python corpus/extract_fact_sheets.py --source primock|aci|trapblind|authored
         [--prompt-version v1|v2] [--input-view full|transcript-only]
         [--workers 12] [--only id1,id2] [--limit N] [--max-seconds S]
Output: master/fact_sheets_raw_<stem>.json (+ extract_failures/ raw dumps for any parse miss)
        stem = source, except authored -> `authored_extracted`.

Idempotent BY ID, PROMPT VERSION AND INPUT VIEW: ids already present in the output file with a
shape-valid fact_sheet extracted under the REQUESTED prompt version + input view are skipped
(rows from another version/view are stale and re-extracted - a stratum must be one instrument,
never a mixture); rerun to retry failures. Partial progress is saved after every completed consult. --limit N and
--max-seconds S bound one invocation so a long source can be run as several blocking chunks
(exit code 0 either way; rerun until the count is right).
Then: critique_extracted.py --source <source>.
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from common import claude, load_primock, validate_fact_sheet, HERE

MODEL = "claude-opus-5"
EFFORT = "medium"
TIMEOUT = 600
ATTEMPTS = 3   # one attempt plus two retries
T0 = time.time()

# The pre-registered extraction prompt, verbatim, with the `importance` instruction appended
# to the salience_traps item spec (the only later change to this prompt).
PROMPT_V1 = """You are a UK GP and a meticulous clinical auditor working on an AI-evaluation research study.
Below is the TRANSCRIPT of a primary-care consultation. Your job is to reverse-engineer the
ground-truth FACT SHEET that a perfect clinical note of this consultation would be judged
against. Work ONLY from the transcript - you have no other source of truth, and you must not
import outside clinical knowledge as if it were said (clinical judgement may guide what is
LOAD-BEARING, never what is TRUE).

TRANSCRIPT:
{transcript}

Produce a fact_sheet with exactly three keys:

1. "must_contain": the facts a CORRECT and complete note of THIS consultation must include.
   Cover: the presenting complaint; key positive findings; the pertinent NEGATIVES the patient
   explicitly denied or the clinician explicitly noted as absent; examination findings (or the
   explicit absence of examination if the consult is remote); the working diagnosis/impression
   AS STATED (preserve any hedging); every element of the plan including drug names, doses,
   frequencies and durations as actually said; safety-netting instructions; and material social
   or risk history. Each item is an object:
     { "fact": "<one atomic fact, precisely stated>",
       "evidence": "<a short VERBATIM quote from the transcript (max 25 words) that supports it>",
       "load_bearing": "high" | "medium" }
   Rules: one fact per item, no compounds. Only include facts a reasonable clinician would
   expect in the note - do not pad with trivia. If it is not supported by a quotable span,
   it does not go in.

2. "must_not_contain": assertions that would be ERRORS for a note of THIS consultation - each
   one a specific wrong claim a scribe could plausibly make here. Cover, where applicable: a
   hardened/confirmed diagnosis where the clinician stayed tentative; an examination, test or
   observation that did not happen or whose result is not back; an inverted negative; a wrong
   dose/frequency; a flipped laterality or wrong body site; a decision recorded as taken when
   it was only discussed, deferred or offered; a fabricated medication or referral. Each item:
     { "assertion": "<the erroneous claim>",
       "why_wrong": "<one line: what the transcript actually establishes>" }
   Do NOT include generic errors that could apply to any consult - every item must be anchored
   in this transcript.

3. "salience_traps": moments in THIS transcript where a scribe could plausibly go wrong - only
   traps that genuinely exist in the dialogue; an unremarkable consult may have few. Each item:
     { "trap": "<what is tempting to get wrong>",
       "correct_handling": "<what a correct note does>",
       "mode": <one of: omission, negation, dose_value, laterality, attribution,
               modality_hardening, temporal, fabrication, decision_status, anchoring> }
   Add a fourth field "importance": "critical" if a careful clinician would consider missing
   this a significant safety or care gap, "supporting" if it's clinically relevant but wouldn't
   itself change management, "peripheral" if a competent note could reasonably compress it.
   Look especially for: hedged or uncertain statements; denied symptoms; deferred or
   conditional decisions; back-up/delayed prescriptions; remote consults where no examination
   was possible; stopped or changed medications; attribution of findings to the right person;
   self-management/OTC advice that notes commonly drop.

CRITICAL: derive everything ONLY from the transcript. Never invent a fact, never list a trap
with no corresponding moment, never mark an assertion as wrong if the transcript supports it.
If the transcript is ambiguous on a point, the fact sheet must preserve that ambiguity rather
than resolve it.

Output ONLY a single valid JSON object with key "fact_sheet" containing the three keys above.
Escape all newlines inside strings as \\n. No commentary, no markdown fences."""


# v1 plus exactly the four fixes for the root causes documented in
# master/critique_extracted_primock_v1.md, plus the rubric-anchored importance instruction
# (condensed faithfully from the study's severity rubric; replaces the earlier one-liner).
# Schema unchanged. Every added/changed passage is grounded in a flagged v1 failure:
#   fix 1 (ASR-uncertainty)   - e.g. "temperature was not measured" asserted high-load from the
#                               garbled "I didn't mention my temperature, no"; night-sweats denial
#                               read out of a bare "Uh, no." after a compound question.
#   fix 2 (equivalence)       - e.g. must_not_contain "Paracetamol 1g four times daily" penalising
#                               the faithful conversion of "two tablets up to four times a day".
#   fix 3 (note-bearing only) - e.g. "patient confirmed the plan was clear and had no further
#                               questions" and virus-education phrasing required at medium load.
#   fix 4 (transcript-only traps) - e.g. traps built on "notes commonly drop OTC advice" / fit
#                               notes never mentioned; v1's own "that notes commonly drop" tail
#                               seeded this framing and is rewritten here.
PROMPT_V2 = """You are a UK GP and a meticulous clinical auditor working on an AI-evaluation research study.
Below is the TRANSCRIPT of a primary-care consultation. Your job is to reverse-engineer the
ground-truth FACT SHEET that a perfect clinical note of this consultation would be judged
against. Work ONLY from the transcript - you have no other source of truth, and you must not
import outside clinical knowledge as if it were said (clinical judgement may guide what is
LOAD-BEARING, never what is TRUE).

The transcript is an automatic speech-recognition rendering of a REAL recorded consultation.
It can contain mistranscribed words, <UNSURE>...</UNSURE> tags around low-confidence words,
<UNIN/> and <INAUDIBLE_SPEECH/> gaps, garbled or truncated turns, and compound questions
answered with a single unattributable reply. ASR-UNCERTAINTY RULE: a garbled, ambiguous, or
contradicted turn must NEVER become a confident fact. Either omit the point, or state the
uncertainty inside the fact itself (e.g. "patient appears to deny night sweats, though the
reply is ambiguous"). A must_contain item may rest ONLY on clean evidence: a quote that is
intact, clearly attributable to one speaker answering one identifiable question, and not
contradicted elsewhere in the transcript. A reply tagged <UNSURE>, a turn interrupted by an
inaudible gap at the load-bearing word, a bare "yes"/"no" after a multi-part question, or a
clinician's leading summary the patient never clearly confirms is NOT clean evidence.

TRANSCRIPT:
{transcript}

Produce a fact_sheet with exactly three keys:

1. "must_contain": the facts a CORRECT and complete note of THIS consultation must include.
   Cover: the presenting complaint; key positive findings; the pertinent NEGATIVES the patient
   explicitly denied or the clinician explicitly noted as absent; examination findings (or the
   explicit absence of examination if the consult is remote); the working diagnosis/impression
   AS STATED (preserve any hedging); every element of the plan including drug names, doses,
   frequencies and durations as actually said; safety-netting instructions; and material social
   or risk history. Each item is an object:
     { "fact": "<one atomic fact, precisely stated>",
       "evidence": "<a short VERBATIM quote from the transcript (max 25 words) that supports it>",
       "load_bearing": "high" | "medium" }
   Rules: one fact per item, no compounds. Only include facts a reasonable clinician would
   expect in the note - do not pad with trivia. If it is not supported by a quotable span,
   it does not go in. NOTE-BEARING CONTENT ONLY: conversational closure ("no further
   questions", thanks and goodbyes), identity-confirmation and secure-location rituals,
   reassurance boilerplate ("nothing to worry about"), and patient-education explanations are
   NOT required note content - include such material only where it is clinically load-bearing
   for THIS consult (a specific safety-netting instruction, a stated reason the plan was
   chosen, demographics the assessment actually turns on). The clinical substance behind the
   phrasing (the working impression, the plan, the safety-net) belongs in its own item; the
   conversational wrapper does not.

2. "must_not_contain": assertions that would be ERRORS for a note of THIS consultation - each
   one a specific wrong claim a scribe could plausibly make here. Cover, where applicable: a
   hardened/confirmed diagnosis where the clinician stayed tentative; an examination, test or
   observation that did not happen or whose result is not back; an inverted negative; a wrong
   dose/frequency; a flipped laterality or wrong body site; a decision recorded as taken when
   it was only discussed, deferred or offered; a fabricated medication or referral. Each item:
     { "assertion": "<the erroneous claim>",
       "why_wrong": "<one line: what the transcript actually establishes>" }
   Do NOT include generic errors that could apply to any consult - every item must be anchored
   in this transcript. EQUIVALENCE RULE: an assertion may be listed ONLY if it is clinically
   WRONG under every standard reading of the transcript - never forbid a rendering a competent
   clinician could correctly write. Correct unit/dose/frequency conversions and standard
   clinical phrasing of what was actually said are explicitly protected: if the clinician
   advises paracetamol "two tablets up to four times a day", a note reading "paracetamol 1g up
   to four times daily" is a faithful conversion (2 x 500mg tablets), NOT an error - what may
   be forbidden is a rendering that changes the clinical content, such as dropping the "up to"
   or recording "twice daily". Where a turn is ambiguous, any defensible reading of it is not
   an error: scope each assertion so it catches only the genuinely wrong version (name the
   dropped qualifier or the contradicting turn in why_wrong), never a faithful one.

3. "salience_traps": moments in THIS transcript where a scribe could plausibly go wrong - only
   traps that genuinely exist in the dialogue; an unremarkable consult may have few. Each item:
     { "trap": "<what is tempting to get wrong>",
       "correct_handling": "<what a correct note does>",
       "mode": <one of: omission, negation, dose_value, laterality, attribution,
               modality_hardening, temporal, fabrication, decision_status, anchoring> }
   Add a fourth field "importance", graded per the study's severity rubric. Grade the clinical
   consequence of the note getting THIS item wrong (omitting, fabricating or altering it) - the
   consequence of the error in the note, not the drama of the topic:
     "critical" - the error would plausibly change clinical action, delay or misdirect care, or
       create a safety risk: an omitted working diagnosis or a hedged impression hardened into
       certainty; omitted red-flag safety-netting; an omitted or wrong drug, allergy, dose,
       frequency or duration; a conditional plan made definite; a fabricated examination,
       finding or result; an omitted pertinent negative that licenses the management decision;
       wrong patient identity attributes used clinically.
     "supporting" - degrades the note's completeness, clarity or defensibility but is unlikely
       to change what happens next: duration/onset detail that colours but does not gate the
       plan; social context with no immediate action attached; compressed phrasing that loses
       nuance without inverting meaning; a secondary symptom that neither supports nor
       threatens the diagnosis.
     "peripheral" - no plausible clinical consequence: conversational colour, rapport,
       patient-education phrasing, administrative closure, redundant restatement of something
       already captured.
   Decide in order: (1) ACTION test - would a competent GP, or the next clinician reading the
   note, plausibly DO something different if this error stood? yes -> critical. (2) SAFETY
   test - even if action today is unchanged, does the error remove a safety net or create a
   latent risk? yes -> critical. (3) RECORD-QUALITY test - does it materially weaken the note
   as a clinical record? yes -> supporting. (4) Otherwise peripheral. If genuinely torn between
   two grades after the procedure, take the LOWER grade. Grade each trap independently; never
   curve to a target distribution.
   Look especially for: hedged or uncertain statements; denied symptoms; deferred or
   conditional decisions; back-up/delayed prescriptions; remote consults where no examination
   was possible; stopped or changed medications; attribution of findings to the right person;
   self-management/OTC advice given in THIS consultation. TRANSCRIPT-ONLY TRAPS: every trap
   must name a specific moment of THIS dialogue (a hedge, a self-correction, a garbled or
   ambiguous reply, a deferred decision) that makes it a trap. Never write a trap from
   meta-knowledge of how scribes or notes typically fail ("notes commonly drop...", "easily
   compressed to..."), never describe scribe or documentation habits in the trap or its
   correct_handling, and never build a trap around a documentation convention the transcript
   never mentions (fit notes, PRN, referral pathways).

CRITICAL: derive everything ONLY from the transcript. Never invent a fact, never list a trap
with no corresponding moment, never mark an assertion as wrong if the transcript supports it.
If the transcript is ambiguous on a point, the fact sheet must preserve that ambiguity rather
than resolve it.

Output ONLY a single valid JSON object with key "fact_sheet" containing the three keys above.
Escape all newlines inside strings as \\n. No commentary, no markdown fences."""

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


SOURCE = _arg("--source")
PV = _arg("--prompt-version", "v1")   # v2 for the PriMock re-run;
assert PV in PROMPTS, "usage: --prompt-version v1|v2"   # default v1 for every other stratum
# PriMock is FORCED transcript-only (the presenting_complaint header
# is actor case-card metadata, never ground truth - re-admitting it would re-contaminate the
# stratum). Other sources default to the full record view, unchanged.
INPUT_VIEW = "transcript-only" if SOURCE == "primock" else _arg("--input-view", "full")
assert INPUT_VIEW in ("full", "transcript-only"), "usage: --input-view full|transcript-only"
WORKERS = int(_arg("--workers", "12"))
ONLY = set((_arg("--only") or "").split(",")) - {""}
LIMIT = int(_arg("--limit", "0"))
MAX_SECONDS = float(_arg("--max-seconds", "0"))
STEMS = {"primock": "primock", "aci": "aci", "trapblind": "trapblind",
         "authored": "authored_extracted"}
assert SOURCE in STEMS, "usage: --source primock|aci|trapblind|authored"
OUTFILE = os.path.join(HERE, "master", f"fact_sheets_raw_{STEMS[SOURCE]}.json")
FAILDIR = os.path.join(HERE, "extract_failures")


def load_consults(source):
    """Normalise each source to {id, transcript, (+presenting_complaint / split)}.

    NOTE for `authored`: only id/presenting_complaint/transcript are read - the authored
    fact_sheet is deliberately dropped so extraction stays blind.
    NOTE for `primock` (always transcript-only): the
    presenting_complaint header is STRUCK from the record - it is the actor's case-card
    prompt, not a fact about the consultation, and must reach no extraction or critique view."""
    if source == "primock":
        return [{"id": r["id"], "source": "primock",
                 "transcript": r["transcript"]} for r in load_primock()]
    if source == "aci":
        rows = json.load(open(os.path.join(HERE, "master", "aci_subsample.json")))
        return [{"id": r["id"], "source": "aci", "split": r["split"],
                 "transcript": r["transcript"]} for r in rows]
    if source == "authored":
        rows = json.load(open(os.path.join(HERE, "authored_scenarios.json")))
        return [{"id": r["id"], "source": "authored",
                 "presenting_complaint": r.get("presenting_complaint", ""),
                 "transcript": r["transcript"]} for r in rows]
    rows = json.load(open(os.path.join(HERE, "master", "trapblind_scenarios_critiqued.json")))
    return [{"id": r["id"], "source": "trapblind",
             "presenting_complaint": r.get("presenting_complaint", ""),
             "transcript": r["transcript"]} for r in rows]


def validate(fs):
    """Shape-check a fact_sheet (the shared schema in common.validate_fact_sheet, so the
    critic/revise path in critique_extracted.py holds revisions to the same contract)."""
    return validate_fact_sheet(fs)


def main():
    consults = load_consults(SOURCE)
    if INPUT_VIEW == "transcript-only":          # general strip (primock is pre-stripped anyway)
        for c in consults:
            c.pop("presenting_complaint", None)
    if ONLY:
        consults = [c for c in consults if c["id"] in ONLY]

    done = {}
    if os.path.exists(OUTFILE):
        # idempotence is keyed on id AND prompt version AND input view: a row extracted under
        # another prompt version or record view is stale for this run (one stratum = one
        # instrument, never a mixture). Rows predating the fields were v1 / full-view.
        done = {r["id"]: r for r in json.load(open(OUTFILE))
                if "fact_sheet" in r and not validate(r["fact_sheet"])
                and r.get("extraction_provenance", {}).get("prompt_version", "v1") == PV
                and r.get("extraction_provenance", {}).get("input_view", "full") == INPUT_VIEW}
    todo = [c for c in consults if c["id"] not in done]
    if LIMIT:
        todo = todo[:LIMIT]
    print(f"[{SOURCE}] {len(consults)} consults | {len(done)} already extracted on {PV}/"
          f"{INPUT_VIEW} | "
          f"{len(todo)} to run on {MODEL} (prompt {PV}, effort={EFFORT}, {WORKERS} workers"
          + (f", limit {LIMIT}" if LIMIT else "")
          + (f", budget {MAX_SECONDS:.0f}s" if MAX_SECONDS else "") + ")", flush=True)
    if not todo:
        report(consults, done)
        return

    os.makedirs(FAILDIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
    lock = threading.Lock()
    order = {c["id"]: i for i, c in enumerate(consults)}
    n_done = [len(done)]

    def save():
        rows = sorted(done.values(), key=lambda r: order.get(r["id"], 1_000_000))
        tmp = OUTFILE + ".tmp"
        json.dump(rows, open(tmp, "w"), ensure_ascii=False, indent=1)
        os.replace(tmp, OUTFILE)

    skipped = [0]

    def one(c):
        if MAX_SECONDS and time.time() - T0 > MAX_SECONDS:
            with lock:
                skipped[0] += 1
            return None                       # wall-clock budget spent; rerun picks this up
        prompt = PROMPTS[PV].replace("{transcript}", c["transcript"])
        t0 = time.time()
        for attempt in range(1, ATTEMPTS + 1):
            raw = claude(prompt, model=MODEL, effort=EFFORT, timeout=TIMEOUT, retries=0)
            m = re.search(r"(\{.*\})", raw, re.S)
            obj = None
            if m:
                try:
                    obj = json.loads(m.group(1))
                except Exception:
                    obj = None
            fs = (obj or {}).get("fact_sheet")
            probs = validate(fs)
            if not probs:
                rec = dict(c)
                rec["fact_sheet"] = fs
                rec["extraction_provenance"] = {
                    "model": MODEL, "route": "plan", "transport": "claude -p",
                    "effort": EFFORT, "attempt": attempt, "blind": True,
                    "prompt_version": PV, "input_view": INPUT_VIEW,
                    **({"instrument": "v2.1"} if PV == "v2"
                       and INPUT_VIEW == "transcript-only" else {}),
                    "spec": "blind fact-sheet extraction, prompt " + PV
                            + (" (rubric-anchored importance)"
                               if PV == "v2" else "")
                            + (" + transcript-only input view"
                               if INPUT_VIEW == "transcript-only" else "")
                            + (" + additive re-extraction" if SOURCE == "authored" else "")}
                with lock:
                    done[c["id"]] = rec
                    n_done[0] += 1
                    save()
                    print(f"  [{n_done[0]}/{len(consults)}] {c['id']:28} ok "
                          f"(mc {len(fs['must_contain']):2d} | mnc {len(fs['must_not_contain']):2d} | "
                          f"traps {len(fs['salience_traps']):2d}) attempt {attempt}, "
                          f"{time.time() - t0:.0f}s", flush=True)
                return True
            dump = os.path.join(FAILDIR, f"{SOURCE}__{c['id']}__a{attempt}.txt")
            open(dump, "w").write((raw or "(empty completion / timeout)")
                                  + "\n\n--- validation problems ---\n" + "\n".join(probs))
            with lock:
                print(f"  {c['id']}: attempt {attempt} failed "
                      f"({probs[0] if probs else 'no JSON'}) -> {os.path.basename(dump)}", flush=True)
        return False

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(one, todo))

    failed = [c["id"] for c, ok in zip(todo, results) if ok is False]
    if failed:
        print(f"\nFAILED after {ATTEMPTS} attempts (rerun to retry): {failed}", flush=True)
    if skipped[0]:
        print(f"\nBUDGET HIT after {time.time() - T0:.0f}s - {skipped[0]} consults not attempted; "
              f"rerun the same command to continue.", flush=True)
    report(consults, done)


def report(consults, done):
    rows = [done[c["id"]] for c in consults if c["id"] in done]
    if not rows:
        return
    mc = [len(r["fact_sheet"]["must_contain"]) for r in rows]
    mnc = [len(r["fact_sheet"]["must_not_contain"]) for r in rows]
    tr = [len(r["fact_sheet"]["salience_traps"]) for r in rows]
    from collections import Counter
    imp = Counter(t["importance"] for r in rows for t in r["fact_sheet"]["salience_traps"])
    n_traps = sum(tr) or 1
    print(f"\n[{SOURCE}] extracted {len(rows)}/{len(consults)} -> {os.path.relpath(OUTFILE, HERE)}")
    print(f"  mean must_contain {sum(mc)/len(rows):.1f} | must_not_contain {sum(mnc)/len(rows):.1f} | "
          f"traps {sum(tr)/len(rows):.1f}")
    print("  importance: " + " | ".join(
        f"{k} {imp.get(k, 0)} ({100*imp.get(k, 0)/n_traps:.0f}%)"
        for k in ("critical", "supporting", "peripheral")), flush=True)


if __name__ == "__main__":
    main()
