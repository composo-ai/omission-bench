"""The ground-truth gate: the audit a pair set had to pass before any judge scored it.

The audit runs over one master substrate, routes the auditor through OpenRouter,
reports per stratum, and carves a GEPA dev pool out of the frozen set at the freeze
step for the prompt-optimisation campaigns. Substrate:

  notes  137 = 30 authored (ideal_notes_balanced.json vs authored_scenarios.json
               fact sheets) + 9 trapblind + 53 primock + 45 aci
               (master/ideal_notes_master.json vs master/fact_sheets_<src>_core.json,
               CORE view: core must_contain items; post-excise must_not_contain;
               all salience_traps)
  pairs  411 = master/hard_negatives_master.json (authored 90 preserved + 321 new)

Routing (models.lock role "auditor"): auditor + semantic checker = openai/gpt-5.5 via
OpenRouter (common.llm - generation ids + credit cost recorded per call,
reasoning_effort high, pinned seeds). Repairs/regenerations/severity grades =
claude-opus-5, max 3-4 workers.

TRANSPORT CHANGE (disclosed): the construction model stays claude-opus-5 at effort medium
but moved OFF the `claude -p` path, which bills against a Claude subscription, ONTO
OpenRouter (models.lock role "constructor" = anthropic/claude-opus-5) so that the
subscription's rate limits stayed free. The
note-repair rounds 0-3 ran on the subscription path before the switch; every construction
call after it is an OpenRouter credit call with a
Run manifest. The auditor role and all pinned auditor settings are UNCHANGED, and every
constructed artefact is re-audited by the cross-family auditor whatever built it.

COST LEVER (pre-registered): "1-seed first pass with 3-seed
confirmation only on flagged items". Screens run seed 11 (audit) / 21 (edit check);
any failing item/field triggers the remaining two seeds (12/13 / 22/23) and the
per-item/per-field MAJORITY of all three decides - identical majority semantics for
every flag, single-seed acceptance for first-pass passes (disclosed in the report).
HARD BUDGET STOP: cumulative OpenRouter spend >= BUDGET_STOP_USD -> save state, exit 3.

ADJUDICATION POLICY (clinical adjudication by a human is replaced by a conservative
automated default; every application is logged and queued for human review in
master/sitting_wa_items.json):
  - NEW strata (trapblind/primock/aci): every majority-confirmed flag is treated as a
    CONFIRMED defect -> REPAIR loop (max 3 rounds) -> full re-audit. Notes still failing
    after 3 rounds, where the protocol calls for manual repair by a person, are EXCLUDED
    from the freeze with their pairs, logged, queued.
  - AUTHORED stratum (pre-registered as preserved): flags are REPORT-ONLY, never
    repaired. A note is excluded (with its 3 pairs) only on a UNANIMOUS 3/3 flag in a
    checklist section (mc/mnc/traps - not the open-ended unsupported section, the most
    false-alarm-prone). Authored pairs failing the SEMANTIC majority on a load-bearing
    field (single_edit/type_match/is_error/truly_absent) are excluded; algorithmic-only
    or description_match-only failures are kept + flagged (the judges never see the
    injector description; the semantic single-edit standard is the load-bearing one).
  Under this policy the first-pass note defect rate is an auditor-majority rate, an
  UPPER BOUND on the human-confirmed rate; auditor precision against a human is n/a
  and reported as such.

State: results/w-a-state/wa_state.json (results/ is exempt from Run's dirty-tree check,
so chunked stages never block the next Run). Chunked, idempotent, per-unit resume;
all batches synchronous foreground ThreadPools. master/ outputs are written only by
`adjudicate` and `freeze` (commit between stages).

Usage (loop until each stage reports stable/complete):
  python3 benchmark/w_a_master.py status|selftest|check0
  python3 benchmark/w_a_master.py audit   [--chunk N] [--workers 6]   # OpenRouter, Run-wrapped
  python3 benchmark/w_a_master.py adjudicate                          # free
  python3 benchmark/w_a_master.py repair  [--workers 3]               # constructor, Run-wrapped
  python3 benchmark/w_a_master.py regen   [--workers 3]               # constructor, Run-wrapped
  python3 benchmark/w_a_master.py editcheck [--chunk N] [--workers 6] # OpenRouter, Run-wrapped
  python3 benchmark/w_a_master.py severity-backfill [--workers 3]     # constructor (authored 90)
  python3 benchmark/w_a_master.py freeze
Exit: 0 stage complete/stable | 2 pending work left (rerun) | 3 budget/rate-limit stop.
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import difflib, hashlib, json, os, random, re, sys, threading, time, unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from common import HERE, llm, Run, resolve_model, _sha256
import hard_negatives_master as hnm
from hard_negatives_master import (claude_json as plan_json, STOP as PLAN_STOP,
                                   STOP_REASON as PLAN_STOP_REASON, run_rubric_grade,
                                   TARGET_TRAP, TARGET_FACT, valid_errored)
from hard_negatives_balanced import INJECT, TYPES

SPEC = "ground-truth verification of the pair set"
MASTER = os.path.join(HERE, "master")
STATE_DIR = os.path.join(HERE, "results", "w-a-state")
STATE_FILE = os.path.join(STATE_DIR, "wa_state.json")
BUDGET_STOP_USD = 450.0   # hard stop for one full gate run: save state and exit
                          # cleanly rather than spend past it (resume re-buys nothing).
# ---- TRANSPORT CHANGE (disclosed) ------------------------------------------------------
# Repairs, regenerations and severity rubric grades used to run against a Claude
# subscription (`claude -p --model claude-opus-5 --effort medium`). They now run on the
# SAME model via OpenRouter (models.lock role "constructor" -> anthropic/claude-opus-5) so
# that the subscription's rate limits stayed free. Same model, different transport: the auditor role
# and all pinned auditor settings are untouched, and every constructed artefact is
# re-audited by the cross-family auditor regardless of how it was built.
CONSTRUCT_ROLE = "constructor"
CONSTRUCT_EFFORT = "medium"     # mirrors the subscription path's --effort medium
CONSTRUCT_TRANSPORT = "openrouter_api"
CONSTRUCT_MAX_WORKERS = 8       # the old cap of 4 existed to spare the subscription's rate
                                # limits; on OpenRouter credits that reason is gone (llm()
                                # already retries with backoff), so the cap doubles for
                                # throughput
AUDIT_SEEDS = (11, 12, 13)      # screen = 11; confirmation = 12, 13 (the cost lever)
EDIT_SEEDS = (21, 22, 23)       # screen = 21; confirmation = 22, 23 (the cost lever)
REASONING = "high"              # the pinned auditor setting in models.lock
MAX_TOKENS = 16000              # reasoning + JSON headroom; R1 guard raises on empty
SECTIONS = ("mc", "mnc", "traps", "unsupported")
MAX_REPAIR_ROUNDS = 3           # pre-registered repair-round cap
MAX_REGEN_ATTEMPTS = 3          # pre-registered regeneration cap
MAX_FALLBACK_ATTEMPTS = 1       # BUG FIX: an omit pair that exhausted its 3
                                # regenerations was routed to the programmatic omit fallback
                                # with NO attempt cap, so a fallback that fails verification
                                # was re-queued forever - and programmatic_omit is
                                # deterministic, so every retry rebuilt byte-identical text
                                # and re-spent semantic-check credits on the same verdict
                                # (observed: 27 pairs looping, 1-3 identical attempts each).
                                # The protocol gives the stubborn-omit fallback ONE shot;
                                # if that fails verification there is no further automated
                                # repair, so the pair is excluded and queued for human review
                                # exactly like a 3-regen add/change failure.
MAX_UNITS, MAX_REGIONS, MAX_CHANGED_WORDS = 2, 2, 30   # the algorithmic rule, v1 as frozen
# ---- ALGORITHMIC CONSTANTS REVISION v2 - the single revision the pre-registration
# allows, exercised ONCE before any semantic EDIT_CHECK call had run, applied by default
# and queued for human review, and recorded with reasons and both fail-counts in
# wa_gate_report.json. Evidence (free pre-check over all
# 411 pairs): v1 rejected 210/411 (51%) incl. 20/90 of the pre-registered preserved
# authored stratum (13 of them omit pairs - the study's headline error class), dominated
# by (a) one-block contiguous deletes of 3-5 units (whole Impression/safety-net sections
# removed as ONE atomic omission), (b) adds/omits realized as natural 1:1 line
# replacements failing the insert-only/delete-only word-op purity proxy, (c) 2-unit
# changes with 3 tiny word-regions. The load-bearing invariant is UNCHANGED: exactly ONE
# non-equal opcode block (one location; everything else verbatim) plus the 30-changed-
# word blast-radius cap. Dropped: word-op purity + MAX_REGIONS proxies (edit DIRECTION
# is verified by the semantic layer's type_match/is_error/truly_absent under 3-seed
# majority). Raised: pure insert (add) / pure delete (omit) contiguous block 2 -> 6
# units; replace-form spans 2 -> 3 units per side. v2 fail-count: 97/411, all genuinely
# multi-location (2-4 blocks: 62) or wholesale rewrites (35). NO FURTHER REVISION is
# permitted under the clause - v2 is final whatever it yields.
PURE_BLOCK_MAX_UNITS = 6     # v2: contiguous insert(add)/delete(omit) unit cap
REPLACE_MAX_UNITS = 3        # v2: replace-form per-side unit cap
ALGO_RULE_VERSION = "v2 (the single permitted revision; v1 counts recorded)"
SEM_FIELDS = ("single_edit", "type_match", "description_match", "is_error", "truly_absent")
LOAD_BEARING_FIELDS = ("single_edit", "type_match", "is_error", "truly_absent")
# ---- truly_absent CORRECTION -----------------------------------------------------------
# EDIT_CHECK item 5 used to require genuine absence AND a MUST_CONTAIN index for the removed
# fact, and sem_majority() failed the pair when no index came back. mc_idx is now recorded
# metadata that gates nothing. Diff + both prompt hashes: master/truly_absent_prompt_diff.md.
SUPERSEDE_REASON = (
    "truly_absent correction: EDIT_CHECK "
    "item 5 conjoined genuine absence with 'corresponds to a MUST_CONTAIN item', and a "
    "missing index failed the pair. mc_idx is now recorded metadata only. The 36 omit pairs "
    "excluded on truly_absent plus the 4 excluded on single_edit were re-verified under the "
    "corrected instrument at the same auditor, settings and 3-seed discipline")
CARVE_SEED = 20260728           # fixed seed for the dev-pool draw
CARVE_N = {"authored": 8, "primock": 14}
JSON_SUFFIX = "\nReturn ONLY the JSON object, no prose."

INPUT_FILES = ["master/ideal_notes_master.json", "master/hard_negatives_master.json",
               "master/fact_sheets_trapblind_core.json", "master/fact_sheets_primock_core.json",
               "master/fact_sheets_aci_core.json", "authored_scenarios.json",
               "ideal_notes_balanced.json", "hard_negatives_balanced.json"]

# ---------------------------------------------------------------- prompts (verbatim)
AUDIT_MC = """You are auditing a clinical note against its ground-truth fact sheet. The fact sheet is
authoritative; the transcript is provided only to resolve detail-level accuracy (doses,
durations, values, laterality, attribution). Audit ONLY the listed items - do not comment
on anything else in the note.

For EVERY item in MUST_CONTAIN decide:
- present: is the fact actually stated in the note (explicitly, or by unambiguous standard
  clinical shorthand)? Implied-but-never-stated counts as ABSENT.
- accurate: where present, are all its details correct per the fact sheet and transcript
  (values, doses, durations, laterality, who it is attributed to, and the stated degree of
  certainty - a hedged fact recorded as definite is INACCURATE)?
- evidence: the exact quote from the note carrying the fact (null if absent).

Return ONLY JSON:
{"items":[{"idx":<0-based index into MUST_CONTAIN>,"present":true|false,
"accurate":true|false|null,"evidence":"<quote or null>",
"reason":"<one line, only when present=false or accurate=false>"}]}

MUST_CONTAIN:
{mc}

NOTE:
{note}

TRANSCRIPT (for accuracy checks only):
{t}"""

AUDIT_MNC = """You are auditing a clinical note against a list of things it must NOT assert. Read each
item's wording carefully: many prohibit a CONFIRMED claim (e.g. a labelled diagnosis)
while the note may legitimately record a suspicion, a differential, or an explicitly
uncertain impression. "Asserted" means the note commits to the item as true or as the
clinician's conclusion.

For EVERY item in MUST_NOT_CONTAIN decide whether the note asserts it.

Return ONLY JSON:
{"items":[{"idx":<0-based index>,"asserted":true|false,"evidence":"<quote or null>",
"reason":"<one line, only when asserted=true>"}]}

MUST_NOT_CONTAIN:
{mnc}

NOTE:
{note}"""

AUDIT_TRAPS = """You are auditing how a clinical note handles known documentation traps from the source
consultation. Each trap comes with its correct handling. Mark handled=false if the note
commits the trap (e.g. hardens stated uncertainty into fact, drops a required hedge,
commits to a precise value the patient never gave, misattributes a finding) OR silently
omits material the correct handling requires recording.

For EVERY trap decide handled true/false.

Return ONLY JSON:
{"items":[{"idx":<0-based index>,"handled":true|false,"evidence":"<quote or null>",
"reason":"<one line, only when handled=false>"}]}

SALIENCE TRAPS (trap -> correct handling):
{traps}

NOTE:
{note}"""

AUDIT_UNSUPPORTED = """List every clinical assertion in the note that is NOT supported by the transcript:
fabricated findings, examinations never performed, values or durations never stated,
claims hardened beyond what was said. IGNORE: section headers, standard documentation
phrasing, reasonable clinical synonyms and shorthand for things that WERE said, and
safe-completion boilerplate. If everything is supported, return an empty list.

Return ONLY JSON:
{"unsupported":[{"assertion":"<quote from the note>","reason":"<one line>"}]}

NOTE:
{note}

TRANSCRIPT:
{t}"""

REPAIR = """Below is a clinical note that failed an audit against its ground-truth fact sheet, with
the exact audit failures. Produce a corrected note that fixes EVERY listed failure and
changes NOTHING else - keep the wording, structure, section order and length of the
original wherever it was not at fault. Return ONLY JSON {"note":"<corrected note>"}.

AUDIT FAILURES (fix all of these):
{failures}

FACT SHEET (authoritative):
MUST_CONTAIN:
{mc}
MUST_NOT_CONTAIN (never assert):
{mnc}
SALIENCE TRAPS (handle each correctly):
{traps}

ORIGINAL NOTE:
{note}

TRANSCRIPT:
{t}"""

EDIT_CHECK = """A clean clinical note had EXACTLY ONE error deliberately injected, producing the errored
note. You are given the diff between them, the intended error TYPE, and the injector's
own description of the edit. Verify:

1. single_edit: the diff is ONE atomic factual edit (one fact added, changed, or removed)
   - not multiple edits, not rephrasing or reformatting elsewhere.
2. type_match: the edit is of the recorded TYPE. add = asserts something absent from the
   clean note; change = alters an existing assertion; omit = removes an assertion and adds
   nothing.
3. description_match: the edit matches the injector's one-line description.
4. is_error (add/change only, else null): the injected or altered content is genuinely
   unsupported by or contradicted by the TRANSCRIPT - it must not happen to be true.
5. truly_absent (omit only, else null): the removed fact was present in the clean note and
   survives NOWHERE in the errored note (no partial or paraphrased residual mention).

Separately, for omit only, record mc_idx: the 0-based index of the MUST_CONTAIN item the
removed fact corresponds to, or -1 if it corresponds to none. mc_idx is METADATA - it must
NOT influence truly_absent or any other verdict above.

Return ONLY JSON:
{"single_edit":true|false,"type_match":true|false,"description_match":true|false,
"is_error":true|false|null,"truly_absent":true|false|null,"mc_idx":<int or null>,
"reason":"<one line covering any false>"}

TYPE: {typ}
INJECTOR DESCRIPTION: {change} | fact: {what}

DIFF (unified, clean -> errored):
{diff}

ERRORED NOTE (full, for the residual-mention check):
{errored}

MUST_CONTAIN:
{mc}

TRANSCRIPT:
{t}"""

REGEN_SUFFIX = """

A PREVIOUS ATTEMPT FAILED VERIFICATION for these reasons - avoid them:
{failure_reasons}
Hard constraints this time: your edit may touch at most {max_units} sentences; every
other sentence must be copied VERBATIM from the note - no rephrasing, reordering, or
reformatting anywhere outside the single edit."""

PROMPTS = {"AUDIT_MC": AUDIT_MC, "AUDIT_MNC": AUDIT_MNC, "AUDIT_TRAPS": AUDIT_TRAPS,
           "AUDIT_UNSUPPORTED": AUDIT_UNSUPPORTED, "REPAIR": REPAIR, "EDIT_CHECK": EDIT_CHECK,
           "REGEN_SUFFIX": REGEN_SUFFIX, "INJECT (verbatim import)": INJECT,
           "TYPE_add": TYPES["add"], "TYPE_change": TYPES["change"], "TYPE_omit": TYPES["omit"]}


def sha16(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


# ---------------------------------------------------------------- corpus
def load_corpus():
    """-> notes {key: {...}}, pairs [..] with per-consult views. key = 'stratum|id'."""
    notes = {}
    scen = {s["id"]: s for s in json.load(open(os.path.join(HERE, "authored_scenarios.json")))}
    for rec in json.load(open(os.path.join(HERE, "ideal_notes_balanced.json"))):
        s = scen[rec["id"]]
        fs = s["fact_sheet"]
        notes["authored|" + rec["id"]] = {
            "stratum": "authored", "id": rec["id"], "orig_note": rec["note"],
            "transcript": s["transcript"],
            "mc": list(fs["must_contain"]), "mc_orig_idx": list(range(len(fs["must_contain"]))),
            "mnc": list(fs["must_not_contain"]),
            "traps": [(t["trap"], t["correct_handling"]) for t in fs["salience_traps"]],
            "trap_meta": fs["salience_traps"]}
    master_notes = {(n["stratum"], n["id"]): n for n in
                    json.load(open(os.path.join(MASTER, "ideal_notes_master.json")))}
    for src in ("trapblind", "primock", "aci"):
        for rec in json.load(open(os.path.join(MASTER, f"fact_sheets_{src}_core.json"))):
            fs = rec["fact_sheet"]
            core = [(i, m) for i, m in enumerate(fs["must_contain"]) if m.get("scope") == "core"]
            mn = master_notes[(src, rec["id"])]
            notes[f"{src}|{rec['id']}"] = {
                "stratum": src, "id": rec["id"], "orig_note": mn["note"],
                "transcript": rec["transcript"],
                "mc": [m["fact"] for _, m in core], "mc_orig_idx": [i for i, _ in core],
                "mnc": [f"{m['assertion']} (why wrong: {m['why_wrong']})"
                        for m in fs["must_not_contain"]],
                "traps": [(t["trap"], t["correct_handling"]) for t in fs["salience_traps"]],
                "trap_meta": fs["salience_traps"]}
    pairs = json.load(open(os.path.join(MASTER, "hard_negatives_master.json")))
    return notes, pairs


def render_mc(note):
    return "\n".join(f"{i}. {t}" for i, t in enumerate(note["mc"]))


def render_mnc(note):
    return "\n".join(f"{i}. {t}" for i, t in enumerate(note["mnc"])) or "(none)"


def render_traps(note):
    return "\n".join(f"{i}. {trap} -> {correct}" for i, (trap, correct) in
                     enumerate(note["traps"])) or "(none)"


# ---------------------------------------------------------------- state
_state_lock = threading.Lock()


def load_state():
    os.makedirs(STATE_DIR, exist_ok=True)
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {"created": now_utc(),
            "spend": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "reasoning_tokens": 0, "cached_tokens": 0, "cost_usd": 0.0,
                      "by_stage": {}, "plan_calls": 0},
            "audits": {}, "notes": {}, "pairs": {}, "severity_backfill": {},
            "run_ids": [], "lever": "1-seed screen + 3-seed majority confirmation on flags"}


def save_state(state):
    with _state_lock:
        tmp = STATE_FILE + ".tmp"
        json.dump(state, open(tmp, "w"), ensure_ascii=False)
        os.replace(tmp, STATE_FILE)


BUDGET_HIT = threading.Event()


def add_spend(state, stage, meta):
    with _state_lock:
        sp = state["spend"]
        u = meta.get("usage", {})
        sp["calls"] += len(meta.get("requests") or [1])
        for f in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens"):
            sp[f] += u.get(f, 0)
        sp["cost_usd"] = round(sp["cost_usd"] + (meta.get("cost_usd_reported") or 0), 6)
        st = sp["by_stage"].setdefault(stage, {"calls": 0, "cost_usd": 0.0})
        st["calls"] += len(meta.get("requests") or [1])
        st["cost_usd"] = round(st["cost_usd"] + (meta.get("cost_usd_reported") or 0), 6)
        if sp["cost_usd"] >= BUDGET_STOP_USD:
            BUDGET_HIT.set()


def budget_ok(state):
    return state["spend"]["cost_usd"] < BUDGET_STOP_USD and not BUDGET_HIT.is_set()


# ---------------------------------------------------------------- auditor call
class LockedRun(Run):
    """Run whose log_call is thread-safe (per_call.jsonl.gz writes + counters)."""
    _lock = threading.Lock()

    def log_call(self, *a, **kw):
        with self._lock:
            return super().log_call(*a, **kw)


def parse_json_blob(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def auditor_json(state, stage, prompt, seed):
    """One auditor call (retry protocol: retry, then rerun with a JSON-only suffix).
    Returns (obj, meta_summary) or (None, error)."""
    last_err = None
    for i, p in enumerate((prompt, prompt, prompt + JSON_SUFFIX)):
        if not budget_ok(state):
            return None, {"error": "budget_stop"}
        try:
            texts, meta = llm(p, role="auditor", temperature=1.0, k=1, seed=seed,
                              reasoning_effort=REASONING, max_tokens=MAX_TOKENS,
                              timeout=330, max_retries=1)
        except Exception as ex:
            last_err = str(ex)[:300]
            continue
        add_spend(state, stage, meta)
        obj = parse_json_blob(texts[0])
        if obj is not None:
            return obj, {"generation_ids": meta.get("generation_ids"),
                         "returned_models": meta.get("returned_models"),
                         "cost": meta.get("cost_usd_reported"), "seed": seed,
                         "json_retry": i}
        last_err = "unparseable JSON"
    return None, {"error": last_err or "failed"}


# ---------------------------------------------------------------- constructor call
# (transport change - see CONSTRUCT_* at the top of the file)
_CONSTRUCT_CTX = {"state": None, "stage": "severity"}


def bump_construct(state):
    state["spend"]["construct_calls"] = state["spend"].get("construct_calls", 0) + 1


def construct_json(state, stage, prompt, timeout=420, max_tokens=MAX_TOKENS):
    """One construction call (repair / regen / severity rubric grade) on the SAME model the
    subscription path used (claude-opus-5) but over OpenRouter - models.lock role 'constructor'.
    Returns the parsed JSON object, or None if it could not be produced. Spend is recorded
    against `stage` exactly like every other credit call; inside a Run context the per-call
    generation ids + usage land in that run's manifest."""
    bump_construct(state)
    for p in (prompt, prompt + JSON_SUFFIX):
        if not budget_ok(state):
            return None
        try:
            texts, meta = llm(p, role=CONSTRUCT_ROLE, temperature=1.0, k=1, seed=None,
                              reasoning_effort=CONSTRUCT_EFFORT, max_tokens=max_tokens,
                              timeout=timeout, max_retries=2)
        except Exception as ex:
            print(f"  constructor call error: {str(ex)[:200]}", flush=True)
            continue
        add_spend(state, stage, meta)
        obj = parse_json_blob(texts[0])
        if obj is not None:
            return obj
    return None


def _construct_claude_json(prompt, timeout=420, model=None, effort=None, retries=0):
    """Drop-in for hard_negatives_master.claude_json so run_rubric_grade (severity backfill)
    also leaves the subscription path. Signature-compatible; state/stage come from _CONSTRUCT_CTX."""
    st = _CONSTRUCT_CTX["state"]
    if st is None:
        raise RuntimeError("_CONSTRUCT_CTX state not set before a rubric-grade call")
    return construct_json(st, _CONSTRUCT_CTX["stage"], prompt, timeout=timeout)


hnm.claude_json = _construct_claude_json


# ---------------------------------------------------------------- audit sections
def section_prompt(note, section, text):
    if section == "mc":
        return (AUDIT_MC.replace("{mc}", render_mc(note)).replace("{note}", text)
                .replace("{t}", note["transcript"][:40000]))
    if section == "mnc":
        return AUDIT_MNC.replace("{mnc}", render_mnc(note)).replace("{note}", text)
    if section == "traps":
        return AUDIT_TRAPS.replace("{traps}", render_traps(note)).replace("{note}", text)
    return (AUDIT_UNSUPPORTED.replace("{note}", text)
            .replace("{t}", note["transcript"][:40000]))


def section_n_items(note, section):
    return {"mc": len(note["mc"]), "mnc": len(note["mnc"]), "traps": len(note["traps"]),
            "unsupported": None}[section]


def parse_section(obj, note, section):
    """-> {'fails': {idx: {...}} or [assertion dicts], 'coverage': ok} or None on shape fail."""
    if section == "unsupported":
        lst = obj.get("unsupported")
        if not isinstance(lst, list):
            return None
        out = []
        for it in lst:
            if isinstance(it, dict) and it.get("assertion"):
                out.append({"assertion": str(it["assertion"]).strip(),
                            "reason": str(it.get("reason") or "").strip()})
        return {"fails": out, "n": len(out)}
    items = obj.get("items")
    if not isinstance(items, list):
        return None
    n = section_n_items(note, section)
    seen, fails = set(), {}
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            idx = int(it.get("idx"))
        except (TypeError, ValueError):
            continue
        if not (0 <= idx < n) or idx in seen:
            continue
        seen.add(idx)
        if section == "mc":
            present = bool(it.get("present"))
            accurate = it.get("accurate")
            if not present:
                fails[idx] = {"kind": "mc_missing", "reason": it.get("reason"),
                              "evidence": it.get("evidence")}
            elif accurate is False:
                fails[idx] = {"kind": "mc_inaccurate", "reason": it.get("reason"),
                              "evidence": it.get("evidence")}
        elif section == "mnc":
            if bool(it.get("asserted")):
                fails[idx] = {"kind": "mnc_asserted", "reason": it.get("reason"),
                              "evidence": it.get("evidence")}
        elif section == "traps":
            if it.get("handled") is False:
                fails[idx] = {"kind": "trap_mishandled", "reason": it.get("reason"),
                              "evidence": it.get("evidence")}
    if len(seen) < n:
        return {"fails": fails, "coverage_missing": sorted(set(range(n)) - seen)}
    return {"fails": fails}


def _norm_assertion(a):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", a.lower())).strip()


def cluster_unsupported(runs):
    """runs: {seed: [ {assertion, reason} ]} -> clusters [{assertion, seeds, reasons}]."""
    clusters = []
    for seed, lst in runs.items():
        for it in lst:
            key = set(_norm_assertion(it["assertion"]).split())
            if not key:
                continue
            best, best_j = None, 0.0
            for c in clusters:
                j = len(key & c["key"]) / max(len(key | c["key"]), 1)
                if j > best_j:
                    best, best_j = c, j
            if best is not None and best_j >= 0.5:
                best["seeds"].add(seed)
                best["reasons"].append(it.get("reason"))
                best["key"] |= key
            else:
                clusters.append({"assertion": it["assertion"], "key": key,
                                 "seeds": {seed}, "reasons": [it.get("reason")]})
    return clusters


def audit_key(key, note_sha, section):
    return f"{key}|{note_sha}|{section}"


def note_entry(state, corpus_notes, key):
    e = state["notes"].get(key)
    if e is None:
        e = state["notes"][key] = {
            "stratum": key.split("|")[0], "id": key.split("|", 1)[1],
            "text": corpus_notes[key]["orig_note"], "round": 0, "repaired": False,
            "status": "auditing", "history": []}
    return e


def section_state(state, akey):
    e = state["audits"].get(akey)
    if e is None:
        e = state["audits"][akey] = {"runs": {}, "done": False, "errors": []}
    return e


def run_audit_stage(state, corpus_notes, chunk, workers):
    """Screens + confirmations for the CURRENT text of every active note. Returns pending count."""
    jobs = []   # (key, section, seed)
    for key in sorted(corpus_notes):
        ne = note_entry(state, corpus_notes, key)
        if ne["status"] in ("excluded",):
            continue
        nsha = sha16(ne["text"])
        for section in SECTIONS:
            akey = audit_key(key, nsha, section)
            se = section_state(state, akey)
            if se["done"]:
                continue
            s0 = str(AUDIT_SEEDS[0])
            if s0 not in se["runs"]:
                jobs.append((key, section, AUDIT_SEEDS[0]))
                continue
            r0 = se["runs"][s0]
            has_fail = bool(r0["fails"])
            if not has_fail:
                se["done"] = True   # lever: clean screen = accepted at 1 seed
                continue
            need = [s for s in AUDIT_SEEDS[1:] if str(s) not in se["runs"]]
            if need:
                jobs.extend((key, section, s) for s in need)
            else:
                se["done"] = True
    save_state(state)
    pending_total = len(jobs)
    if not jobs:
        return 0
    if chunk:
        jobs = jobs[:chunk]
    done_ct = {"n": 0}

    def one(job):
        key, section, seed = job
        if not budget_ok(state) or past_deadline():
            return
        note = corpus_notes[key]
        ne = state["notes"][key]
        nsha = sha16(ne["text"])
        akey = audit_key(key, nsha, section)
        obj, meta = auditor_json(state, "audit", section_prompt(note, section, ne["text"]), seed)
        with _state_lock:
            se = state["audits"][akey]
            if obj is None:
                se["errors"].append({"seed": seed, **meta})
                return
            parsed = parse_section(obj, note, section)
            if parsed is None:
                se["errors"].append({"seed": seed, "error": "bad shape", **meta})
                return
            se["runs"][str(seed)] = {"fails": parsed["fails"], "meta": meta,
                                     **({"coverage_missing": parsed["coverage_missing"]}
                                        if parsed.get("coverage_missing") else {}),
                                     **({"n": parsed["n"]} if "n" in parsed else {})}
            done_ct["n"] += 1
        if done_ct["n"] % 25 == 0:
            save_state(state)
            print(f"  ... {done_ct['n']}/{len(jobs)} calls, spend ${state['spend']['cost_usd']:.2f}",
                  flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, jobs))
    save_state(state)
    print(f"audit: ran {done_ct['n']}/{len(jobs)} calls (of {pending_total} pending) | "
          f"spend ${state['spend']['cost_usd']:.2f}", flush=True)
    return pending_total - done_ct["n"] + max(0, pending_total - len(jobs))


# ---------------------------------------------------------------- flag aggregation + adjudication
def majority_flags(state, corpus_notes, key):
    """Flags for the CURRENT version of a note (post-confirmation majority). None if not done."""
    ne = state["notes"][key]
    note = corpus_notes[key]
    nsha = sha16(ne["text"])
    flags = []
    for section in SECTIONS:
        se = state["audits"].get(audit_key(key, nsha, section))
        if se is None or not se["done"]:
            return None
        runs = se["runs"]
        n_runs = len(runs)
        if section == "unsupported":
            cl = cluster_unsupported({s: r["fails"] for s, r in runs.items()})
            for c in cl:
                votes = sorted(c["seeds"])
                if len(votes) >= 2:
                    flags.append({"section": section, "idx": None, "kind": "unsupported_assertion",
                                  "item": c["assertion"], "votes": votes, "n_fail": len(votes),
                                  "n_runs": n_runs, "unanimous": len(votes) == n_runs == 3,
                                  "reasons": [r for r in c["reasons"] if r]})
            continue
        counts = defaultdict(list)
        for s, r in runs.items():
            for idx, f in r["fails"].items():
                counts[int(idx)].append((s, f))
        for idx, votes in sorted(counts.items()):
            if len(votes) >= 2:   # majority of 3 (screen fail + >=1 confirmation)
                item_text = {"mc": note["mc"], "mnc": note["mnc"]}.get(section)
                item_text = (item_text[idx] if item_text else
                             f"{note['traps'][idx][0]} -> {note['traps'][idx][1]}")
                flags.append({"section": section, "idx": idx, "kind": votes[0][1]["kind"],
                              "item": item_text, "votes": sorted(s for s, _ in votes),
                              "n_fail": len(votes), "n_runs": n_runs,
                              "unanimous": len(votes) == n_runs == 3,
                              "reasons": [f.get("reason") for _, f in votes if f.get("reason")],
                              "evidence": next((f.get("evidence") for _, f in votes
                                                if f.get("evidence")), None)})
    return flags


def adjudicate_stage(state, corpus_notes):
    """Apply the automated adjudication policy to every note whose current audit is done."""
    changed, pending = 0, 0
    for key in sorted(corpus_notes):
        ne = state["notes"][key]
        if ne["status"] not in ("auditing",):
            continue
        flags = majority_flags(state, corpus_notes, key)
        if flags is None:
            pending += 1
            continue
        stratum = ne["stratum"]
        ne.setdefault("flag_history", []).append(
            {"round": ne["round"], "n_flags": len(flags), "flags": flags})
        if not flags:
            ne["status"] = "clean"
            changed += 1
            continue
        if stratum == "authored":
            hard = [f for f in flags if f["unanimous"] and f["section"] in ("mc", "mnc", "traps")]
            if hard:
                ne["status"] = "excluded"
                ne["excluded_reason"] = ("authored note: unanimous 3/3 cross-family checklist "
                                         "flag(s) - genuinely broken per policy; preserved "
                                         "stratum is never repaired: "
                                         + "; ".join(f"{f['section']}[{f['idx']}]" for f in hard))
            else:
                ne["status"] = "clean"          # kept; flags REPORTED, queued for review
                ne["kept_flagged"] = True
            changed += 1
            continue
        # new strata: majority flags = confirmed defects -> repair (max 3 rounds)
        if ne["round"] >= MAX_REPAIR_ROUNDS:
            ne["status"] = "excluded"
            ne["excluded_reason"] = (f"unrepaired after {MAX_REPAIR_ROUNDS} repair rounds "
                                     "(the protocol reserves manual repair for a human - "
                                     "excluded by the automated default, queued for review)")
        else:
            ne["status"] = "repair_pending"
        changed += 1
    save_state(state)
    return changed, pending


# ---------------------------------------------------------------- repair
def render_failures(flags):
    lines = []
    for i, f in enumerate(flags):
        loc = f"{f['section']}" + (f" item {f['idx']}" if f["idx"] is not None else "")
        why = "; ".join(dict.fromkeys(f["reasons"]))[:400] if f["reasons"] else f["kind"]
        ev = f" | note evidence: {f.get('evidence')}" if f.get("evidence") else ""
        lines.append(f"{i + 1}. [{loc}] {f['kind']}: {f['item']} -> {why}{ev}")
    return "\n".join(lines)


def repair_stage(state, corpus_notes, workers, chunk=None):
    todo = sorted(k for k, ne in state["notes"].items() if ne["status"] == "repair_pending")
    pending = len(todo)
    if not todo:
        return 0
    if chunk:
        todo = todo[:chunk]
    print(f"repair: {len(todo)} of {pending} pending notes this invocation "
          f"({CONSTRUCT_TRANSPORT}, {workers} workers)", flush=True)

    def one(key):
        if not budget_ok(state) or past_deadline():
            return
        ne = state["notes"][key]
        note = corpus_notes[key]
        flags = ne["flag_history"][-1]["flags"]
        prompt = (REPAIR.replace("{failures}", render_failures(flags))
                  .replace("{mc}", render_mc(note)).replace("{mnc}", render_mnc(note))
                  .replace("{traps}", render_traps(note)).replace("{note}", ne["text"])
                  .replace("{t}", note["transcript"][:40000]))
        res = construct_json(state, "repair", prompt, timeout=540)
        if not (isinstance(res, dict) and isinstance(res.get("note"), str)
                and len(res["note"].strip()) > 150):
            print(f"  {key}: repair call FAILED (rerun to retry)", flush=True)
            return
        with _state_lock:
            ne["history"].append({"round": ne["round"], "old_sha": sha16(ne["text"]),
                                  "flags_fixed": len(flags), "utc": now_utc()})
            ne["text"] = res["note"].strip()
            ne["round"] += 1
            ne["repaired"] = True
            ne["status"] = "auditing"     # new version -> fresh audit entries pend automatically
        save_state(state)
        print(f"  {key}: repaired (round {ne['round']})", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, todo))
    save_state(state)
    if BUDGET_HIT.is_set():
        print(f"BUDGET STOP: ${state['spend']['cost_usd']:.2f} >= ${BUDGET_STOP_USD}", flush=True)
        sys.exit(3)
    return len(todo)


# ---------------------------------------------------------------- algorithmic single-edit check
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def units_of(text):
    """Unit normalisation: NFC, per-line strip, collapse whitespace runs; split lines; sentence-split
    within lines; a line ending ':' is its own unit. Returns normalized unit strings."""
    text = unicodedata.normalize("NFC", text or "")
    units = []
    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line.strip())
        if not line:
            continue
        if line.endswith(":"):
            units.append(line)
            continue
        units.extend(p.strip() for p in SENT_SPLIT.split(line) if p.strip())
    return units


def word_diff_stats(a_text, b_text):
    aw, bw = a_text.split(), b_text.split()
    ops = [op for op in difflib.SequenceMatcher(None, aw, bw).get_opcodes()
           if op[0] != "equal"]
    changed = sum((op[2] - op[1]) + (op[4] - op[3]) for op in ops)
    kinds = {op[0] for op in ops}
    return {"regions": len(ops), "changed_words": changed, "kinds": sorted(kinds)}


def algo_check(clean, errored, typ):
    """The algorithmic acceptance rule at ALGO_RULE_VERSION (v2 - see the revision block
    above for the v1 text, evidence, and what stayed frozen). One non-equal unit block; pure
    insert/delete <= PURE_BLOCK_MAX_UNITS (type-gated); replace-form <= REPLACE_MAX_UNITS
    per side and <= MAX_CHANGED_WORDS word tokens; change must be replace-form."""
    uc, ue = units_of(clean), units_of(errored)
    ops = difflib.SequenceMatcher(None, uc, ue).get_opcodes()
    ne = [op for op in ops if op[0] != "equal"]
    v = {"pass": False, "rule": ALGO_RULE_VERSION, "n_blocks": len(ne),
         "units_clean": len(uc), "units_err": len(ue)}
    if len(ne) != 1:
        v["reason"] = (f"{len(ne)} non-equal opcode blocks at unit level "
                       "(rule requires exactly 1)")
        if len(ne) == 0:
            v["reason"] = "no difference between clean and errored at unit level"
        return v
    tag, i1, i2, j1, j2 = ne[0]
    v.update({"tag": tag, "clean_span": [i1, i2], "err_span": [j1, j2]})
    du, iu = i2 - i1, j2 - j1
    if tag == "insert":
        if typ == "add" and iu <= PURE_BLOCK_MAX_UNITS:
            v["pass"] = True
        else:
            v["reason"] = f"{typ}: got insert of {iu} units (add-only, max {PURE_BLOCK_MAX_UNITS})"
    elif tag == "delete":
        if typ == "omit" and du <= PURE_BLOCK_MAX_UNITS:
            v["pass"] = True
        else:
            v["reason"] = f"{typ}: got delete of {du} units (omit-only, max {PURE_BLOCK_MAX_UNITS})"
    else:   # replace-form, any type; direction verified by the semantic layer
        w = word_diff_stats(" ".join(uc[i1:i2]), " ".join(ue[j1:j2]))
        v["word_diff"] = w
        if du <= REPLACE_MAX_UNITS and iu <= REPLACE_MAX_UNITS \
                and w["changed_words"] <= MAX_CHANGED_WORDS:
            v["pass"] = True
        else:
            v["reason"] = (f"replace {du}->{iu} units / {w['changed_words']} changed words "
                           f"(max {REPLACE_MAX_UNITS} per side / {MAX_CHANGED_WORDS} words)")
    return v


def algo_check_v1(clean, errored, typ):
    """The ORIGINAL rule exactly as frozen at pre-registration - kept verbatim so the gate
    report can recount both versions over the as-built pairs (audit trail for the v2
    revision). Never used for gating once v2 was adopted."""
    uc, ue = units_of(clean), units_of(errored)
    ne = [op for op in difflib.SequenceMatcher(None, uc, ue).get_opcodes()
          if op[0] != "equal"]
    if len(ne) != 1:
        return False
    tag, i1, i2, j1, j2 = ne[0]
    du, iu = i2 - i1, j2 - j1
    if typ == "add":
        if tag == "insert" and iu <= MAX_UNITS:
            return True
        if tag == "replace" and du == 1 and iu == 1:
            return word_diff_stats(uc[i1], ue[j1])["kinds"] == ["insert"]
        return False
    if typ == "omit":
        if tag == "delete" and du <= MAX_UNITS:
            return True
        if tag == "replace" and du == 1 and iu == 1:
            return word_diff_stats(uc[i1], ue[j1])["kinds"] == ["delete"]
        return False
    if typ == "change":
        if tag == "replace" and du <= MAX_UNITS and iu <= MAX_UNITS:
            w = word_diff_stats(" ".join(uc[i1:i2]), " ".join(ue[j1:j2]))
            return w["regions"] <= MAX_REGIONS and w["changed_words"] <= MAX_CHANGED_WORDS
        return False
    return False


def unit_diff_text(clean, errored):
    return "\n".join(difflib.unified_diff(units_of(clean), units_of(errored),
                                          fromfile="clean", tofile="errored",
                                          lineterm="", n=2))


# ---------------------------------------------------------------- semantic check
def edit_prompt(pair, note):
    what = pair.get("what")
    if not isinstance(what, str):
        what = json.dumps(what, ensure_ascii=False)
    return (EDIT_CHECK.replace("{typ}", pair["type"])
            .replace("{change}", str(pair.get("change") or ""))
            .replace("{what}", what or "")
            .replace("{diff}", unit_diff_text(pair["clean"], pair["errored"]))
            .replace("{errored}", pair["errored"])
            .replace("{mc}", render_mc(note))
            .replace("{t}", note["transcript"][:40000]))


def parse_edit(obj, typ):
    out = {}
    for f in SEM_FIELDS:
        v = obj.get(f)
        if f in ("is_error",) and typ == "omit":
            out[f] = None
        elif f in ("truly_absent",) and typ != "omit":
            out[f] = None
        elif isinstance(v, bool):
            out[f] = v
        else:
            return None
    mi = obj.get("mc_idx")
    out["mc_idx"] = int(mi) if isinstance(mi, (int, float)) and typ == "omit" else None
    out["reason"] = obj.get("reason")
    return out


def sem_majority(runs, typ):
    """runs {seed: parsed} -> (verdict {field: bool|None}, mc_idx, pass, fail_fields)."""
    verdict, fails = {}, []
    n = len(runs)
    for f in SEM_FIELDS:
        vals = [r[f] for r in runs.values() if r[f] is not None]
        if not vals:
            verdict[f] = None
            continue
        true_n = sum(1 for v in vals if v)
        maj = true_n > len(vals) / 2 if n > 1 else bool(vals[0])
        verdict[f] = maj
        if not maj:
            fails.append(f)
    mc_idx = None
    if typ == "omit":
        # CORRECTION: mc_idx is recorded
        # metadata ONLY. It no longer gates `ok`, and the branch on truly_absent that used
        # to precede this (behaviourally identical in both arms) is collapsed, so nothing
        # about mc_idx depends on a verdict or vice versa.
        cand = [r["mc_idx"] for r in runs.values()
                if r.get("mc_idx") is not None and r["mc_idx"] >= 0]
        if cand:
            mc_idx = Counter(cand).most_common(1)[0][0]
    ok = all(verdict[f] is not False for f in SEM_FIELDS)
    return verdict, mc_idx, ok, fails


def pair_entry(state, pair):
    e = state["pairs"].get(pair["pair_id"])
    if e is None:
        e = state["pairs"][pair["pair_id"]] = {
            "stratum": pair["stratum"], "id": pair["id"], "type": pair["type"],
            "status": "pending", "attempts": [], "current": {
                "kind": "original", "clean_sha": sha16(pair["clean"]),
                "errored": pair["errored"], "change": pair.get("change"),
                "what": pair.get("what")}}
    return e


def current_pair_view(state, corpus_notes, pair):
    """The pair as it should be verified NOW: clean = current note text."""
    key = f"{pair['stratum']}|{pair['id']}"
    ne = state["notes"].get(key)
    clean = ne["text"] if ne else pair["clean"]
    pe = state["pairs"][pair["pair_id"]]
    cur = pe["current"]
    return {**pair, "clean": clean, "errored": cur["errored"],
            "change": cur.get("change", pair.get("change")),
            "what": cur.get("what", pair.get("what"))}


def editcheck_stage(state, corpus_notes, pairs, chunk, workers):
    """Algorithmic (free) + semantic screens/confirmations on every active pair whose
    clean note is settled. Marks failures for regen/fallback/exclusion. Returns pending."""
    by_id = {p["pair_id"]: p for p in pairs}
    jobs = []
    for pair in pairs:
        pe = pair_entry(state, pair)
        key = f"{pair['stratum']}|{pair['id']}"
        ne = state["notes"].get(key)
        if ne is None or ne["status"] == "auditing" or ne["status"] == "repair_pending":
            continue    # note not settled yet
        if ne["status"] == "excluded":
            if pe["status"] != "excluded":
                pe["status"], pe["excluded_reason"] = "excluded", "note_excluded: " + \
                    (ne.get("excluded_reason") or "")
            continue
        if pe["status"] in ("passed", "excluded", "regen_pending", "fallback_pending"):
            continue
        cur = pe["current"]
        if cur["clean_sha"] != sha16(ne["text"]):
            if pair["stratum"] == "authored":
                raise AssertionError(f"{pair['pair_id']}: authored clean drifted - pipeline bug")
            pe["status"] = "regen_pending"   # post-repair rebuild: this becomes the pair's
            pe["rebuild"] = True             # FIRST-PASS attempt in the failure-rate metric
            continue
        # 1) algorithmic (free, once per attempt)
        if "algo" not in cur:
            cur["algo"] = algo_check(ne["text"], cur["errored"], pair["type"])
        # 2) semantic screens/confirmations
        sem = cur.setdefault("sem_runs", {})
        if pair["stratum"] != "authored" and not cur["algo"]["pass"]:
            # semantic runs on algorithmic passes only (regen on an algorithmic fail)
            pe["status"] = "sem_done"
            continue
        s0 = str(EDIT_SEEDS[0])
        if s0 not in sem:
            jobs.append(pair["pair_id"])
            continue
        p0 = sem[s0]
        _, _, ok0, _ = sem_majority({s0: p0}, pair["type"])
        if ok0:
            pe["status"] = "sem_done"
            continue
        need = [s for s in EDIT_SEEDS[1:] if str(s) not in sem]
        if need:
            jobs.append(pair["pair_id"])
        else:
            pe["status"] = "sem_done"
    save_state(state)
    pending_total = len(jobs)
    if chunk:
        jobs = jobs[:chunk]
    done_ct = {"n": 0}

    def one(pid):
        if not budget_ok(state) or past_deadline():
            return
        pair = by_id[pid]
        pe = state["pairs"][pid]
        cur = pe["current"]
        view = current_pair_view(state, corpus_notes, pair)
        note = corpus_notes[f"{pair['stratum']}|{pair['id']}"]
        sem = cur["sem_runs"]
        s0 = str(EDIT_SEEDS[0])
        seeds = [EDIT_SEEDS[0]] if s0 not in sem else \
            [s for s in EDIT_SEEDS[1:] if str(s) not in sem]
        prompt = edit_prompt(view, note)
        for seed in seeds:
            obj, meta = auditor_json(state, "editcheck", prompt, seed)
            if obj is None:
                with _state_lock:
                    cur.setdefault("sem_errors", []).append({"seed": seed, **meta})
                continue
            parsed = parse_edit(obj, pair["type"])
            with _state_lock:
                if parsed is None:
                    cur.setdefault("sem_errors", []).append(
                        {"seed": seed, "error": "bad shape", "gen": meta.get("generation_ids")})
                else:
                    parsed["gen"] = meta.get("generation_ids")
                    sem[str(seed)] = parsed
            # after screen: stop early if screen passed
            if seed == EDIT_SEEDS[0] and parsed is not None:
                _, _, ok0, _ = sem_majority({s0: parsed}, pair["type"])
                if ok0:
                    break
        with _state_lock:
            done_ct["n"] += 1
        if done_ct["n"] % 20 == 0:
            save_state(state)
            print(f"  ... {done_ct['n']}/{len(jobs)} pairs, spend "
                  f"${state['spend']['cost_usd']:.2f}", flush=True)

    if jobs:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, jobs))
    save_state(state)
    # resolve sem_done pairs into passed / failure routing
    resolved = 0
    for pair in pairs:
        pe = state["pairs"].get(pair["pair_id"])
        if pe is None or pe["status"] != "sem_done":
            continue
        cur = pe["current"]
        typ = pair["type"]
        algo_ok = cur.get("algo", {}).get("pass", False)
        sem_runs = {s: p for s, p in cur.get("sem_runs", {}).items()}
        verdict, mc_idx, sem_ok, fails = (None, None, False, ["no_sem_runs"])
        if sem_runs:
            verdict, mc_idx, sem_ok, fails = sem_majority(sem_runs, typ)
        attempt_rec = {"kind": cur.get("kind", "original"), "algo": cur.get("algo"),
                       "sem_verdict": verdict, "sem_fails": fails, "mc_idx": mc_idx,
                       "n_sem_seeds": len(sem_runs), "utc": now_utc()}
        pe["attempts"].append(attempt_rec)
        if pair["stratum"] == "authored":
            lb_fail = [f for f in fails if f in LOAD_BEARING_FIELDS]
            if sem_runs and not lb_fail:
                pe["status"] = "passed"
                pe["mc_idx"] = mc_idx
                pe["waivers"] = ([] if algo_ok else ["algorithmic_nonconformance"]) + \
                    (["description_match"] if verdict and verdict.get("description_match")
                     is False else [])
            else:
                pe["status"] = "excluded"
                pe["excluded_reason"] = ("authored pair failed semantic majority on "
                                        f"load-bearing field(s) {lb_fail} - genuinely broken; "
                                        "preserved stratum is never rewritten")
        else:
            if algo_ok and sem_ok:
                pe["status"] = "passed"
                pe["mc_idx"] = mc_idx
            else:
                n_regens = sum(1 for a in pe["attempts"] if a["kind"] == "regen")
                if n_regens < MAX_REGEN_ATTEMPTS:
                    pe["status"] = "regen_pending"
                    parts = []
                    if not algo_ok:
                        parts.append("algorithmic: " + (cur.get("algo", {}).get("reason")
                                                        or "not a single localized edit"))
                    if not sem_ok:
                        sem_reasons = "; ".join(str(r.get("reason")) for r in
                                                sem_runs.values() if r.get("reason"))
                        parts.append("semantic: fields failed = " + ", ".join(fails)
                                     + ("; " + sem_reasons if sem_reasons else ""))
                    pe["fail_reasons"] = "; ".join(parts)[:800]
                elif typ == "omit" and sum(1 for a in pe["attempts"]
                                           if a["kind"] == "fallback") < MAX_FALLBACK_ATTEMPTS:
                    pe["status"] = "fallback_pending"
                elif typ == "omit":
                    pe["status"] = "excluded"
                    pe["excluded_reason"] = (
                        f"omit pair failed after {MAX_REGEN_ATTEMPTS} regenerations AND its "
                        f"{MAX_FALLBACK_ATTEMPTS} programmatic-deletion fallback "
                        f"(last semantic failure: {', '.join(fails) or 'n/a'}); the fallback is "
                        "deterministic, so retrying rebuilds identical text - excluded by the "
                        "automated default, queued for review (the protocol reserves the "
                        "hand-written edit for a human)")
                else:
                    pe["status"] = "excluded"
                    pe["excluded_reason"] = (f"{typ} pair failed after {MAX_REGEN_ATTEMPTS} "
                                             "regenerations (the protocol's fallback is a "
                                             "hand-written edit - excluded by the automated "
                                             "default, queued for review)")
        resolved += 1
    save_state(state)
    left = pending_total - len(jobs) + sum(
        1 for p in pairs if state["pairs"].get(p["pair_id"], {}).get("status")
        in ("pending", "sem_done"))
    print(f"editcheck: {done_ct['n']} pair-batches called, {resolved} resolved | "
          f"spend ${state['spend']['cost_usd']:.2f}", flush=True)
    return left


# ---------------------------------------------------------------- regen + fallback
def target_instr(pair):
    instr = TYPES[pair["type"]]
    t = pair.get("target")
    if t and t.get("kind") == "trap":
        instr += TARGET_TRAP.format(trap=t["trap"], correct=t["correct_handling"])
    elif t and t.get("kind") == "must_contain":
        instr += TARGET_FACT.format(fact=t["fact"])
    return instr


def regen_stage(state, corpus_notes, pairs, workers, chunk=None):
    by_id = {p["pair_id"]: p for p in pairs}
    todo = sorted(pid for pid, pe in state["pairs"].items()
                  if pe["status"] in ("regen_pending", "fallback_pending"))
    pending = len(todo)
    if not todo:
        return 0
    if chunk:
        todo = todo[:chunk]
    print(f"regen: {len(todo)} of {pending} pending pairs this invocation "
          f"({CONSTRUCT_TRANSPORT}, {workers} workers)", flush=True)

    def one(pid):
        if not budget_ok(state) or past_deadline():
            return
        pair = by_id[pid]
        pe = state["pairs"][pid]
        key = f"{pair['stratum']}|{pair['id']}"
        ne = state["notes"][key]
        clean = ne["text"]
        if pe["status"] == "fallback_pending":
            new = programmatic_omit(clean, pair, pe)
            with _state_lock:
                if new is None:
                    pe["status"] = "excluded"
                    pe["excluded_reason"] = ("omit fallback: target fact spans non-adjacent "
                                             "units - programmatic single-block deletion "
                                             "impossible; excluded (queued)")
                else:
                    pe["current"] = {"kind": "fallback", "clean_sha": sha16(clean),
                                     "errored": new["errored"], "change": new["change"],
                                     "what": new["what"]}
                    pe["status"] = "pending"
            save_state(state)
            return
        rebuild = bool(pe.pop("rebuild", False))
        instr = target_instr(pair)
        if not rebuild:   # failure regeneration gets the corrective addendum
            instr += REGEN_SUFFIX.replace(
                "{failure_reasons}", pe.get("fail_reasons") or "previous edit not a single "
                "localized edit").replace("{max_units}", str(MAX_UNITS))
        prompt = (INJECT.format(t=corpus_notes[key]["transcript"][:40000], note=clean,
                                instr=instr))
        res = construct_json(state, "regen", prompt, timeout=420)
        ok = (isinstance(res, dict) and isinstance(res.get("note"), str)
              and valid_errored(clean, res["note"].strip(), pair["type"]))
        with _state_lock:
            if not ok:
                if rebuild:
                    pe["rebuild"] = True   # keep the marker so the retry stays a rebuild
                print(f"  {pid}: regen call FAILED (rerun to retry)", flush=True)
                return
            pe["current"] = {"kind": "original" if rebuild else "regen",
                             "rebuilt": rebuild, "clean_sha": sha16(clean),
                             "errored": res["note"].strip(), "change": res.get("change"),
                             "what": res.get("what")}
            pe["status"] = "pending"
        save_state(state)
        print(f"  {pid}: {'rebuilt post-repair' if rebuild else 'regenerated'}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, todo))
    save_state(state)
    if BUDGET_HIT.is_set():
        print(f"BUDGET STOP: ${state['spend']['cost_usd']:.2f} >= ${BUDGET_STOP_USD}", flush=True)
        sys.exit(3)
    return len(todo)


def _fact_of(pair):
    t = pair.get("target") or {}
    if t.get("kind") == "must_contain":
        return t.get("fact") or ""
    w = pair.get("what")
    return w if isinstance(w, str) else (t.get("trap") or json.dumps(w or ""))


def programmatic_omit(clean, pair, pe):
    """Deterministic fallback for stubborn omit pairs: delete the raw-text span of the
    unit(s) carrying the target fact (1-2 ADJACENT units so the algorithmic rule holds)."""
    fact_toks = set(_norm_assertion(_fact_of(pair)).split()) - {
        "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "with", "no", "not"}
    if not fact_toks:
        return None
    text = unicodedata.normalize("NFC", clean)
    unit_texts = units_of(text)
    # score units by overlap with the target fact's content tokens
    scored = sorted(((len(fact_toks & set(_norm_assertion(u).split()))
                      / max(len(fact_toks), 1), i) for i, u in enumerate(unit_texts)),
                    reverse=True)
    if not scored or scored[0][0] == 0:
        return None
    best_i = scored[0][1]
    # delete by whitespace-tolerant exact-text removal of the unit(s); try the single best
    # unit, then it plus one ADJACENT unit (adjacent deletions stay one opcode block)
    candidates = [[best_i]]
    if best_i + 1 < len(unit_texts):
        candidates.append([best_i, best_i + 1])
    if best_i - 1 >= 0:
        candidates.append([best_i - 1, best_i])
    for units in candidates:
        new = text
        ok = True
        for ui in sorted(set(units), reverse=True):
            pat = re.compile(re.escape(unit_texts[ui]).replace(r"\ ", r"\s+"))
            new, n_sub = pat.subn("", new, count=1)
            ok = ok and n_sub == 1
        if not ok:
            continue
        new = re.sub(r"[ \t]{2,}", " ", new)
        new = re.sub(r" +\n", "\n", new)
        new = re.sub(r"\n{3,}", "\n\n", new).strip()
        v = algo_check(clean, new, "omit")
        if v["pass"]:
            return {"errored": new,
                    "change": "programmatic deletion of the unit carrying the target fact",
                    "what": _fact_of(pair)}
    return None


# ---------------------------------------------------------------- severity backfill (authored)
def severity_stage(state, corpus_notes, pairs, workers):
    todo = [p for p in pairs if p["stratum"] == "authored"
            and p["pair_id"] not in state["severity_backfill"]
            and state["pairs"].get(p["pair_id"], {}).get("status") == "passed"]
    # also regenerated/fallback new-stratum pairs that lost their rubric grade context:
    regraded = [p for p in pairs if p["stratum"] != "authored"
                and state["pairs"].get(p["pair_id"], {}).get("status") == "passed"
                and (state["pairs"][p["pair_id"]]["current"].get("kind") in
                     ("regen", "fallback")
                     or state["pairs"][p["pair_id"]]["current"].get("rebuilt"))
                and (p.get("target") or {}).get("kind") != "trap"
                and p["pair_id"] not in state["severity_backfill"]]
    todo += regraded
    if not todo:
        return 0
    print(f"severity-backfill: {len(todo)} pairs ({CONSTRUCT_TRANSPORT})", flush=True)
    _CONSTRUCT_CTX["state"], _CONSTRUCT_CTX["stage"] = state, "severity"

    def one(pair):
        if not budget_ok(state) or past_deadline():
            return
        pe = state["pairs"][pair["pair_id"]]
        cur = pe["current"]
        inj = {"change": cur.get("change") or pair.get("change"),
               "what": cur.get("what") or pair.get("what")}
        note = corpus_notes[f"{pair['stratum']}|{pair['id']}"]
        got = run_rubric_grade(note["transcript"], pair["type"], inj)
        with _state_lock:
            if got is None:
                print(f"  {pair['pair_id']}: rubric grade FAILED (rerun to retry)", flush=True)
                return
            state["severity_backfill"][pair["pair_id"]] = got
        save_state(state)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, todo))
    save_state(state)
    if BUDGET_HIT.is_set():
        print(f"BUDGET STOP: ${state['spend']['cost_usd']:.2f} >= ${BUDGET_STOP_USD}", flush=True)
        sys.exit(3)
    left = [p for p in todo if p["pair_id"] not in state["severity_backfill"]]
    return len(left)


# ---------------------------------------------------------------- stats helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return {"k": k, "n": n, "p": round(p, 4), "lo": round(max(0.0, c - h), 4),
            "hi": round(min(1.0, c + h), 4)}


# ---------------------------------------------------------------- stage: check0
def check0(state, corpus_notes, pairs):
    problems = []
    by_consult = defaultdict(list)
    for p in pairs:
        by_consult[(p["stratum"], p["id"])].append(p)
    for (stratum, cid), ps in sorted(by_consult.items()):
        types = sorted(x["type"] for x in ps)
        if types != ["add", "change", "omit"]:
            problems.append(f"{stratum}|{cid}: pair types {types}")
        cleans = {x["clean"] for x in ps}
        if len(cleans) != 1:
            problems.append(f"{stratum}|{cid}: {len(cleans)} distinct clean texts")
        note = corpus_notes.get(f"{stratum}|{cid}")
        if note is None:
            problems.append(f"{stratum}|{cid}: no ideal note record")
        elif ps[0]["clean"] != note["orig_note"]:
            problems.append(f"{stratum}|{cid}: clean != ideal note")
    missing_pairs = set(corpus_notes) - {f"{s}|{c}" for s, c in by_consult}
    if missing_pairs:
        problems.append(f"notes without pairs: {sorted(missing_pairs)}")
    state["check0"] = {"pass": not problems, "problems": problems,
                       "n_consults": len(by_consult), "n_pairs": len(pairs), "utc": now_utc()}
    save_state(state)
    if problems:
        print("CHECK 0 FAILED (pipeline bug - fix before anything runs):")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print(f"check 0 PASS: {len(by_consult)} consults x3 pairs = {len(pairs)}; clean texts "
          "byte-identical to their ideal notes")


# ---------------------------------------------------------------- freeze + reports
def freeze(state, corpus_notes, pairs):
    # gate assertions
    unresolved_notes = {k: v["status"] for k, v in state["notes"].items()
                        if v["status"] not in ("clean", "excluded")}
    unresolved_pairs = {pid: pe["status"] for pid, pe in state["pairs"].items()
                       if pe["status"] not in ("passed", "excluded")}
    assert not unresolved_notes, f"notes unresolved: {unresolved_notes}"
    assert not unresolved_pairs, f"pairs unresolved: {unresolved_pairs}"
    for p in pairs:
        assert p["pair_id"] in state["pairs"], f"pair never processed: {p['pair_id']}"

    excluded_notes = {k for k, v in state["notes"].items() if v["status"] == "excluded"}
    survivors = []
    for pair in pairs:
        pe = state["pairs"][pair["pair_id"]]
        if pe["status"] != "passed":
            continue
        key = f"{pair['stratum']}|{pair['id']}"
        ne = state["notes"][key]
        cur = pe["current"]
        assert cur["clean_sha"] == sha16(ne["text"]), f"{pair['pair_id']}: stale clean at freeze"
        sev, sev_src = pair.get("severity"), pair.get("severity_source")
        rec_extra = {}
        if pair["pair_id"] in state["severity_backfill"]:
            g = state["severity_backfill"][pair["pair_id"]]
            sev, sev_src = g["importance"], "rubric_graded_at_freeze"
            rec_extra = {"severity_rationale": g.get("rationale"),
                         "severity_test_fired": g.get("test_fired")}
        elif sev_src == "rubric_graded":
            rec_extra = {k: pair[k] for k in ("severity_rationale", "severity_test_fired")
                         if k in pair}
        prov = dict(pair.get("provenance") or {})
        prov["w_a"] = {
            "verified": True, "attempt_kind": cur.get("kind", "original"),
            "n_attempts": len(pe["attempts"]),
            "clean_note_round": ne["round"], "clean_note_repaired": ne["repaired"],
            "waivers": pe.get("waivers") or []}
        survivors.append({
            "pair_id": pair["pair_id"], "id": pair["id"], "stratum": pair["stratum"],
            "type": pair["type"], "clean": ne["text"], "errored": cur["errored"],
            "change": cur.get("change") or pair.get("change"),
            "what": cur.get("what") or pair.get("what"),
            "target": pair.get("target"), "severity": sev, "severity_source": sev_src,
            **rec_extra, "mc_idx": pe.get("mc_idx"),
            "verified": True, "provenance": prov})

    # balance gate over the surviving CORPUS (eval + dev pool - the carve-out is routing,
    # not measurement)
    n = len(survivors)
    tt = Counter(p["type"] for p in survivors)
    balance = {"corpus_pairs": n, "corpus_in_360_450": 360 <= n <= 450,
               "by_type": dict(tt),
               "each_type_within_5pp_of_third":
                   all(abs(tt[t] / n - 1 / 3) <= 0.05 for t in ("add", "change", "omit")),
               "by_stratum_type": {}}
    for s in ("authored", "trapblind", "primock", "aci"):
        balance["by_stratum_type"][s] = dict(
            Counter(p["type"] for p in survivors if p["stratum"] == s))

    # GEPA dev-pool carve for the prompt-optimisation campaigns: seeded draw over
    # SURVIVING consult ids, sorted, seed 20260728; ~8 authored + ~14 primock.
    surviving_consults = defaultdict(set)
    for p in survivors:
        surviving_consults[p["stratum"]].add(p["id"])
    rng = random.Random(CARVE_SEED)
    dev_ids = {}
    for s, k in CARVE_N.items():
        pool = sorted(surviving_consults[s])
        dev_ids[s] = sorted(rng.sample(pool, min(k, len(pool))))
    dev_set = {(s, i) for s, ids in dev_ids.items() for i in ids}
    frozen_pairs = [p for p in survivors if (p["stratum"], p["id"]) not in dev_set]
    dev_pairs = [p for p in survivors if (p["stratum"], p["id"]) in dev_set]

    # final ideal notes (all audited consults, final text)
    ideal_out, sheets = [], {}
    for key in sorted(corpus_notes):
        ne = state["notes"][key]
        stratum, cid = ne["stratum"], ne["id"]
        ideal_out.append({
            "id": cid, "stratum": stratum, "note": ne["text"], "ok": ne["status"] == "clean",
            "repaired": ne["repaired"], "repair_rounds": ne["round"],
            "status": ne["status"], "excluded_reason": ne.get("excluded_reason"),
            "kept_flagged": ne.get("kept_flagged", False)})
    prompt_hashes = {k: sha16(v) for k, v in PROMPTS.items()}

    frozen = {
        "frozen_utc": now_utc(), "spec": SPEC,
        "frozen_strata": sorted({p["stratum"] for p in frozen_pairs}),
        "constants": {"algo_rule_version": ALGO_RULE_VERSION,
                      "v1_frozen": {"MAX_UNITS": MAX_UNITS, "MAX_REGIONS": MAX_REGIONS,
                                    "MAX_CHANGED_WORDS": MAX_CHANGED_WORDS},
                      "v2": {"single_block": "unchanged (load-bearing)",
                             "PURE_BLOCK_MAX_UNITS": PURE_BLOCK_MAX_UNITS,
                             "REPLACE_MAX_UNITS": REPLACE_MAX_UNITS,
                             "MAX_CHANGED_WORDS": MAX_CHANGED_WORDS,
                             "dropped": "word-op purity + MAX_REGIONS proxies (edit "
                                        "direction verified semantically)"},
                      "revision_note": "the single pre-registered revision, "
                                       "exercised before any semantic call; "
                                       "evidence + both fail-counts in wa_gate_report.json",
                      "audit_seeds": AUDIT_SEEDS, "edit_seeds": EDIT_SEEDS,
                      "lever": state["lever"]},
        "models": {"auditor": {k: v for k, v in resolve_model("auditor").items()
                               if k != "notes"},
                   "construction": {"model": "claude-opus-5", "effort": "medium",
                                    "route": "Claude subscription (rounds 0-3 note repairs) "
                                             "-> openrouter anthropic/claude-opus-5",
                                    "transport_change_utc": "2026-08-10",
                                    "transport_change_note":
                                        "construction moved off the "
                                        "Claude subscription onto OpenRouter credits (models.lock "
                                        "role 'constructor'), budget stop 350 -> 450. SAME "
                                        "model and effort, different transport; the auditor "
                                        "role and every pinned auditor setting are unchanged, "
                                        "and every constructed artefact is re-audited by the "
                                        "cross-family auditor regardless of transport. "
                                        "Subscription calls made before the switch: "
                                        + str(state["spend"].get("plan_calls", 0))
                                        + "; OpenRouter constructor calls after it: "
                                        + str(state["spend"].get("construct_calls", 0))}},
        "prompt_sha256_16": prompt_hashes,
        "counts": {"pairs_frozen": len(frozen_pairs), "pairs_dev_pool": len(dev_pairs),
                   "pairs_excluded": sum(1 for pe in state["pairs"].values()
                                         if pe["status"] == "excluded"),
                   "notes_excluded": len(excluded_notes),
                   "by_stratum_type_frozen": {
                       s: dict(Counter(p["type"] for p in frozen_pairs if p["stratum"] == s))
                       for s in sorted({p["stratum"] for p in frozen_pairs})}},
        "balance_gate": balance,
        "gepa_dev_pool_ids": dev_ids,
        "ideal_notes": [i for i in ideal_out
                        if (i["stratum"], i["id"]) not in dev_set and i["status"] == "clean"],
        "pairs": frozen_pairs,
    }
    tmp = os.path.join(MASTER, "pairs_master_frozen.json.tmp")
    json.dump(frozen, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(MASTER, "pairs_master_frozen.json"))
    frozen_sha = _sha256(os.path.join(MASTER, "pairs_master_frozen.json"))
    # byte-identical root copy: check_manifests expects inputs named
    # 'pairs_master_frozen.json', the corpus build names master/ - both recorded
    with open(os.path.join(MASTER, "pairs_master_frozen.json"), "rb") as fsrc, \
            open(os.path.join(HERE, "pairs_master_frozen.json"), "wb") as fdst:
        fdst.write(fsrc.read())

    # dev pool (audited + verified, same schema)
    fs_by = {}
    for src in ("trapblind", "primock", "aci"):
        for rec in json.load(open(os.path.join(MASTER, f"fact_sheets_{src}_core.json"))):
            fs_by[(src, rec["id"])] = rec["fact_sheet"]
    for s in json.load(open(os.path.join(HERE, "authored_scenarios.json"))):
        fs_by[("authored", s["id"])] = s["fact_sheet"]
    dev_out = {"generated_utc": now_utc(), "seed": CARVE_SEED,
               "spec": "fully audited, routed out of the frozen eval set; the training "
                       "resource for the prompt-optimisation campaigns",
               "ids": dev_ids,
               "consultations": [
                   {"id": cid, "stratum": s,
                    "fact_sheet": fs_by[(s, cid)],
                    "ideal_note": state["notes"][f"{s}|{cid}"]["text"],
                    "pairs": [p for p in dev_pairs if p["stratum"] == s and p["id"] == cid]}
                   for s, ids in dev_ids.items() for cid in ids]}
    json.dump(dev_out, open(os.path.join(MASTER, "gepa_dev_pool.json"), "w"),
              ensure_ascii=False, indent=1)
    json.dump({"seed": CARVE_SEED, "ids": dev_ids, "generated_utc": now_utc()},
              open(os.path.join(MASTER, "gepa_dev_pool_ids.json"), "w"), indent=1)

    # freeze_hashes.json, read by check_manifests
    fh = {}
    if os.path.exists(os.path.join(HERE, "freeze_hashes.json")):
        fh = json.load(open(os.path.join(HERE, "freeze_hashes.json")))
    fh["pairs_master_frozen.json"] = frozen_sha
    fh["master/pairs_master_frozen.json"] = frozen_sha
    json.dump(fh, open(os.path.join(HERE, "freeze_hashes.json"), "w"), indent=1)
    return frozen, frozen_sha, dev_out, ideal_out


def _revision_record(pairs):
    """Recount BOTH versions of the algorithmic rule over the as-built pairs (self-verifying
    audit trail for the single pre-registered constants revision)."""
    v1f, v2f = Counter(), Counter()
    for p in pairs:
        if not algo_check_v1(p["clean"], p["errored"], p["type"]):
            v1f[p["stratum"]] += 1
        if not algo_check(p["clean"], p["errored"], p["type"])["pass"]:
            v2f[p["stratum"]] += 1
    return {
        "clause": "the constants were frozen at pre-registration under a clause allowing "
                  "one revision: 'if they prove badly calibrated in the first 10 pairs, we "
                  "may revise ONCE, before the full run, and record the revision + reason' "
                  "- exercised BEFORE any semantic EDIT_CHECK call, applied by default and "
                  "queued for human review",
        "evidence": "v1 rejected 20/90 of the pre-registered preserved authored stratum "
                    "(the calibration stratum; 13 omit pairs among them - "
                    "the study's headline error class) and 51% corpus-wide; dominant "
                    "false-fail classes were one-block contiguous section deletes of 3-5 "
                    "units and adds/omits realized as natural 1:1 line replacements - "
                    "all single-location edits satisfying the rule's stated intent "
                    "('exactly one localized edit, everything else verbatim')",
        "kept_frozen": "single non-equal opcode block (the localization invariant) + "
                       "MAX_CHANGED_WORDS=30 blast-radius cap + REGEN_SUFFIX max_units=2",
        "changed": {"pure_insert_delete_unit_cap": f"2 -> {PURE_BLOCK_MAX_UNITS}",
                    "replace_span_unit_cap": f"2 -> {REPLACE_MAX_UNITS} per side",
                    "dropped_proxies": "insert-only/delete-only word-op purity; "
                                       "MAX_REGIONS=2 (direction + atomicity now "
                                       "verified by EDIT_CHECK 3-seed majority)"},
        "fail_counts_as_built": {"v1": {"total": sum(v1f.values()), **dict(v1f)},
                                 "v2": {"total": sum(v2f.values()), **dict(v2f)}},
        "no_further_revision": True}


def exclusion_why(state):
    """One generated sentence accounting for every excluded pair, by cause and type - so the
    verdict never has to be hand-amended after a re-freeze."""
    buckets = Counter()
    for pid, pe in state["pairs"].items():
        if pe["status"] != "excluded":
            continue
        r = pe.get("excluded_reason") or ""
        cause = ("note excluded (unrepairable clean note)" if r.startswith("note_excluded")
                 else "authored pair failed the semantic load-bearing majority"
                 if r.startswith("authored pair failed")
                 else "omit pair failed 3 regenerations and its programmatic fallback"
                 if "programmatic-deletion fallback" in r
                 else "add/change pair failed 3 regenerations"
                 if "regenerations" in r else "other")
        buckets[(cause, pe["type"])] += 1
    n = sum(buckets.values())
    parts = "; ".join(f"{v} {t} - {c}" for (c, t), v in sorted(buckets.items(),
                                                              key=lambda x: -x[1]))
    return (f"{n} of {len(state['pairs'])} constructed pairs were excluded by the "
            f"ground-truth gate ({parts}), "
            f"leaving {len(state['pairs']) - n}.")


def _truly_absent_record(state):
    """Self-verifying account of the truly_absent correction: which pairs were re-verified
    under the corrected EDIT_CHECK, and what changed. Built from the state, not a list."""
    rows = []
    for pid, pe in sorted(state["pairs"].items()):
        if not pe.get("reverify_2026_08_10"):
            continue
        old = pe["current"].get("sem_runs_v1_conflated") or {}
        prior = [a for a in pe["attempts"][:-1]]
        rows.append({
            "pair_id": pid,
            "sem_fails_before": (prior[-1].get("sem_fails") if prior else None),
            "sem_fails_after": pe["attempts"][-1].get("sem_fails"),
            "status_after": pe["status"],
            "mc_idx_after": pe["attempts"][-1].get("mc_idx"),
            "n_seeds_before": len(old), "n_seeds_after": pe["attempts"][-1].get("n_sem_seeds"),
            "reason_after": next((r.get("reason") for r in
                                  pe["current"].get("sem_runs", {}).values() if r.get("reason")),
                                 None)})
    restored = [r["pair_id"] for r in rows if r["status_after"] == "passed"]
    still = [r for r in rows if r["status_after"] != "passed"]
    return {
        "what_changed": SUPERSEDE_REASON,
        "prompt_diff": "master/truly_absent_prompt_diff.md",
        "edit_check_sha256_16": {"before": "0cffb4e7fb568030", "after": sha16(EDIT_CHECK)},
        "n_reverified": len(rows), "n_restored": len(restored),
        "n_still_excluded": len(still),
        "restored_pairs": restored,
        "still_failing_on": dict(Counter(", ".join(r["sem_fails_after"] or ["-"])
                                         for r in still)),
        "finding": "the pre-correction diagnosis - that the "
                   "MUST_CONTAIN-index conjunct was the clause doing the failing - is NOT "
                   "what the re-verification shows. Only the pairs where no index could be "
                   "produced at all were freed by the fix; the rest fail the FIRST clause, "
                   "the auditor finding a partial or paraphrased residual of the removed "
                   "fact still in the errored note. The correction is right on its own terms "
                   "and is kept, but it does not restore the corpus to the balance band",
        "pairs": rows}


def build_reports(state, corpus_notes, pairs, frozen, frozen_sha, dev_out, ideal_out):
    strata = ("authored", "trapblind", "primock", "aci")
    # ---- M1: first-pass note defect rate (round-0 majority flags), per stratum
    m1, m1_cat = {}, Counter()
    first_flags_by_note = {}
    for key, ne in state["notes"].items():
        fh0 = next((h for h in ne.get("flag_history", []) if h["round"] == 0), None)
        first_flags_by_note[key] = fh0["flags"] if fh0 else []
        for f in (fh0["flags"] if fh0 else []):
            m1_cat[f["kind"]] += 1
    for s in strata + ("all",):
        keys = [k for k in corpus_notes if s == "all" or k.startswith(s + "|")]
        m1[s] = wilson(sum(1 for k in keys if first_flags_by_note.get(k)), len(keys))
    # ---- M2: first-pass pair failure rate (original attempt), by type/stratum/layer
    m2 = {"overall": None, "by_type": {}, "by_stratum": {}, "by_layer": Counter()}
    first_fail = {}
    for pair in pairs:
        pe = state["pairs"].get(pair["pair_id"])
        a0 = next((a for a in (pe or {}).get("attempts", []) if a["kind"] == "original"), None)
        algo_ok = bool(a0 and a0.get("algo", {}).get("pass"))
        sem_ok = bool(a0 and not a0.get("sem_fails"))
        if a0 is None:
            # original never reached resolution (e.g. note excluded first, or replaced
            # pre-editcheck after repair) - count algo layer from stored current if original
            first_fail[pair["pair_id"]] = None
            continue
        failed = (not algo_ok) or (not sem_ok)
        first_fail[pair["pair_id"]] = {"failed": failed, "algo": not algo_ok,
                                       "sem": bool(a0.get("sem_fails")) and algo_ok}
        if failed:
            m2["by_layer"]["algorithmic" if not algo_ok else "semantic"] += 1
    scored = {pid: v for pid, v in first_fail.items() if v is not None}
    m2["overall"] = wilson(sum(1 for v in scored.values() if v["failed"]), len(scored))
    for t in ("add", "change", "omit"):
        sub = [v for pid, v in scored.items() if pid.rsplit("|", 1)[1] == t]
        m2["by_type"][t] = wilson(sum(1 for v in sub if v["failed"]), len(sub))
    for s in strata:
        sub = [v for pid, v in scored.items() if pid.startswith(s + "|")]
        m2["by_stratum"][s] = wilson(sum(1 for v in sub if v["failed"]), len(sub))
    m2["by_layer"] = dict(m2["by_layer"])
    m2["unscored_originals"] = sum(1 for v in first_fail.values() if v is None)
    # ---- majority-confirmation KILL RATE: what fraction of 1-seed screen flags die at
    # 3-seed confirmation - the empirical answer to the auditor-pedantry worry. Plus
    # pairwise agreement on 3-seed sections (the reproducibility metric).
    conf = {"screen_flag_items": 0, "confirmed_items": 0,
            "by_section": {s: {"screen": 0, "confirmed": 0} for s in SECTIONS}}
    agree_vals = []
    agree_by_section = defaultdict(list)     # agreement coverage fix, see below
    m4_cov = {"section_audits_total": len(state["audits"]),
              "escalated_to_3_seeds": 0, "screen_clean_1_seed_only": 0,
              "screen_clean_all_flagsets_empty": True}
    s0 = str(AUDIT_SEEDS[0])
    for akey, se in state["audits"].items():
        if len(se["runs"]) < 3:
            m4_cov["screen_clean_1_seed_only"] += 1
            if any(r.get("fails") for r in se["runs"].values()):
                m4_cov["screen_clean_all_flagsets_empty"] = False
            continue
        m4_cov["escalated_to_3_seeds"] += 1
        section = akey.rsplit("|", 1)[1]
        runs = se["runs"]
        if section == "unsupported":
            cl = cluster_unsupported({s: r["fails"] for s, r in runs.items()})
            screen_cl = [c for c in cl if s0 in c["seeds"]]
            conf["screen_flag_items"] += len(screen_cl)
            conf["by_section"][section]["screen"] += len(screen_cl)
            kept = sum(1 for c in screen_cl if len(c["seeds"]) >= 2)
            conf["confirmed_items"] += kept
            conf["by_section"][section]["confirmed"] += kept
            # COVERAGE FIX: the open-ended section used to `continue` here, so it
            # contributed nothing to the agreement figure at all - which silently covered
            # only mc/mnc/traps. Agreement over clustered assertions is computable from the
            # same stored runs (no new calls), so it is now included in the extended figure.
            seeds_u = list(runs)
            for a_i in range(len(seeds_u)):
                for b_i in range(a_i + 1, len(seeds_u)):
                    fa = {i for i, c in enumerate(cl) if seeds_u[a_i] in c["seeds"]}
                    fb = {i for i, c in enumerate(cl) if seeds_u[b_i] in c["seeds"]}
                    universe = fa | fb
                    agree_by_section[section].append(
                        1.0 if not universe else len(fa & fb) / len(universe))
            continue
        screen_fails = set(int(i) for i in runs.get(s0, {}).get("fails", {}))
        conf["screen_flag_items"] += len(screen_fails)
        conf["by_section"][section]["screen"] += len(screen_fails)
        for i in screen_fails:
            votes = sum(1 for r in runs.values()
                        if i in {int(x) for x in r["fails"]})
            if votes >= 2:
                conf["confirmed_items"] += 1
                conf["by_section"][section]["confirmed"] += 1
        # pairwise per-item agreement over the union of flagged idxs
        seeds = list(runs)
        for a_i in range(len(seeds)):
            for b_i in range(a_i + 1, len(seeds)):
                fa = {int(x) for x in runs[seeds[a_i]]["fails"]}
                fb = {int(x) for x in runs[seeds[b_i]]["fails"]}
                universe = fa | fb
                v = 1.0 if not universe else len(fa & fb) / len(universe)
                agree_vals.append(v)
                agree_by_section[section].append(v)
    confirmation_rate = (conf["confirmed_items"] / conf["screen_flag_items"]
                         if conf["screen_flag_items"] else None)
    kill_rate = (1 - confirmation_rate) if confirmation_rate is not None else None

    def _mean4(v):
        return round(sum(v) / len(v), 4) if v else None
    m4 = _mean4(agree_vals)                                  # checklist sections only
    all_vals = [v for vs in agree_by_section.values() for v in vs]
    m4_all = _mean4(all_vals)                                # + the open-ended section
    # ---- M5 repair convergence
    m5 = {"repaired_notes": sum(1 for ne in state["notes"].values() if ne["repaired"]),
          "rounds_hist": dict(Counter(ne["round"] for ne in state["notes"].values()
                                      if ne["repaired"])),
          "excluded_notes": sum(1 for ne in state["notes"].values()
                                if ne["status"] == "excluded"),
          "fallback_pairs": sum(1 for pe in state["pairs"].values()
                                if pe["status"] == "passed"
                                and pe["current"].get("kind") == "fallback"),
          "regen_pairs": sum(1 for pe in state["pairs"].values()
                             if pe["status"] == "passed"
                             and pe["current"].get("kind") == "regen")}
    # ---- M6 omit-salience linkage
    omit_pass = [pe for pid, pe in state["pairs"].items()
                 if pe["status"] == "passed" and pid.endswith("|omit")]
    m6 = wilson(sum(1 for pe in omit_pass if (pe.get("mc_idx") or -1) >= 0), len(omit_pass))
    report = {
        "generated_utc": now_utc(), "spec": SPEC,
        "spec_output_name": "the pre-registered output name is w_a_report.json; delivered "
                            "as master/wa_gate_report.json",
        "gate": {"pass": True,
                 "pass_means": "the ground-truth gate itself completed: every note is "
                               "clean-or-excluded, every pair is passed-or-excluded, and the "
                               "surviving set is frozen. It does NOT assert the corpus-size "
                               "and balance requirement - read balance_gate_verdict below "
                               "and balance_gate for that.",
                 "balance_gate_verdict": {
                     "pass": bool(frozen["balance_gate"]["corpus_in_360_450"]
                                  and frozen["balance_gate"]["each_type_within_5pp_of_third"]),
                     "rule": "the pre-registered corpus rule: final corpus within 360-450 pairs AND each of "
                             "add/change/omit within +/-5 percentage points of one third",
                     "corpus_pairs": frozen["balance_gate"]["corpus_pairs"],
                     "corpus_in_360_450": frozen["balance_gate"]["corpus_in_360_450"],
                     "by_type": frozen["balance_gate"]["by_type"],
                     "type_shares": {t: round(c / frozen["balance_gate"]["corpus_pairs"], 4)
                                     for t, c in frozen["balance_gate"]["by_type"].items()},
                     "pp_from_one_third": {
                         t: round(100 * (c / frozen["balance_gate"]["corpus_pairs"] - 1 / 3), 1)
                         for t, c in frozen["balance_gate"]["by_type"].items()},
                     "each_type_within_5pp_of_third":
                         frozen["balance_gate"]["each_type_within_5pp_of_third"],
                     "why": exclusion_why(state),
                     "note": "the pre-registered remedy for a broken balance ('drop the "
                             "surplus type's weakest pairs') cannot apply when the corpus is "
                             "BELOW the band - dropping shrinks it further. A corpus under 360 "
                             "needs more consultations or re-injected pairs for the excluded "
                             "ones, which is a human decision, not this driver's."},
                 "frozen_file": "master/pairs_master_frozen.json",
                 "sha256": frozen_sha,
                 "statement": f"GROUND-TRUTH GATE: pairs_master_frozen.json "
                              f"sha256={frozen_sha} "
                              f"({frozen['counts']['pairs_frozen']} pairs, strata "
                              f"{frozen['frozen_strata']}) - every downstream judge run "
                              "loads ONLY this file, asserted by hash",
                 "root_copy": "pairs_master_frozen.json (byte-identical; both paths in "
                              "freeze_hashes.json)"},
        "lever_disclosure": state["lever"] + " (pre-registered; screens "
            "= seed 11/21; confirmations = 12,13/22,23; majority-of-3 semantics on every "
            "flag; single-seed acceptance for clean screens)",
        "adjudication_policy": {
            "mode": "automated; human clinical adjudication is queued for review in "
                    "master/sitting_wa_items.json",
            "new_strata": "majority flag = confirmed defect -> repair (max 3 rounds) -> "
                          "exclude",
            "authored": "report-only; exclude note only on unanimous 3/3 checklist-section "
                        "flag; exclude pair only on semantic load-bearing majority failure",
            "M1_caveat": "the first-pass note defect rate is an auditor-majority rate - an "
                         "upper bound on the human-confirmed rate the protocol defines; "
                         "auditor precision against a human is n/a"},
        "algorithmic_constants_revision": _revision_record(pairs),
        "truly_absent_correction": _truly_absent_record(state),
        "check0": state.get("check0"),
        "M1_first_pass_note_defect_rate": m1,
        "M1_defect_categories": dict(m1_cat),
        "M2_first_pass_pair_failure_rate": m2,
        "M3_auditor_precision": {"value": None,
                                 "reason": "no human adjudication in the loop (see policy); "
                                           "the majority-confirmation kill rate is reported "
                                           "instead",
                                 "majority_confirmation_kill_rate":
                                     round(kill_rate, 4) if kill_rate is not None else None,
                                 "kill_rate_meaning": "fraction of 1-seed screen flags that "
                                     "DIE at 3-seed majority confirmation - empirical answer "
                                     "to the auditor-pedantry worry",
                                 "auditor_confirmation_rate":
                                     round(confirmation_rate, 4)
                                     if confirmation_rate is not None else None,
                                 "screen_flag_items": conf["screen_flag_items"],
                                 "confirmed_items": conf["confirmed_items"],
                                 "by_section": conf["by_section"]},
        "M4_audit_reproducibility": {
            "mean_pairwise_agreement_flag_items": m4,
            "figure_covers": "checklist sections (mc/mnc/traps) that escalated to 3 seeds - "
                             "the figure originally reported; the open-ended 'unsupported' "
                             "section was silently omitted from it",
            "mean_pairwise_agreement_incl_unsupported": m4_all,
            "by_section": {s: {"mean": _mean4(v), "seed_pairs": len(v)}
                           for s, v in sorted(agree_by_section.items())},
            "coverage": {**m4_cov,
                         "seed_pairs_scored": len(all_vals),
                         "meaning": "under the cost lever a section is audited at 3 seeds ONLY "
                                    "if the seed-11 screen flagged something; a clean screen "
                                    "is accepted on one seed, so no second opinion exists for "
                                    "it and its agreement is UNMEASURED, not 1.0"},
            "restriction": "BOTH figures are conditioned on the screen having flagged "
                           "something, i.e. computed over the disputed sections only. They "
                           "are a lower bound on whole-corpus reproducibility and are NOT "
                           "comparable to the pre-registered 0.90 line, which contemplates "
                           "all sections. Computing the unconditional figure needs 2 extra "
                           "seeds on the "
                           + str(m4_cov["screen_clean_1_seed_only"]) + " screen-clean "
                           "sections; not spent",
            "assumption_conditional_ceiling_not_measured":
                (round((sum(all_vals) + m4_cov["screen_clean_1_seed_only"] * 3)
                       / (len(all_vals) + m4_cov["screen_clean_1_seed_only"] * 3), 4)
                 if all_vals else None),
            "ceiling_meaning": "arithmetic only: what the unconditional mean WOULD be if "
                               "every screen-clean section had produced an empty flag set on "
                               "all 3 seeds. An assumption, not a measurement - do not quote "
                               "it as an observed value"},
        "M5_repair_convergence": m5,
        "M6_omit_salience_linkage": {
            **m6,
            "caveat": "BEFORE the truly_absent correction this metric was 1.0 BY "
                      "CONSTRUCTION - an omit pair could not pass verification unless a "
                      "MUST_CONTAIN index had been found for it, so k==n was forced. From "
                      "this run on, mc_idx is recorded but does not gate acceptance, so the "
                      "figure is a genuine measurement of how often a verified omission "
                      "maps to a checklist item"},
        "exclusions": {
            "notes": {k: ne.get("excluded_reason") for k, ne in state["notes"].items()
                      if ne["status"] == "excluded"},
            "pairs": {pid: pe.get("excluded_reason") for pid, pe in state["pairs"].items()
                      if pe["status"] == "excluded"}},
        "kept_flagged_authored_notes": [k for k, ne in state["notes"].items()
                                        if ne.get("kept_flagged")],
        "balance_gate": frozen["balance_gate"],
        "gepa_dev_pool_n": {"consults": {s: len(ids) for s, ids in
                                         frozen["gepa_dev_pool_ids"].items()},
                            "pairs": frozen["counts"]["pairs_dev_pool"]},
        "spend": state["spend"],
        "run_ids": state["run_ids"],
        "models": frozen["models"],
        "prompt_sha256_16": frozen["prompt_sha256_16"],
        "readings": {},
    }
    # pre-registered reading per stratum
    for s in strata:
        r = m1[s]
        if r and r["n"]:
            reading = "A (0-10%)" if r["p"] <= 0.10 else \
                      "B (10-30%)" if r["p"] <= 0.30 else "C (>30%)"
            report["readings"][s] = {
                "M1": r, "reading": reading,
                "note": ("Reading C consequence (instrument rewrite) NOT executed - "
                         "out of scope for this run, which did no further instrument "
                         "iteration; repaired to pass instead, flagged for the paper"
                         if reading.startswith("C") else "within pre-registered band")}
    return report


def write_flags_file(state, corpus_notes):
    rows = []
    for key, ne in state["notes"].items():
        for h in ne.get("flag_history", []):
            for f in h["flags"]:
                row = {"note": key, "stratum": ne["stratum"], "round": h["round"],
                       **{k: v for k, v in f.items() if k != "key"},
                       "adjudication": ("auto-confirmed -> repair"
                                        if ne["stratum"] != "authored"
                                        else ("excluded" if ne["status"] == "excluded"
                                              else "report-only (kept, queued)")),
                       "note_final_status": ne["status"]}
                if (ne.get("adjudication_override") and f.get("unanimous")
                        and f["section"] == "mnc"):
                    row["adjudication"] = "OVERTURNED (transcript-grounded)"
                    row["adjudication_override"] = ne["adjudication_override"]["reason"]
                rows.append(row)
    out = {"generated_utc": now_utc(), "policy": "see master/wa_gate_report.json "
           "adjudication_policy", "n_flags": len(rows), "flags": rows}
    json.dump(out, open(os.path.join(MASTER, "wa_audit_flags.json"), "w"),
              ensure_ascii=False, indent=1)
    return out


def write_repair_log(state):
    out = {"generated_utc": now_utc(),
           "model": "claude-opus-5, effort medium (Claude subscription for the note-repair "
                    "rounds, then OpenRouter anthropic/claude-opus-5 - same model, "
                    "disclosed transport change)",
           "notes": {k: {"rounds": ne["round"], "repaired": ne["repaired"],
                         "status": ne["status"], "history": ne["history"],
                         "flag_history": [{"round": h["round"], "n_flags": h["n_flags"]}
                                          for h in ne.get("flag_history", [])]}
                     for k, ne in state["notes"].items() if ne["repaired"]
                     or ne["status"] == "excluded"}}
    json.dump(out, open(os.path.join(MASTER, "wa_repair_log.json"), "w"),
              ensure_ascii=False, indent=1)


def write_injection_checks(state):
    out = {"generated_utc": now_utc(),
           "pairs": {pid: {"status": pe["status"], "attempts": pe["attempts"],
                           "mc_idx": pe.get("mc_idx"), "waivers": pe.get("waivers"),
                           "final_kind": pe["current"].get("kind"),
                           "excluded_reason": pe.get("excluded_reason")}
                     for pid, pe in state["pairs"].items()}}
    json.dump(out, open(os.path.join(MASTER, "wa_injection_checks.json"), "w"),
              ensure_ascii=False, indent=1)


def write_v2_files(state, corpus_notes, ideal_out, frozen, dev_out):
    json.dump({"generated_utc": now_utc(), "notes": ideal_out},
              open(os.path.join(MASTER, "ideal_notes_v2.json"), "w"),
              ensure_ascii=False, indent=1)
    # hard_negatives_master_v2.json = ALL surviving verified pairs (eval + dev pool)
    allp = frozen["pairs"] + [p for c in dev_out["consultations"] for p in c["pairs"]]
    json.dump({"generated_utc": now_utc(), "n": len(allp), "pairs": allp},
              open(os.path.join(MASTER, "hard_negatives_master_v2.json"), "w"),
              ensure_ascii=False, indent=1)


def write_sitting(state, corpus_notes, report):
    items = []
    for key, ne in state["notes"].items():
        if ne["stratum"] == "authored" and (ne.get("kept_flagged")
                                            or ne["status"] == "excluded"):
            fh0 = ne["flag_history"][-1] if ne.get("flag_history") else {"flags": []}
            item = {"kind": "authored_note_flags", "note": key,
                    "outcome": ne["status"],
                    "ask": "confirm/overturn each flag (report-only policy applied; "
                           "overturns need no action, confirms may motivate a corpus "
                           "revision)",
                    "flags": [{k: f[k] for k in ("section", "idx", "kind", "item",
                                                 "votes", "reasons", "evidence")
                               if k in f} for f in fh0["flags"]]}
            if ne.get("adjudication_override"):
                item["agent_adjudication"] = ne["adjudication_override"]
                item["ask"] = ("REVIEW: a unanimous auditor mnc flag was OVERTURNED "
                               "on transcript grounding (reason attached) and the note "
                               "kept - confirm or reinstate the exclusion")
            items.append(item)
        elif ne["status"] == "excluded":
            items.append({"kind": "excluded_note", "note": key,
                          "reason": ne.get("excluded_reason"),
                          "ask": "sanity-check the exclusion (conservative default applied)"})
        elif ne["repaired"]:
            items.append({"kind": "repaired_note_review", "note": key,
                          "rounds": ne["round"],
                          "n_confirmed_flags_round0": (ne["flag_history"][0]["n_flags"]
                                                       if ne.get("flag_history") else None),
                          "ask": "spot-check the repair (auto-confirmed flags; re-audited "
                                 "to pass)"})
    for pid, pe in state["pairs"].items():
        if pe["status"] == "excluded" and not pe.get("excluded_reason", "").startswith("note_"):
            items.append({"kind": "excluded_pair", "pair": pid,
                          "reason": pe.get("excluded_reason"),
                          "last_attempt": pe["attempts"][-1] if pe["attempts"] else None,
                          "ask": "hand-write the edit if the pair should be rescued "
                                 "(the protocol's fallback, reserved for a human)"})
        elif pe.get("waivers"):
            items.append({"kind": "authored_pair_waiver", "pair": pid,
                          "waivers": pe["waivers"],
                          "ask": "confirm keeping (passed semantic majority; algorithmic "
                                 "or description nonconformance waived per policy)"})
    # 5-8 representative CONFIRMED flags (mixed kinds/strata) so the review can
    # spot-check the AUDITOR itself, not just the ambiguous calls
    confirmed = []
    for key, ne in state["notes"].items():
        fh0 = next((h for h in ne.get("flag_history", []) if h["round"] == 0), None)
        for f in (fh0["flags"] if fh0 else []):
            confirmed.append((key, ne["stratum"], f))
    rng = random.Random(CARVE_SEED)
    rng.shuffle(confirmed)
    picked, per_kind = [], Counter()
    for key, stratum, f in confirmed:
        if per_kind[f["kind"]] >= 2 or len(picked) >= 8:
            continue
        per_kind[f["kind"]] += 1
        picked.append({"kind": "auditor_spot_check", "note": key, "stratum": stratum,
                       "flag_kind": f["kind"], "section": f["section"], "idx": f["idx"],
                       "item": f["item"], "votes": f["votes"], "unanimous": f["unanimous"],
                       "reasons": f["reasons"][:3], "evidence": f.get("evidence"),
                       "ask": "spot-check the auditor: is this a real defect by the "
                              "study's standards? (confirmed by 3-seed majority; "
                              "repaired if new-stratum)"})
    items.extend(picked)
    items.append({
        "kind": "algorithmic_constants_revision", "decision_utc": "2026-08-10",
        "what": "the single pre-registered constants revision exercised (v1 -> v2) "
                "before any semantic call; full record in wa_gate_report.json "
                "'algorithmic_constants_revision'",
        "headline": "v1 rejected 51% of as-built pairs incl. 20/90 preserved authored "
                    "(13 omits); v2 keeps the one-block + 30-word invariants, drops the "
                    "word-op purity/regions proxies, caps pure blocks at 6 units",
        "ask": "review: if v1 strictness should come back, the free pre-check reruns in "
               "seconds and the then-failing pairs regenerate - but that reshapes ~46% "
               "of the corpus toward verifier-convenient minimal edits"})
    out = {"generated_utc": now_utc(),
           "purpose": "the human-adjudication queue for the ground-truth gate: conservative "
                      "defaults were applied and nothing was blocked pending review",
           "n_items": len(items), "items": items}
    json.dump(out, open(os.path.join(MASTER, "sitting_wa_items.json"), "w"),
              ensure_ascii=False, indent=1)
    return out


def complete_pair_verification(state, frozen, frozen_sha, report):
    pv = json.load(open(os.path.join(MASTER, "pair_verification.json")))
    # A re-freeze must never silently replace the sha a previous freeze published: keep the
    # superseded record so both hashes stay quotable (added with the truly_absent
    # correction, the first time this file was rewritten after a published freeze).
    prev = pv.get("verification")
    if prev and prev.get("frozen_sha256", {}).get(
            "master/pairs_master_frozen.json") not in (None, frozen_sha):
        hist = pv.setdefault("superseded_verifications", [])
        if not any(h.get("frozen_sha256") == prev["frozen_sha256"] for h in hist):
            hist.append({
                "superseded_utc": now_utc(),
                "completed_utc": prev.get("completed_utc"),
                "frozen_sha256": prev.get("frozen_sha256"),
                "gate": prev.get("gate"),
                "balance_gate_post_wa": prev.get("balance_gate_post_wa"),
                "spend_usd_openrouter": prev.get("spend_usd_openrouter"),
                "superseded_because": SUPERSEDE_REASON})
    pv["status"] = "frozen"
    pv["verification"] = {
        "completed_utc": now_utc(), "by": "w_a_master.py (" + SPEC + ")",
        "auditor": {"model": "openai/gpt-5.5", "transport": "openrouter",
                    "reasoning_effort": REASONING,
                    "seeds": {"audit": AUDIT_SEEDS, "edit_check": EDIT_SEEDS},
                    "lever": state["lever"]},
        "construction_side": {"model": "claude-opus-5", "effort": "medium",
                              "route": "Claude subscription (note-repair rounds) -> "
                                       "openrouter anthropic/claude-opus-5",
                              "plan_calls": state["spend"].get("plan_calls", 0),
                              "openrouter_construct_calls":
                                  state["spend"].get("construct_calls", 0)},
        "gate": {"pass": True, "pairs_frozen": frozen["counts"]["pairs_frozen"],
                 "pairs_dev_pool": frozen["counts"]["pairs_dev_pool"],
                 "pairs_excluded": frozen["counts"]["pairs_excluded"],
                 "notes_excluded": frozen["counts"]["notes_excluded"]},
        "frozen_sha256": {"master/pairs_master_frozen.json": frozen_sha},
        "balance_gate_post_wa": frozen["balance_gate"],
        "report": "master/wa_gate_report.json",
        "spend_usd_openrouter": state["spend"]["cost_usd"]}
    tmp = os.path.join(MASTER, "pair_verification.json.tmp")
    json.dump(pv, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(MASTER, "pair_verification.json"))


# ---------------------------------------------------------------- selftest
def selftest():
    clean = ("Presenting complaint:\nChest pain for 3 days. Denies breathlessness.\n\n"
             "Plan:\nStart amoxicillin 500mg TDS for 5 days. Safety-netting advised: "
             "return if fevers persist. Review in 2 weeks.")
    add = clean.replace("Review in 2 weeks.",
                        "Review in 2 weeks. Chest X-ray showed clear lung fields.")
    omit = clean.replace(" Safety-netting advised: return if fevers persist.", "")
    change = clean.replace("500mg TDS", "250mg BD")
    multi = change.replace("Denies breathlessness", "Reports breathlessness")
    rewrap = clean.replace("Chest pain for 3 days. Denies breathlessness.",
                           "Chest pain for 3 days.\nDenies breathlessness.")
    # v2-specific: one-block 3-unit section removal passes; 1:1 line-replacement add passes
    omit3 = clean.replace("Plan:\nStart amoxicillin 500mg TDS for 5 days. Safety-netting "
                          "advised: return if fevers persist. Review in 2 weeks.", "Plan:")
    add_repl = clean.replace("Denies breathlessness.",
                             "O/E chest clear, no wheeze, sats 98% on air.")
    for name, err, typ, want in [("add", add, "add", True), ("omit", omit, "omit", True),
                                 ("change", change, "change", True),
                                 ("multi-as-change", multi, "change", False),
                                 ("rewrap-only-as-change", rewrap, "change", False),
                                 ("v2-omit-3unit-block", omit3, "omit", True),
                                 ("v2-add-as-replace", add_repl, "add", True)]:
        v = algo_check(clean, err, typ)
        status = "ok" if v["pass"] == want else "MISMATCH"
        print(f"  {name:24} pass={v['pass']} (want {want}) {status} "
              f"{v.get('reason') or ''}")
        assert v["pass"] == want, name
    # rewrap normalizes to zero-diff (n_blocks=0) - correct: not a valid change pair
    print("selftest OK (frozen constants MAX_UNITS=2 MAX_REGIONS=2 MAX_CHANGED_WORDS=30)")


# ---------------------------------------------------------------- main
T0 = time.time()
DISPATCH_DEADLINE_S = 240.0   # stop dispatching new auditor jobs after this; in-flight
                              # calls drain and the Run finalizes cleanly inside the
                              # 10-minute cap on a single invocation (a killed invocation
                              # orphans its manifest)


def past_deadline():
    return time.time() - T0 > DISPATCH_DEADLINE_S


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "status"
    chunk = int(_arg("--chunk", "0")) or None
    workers = int(_arg("--workers", "6"))
    state = load_state()
    corpus_notes, pairs = load_corpus()
    for key in corpus_notes:
        note_entry(state, corpus_notes, key)

    if stage == "selftest":
        selftest()
        return
    if stage == "status":
        ns = Counter(ne["status"] for ne in state["notes"].values())
        ps = Counter(pe["status"] for pe in state["pairs"].values())
        ps["never_touched"] = len(pairs) - sum(ps.values())
        print(f"notes: {dict(ns)}")
        print(f"pairs: {dict(ps)}")
        print(f"spend: ${state['spend']['cost_usd']:.2f} of ${BUDGET_STOP_USD:.0f} "
              f"({state['spend']['calls']} auditor calls, "
              f"{state['spend']['plan_calls']} subscription calls, "
              f"{state['spend'].get('construct_calls', 0)} openrouter constructor calls)")
        print(f"by stage: {json.dumps(state['spend']['by_stage'])}")
        return
    if stage == "check0":
        check0(state, corpus_notes, pairs)
        return
    if stage == "audit":
        if not state.get("check0", {}).get("pass"):
            print("run check0 first")
            sys.exit(1)
        with LockedRun("w-a-audit", params={"stage": "audit", "lever": state["lever"],
                                            "seeds": AUDIT_SEEDS,
                                            "reasoning_effort": REASONING},
                       replicate=len(state["run_ids"]) + 1, seed=AUDIT_SEEDS[0],
                       inputs=INPUT_FILES, spec=SPEC) as run:
            run.register_prompts(PROMPTS)
            state["run_ids"].append(run.run_id)
            left = run_audit_stage(state, corpus_notes, chunk, workers)
            run.save("stage_summary.json", {"stage": "audit", "pending_after": left,
                                            "spend": state["spend"]})
        save_state(state)
        if BUDGET_HIT.is_set():
            print(f"BUDGET STOP: ${state['spend']['cost_usd']:.2f} >= ${BUDGET_STOP_USD}")
            sys.exit(3)
        sys.exit(0 if left == 0 else 2)
    if stage == "adjudicate":
        changed, pending = adjudicate_stage(state, corpus_notes)
        ns = Counter(ne["status"] for ne in state["notes"].values())
        print(f"adjudicate: {changed} notes moved, {pending} awaiting audit | {dict(ns)}")
        write_flags_file(state, corpus_notes)
        print("wrote master/wa_audit_flags.json")
        sys.exit(0 if pending == 0 else 2)
    if stage in ("repair", "regen"):
        # construction stages: same model as the old subscription path, now on OpenRouter,
        # so they are credit-spending paper-bound calls and get a Run manifest like
        # audit/editcheck.
        with LockedRun("w-a-construct", params={"stage": stage, "role": CONSTRUCT_ROLE,
                                                "transport": CONSTRUCT_TRANSPORT,
                                                "reasoning_effort": CONSTRUCT_EFFORT},
                       replicate=len(state["run_ids"]) + 1, seed=None,
                       inputs=INPUT_FILES, spec=SPEC) as run:
            run.register_prompts(PROMPTS)
            state["run_ids"].append(run.run_id)
            n = (repair_stage(state, corpus_notes, min(workers, CONSTRUCT_MAX_WORKERS), chunk)
                 if stage == "repair" else
                 regen_stage(state, corpus_notes, pairs, min(workers, CONSTRUCT_MAX_WORKERS), chunk))
            run.save("stage_summary.json", {"stage": stage, "ran": n,
                                            "spend": state["spend"]})
        save_state(state)
        sys.exit(0 if n == 0 else 2)
    if stage == "editcheck":
        with LockedRun("w-a-audit", params={"stage": "editcheck", "lever": state["lever"],
                                            "seeds": EDIT_SEEDS,
                                            "reasoning_effort": REASONING},
                       replicate=len(state["run_ids"]) + 1, seed=EDIT_SEEDS[0],
                       inputs=INPUT_FILES, spec=SPEC) as run:
            run.register_prompts(PROMPTS)
            state["run_ids"].append(run.run_id)
            left = editcheck_stage(state, corpus_notes, pairs, chunk, workers)
            run.save("stage_summary.json", {"stage": "editcheck", "pending_after": left,
                                            "spend": state["spend"]})
        save_state(state)
        if BUDGET_HIT.is_set():
            print(f"BUDGET STOP: ${state['spend']['cost_usd']:.2f} >= ${BUDGET_STOP_USD}")
            sys.exit(3)
        sys.exit(0 if left == 0 else 2)
    if stage == "severity-backfill":
        with LockedRun("w-a-construct", params={"stage": "severity", "role": CONSTRUCT_ROLE,
                                                "transport": CONSTRUCT_TRANSPORT,
                                                "reasoning_effort": CONSTRUCT_EFFORT},
                       replicate=len(state["run_ids"]) + 1, seed=None,
                       inputs=INPUT_FILES, spec=SPEC) as run:
            state["run_ids"].append(run.run_id)
            n = severity_stage(state, corpus_notes, pairs, min(workers, CONSTRUCT_MAX_WORKERS))
            run.save("stage_summary.json", {"stage": "severity", "pending_after": n,
                                            "spend": state["spend"]})
        save_state(state)
        sys.exit(0 if n == 0 else 2)
    if stage == "freeze":
        frozen, frozen_sha, dev_out, ideal_out = freeze(state, corpus_notes, pairs)
        report = build_reports(state, corpus_notes, pairs, frozen, frozen_sha,
                               dev_out, ideal_out)
        write_flags_file(state, corpus_notes)
        write_repair_log(state)
        write_injection_checks(state)
        write_v2_files(state, corpus_notes, ideal_out, frozen, dev_out)
        sit = write_sitting(state, corpus_notes, report)
        report["sitting_queue_n"] = sit["n_items"]
        json.dump(report, open(os.path.join(MASTER, "wa_gate_report.json"), "w"),
                  ensure_ascii=False, indent=1)
        complete_pair_verification(state, frozen, frozen_sha, report)
        print(report["gate"]["statement"])
        print(f"frozen: {frozen['counts']} | sitting queue: {sit['n_items']} | "
              f"spend ${state['spend']['cost_usd']:.2f}")
        return
    print(f"unknown stage {stage!r}")
    sys.exit(1)


if __name__ == "__main__":
    main()
