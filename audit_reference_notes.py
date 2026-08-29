"""W-D step 6 - reference-note audit-and-repair (the WD-R3 reading) over the kept
PriMock/ACI consults.

Per consult, ONE claude-opus-5 plan-path call runs the spec 3.3 audit-and-repair prompt
VERBATIM (transcript + CORE fact sheet + original clinician reference note in; audit
discrepancies + minimally-repaired verified note out), with one Amendment-H-style addendum
appended to the Step-1 item spec: each audit item also carries "rubric_severity" (graded per
specs/severity-rubric.md, condensed exactly as the v2 extraction prompt condensed it) and a
short verbatim "evidence" quote. The 3.3 severity axis (material|minor) is untouched - it is
what the pre-registered WD-R3 reading (spec section 6.5) is computed on; rubric_severity is
additional reporting granularity.

Then a SEPARATE verification call per repaired note (prompt constructed for this step - the
spec's section 3 defines no verification prompt; built per the spec's audit principles):
does the note carry the clinical substance of EVERY core must_contain item, assert NO
must_not_contain item, and contain nothing unsupported by the transcript? Pass/fail is
computed from the per-item verdicts, never trusted from the model's own flag. On failure:
one fix call (effort high - this stage's revise() analog) with the exact failure list,
then one recheck. Still-failing consults are logged, never silently kept as clean.

Inputs (the finalized master corpus - kept sheets only):
  primock  53 kept sheets from master/fact_sheets_primock_core.json (v2.1 instrument,
           transcript-only view); original clinician note = the `summary` field of
           experiments/ai_scribe/dataset/primock57_parsed.json, joined by id.
  aci      45 kept sheets from master/fact_sheets_aci_core.json; original reference note =
           the `ref_note` field of master/aci_subsample.json, joined by id.
The CORE fact-sheet view passed to the auditor: must_contain filtered to scope=="core"
(Amendment Q: ideal notes and pairs build from core items only - the verified note is that
substrate, so core is the coverage bar), must_not_contain as kept (post-excise),
salience_traps with importance (pair_eligible stripped - pair machinery, not audit input).

Outputs:
  master/refs_audited_<source>.json   per consult: id, ref_note_original, audit[],
                                      ref_note_verified, edit_summary, repair_edit_count,
                                      verification, audit_provenance
  master/refs_<source>.json           identical copy under the spec section-5 step-6 name
                                      (the W5 handoff filename; both kept in sync)
  master/wd_r3_report.json            --report: WD-R3 per spec section 6.5 - per-corpus
                                      fraction with >=1 material discrepancy (Wilson 95% CI),
                                      breakdown by kind/severity/rubric_severity, the >=20%
                                      pre-registered verdict, examples, verification outcomes

Idempotent by id + fact-sheet sha + note sha at STAGE granularity (audit -> verify -> fix ->
reverify; state: master/refs_audit_state_<source>.json) - a budget-interrupted consult resumes
at its next missing stage. --max-seconds bounds one invocation (no NEW call starts after the
budget; in-flight calls finish) so a source runs as several blocking chunks; rerun until
complete. Model: claude-opus-5, plan path via common.claude_json (Amendments E/M).

Usage:
  python audit_reference_notes.py --source primock|aci [--workers 6] [--max-seconds S]
                                  [--only id1,id2] [--limit N]
  python audit_reference_notes.py --report
"""
import hashlib, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from common import claude_json, load_primock, HERE

MODEL = "claude-opus-5"   # Amendment 2026-07-30 part M
EFFORT = "medium"         # spec section 4 construction row (ref-audit at medium)
FIX_EFFORT = "high"       # the fix pass is this stage's revise() analog (revise runs high)
TIMEOUT = 420
ATTEMPTS = 3
T0 = time.time()
M = lambda *p: os.path.join(HERE, "master", *p)

KINDS = {"missing_fact", "contains_error", "unsupported", "hardened_uncertainty"}
SEVERITIES = {"material", "minor"}
RUBRIC = {"critical", "supporting", "peripheral"}

# ---------------------------------------------------------------- prompts
# Spec 3.3 VERBATIM. One addendum (marked) appended to the Step-1 item spec, in the exact
# pattern Amendment 2026-07-29b H used on the 3.1 prompt: rubric_severity graded per
# specs/severity-rubric.md (condensed as in the v2 extraction prompt) + a verbatim evidence
# quote per item. Everything else is byte-faithful to the spec.
AUDIT_PROMPT = """You are a UK GP auditing an existing clinician-written note of a consultation against (1) the
consultation TRANSCRIPT (the only source of truth) and (2) an audited FACT SHEET of that transcript.

TRANSCRIPT:
{transcript}

FACT_SHEET:
{fact_sheet_json}

EXISTING NOTE:
{ref_note}

Step 1 - AUDIT. List every discrepancy in the existing note, each as:
  { "kind": "missing_fact" | "contains_error" | "unsupported" | "hardened_uncertainty",
    "detail": "<what exactly>",
    "fact_ref": "<the must_contain fact or must_not_contain assertion involved, or null>",
    "severity": "material" | "minor" }
missing_fact = a must_contain fact absent from the note. contains_error = the note asserts a
must_not_contain item or contradicts the transcript. unsupported = a note claim with no transcript
support (even if clinically plausible). hardened_uncertainty = the transcript hedged, the note is
definite. Judge severity clinically: material = could change patient understanding or management.
Add two further fields to each audit item:
  "rubric_severity": the clinical consequence of THIS discrepancy standing in the note, graded
    per the study's severity rubric - "critical" if the error would plausibly change clinical
    action, delay or misdirect care, or create a safety risk (an omitted working diagnosis or a
    hedged impression hardened into certainty; omitted red-flag safety-netting; an omitted or
    wrong drug, allergy, dose, frequency or duration; a conditional plan made definite; a
    fabricated examination, finding or result; an omitted pertinent negative that licenses the
    management decision; wrong patient identity attributes used clinically); "supporting" if it
    degrades the note's completeness, clarity or defensibility but is unlikely to change what
    happens next; "peripheral" if it has no plausible clinical consequence. Decide in order:
    (1) ACTION test - would a competent GP, or the next clinician reading the note, plausibly DO
    something different if this error stood? yes -> critical. (2) SAFETY test - even if action
    today is unchanged, does the error remove a safety net or create a latent risk? yes ->
    critical. (3) RECORD-QUALITY test - does it materially weaken the note as a clinical record?
    yes -> supporting. (4) Otherwise peripheral. If genuinely torn between two grades, take the
    LOWER grade; grade each discrepancy independently and never curve to a distribution.
  "evidence": a short VERBATIM quote (max 25 words) from the transcript or the existing note
    that shows the discrepancy.

Step 2 - REPAIR. Produce a VERIFIED version of the note that fixes every material discrepancy with
MINIMAL edits: add missing facts in the appropriate section, delete or correct erroneous/unsupported
content, restore hedging. Preserve the original note's structure, section headings, ordering and
style wherever untouched - this must read as the same clinician's note, corrected, not a rewrite.
Fix minor discrepancies only when the fix is a strict improvement with no style cost.

Output ONLY JSON: { "audit": [ ... ], "verified_note": "<full corrected note>",
"edit_summary": "<one line per edit made>" }. Escape newlines inside strings as \\n."""

# Constructed for this step (no verification prompt exists in the spec's section 3); built per
# the spec's audit principles: transcript-only ground truth, semantic not string matching, the
# v2 equivalence protection (faithful conversions are never failures), and a computed - never
# self-reported - pass.
VERIFY_PROMPT = """You are a senior UK GP verifying a corrected clinical note for an AI-evaluation research study.
Below are the consultation TRANSCRIPT (the only source of truth), the numbered core facts and
forbidden assertions from an audited fact sheet of that transcript, and a REPAIRED NOTE produced
by an audit-and-repair pass. Run three checks:

1. COVERAGE - for EVERY numbered must_contain fact: is its clinical substance present in the
   note? Judge semantic equivalence, not string match: notes compress, and a faithful clinical
   rendering (including correct unit/dose/frequency conversions of what was said) counts as
   present.
2. ERRORS - does the note assert any numbered must_not_contain assertion, or a clinically
   equivalent rendering of it? Use each assertion's why_wrong to scope it: only the genuinely
   wrong version counts, never a faithful rendering of what was actually said.
3. SUPPORT - does the note contain any claim the transcript does not establish (even if
   clinically plausible), or state as definite anything the transcript only hedged? Standard
   clinical phrasing, section headings, and faithful conversions are not failures; flag only
   content that adds or hardens clinical meaning beyond the transcript.

TRANSCRIPT:
{transcript}

MUST_CONTAIN (numbered):
{mc}

MUST_NOT_CONTAIN (numbered, each with why it is wrong):
{mnc}

REPAIRED NOTE:
{note}

Output ONLY JSON:
{ "coverage": [ { "item": <number>, "present": true | false, "where_or_why": "<one line>" } ],
  "errors_asserted": [ { "item": <number>, "where": "<one line: where the note asserts it>" } ],
  "unsupported": [ { "claim": "<the note's claim>", "why": "<one line>" } ] }
"coverage" carries one entry per numbered must_contain fact, every number exactly once;
"errors_asserted" and "unsupported" are empty lists when clean. Escape newlines inside strings
as \\n."""

# Constructed for this step - the fix half of the one fix-and-recheck cycle.
FIX_PROMPT = """You are a UK GP making a minimal correction pass on a clinical note for an AI-evaluation
research study. The note below was repaired once already, but an independent verification found
the remaining FAILURES listed. Fix EVERY listed failure with MINIMAL edits and change nothing
else: add each missing must_contain fact in the appropriate section; remove or correct each
asserted error; delete or reword each unsupported claim so the note claims nothing the
TRANSCRIPT (the only source of truth) does not establish, restoring hedging rather than deleting
where the transcript hedged. Preserve the note's structure, section headings, ordering and style
wherever untouched - it must still read as the same clinician's note, corrected, not a rewrite.

TRANSCRIPT:
{transcript}

MUST_CONTAIN (numbered):
{mc}

MUST_NOT_CONTAIN (numbered, each with why it is wrong):
{mnc}

NOTE TO FIX:
{note}

FAILURES:
{failures}

Output ONLY JSON: { "verified_note": "<full corrected note>", "edit_summary": "<one line per
edit made>" }. Escape newlines inside strings as \\n."""

PROMPT_SHA = {k: hashlib.sha256(v.encode()).hexdigest()[:16] for k, v in
              [("audit", AUDIT_PROMPT), ("verify", VERIFY_PROMPT), ("fix", FIX_PROMPT)]}


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


SOURCE = _arg("--source")
WORKERS = int(_arg("--workers", "6"))
MAX_SECONDS = float(_arg("--max-seconds", "0"))
ONLY = set((_arg("--only") or "").split(",")) - {""}
LIMIT = int(_arg("--limit", "0"))
STEMS = {"primock": "primock", "aci": "aci"}


def out_of_budget():
    return MAX_SECONDS and time.time() - T0 > MAX_SECONDS


def numbered(lines):
    return "\n".join(f"{i + 1}. {t}" for i, t in enumerate(lines))


def sha16(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def fs_sha(rec):
    return hashlib.sha256(json.dumps(rec["fact_sheet"], sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- inputs
def load_rows(source):
    """Kept core sheets joined with the original clinician reference note. PriMock's core
    records are already transcript-only (v2.1); the reference note is the AUDIT SUBJECT the
    3.3 prompt takes as input, not a ground-truth view, so joining it here is per spec."""
    stem = STEMS[source]
    recs = json.load(open(M(f"fact_sheets_{stem}_core.json")))
    if source == "primock":
        notes = {r["id"]: r["summary"] for r in load_primock()}
    else:
        notes = {r["id"]: r["ref_note"] for r in json.load(open(M("aci_subsample.json")))}
    rows = []
    for r in recs:
        note = (notes.get(r["id"]) or "").strip()
        assert note, f"{r['id']}: no original reference note found"
        rows.append({"rec": r, "note": note})
    return rows


def core_view(rec):
    """The CORE fact-sheet view the auditor sees (Amendment Q: core is the coverage bar)."""
    fs = rec["fact_sheet"]
    return {
        "must_contain": [{"fact": it["fact"], "evidence": it["evidence"],
                          "load_bearing": it.get("load_bearing")}
                         for it in fs["must_contain"] if it.get("scope") == "core"],
        "must_not_contain": [{"assertion": it["assertion"], "why_wrong": it["why_wrong"]}
                             for it in fs["must_not_contain"]],
        "salience_traps": [{"trap": t["trap"], "correct_handling": t["correct_handling"],
                            "mode": t.get("mode"), "importance": t.get("importance")}
                           for t in fs["salience_traps"]],
    }


def mc_mnc_blocks(rec):
    view = core_view(rec)
    mc = numbered([it["fact"] for it in view["must_contain"]])
    mnc = numbered([f"{it['assertion']} (why wrong: {it['why_wrong']})"
                    for it in view["must_not_contain"]]) or "(none)"
    return view, mc, mnc


# ---------------------------------------------------------------- stages
def valid_audit(obj):
    """Shape-check the audit call's output; return (audit, verified_note, edit_summary) or None."""
    if not isinstance(obj, dict):
        return None
    audit, note, summ = obj.get("audit"), obj.get("verified_note"), obj.get("edit_summary", "")
    if not isinstance(audit, list) or not isinstance(note, str) or len(note.strip()) < 40:
        return None
    for it in audit:
        if not (isinstance(it, dict) and it.get("kind") in KINDS
                and isinstance(it.get("detail"), str) and it["detail"].strip()
                and it.get("severity") in SEVERITIES
                and it.get("rubric_severity") in RUBRIC):
            return None
        it.setdefault("fact_ref", None)
        if not isinstance(it.get("evidence"), str):
            it["evidence"] = ""
    if not isinstance(summ, str):
        summ = json.dumps(summ, ensure_ascii=False) if summ else ""
    return audit, note.strip(), summ


def run_audit(row):
    view = core_view(row["rec"])
    prompt = (AUDIT_PROMPT
              .replace("{transcript}", row["rec"]["transcript"])
              .replace("{fact_sheet_json}", json.dumps(view, ensure_ascii=False, indent=1))
              .replace("{ref_note}", row["note"]))
    for attempt in range(1, ATTEMPTS + 1):
        r = claude_json(prompt, model=MODEL, effort=EFFORT, timeout=TIMEOUT, retries=0)
        got = valid_audit(r)
        if got:
            return {"audit": got[0], "verified_note": got[1], "edit_summary": got[2],
                    "attempt": attempt}
        if out_of_budget():
            break
    return None


def valid_verify(obj, n_mc, n_mnc):
    """Shape-check the verification output and COMPUTE pass from the per-item verdicts."""
    if not isinstance(obj, dict):
        return None
    cov = obj.get("coverage")
    if not (isinstance(cov, list) and len(cov) == n_mc
            and all(isinstance(c, dict) and isinstance(c.get("item"), int)
                    and isinstance(c.get("present"), bool) for c in cov)
            and {c["item"] for c in cov} == set(range(1, n_mc + 1))):
        return None
    errs = [e for e in (obj.get("errors_asserted") or [])
            if isinstance(e, dict) and isinstance(e.get("item"), int)
            and 1 <= e["item"] <= n_mnc]
    uns = [u for u in (obj.get("unsupported") or [])
           if isinstance(u, dict) and isinstance(u.get("claim"), str) and u["claim"].strip()]
    missing = sorted(c["item"] for c in cov if not c["present"])
    return {"coverage": sorted(cov, key=lambda c: c["item"]), "missing": missing,
            "errors_asserted": errs, "unsupported": uns,
            "pass": not missing and not errs and not uns}


def run_verify(row, note):
    view, mc, mnc = mc_mnc_blocks(row["rec"])
    n_mc, n_mnc = len(view["must_contain"]), len(view["must_not_contain"])
    prompt = (VERIFY_PROMPT
              .replace("{transcript}", row["rec"]["transcript"])
              .replace("{mc}", mc).replace("{mnc}", mnc).replace("{note}", note))
    for _ in range(ATTEMPTS):
        r = claude_json(prompt, model=MODEL, effort=EFFORT, timeout=TIMEOUT, retries=0)
        got = valid_verify(r, n_mc, n_mnc)
        if got:
            return got
        if out_of_budget():
            break
    return None


def failure_lines(row, verify):
    view = core_view(row["rec"])
    lines = []
    for c in verify["coverage"]:
        if not c["present"]:
            lines.append(f"missing (must_contain {c['item']}): "
                         f"{view['must_contain'][c['item'] - 1]['fact']} - {c.get('where_or_why', '')}")
    for e in verify["errors_asserted"]:
        lines.append(f"asserts error (must_not_contain {e['item']}): "
                     f"{view['must_not_contain'][e['item'] - 1]['assertion']} - {e.get('where', '')}")
    for u in verify["unsupported"]:
        lines.append(f"unsupported: {u['claim']} - {u.get('why', '')}")
    return "\n".join(lines)


def run_fix(row, note, verify):
    _, mc, mnc = mc_mnc_blocks(row["rec"])
    prompt = (FIX_PROMPT
              .replace("{transcript}", row["rec"]["transcript"])
              .replace("{mc}", mc).replace("{mnc}", mnc).replace("{note}", note)
              .replace("{failures}", failure_lines(row, verify)))
    for _ in range(ATTEMPTS):
        r = claude_json(prompt, model=MODEL, effort=FIX_EFFORT, timeout=TIMEOUT, retries=0)
        if (isinstance(r, dict) and isinstance(r.get("verified_note"), str)
                and len(r["verified_note"].strip()) >= 40):
            summ = r.get("edit_summary", "")
            if not isinstance(summ, str):
                summ = json.dumps(summ, ensure_ascii=False) if summ else ""
            return {"verified_note": r["verified_note"].strip(), "edit_summary": summ}
        if out_of_budget():
            break
    return None


# ---------------------------------------------------------------- driver
def _count_edits(*summaries):
    n = 0
    for s in summaries:
        for line in (s or "").split("\n"):
            t = line.strip().strip("-* ").lower()
            if t and not t.startswith(("no edit", "none", "n/a", "no change", "no correction")):
                n += 1
    return n


def run_source(source):
    stem = STEMS[source]
    rows = load_rows(source)
    if ONLY:
        rows = [r for r in rows if r["rec"]["id"] in ONLY]
    statefile = M(f"refs_audit_state_{stem}.json")
    state = json.load(open(statefile)) if os.path.exists(statefile) else {}
    # drop stale state (sheet or note changed since - e.g. a stratum re-run)
    keyed = {r["rec"]["id"]: r for r in rows}
    state = {rid: e for rid, e in state.items()
             if rid in keyed and e.get("fs_sha") == fs_sha(keyed[rid]["rec"])
             and e.get("note_sha") == sha16(keyed[rid]["note"])}

    def stage_of(rid):
        e = state.get(rid)
        if not e or "audit" not in e:
            return "audit"
        if "verify1" not in e:
            return "verify1"
        if e["verify1"]["pass"]:
            return "done"
        if "fix" not in e:
            return "fix"
        if e["fix"] is None:
            return "done"          # fix call itself failed after ATTEMPTS on a full run - logged
        if "verify2" not in e:
            return "verify2"
        return "done"

    todo = [r for r in rows if stage_of(r["rec"]["id"]) != "done"]
    if LIMIT:
        todo = todo[:LIMIT]
    n_done = len(rows) - len([r for r in rows if stage_of(r["rec"]["id"]) != "done"])
    print(f"[{source}] {len(rows)} kept consults | {n_done} complete | {len(todo)} to advance "
          f"on {MODEL} (plan path, effort={EFFORT}/fix {FIX_EFFORT}, {WORKERS} workers"
          + (f", limit {LIMIT}" if LIMIT else "")
          + (f", budget {MAX_SECONDS:.0f}s" if MAX_SECONDS else "") + ")", flush=True)

    lock = threading.Lock()

    def save_state():
        tmp = statefile + ".tmp"
        json.dump(state, open(tmp, "w"), ensure_ascii=False, indent=1)
        os.replace(tmp, statefile)

    def note_of(e):
        return (e.get("fix") or {}).get("verified_note") or e["audit_out"]["verified_note"]

    def one(row):
        rid = row["rec"]["id"]
        while True:
            with lock:
                e = state.setdefault(rid, {"fs_sha": fs_sha(row["rec"]),
                                           "note_sha": sha16(row["note"])})
                st = stage_of(rid)
            if st == "done" or out_of_budget():
                return
            if st == "audit":
                got = run_audit(row)
                with lock:
                    if got is None:
                        print(f"  {rid}: audit FAILED (rerun to retry)", flush=True)
                        return
                    e["audit"] = got["audit"]
                    e["audit_out"] = {"verified_note": got["verified_note"],
                                      "edit_summary": got["edit_summary"],
                                      "attempt": got["attempt"]}
                    n_mat = sum(1 for a in got["audit"] if a["severity"] == "material")
                    print(f"  {rid:24} audited: {len(got['audit'])} discrepancies "
                          f"({n_mat} material), attempt {got['attempt']}", flush=True)
                    save_state()
            elif st in ("verify1", "verify2"):
                with lock:
                    note = e["audit_out"]["verified_note"] if st == "verify1" else note_of(e)
                got = run_verify(row, note)
                with lock:
                    if got is None:
                        print(f"  {rid}: {st} FAILED (rerun to retry)", flush=True)
                        return
                    e[st] = got
                    tag = "PASS" if got["pass"] else (f"fail (missing {len(got['missing'])}, "
                                                     f"errs {len(got['errors_asserted'])}, "
                                                     f"unsupported {len(got['unsupported'])})")
                    print(f"  {rid:24} {st}: {tag}", flush=True)
                    save_state()
            elif st == "fix":
                with lock:
                    v1 = e["verify1"]
                got = run_fix(row, e["audit_out"]["verified_note"], v1)
                with lock:
                    e["fix"] = got   # None = fix call failed; stage_of treats as done + logged
                    print(f"  {rid:24} fix: {'ok' if got else 'FAILED (kept audit-pass note)'}",
                          flush=True)
                    save_state()

    if todo:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(one, todo))

    incomplete = [r["rec"]["id"] for r in rows if stage_of(r["rec"]["id"]) != "done"]
    if incomplete:
        print(f"[{source}] INCOMPLETE - {len(incomplete)} consults mid-pipeline (rerun): "
              f"{incomplete[:6]}{'...' if len(incomplete) > 6 else ''}", flush=True)
        return False

    # ---- finalize: write refs_audited_<source>.json (+ the spec-named refs_<source>.json copy)
    out = []
    for row in rows:
        rid = row["rec"]["id"]
        e = state[rid]
        fix = e.get("fix") or None
        final_verify = e.get("verify2") or e["verify1"]
        final_pass = bool(final_verify["pass"])
        rec = {
            "id": rid, "source": stem,
            "ref_note_original": row["note"],
            "audit": e["audit"],
            "ref_note_verified": note_of(e),
            "edit_summary": e["audit_out"]["edit_summary"]
                            + (("\n" + fix["edit_summary"]) if fix else ""),
            "repair_edit_count": _count_edits(e["audit_out"]["edit_summary"],
                                              fix and fix["edit_summary"]),
            "verification": {
                "pass": final_pass,
                "first_pass": bool(e["verify1"]["pass"]),
                "fix_applied": bool(fix),
                "fix_call_failed": "fix" in e and e["fix"] is None,
                "final": {k: final_verify[k] for k in ("missing", "errors_asserted",
                                                       "unsupported")},
                "core_mc_items": sum(1 for it in row["rec"]["fact_sheet"]["must_contain"]
                                     if it.get("scope") == "core"),
            },
            "audit_provenance": {
                "model": MODEL, "route": "plan", "transport": "claude -p",
                "effort": EFFORT, "fix_effort": FIX_EFFORT,
                "audit_attempt": e["audit_out"]["attempt"],
                "fact_sheet_view": "core must_contain only (Amendment Q); post-excise "
                                   "must_not_contain; traps with importance",
                "prompts": {"audit": "spec 3.3 verbatim + rubric_severity/evidence addendum "
                                     "(Amendment-H style, condensed from specs/severity-rubric.md)",
                            "verify": "constructed for this step (no verification prompt in "
                                      "spec section 3)",
                            "fix": "constructed for this step (fix half of the one "
                                   "fix-and-recheck cycle)"},
                "prompt_sha256_16": PROMPT_SHA,
                "spec": "w-d-master-dataset.md 3.3 + section 5 step 6 + Amendments M (opus-5), "
                        "Q (core view), R (severity rubric)"
                        + (", T1 (transcript-only view)" if source == "primock" else ""),
            },
        }
        out.append(rec)
    for name in (f"refs_audited_{stem}.json", f"refs_{stem}.json"):
        tmp = M(name) + ".tmp"
        json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1)
        os.replace(tmp, M(name))
    n_mat = sum(1 for r in out if any(a["severity"] == "material" for a in r["audit"]))
    still = [r["id"] for r in out if not r["verification"]["pass"]]
    print(f"[{source}] complete -> master/refs_audited_{stem}.json (+ refs_{stem}.json copy) | "
          f"{len(out)} notes | >=1 material: {n_mat} ({100 * n_mat / len(out):.0f}%) | "
          f"verification still-failing: {len(still)}{' ' + str(still) if still else ''}",
          flush=True)
    return True


# ---------------------------------------------------------------- WD-R3 report
def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def pick_examples(rows, n=3):
    mats = [(r["id"], a) for r in rows for a in r["audit"] if a["severity"] == "material"]
    out, seen_kinds, seen = [], set(), set()
    for pass_unique in (True, False):
        for rid, a in mats:
            key = (rid, a["kind"], a["detail"][:80])
            if key in seen or (pass_unique and a["kind"] in seen_kinds):
                continue
            out.append({"id": rid, "kind": a["kind"], "severity": a["severity"],
                        "rubric_severity": a["rubric_severity"], "detail": a["detail"],
                        "fact_ref": a.get("fact_ref"), "evidence": a.get("evidence", "")})
            seen.add(key), seen_kinds.add(a["kind"])
            if len(out) == n:
                return out
    return out


def run_report():
    corpora = {}
    for source, stem in STEMS.items():
        path = M(f"refs_audited_{stem}.json")
        if not os.path.exists(path):
            print(f"[report] refs_audited_{stem}.json missing - run --source {source} first")
            return False
        rows = json.load(open(path))
        n = len(rows)
        n_mat = sum(1 for r in rows if any(a["severity"] == "material" for a in r["audit"]))
        by_kind = {k: {"total": 0, "material": 0, "minor": 0} for k in sorted(KINDS)}
        by_rubric = {k: 0 for k in ("critical", "supporting", "peripheral")}
        n_disc = n_disc_mat = 0
        for r in rows:
            for a in r["audit"]:
                n_disc += 1
                by_kind[a["kind"]]["total"] += 1
                by_kind[a["kind"]][a["severity"]] += 1
                by_rubric[a["rubric_severity"]] += 1
                n_disc_mat += a["severity"] == "material"
        frac = n_mat / n
        corpora[stem] = {
            "n_notes_audited": n,
            "n_with_material_discrepancy": n_mat,
            "fraction_with_material": round(frac, 4),
            "wilson_95ci": wilson(n_mat, n),
            "pre_registered_verdict": (
                "SUBSECTION SHIPS (>=20% material - 'the wall is made of the same bricks'; "
                "W5's original-vs-verified comparison becomes a headline figure)"
                if frac >= 0.20 else
                "below 20% - reported as a robustness check only"),
            "discrepancies": {
                "total": n_disc, "material": n_disc_mat, "minor": n_disc - n_disc_mat,
                "mean_per_note": round(n_disc / n, 2),
                "mean_material_per_note": round(n_disc_mat / n, 2),
                "by_kind": by_kind, "by_rubric_severity": by_rubric},
            "repair": {
                "mean_edit_count": round(sum(r["repair_edit_count"] for r in rows) / n, 2),
                "notes_with_zero_edits": sum(1 for r in rows if r["repair_edit_count"] == 0)},
            "verification": {
                "first_pass": sum(1 for r in rows if r["verification"]["first_pass"]),
                "needed_fix": sum(1 for r in rows if r["verification"]["fix_applied"]),
                "pass_after_fix": sum(1 for r in rows if r["verification"]["fix_applied"]
                                      and r["verification"]["pass"]),
                "still_failing": [r["id"] for r in rows if not r["verification"]["pass"]]},
            "examples": pick_examples(rows),
        }
    report = {
        "spec": "w-d-master-dataset.md section 1 WD-R3 + section 6.5; audit instrument: "
                "section 3.3 verbatim + rubric_severity addendum (severity-rubric.md)",
        "generated_by": "audit_reference_notes.py --report",
        "pre_registered_reading": (
            ">=20% of reference notes in either corpus contain at least one MATERIAL "
            "discrepancy (spec 3.3 severity axis) -> the paper gains a standalone subsection "
            "and W5 quantifies the downstream judging delta on the original-vs-verified pair; "
            "a null (references nearly clean) is also publishable."),
        "denominator_note": "fractions are over the KEPT master-corpus consults (53 PriMock "
                            "v2.1 + 45 ACI sheets that passed extraction critique); dropped "
                            "consults have no audited fact sheet and are not audited.",
        "model": {"model": MODEL, "route": "plan", "effort": EFFORT, "fix_effort": FIX_EFFORT},
        "per_corpus": corpora,
        "headline": {stem: {"fraction_with_material": c["fraction_with_material"],
                            "wilson_95ci": c["wilson_95ci"],
                            "meets_20pct": c["fraction_with_material"] >= 0.20}
                     for stem, c in corpora.items()},
    }
    out = M("wd_r3_report.json")
    json.dump(report, open(out, "w"), ensure_ascii=False, indent=1)
    for stem, c in corpora.items():
        print(f"[{stem}] {c['n_with_material_discrepancy']}/{c['n_notes_audited']} notes with "
              f">=1 material ({100 * c['fraction_with_material']:.1f}%, CI "
              f"{c['wilson_95ci']}) - {'>=20% MET' if c['fraction_with_material'] >= .2 else 'below 20%'}")
    print(f"-> {os.path.relpath(out, HERE)}")
    return True


if __name__ == "__main__":
    if "--report" in sys.argv:
        ok = run_report()
    else:
        assert SOURCE in STEMS, "usage: --source primock|aci | --report"
        ok = run_source(SOURCE)
    sys.exit(0 if ok is not False else 0)   # incomplete chunks exit 0; rerun until complete
