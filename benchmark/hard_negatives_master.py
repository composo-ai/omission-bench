"""Build the master ideal notes and the balanced hard-negative pairs over the whole corpus.

Last of three, and the one to run. It imports the recipes from `hard_negatives.py` and
the balancing from `hard_negatives_balanced.py`; neither is superseded by this file.

Generalisation of hard_negatives_balanced.py to the master consult list. What it builds:

  master/ideal_notes_master.json   107 clean notes:
    - trapblind (9):  build_ideal from hard_negatives.py (reused verbatim; model
      patched to claude-opus-5) over the CORE fact-sheet view: core-scope
      must_contain only; post-excise must_not_contain; all salience_traps
      handled correctly.
    - primock (53) + aci (45): the VERIFIED repaired reference note copied
      through from master/refs_<source>.json field ref_note_verified (the
      clean-note substrate rule - NEVER the original clinician reference, which
      is kept only for the original-vs-verified reference comparison). No
      regeneration.

  master/hard_negatives_master.json   pairs, one list:
    - authored stratum: the existing 90 pairs preserved VERBATIM from
      hard_negatives_balanced.json (pair ids preserved, no regeneration); only
      stratum/pair_id/provenance metadata added.
    - trapblind/primock/aci: per consult exactly 3 pairs (add/change/omit), one
      injected error each, TYPES + INJECT reused verbatim from
      hard_negatives_balanced.py. Driver-level named diffs only:
      the clean note comes from build_ideal (trapblind) or ref_note_verified
      (primock/aci), and each injection is TARGETED - pair-eligible traps
      preferentially (peripheral traps are excluded), core
      must_contain facts as the omit fallback, untargeted generic otherwise.
      Severity per pair: the target trap's importance grade; trap-less pairs are
      graded against the severity rubric (one claude-opus-5 call over the
      injected error).

  master/pair_verification.json   construction-manifest STUB only (counts,
    model, prompt shas, balance check). The FULL single-edit verification and
    the ground-truth audit belong to the ground-truth gate and have NOT run.

Model: claude-opus-5, run against a Claude subscription rather than an API key,
effort medium. Chunked + resumable: per-(stratum,id,type) state in
master/hn_master_state.json, saved
after every completed call; --max-seconds N exits cleanly mid-batch (in-flight
calls finish); rerun until complete. Plan rate/usage-limit errors are detected
on stderr and STOP the batch (exit 3) instead of grinding against the subscription's limit.

Usage:
  python3 benchmark/hard_negatives_master.py [--workers 4] [--max-seconds N]
                                   [--only id1,id2] [--limit N] [--manifest]
Exit codes: 0 complete (outputs written) | 2 incomplete (rerun) | 3 plan-limit stop.
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import hashlib, json, os, random, re, subprocess, sys, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from common import HERE
import hard_negatives
from hard_negatives import IDEAL, build_ideal
from hard_negatives_balanced import INJECT, TYPES

MODEL = "claude-opus-5"   # the construction model
EFFORT = "medium"         # the construction reasoning effort
TIMEOUT = 420
ATTEMPTS = 3
SEED = 20260728           # the study seed
T0 = time.time()


def M(name):
    return os.path.join(HERE, "master", name)


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


WORKERS = int(_arg("--workers", "4"))   # cap 3-4: the subscription is shared with interactive use
MAX_SECONDS = float(_arg("--max-seconds", "0"))
ONLY = set((_arg("--only") or "").split(",")) - {""}
LIMIT = int(_arg("--limit", "0"))
MANIFEST_ONLY = "--manifest" in sys.argv


def out_of_budget():
    return MAX_SECONDS and time.time() - T0 > MAX_SECONDS


def sha16(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def fs_sha(rec):
    return hashlib.sha256(json.dumps(rec["fact_sheet"], sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- transport
# common.claude-equivalent invocation (same argv, cwd=/tmp) that ALSO captures
# stderr: a subscription rate/usage-limit error sets STOP so the batch halts gracefully
# instead of burning the subscription's quota on retries. Everything else degrades to ""
# exactly like common.claude.
LIMIT_PATTERNS = ("usage limit", "rate limit", "limit reached", "hit your limit",
                  "out of extra usage", "overloaded", "quota", "insufficient credit",
                  "credit balance")
STOP = threading.Event()
STOP_REASON = []


def claude_text(prompt, model, effort, timeout):
    cmd = ["claude", "-p", prompt, "--output-format", "text", "--strict-mcp-config",
           "--model", model, "--effort", effort]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd="/tmp")
    except Exception:
        return ""
    out = (r.stdout or "").strip()
    if r.returncode != 0 or not out:
        blob = ((r.stdout or "") + "\n" + (r.stderr or "")).lower()
        if any(p in blob for p in LIMIT_PATTERNS):
            if not STOP.is_set():
                STOP_REASON.append(((r.stderr or r.stdout or "").strip())[-400:])
                STOP.set()
        return ""
    return out


def claude_json(prompt, timeout=TIMEOUT, model=MODEL, effort=EFFORT, retries=0):
    """Signature-compatible with common.claude_json (build_ideal is routed here)."""
    for _ in range(retries + 1):
        out = claude_text(prompt, model, effort, timeout)
        m = re.search(r"(\{.*\}|\[.*\])", out, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        if STOP.is_set():
            return None
    return None


# route build_ideal's calls through the limit-aware transport + the amended model.
# build_ideal's own logic/prompt (IDEAL) is reused untouched.
hard_negatives.claude_json = claude_json
hard_negatives.MODEL = MODEL


# ---------------------------------------------------------------- prompts (driver additions)
# TYPES + INJECT are verbatim imports. The only prompt additions live in the
# driver, never in the imported prompts: a target line appended
# to the {instr} slot, and the rubric-grade prompt for trap-less pairs.
TARGET_TRAP = """

Realise the error at THIS specific point in the consultation (a documented trap moment where
a real scribe could plausibly go wrong):
TRAP: {trap}
The note currently handles it correctly: {correct}
Your single edit must make the note get exactly this point wrong."""

TARGET_FACT = """

Realise the omission at THIS specific fact (currently correctly present in the note):
FACT: {fact}
Remove exactly this fact from the note."""

#: The severity rubric. Not carried by this repository: it is released with the data, as
#: the `rubric` field of `validation/sitting_severity_items.json` in the dataset
#: repository, beside the items the graders saw and the sha256 they were hashed against.
#: Save it to a file and point SEVERITY_RUBRIC at it, or drop it at the path below.
RUBRIC_FILE = os.environ.get("SEVERITY_RUBRIC",
                             os.path.join(HERE, "master", "severity-rubric.md"))
_RUBRIC = None


def read_rubric():
    """The rubric text, read on first use so that importing this module does not need it."""
    global _RUBRIC
    if _RUBRIC is None:
        if not os.path.exists(RUBRIC_FILE):
            raise SystemExit(
                "cannot find the severity rubric at %s. It is released as the `rubric` "
                "field of validation/sitting_severity_items.json in the dataset "
                "repository; save it to a file and set SEVERITY_RUBRIC to its path."
                % RUBRIC_FILE)
        with open(RUBRIC_FILE, encoding="utf-8") as fh:
            _RUBRIC = fh.read()
    return _RUBRIC


def __getattr__(name):
    # Serves `hard_negatives_master.RUBRIC` for the three modules that read it that way.
    # Bare `RUBRIC` inside this module does NOT route here - those call read_rubric().
    if name == "RUBRIC":
        return read_rubric()
    raise AttributeError(name)

RUBRIC_GRADE = """You are a senior UK GP grading, for an AI-evaluation research study, the clinical
importance of ONE known defect deliberately introduced into an otherwise-correct clinical
note of the consultation below. Grade how much this defect matters: if the note carried this
error, how much would it matter for the patient or the next clinician who reads the note?
Grade STRICTLY by the rubric below - apply its decision procedure in order and its
tie-break; do not substitute your own scale.

===== RUBRIC (verbatim; the instrument under evaluation) =====
{rubric}
===== END RUBRIC =====

TRANSCRIPT:
{t}

THE DEFECT (type: {typ} - add = fabricated content, change = altered assertion,
omit = removed content):
edit made: {change}
specific fact involved: {what}

Walk the rubric's decision procedure (1 action test, 2 safety test, 3 record-quality test,
4 otherwise peripheral) and apply the conservative tie-break if genuinely torn.
Output ONLY JSON: {{"importance": "critical"|"supporting"|"peripheral",
"test_fired": "action"|"safety"|"record_quality"|"none", "rationale": "<one line>"}}.
No commentary, no markdown fences."""

PROMPT_SHA = {
    "ideal (hard_negatives.IDEAL, verbatim)": sha16(IDEAL),
    "inject (hard_negatives_balanced.INJECT, verbatim)": sha16(INJECT),
    "type_add (verbatim)": sha16(TYPES["add"]),
    "type_change (verbatim)": sha16(TYPES["change"]),
    "type_omit (verbatim)": sha16(TYPES["omit"]),
    "target_trap_line (driver addition)": sha16(TARGET_TRAP),
    "target_fact_line (driver addition)": sha16(TARGET_FACT),
    "rubric_grade (driver addition)": sha16(RUBRIC_GRADE),
}


def prompt_sha():
    """`PROMPT_SHA` with the rubric's own hash, for a run manifest. Needs the rubric file."""
    return dict(PROMPT_SHA, severity_rubric_md=sha16(read_rubric()))


# ---------------------------------------------------------------- inputs
STRATA = ("trapblind", "primock", "aci")
REF_STRATA = ("primock", "aci")


def load_rows():
    refs = {src: {r["id"]: r for r in json.load(open(M(f"refs_{src}.json")))}
            for src in REF_STRATA}
    rows = []
    for src in STRATA:
        recs = json.load(open(M(f"fact_sheets_{src}_core.json")))
        for rec in recs:
            row = {"stratum": src, "id": rec["id"], "rec": rec}
            if src in REF_STRATA:
                ref = refs[src].get(rec["id"])
                assert ref and (ref.get("ref_note_verified") or "").strip(), \
                    f"{src}/{rec['id']}: no ref_note_verified"
                row["ref"] = ref
            rows.append(row)
        n_refs = len(refs.get(src, {}))
        if src in REF_STRATA:
            assert n_refs == len(recs), f"{src}: refs ({n_refs}) != core sheets ({len(recs)})"
    return rows


def load_authored_pairs():
    pairs = json.load(open(os.path.join(HERE, "hard_negatives_balanced.json")))
    keys = Counter((p["id"], p["type"]) for p in pairs)
    ids = {p["id"] for p in pairs}
    assert len(pairs) == 90 and len(ids) == 30 and all(v == 1 for v in keys.values()), \
        f"authored pairs not the expected 30x3 set: n={len(pairs)}, ids={len(ids)}"
    ideals = json.load(open(os.path.join(HERE, "ideal_notes_balanced.json")))
    gaps = [i["id"] for i in ideals if not i.get("ok")]
    assert not gaps, f"authored-stratum ideal-note gaps (would need build_ideal): {gaps}"
    return pairs


# ---------------------------------------------------------------- target selection
# Deterministic seeded choice (seed 20260728 + stratum|id|type). Preference:
# importance critical > supporting (pair_eligible already excludes peripheral),
# then - for change - the modes TYPES["change"] names explicitly,
# then a per-candidate seeded jitter.
IMP_RANK = {"critical": 0, "supporting": 1}
CHANGE_MODES = ["negation", "dose_value", "modality_hardening", "decision_status",
                "laterality", "attribution", "temporal", "anchoring"]


def pick_target(rec, stratum, typ):
    fs = rec["fact_sheet"]
    rng = random.Random(f"{SEED}|{stratum}|{rec['id']}|{typ}")
    traps = [(i, t) for i, t in enumerate(fs["salience_traps"]) if t.get("pair_eligible")]

    def trap_target(cands, mode_rank):
        if not cands:
            return None
        jit = {i: rng.random() for i, _ in cands}
        i, t = min(cands, key=lambda it: (IMP_RANK.get(it[1].get("importance"), 2),
                                          mode_rank.get(it[1].get("mode"), 99), jit[it[0]]))
        return {"kind": "trap", "index": i, "trap": t["trap"],
                "correct_handling": t["correct_handling"], "mode": t.get("mode"),
                "importance": t.get("importance")}

    if typ == "omit":
        got = trap_target([(i, t) for i, t in traps if t.get("mode") == "omission"], {})
        if got:
            return got
        mcs = [(i, m) for i, m in enumerate(fs["must_contain"]) if m.get("scope") == "core"]
        jit = {i: rng.random() for i, _ in mcs}
        i, m = min(mcs, key=lambda it: (0 if it[1].get("load_bearing") == "high" else 1,
                                        jit[it[0]]))
        return {"kind": "must_contain", "index": i, "fact": m["fact"],
                "load_bearing": m.get("load_bearing")}
    if typ == "change":
        rank = {m: r for r, m in enumerate(CHANGE_MODES)}
        return trap_target([(i, t) for i, t in traps if t.get("mode") in rank], rank)
    if typ == "add":
        return trap_target([(i, t) for i, t in traps if t.get("mode") == "fabrication"], {})
    raise ValueError(typ)


# ---------------------------------------------------------------- stages
def norm(text):
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def valid_errored(clean, errored, typ):
    if not errored or norm(errored) == norm(clean):
        return False
    ratio = len(errored) / max(len(clean), 1)
    return 0.3 <= ratio <= 2.5


def run_inject(transcript, clean, typ, target):
    instr = TYPES[typ]
    if target and target["kind"] == "trap":
        instr += TARGET_TRAP.format(trap=target["trap"], correct=target["correct_handling"])
    elif target and target["kind"] == "must_contain":
        instr += TARGET_FACT.format(fact=target["fact"])
    prompt = INJECT.format(t=transcript[:40000], note=clean, instr=instr)
    for attempt in range(1, ATTEMPTS + 1):
        res = claude_json(prompt)
        if isinstance(res, dict) and isinstance(res.get("note"), str):
            err = res["note"].strip()
            if valid_errored(clean, err, typ):
                return {"errored": err, "change": res.get("change"),
                        "what": res.get("what"), "attempt": attempt}
        if STOP.is_set() or out_of_budget():
            break
    return None


def run_rubric_grade(transcript, typ, inj):
    prompt = RUBRIC_GRADE.format(rubric=read_rubric(), t=transcript[:40000], typ=typ,
                                 change=inj.get("change") or "",
                                 what=json.dumps(inj.get("what"), ensure_ascii=False)
                                 if not isinstance(inj.get("what"), str) else inj["what"])
    for attempt in range(1, ATTEMPTS + 1):
        res = claude_json(prompt)
        if (isinstance(res, dict)
                and res.get("importance") in ("critical", "supporting", "peripheral")):
            return {"importance": res["importance"],
                    "test_fired": res.get("test_fired"),
                    "rationale": res.get("rationale"), "attempt": attempt}
        if STOP.is_set() or out_of_budget():
            break
    return None


# ---------------------------------------------------------------- driver
def main():
    rows = load_rows()
    authored_pairs = load_authored_pairs()
    if ONLY:
        rows = [r for r in rows if r["id"] in ONLY]

    statefile = M("hn_master_state.json")
    state = (json.load(open(statefile)) if os.path.exists(statefile)
             else {"ideals": {}, "pairs": {}})
    lock = threading.Lock()

    def save_state():
        tmp = statefile + ".tmp"
        json.dump(state, open(tmp, "w"), ensure_ascii=False, indent=1)
        os.replace(tmp, statefile)

    # ---- stale-state guards (sheet or clean-note substrate changed since)
    by_id = {r["id"]: r for r in rows}
    state["ideals"] = {rid: e for rid, e in state["ideals"].items()
                       if rid in by_id and e.get("fs_sha") == fs_sha(by_id[rid]["rec"])}

    def clean_note_of(row):
        if row["stratum"] in REF_STRATA:
            return row["ref"]["ref_note_verified"].strip()
        e = state["ideals"].get(row["id"])
        return (e or {}).get("note") or None

    def pair_key(row, typ):
        return f"{row['stratum']}|{row['id']}|{typ}"

    fresh_pairs = {}
    for row in rows:
        clean = clean_note_of(row)
        for typ in TYPES:
            k = pair_key(row, typ)
            e = state["pairs"].get(k)
            if e and e.get("fs_sha") == fs_sha(row["rec"]) and clean \
                    and e.get("clean_sha") == sha16(clean):
                fresh_pairs[k] = e
    state["pairs"] = fresh_pairs

    # ---- worklists
    ideal_todo = [r for r in rows if r["stratum"] == "trapblind"
                  and not (state["ideals"].get(r["id"], {}).get("ok"))]

    def pair_stage(row, typ):
        e = state["pairs"].get(pair_key(row, typ))
        if not e or "inject" not in e or e["inject"] is None:
            return "inject"
        needs_grade = e["target"] is None or e["target"]["kind"] != "trap"
        if needs_grade and not e.get("severity"):
            return "grade"
        return "done"

    def pending_pairs():
        out = []
        for row in rows:
            if clean_note_of(row) is None:
                continue   # trapblind ideal not built yet
            for typ in TYPES:
                if pair_stage(row, typ) != "done":
                    out.append((row, typ))
        return out

    n_pairs_done = sum(1 for row in rows for typ in TYPES
                       if clean_note_of(row) is not None and pair_stage(row, typ) == "done")
    print(f"consults: {len(rows)} ({Counter(r['stratum'] for r in rows)}) | "
          f"authored pairs preserved: {len(authored_pairs)} | "
          f"ideals pending: {len(ideal_todo)} | pairs done: {n_pairs_done}/{3 * len(rows)} | "
          f"{MODEL} plan path, effort {EFFORT}, {WORKERS} workers"
          + (f", budget {MAX_SECONDS:.0f}s" if MAX_SECONDS else "")
          + (f", limit {LIMIT}" if LIMIT else ""), flush=True)

    if not MANIFEST_ONLY:
        # ---- phase 1: trapblind ideal notes (gate their pairs)
        if ideal_todo and not out_of_budget():
            def one_ideal(row):
                if STOP.is_set() or out_of_budget():
                    return
                fs = row["rec"]["fact_sheet"]
                shim = {"id": row["id"], "transcript": row["rec"]["transcript"],
                        "fact_sheet": {
                            "must_contain": [m["fact"] for m in fs["must_contain"]
                                             if m.get("scope") == "core"],
                            "must_not_contain": [f"{m['assertion']} (why wrong: {m['why_wrong']})"
                                                 for m in fs["must_not_contain"]],
                            "salience_traps": [{"trap": t["trap"],
                                                "correct_handling": t["correct_handling"]}
                                               for t in fs["salience_traps"]]}}
                got = build_ideal(shim)
                with lock:
                    if got["ok"]:
                        state["ideals"][row["id"]] = {"fs_sha": fs_sha(row["rec"]),
                                                      "note": got["note"], "ok": True}
                        print(f"  ideal {row['id']:28} built ({len(got['note'])} chars)",
                              flush=True)
                        save_state()
                    else:
                        print(f"  ideal {row['id']:28} FAILED (rerun to retry)", flush=True)
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                list(ex.map(one_ideal, ideal_todo))

        # ---- phase 2: pairs
        todo = pending_pairs()
        if LIMIT:
            todo = todo[:LIMIT]

        def one_pair(job):
            row, typ = job
            if STOP.is_set() or out_of_budget():
                return
            k = pair_key(row, typ)
            clean = clean_note_of(row)
            with lock:
                e = state["pairs"].setdefault(k, {"fs_sha": fs_sha(row["rec"]),
                                                  "clean_sha": sha16(clean)})
                if "target" not in e:
                    e["target"] = pick_target(row["rec"], row["stratum"], typ)
                st = pair_stage(row, typ)
                target = e["target"]
            if st == "inject":
                got = run_inject(row["rec"]["transcript"], clean, typ, target)
                with lock:
                    if got is None:
                        print(f"  {k:44} inject FAILED (rerun to retry)", flush=True)
                        return
                    e["inject"] = got
                    tgt = (f"trap#{target['index']}({target['importance']})"
                           if target and target["kind"] == "trap"
                           else f"mc#{target['index']}" if target else "untargeted")
                    print(f"  {k:44} injected [{tgt}] attempt {got['attempt']}", flush=True)
                    save_state()
                st = pair_stage(row, typ)
            if st == "grade" and not (STOP.is_set() or out_of_budget()):
                with lock:
                    inj = e["inject"]
                got = run_rubric_grade(row["rec"]["transcript"], typ, inj)
                with lock:
                    if got is None:
                        print(f"  {k:44} rubric-grade FAILED (rerun to retry)", flush=True)
                        return
                    e["severity"] = got
                    print(f"  {k:44} rubric-graded {got['importance']}", flush=True)
                    save_state()

        if todo and not (STOP.is_set() or out_of_budget()):
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                list(ex.map(one_pair, todo))

    if STOP.is_set():
        save_state()
        print(f"\nPLAN-LIMIT STOP - halting gracefully; progress saved. Detail: "
              f"{(STOP_REASON or ['(pattern match in claude -p output)'])[0]}", flush=True)
        sys.exit(3)

    # ---- completeness check
    missing_ideals = [r["id"] for r in rows if r["stratum"] == "trapblind"
                      and not state["ideals"].get(r["id"], {}).get("ok")]
    unfinished = [pair_key(row, typ) for row in rows for typ in TYPES
                  if clean_note_of(row) is None or pair_stage(row, typ) != "done"]
    if missing_ideals or unfinished:
        save_state()
        print(f"\nINCOMPLETE - ideals missing: {len(missing_ideals)}, pairs unfinished: "
              f"{len(unfinished)} (rerun to resume): {unfinished[:5]}"
              f"{'...' if len(unfinished) > 5 else ''}", flush=True)
        sys.exit(2)

    if ONLY or LIMIT:
        save_state()
        print("\ncomplete for the requested subset - rerun without --only/--limit to finalize",
              flush=True)
        sys.exit(2)

    finalize(rows, authored_pairs, state)


# ---------------------------------------------------------------- finalize
def finalize(rows, authored_pairs, state):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ideal_notes_master.json - 107 rows (authored clean notes live on in
    # ideal_notes_balanced.json; their pairs are preserved, so no row here)
    ideals_out = []
    for row in rows:
        if row["stratum"] == "trapblind":
            e = state["ideals"][row["id"]]
            ideals_out.append({
                "id": row["id"], "stratum": "trapblind", "note": e["note"], "ok": True,
                "provenance": {
                    "method": "built_ideal_core",
                    "detail": "hard_negatives.build_ideal (IDEAL prompt verbatim) over the "
                              "core fact-sheet view: core-scope must_contain, post-excise "
                              "must_not_contain (assertion + why_wrong), all salience_traps",
                    "model": MODEL, "route": "plan", "transport": "claude -p",
                    "effort": EFFORT}})
        else:
            v = row["ref"]["verification"]
            ideals_out.append({
                "id": row["id"], "stratum": row["stratum"],
                "note": row["ref"]["ref_note_verified"].strip(), "ok": True,
                "provenance": {
                    "method": "copied_ref_note_verified",
                    "detail": f"copied verbatim from master/refs_{row['stratum']}.json field "
                              "ref_note_verified (the reference-note audit-and-repair "
                              "output; the clean-note substrate rule - never the original "
                              "reference)",
                    "verification_pass": bool(v.get("pass")),
                    "repair_edit_count": row["ref"].get("repair_edit_count")}})

    # hard_negatives_master.json - authored preserved + 321 new pairs
    pairs_out = []
    for p in authored_pairs:
        q = dict(p)
        q["pair_id"] = f"authored|{p['id']}|{p['type']}"
        q["stratum"] = "authored"
        q["provenance"] = {
            "construction": "preserved verbatim from hard_negatives_balanced.json "
                            "(pair ids preserved, no regeneration); metadata fields "
                            "added only",
            "model": "claude-opus-4-8",
            "clean_note_source": "ideal_notes_balanced.json (build_ideal, authored fact sheet)"}
        pairs_out.append(q)

    def clean_note_of(row):
        return (row["ref"]["ref_note_verified"].strip() if row["stratum"] in REF_STRATA
                else state["ideals"][row["id"]]["note"])

    for row in rows:
        for typ in TYPES:
            e = state["pairs"][f"{row['stratum']}|{row['id']}|{typ}"]
            inj, target = e["inject"], e["target"]
            if target and target["kind"] == "trap":
                severity, sev_src = target["importance"], "trap_importance"
            else:
                severity, sev_src = e["severity"]["importance"], "rubric_graded"
            rec = {
                "pair_id": f"{row['stratum']}|{row['id']}|{typ}",
                "id": row["id"], "stratum": row["stratum"], "type": typ,
                "clean": clean_note_of(row), "errored": inj["errored"],
                "change": inj.get("change"), "what": inj.get("what"),
                "target": target, "severity": severity, "severity_source": sev_src,
                "provenance": {
                    "model": MODEL, "route": "plan", "effort": EFFORT,
                    "inject_attempt": inj["attempt"],
                    "clean_note_source": ("ref_note_verified" if row["stratum"] in REF_STRATA
                                          else "built_ideal_core")}}
            if sev_src == "rubric_graded":
                rec["severity_rationale"] = e["severity"].get("rationale")
                rec["severity_test_fired"] = e["severity"].get("test_fired")
            pairs_out.append(rec)

    for name, obj in (("ideal_notes_master.json", ideals_out),
                      ("hard_negatives_master.json", pairs_out)):
        tmp = M(name) + ".tmp"
        json.dump(obj, open(tmp, "w"), ensure_ascii=False, indent=1)
        os.replace(tmp, M(name))

    # pair_verification.json - construction manifest STUB (the ground-truth gate runs the
    # real verification)
    new_pairs = [p for p in pairs_out if p["stratum"] != "authored"]
    per = {}
    for st in ("authored",) + STRATA:
        per[st] = dict(Counter(p["type"] for p in pairs_out if p["stratum"] == st))
    type_tot = Counter(p["type"] for p in pairs_out)
    n = len(pairs_out)
    share_ok = all(abs(type_tot[t] / n - 1 / 3) <= 0.05 for t in TYPES)
    inputs = ["master/fact_sheets_trapblind_core.json", "master/fact_sheets_primock_core.json",
              "master/fact_sheets_aci_core.json", "master/refs_primock.json",
              "master/refs_aci.json", "hard_negatives_balanced.json",
              "ideal_notes_balanced.json", "master/severity-rubric.md"]
    manifest = {
        "status": "pending_ground_truth_gate",
        "note": "Construction-manifest STUB only. The FULL pair verification - the single-edit "
                "check plus the ground-truth audit (ideal-note audit + EDIT_CHECK) - has NOT "
                "run; the ground-truth gate owns it and produces pairs_master_frozen.json. "
                "Nothing downstream may consume hard_negatives_master.json directly.",
        "generated_by": "hard_negatives_master.py", "generated_utc": now,
        "construction_rules": ("authored pairs preserved verbatim; claude-opus-5 run against "
                               "a Claude subscription; severity from the target trap's "
                               "importance grade or else from the rubric; core-scope "
                               "substrate and pair-eligible traps only"),
        "construction": {"model": MODEL, "route": "plan",
                         "transport": "claude -p (common.claude-equivalent argv + stderr "
                                      "limit detection)",
                         "effort": EFFORT, "attempts_per_call": ATTEMPTS, "seed": SEED,
                         "workers": WORKERS,
                         "authored_stratum": "preserved verbatim, claude-opus-4-8 (no calls)"},
        "inputs_sha256_16": {f: sha16(open(os.path.join(HERE, f)).read()) for f in inputs},
        "prompt_sha256_16": prompt_sha(),
        "counts": {
            "pairs_total": n, "pairs_by_stratum_type": per,
            "pairs_by_type": dict(type_tot),
            "consults": {"authored": 30, **{s: sum(1 for r in rows if r["stratum"] == s)
                                            for s in STRATA}},
            "ideal_notes_master": dict(Counter(i["provenance"]["method"] for i in ideals_out)),
            "authored_ideal_gaps_rebuilt": 0,
            "targeting": dict(Counter(
                (p["target"] or {}).get("kind") or "untargeted" for p in new_pairs)),
            "severity_new_pairs": dict(Counter(p["severity"] for p in new_pairs)),
            "severity_source": dict(Counter(p["severity_source"] for p in new_pairs)),
        },
        "balance_gate": {
            "corpus_in_360_450": 360 <= n <= 450,
            "each_type_within_5pp_of_third": share_ok,
            "note": "pre-QA construction balance; re-checked after the ground-truth gate "
                    "drops pairs"},
        "verification": None,
    }
    tmp = M("pair_verification.json") + ".tmp"
    json.dump(manifest, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, M("pair_verification.json"))

    print(f"\ncomplete -> master/ideal_notes_master.json ({len(ideals_out)} notes: "
          f"{manifest['counts']['ideal_notes_master']}) | master/hard_negatives_master.json "
          f"({n} pairs: {per}) | master/pair_verification.json (stub, pending the "
          "ground-truth gate)", flush=True)
    print(f"targeting: {manifest['counts']['targeting']} | severity (new pairs): "
          f"{manifest['counts']['severity_new_pairs']} via {manifest['counts']['severity_source']}",
          flush=True)


if __name__ == "__main__":
    main()
