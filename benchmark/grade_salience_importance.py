"""Severity and salience grades for the 30 authored scenarios, backfilled onto traps
that predate the rubric.

Every salience_trap in authored_scenarios.json gains a fourth field
    "importance": "critical" | "supporting" | "peripheral"
graded per the instruction text below. Procedure:
  1. One opus call per scenario (plan path): transcript + all traps (trap + correct_handling +
     mode) in, per-trap importance grade + one-line rationale out.
  2. Critic check: the extraction pipeline's `materiality` critic restricted to one check -
     "any `importance` grade that is clearly wrong in either direction" - one call per scenario.
  3. Any trap the critic flags as materially wrong is re-graded ONCE with the critic's objection
     in context; flips are logged.
  4. Grades are written back INTO authored_scenarios.json (backup kept at
     authored_scenarios.json.bak-pre-severity - committed, the audit trail is the point).
     Rationales, critic flags and flips land in master/salience_importance_rationales.json.

Usage: python benchmark/grade_salience_importance.py [--only id1,id2] [--workers 5]
Idempotent: a scenario whose traps ALL already carry `importance` skips step 1; a scenario whose
sidecar entry already has a critic result skips step 2. Rerun to fill gaps after failures.
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import json, os, shutil, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from common import claude_json, HERE

MODEL = "claude-opus-4-8"
EFFORT = "medium"
TIMEOUT = 420
SCN_FILE = os.path.join(HERE, "authored_scenarios.json")
BAK_FILE = SCN_FILE + ".bak-pre-severity"
SIDECAR = os.path.join(HERE, "master", "salience_importance_rationales.json")
VALID = {"critical", "supporting", "peripheral"}


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


WORKERS = int(_arg("--workers", "5"))
ONLY = set((_arg("--only") or "").split(",")) - {""}

# The grading instruction appended verbatim to the salience_traps item spec, and the
# three grade definitions.
GRADE_INSTRUCTION = (
    'Add a fourth field "importance": "critical" if a careful clinician would consider missing this '
    'a significant safety or care gap, "supporting" if it\'s clinically relevant but wouldn\'t itself '
    'change management, "peripheral" if a competent note could reasonably compress it.'
)
DEFINITIONS = """\
"critical" = a careful clinician would consider missing this a significant safety/care gap (a working
diagnosis, a red-flag safety-net, a drug/allergy, a stated red-flag negative).
"supporting" = clinically relevant detail whose absence doesn't itself change management.
"peripheral" = present but a competent note could reasonably compress it."""

# The `materiality` critic, restricted to the importance check (the other checks are
# extraction-pipeline concerns and are audited elsewhere).
MATERIALITY_D = """You are a senior UK GP reviewing a fact sheet extracted from a consultation transcript for use in
scoring AI scribe notes. Judge CLINICAL MATERIALITY, not support (support is audited separately).
Flag ONLY: (d) any `importance` grade that is clearly wrong in either direction - in particular any
trap graded `peripheral` that a careful clinician would treat as a missed diagnosis, drug, or
safety-net. Be strict but fair - style is not materiality."""


def _traps_json(traps, with_importance=False):
    rows = []
    for i, t in enumerate(traps):
        row = {"index": i, "trap": t["trap"], "correct_handling": t["correct_handling"],
               "mode": t["mode"]}
        if with_importance:
            row["importance"] = t.get("importance")
        rows.append(row)
    return json.dumps(rows, ensure_ascii=False, indent=1)


def grade_scenario(scn):
    traps = scn["fact_sheet"]["salience_traps"]
    prompt = f"""You are a senior UK GP grading, for an AI-evaluation research study, how much each
"salience trap" in a mock consultation matters clinically. Each trap is a moment where an AI scribe
note could plausibly go wrong; you grade how important it is that a note handles it correctly.

{GRADE_INSTRUCTION}

Definitions:
{DEFINITIONS}

TRANSCRIPT:
{scn['transcript']}

SALIENCE_TRAPS (grade every one):
{_traps_json(traps)}

Output ONLY JSON: {{"grades": [ {{"index": <int>, "importance": "critical"|"supporting"|"peripheral",
"rationale": "<one line>"}} ]}} - exactly one entry per trap, in index order. Escape newlines inside
strings as \\n. No commentary, no markdown fences."""
    r = claude_json(prompt, model=MODEL, effort=EFFORT, timeout=TIMEOUT, retries=2)
    grades = (r or {}).get("grades") or []
    by_idx = {g.get("index"): g for g in grades
              if isinstance(g, dict) and g.get("importance") in VALID}
    if set(by_idx) != set(range(len(traps))):
        return None
    return [by_idx[i] for i in range(len(traps))]


def critic_scenario(scn):
    traps = scn["fact_sheet"]["salience_traps"]
    prompt = (f"{MATERIALITY_D}\n\nTRANSCRIPT:\n{scn['transcript']}\n\n"
              f"SALIENCE_TRAPS (each with its assigned `importance` grade):\n"
              f"{_traps_json(traps, with_importance=True)}\n\n"
              'Output ONLY JSON: {"verdict": "ok" | "issues", "issues": [ {"index": <int>, '
              '"graded": "<current grade>", "should_be": "critical"|"supporting"|"peripheral", '
              '"why": "<one line>", "severity": "minor"|"material"} ]}. Empty issues list if verdict '
              "is ok. Flag ONLY grades that are clearly wrong - do not relitigate borderline calls.")
    r = claude_json(prompt, model=MODEL, effort=EFFORT, timeout=TIMEOUT, retries=2)
    if not r:
        return None
    return {"verdict": r.get("verdict", "error"), "issues": r.get("issues") or []}


def regrade_trap(scn, idx, objection):
    t = scn["fact_sheet"]["salience_traps"][idx]
    prompt = f"""You are a senior UK GP. A review panel challenged the `importance` grade assigned to
one salience trap in a mock consultation used to score AI scribe notes. Re-grade it, weighing the
panel's objection on its merits - agree with the panel only if it is right.

{GRADE_INSTRUCTION}

Definitions:
{DEFINITIONS}

TRANSCRIPT:
{scn['transcript']}

TRAP:
{json.dumps({k: t[k] for k in ('trap', 'correct_handling', 'mode')}, ensure_ascii=False, indent=1)}

CURRENT GRADE: {t.get('importance')}
PANEL OBJECTION: {objection.get('why', '')} (panel suggests: {objection.get('should_be')})

Output ONLY JSON: {{"importance": "critical"|"supporting"|"peripheral", "rationale": "<one line>"}}.
No commentary, no markdown fences."""
    r = claude_json(prompt, model=MODEL, effort="high", timeout=TIMEOUT, retries=2)
    if r and r.get("importance") in VALID:
        return r
    return None


def main():
    scns = json.load(open(SCN_FILE))
    if not os.path.exists(BAK_FILE):
        shutil.copyfile(SCN_FILE, BAK_FILE)
        print(f"Backup written: {os.path.basename(BAK_FILE)}")

    sidecar = json.load(open(SIDECAR)) if os.path.exists(SIDECAR) else {}
    os.makedirs(os.path.dirname(SIDECAR), exist_ok=True)

    def save():
        json.dump(scns, open(SCN_FILE, "w"), ensure_ascii=False, indent=1)
        json.dump(sidecar, open(SIDECAR, "w"), ensure_ascii=False, indent=1)

    pick = [s for s in scns if not ONLY or s["id"] in ONLY]

    # ---- step 1: grade (skip scenarios already fully graded) ----
    todo = [s for s in pick
            if not all(t.get("importance") in VALID for t in s["fact_sheet"]["salience_traps"])]
    print(f"Grading {len(todo)}/{len(pick)} scenarios on {MODEL} ({WORKERS} workers).")
    if todo:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            results = list(ex.map(lambda s: (s["id"], grade_scenario(s)), todo))
        failed = []
        for sid, grades in results:
            if grades is None:
                failed.append(sid)
                continue
            scn = next(s for s in scns if s["id"] == sid)
            for i, g in enumerate(grades):
                scn["fact_sheet"]["salience_traps"][i]["importance"] = g["importance"]
            sidecar[sid] = {"grades": [{"index": g["index"], "importance": g["importance"],
                                        "rationale": g.get("rationale", "")} for g in grades]}
        save()
        if failed:
            print(f"GRADING FAILED (rerun to retry): {failed}")

    # ---- step 2: materiality critic, check (d) only (skip ids already checked) ----
    graded = [s for s in pick
              if all(t.get("importance") in VALID for t in s["fact_sheet"]["salience_traps"])]
    todo_c = [s for s in graded if "critic" not in sidecar.get(s["id"], {})]
    print(f"Critic check (d) on {len(todo_c)}/{len(graded)} scenarios.")
    if todo_c:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            results = list(ex.map(lambda s: (s["id"], critic_scenario(s)), todo_c))
        for sid, res in results:
            if res is None:
                print(f"CRITIC FAILED (rerun to retry): {sid}")
                continue
            sidecar.setdefault(sid, {})["critic"] = res
        save()

    # ---- step 3: re-grade materially-flagged traps once, log flips ----
    jobs = []
    for s in graded:
        entry = sidecar.get(s["id"], {})
        done_idx = {f["index"] for f in entry.get("flips", [])}
        for iss in entry.get("critic", {}).get("issues", []):
            idx = iss.get("index")
            if (iss.get("severity") == "material" and isinstance(idx, int)
                    and 0 <= idx < len(s["fact_sheet"]["salience_traps"])
                    and idx not in done_idx):
                jobs.append((s, idx, iss))
    print(f"Re-grading {len(jobs)} critic-flagged traps (material only).")
    if jobs:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            results = list(ex.map(lambda j: (j[0]["id"], j[1], j[2], regrade_trap(*j)), jobs))
        for sid, idx, iss, r in results:
            if r is None:
                print(f"RE-GRADE FAILED (rerun to retry): {sid}[{idx}]")
                continue
            scn = next(s for s in scns if s["id"] == sid)
            old = scn["fact_sheet"]["salience_traps"][idx].get("importance")
            scn["fact_sheet"]["salience_traps"][idx]["importance"] = r["importance"]
            sidecar.setdefault(sid, {}).setdefault("flips", []).append(
                {"index": idx, "from": old, "to": r["importance"], "flipped": old != r["importance"],
                 "critic_objection": iss.get("why", ""), "critic_suggested": iss.get("should_be"),
                 "regrade_rationale": r.get("rationale", "")})
        save()

    # ---- sanity report ----
    all_traps = [t for s in scns for t in s["fact_sheet"]["salience_traps"]]
    dist = Counter(t.get("importance", "UNGRADED") for t in all_traps)
    n_flags = sum(len(sidecar.get(s["id"], {}).get("critic", {}).get("issues", [])) for s in scns)
    n_material = sum(1 for s in scns
                     for i in sidecar.get(s["id"], {}).get("critic", {}).get("issues", [])
                     if i.get("severity") == "material")
    flips = [f for s in scns for f in sidecar.get(s["id"], {}).get("flips", []) if f["flipped"]]
    print(f"\nSanity report: {len(all_traps)} traps across {len(scns)} scenarios")
    for k in ("critical", "supporting", "peripheral", "UNGRADED"):
        if dist.get(k):
            print(f"  {k:11} {dist[k]:4d}  ({100 * dist[k] / len(all_traps):.1f}%)")
    print(f"  critic flags: {n_flags} ({n_material} material) | re-grades run: "
          f"{sum(len(sidecar.get(s['id'], {}).get('flips', [])) for s in scns)} | actual flips: {len(flips)}")
    for f in flips:
        print(f"    flip: {f['from']} -> {f['to']} | {f['critic_objection'][:90]}")


if __name__ == "__main__":
    main()
