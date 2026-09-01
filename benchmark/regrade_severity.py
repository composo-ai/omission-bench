"""Regrade every trap's severity against the written rubric, with two independent graders.

An earlier blind-recovery reading of fact-sheet provenance
(master/wd_provenance_report.json) put cross-model agreement on the severity axis at kappa
0.177 - but the two graders worked WITHOUT a shared written standard (opus-4-8 graded from
a one-paragraph instruction; opus-5 graded natively inside the extraction call), on
independently-worded trap lists joined by a semantic matcher. This run asks whether a
written rubric stabilises the axis: regrade every trap under the FULL rubric (version 1 of
the instrument) with two models independently, and report with-rubric vs without-rubric
agreement - either answer is the reported finding.

Scope: the 211 authored traps (authored_scenarios.json, 30 scenarios) + the 472 traps on
the 27 kept blind-extracted sheets (master/fact_sheets_authored_extracted.json). Neither
source file is modified; all outputs are new files under master/.

Arms - each grades all 57 sheets, one call per sheet; graders see trap + correct_handling
only (blinded to mode and to every prior grade):
  A  opus5  claude-opus-5, effort medium, plan path (common.claude_json), 4 workers
  B  gpt55  gpt-5.5 via OpenRouter (common.llm, role "auditor" in models.lock.json:
            temperature 1.0, reasoning_effort high per the lock's auditor notes; k=1,
            seed recorded per call; generation ids + credit cost recorded per call)

Usage:
  python benchmark/regrade_severity.py --arm opus5 [--workers 4] [--limit N]   # grade (resumable)
  python benchmark/regrade_severity.py --arm gpt55 [--workers 6] [--limit N]
  python benchmark/regrade_severity.py --analyse                               # no LLM calls

Idempotent per (arm, sheet, scenario): finished units are skipped on rerun; grades,
rationales, generation ids and spend accumulate in master/severity_regrade_state.json
(atomic tmp+replace writes). --analyse then writes:
  master/severity_rubric_grades.json      per-trap: both rubric grades + rationales, the
                                          opus48_backfill / opus5_native joins, consensus
  master/severity_stability_report.json   the kappa table (with cluster-bootstrap CIs),
                                          distributions, confusions, flagged disagreements
  master/sitting_severity_items.json      20 blinded items for the author-clinician grading
                                          sitting (seed 20260809)

Join note: master/wd_provenance_matches.json holds two generations of match calls
(early un-hashed keys, later content-hashed keys). Joins here use the entry keyed to the
CURRENT sheet content - hashed key first, un-hashed fallback only when its trap counts
match the current sheets. Under that rule 170 matched pairs join cleanly; the report's
171st pair traces to a superseded pre-revision sheet (chest_pain_onset, 20-trap v0) and is
not joinable to current content. The restated earlier kappa is quoted from the report
verbatim regardless.
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import hashlib, json, os, random, re, sys, threading, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from common import claude_json, llm, HERE

# wd_provenance_compare: reuse the verified kappa + cluster-bootstrap implementation and
# the cache-key function (import executes only cheap module-level setup; main() is guarded)
import wd_provenance_compare as wpc

OPUS_MODEL = "claude-opus-5"
OPUS_EFFORT = "medium"
OPUS_TIMEOUT = 540
GPT_ROLE = "auditor"            # models.lock.json -> openai/gpt-5.5, route order ["openai"]
GPT_TEMPERATURE = 1.0           # lock's grid convention; auditor entry pins no other value
GPT_REASONING = "high"          # per the lock's auditor notes ("reasoning_effort high")
GPT_MAX_TOKENS = 16000          # covers reasoning + ~23 grades; llm() raises on empty
GPT_SEED0 = 20260809
SITTING_SEED = 20260809
NBOOT = 10000
VALID = ("critical", "supporting", "peripheral")
RANK = {"critical": 2, "supporting": 1, "peripheral": 0}   # lower rank = lower grade
TESTS = ("action", "safety", "record_quality", "none")

SCN_FILE = os.path.join(HERE, "authored_scenarios.json")
EXT_FILE = os.path.join(HERE, "master", "fact_sheets_authored_extracted.json")
#: The severity rubric, released with the data as the `rubric` field of
#: validation/sitting_severity_items.json. Save it beside the corpus or point
#: SEVERITY_RUBRIC at it.
RUBRIC_FILE = os.environ.get("SEVERITY_RUBRIC",
                             os.path.join(HERE, "master", "severity-rubric.md"))
MATCHES_FILE = os.path.join(HERE, "master", "wd_provenance_matches.json")
PROV_REPORT = os.path.join(HERE, "master", "wd_provenance_report.json")
# per-arm state files so the two arms can run as concurrent PROCESSES without a
# read-modify-write race on one shared file (the threading.Lock only covers one process)
STATE_FILES = {arm: os.path.join(HERE, "master", f"severity_regrade_state_{arm}.json")
               for arm in ("opus5", "gpt55")}
GRADES_OUT = os.path.join(HERE, "master", "severity_rubric_grades.json")
REPORT_OUT = os.path.join(HERE, "master", "severity_stability_report.json")
SITTING_OUT = os.path.join(HERE, "master", "sitting_severity_items.json")


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


ARM = _arg("--arm")
WORKERS = int(_arg("--workers", "4"))
LIMIT = int(_arg("--limit", "0"))          # 0 = no cap this invocation
ANALYSE = "--analyse" in sys.argv

_RUBRIC = None


def read_rubric():
    """The rubric text, read on first use so that importing this module does not need it.

    Not carried by this repository: it is released with the data, as the `rubric` field of
    `validation/sitting_severity_items.json` in the dataset repository, beside the sha256
    every run in this file hashes it against. Set SEVERITY_RUBRIC to a copy of it.
    """
    global _RUBRIC
    if _RUBRIC is None:
        path = os.environ.get("SEVERITY_RUBRIC", RUBRIC_FILE)
        if not os.path.exists(path):
            raise SystemExit(
                "cannot find the severity rubric at %s. It is released as the `rubric` "
                "field of validation/sitting_severity_items.json in the dataset "
                "repository; save it to a file and set SEVERITY_RUBRIC to its path."
                % path)
        with open(path, encoding="utf-8") as fh:
            _RUBRIC = fh.read()
    return _RUBRIC


def rubric_sha():
    """The rubric's sha256, which every run in this file records and checks its state against."""
    return hashlib.sha256(read_rubric().encode()).hexdigest()


def __getattr__(name):
    # Serves qualified access from another module. Bare `RUBRIC` / `RUBRIC_SHA` inside this
    # module does NOT route here - those call read_rubric() and rubric_sha().
    if name == "RUBRIC":
        return read_rubric()
    if name == "RUBRIC_SHA":
        return rubric_sha()
    raise AttributeError(name)

PROMPT = """You are a senior UK GP grading, for an AI-evaluation research study, the clinical
importance of "salience traps" in a consultation - moments where an AI scribe note could
plausibly go wrong. For each trap you grade how much it matters that the note gets this
RIGHT vs wrong. Grade STRICTLY by the rubric below - apply its decision procedure in order
and its tie-break; do not substitute your own scale.

===== RUBRIC (verbatim; the instrument under evaluation) =====
{rubric}
===== END RUBRIC =====

TRANSCRIPT:
{transcript}

SALIENCE_TRAPS (grade every one; each = what is tempting to get wrong + what a correct
note does):
{traps}

For each trap, walk the rubric's decision procedure (1 action test, 2 safety test,
3 record-quality test, 4 otherwise peripheral) and apply the conservative tie-break if
genuinely torn. Grade each trap independently; do not curve to a distribution.

Output ONLY JSON:
{{"grades": [ {{"index": <int>, "importance": "critical"|"supporting"|"peripheral",
"test_fired": "action"|"safety"|"record_quality"|"none",
"rationale": "<ONE line: which rubric test fired (or that none did) and why, for THIS trap>"}} ]}}
Exactly one entry per trap, in index order. "test_fired" = the first procedure step that
fired ("none" if it fell through to peripheral). Escape newlines inside strings as \\n.
No commentary, no markdown fences."""


# ---------------------------------------------------------------- units + state
def load_units():
    """57 grading units in deterministic order: 30 authored sheets + 27 extracted sheets."""
    authored = json.load(open(SCN_FILE))
    extracted = json.load(open(EXT_FILE))
    units = []
    for s in authored:
        units.append({"unit": f"authored|{s['id']}", "sheet": "authored", "sid": s["id"],
                      "transcript": s["transcript"],
                      "traps": s["fact_sheet"]["salience_traps"]})
    for e in extracted:
        units.append({"unit": f"extracted|{e['id']}", "sheet": "extracted", "sid": e["id"],
                      "transcript": e["transcript"],
                      "traps": e["fact_sheet"]["salience_traps"]})
    return units


_state_lock = threading.Lock()


def load_state(arm):
    f = STATE_FILES[arm]
    return json.load(open(f)) if os.path.exists(f) else {
        "rubric_sha256": rubric_sha(), "units": {}}


def save_state(arm, state):
    f = STATE_FILES[arm]
    json.dump(state, open(f + ".tmp", "w"), ensure_ascii=False, indent=1)
    os.replace(f + ".tmp", f)


def _traps_payload(traps):
    """Graders see trap + correct_handling ONLY - no mode, no prior importance grade."""
    return json.dumps([{"index": i, "trap": t["trap"],
                        "correct_handling": t["correct_handling"]}
                       for i, t in enumerate(traps)], ensure_ascii=False, indent=1)


def _validate(r, n):
    """Exactly one valid entry per index -> normalized list, else None."""
    grades = (r or {}).get("grades") or []
    by_idx = {}
    for g in grades:
        if (isinstance(g, dict) and isinstance(g.get("index"), int)
                and g.get("importance") in VALID and g["index"] not in by_idx):
            by_idx[g["index"]] = {
                "importance": g["importance"],
                "test_fired": g.get("test_fired") if g.get("test_fired") in TESTS else None,
                "rationale": str(g.get("rationale", ""))[:400]}
    if set(by_idx) != set(range(n)):
        return None
    return [by_idx[i] for i in range(n)]


def grade_unit(arm, u, state):
    prompt = PROMPT.format(rubric=read_rubric(), transcript=u["transcript"],
                           traps=_traps_payload(u["traps"]))
    n = len(u["traps"])
    if arm == "opus5":
        for _ in range(3):
            r = claude_json(prompt, model=OPUS_MODEL, effort=OPUS_EFFORT,
                            timeout=OPUS_TIMEOUT, retries=1)
            grades = _validate(r, n)
            if grades:
                rec = {"grades": grades, "model": OPUS_MODEL, "effort": OPUS_EFFORT,
                       "route": "plan (claude -p)", "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
                break
        else:
            return False
    else:  # gpt55 via OpenRouter, one attempt per seed so a bad sample can't repeat itself
        rec = None
        for attempt in range(3):
            seed = GPT_SEED0 + attempt
            try:
                texts, meta = llm(prompt, role=GPT_ROLE, temperature=GPT_TEMPERATURE, k=1,
                                  seed=seed, reasoning_effort=GPT_REASONING,
                                  max_tokens=GPT_MAX_TOKENS, timeout=600, max_retries=3)
            except Exception as ex:
                print(f"  {u['unit']:44} gpt55 call error: {str(ex)[:120]}", flush=True)
                continue
            m = re.search(r"(\{.*\})", texts[0], re.S)
            grades = _validate(json.loads(m.group(1)) if m else None, n) if m else None
            if grades:
                rec = {"grades": grades, "model": meta["model"], "seed": seed,
                       "temperature": GPT_TEMPERATURE, "reasoning_effort": GPT_REASONING,
                       "providers": meta["providers"],
                       "generation_ids": meta["generation_ids"],
                       "usage": meta["usage"],
                       "cost_usd_reported": meta["cost_usd_reported"],
                       "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
                break
        if rec is None:
            return False
    with _state_lock:
        state["units"][u["unit"]] = rec
        save_state(arm, state)
    dist = Counter(g["importance"] for g in rec["grades"])
    cost = f" ${rec.get('cost_usd_reported', 0):.3f}" if arm == "gpt55" else ""
    print(f"  {u['unit']:44} {arm} ok: {n} traps "
          f"(c{dist.get('critical',0)}/s{dist.get('supporting',0)}/p{dist.get('peripheral',0)})"
          f"{cost}", flush=True)
    return True


def run_arm(arm):
    units = load_units()
    state = load_state(arm)
    assert state.get("rubric_sha256") == rubric_sha(), \
        "rubric changed since state was started - clear state or reconcile first"
    done = state["units"]
    todo = [u for u in units if u["unit"] not in done]
    if LIMIT:
        todo = todo[:LIMIT]
    print(f"arm {arm}: {len(done)}/{len(units)} units done; grading {len(todo)} now "
          f"({WORKERS} workers)", flush=True)
    if todo:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            results = list(ex.map(lambda u: grade_unit(arm, u, state), todo))
        print(f"chunk: {sum(results)}/{len(todo)} succeeded", flush=True)
    # verify from file, not memory
    on_disk = json.load(open(STATE_FILES[arm]))["units"]
    n_traps = sum(len(v["grades"]) for v in on_disk.values())
    spend = sum(v.get("cost_usd_reported") or 0 for v in on_disk.values())
    print(f"state file: {len(on_disk)}/{len(units)} units, {n_traps} trap grades"
          + (f", OpenRouter spend ${spend:.2f}" if arm == "gpt55" else ""), flush=True)


# ---------------------------------------------------------------- provenance join
def provenance_join(authored_by, ext_by, cache):
    """(sid, authored_idx0) -> (extracted_idx0, via) for matcher-affirmed same-moment pairs,
    using the cache entry keyed to CURRENT sheet content (see module docstring)."""
    fwd = {}
    for sid, e in ext_by.items():
        afs, bfs = authored_by[sid]["fact_sheet"], e["fact_sheet"]
        na, nb = len(afs["salience_traps"]), len(bfs["salience_traps"])
        entry, via = cache.get(wpc.ckey("kept", sid, "traps", afs, bfs)), "hashed"
        if entry is None:
            u = cache.get(f"kept|{sid}|traps")
            if u and u.get("n_a") == na and u.get("n_b") == nb:
                entry, via = u, "unhashed-count-validated"
        if entry is None:
            continue
        for m in entry["pairs"]:
            i, j = m.get("a"), m.get("b")
            if (m.get("same_moment") and isinstance(i, int) and isinstance(j, int)
                    and 1 <= i <= na and 1 <= j <= nb):
                fwd[(sid, i - 1)] = (j - 1, via)
    return fwd


# ---------------------------------------------------------------- analysis
def kappa_block(rows, a_key, b_key, label, note=None):
    """Kappa + consultation-level cluster-bootstrap CI + confusion for rows carrying both
    grades. rows: dicts with 'consultation' and grade fields (may be None)."""
    use = [r for r in rows if r.get(a_key) in VALID and r.get(b_key) in VALID]
    pairs = [(r[a_key], r[b_key]) for r in use]
    by_c = defaultdict(list)
    for r in use:
        by_c[r["consultation"]].append((r[a_key], r[b_key]))
    clusters = list(by_c.values())
    k = wpc.cohens_kappa(pairs)
    boot = wpc.cluster_bootstrap(
        clusters, lambda d: wpc.cohens_kappa([g for c in d for g in c]), nboot=NBOOT)
    conf = {a: {b: 0 for b in VALID} for a in VALID}
    for a, b in pairs:
        conf[a][b] += 1
    return {
        "pairing": label, "n_pairs": len(pairs), "n_consultation_clusters": len(clusters),
        "raw_agreement": round(sum(1 for a, b in pairs if a == b) / len(pairs), 4) if pairs else None,
        "cohens_kappa": round(k, 4) if k is not None else None,
        "kappa_cluster_bootstrap_95ci": (None if not boot or boot.get("lo") is None else
                                         [round(boot["lo"], 4), round(boot["hi"], 4)]),
        "bootstrap_detail": boot,
        "confusion_rows_a_cols_b": conf,
        **({"note": note} if note else {})}


def analyse():
    units = load_units()
    arms = {}
    for arm in ("opus5", "gpt55"):
        st = load_state(arm)
        assert st.get("rubric_sha256") == rubric_sha(), f"{arm} state graded a different rubric"
        arms[arm] = st["units"]
    missing = {arm: [u["unit"] for u in units if u["unit"] not in arms[arm]]
               for arm in ("opus5", "gpt55")}
    if any(missing.values()):
        print("INCOMPLETE - missing units:", json.dumps(missing, indent=1))
        sys.exit(1)

    authored_by = {s["id"]: s for s in json.load(open(SCN_FILE))}
    ext_by = {e["id"]: e for e in json.load(open(EXT_FILE))}
    cache = json.load(open(MATCHES_FILE))
    fwd = provenance_join(authored_by, ext_by, cache)
    rev = defaultdict(list)                       # (sid, b_idx0) -> [a_idx0, ...]
    for (sid, ai), (bi, _via) in fwd.items():
        rev[(sid, bi)].append(ai)

    # ---- per-trap rows ----
    rows, ambiguous_rev = [], 0
    for u in units:
        o5 = arms["opus5"][u["unit"]]["grades"]
        g5 = arms["gpt55"][u["unit"]]["grades"]
        assert len(o5) == len(g5) == len(u["traps"])
        for i, t in enumerate(u["traps"]):
            if u["sheet"] == "authored":
                backfill = t.get("importance")
                m = fwd.get((u["sid"], i))
                native = (ext_by[u["sid"]]["fact_sheet"]["salience_traps"][m[0]]
                          .get("importance") if m else None)
                join = {"matched_extracted_index": m[0] if m else None,
                        "via": m[1] if m else None}
            else:
                native = t.get("importance")
                ais = rev.get((u["sid"], i), [])
                if len(ais) == 1:
                    backfill = (authored_by[u["sid"]]["fact_sheet"]["salience_traps"][ais[0]]
                                .get("importance"))
                elif len(ais) > 1:
                    gs = {authored_by[u["sid"]]["fact_sheet"]["salience_traps"][a]
                          .get("importance") for a in ais}
                    backfill = gs.pop() if len(gs) == 1 else None
                    if backfill is None:
                        ambiguous_rev += 1
                else:
                    backfill = None
                join = {"matched_authored_indices": ais or None}
            a, b = o5[i]["importance"], g5[i]["importance"]
            agree = a == b
            consensus = a if agree else min((a, b), key=lambda g: RANK[g])
            rows.append({
                "uid": f"{u['unit']}|{i}",
                "consultation": u["sid"], "sheet": u["sheet"], "trap_index": i,
                "trap": t["trap"], "correct_handling": t["correct_handling"],
                "mode": t.get("mode"),
                "opus5_rubric": o5[i], "gpt55_rubric": g5[i],
                "opus48_backfill": backfill, "opus5_native": native,
                "provenance_join": join,
                "consensus": {"grade": consensus, "flagged": not agree,
                              "source": "agreement" if agree
                              else "disagreement_resolved_to_lower"}})

    n_auth = sum(1 for r in rows if r["sheet"] == "authored")
    n_ext = len(rows) - n_auth
    assert n_auth == 211, f"expected 211 authored traps, got {n_auth}"
    assert n_ext == 472, f"expected 472 extracted traps, got {n_ext}"

    gpt_units = arms["gpt55"]
    spend = round(sum(v.get("cost_usd_reported") or 0 for v in gpt_units.values()), 4)
    usage = {f: sum(v["usage"].get(f, 0) for v in gpt_units.values())
             for f in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens")}

    grades_doc = {
        "spec": "rubric-anchored severity regrade of every trap in the master dataset",
        "rubric": {"file": RUBRIC_FILE, "sha256": rubric_sha(),
                   "committed": "1fb00db07"},
        "arms": {
            "opus5_rubric": {"model": OPUS_MODEL, "effort": OPUS_EFFORT,
                             "route": "plan (claude -p)", "workers": 4},
            "gpt55_rubric": {"model": "openai/gpt-5.5 (models.lock.json role 'auditor')",
                             "route": "openrouter, provider order ['openai']",
                             "temperature": GPT_TEMPERATURE,
                             "reasoning_effort": GPT_REASONING, "k": 1,
                             "seed_base": GPT_SEED0,
                             "cost_usd_reported_total": spend, "usage_total": usage}},
        "blinding": "graders saw trap + correct_handling only - no mode, no prior grades",
        "joins": {
            "opus48_backfill": "authored traps: own no-rubric backfill grade; extracted "
                               "traps: reverse provenance match where unambiguous",
            "opus5_native": "extracted traps: own extraction-call grade; authored traps: "
                            "forward provenance match where one exists",
            "provenance_matching": "master/wd_provenance_matches.json, entries keyed to current "
                             "sheet content (hashed keys; count-validated unhashed fallback "
                             "for ha_sudden_thunderclap)",
            "matched_pairs_joined": len(fwd),
            "ambiguous_reverse_matches_left_null": ambiguous_rev},
        "n_traps": {"authored": n_auth, "extracted": n_ext, "total": len(rows)},
        "traps": rows}
    json.dump(grades_doc, open(GRADES_OUT, "w"), ensure_ascii=False, indent=1)

    # ---- stability report ----
    prov = json.load(open(PROV_REPORT))
    r5 = prov["analyses"]["kept"]["WD_R5"]

    auth_rows = [r for r in rows if r["sheet"] == "authored"]
    ext_rows = [r for r in rows if r["sheet"] == "extracted"]

    def g(r, fam):
        return (r.get(fam) or {}).get("importance") if fam.endswith("_rubric") else r.get(fam)

    flat = [{**{k: r[k] for k in ("consultation", "sheet")},
             "o5r": g(r, "opus5_rubric"), "g5r": g(r, "gpt55_rubric"),
             "bf": r["opus48_backfill"], "nat": r["opus5_native"]} for r in rows]
    fa = [f for f in flat if f["sheet"] == "authored"]
    fe = [f for f in flat if f["sheet"] == "extracted"]

    pairings = [
        kappa_block(flat, "o5r", "g5r",
                    "opus5_rubric vs gpt55_rubric (all 683 traps, same-item)",
                    note="THE rubric-anchored cross-model agreement. Same trap graded by "
                         "both models - no matching noise. The design difference against "
                         "the earlier no-rubric reading (matched items across sheets) is "
                         "quantified by the design-matched rows below."),
        kappa_block(fa, "o5r", "g5r", "opus5_rubric vs gpt55_rubric (authored 211 only)"),
        kappa_block(fe, "o5r", "g5r", "opus5_rubric vs gpt55_rubric (extracted 472 only)"),
        {"pairing": "opus48_backfill vs opus5_native (earlier no-rubric reading, verbatim)",
         "n_pairs": r5["matched_trap_pairs"],
         "raw_agreement": r5["raw_agreement"],
         "cohens_kappa": r5["cohens_kappa"],
         "kappa_cluster_bootstrap_95ci": r5["kappa_cluster_bootstrap_95ci"],
         "confusion_rows_a_cols_b": r5["confusion_matrix_authored_rows_x_extracted_cols"],
         "note": "no shared rubric, different models AND independently-worded trap lists "
                 "joined by a semantic matcher (cross-model + cross-sheet). Quoted from "
                 "master/wd_provenance_report.json (kept set)."},
        kappa_block(fa, "o5r", "bf",
                    "opus5_rubric vs opus48_backfill (authored 211, same-item)",
                    note="does the rubric move the Claude family's grades off the "
                         "no-rubric backfill? Same trap, same family, rubric vs none."),
        kappa_block(fe, "o5r", "nat",
                    "opus5_rubric vs opus5_native (extracted 472, same-item)",
                    note="same model, same trap, rubric vs native extraction-call grade - "
                         "the purest read on what the rubric itself changes."),
    ]

    # design-matched successor to the earlier no-rubric reading: cross-model AND cross-sheet
    # on the joined pairs, rubric-anchored - the like-for-like comparison against the 0.177
    # design
    dm_rows = []
    ext_traps_g = {(u["sid"], i): (arms["opus5"][u["unit"]]["grades"][i]["importance"],
                                   arms["gpt55"][u["unit"]]["grades"][i]["importance"])
                   for u in units if u["sheet"] == "extracted" for i in range(len(u["traps"]))}
    auth_traps_g = {(u["sid"], i): (arms["opus5"][u["unit"]]["grades"][i]["importance"],
                                    arms["gpt55"][u["unit"]]["grades"][i]["importance"])
                    for u in units if u["sheet"] == "authored" for i in range(len(u["traps"]))}
    for (sid, ai), (bi, _via) in fwd.items():
        ao, ag = auth_traps_g[(sid, ai)]
        bo, bg = ext_traps_g[(sid, bi)]
        dm_rows.append({"consultation": sid, "a_o5": ao, "a_g5": ag,
                        "b_o5": bo, "b_g5": bg})
    pairings.append(kappa_block(
        [{"consultation": d["consultation"], "x": d["a_o5"], "y": d["b_g5"]} for d in dm_rows],
        "x", "y", "opus5_rubric(authored trap) vs gpt55_rubric(matched extracted trap)",
        note="design-matched to the earlier no-rubric reading: different models, different "
             "sheets, semantic-matcher join - but both rubric-anchored. Direct like-for-like "
             "against kappa 0.177."))
    pairings.append(kappa_block(
        [{"consultation": d["consultation"], "x": d["a_g5"], "y": d["b_o5"]} for d in dm_rows],
        "x", "y", "gpt55_rubric(authored trap) vs opus5_rubric(matched extracted trap)",
        note="same design-matched comparison, arms swapped"))
    pairings.append(kappa_block(
        [{"consultation": d["consultation"], "x": d["a_o5"], "y": d["b_o5"]} for d in dm_rows],
        "x", "y", "opus5_rubric(authored trap) vs opus5_rubric(matched extracted trap)",
        note="SAME model both sides across the matched sheets: isolates item-wording + "
             "matching noise from model disagreement"))

    dists = {
        "opus5_rubric": dict(Counter(f["o5r"] for f in flat)),
        "gpt55_rubric": dict(Counter(f["g5r"] for f in flat)),
        "opus48_backfill (authored 211)": dict(Counter(f["bf"] for f in fa)),
        "opus5_native (extracted 472)": dict(Counter(f["nat"] for f in fe)),
        "consensus": dict(Counter(r["consensus"]["grade"] for r in rows)),
        "by_sheet": {
            "authored": {"opus5_rubric": dict(Counter(f["o5r"] for f in fa)),
                         "gpt55_rubric": dict(Counter(f["g5r"] for f in fa))},
            "extracted": {"opus5_rubric": dict(Counter(f["o5r"] for f in fe)),
                          "gpt55_rubric": dict(Counter(f["g5r"] for f in fe))}}}

    flagged = [r for r in rows if r["consensus"]["flagged"]]
    flagged_list = [{"uid": r["uid"], "sheet": r["sheet"],
                     "opus5": r["opus5_rubric"]["importance"],
                     "gpt55": r["gpt55_rubric"]["importance"],
                     "consensus": r["consensus"]["grade"], "trap": r["trap"][:120]}
                    for r in flagged]

    main_k = pairings[0]["cohens_kappa"]
    main_ci = pairings[0]["kappa_cluster_bootstrap_95ci"]
    old_k = r5["cohens_kappa"]
    old_ci = r5["kappa_cluster_bootstrap_95ci"]
    delta = round(main_k - old_k, 4) if main_k is not None else None
    separated = (main_ci and old_ci and main_ci[0] > old_ci[1])
    if main_k is None:
        headline = "kappa could not be computed - inspect the grades file."
    elif delta > 0:
        headline = (
            f"Under the written rubric, cross-model agreement on the severity axis is kappa "
            f"{main_k:.3f} (95% CI {main_ci[0]:.3f}-{main_ci[1]:.3f}, opus-5 vs gpt-5.5, "
            f"{pairings[0]['n_pairs']} same-item trap pairs) vs 0.177 (95% CI "
            f"{old_ci[0]:.3f}-{old_ci[1]:.3f}) without one - a gain of {delta:+.3f}"
            + (", with non-overlapping CIs" if separated else
               ", though the CIs overlap") +
            "; note the rubric comparison is same-item while the earlier no-rubric reading "
            "crossed matched items - the design-matched rows isolate that difference.")
    else:
        headline = (
            f"The written rubric did NOT raise cross-model agreement: kappa {main_k:.3f} "
            f"(95% CI {main_ci[0]:.3f}-{main_ci[1]:.3f}) vs 0.177 without it "
            f"({delta:+.3f}); severity-axis instability persists under a shared written "
            f"standard and is reported as a finding.")

    report = {
        "spec": "rubric-anchored severity regrade of every trap in the master dataset",
        "rubric_sha256": rubric_sha(), "seed_bootstrap": wpc.SEED, "nboot": NBOOT,
        "kappa_selfcheck": wpc._kappa_selfcheck(),
        "headline": headline,
        "kappa_table": pairings,
        "grade_distributions": dists,
        "consensus": {"n_traps": len(rows), "n_agreement": len(rows) - len(flagged),
                      "n_flagged_disagreements": len(flagged),
                      "flag_rate": round(len(flagged) / len(rows), 4),
                      "disagreement_pairs": dict(Counter(
                          f"{r['opus5_rubric']['importance']}|{r['gpt55_rubric']['importance']}"
                          for r in flagged)),
                      "flagged": flagged_list},
        "openrouter_spend": {"cost_usd_reported": spend, "usage": usage,
                             "n_calls": len(gpt_units)},
        "files": {"grades": os.path.relpath(GRADES_OUT, HERE),
                  "sitting": os.path.relpath(SITTING_OUT, HERE)}}
    json.dump(report, open(REPORT_OUT, "w"), ensure_ascii=False, indent=1)

    build_sitting(rows)

    print(f"\nheadline: {headline}\n")
    print(f"{'pairing':74} {'n':>5} {'raw':>6} {'kappa':>7}  95% CI")
    for p in pairings:
        ci = p.get("kappa_cluster_bootstrap_95ci")
        print(f"{p['pairing'][:74]:74} {p['n_pairs']:>5} "
              f"{p['raw_agreement'] if p['raw_agreement'] is not None else '-':>6} "
              f"{p['cohens_kappa'] if p['cohens_kappa'] is not None else '-':>7}  "
              f"{ci if ci else '-'}")
    print(f"\nconsensus: {len(flagged)}/{len(rows)} flagged "
          f"({100 * len(flagged) / len(rows):.1f}%)")
    print(f"OpenRouter spend: ${spend:.2f} over {len(gpt_units)} calls")
    print(f"-> {os.path.relpath(GRADES_OUT, HERE)}\n-> {os.path.relpath(REPORT_OUT, HERE)}"
          f"\n-> {os.path.relpath(SITTING_OUT, HERE)}")


# ---------------------------------------------------------------- sitting pack
_STOP = set("a an and are as at be been but by did do does for from had has have he her his i "
            "in is it its me my no not of on or our she so that the their them they this to "
            "was we were with you your".split())


def _words(s):
    return {w for w in re.findall(r"[a-z']+", s.lower()) if w not in _STOP and len(w) > 2}


def excerpt(transcript, trap, correct_handling, width=1600):
    """Deterministic excerpt: the ~width chars of contiguous transcript turns that best
    overlap the trap wording, [...]-marked when truncated."""
    lines = transcript.splitlines()
    target = _words(trap + " " + correct_handling)
    scores = [len(target & _words(l)) for l in lines]
    best = max(range(len(lines)), key=lambda i: (scores[i], -i))
    lo = hi = best
    size = len(lines[best])
    while size < width:
        up = scores[lo - 1] if lo > 0 else -1
        dn = scores[hi + 1] if hi < len(lines) - 1 else -1
        if up < 0 and dn < 0:
            break
        if up >= dn:
            lo -= 1
            size += len(lines[lo])
        else:
            hi += 1
            size += len(lines[hi])
    text = "\n".join(lines[lo:hi + 1]).strip()
    return (("[...]\n" if lo > 0 else "") + text + ("\n[...]" if hi < len(lines) - 1 else ""))


def _stratified(items, keyf, n, rng):
    """Round-robin draw across strata (sorted keys, shuffled within) until n taken."""
    strata = defaultdict(list)
    for it in items:
        strata[keyf(it)].append(it)
    for v in strata.values():
        rng.shuffle(v)
    order = sorted(strata)
    out, i = [], 0
    while len(out) < n and any(strata[k] for k in order):
        k = order[i % len(order)]
        i += 1
        if strata[k]:
            out.append(strata[k].pop())
    return out


def build_sitting(rows, n_total=20, max_flagged=14):
    rng = random.Random(SITTING_SEED)
    tmap = {}
    for u in load_units():
        for i, t in enumerate(u["traps"]):
            tmap[f"{u['unit']}|{i}"] = u["transcript"]
    flagged = [r for r in rows if r["consensus"]["flagged"]]
    clean = [r for r in rows if not r["consensus"]["flagged"]]
    n_flag = min(len(flagged), max_flagged)
    picked = _stratified(
        flagged, lambda r: (r["sheet"], "|".join(sorted([r["opus5_rubric"]["importance"],
                                                         r["gpt55_rubric"]["importance"]]))),
        n_flag, rng)
    picked += _stratified(clean, lambda r: (r["sheet"], r["consensus"]["grade"]),
                          n_total - len(picked), rng)
    rng.shuffle(picked)   # flagged and clean interleaved so position carries no signal

    items = []
    for k, r in enumerate(picked, 1):
        items.append({
            "item_id": f"sev_{k:02d}",
            "consultation": r["consultation"], "sheet": r["sheet"], "uid": r["uid"],
            "trap": r["trap"], "correct_handling": r["correct_handling"],
            "transcript_excerpt": excerpt(tmap[r["uid"]], r["trap"], r["correct_handling"]),
            "blinded": {
                "mode": r["mode"],
                "opus5_rubric": r["opus5_rubric"], "gpt55_rubric": r["gpt55_rubric"],
                "opus48_backfill": r["opus48_backfill"], "opus5_native": r["opus5_native"],
                "consensus": r["consensus"]}})
    doc = {
        "purpose": "severity items for the single author-clinician sitting that anchors "
                   "the rubric grades to a human. Grade each item with the rubric "
                   "below, seeing ONLY trap + correct_handling + transcript excerpt - the "
                   "'blinded' key holds every model opinion and must not be shown during "
                   "grading.",
        "instructions_for_grader": "For each item: read the excerpt, then grade the "
                                   "importance of a note getting this right vs wrong - "
                                   "critical / supporting / peripheral - by the rubric's "
                                   "decision procedure and tie-break.",
        "seed": SITTING_SEED,
        "composition": {"n_items": len(items),
                        "n_from_flagged_disagreements": sum(
                            1 for it in items if it["blinded"]["consensus"]["flagged"]),
                        "n_clean_agreement": sum(
                            1 for it in items if not it["blinded"]["consensus"]["flagged"]),
                        "sampling": "seeded round-robin stratified draw: disagreements by "
                                    "(sheet, grade-pair), capped at 14; topped up with "
                                    "agreements by (sheet, grade) to 20; order shuffled"},
        "rubric_sha256": rubric_sha(),
        "rubric": read_rubric(),
        "items": items}
    json.dump(doc, open(SITTING_OUT, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    if ANALYSE:
        analyse()
    elif ARM in ("opus5", "gpt55"):
        run_arm(ARM)
    else:
        print(__doc__)
        sys.exit(2)
