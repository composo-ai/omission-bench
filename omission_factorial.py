"""omission_factorial.py - rubric severity per fact, then the severity x residual allocation.

Stage 2 of the from-scratch omission design (docs/FROM-SCRATCH.md, the lead author's GO 2026-08-12).
The paper's central figure is a SURFACE: detection across how much of the fact survives
(the residual axis) crossed with how much its loss matters (the severity axis). The frozen
corpus can draw neither axis - 255 critical / 26 supporting / 0 peripheral, every omission
complete. This file decides what to build so both axes have range, and says honestly what
the corpus can and cannot support BEFORE anything is built.

  1. eligible facts   from master/fact_sites.json: every site located exactly once, sites
                      non-overlapping, a primary nominated.
  2. severity         graded per FACT, never per pair - the matched complete/partial pairs
                      of one fact share it, because the fact is the same and only the
                      residual differs. EVERY fact gets the same instrument, trap or
                      must_contain: two cross-family arms on the VERBATIM
                      hard_negatives_master.RUBRIC_GRADE prompt and the verbatim rubric -
                      opus-5 (constructor, effort medium) and gpt-5.5 (auditor, effort high).
                      Agreement stands; disagreement takes the LOWER grade (the rubric's own
                      conservative tie-break) and flags the fact. A trap's inherited sheet
                      `importance` is recorded beside the grade for comparison, never used as
                      the value - the shakedown found the two sources differ by a level, and
                      a step in the middle of the severity axis is exactly what this paper
                      cannot afford.
  3. residual levels  what each fact can support, from its own site map:
                        complete        remove every site.                     (always)
                        partial-strong  remove every site but one EXPLICIT or PARAPHRASE
                                        residual - the fact is still fully recoverable.
                        partial-weak    remove every site but one PARTIAL residual - only a
                                        fragment survives, missing the detail that drives
                                        action.
                      The primary site is removed in every class, so `primary_site_removed`
                      means the same thing everywhere. Holding the surviving count at ONE is
                      deliberate: it makes residual STRENGTH the manipulated variable instead
                      of confounding it with how many mentions happen to remain.
  4. allocation       greedy fill of the 3x3 grid to TARGET_PER_CELL, balanced across
                      consultations, preferring facts that can carry a MATCHED set (the same
                      fact at two or three residual levels - an exact within-fact contrast).

Outputs:
  master/factorial_severity.json    grade state + report (idempotent per fact+arm)
  master/factorial_allocation.json  the build plan: one row per pair to construct
  docs/FACTORIAL-ALLOCATION.md      the honest allocation report, cell by cell

Usage:
  .venv/bin/python omission_factorial.py --grade --budget 60 -y      # sever  ity, resumable
  .venv/bin/python omission_factorial.py --allocate                  # free
  .venv/bin/python omission_factorial.py --report                    # free
Exit: 0 complete | 2 work left (rerun) | 3 budget stop.
"""
import argparse, json, os, random, sys, threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from common import HERE, Run, resolve_model
import hard_negatives_master as HN
import inject_omissions_v2 as INJ
import omission_sites as OS

SPEC = OS.SPEC
EXPERIMENT = "wd-omission-factorial"
MASTER = OS.MASTER
SEV_FILE = os.path.join(MASTER, "factorial_severity.json")
ALLOC_FILE = os.path.join(MASTER, "factorial_allocation.json")
ALLOC_DEV_FILE = os.path.join(MASTER, "factorial_allocation_dev.json")
REPORT_MD = os.path.join(HERE, "docs", "FACTORIAL-ALLOCATION.md")
REPORT_MD_DEV = os.path.join(HERE, "docs", "FACTORIAL-ALLOCATION-DEV.md")
SEED_SET_FILE = os.path.join(MASTER, "partial_omission_seed_set.json")
FROZEN_FILE = os.path.join(MASTER, "pairs_master_frozen.json")

DATASET_VERSION = "factorial-v1"
ARMS = {"opus5": ("constructor", "medium"), "gpt55": ("auditor", "high")}
GRADE_SEED = 20260809             # the severity-regrade seed, as grade_partial_seed_severity
VALID = ("critical", "supporting", "peripheral")
RANK = {"critical": 2, "supporting": 1, "peripheral": 0}
STRONG = ("explicit", "paraphrase")

LEVELS = ("complete", "partial-strong", "partial-weak")
SEVERITIES = ("critical", "supporting", "peripheral")
TARGET_PER_CELL = 25
MAX_FACTS_PER_CONSULT = 4         # keep one long note from dominating a cell
MAX_PAIRS_PER_CONSULT = 6

_lock = threading.RLock()


# ---------------------------------------------------------------- substrate
def residual_options(fact):
    """{level: surviving_site_id or None} for one mapped fact.

    The survivor is always a NON-primary site (the primary is removed in every class), and
    where several qualify the strongest-then-earliest one wins, so the choice is
    deterministic and the strong level is genuinely strong.
    """
    sites = fact["sites"]
    pid = fact.get("primary_site_id")
    opts = {"complete": None}
    if pid is None:
        return {}
    others = [s for s in sites if s["site_id"] != pid]
    strong = [s for s in others if s["strength"] in STRONG]
    weak = [s for s in others if s["strength"] == "partial"]
    if strong:
        strong.sort(key=lambda s: (STRONG.index(s["strength"]), s["char_span"][0]))
        opts["partial-strong"] = strong[0]["site_id"]
    if weak:
        weak.sort(key=lambda s: s["char_span"][0])
        opts["partial-weak"] = weak[0]["site_id"]
    return opts


def eligible(cons_rows):
    """[(cons, fact, options)] over every construction-eligible fact in the map."""
    out = []
    for cons in cons_rows:
        for f in INJ.eligible_facts(cons):
            opts = residual_options(f)
            if not opts:
                continue
            out.append((cons, f, opts))
    return out


def fact_uid(cons, fact):
    return f"{cons['key']}|{fact['fact_key']}"


# ---------------------------------------------------------------- severity
def load_sev():
    if os.path.exists(SEV_FILE):
        return json.load(open(SEV_FILE))
    return {"created": OS.now_utc(), "spend": {"cost_usd": 0.0, "calls": 0, "by_stage": {}},
            "facts": {}}


def save_sev(sev):
    with _lock:
        tmp = SEV_FILE + ".tmp"
        json.dump(sev, open(tmp, "w"), ensure_ascii=False, indent=1)
        os.replace(tmp, SEV_FILE)


def grade_arm(cons, fact, arm, budget):
    """One arm's rubric grade for one fact - the defect being graded is the WHOLE fact
    leaving the note, which is what the complete class does and what the partial class is a
    weakened version of. Residual survival is deliberately not mentioned: folding it in
    would beg the experiment's question."""
    role, effort = ARMS[arm]
    prompt = HN.RUBRIC_GRADE.format(
        rubric=HN.RUBRIC, t=(cons.get("transcript") or "")[:40000], typ="omit",
        change=f"Removed the fact from the note entirely: {fact['fact']}",
        what=fact["fact"])
    obj, meta = OS.call_json(budget, f"rubric_grade_{arm}", prompt, role, effort,
                             seed=GRADE_SEED)
    if not isinstance(obj, dict) or obj.get("importance") not in VALID:
        return {"arm": arm, "error": (meta or {}).get("error") or "bad grade",
                "raw": str(obj)[:200]}
    return {"arm": arm, "role": role, "reasoning_effort": effort, "seed": GRADE_SEED,
            "importance": obj["importance"], "test_fired": obj.get("test_fired"),
            "rationale": obj.get("rationale"),
            "generation_ids": (meta or {}).get("generation_ids"),
            "cost_usd": (meta or {}).get("cost"), "utc": OS.now_utc()}


def consensus(arms):
    """Verbatim from the part-R regrade: agreement stands; disagreement takes the LOWER
    grade and flags the item for a human look."""
    g = {a: r["importance"] for a, r in arms.items() if r.get("importance")}
    if len(g) < 2:
        only = next(iter(g.values()), None)
        return {"grade": only, "flagged": None if only else True, "n_arms": len(g),
                "note": "single arm returned - not a two-model consensus" if only
                        else "no arm returned a valid grade"}
    a, b = g["opus5"], g["gpt55"]
    agree = a == b
    return {"grade": a if agree else min((a, b), key=lambda x: RANK[x]),
            "flagged": not agree, "n_arms": 2,
            "note": "both arms agree" if agree else
                    f"arms disagree (opus5 {a} / gpt55 {b}) - lower grade taken"}


def grade_order(cands, sev, include_single=False, eval_only=True, dev_only=False,
                seed=OS.SEED):
    """Which facts to spend grading calls on, in priority order.

    EVERY fact is graded on the rubric, traps included. The first shakedown of this pass
    found the inherited trap `importance` grades sitting a level above a fresh rubric grade of
    "this fact left the note entirely" - graded, they came back supporting and peripheral
    where the sheet said critical and supporting. Two severity sources that disagree
    systematically would put a step in the middle of the axis the paper is about, so the axis
    gets ONE instrument and the inherited grade is kept beside it as a comparison rather than
    used as a value. At ~$0.02 a call this costs tens of dollars, not hundreds.

    Multi-site facts first (only they can carry a partial pair, which is where the scarce
    cells are), low-severity-proxy buckets ahead of the abundant load-bearing-core bucket,
    then round-robin across consultations so a budget stop leaves a spread rather than the
    first N notes.
    """
    rng = random.Random(f"{seed}|factorial-grade")
    prio = {"mc_contextual": 0, "trap_peripheral": 0, "mc_core_medium": 1,
            "trap_supporting": 1, "mc_core_high": 2, "trap_critical": 2}
    rows = []
    for cons, f, opts in cands:
        if len(opts) < 2 and not include_single:
            continue
        if dev_only and cons["split"] != "gepa_dev":
            continue
        if eval_only and not dev_only and cons["split"] != "eval":
            continue          # the allocation is eval-split only; grading the dev pool would
                              # spend on facts no cell can ever use
        rows.append((0 if len(opts) > 1 else 1, prio.get(f.get("bucket"), 3),
                     rng.random(), cons, f))
    rows.sort(key=lambda r: r[:3])
    by_consult = defaultdict(list)
    for r in rows:
        by_consult[r[3]["key"]].append(r)
    out, i = [], 0
    while any(len(v) > i for v in by_consult.values()):
        for k in sorted(by_consult):
            if len(by_consult[k]) > i:
                out.append((by_consult[k][i][3], by_consult[k][i][4]))
        i += 1
    return out


def run_grades(todo, sev, budget, args):
    print(f"severity: {len(todo)} facts with grading work | arms "
          + ", ".join(f"{a}={resolve_model(r)['resolved']} effort {e}"
                      for a, (r, e) in ARMS.items())
          + f" | budget ${budget.spent:.2f}/${budget.cap:.0f}", flush=True)

    def one(job):
        cons, f = job
        uid = fact_uid(cons, f)
        with _lock:
            rec = sev["facts"].setdefault(uid, {
                "fact_uid": uid, "key": cons["key"], "stratum": cons["stratum"],
                "id": cons["id"], "split": cons["split"], "fact_key": f["fact_key"],
                "kind": f["kind"], "bucket": f.get("bucket"), "scope": f.get("scope"),
                "load_bearing": f.get("load_bearing"),
                "inherited_importance": f.get("importance"), "fact": f["fact"], "arms": {}})
        for arm in ARMS:
            if rec["arms"].get(arm, {}).get("importance"):
                continue
            if not budget.ok():
                return
            got = grade_arm(cons, f, arm, budget)
            with _lock:
                rec["arms"][arm] = got
                save_sev(sev)
        with _lock:
            rec["consensus"] = consensus(rec["arms"])
            save_sev(sev)
        c = rec["consensus"]
        print(f"  {uid:44} {(rec.get('bucket') or '-'):15} "
              + "/".join(rec["arms"].get(a, {}).get("importance", "ERR") for a in ARMS)
              + f" -> {c['grade']}{'  FLAGGED' if c['flagged'] else ''} ${budget.spent:.2f}",
              flush=True)

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(one, todo))
    save_sev(sev)


def severity_of(cons, fact, sev):
    """(grade, source, extra) for one fact. The fresh 2-arm rubric consensus is the value of
    record for every fact kind; a trap's inherited sheet grade rides along for comparison and
    is used only if the grading pass never reached the fact."""
    uid = fact_uid(cons, fact)
    rec = sev["facts"].get(uid) or {}
    con = rec.get("consensus") or {}
    if con.get("grade"):
        arms = rec.get("arms") or {}
        out = {"severity": con["grade"], "severity_source": "rubric_2arm_consensus",
               "severity_flagged": bool(con.get("flagged")),
               "severity_arms": {a: arms.get(a, {}).get("importance") for a in ARMS},
               "severity_rationale": (arms.get("opus5") or {}).get("rationale"),
               "severity_test_fired": (arms.get("opus5") or {}).get("test_fired")}
        if fact.get("importance"):
            out["inherited_importance"] = fact["importance"]
            out["inherited_agrees"] = fact["importance"] == con["grade"]
        return out
    if fact["kind"] == "trap" and fact.get("importance"):
        return {"severity": fact["importance"],
                "severity_source": "trap_importance_fallback_ungraded",
                "severity_flagged": False, "inherited_importance": fact["importance"]}
    return {"severity": None, "severity_source": "ungraded", "severity_flagged": None}


# ---------------------------------------------------------------- existing corpus
def carried_forward():
    """What the corpus already holds that belongs in a factorial cell:
    the frozen complete-omission pairs, and the 34 relabelled partials with their verified
    residual strength deciding which partial level they sit in."""
    out = {"complete": Counter(), "partial-strong": Counter(), "partial-weak": Counter(),
           "detail": []}
    fz = json.load(open(FROZEN_FILE))
    for p in fz["pairs"]:
        if p["type"] != "omit" or not p.get("severity"):
            continue
        out["complete"][p["severity"]] += 1
        out["detail"].append({"pair_id": p["pair_id"], "level": "complete",
                              "severity": p["severity"], "source": "frozen_281"})
    if os.path.exists(SEED_SET_FILE):
        for p in json.load(open(SEED_SET_FILE))["pairs"]:
            res = (p.get("residual") or {})
            strongest = res.get("strongest") or res.get("max_strength")
            lvl = "partial-strong" if strongest in STRONG else (
                "partial-weak" if strongest == "partial" else None)
            if not lvl or not p.get("severity"):
                continue
            out[lvl][p["severity"]] += 1
            out["detail"].append({"pair_id": p["pair_id"], "level": lvl,
                                  "severity": p["severity"], "split": p.get("split"),
                                  "source": "partial_seed_set"})
    return out


# ---------------------------------------------------------------- allocation
_BUILT_STATE = "omission_v2_state.json"


def already_built():
    """(fact_uid, residual_level) pairs that have already been constructed. A re-allocation
    at a higher target must keep them - otherwise the plan drifts away from the artefact and
    the build spends again on facts it has already used."""
    f = os.path.join(MASTER, _BUILT_STATE)
    if not os.path.exists(f):
        return set()
    return {(p.get("fact_uid"), p.get("residual_level"))
            for p in json.load(open(f)).get("pairs", {}).values()}


def allocate(cands, sev, target=TARGET_PER_CELL, seed=OS.SEED, eval_only=True,
             dev_only=False):
    """Greedy fill of the 3x3 grid, balanced across consultations, matched sets preferred."""
    rng = random.Random(f"{seed}|factorial-alloc")
    built = already_built()
    rows = []
    for cons, f, opts in cands:
        want = "gepa_dev" if dev_only else "eval"
        if (dev_only or eval_only) and cons["split"] != want:
            continue
        s = severity_of(cons, f, sev)
        if not s["severity"]:
            continue
        rows.append({"cons": cons, "fact": f, "opts": opts, "sev": s,
                     "uid": fact_uid(cons, f), "jitter": rng.random()})
    cells = {(l, g): [] for l in LEVELS for g in SEVERITIES}
    used = defaultdict(set)              # uid -> levels already taken
    facts_per_consult, pairs_per_consult = Counter(), Counter()

    def candidates_for(level, grade, cap_facts=MAX_FACTS_PER_CONSULT,
                       cap_pairs=MAX_PAIRS_PER_CONSULT):
        out = []
        for r in rows:
            if r["sev"]["severity"] != grade or level not in r["opts"]:
                continue
            if level in used[r["uid"]]:
                continue
            k = r["cons"]["key"]
            if pairs_per_consult[k] >= cap_pairs:
                continue
            if (not used[r["uid"]]) and facts_per_consult[k] >= cap_facts:
                continue
            out.append(r)
        # already-built pairs first (so a re-allocation is a superset, never a reshuffle),
        # then matched (this fact already carries another level), then the consultation that
        # has given the least so far, then the seeded jitter
        out.sort(key=lambda r: (0 if (r["uid"], level) in built else 1,
                                0 if used[r["uid"]] else 1,
                                pairs_per_consult[r["cons"]["key"]], r["jitter"]))
        return out

    # pass 1 fills under the balance caps; pass 2 revisits only the cells that came up short
    # and lets one consultation contribute more, because an under-filled cell costs the design
    # more than a slightly lumpier one. Rows placed in pass 2 are tagged `cap_relaxed`.
    caps = [(MAX_FACTS_PER_CONSULT, MAX_PAIRS_PER_CONSULT, False),
            (MAX_FACTS_PER_CONSULT + 3, MAX_PAIRS_PER_CONSULT + 4, True)]
    for cap_facts, cap_pairs, relaxed in caps:
      exhausted = set()
      while True:
        need = [(target - len(v), c) for c, v in cells.items()
                if len(v) < target and c not in exhausted]
        if not need:
            break
        need.sort(reverse=True)
        cell = need[0][1]
        cand = candidates_for(*cell, cap_facts=cap_facts, cap_pairs=cap_pairs)
        if not cand:
            exhausted.add(cell)
            continue
        r = cand[0]
        level, grade = cell
        k = r["cons"]["key"]
        if not used[r["uid"]]:
            facts_per_consult[k] += 1
        used[r["uid"]].add(level)
        pairs_per_consult[k] += 1
        cells[cell].append({
            "cap_relaxed": relaxed,
            "pair_id": f"{k}|{INJ.LEVEL_TAG[level]}|{INJ.slug(r['fact']['fact_key'])}",
            "key": k, "stratum": r["cons"]["stratum"], "id": r["cons"]["id"],
            "split": r["cons"]["split"], "fact_uid": r["uid"],
            "fact_key": r["fact"]["fact_key"], "fact_kind": r["fact"]["kind"],
            "bucket": r["fact"].get("bucket"), "residual_level": level,
            "keep_site_id": r["opts"][level],
            "keep_site_strength": next((s["strength"] for s in r["fact"]["sites"]
                                        if s["site_id"] == r["opts"][level]), None),
            "n_sites": r["fact"]["n_sites"],
            "primary_site_id": r["fact"]["primary_site_id"],
            "already_built": (r["uid"], level) in built,
            "cell": f"{level}|{grade}", **r["sev"]})
    # matched-set bookkeeping
    plan = [p for v in cells.values() for p in v]
    per_fact = defaultdict(list)
    for p in plan:
        per_fact[p["fact_uid"]].append(p["residual_level"])
    for p in plan:
        p["matched_levels"] = sorted(per_fact[p["fact_uid"]])
    plan.sort(key=lambda p: p["pair_id"])
    return plan, cells, rows


def cell_supply(rows):
    """How many DISTINCT facts could serve each cell if nothing else competed - the honest
    ceiling, before the per-consultation caps and the one-fact-one-level-per-cell rule."""
    sup = Counter()
    for r in rows:
        for lvl in r["opts"]:
            sup[(lvl, r["sev"]["severity"])] += 1
    return sup


# ---------------------------------------------------------------- reporting
def write_allocation(doc, plan, cells, rows, sev, cf, args):
    sup = cell_supply(rows)
    out = {
        "generated_utc": OS.now_utc(), "spec": SPEC, "dataset_version": DATASET_VERSION,
        "what": "the severity x residual factorial build plan: one row per omission pair to "
                "construct, with the fact, the sites to remove and the single site to leave "
                "surviving. Severity is graded per FACT and shared by that fact's matched "
                "pairs; the residual level is the manipulated variable.",
        "site_map": {"file": "master/fact_sites.json",
                     "generated_utc": doc.get("generated_utc"),
                     "instrument_version": doc.get("instrument_version"),
                     "candidate_set_version": doc.get("candidate_set_version")},
        "design": {
            "levels": list(LEVELS), "severities": list(SEVERITIES),
            "target_per_cell": args.target,
            "residual_rule": "every class removes the primary site; partial classes leave "
                             "EXACTLY ONE surviving site, chosen for its strength, so "
                             "residual strength is manipulated and residual count is held "
                             "at 1",
            "severity_rule": "graded per fact, never per pair; traps inherit their consensus "
                             "importance, must_contain facts get two cross-family rubric "
                             "arms with the conservative tie-break",
            "caps": {"max_target_facts_per_consultation": MAX_FACTS_PER_CONSULT,
                     "max_pairs_per_consultation": MAX_PAIRS_PER_CONSULT},
            "scope": ("gepa_dev split only" if args.dev_only
                      else "eval split only" if args.eval_only else "eval + gepa_dev")},
        "counts": {
            "planned_pairs": len(plan),
            "by_cell": {f"{l}|{g}": len(cells[(l, g)]) for l in LEVELS for g in SEVERITIES},
            "supply_by_cell": {f"{l}|{g}": sup[(l, g)] for l in LEVELS for g in SEVERITIES},
            "shortfall_by_cell": {f"{l}|{g}": max(0, args.target - len(cells[(l, g)]))
                                  for l in LEVELS for g in SEVERITIES},
            "by_stratum": dict(Counter(p["stratum"] for p in plan)),
            "by_severity_source": dict(Counter(p["severity_source"] for p in plan)),
            "severity_flagged": sum(1 for p in plan if p.get("severity_flagged")),
            "cap_relaxed": sum(1 for p in plan if p.get("cap_relaxed")),
            "consultations": len({p["key"] for p in plan}),
            "target_facts": len({p["fact_uid"] for p in plan}),
            "matched_sets": dict(Counter(
                len(v) for v in
                {p["fact_uid"]: p["matched_levels"] for p in plan}.values())),
            "carried_forward": {l: dict(cf[l]) for l in LEVELS},
        },
        "severity_grading": severity_summary(sev),
        "plan": plan,
    }
    path = ALLOC_DEV_FILE if args.dev_only else ALLOC_FILE
    tmp = path + ".tmp"
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return out


def severity_summary(sev):
    facts = list(sev.get("facts", {}).values())
    graded = [f for f in facts if (f.get("consensus") or {}).get("grade")]
    audit = [f for f in graded if f.get("inherited_importance")]
    return {
        "facts_graded": len(graded), "facts_attempted": len(facts),
        "by_grade": dict(Counter(f["consensus"]["grade"] for f in graded)),
        "by_bucket_grade": {b: dict(Counter(f["consensus"]["grade"] for f in graded
                                            if f.get("bucket") == b))
                            for b in sorted({f.get("bucket") for f in graded})},
        "arms_disagreed": sum(1 for f in graded if f["consensus"].get("flagged")),
        "arm_disagreement_rate": round(
            sum(1 for f in graded if f["consensus"].get("flagged")) / max(len(graded), 1), 4),
        "trap_audit": {
            "n": len(audit),
            "agrees_with_inherited": sum(1 for f in audit
                                         if f["consensus"]["grade"] == f["inherited_importance"]),
            "confusion": dict(Counter(f"{f['inherited_importance']}->{f['consensus']['grade']}"
                                      for f in audit))},
        "spend_usd": sev["spend"]["cost_usd"], "calls": sev["spend"]["calls"],
    }


def write_md(out, cf, dev=False):
    c, s = out["counts"], out["severity_grading"]
    L = []
    L.append("# The severity x residual allocation - what the corpus can actually support")
    L.append("")
    L.append(f"Generated {out['generated_utc']} | dataset_version `{out['dataset_version']}` "
             f"| site map instrument `{out['site_map']['instrument_version']}`, candidate set "
             f"`{out['site_map']['candidate_set_version']}`.")
    L.append("")
    L.append("Written BEFORE the pairs are built, so the design is judged on what the notes "
             "hold rather than on what the build managed to find.")
    L.append("")
    L.append("## Severity grading")
    L.append("")
    L.append(f"{s['facts_graded']} facts carry a grade. Distribution: "
             + ", ".join(f"{k} {v}" for k, v in sorted(s['by_grade'].items()))
             + f". The two rubric arms disagreed on {s['arms_disagreed']} "
             f"({100 * s['arm_disagreement_rate']:.1f}%); every disagreement took the lower "
             "grade and is flagged in the artefact.")
    L.append("")
    L.append("| candidate bucket | critical | supporting | peripheral |")
    L.append("|---|---|---|---|")
    for b, d in sorted(s["by_bucket_grade"].items()):
        L.append(f"| {b} | {d.get('critical', 0)} | {d.get('supporting', 0)} | "
                 f"{d.get('peripheral', 0)} |")
    L.append("")
    ta = s["trap_audit"]
    if ta["n"]:
        L.append(f"**Trap-severity audit.** {ta['n']} trap facts whose importance the plan "
                 f"inherits were also graded fresh on the rubric by both arms: "
                 f"{ta['agrees_with_inherited']}/{ta['n']} agree. Transitions: {ta['confusion']}.")
        L.append("")
    L.append("## The grid")
    L.append("")
    L.append(f"Target {out['design']['target_per_cell']} per cell, {out['design']['scope']}. "
             "*supply* = distinct facts that could serve the cell; *planned* = what the "
             "allocator could actually place under the per-consultation caps; *carried* = "
             "pairs the corpus already has that belong in the cell.")
    L.append("")
    L.append("| cell | supply | planned | shortfall | carried forward |")
    L.append("|---|---|---|---|---|")
    for l in LEVELS:
        for g in SEVERITIES:
            k = f"{l}|{g}"
            L.append(f"| {l} x {g} | {c['supply_by_cell'][k]} | {c['by_cell'][k]} | "
                     f"{c['shortfall_by_cell'][k]} | {cf[l].get(g, 0)} |")
    L.append("")
    L.append(f"Total planned: **{c['planned_pairs']} pairs** over {c['consultations']} "
             f"consultations and {c['target_facts']} target facts "
             f"({c['by_stratum']}). Severity sources: {c['by_severity_source']}; "
             f"{c['severity_flagged']} pairs carry a flagged (arms-disagreed) grade.")
    L.append("")
    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    open(REPORT_MD_DEV if dev else REPORT_MD, "w").write("\n".join(L) + "\n")


def report(out, cf):
    c = out["counts"]
    print(f"\nmaster/factorial_allocation.json - {c['planned_pairs']} pairs planned over "
          f"{c['consultations']} consultations")
    print(f"  severity: {out['severity_grading']['by_grade']} | "
          f"{out['severity_grading']['arms_disagreed']} arm disagreements")
    print("  cell            supply  planned  short  carried")
    for l in LEVELS:
        for g in SEVERITIES:
            k = f"{l}|{g}"
            print(f"   {l:15}{g:12} {c['supply_by_cell'][k]:5} {c['by_cell'][k]:7}"
                  f" {c['shortfall_by_cell'][k]:6} {cf[l].get(g, 0):7}")
    print(f"  strata {c['by_stratum']} | severity sources {c['by_severity_source']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--grade", action="store_true", help="run the severity grading pass")
    ap.add_argument("--allocate", action="store_true", help="free: write the allocation")
    ap.add_argument("--report", action="store_true", help="free: reprint the allocation")
    ap.add_argument("--grade-plan", action="store_true",
                    help="grade exactly the facts the current allocation used that still "
                         "carry a fallback grade - keeps the severity axis one instrument")
    ap.add_argument("--include-single-site", action="store_true",
                    help="also grade single-site facts (they can only serve complete cells)")
    ap.add_argument("--grade-limit", type=int, default=0, help="0 = every eligible fact")
    ap.add_argument("--target", type=int, default=TARGET_PER_CELL)
    ap.add_argument("--eval-only", action="store_true", default=True)
    ap.add_argument("--dev-only", action="store_true",
                    help="work the GEPA dev split instead: those consultations are already "
                         "burned for evaluation, so pairs built on them cost the eval set "
                         "nothing and give W-F the training signal it is short of")
    ap.add_argument("--include-dev-pool", dest="eval_only", action="store_false")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget", type=float, default=60.0)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    if args.dev_only:
        globals()["_BUILT_STATE"] = "omission_v2_dev_state.json"
    doc, cons_rows = INJ.load_sites()
    cands = eligible(cons_rows)
    left = 0
    sev = load_sev()
    cf = carried_forward()
    print(f"site map: {len(cons_rows)} consultations | {len(cands)} construction-eligible "
          f"facts | multi-site {sum(1 for _, _, o in cands if len(o) > 1)}")

    if args.grade:
        budget = OS.Budget(sev, args.budget)
        # retry an arm that errored, not just one that was never attempted: grade_arm stores
        # a record either way, so keying on presence alone would strand every failed call
        if args.grade_plan and os.path.exists(ALLOC_DEV_FILE if args.dev_only else ALLOC_FILE):
            af = ALLOC_DEV_FILE if args.dev_only else ALLOC_FILE
            want = {r["fact_uid"] for r in json.load(open(af))["plan"]
                    if r.get("severity_source") != "rubric_2arm_consensus"}
            queue = [(c, f) for c, f, _ in cands if fact_uid(c, f) in want]
        else:
            queue = grade_order(cands, sev, args.include_single_site, args.eval_only,
                                args.dev_only)
        todo = [(c, f) for c, f in queue
                if any(not (sev["facts"].get(fact_uid(c, f), {}).get("arms") or {})
                       .get(a, {}).get("importance") for a in ARMS)]
        if args.grade_limit:
            todo = todo[:args.grade_limit]
        if not args.yes and todo:
            try:
                if input(f"grade {len(todo)} facts x 2 arms (cap ${args.budget:.0f})? [y/N] "
                         ).strip().lower() != "y":
                    sys.exit("aborted")
            except EOFError:
                sys.exit("aborted (non-interactive)")
        with Run(EXPERIMENT, params={"stage": "severity", "n_facts": len(todo),
                                     "include_single_site": args.include_single_site,
                                     "budget_cap_usd": args.budget},
                 seed=GRADE_SEED, inputs=["master/fact_sites.json"], spec=SPEC,
                 allow_dirty=args.allow_dirty) as run:
            run.register_prompts({"RUBRIC_GRADE": HN.RUBRIC_GRADE, "SEVERITY_RUBRIC": HN.RUBRIC})
            run_grades(todo, sev, budget, args)
            run.save("severity_summary.json", severity_summary(sev))
        print(json.dumps(severity_summary(sev)["by_grade"], indent=1))
        if budget.hit.is_set():
            print(f"\nBUDGET STOP at ${budget.spent:.2f} (cap ${budget.cap:.0f})")
            return 3
        left = sum(1 for c, f in queue
                   if any(not (sev["facts"].get(fact_uid(c, f), {}).get("arms") or {})
                          .get(a, {}).get("importance") for a in ARMS))

    if args.grade or args.allocate or args.report:
        plan, cells, rows = allocate(cands, sev, args.target, eval_only=args.eval_only,
                                     dev_only=args.dev_only)
        out = write_allocation(doc, plan, cells, rows, sev, cf, args)
        write_md(out, cf, args.dev_only)
        report(out, cf)
        if args.grade and left:
            print(f"\n{left} facts still ungraded (failed calls) - rerun to retry")
            return 2
        return 0
    sys.exit("pick a stage: --grade, --allocate or --report")


if __name__ == "__main__":
    sys.exit(main())
