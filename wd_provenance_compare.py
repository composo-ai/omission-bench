"""W-D Amendment 2026-07-30 part N - the two provenance readings, WD-R4 and WD-R5.

The authored 30 now have TWO fact sheets each: the June authored sheet (written by the same author
who wrote the transcript, `authored_scenarios.json`) and a blind sheet extracted from the
transcript alone by the same instrument used on PriMock/ACI/trap-blind
(`master/fact_sheets_authored_extracted.json`). Same 30 transcripts, two provenances - which is
what isolates the sheet-provenance effect from the trap-seeding effect WD-R2 tests.

WD-R4 (sheet-provenance effect). must_contain / must_not_contain / trap counts either side, the
importance distribution either side, and the headline: what fraction of authored must_contain
facts does blind extraction RECOVER? Wording, granularity and ordering differ freely between the
two sheets, so string matching would measure prose style, not recall - recovery is judged
SEMANTICALLY, one `claude-opus-5` call per scenario (plan path), each call seeing the transcript
plus both numbered lists and returning a per-authored-fact recovered/not verdict. Reported with a
Wilson 95% CI against the pre-registered >=80% threshold, plus a consultation-level cluster
bootstrap CI because facts are clustered within consultations and Wilson is not.

WD-R5 (severity-grade agreement). The authored traps carry `importance` grades backfilled by
`claude-opus-4-8` (Amendment 2026-07-29b part H); the blind sheets carry `claude-opus-5` native
grades. TRAP-MATCHING METHOD: traps are matched semantically, one `claude-opus-5` call per
scenario, pairing each authored trap with the extracted trap that refers to the SAME moment and
failure mode in the dialogue (mode label ignored - a trap can be graded the same while the two
sheets label its mode differently); only pairs the matcher affirms as the same moment enter the
agreement analysis, and unmatched traps on either side are reported as counts, never imputed.
Agreement on the critical/supporting/peripheral scale is Cohen's kappa (unweighted; the scale is
ordered but the pre-registration says kappa, so kappa is what ships) with a consultation-level
cluster-bootstrap CI.

Primary analysis runs on the KEPT (post-critique) extracted sheets - that is the artifact the
study consumes. A sensitivity pass over all 30 RAW extracted sheets is reported alongside so the
gate-2 drops cannot be mistaken for a selection effect.

Usage: python wd_provenance_compare.py [--workers 10] [--set kept|raw|both] [--boot 10000]
Output: master/wd_provenance_report.json  (+ master/wd_provenance_matches.json, the cached
        per-scenario semantic match calls, so reruns are free and auditable)
"""
import hashlib, json, math, os, random, statistics, sys, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from common import claude_json, HERE

MODEL = "claude-opus-5"
EFFORT = "medium"
TIMEOUT = 420
SEED = 20260728
IMPORTANCE = ("critical", "supporting", "peripheral")


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


WORKERS = int(_arg("--workers", "10"))
WHICH = _arg("--set", "both")
NBOOT = int(_arg("--boot", "10000"))
CACHE = os.path.join(HERE, "master", "wd_provenance_matches.json")
OUT = os.path.join(HERE, "master", "wd_provenance_report.json")

MATCH_MC = """You are auditing whether two independently-produced ground-truth fact sheets of the SAME
consultation capture the same clinical facts. SHEET_A was written by the author who also wrote the
consultation. SHEET_B was extracted blind from the transcript alone by a different process.
Wording, granularity, ordering and level of detail differ freely between them - judge MEANING, not
words.

TRANSCRIPT:
{transcript}

SHEET_A must_contain (the reference list):
{a_list}

SHEET_B must_contain (the blind extraction):
{b_list}

For EVERY numbered item in SHEET_A decide whether SHEET_B captures the same clinical fact. It
counts as RECOVERED if a clinician reading SHEET_B would come away knowing that fact - even if
SHEET_B splits it across several items, states it more or less precisely, or embeds it inside a
larger item. It is NOT recovered if SHEET_B merely mentions the same topic without the fact, or
states something materially different.

Output ONLY JSON:
{"matches": [ {"a": <SHEET_A item number>, "b": [<SHEET_B item numbers that carry it, [] if none>],
"recovered": true|false, "note": "<12 words max>"} ]}
One entry per SHEET_A item, in order. No commentary, no markdown fences."""

MATCH_TRAPS = """Two independently-produced lists of SALIENCE TRAPS for the SAME consultation - moments in
the dialogue where a clinical scribe could plausibly go wrong. LIST_A was written by the author of
the consultation; LIST_B was derived blind from the transcript alone. They overlap partially and
word things differently.

TRANSCRIPT:
{transcript}

LIST_A:
{a_list}

LIST_B:
{b_list}

For each trap in LIST_A, find the ONE trap in LIST_B that refers to the SAME MOMENT in the
dialogue and the SAME way of getting it wrong. Ignore how each list labels the failure mode and
ignore wording - two entries match if a clinician would say they are about the same risk at the
same point in the consultation. If nothing in LIST_B refers to that moment, return null.

Output ONLY JSON:
{"pairs": [ {"a": <LIST_A number>, "b": <LIST_B number or null>,
"same_moment": true|false, "note": "<12 words max>"} ]}
One entry per LIST_A trap, in order. No commentary, no markdown fences."""


# ---------------------------------------------------------------- stats
def wilson(k, n, z=1.96):
    """Wilson score interval - the study-wide standard for proportions (spec section 6)."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, centre - half), min(1.0, centre + half)


def cohens_kappa(pairs, cats=IMPORTANCE):
    """Unweighted Cohen's kappa over (rater_a, rater_b) label pairs.
    kappa = (po - pe) / (1 - pe); pe from the two raters' marginal distributions."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    ma = Counter(a for a, _ in pairs)
    mb = Counter(b for _, b in pairs)
    pe = sum((ma.get(c, 0) / n) * (mb.get(c, 0) / n) for c in cats)
    if abs(1 - pe) < 1e-12:
        return None            # no chance-corrected signal available (one category only)
    return (po - pe) / (1 - pe)


def _kappa_selfcheck():
    """Sanity-check the kappa implementation on cases with known answers before trusting it."""
    perfect = [("critical", "critical")] * 6 + [("supporting", "supporting")] * 4
    assert abs(cohens_kappa(perfect) - 1.0) < 1e-9, "kappa of perfect agreement must be 1.0"
    # total disagreement with identical marginals -> negative kappa
    swap = [("critical", "supporting")] * 5 + [("supporting", "critical")] * 5
    assert cohens_kappa(swap) < 0, "kappa of systematic disagreement must be negative"
    # textbook case: po=0.7, marginals a=(.6,.4) b=(.7,.3) -> pe=.54, kappa=(.7-.54)/.46=.3478
    tb = ([("critical", "critical")] * 6 + [("supporting", "supporting")] * 1
          + [("supporting", "critical")] * 1 + [("critical", "supporting")] * 2)
    k = cohens_kappa(tb)
    exp = (0.7 - (0.8 * 0.7 + 0.2 * 0.3)) / (1 - (0.8 * 0.7 + 0.2 * 0.3))
    assert abs(k - exp) < 1e-9, f"kappa mismatch {k} vs {exp}"
    # independence -> kappa ~ 0
    rng = random.Random(1)
    ind = [(rng.choice(IMPORTANCE), rng.choice(IMPORTANCE)) for _ in range(20000)]
    assert abs(cohens_kappa(ind)) < 0.03, "kappa under independence must be ~0"
    return {"perfect": 1.0, "systematic_disagreement_negative": True,
            "textbook_case_kappa": round(exp, 6), "independence_kappa_abs_lt": 0.03,
            "verdict": "kappa implementation verified"}


def cluster_bootstrap(clusters, stat, nboot=None, seed=SEED):
    """Percentile CI resampling CONSULTATIONS with replacement (spec section 6: cluster bootstrap
    at consultation level). `clusters` = list of per-consultation payloads; `stat` maps a list of
    payloads to a scalar (or None when the resample is degenerate)."""
    nboot = nboot or NBOOT
    if not clusters:
        return None
    rng = random.Random(seed)
    n, vals = len(clusters), []
    for _ in range(nboot):
        draw = [clusters[rng.randrange(n)] for _ in range(n)]
        v = stat(draw)
        if v is not None:
            vals.append(v)
    if len(vals) < nboot * 0.5:
        return {"lo": None, "hi": None, "n_valid_draws": len(vals), "nboot": nboot}
    vals.sort()
    return {"lo": vals[int(0.025 * len(vals))], "hi": vals[int(0.975 * len(vals)) - 1],
            "n_valid_draws": len(vals), "nboot": nboot}


# ---------------------------------------------------------------- matching
_lock = threading.Lock()
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def _numbered(items):
    return "\n".join(f"{i + 1}. {t}" for i, t in enumerate(items))


def ckey(tag, sid, kind, a_sheet, b_sheet):
    """Cache key includes a hash of BOTH sheets: if a sheet is revised (a critic revision cycle,
    a re-extraction) the cached match silently becomes wrong, so the key must move with it."""
    h = hashlib.sha256(json.dumps([a_sheet, b_sheet], sort_keys=True).encode()).hexdigest()[:12]
    return f"{tag}|{sid}|{kind}|{h}"


def mc_text(sheet):
    """authored must_contain items are plain strings (June schema); extracted ones are objects."""
    out = []
    for it in sheet["must_contain"]:
        out.append(it if isinstance(it, str) else it.get("fact", ""))
    return out


def trap_text(sheet):
    return [f"{t.get('trap','')} -- correct handling: {t.get('correct_handling','')}"
            for t in sheet["salience_traps"]]


def match_one(job):
    """One scenario: the must_contain recovery call and the trap-pairing call (cached by key)."""
    tag, sid, transcript, a_sheet, b_sheet = job
    for kind, prompt, a_items, b_items in (
            ("mc", MATCH_MC, mc_text(a_sheet), mc_text(b_sheet)),
            ("traps", MATCH_TRAPS, trap_text(a_sheet), trap_text(b_sheet))):
        key = ckey(tag, sid, kind, a_sheet, b_sheet)
        with _lock:
            if key in cache:
                continue
        p = (prompt.replace("{transcript}", transcript)
                   .replace("{a_list}", _numbered(a_items))
                   .replace("{b_list}", _numbered(b_items)))
        res, want = None, "matches" if kind == "mc" else "pairs"
        for _ in range(3):
            r = claude_json(p, model=MODEL, effort=EFFORT, timeout=TIMEOUT, retries=1)
            if isinstance(r, dict) and isinstance(r.get(want), list):
                res = r
                break
        with _lock:
            if res is None:
                print(f"  {sid:26} {kind:5} MATCH CALL FAILED (rerun to retry)", flush=True)
            else:
                cache[key] = {"n_a": len(a_items), "n_b": len(b_items), **res}
                json.dump(cache, open(CACHE + ".tmp", "w"), ensure_ascii=False, indent=1)
                os.replace(CACHE + ".tmp", CACHE)
                if kind == "mc":
                    rec = sum(1 for m in res["matches"] if m.get("recovered"))
                    print(f"  {sid:26} mc    {rec}/{len(a_items)} authored facts recovered "
                          f"(B has {len(b_items)})", flush=True)
                else:
                    mt = sum(1 for m in res["pairs"] if m.get("same_moment") and m.get("b"))
                    print(f"  {sid:26} traps {mt}/{len(a_items)} traps matched "
                          f"(B has {len(b_items)})", flush=True)


# ---------------------------------------------------------------- analysis
def analyse(tag, authored, extracted):
    ids = [s["id"] for s in authored if s["id"] in extracted]
    a_by = {s["id"]: s for s in authored}
    per = []
    missing = []
    for sid in ids:
        a, b = a_by[sid]["fact_sheet"], extracted[sid]["fact_sheet"]
        km, kt = ckey(tag, sid, "mc", a, b), ckey(tag, sid, "traps", a, b)
        if km not in cache or kt not in cache:
            missing.append(sid)
            continue
        mc = cache[km]["matches"]
        a_mc = mc_text(a)
        seen, rows = set(), []
        for m in mc:                       # guard against duplicate/out-of-range indices
            i = m.get("a")
            if not isinstance(i, int) or not (1 <= i <= len(a_mc)) or i in seen:
                continue
            seen.add(i)
            rows.append(bool(m.get("recovered")))
        # traps: only matcher-affirmed same-moment pairs enter the grade comparison
        gp = []
        b_traps = b["salience_traps"]
        a_traps = a["salience_traps"]
        for m in cache[kt]["pairs"]:
            i, j = m.get("a"), m.get("b")
            if not (m.get("same_moment") and isinstance(i, int) and isinstance(j, int)):
                continue
            if not (1 <= i <= len(a_traps) and 1 <= j <= len(b_traps)):
                continue
            ga, gb = a_traps[i - 1].get("importance"), b_traps[j - 1].get("importance")
            if ga in IMPORTANCE and gb in IMPORTANCE:
                gp.append((ga, gb))
        per.append({
            "id": sid,
            "authored_mc": len(a["must_contain"]), "extracted_mc": len(b["must_contain"]),
            "authored_mnc": len(a["must_not_contain"]), "extracted_mnc": len(b["must_not_contain"]),
            "authored_traps": len(a_traps), "extracted_traps": len(b_traps),
            "mc_judged": len(rows), "mc_recovered": sum(rows),
            "traps_matched": len(gp), "grade_pairs": gp,
            "authored_importance": dict(Counter(t.get("importance") for t in a_traps)),
            "extracted_importance": dict(Counter(t.get("importance") for t in b_traps))})

    def mean(f):
        v = [f(p) for p in per]
        return {"mean": round(statistics.mean(v), 3) if v else None,
                "sd": round(statistics.stdev(v), 3) if len(v) > 1 else None,
                "total": sum(v)}

    # WD-R4 headline: recovery of authored must_contain facts by blind extraction
    k = sum(p["mc_recovered"] for p in per)
    n = sum(p["mc_judged"] for p in per)
    p_hat, lo, hi = wilson(k, n)
    boot = cluster_bootstrap(
        per, lambda d: (sum(x["mc_recovered"] for x in d) / sum(x["mc_judged"] for x in d)
                        if sum(x["mc_judged"] for x in d) else None))
    per_consult = [p["mc_recovered"] / p["mc_judged"] for p in per if p["mc_judged"]]

    # WD-R5: severity-grade agreement on matched traps
    gp = [g for p in per for g in p["grade_pairs"]]
    kappa = cohens_kappa(gp)
    kboot = cluster_bootstrap(
        [p for p in per if p["grade_pairs"]],
        lambda d: cohens_kappa([g for x in d for g in x["grade_pairs"]]))
    conf = {a: {b: 0 for b in IMPORTANCE} for a in IMPORTANCE}
    for a, b in gp:
        conf[a][b] += 1

    return {
        "set": tag, "n_consultations": len(per), "ids": [p["id"] for p in per],
        "missing_match_calls": missing,
        "counts": {
            "authored_must_contain": mean(lambda p: p["authored_mc"]),
            "extracted_must_contain": mean(lambda p: p["extracted_mc"]),
            "authored_must_not_contain": mean(lambda p: p["authored_mnc"]),
            "extracted_must_not_contain": mean(lambda p: p["extracted_mnc"]),
            "authored_traps": mean(lambda p: p["authored_traps"]),
            "extracted_traps": mean(lambda p: p["extracted_traps"])},
        "importance_distribution": {
            "authored_opus_4_8_backfill": dict(Counter(
                g for p in per for g, c in p["authored_importance"].items() for _ in range(c))),
            "extracted_opus_5_native": dict(Counter(
                g for p in per for g, c in p["extracted_importance"].items() for _ in range(c)))},
        "WD_R4": {
            "definition": "fraction of AUTHORED must_contain facts a clinician would recover from "
                          "the blind-extracted sheet, judged semantically by claude-opus-5, one "
                          "call per consultation",
            "authored_facts_judged": n, "recovered": k,
            "recovery": round(p_hat, 4),
            "wilson_95ci": [round(lo, 4), round(hi, 4)],
            "wilson_caveat": "Wilson treats the facts as independent; they are clustered within "
                             "consultations, so the cluster bootstrap below is the honest CI",
            "cluster_bootstrap_95ci": (None if not boot else
                                       [round(boot["lo"], 4), round(boot["hi"], 4)]),
            "cluster_bootstrap_detail": boot,
            "per_consultation_recovery": {
                "mean": round(statistics.mean(per_consult), 4) if per_consult else None,
                "sd": round(statistics.stdev(per_consult), 4) if len(per_consult) > 1 else None,
                "min": round(min(per_consult), 4) if per_consult else None,
                "max": round(max(per_consult), 4) if per_consult else None},
            "threshold": 0.80,
            "clears_threshold_point_estimate": p_hat >= 0.80,
            "clears_threshold_ci_lower_bound": (lo >= 0.80),
            "reading": None},
        "WD_R5": {
            "trap_matching_method": "semantic pairing, one claude-opus-5 call per consultation; "
                                    "authored trap -> extracted trap referring to the SAME moment "
                                    "and same way of going wrong, mode label ignored; only pairs "
                                    "the matcher affirms as same_moment enter the analysis; "
                                    "unmatched traps are counted, never imputed",
            "graders": {"authored": "claude-opus-4-8 (Amendment 2026-07-29b H backfill)",
                        "extracted": "claude-opus-5 (native, in the extraction call)"},
            "matched_trap_pairs": len(gp),
            "authored_traps_total": sum(p["authored_traps"] for p in per),
            "extracted_traps_total": sum(p["extracted_traps"] for p in per),
            "match_rate_of_authored_traps": (round(len(gp) / sum(p["authored_traps"] for p in per), 4)
                                             if per else None),
            "raw_agreement": round(sum(1 for a, b in gp if a == b) / len(gp), 4) if gp else None,
            "cohens_kappa": round(kappa, 4) if kappa is not None else None,
            "kappa_cluster_bootstrap_95ci": (None if not kboot or kboot["lo"] is None else
                                             [round(kboot["lo"], 4), round(kboot["hi"], 4)]),
            "kappa_cluster_bootstrap_detail": kboot,
            "kappa_weighting": "unweighted (pre-registration says Cohen's kappa; the scale is "
                               "ordered so unweighted kappa is the conservative reading)",
            "confusion_matrix_authored_rows_x_extracted_cols": conf},
        "per_consultation": per}


def main():
    authored = json.load(open(os.path.join(HERE, "authored_scenarios.json")))
    sets = {}
    if WHICH in ("kept", "both"):
        sets["kept"] = {r["id"]: r for r in json.load(open(
            os.path.join(HERE, "master", "fact_sheets_authored_extracted.json")))}
    if WHICH in ("raw", "both"):
        sets["raw"] = {r["id"]: r for r in json.load(open(
            os.path.join(HERE, "master", "fact_sheets_raw_authored_extracted.json")))}

    jobs = []
    for tag, ext in sets.items():
        for s in authored:
            if s["id"] in ext:
                jobs.append((tag, s["id"], s["transcript"], s["fact_sheet"],
                             ext[s["id"]]["fact_sheet"]))
    todo = [j for j in jobs
            if ckey(j[0], j[1], "mc", j[3], j[4]) not in cache
            or ckey(j[0], j[1], "traps", j[3], j[4]) not in cache]
    print(f"semantic matching on {MODEL}: {len(jobs)} scenario-set pairs, {len(todo)} need calls "
          f"({WORKERS} workers)", flush=True)
    if todo:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(match_one, todo))

    report = {
        "spec": "specs/w-d-master-dataset.md Amendment 2026-07-30 part N (WD-R4, WD-R5)",
        "model": MODEL, "effort": EFFORT, "route": "plan (claude -p)", "seed": SEED,
        "nboot": NBOOT, "kappa_selfcheck": _kappa_selfcheck(),
        "note": "primary = `kept` (post-critique sheets, the artifact the study consumes); "
                "`raw` = all pre-critique extracted sheets, reported so the gate-2 drops cannot "
                "be mistaken for a selection effect",
        "analyses": {}}
    for tag in sets:
        report["analyses"][tag] = analyse(tag, authored, sets[tag])

    # pre-registered reading, stated mechanically from the primary (kept) analysis
    prim = report["analyses"].get("kept") or next(iter(report["analyses"].values()))
    r4 = prim["WD_R4"]
    r4["reading"] = (
        "HIGH RECOVERY (>=80%): blind extraction is a faithful substitute for the authored sheet; "
        "the extracted strata are not systematically thinner, which is what licenses pooling "
        "strata in W2's primary analysis."
        if r4["clears_threshold_point_estimate"] else
        "LOW RECOVERY (<80%): the authored and extracted sheets are not equivalent instruments; "
        "per Amendment 2026-07-30 part O.2 the pre-registered response is PER-STRATUM REPORTING "
        "as primary, NOT retuning the extraction prompt until the number improves.")
    json.dump(report, open(OUT, "w"), indent=1)

    for tag, a in report["analyses"].items():
        r4, r5 = a["WD_R4"], a["WD_R5"]
        print(f"\n===== {tag}: {a['n_consultations']} consultations =====")
        print(f"WD-R4 recovery {r4['recovery']:.1%} ({r4['recovered']}/{r4['authored_facts_judged']}"
              f" authored facts) | Wilson 95% CI [{r4['wilson_95ci'][0]:.1%}, "
              f"{r4['wilson_95ci'][1]:.1%}]"
              + (f" | cluster boot [{r4['cluster_bootstrap_95ci'][0]:.1%}, "
                 f"{r4['cluster_bootstrap_95ci'][1]:.1%}]" if r4["cluster_bootstrap_95ci"] else ""))
        print(f"      vs >=80% threshold: point estimate "
              f"{'CLEARS' if r4['clears_threshold_point_estimate'] else 'DOES NOT CLEAR'}; "
              f"CI lower bound {'clears' if r4['clears_threshold_ci_lower_bound'] else 'below'}")
        print(f"      must_contain mean: authored {a['counts']['authored_must_contain']['mean']} "
              f"vs extracted {a['counts']['extracted_must_contain']['mean']}")
        print(f"      traps mean: authored {a['counts']['authored_traps']['mean']} "
              f"vs extracted {a['counts']['extracted_traps']['mean']}")
        print(f"WD-R5 kappa {r5['cohens_kappa']} on {r5['matched_trap_pairs']} matched trap pairs "
              f"(raw agreement {r5['raw_agreement']})"
              + (f" | cluster boot 95% CI {r5['kappa_cluster_bootstrap_95ci']}"
                 if r5["kappa_cluster_bootstrap_95ci"] else ""))
    print(f"\n-> {os.path.relpath(OUT, HERE)}", flush=True)


if __name__ == "__main__":
    main()
