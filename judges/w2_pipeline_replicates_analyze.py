"""w2_pipeline_replicates_analyze.py - read-out for the pipeline REPLICATE round (25 Aug 2026).

Why this exists. Every published pipeline number is a single replicate: B2's 0.801 paired
discrimination on omissions, B3's 0.752, and the per-fact critical rule's 20.6% detection at a
2.1% false-alarm rate. They are quoted against a monolithic baseline whose OWN three-run spread
is 0.118 wide (0.531 to 0.649), and the published head-to-head compares route two read as a
MAJORITY OF THREE against route one read as ONE DRAW. This round buys replicates 2 and 3 of both
tiers over the identical 198 held-out notes (seeds 22 and 33, against replicate 1's 11) and
reports what the single-run numbers become.

Two deliberate departures from a naive implementation, both to protect existing artifacts:

  1. The new replicates are NOT merged into results/w2-pipeline/_state/confirm-B{2,3}.jsonl.
     w2_evalfull_analyze.py builds `b3_by_note` with "one replicate; last write wins" (its
     line 104), so appending replicates 2 and 3 to that store would silently change the
     published pipeline numbers. Each replicate keeps its own tag store and this script
     reads all three explicitly, filtering on `replicate`.
  2. `paired`, `wilson` and `mcnemar_exact` are IMPORTED from the modules that produced the
     published figures rather than reimplemented, so the convention cannot drift. Spot-checked
     on import: wilson(27,131) -> 0.2061 and mcnemar_exact(25,7) -> 0.002102, both matching
     what the paper already reports.

Substrates. The held-out subset is the 198 notes of master/arms_confirm_subset.json (151 pairs
over 47 consultations: 131 omissions, 20 commissions, 47 clean twins). The evaluation set is the
607 notes of dataset factorial-v1 (293 omission notes, 112 clean twins) - route one has one
replicate there and this round did not buy more, so the evaluation-set contrast is equalised in
the other direction, by reading route two as single draws.

    python3 judges/w2_pipeline_replicates_analyze.py
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import collections
import json
import os
import statistics

from common import RESULTS
import w2_pipeline_analyze as PA          # paired()  - tie-adjusted paired discrimination
import w2_evalfull_analyze as EV          # wilson(), mcnemar_exact(), FLAG_BELOW

OUT_DIR = os.path.join(RESULTS, "w2-pipeline-replicates")
STATE = os.path.join(RESULTS, "w2-pipeline", "_state")
HELD_OUT_DV = "factorial-v1-armconfirm"

# tag store -> which replicate it holds. Replicate 1 lives in the original confirm store; the
# 25 Aug round wrote one store per replicate so nothing existing was touched.
# Tag names carry no hyphen after "confirm" on purpose. w2_pipeline_analyze.py resolves a tier
# by globbing "<tag>-*.jsonl" and taking the filename's last hyphen-separated field, so a
# replicate tagged "confirm-b3r2" produces confirm-b3r2-B3.jsonl, which MATCHES the glob for
# --tag confirm and, sorting after confirm-B3.jsonl, silently replaced replicate 1 as "the" B3
# store. Renamed 2026-08-25 after the runs finished; the run manifests record the tag the run
# was bought under, so a manifest's record_stores path is the pre-rename name.
B2_STORES = {1: "confirm-B2.jsonl", 2: "confirmr2-B2.jsonl", 3: "confirmr3-B2.jsonl"}
B3_STORES = {1: "confirm-B3.jsonl", 2: "confirmr2-B3.jsonl", 3: "confirmr3-B3.jsonl"}


# A held-out replicate is 198 notes. A store with fewer is a run still in flight, and its
# rates are a partial-data artifact, not a measurement - the 25 Aug smoke test read 6.9% and
# 13.7% detection off two half-finished B3 runs against replicate 1's true 20.6%. Anything
# short of the full count is refused rather than reported.
HELD_OUT_NOTES = 198


def read_records(fname, replicate, dataset_version=None, require=None):
    """Records from one tag store, filtered to one replicate (and optionally one substrate).

    Returns None for absent or INCOMPLETE stores, so a run still in flight can never reach
    the read-out as if it were a measurement.
    """
    path = os.path.join(STATE, fname)
    if not os.path.exists(path):
        return None
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue                       # torn final line from a killed process
        if r.get("replicate") != replicate:
            continue
        if dataset_version and r.get("dataset_version") != dataset_version:
            continue
        out.append(r)
    if require is not None and len(out) != require:
        return {"__incomplete__": True, "have": len(out), "want": require, "path": fname}
    return out


def rule_flag(rec):
    """Route one's deployed predicate: flag the note if any fact graded critical is absent.

    Identical read to w2_evalfull_analyze.b3_flag and w2_pipeline_analyze:180 - the per-fact
    verdicts live in detail.missing_facts, already bought; nothing is re-judged here.
    """
    for f in (rec.get("detail") or {}).get("missing_facts") or []:
        if f.get("severity") == "critical" and f.get("verdict") == "absent":
            return True
    return False


def roster_of(recs):
    """note_key -> the fields the splits need, taken from the tier's own records."""
    out = {}
    for r in recs:
        nk = r.get("note_key")
        if nk:
            out[nk] = {"role": r.get("note_role"), "pair_class": r.get("pair_class"),
                       "pair_type": r.get("pair_type"), "clean_key": r.get("clean_key"),
                       "severity": r.get("severity"),
                       "consultation": f"{r.get('stratum')}|{r.get('consultation')}"}
    return out


def paired_for(recs, want):
    """Tie-adjusted paired discrimination over (errored, its own clean twin) score pairs.

    `want` is 'omit' or a commission set; the clean twin is looked up by the record's own
    clean_key, so a note is never paired against anything but its twin.
    """
    score = {r["note_key"]: r.get("aggregate") for r in recs}
    rows = [(score.get(r["note_key"]), score.get(r.get("clean_key")))
            for r in recs
            if r.get("note_role") == "errored"
            and (r.get("pair_type") == "omit" if want == "omit"
                 else r.get("pair_type") != "omit")]
    return PA.paired(rows)


def spread(values):
    """The study's reporting convention for >=3 runs: mean, sd, and the observed range."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    out = {"per_replicate": [round(v, 4) for v in vals], "n_runs": len(vals),
           "mean": round(statistics.mean(vals), 4),
           "min": round(min(vals), 4), "max": round(max(vals), 4),
           "range": round(max(vals) - min(vals), 4)}
    out["sd"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else None
    return out


def rate(k, n):
    p, lo, hi = EV.wilson(k, n)
    return {"k": k, "n": n, "p": round(p, 4), "wilson": [round(lo, 4), round(hi, 4)]}


def head_to_head(omission_notes, roster, flag_a, flag_b):
    """Consultation-collapsed 2x2 + exact McNemar, the unit the published head-to-head uses.

    a = route two (the budgeted single call), b = route one (the per-fact critical rule),
    matching the orientation of the published head_to_head_consultation block so the counts
    are read the same way round.
    """
    by_cons = collections.defaultdict(list)
    for nk in omission_notes:
        by_cons[roster[nk]["consultation"]].append(nk)
    both = a_only = b_only = neither = 0
    for _cons, nks in sorted(by_cons.items()):
        a = any(flag_a(nk) for nk in nks)
        b = any(flag_b(nk) for nk in nks)
        both += a and b
        a_only += a and not b
        b_only += b and not a
        neither += (not a) and (not b)
    return {"n_consultations": len(by_cons), "both": both,
            "route_two_only": a_only, "rule_only": b_only, "neither": neither,
            "union": both + a_only + b_only,
            "mcnemar_p": round(EV.mcnemar_exact(a_only, b_only), 6)}


def route_two_scores(dataset_version):
    """note_key -> replicate -> score, for the budgeted single call (arm power-gepa-high)."""
    path = os.path.join(RESULTS, "w2-power", "_state", "power.jsonl")
    scores = collections.defaultdict(dict)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("arm") != "power-gepa-high":
            continue
        if dataset_version and r.get("dataset_version") != dataset_version:
            continue
        scores[r["note_key"]][r["replicate"]] = r.get("aggregate")
    return scores


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out = {"substrate": {}, "missing_replicates": []}

    # ---------------------------------------------------------------- load both tiers
    tiers = {}
    for tier, stores in (("B2", B2_STORES), ("B3", B3_STORES)):
        got = {}
        for rep, fname in stores.items():
            recs = read_records(fname, rep, HELD_OUT_DV, require=HELD_OUT_NOTES)
            if isinstance(recs, dict) and recs.get("__incomplete__"):
                out["missing_replicates"].append(
                    f"{tier} replicate {rep} INCOMPLETE ({recs['have']}/{recs['want']} notes "
                    f"in {fname}) - run still in flight, excluded")
            elif recs:
                got[rep] = recs
            else:
                out["missing_replicates"].append(f"{tier} replicate {rep} absent ({fname})")
        tiers[tier] = got

    # roster + note sets from B3 replicate 1 (every tier judges the identical 198 notes)
    base = tiers["B3"].get(1) or tiers["B2"].get(1)
    roster = roster_of(base)
    omission_notes = [nk for nk, m in roster.items()
                      if m["role"] == "errored" and str(m["pair_class"]).startswith("omit")]
    clean_notes = [nk for nk, m in roster.items() if m["role"] == "clean"]
    critical_notes = [nk for nk in omission_notes if roster[nk].get("severity") == "critical"]
    out["substrate"]["held_out"] = {
        "dataset_version": HELD_OUT_DV, "notes": len(roster),
        "omission_notes": len(omission_notes), "clean_notes": len(clean_notes),
        "critical_omission_notes": len(critical_notes),
        "consultations": len({m["consultation"] for m in roster.values()}),
        "replicates_available": {t: sorted(v) for t, v in tiers.items()}}

    # ------------------------------------------- 1. paired discrimination, per replicate
    out["paired_discrimination"] = {}
    for tier, got in tiers.items():
        om = {rep: (paired_for(recs, "omit") or {}).get("paired")
              for rep, recs in sorted(got.items())}
        cm = {rep: (paired_for(recs, "comm") or {}).get("paired")
              for rep, recs in sorted(got.items())}
        out["paired_discrimination"][tier] = {
            "omissions": spread(list(om.values())),
            "omissions_by_replicate": {str(k): v for k, v in om.items()},
            "commissions": spread(list(cm.values())),
            "commissions_by_replicate": {str(k): v for k, v in cm.items()}}

    # ------------------------------- 2. the per-fact critical rule, per replicate + majority
    b3 = tiers["B3"]
    flags = {rep: {r["note_key"]: rule_flag(r) for r in recs} for rep, recs in b3.items()}

    def majority(nk):
        votes = [flags[rep].get(nk) for rep in sorted(flags) if nk in flags[rep]]
        votes = [v for v in votes if v is not None]
        return sum(1 for v in votes if v) * 2 > len(votes) if votes else False

    per_rep = {}
    for rep in sorted(flags):
        f = flags[rep]
        per_rep[str(rep)] = {
            "detection": rate(sum(1 for nk in omission_notes if f.get(nk)), len(omission_notes)),
            "false_alarms": rate(sum(1 for nk in clean_notes if f.get(nk)), len(clean_notes)),
            "critical": rate(sum(1 for nk in critical_notes if f.get(nk)), len(critical_notes))}
    out["critical_rule"] = {
        "per_replicate": per_rep,
        "detection_spread": spread([per_rep[r]["detection"]["p"] for r in per_rep]),
        "false_alarm_spread": spread([per_rep[r]["false_alarms"]["p"] for r in per_rep]),
        "critical_spread": spread([per_rep[r]["critical"]["p"] for r in per_rep]),
        "majority_of_three": {
            "detection": rate(sum(1 for nk in omission_notes if majority(nk)),
                              len(omission_notes)),
            "false_alarms": rate(sum(1 for nk in clean_notes if majority(nk)),
                                 len(clean_notes)),
            "critical": rate(sum(1 for nk in critical_notes if majority(nk)),
                             len(critical_notes)),
            "n_replicates_voting": len(flags)}}

    # ------------------------------------- 3. the head-to-head with estimators made equal
    #
    # The objection is that the published head-to-head compares route two's MAJORITY OF
    # THREE against route one's ONE DRAW. There are two honest ways to equalise, and they
    # point opposite ways, so both are reported and neither is chosen here.
    r2_held = route_two_scores(HELD_OUT_DV)

    def r2_flag_rep(nk, rep):
        s = r2_held.get(nk, {}).get(rep)
        return s is not None and s < EV.FLAG_BELOW

    def r2_flag_majority(nk):
        votes = [1 if (s is not None and s < EV.FLAG_BELOW) else 0
                 for s in r2_held.get(nk, {}).values()]
        return sum(votes) * 2 > len(votes) if votes else False

    # Route two's OWN operating point under each reading. This is not decoration: a majority
    # of three fires only when 2 of 3 runs fire, so it is strictly more conservative than any
    # single run. Levelling the head-to-head down to single draws therefore moves route two to
    # a different, noisier operating point, and the contrast must be read with that in view
    # rather than as a free gain. Reproduces the published per-replicate figures.
    r2_rates = {}
    for rep in sorted({r for v in r2_held.values() for r in v}):
        r2_rates[str(rep)] = {
            "detection": rate(sum(1 for nk in omission_notes if r2_flag_rep(nk, rep)),
                              len(omission_notes)),
            "false_alarms": rate(sum(1 for nk in clean_notes if r2_flag_rep(nk, rep)),
                                 len(clean_notes))}
    out["route_two_operating_point_held_out"] = {
        "per_replicate": r2_rates,
        "majority_of_three": {
            "detection": rate(sum(1 for nk in omission_notes if r2_flag_majority(nk)),
                              len(omission_notes)),
            "false_alarms": rate(sum(1 for nk in clean_notes if r2_flag_majority(nk)),
                                 len(clean_notes))}}

    hh = {}
    # (a) level UP: majority against majority, both routes read as three runs
    if len(flags) >= 3:
        hh["majority_vs_majority"] = head_to_head(
            omission_notes, roster, r2_flag_majority, majority)
    # (b) level DOWN: one draw against one draw, replicate index matched
    hh["single_vs_single_by_replicate"] = {
        str(rep): head_to_head(omission_notes, roster,
                               lambda nk, rp=rep: r2_flag_rep(nk, rp),
                               lambda nk, rp=rep: flags[rp].get(nk, False))
        for rep in sorted(flags)}
    # (c) the published asymmetric reading, recomputed here as the anchor
    if 1 in flags:
        hh["published_asymmetric_majority_vs_single"] = head_to_head(
            omission_notes, roster, r2_flag_majority, lambda nk: flags[1].get(nk, False))
    out["head_to_head_held_out"] = hh

    # --------- 4. the evaluation set: equalise for free by reading route two as single draws
    #
    # Route one has ONE replicate on the 607-note evaluation set and this round did not buy
    # more (that would have been another ~$36, past the spend cap). So the published headline
    # is equalised in the only direction that costs nothing: route two read one run at a time.
    b3_eval = read_records(B3_STORES[1], 1)
    if b3_eval:
        eroster = roster_of(b3_eval)
        eom = [nk for nk, m in eroster.items()
               if m["role"] == "errored" and str(m["pair_class"]).startswith("omit")]
        eflag = {r["note_key"]: rule_flag(r) for r in b3_eval}
        r2_eval = route_two_scores(None)

        def er2(nk, rep):
            s = r2_eval.get(nk, {}).get(rep)
            return s is not None and s < EV.FLAG_BELOW

        def er2_majority(nk):
            votes = [1 if (s is not None and s < EV.FLAG_BELOW) else 0
                     for s in r2_eval.get(nk, {}).values()]
            return sum(votes) * 2 > len(votes) if votes else False

        reps = sorted({rep for v in r2_eval.values() for rep in v})
        out["substrate"]["evaluation_set"] = {
            "notes": len(eroster), "omission_notes": len(eom),
            "clean_notes": sum(1 for m in eroster.values() if m["role"] == "clean"),
            "consultations": len({m["consultation"] for m in eroster.values()}),
            "route_one_replicates": 1, "route_two_replicates": len(reps)}
        out["head_to_head_evaluation_set"] = {
            "published_majority_vs_single": head_to_head(
                eom, eroster, er2_majority, lambda nk: eflag.get(nk, False)),
            "single_vs_single_by_route_two_replicate": {
                str(rep): head_to_head(eom, eroster, lambda nk, rp=rep: er2(nk, rp),
                                       lambda nk: eflag.get(nk, False))
                for rep in reps}}
        ecl = [nk for nk, m in eroster.items() if m["role"] == "clean"]
        out["route_two_operating_point_evaluation_set"] = {
            "per_replicate": {
                str(rep): {"detection": rate(sum(1 for nk in eom if er2(nk, rep)), len(eom)),
                           "false_alarms": rate(sum(1 for nk in ecl if er2(nk, rep)), len(ecl))}
                for rep in reps},
            "majority_of_three": {
                "detection": rate(sum(1 for nk in eom if er2_majority(nk)), len(eom)),
                "false_alarms": rate(sum(1 for nk in ecl if er2_majority(nk)), len(ecl))},
            "route_one_single_replicate": {
                "detection": rate(sum(1 for nk in eom if eflag.get(nk)), len(eom)),
                "false_alarms": rate(sum(1 for nk in ecl if eflag.get(nk)), len(ecl))}}

    path = os.path.join(OUT_DIR, "analysis.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=False)
    print(json.dumps(out, indent=2)[:4000])
    print(f"\nwrote {os.path.relpath(path)}")
    if out["missing_replicates"]:
        print("INCOMPLETE - still missing: " + "; ".join(out["missing_replicates"]))


if __name__ == "__main__":
    main()
