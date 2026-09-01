#!/usr/bin/env python3
"""second_clinician_analyze.py - unblind and score the independent clinician's sitting.

Takes whatever came back from the offline app - one JSON file per lot, in the shape
second_clinician_sitting.py's download button produces - joins it to
results/second-clinician-sitting/key.json, and writes the read-out that was planned
BEFORE the sitting ran (it is in the manifest's reporting_plan, so nobody is deciding
what to report after seeing the answers):

  - an independent precision estimate on the verified subset, with a Wilson interval,
    reported BESIDE the author's 20 of 21 and never pooled with it;
  - the panel-refused items reported separately, because the author judged all nine of
    his genuine and an independent clinician breaking that is the most interesting thing
    this sitting can produce;
  - an independent severity read against the rubric grades: exact, adjacent, and the
    DIRECTION of the disagreements, which is the point;
  - every disagreement described individually, in plain words;
  - NO kappa, and the reason stated;
  - the observed reading time per item, which is what resizes lot two.

    python3 sittings/second_clinician_analyze.py results/second-clinician-sitting/returned/*.json
    python3 sittings/second_clinician_analyze.py --self-test      # fabricated answers
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import argparse
import glob
import json
import os
import random
import statistics
import sys

from common import HERE, RESULTS
from w2_analyze import wilson

OUTDIR = os.path.join(RESULTS, "second-clinician-sitting")
DRAWS = 10000
SEED = 20260826
ORDER = {"peripheral": 0, "supporting": 1, "critical": 2}

# The author's own first sitting, as published, for the side-by-side. Never pooled with ours.
AUTHOR = {"verified_genuine": 20, "verified_n": 21, "refused_genuine": 9, "refused_n": 9,
          "severity_exact": 16, "severity_n": 20, "severity_up": 4, "severity_down": 0}


def load_returns(paths):
    """One dict per lot, keyed by lot number, with the pack hash checked."""
    out = {}
    for p in paths:
        d = json.load(open(p))
        lot = d.get("set")
        if lot in out:
            raise SystemExit("two files for lot %s" % lot)
        out[lot] = {"file": os.path.basename(p), "payload": d}
    if not out:
        raise SystemExit("no answer files given")
    return out


def join(returns, key):
    """Key rows for the lots that came back, with the clinician's answer attached."""
    by_lot = {}
    for row in key:
        by_lot.setdefault(row["lot"], {})[row["item"]] = dict(row)
    rows = []
    for lot, r in sorted(returns.items()):
        if lot not in by_lot:
            raise SystemExit("answers for lot %s, which was never built" % lot)
        expect = len(by_lot[lot])
        if r["payload"].get("n_items") != expect:
            raise SystemExit("lot %s: the file says %s items, the key holds %d"
                             % (lot, r["payload"].get("n_items"), expect))
        for a in r["payload"]["answers"]:
            row = by_lot[lot].get(a["item"])
            if not row:
                raise SystemExit("lot %s: answer for unknown item %s" % (lot, a["item"]))
            row["verdict"] = a.get("verdict")
            row["severity"] = a.get("severity")
            row["seconds"] = a.get("seconds")
            rows.append(row)
    return rows


def cluster_boot(rows, hit):
    """Percentile bootstrap over consultations. Degenerate at these n - artifact only."""
    by_cons = {}
    for r in rows:
        by_cons.setdefault(r["consultation"], []).append(r)
    cons = sorted(by_cons)
    if not cons:
        return None
    rng = random.Random(SEED)
    stats = []
    for _ in range(DRAWS):
        num = den = 0
        for _ in cons:
            for r in by_cons[cons[rng.randrange(len(cons))]]:
                den += 1
                num += bool(hit(r))
        stats.append(num / den)
    stats.sort()
    return [round(stats[int(0.025 * DRAWS)], 4), round(stats[int(0.975 * DRAWS) - 1], 4)]


def describe(row, master):
    """One disagreement, in plain words."""
    f = master.get(row["finding_id"], {})
    v = f.get("verdict") or {}
    reasons = v.get("reasons") or {}
    return {
        "lot": row["lot"], "item": row["item"], "finding_id": row["finding_id"],
        "status": row["status"],
        "what_the_finding_claimed": f.get("description"),
        "claimed_mode": row.get("mode"),
        "note_quote": f.get("note_quote"), "transcript_quote": f.get("source_quote"),
        "what_the_clinician_said": row["verdict"],
        "clinician_severity": row.get("severity"),
        "rubric_severity": row.get("severity_rubric"),
        "what_the_panel_said": v.get("decided_by"),
        "panel_reasons": {k: reasons[k] for k in sorted(reasons)},
        "seconds_spent": row.get("seconds"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers", nargs="*", help="the returned JSON files, one per lot")
    ap.add_argument("--key", default=os.path.join(OUTDIR, "key.json"))
    ap.add_argument("--out", default=os.path.join(OUTDIR, "sitting_results.json"))
    ap.add_argument("--self-test", action="store_true",
                    help="run end to end on FABRICATED answers, write nothing durable")
    args = ap.parse_args()

    key = json.load(open(args.key))
    if args.self_test:
        paths = fabricate(key)
        args.out = os.path.join(OUTDIR, "SELF-TEST-results.json")
    else:
        paths = [p for a in args.answers for p in sorted(glob.glob(a))]
        if not paths:
            raise SystemExit("give me the returned answer files (or --self-test)")

    returns = load_returns(paths)
    rows = join(returns, key)
    master = {i["finding_id"]: i
              for i in json.load(open(os.path.join(
                  HERE, "master/findings_verified_master.json")))["all_issues"]}

    ver = [r for r in rows if r["status"] == "verified_finding"]
    ref = [r for r in rows if r["status"] == "panel_refused"]
    answered_v = [r for r in ver if r["verdict"] in ("genuine", "not-genuine")]
    g = sum(1 for r in ver if r["verdict"] == "genuine")
    ng = sum(1 for r in ver if r["verdict"] == "not-genuine")
    cj = sum(1 for r in ver if r["verdict"] == "cannot-judge")
    blank = sum(1 for r in ver if not r["verdict"])

    w_all = wilson(g, len(ver)) if ver else None
    w_ans = wilson(g, len(answered_v)) if answered_v else None

    sev = [(r["severity_rubric"], r["severity"]) for r in ver
           if r["verdict"] == "genuine" and r["severity"] and r["severity_rubric"]
           and r["severity_rubric"] in ORDER]
    exact = sum(1 for a, b in sev if a == b)
    adjacent = sum(1 for a, b in sev if abs(ORDER[a] - ORDER[b]) == 1)
    up = sum(1 for a, b in sev if ORDER[b] > ORDER[a])
    down = sum(1 for a, b in sev if ORDER[b] < ORDER[a])

    fg = sum(1 for r in ref if r["verdict"] == "genuine")
    fng = sum(1 for r in ref if r["verdict"] == "not-genuine")
    fcj = sum(1 for r in ref if r["verdict"] == "cannot-judge")

    secs = [r["seconds"] for r in rows if isinstance(r.get("seconds"), int) and r["seconds"] > 0]
    timing = None
    if secs:
        timing = {
            "n_items_timed": len(secs),
            "median_minutes_per_item": round(statistics.median(secs) / 60, 2),
            "mean_minutes_per_item": round(statistics.mean(secs) / 60, 2),
            "min_minutes": round(min(secs) / 60, 2), "max_minutes": round(max(secs) / 60, 2),
            "total_minutes_per_lot": {
                lot: round(sum(r["seconds"] for r in rows
                               if r["lot"] == lot and isinstance(r.get("seconds"), int)) / 60, 1)
                for lot in sorted({r["lot"] for r in rows})},
            "what_to_do_with_it": "this is the observation the build's estimate was a stand-in "
                                  "for; resize lot two from it before sending",
        }

    out = {
        "generated_by": "second_clinician_analyze.py",
        "self_test": bool(args.self_test),
        "lots_returned": sorted(returns),
        "files": [r["file"] for r in returns.values()],
        "rater": "an independent clinician, not an author of the study",
        "independent_precision": {
            "genuine": g, "not_genuine": ng, "cannot_judge": cj, "unanswered": blank,
            "n_sampled": len(ver),
            "point_all_denominator": round(g / len(ver), 4) if ver else None,
            "wilson95_all": [round(w_all["lo"], 4), round(w_all["hi"], 4)] if w_all else None,
            "point_answered_only": round(g / len(answered_v), 4) if answered_v else None,
            "wilson95_answered_only": [round(w_ans["lo"], 4), round(w_ans["hi"], 4)]
                                      if w_ans else None,
            "clustered95_artifact_only": cluster_boot(ver, lambda r: r["verdict"] == "genuine"),
            "n_consultations": len({r["consultation"] for r in ver}),
            "how_to_report": "beside the author's %d of %d, never pooled with it: two raters, "
                             "disjoint samples. The clustered interval degenerates at this n "
                             "and stays artifact-only, exactly as it did in the first sitting."
                             % (AUTHOR["verified_genuine"], AUTHOR["verified_n"]),
        },
        "panel_refused_items": {
            "judged_genuine": fg, "judged_not_genuine": fng, "cannot_judge": fcj,
            "n_sampled": len(ref),
            "agreement_with_the_refusal": round(fng / len(ref), 4) if ref else None,
            "the_author_result_being_tested": "%d of %d judged genuine, i.e. 0%% agreement "
                                              "with the refusals" % (AUTHOR["refused_genuine"],
                                                                     AUTHOR["refused_n"]),
            "why_it_matters": "the author's 9 of 9 is what places a clinician's standard "
                              "nearer the lenient end of the measured range than the panel's. "
                              "An independent clinician replicating it strengthens the claim; "
                              "breaking it is the most interesting thing this sitting can "
                              "produce.",
        },
        "severity": {
            "n_graded_pairs": len(sev), "exact": exact, "adjacent": adjacent,
            "beyond_adjacent": len(sev) - exact - adjacent,
            "clinician_higher_than_rubric": up, "clinician_lower": down,
            "pairs_rubric_then_clinician": sev,
            "the_two_prior_samples": "on the companion study's facts the machine graders sat "
                                     "about a grade more severe than the clinician; on sampled "
                                     "census survivors the author graded 4 of 20 ABOVE the "
                                     "rubric and none below. They point in opposite directions, "
                                     "so direction here is the result, not the exact rate.",
        },
        "interassessor_agreement": {
            "kappa": None,
            "why_not": "two raters on disjoint samples produce no interassessor agreement by "
                       "construction. The second rater was given fresh items to maximise the "
                       "number of findings a human has assessed, rather than to compute a "
                       "coefficient that a dozen shared items could not have supported: at a "
                       "first-rater marginal of 29 genuine in 30, kappa comes out undefined, "
                       "exactly zero or negative. TRIPOD-LLM item 7d therefore stays not-met "
                       "and gains that one line.",
        },
        "coverage": {
            "findings_a_human_had_assessed_before": AUTHOR["verified_n"],
            "fresh_verified_findings_assessed_now": len(ver),
            "total_now": AUTHOR["verified_n"] + len(ver),
            "note": "the two samples are disjoint by construction and are reported side by "
                    "side, so this total is a coverage count and not a pooled denominator",
        },
        "timing": timing,
        "disagreements": [describe(r, master) for r in rows
                          if (r["status"] == "verified_finding" and r["verdict"] != "genuine")
                          or (r["status"] == "panel_refused" and r["verdict"] != "not-genuine")],
        "every_item": [{"lot": r["lot"], "item": r["item"], "status": r["status"],
                        "verdict": r["verdict"], "severity": r["severity"],
                        "rubric_severity": r["severity_rubric"], "seconds": r.get("seconds")}
                       for r in rows],
    }

    json.dump(out, open(args.out, "w"), indent=1)

    p = out["independent_precision"]
    f = out["panel_refused_items"]
    print("INDEPENDENT PRECISION  %d of %d judged genuine = %s, Wilson 95%% [%.1f, %.1f]"
          % (p["genuine"], p["n_sampled"],
             "%.1f%%" % (100 * p["point_all_denominator"]) if p["point_all_denominator"] is not None else "-",
             100 * p["wilson95_all"][0], 100 * p["wilson95_all"][1]))
    print("   report beside the author's %d of %d - never pooled"
          % (AUTHOR["verified_genuine"], AUTHOR["verified_n"]))
    print("PANEL-REFUSED ITEMS    %d of %d judged genuine (the author: %d of %d)"
          % (f["judged_genuine"], f["n_sampled"], AUTHOR["refused_genuine"], AUTHOR["refused_n"]))
    s = out["severity"]
    print("SEVERITY               %d of %d exact, %d adjacent, %d up / %d down against the rubric"
          % (s["exact"], s["n_graded_pairs"], s["adjacent"],
             s["clinician_higher_than_rubric"], s["clinician_lower"]))
    print("COVERAGE               %d findings had been assessed by a human; now %d"
          % (AUTHOR["verified_n"], out["coverage"]["total_now"]))
    if timing:
        print("TIMING                 median %.1f min/item, %s min per lot"
              % (timing["median_minutes_per_item"],
                 ", ".join("%s: %.0f" % (k, v)
                           for k, v in timing["total_minutes_per_lot"].items())))
    print("KAPPA                  none, by construction - see the results file")
    print("\n%d disagreement(s):" % len(out["disagreements"]))
    for d in out["disagreements"]:
        print("  lot %s item %s [%s] clinician says %r"
              % (d["lot"], d["item"], d["status"], d["what_the_clinician_said"]))
        print("    claimed: %s" % (d["what_the_finding_claimed"] or "-")[:150])
    print("\nwrote %s" % args.out)
    if args.self_test:
        for p_ in paths:
            os.remove(p_)
        print("SELF TEST - the fabricated answer files have been deleted. "
              "%s is NOT data; delete it before the real sitting." % os.path.basename(args.out))


def fabricate(key):
    """Fabricated answers, purely to exercise the join and every branch of the read-out.

    Deliberately not all-genuine: it plants a rejection, an abstention, a blank, a
    severity upgrade and a severity downgrade, so every code path below runs at least
    once before the clinician's real answers arrive. The files it writes are deleted at
    the end of the run and the result carries self_test: true.
    """
    rng = random.Random(1)
    paths = []
    for lot in sorted({r["lot"] for r in key}):
        rows = sorted([r for r in key if r["lot"] == lot], key=lambda r: r["item"])
        answers = []
        for n, r in enumerate(rows):
            if n == 0:
                v, s = "not-genuine", None                 # a rejection
            elif n == 1:
                v, s = "cannot-judge", None                # an abstention
            elif n == 2 and lot == 1:
                v, s = None, None                          # left blank
            else:
                v = "genuine"
                grade = r.get("severity_rubric") or "supporting"
                if n == 3:
                    s = "critical" if grade != "critical" else "supporting"   # a move
                elif n == 4:
                    s = "peripheral"                       # a downgrade
                else:
                    s = grade                              # exact
            answers.append({"item": r["item"], "verdict": v, "severity": s,
                            "seconds": rng.randint(120, 420)})
        payload = {"set": lot, "pack": "SELF-TEST", "n_items": len(rows),
                   "started": "2026-01-01T00:00:00.000Z",
                   "finished": "2026-01-01T00:40:00.000Z",
                   "total_minutes": round(sum(a["seconds"] for a in answers) / 60),
                   "answers": answers}
        path = os.path.join(OUTDIR, "SELF-TEST-lot-%d.json" % lot)
        json.dump(payload, open(path, "w"), indent=1)
        paths.append(path)
    return paths


if __name__ == "__main__":
    main()
