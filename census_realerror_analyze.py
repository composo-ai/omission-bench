"""census_realerror_analyze.py - read-out for the real-error arm. No model calls.

Everything P1 measures is on constructed absence. This scores three P1 judge designs against
the census's own panel-verified findings on 261 REAL vendor notes: 87 carrying a tier-1
verified omission, and 174 carrying no verified finding at all, drawn stratified by product
and substrate at seed 20260825 (`build_census_realerror_subset.py`, artifact
`master/census_realerror_261.json`).

Each arm is scored at ITS OWN published rule, not at a rule chosen here:

  monolithic FC-score-k8   flag iff the winsorised mean of 8 samples < 8.0 - the grid's
                           pre-registered threshold (specs/w2-ablation-grid.md, and the
                           `flagged` field the store already carries)
  route two                flag iff the score is BELOW 10 - the paper's deployed rule
                           (FINDINGS 31.2 / w2_evalfull_analyze.FLAG_BELOW). NOTE the store's
                           own `flagged` field uses w2_power's `score <= 7`, which is a
                           different rule and is reported beside it rather than used.
  route one (B3)           flag iff any fact the audit graded critical is verdicted absent -
                           the per-fact critical rule of FINDINGS 19.3 / 31.2

A threshold sweep is reported beside each arm's own rule, so a reader can see whether a rate
is a property of the judge or of where its threshold happens to sit on this corpus.

Detection is the flag rate on the 87 verified-omission notes; the false-alarm rate is the flag
rate on the 174 no-finding notes. These are NOT the benchmark's rates and must never be merged
into its tables: the positive class is defined by the census panel's strict standard, and the
negative class is clean only as far as twelve discovery passes found nothing.

    .venv/bin/python census_realerror_analyze.py   # needs scipy, via w2_analyze
"""
import collections
import json
import os
import re

from common import RESULTS
from w2_analyze import wilson              # the canonical implementation; do not fork it

HERE = os.path.dirname(os.path.abspath(__file__))
SUBSET = os.path.join(HERE, "master", "census_realerror_261.json")
OUT_DIR = os.path.join(RESULTS, "w2-realerror")

ARMS = {
    "monolithic_FC_score_k8": {
        "store": "w2-realerror-k8/_state/census-k8-r1.jsonl",
        "rule": "winsorised mean of 8 samples < 8.0 (the grid's pre-registered threshold)",
        "flag": lambda r: (r.get("aggregate") is not None and r["aggregate"] < 8.0),
        "score": lambda r: r.get("aggregate"),
        "sweep": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.5, 10.0],
    },
    "route_two_budgeted_single_call": {
        "store": "w2-realerror-route2/_state/census-route2-r1.jsonl",
        "rule": "score below 10 (the deployed rule of FINDINGS 31.2)",
        "flag": lambda r: (r.get("aggregate") is not None and r["aggregate"] < 10.0),
        "score": lambda r: r.get("aggregate"),
        "sweep": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    },
    "route_one_two_stage_B3": {
        # Two stores: the arm was judged as two disjoint consultation slices by concurrent
        # processes, because its cold-cache stage 1/1b runs at only min(workers,3) and that,
        # not the note judging, was the bottleneck. Disjoint by construction (--skip-blocks),
        # and the merge asserts no note_key appears twice.
        "store": ["w2-realerror/_state/census-b3r1-B3.jsonl",
                  "w2-realerror-sliceb/_state/census-b3r1b-B3.jsonl",
                  "w2-realerror-slicec/_state/census-b3r1c-B3.jsonl"],
        "rule": "any fact graded critical is verdicted absent (FINDINGS 19.3's per-fact rule)",
        "flag": None,                       # set below; reads the per-fact verdicts
        "score": lambda r: r.get("aggregate"),
        "sweep": None,                      # the rule is a predicate, not a threshold
    },
}

STOP = set("""a an and are as at be been but by for from had has have in into is it its of on
or that the their there these this to was were which who will with without not no had""".split())


def critical_absent(rec):
    """Route one's deployed predicate, and the facts it fired on."""
    hits = [f for f in (rec.get("detail") or {}).get("missing_facts") or []
            if f.get("severity") == "critical" and f.get("verdict") == "absent"]
    return bool(hits), hits


ARMS["route_one_two_stage_B3"]["flag"] = lambda r: critical_absent(r)[0]


def load_store(rel, expect=261):
    """Records keyed by census note_key. `rel` may be one store or a list of disjoint slices.

    Returns (None, reason) for an absent or INCOMPLETE arm, so a run still in flight can never
    reach the read-out as if it were a measurement.
    """
    rels = [rel] if isinstance(rel, str) else list(rel)
    recs, pf = {}, 0
    for r_ in rels:
        path = os.path.join(RESULTS, r_)
        if not os.path.exists(path):
            return None, f"absent ({r_})"
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                    # torn final line from a killed process
            if r.get("parse_failure"):
                pf += 1
            nk = (r.get("note_key") or "")
            nk = nk[:-4] if nk.endswith("|err") else nk
            if nk in recs:
                return None, (f"note {nk} appears in more than one slice of {rels} - the "
                              "slices are meant to be disjoint; refusing to score")
            recs[nk] = r
    if len(recs) != expect:
        return None, (f"INCOMPLETE ({len(recs)}/{expect} notes across {len(rels)} store(s)) "
                      "- run still in flight")
    return {"records": recs, "parse_failures": pf}, None


def rate(k, n):
    w = wilson(k, n)
    return {"k": k, "n": n, "p": round(k / n, 4) if n else None,
            "wilson": [w["lo"], w["hi"]] if w else None}


def tokens(text):
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(t) > 2 and t not in STOP}


def overlap(a, b):
    """Containment of the smaller token set in the larger. Indicative, not adjudication."""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sub = json.load(open(SUBSET))
    notes = {n["note_key"]: n for n in sub["notes"]}
    omission = [k for k, n in notes.items() if n["arm"] == "omission"]
    clean = [k for k, n in notes.items() if n["arm"] == "no_finding"]

    out = {"design": {
        "subset": os.path.relpath(SUBSET, HERE), "seed": sub.get("seed"),
        "n_notes": len(notes), "n_omission_notes": len(omission),
        "n_no_finding_notes": len(clean),
        "consultations": len({n["consultation"] for n in notes.values()}),
        "by_product": dict(collections.Counter(n["product_letter"] for n in notes.values())),
        "positive_class": ("notes carrying at least one tier-1 verified OMISSION under the "
                           "census panel's strict standard"),
        "negative_class": ("notes on which twelve discovery passes and the panel found "
                           "nothing at all")},
        "arms": {}, "unavailable": []}

    # -------------------------------------------------------------- per arm, at its own rule
    for name, cfg in ARMS.items():
        store, err = load_store(cfg["store"])
        if err:
            out["unavailable"].append(f"{name}: {err}")
            continue
        recs = store["records"]
        missing = [k for k in notes if k not in recs]
        if missing:
            out["unavailable"].append(f"{name}: {len(missing)} scored notes have no record")
            continue

        det_k = sum(1 for k in omission if cfg["flag"](recs[k]))
        fa_k = sum(1 for k in clean if cfg["flag"](recs[k]))
        blk = {"rule": cfg["rule"], "parse_failures": store["parse_failures"],
               "detection": rate(det_k, len(omission)),
               "false_alarms": rate(fa_k, len(clean))}

        # per product, so a rate is not quietly a property of one vendor's template
        blk["by_product"] = {}
        for letter in sorted({n["product_letter"] for n in notes.values()}):
            om_p = [k for k in omission if notes[k]["product_letter"] == letter]
            cl_p = [k for k in clean if notes[k]["product_letter"] == letter]
            blk["by_product"][letter] = {
                "detection": rate(sum(1 for k in om_p if cfg["flag"](recs[k])), len(om_p)),
                "false_alarms": rate(sum(1 for k in cl_p if cfg["flag"](recs[k])), len(cl_p))}

        # the sweep: is the operating point a property of the judge or of the threshold?
        if cfg["sweep"]:
            sweep = []
            for t in cfg["sweep"]:
                f = lambda r, th=t: (cfg["score"](r) is not None and cfg["score"](r) < th)
                sweep.append({"threshold_below": t,
                              "detection": rate(sum(1 for k in omission if f(recs[k])),
                                                len(omission)),
                              "false_alarms": rate(sum(1 for k in clean if f(recs[k])),
                                                   len(clean))})
            blk["threshold_sweep"] = sweep

        # Route two's rule works on the benchmark only because clean twins park at exactly 10
        # (FINDINGS 31.3: 90-94% of evaluation-set twins, 96-98% held out). That is a
        # calibration of this corpus and judge, which Section 7.7 says in words and this
        # measures: if real vendor notes do not park at 10, the rule's false-alarm rate has
        # nothing to stand on. Reported for every scored arm, since the same question applies
        # to the monolithic judge's threshold.
        if cfg["sweep"]:
            dist = collections.Counter()
            for k in clean:
                s = cfg["score"](recs[k])
                dist[None if s is None else round(float(s), 2)] += 1
            top = max((v for v in dist if v is not None), default=None)
            blk["clean_note_calibration"] = {
                "score_distribution_on_no_finding_notes":
                    {str(k): v for k, v in sorted(dist.items(), key=lambda x: (x[0] is None,
                                                                              x[0]))},
                "at_top_of_scale": rate(dist.get(top, 0), len(clean)) if top is not None
                else None,
                "top_of_scale_value": top}

        # route two's store carries its own, different flag field - report, do not use
        if name == "route_two_budgeted_single_call":
            blk["store_own_flag_rule"] = {
                "rule": "score <= 7 (w2_power.FLAG_RULE, NOT the paper's deployed rule)",
                "detection": rate(sum(1 for k in omission if recs[k].get("flagged")),
                                  len(omission)),
                "false_alarms": rate(sum(1 for k in clean if recs[k].get("flagged")),
                                     len(clean))}
        out["arms"][name] = blk

    # ------------------------------------------- route one only: does the named fact match?
    #
    # The number no injected benchmark can give. Route one's flag names a fact; the census's
    # verified omission is an independent verbalisation of the same absence. Matching the two
    # is a judgement, so this reports a LEXICAL indicator and writes every pair out for
    # adjudication rather than claiming the lexical number is the answer.
    store, err = load_store(ARMS["route_one_two_stage_B3"]["store"])
    if not err:
        recs = store["records"]
        pairs, per_note = [], []
        for k in omission:
            fired, hits = critical_absent(recs[k])
            if not fired:
                continue
            truth = notes[k]["verified_omissions"]
            best_for_note = 0.0
            for h in hits:
                scored = [(max(overlap(h.get("fact"), t.get("description")),
                               overlap(h.get("fact"), t.get("source_quote"))), t)
                          for t in truth]
                scored.sort(key=lambda x: -x[0])
                top, match = (scored[0] if scored else (0.0, None))
                best_for_note = max(best_for_note, top)
                pairs.append({"note_key": k, "product": notes[k]["product_letter"],
                              "judge_fact": h.get("fact"),
                              "census_finding_id": (match or {}).get("finding_id"),
                              "census_description": (match or {}).get("description"),
                              "census_severity": (match or {}).get("severity"),
                              "lexical_overlap": round(top, 3)})
            per_note.append({"note_key": k, "n_flagged_facts": len(hits),
                             "best_overlap": round(best_for_note, 3)})
        for thr in (0.3, 0.4, 0.5):
            n_match = sum(1 for p in per_note if p["best_overlap"] >= thr)
            out.setdefault("route_one_flag_content", {})[f"lexical_at_{thr}"] = \
                rate(n_match, len(per_note)) if per_note else None
        out.setdefault("route_one_flag_content", {}).update({
            "n_flagged_omission_notes": len(per_note),
            "n_flagged_facts": len(pairs),
            "method": ("containment of the smaller token set in the larger, stopworded; an "
                       "INDICATIVE screen, not adjudication - every pair is written to "
                       "flag_content_pairs.json for a read"),
        })
        with open(os.path.join(OUT_DIR, "flag_content_pairs.json"), "w") as fh:
            json.dump({"note": ("Route one's flagged facts beside the census's verified "
                                "omission they best match lexically. Vendor note text is not "
                                "reproduced; these are the judge's fact wording and the "
                                "panel's finding description."),
                       "pairs": sorted(pairs, key=lambda p: -p["lexical_overlap"])}, fh,
                      indent=1)

    path = os.path.join(OUT_DIR, "analysis.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=2)[:1500])
    for name, blk in out["arms"].items():
        d, f = blk["detection"], blk["false_alarms"]
        print(f"\n{name}\n  rule: {blk['rule']}")
        print(f"  detection   {d['k']:3d}/{d['n']:3d} = {100 * d['p']:5.1f}%  {d['wilson']}")
        print(f"  false alarms{f['k']:3d}/{f['n']:3d} = {100 * f['p']:5.1f}%  {f['wilson']}")
    if out["unavailable"]:
        print("\nUNAVAILABLE: " + "; ".join(out["unavailable"]))
    print(f"\nwrote {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    main()
