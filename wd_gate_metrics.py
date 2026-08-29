"""W-D steps 4-5 roll-up: the spec section 6 gate 1 + gate 2 scoreboard, per source.

Reads the per-source critique state and the kept/raw fact sheets and emits one consolidated
record: critic flags by type and severity, revisions, drops (with reasons), gate 2 status, and the
fact-sheet shape stats (must_contain / must_not_contain / trap counts, importance distribution).

Gate 1 (human evidence-quote audit, >=95% of audited must_contain items supported) cannot be
scored here - it is the lead author's audit of master/human_audit_sample.json - so it is reported as PENDING
with the sample it is waiting on. Gate 2 (<10% drops per source) is scored.

Usage: python wd_gate_metrics.py
Output: master/wd_gate_metrics.json
"""
import json, math, os, statistics
from collections import Counter
from common import HERE

SOURCES = {"primock": 57, "aci": 48, "trapblind": 10, "authored_extracted": 30}
IMPORTANCE = ("critical", "supporting", "peripheral")
LOAD_BEARING = ("high", "medium")
OUT = os.path.join(HERE, "master", "wd_gate_metrics.json")


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(p, 4), round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def sheet_stats(rows):
    if not rows:
        return None
    mc = [len(r["fact_sheet"]["must_contain"]) for r in rows]
    mnc = [len(r["fact_sheet"]["must_not_contain"]) for r in rows]
    tr = [len(r["fact_sheet"]["salience_traps"]) for r in rows]
    imp = Counter(t.get("importance") for r in rows for t in r["fact_sheet"]["salience_traps"])
    lb = Counter(i.get("load_bearing") for r in rows for i in r["fact_sheet"]["must_contain"])
    n_tr, n_mc = sum(tr) or 1, sum(mc) or 1

    def m(v):
        return {"mean": round(statistics.mean(v), 2),
                "sd": round(statistics.stdev(v), 2) if len(v) > 1 else None,
                "min": min(v), "max": max(v), "total": sum(v)}
    return {"n_sheets": len(rows), "must_contain": m(mc), "must_not_contain": m(mnc),
            "salience_traps": m(tr),
            "importance": {k: {"n": imp.get(k, 0), "pct": round(100 * imp.get(k, 0) / n_tr, 1)}
                           for k in IMPORTANCE},
            "load_bearing": {k: {"n": lb.get(k, 0), "pct": round(100 * lb.get(k, 0) / n_mc, 1)}
                             for k in LOAD_BEARING}}


def main():
    audit = json.load(open(os.path.join(HERE, "master", "human_audit_sample.json")))
    rep = {"spec": "specs/w-d-master-dataset.md section 6 gates 1-2, steps 4-5",
           "extraction_layer": {"model": "claude-opus-5", "route": "plan (claude -p)",
                                "effort": "medium", "workers": 12,
                                "why_one_transport": "extraction produces the ground-truth "
                                "annotation artifact; keeping the whole layer on `claude -p` (the "
                                "transport every other artifact in this corpus came through) means "
                                "WD-R4 measures authored-vs-extracted, not transport-vs-transport"},
           "critique_layer": {"model": "claude-opus-5", "route": "plan (claude -p)",
                              "effort": "medium (revise: high)",
                              "panel": ["support", "materiality", "leakage"]},
           "gate1": {"status": "PENDING - human audit (the lead author)", "rule": audit["gate"],
                     "sample": "master/human_audit_sample.json",
                     "items_to_check_pre_registered_15":
                         audit["total_must_contain_items_pre_registered_15"]},
           "sources": {}}

    for src, expect in SOURCES.items():
        st = json.load(open(os.path.join(HERE, "master", f"critique_state_{src}.json")))
        raw = json.load(open(os.path.join(HERE, "master", f"fact_sheets_raw_{src}.json")))
        kept = json.load(open(os.path.join(HERE, "master", f"fact_sheets_{src}.json")))
        statuses = Counter(v.get("status", "unscored") for v in st.values())

        flags, material, verdicts = Counter(), Counter(), Counter()
        for v in st.values():
            for c, r in v.get("critics", {}).items():
                verdicts[f"{c}:{r.get('verdict')}"] += 1
                for i in r.get("issues", []):
                    flags[c] += 1
                    if i.get("severity") == "material":
                        material[c] += 1
        dropped = {rid: {"status": v["status"],
                         "residual_material": [i.get("what") for i in
                                               v.get("recheck", {}).get("issues", [])
                                               if i.get("severity") == "material"]}
                   for rid, v in st.items() if str(v.get("status", "")).startswith("dropped")}
        incomplete = statuses.get("critic-error", 0) + statuses.get("revise-pending", 0)
        n_scored = len(raw) - incomplete
        drop_rate = len(dropped) / n_scored if n_scored else 0.0

        rep["sources"][src] = {
            "expected_consultations": expect, "extracted": len(raw),
            "extraction_complete": len(raw) == expect,
            "scored": n_scored, "incomplete_excluded_from_scoring": incomplete,
            "statuses": dict(statuses),
            "critic_flags_total_by_type": dict(flags),
            "critic_flags_material_by_type": dict(material),
            "critic_verdicts": dict(verdicts),
            "consults_with_>=1_material_flag": sum(1 for v in st.values() if v.get("material")),
            "revisions_run": sum(1 for v in st.values() if v.get("revise", {}).get("ok")),
            "revisions_with_transcript_drift_restored":
                sum(1 for v in st.values() if v.get("revise", {}).get("transcript_drift")),
            "re_audits_run": sum(1 for v in st.values() if v.get("recheck")),
            "kept": len(kept), "dropped": len(dropped),
            "drop_rate": round(drop_rate, 4),
            "drop_rate_wilson_95ci": wilson(len(dropped), n_scored),
            "gate2_rule": "<10% drops per source",
            "gate2": "PASS" if drop_rate < 0.10 else "FAIL",
            "gate2_action_if_fail": "STOP this source and flag for human review; do NOT retune the "
                                    "extraction prompt (pre-registration principle, Amendment "
                                    "2026-07-30 part O.2)",
            "dropped_detail": dropped,
            "fact_sheet_stats_kept": sheet_stats(kept),
            "fact_sheet_stats_raw": sheet_stats(raw)}

    tot_raw = sum(v["extracted"] for v in rep["sources"].values())
    tot_kept = sum(v["kept"] for v in rep["sources"].values())
    rep["totals"] = {"extracted": tot_raw, "kept": tot_kept,
                     "dropped": sum(v["dropped"] for v in rep["sources"].values()),
                     "sources_passing_gate2": [s for s, v in rep["sources"].items()
                                               if v["gate2"] == "PASS"],
                     "sources_failing_gate2": [s for s, v in rep["sources"].items()
                                               if v["gate2"] == "FAIL"]}
    json.dump(rep, open(OUT, "w"), indent=1)

    print(f"{'source':20} {'extr':>5} {'scored':>6} {'kept':>5} {'drop':>5} {'rate':>6}  gate2   "
          f"flags(total/material) support|materiality|leakage")
    for s, v in rep["sources"].items():
        f, m = v["critic_flags_total_by_type"], v["critic_flags_material_by_type"]
        print(f"{s:20} {v['extracted']:5d} {v['scored']:6d} {v['kept']:5d} {v['dropped']:5d} "
              f"{v['drop_rate']:6.1%}  {v['gate2']:6} "
              + " | ".join(f"{f.get(c,0)}/{m.get(c,0)}" for c in ('support','materiality','leakage')))
    print(f"\ntotals: extracted {tot_raw} | kept {tot_kept} | "
          f"gate2 PASS {rep['totals']['sources_passing_gate2']} | "
          f"FAIL {rep['totals']['sources_failing_gate2']}")
    print(f"gate1: {rep['gate1']['status']} ({rep['gate1']['items_to_check_pre_registered_15']} "
          f"must_contain items in the pre-registered 15-sheet draw)")
    print(f"-> {os.path.relpath(OUT, HERE)}")


if __name__ == "__main__":
    main()
