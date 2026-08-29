"""Amendment 2026-08-09b gate report v3 - the FULL PriMock instrument trajectory.

One committed artifact reporting the whole story the paper discloses as a single finding:
  v1 (spec 3.1 prompt)                    -> 33.3% drops (19/57)  - gate FAIL
  v2 (part P: 4 ASR-specific prompt fixes)-> 22.8% drops (13/57)  - gate FAIL
  v2.1 (part T1: transcript-only input view - the primock57_parsed.json presenting_complaint
        header is actor case-card metadata, struck from every extraction/critique view)
     + excise rule (part T2: contested must_not_contain-only residuals are deleted, sheet kept)
                                          -> final rate computed here from the run artifacts.

Decomposition: the metadata-defect class (did striking the header resolve those ids?), the
excised items (count + per-item log), the genuine remaining drops with reasons, the fate of
the 8 ids v1 kept but v2's panel dropped (panel churn), and per-sheet size stats per version.

No LLM calls - pure file analysis over committed artifacts. Rerunnable any time.
Per the amendment's reporting commitment the final number is reported WHATEVER it is; no
further prompt iteration regardless (hard stop without the lead author).

Usage:  python primock_trajectory.py
Output: master/primock_instrument_trajectory.json
"""
import json, os, re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
M = lambda *p: os.path.join(HERE, "master", *p)
N = 57


def parse_report(path):
    """Per-id {status, residuals} from a critique report (same pattern as compare_primock_v1_v2)."""
    txt = open(path).read()
    out = {}
    for rid, status, body in re.findall(r"^## (day\S+) - (\S+)\n(.*?)(?=^## |\Z)", txt, re.M | re.S):
        resid = [" ".join(r.split()) for r in re.findall(
            r"\[residual\] (.*?)(?=\n    - \[residual\]|\n  [a-z]|\n\Z)", body, re.S)]
        out[rid] = {"status": status, "residuals": resid}
    return out


def size_stats(rows):
    n = len(rows)
    if not n:
        return {}
    mc = [len(r["fact_sheet"]["must_contain"]) for r in rows]
    mnc = [len(r["fact_sheet"]["must_not_contain"]) for r in rows]
    tr = [len(r["fact_sheet"]["salience_traps"]) for r in rows]
    imp = Counter(t.get("importance") for r in rows for t in r["fact_sheet"]["salience_traps"])
    return {"n_sheets": n,
            "mean_must_contain": round(sum(mc) / n, 1),
            "mean_must_not_contain": round(sum(mnc) / n, 1),
            "mean_salience_traps": round(sum(tr) / n, 1),
            "importance_counts": dict(imp)}


def main():
    v1 = parse_report(M("critique_extracted_primock_v1.md"))
    v2 = parse_report(M("critique_extracted_primock_v2.md"))
    v21 = parse_report(M("critique_extracted_primock.md"))
    state = json.load(open(M("critique_state_primock.json")))
    comparison = json.load(open(M("primock_v1_v2_comparison.json")))
    excise_log = json.load(open(M("excised_mnc_items.json")))
    assert set(v1) == set(v2) == set(v21) == set(state) and len(v1) == N

    kept_files = {"v1": json.load(open(M("fact_sheets_primock_v1.json"))),
                  "v2": json.load(open(M("fact_sheets_primock_v2.json"))),
                  "v2.1": json.load(open(M("fact_sheets_primock.json")))}
    raw_files = {"v1": json.load(open(M("fact_sheets_raw_primock_v1.json"))),
                 "v2": json.load(open(M("fact_sheets_raw_primock_v2.json"))),
                 "v2.1": json.load(open(M("fact_sheets_raw_primock.json")))}
    assert all(r["extraction_provenance"].get("instrument") == "v2.1"
               and "presenting_complaint" not in r for r in raw_files["v2.1"])

    drops = {v: sorted(r for r, e in st.items() if e["status"].startswith("dropped"))
             for v, st in (("v1", v1), ("v2", v2), ("v2.1", v21))}
    excised_ids = sorted(r for r, e in v21.items() if e["status"] == "revised-kept-excised")
    pre_excise_drops = sorted(drops["v2.1"] + excised_ids)   # what v2.1 drops WITHOUT part T2
    entries = [x for x in excise_log["entries"] if x["source"] == "primock"]

    # --- metadata-defect class (part T1's target): the ids v2 dropped wholly or partly on the
    # presenting_complaint header, as classified in the committed part-P gate report.
    meta_only = comparison["v2_drops"]["metadata_only_ids"]
    meta_partial = [r for r, d in comparison["v2_drops"]["detail"].items()
                    if d["cause"] == "metadata_plus_genuine"]
    meta_class = {rid: {"v2_cause": comparison["v2_drops"]["detail"][rid]["cause"],
                        "v21_status": v21[rid]["status"],
                        "resolved": not v21[rid]["status"].startswith("dropped")}
                  for rid in meta_only + meta_partial}

    # --- the 8 panel-churn ids (v1 kept, v2 dropped) under v2.1
    churn = comparison["v2_drops"]["newly_dropped_ids_kept_under_v1"]

    def fate(rid):
        s = v21[rid]["status"]
        return {"clean": "rescued-clean", "minor-only": "rescued-clean",
                "revised-kept": "rescued-after-revision",
                "revised-kept-excised": "rescued-by-excise"}.get(s, "dropped-again")

    churn_detail = {rid: {"v1": v1[rid]["status"], "v2": v2[rid]["status"],
                          "v2.1": v21[rid]["status"], "fate": fate(rid),
                          "excised_items": [x["assertion"] for x in entries if x["id"] == rid]}
                    for rid in churn}

    # --- remaining genuine drops, with the excise classifier's reading where it ran
    drop_detail = {}
    for rid in drops["v2.1"]:
        e = state[rid]
        ex = e.get("excise", {})
        drop_detail[rid] = {
            "status": v21[rid]["status"],
            "v1_status": v1[rid]["status"], "v2_status": v2[rid]["status"],
            "residuals": v21[rid]["residuals"],
            "excise_classifier_verdicts": sorted({c["verdict"] for c in
                                                  ex.get("classifications", [])}) or None,
            "why_not_excisable": ("revise never produced a sheet (transport-exhausted) - "
                                  "nothing to excise from"
                                  if v21[rid]["status"] == "dropped-revise-failed" else
                                  "at least one residual touches must_contain or salience_traps"
                                  if ex.get("decision") == "drop" else None)}

    rate = lambda k: round(len(k) / N, 3)
    report = {
        "spec": "w-d-master-dataset.md Amendment 2026-08-09b (parts T1+T2) - gate report v3, "
                "full instrument trajectory",
        "generated_by": "primock_trajectory.py",
        "reporting_commitment": "the final rate is reported whatever it is; no further prompt "
                                "iteration regardless (hard stop without the lead author). The gate's "
                                "purpose - no silent retuning - is preserved by disclosure.",
        "instruments": {
            "v1": "extraction prompt v1 (spec 3.1 + Amendment 2026-07-29b H), full record view",
            "v2": "extraction prompt v2 (part P: ASR-uncertainty, equivalence, note-bearing-only, "
                  "transcript-only-traps fixes + part R rubric-anchored importance), full record "
                  "view - the presenting_complaint header still reached critic/revise views",
            "v2.1": "same v2 prompt text, transcript-only input view (part T1: the "
                    "primock57_parsed.json presenting_complaint header is the actor's case-card "
                    "prompt, not ground truth - struck from the extraction record and every "
                    "critic/revise view) - a data correction, not a prompt iteration",
            "excise_rule": "part T2, applied on top of v2.1: after the one revision cycle, a "
                           "sheet whose ONLY residual material issues are contested "
                           "must_not_contain items has those items deleted (logged) and is KEPT; "
                           "residuals touching must_contain or salience_traps still drop it",
            "critic_panel": "identical all three runs: support/materiality/leakage, "
                            "claude-opus-5, plan path, one revision cycle, re-audit = support "
                            "critic"},
        "gate2_trajectory": {
            "rule": "drop rate (still-material after one revision) < 10% per source",
            "v1": {"dropped": len(drops["v1"]), "of": N, "drop_rate": rate(drops["v1"]),
                   "verdict": "FAIL"},
            "v2": {"dropped": len(drops["v2"]), "of": N, "drop_rate": rate(drops["v2"]),
                   "verdict": "FAIL",
                   "excluding_metadata_only_drops": comparison["gate2"][
                       "v2_drop_rate_excluding_metadata_only_drops"]},
            "v2.1_pre_excise": {"dropped": len(pre_excise_drops), "of": N,
                                "drop_rate": rate(pre_excise_drops),
                                "ids": pre_excise_drops,
                                "note": "v2.1 input-view correction alone, before part T2"},
            "v2.1_plus_excise_FINAL": {
                "dropped": len(drops["v2.1"]), "of": N, "drop_rate": rate(drops["v2.1"]),
                "excised_sheets": len(excised_ids), "excised_items": len(entries),
                "verdict": "PASS" if len(drops["v2.1"]) / N < 0.10 else "FAIL"}},
        "kept": {v: {"n": len(kept_files[v]),
                     "statuses": dict(Counter(e["status"] for e in st.values()))}
                 for v, st in (("v1", v1), ("v2", v2), ("v2.1", v21))},
        "metadata_defect_class": {
            "definition": "v2 drops caused wholly (metadata_only) or partly "
                          "(metadata_plus_genuine) by the presenting_complaint header "
                          "contradicting the transcript",
            "ids": meta_class,
            "all_resolved": all(d["resolved"] for d in meta_class.values())},
        "excisions": {
            "n_items": len(entries), "n_sheets": len(excised_ids),
            "sheets": excised_ids,
            "log_file": "master/excised_mnc_items.json",
            "items": [{"id": x["id"], "mnc_index": x["mnc_index"],
                       "assertion": x["assertion"], "critic_reason": x["critic_reason"]}
                      for x in entries]},
        "v2.1_drops": {"ids": drops["v2.1"], "detail": drop_detail},
        "v2_churn_ids_under_v2.1": {
            "definition": "the 8 ids v1 kept but v2's panel dropped (panel churn, mostly "
                          "contested-must_not_contain residuals) - their fate under v2.1+excise",
            "counts": dict(Counter(d["fate"] for d in churn_detail.values())),
            "detail": churn_detail},
        "size_stats": {
            "raw_all_sheets": {v: size_stats(raw_files[v]) for v in ("v1", "v2", "v2.1")},
            "kept_sheets": {v: size_stats(kept_files[v]) for v in ("v1", "v2", "v2.1")}},
        "per_id_status_map": {rid: {"v1": v1[rid]["status"], "v2": v2[rid]["status"],
                                    "v2.1": v21[rid]["status"]} for rid in sorted(v1)},
    }
    out = M("primock_instrument_trajectory.json")
    json.dump(report, open(out, "w"), ensure_ascii=False, indent=1)
    g = report["gate2_trajectory"]
    print(f"v1 {g['v1']['drop_rate']:.1%} -> v2 {g['v2']['drop_rate']:.1%} -> "
          f"v2.1 pre-excise {g['v2.1_pre_excise']['drop_rate']:.1%} -> "
          f"v2.1+excise {g['v2.1_plus_excise_FINAL']['drop_rate']:.1%} "
          f"({g['v2.1_plus_excise_FINAL']['verdict']})")
    print(f"metadata class resolved: {report['metadata_defect_class']['all_resolved']} | "
          f"excised {len(entries)} items / {len(excised_ids)} sheets | "
          f"remaining drops: {drops['v2.1']}")
    print(f"-> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
