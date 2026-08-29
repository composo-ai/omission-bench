"""Amendment 2026-08-09 part P gate report - PriMock extraction v1 vs v2.

Compares the archived v1 stratum (master/*_primock_v1.*) against the canonical v2 re-run:
drop rates, per-sheet size stats (raw and kept), fate of every v1-dropped id under v2,
flags-by-critic, and a residual-cause classification that separates drops the extraction
prompt can reach from drops caused by the source-data `presenting_complaint` header
contradicting the transcript (a primock57_parsed.json artifact: the field is carried from
the source file, shown to the critic panel, and untouchable by the revise loop, which only
ever rewrites the fact_sheet).

No LLM calls - pure file analysis. Rerunnable any time; inputs are all committed artifacts.

Usage:  python compare_primock_v1_v2.py
Output: master/primock_v1_v2_comparison.json
"""
import json, os, re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
M = lambda *p: os.path.join(HERE, "master", *p)

META_RE = re.compile(r"presenting_complaint|[Mm]etadata|header field|record's own header")


def parse_report(path):
    """Per-id {status, residuals} + the summary flags-by-critic line from a critique report."""
    txt = open(path).read()
    out = {}
    for rid, status, body in re.findall(r"^## (day\S+) - (\S+)\n(.*?)(?=^## |\Z)", txt, re.M | re.S):
        resid = [" ".join(r.split()) for r in re.findall(
            r"\[residual\] (.*?)(?=\n    - \[residual\]|\n  [a-z]|\n\Z)", body, re.S)]
        out[rid] = {"status": status, "residuals": resid}
    flags = {c: {"total": int(t), "material": int(m)} for c, t, m in
             re.findall(r"(support|materiality|leakage) (\d+)/(\d+)", txt)}
    return out, flags


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


def residual_cause(residuals):
    """metadata_only | metadata_plus_genuine | genuine - is the drop reachable by a prompt?"""
    if not residuals:
        return "none"
    meta = [bool(META_RE.search(r)) for r in residuals]
    if all(meta):
        return "metadata_only"
    if any(meta):
        return "metadata_plus_genuine"
    return "genuine"


def main():
    v1_state, v1_flags = parse_report(M("critique_extracted_primock_v1.md"))
    v2_state, v2_flags = parse_report(M("critique_extracted_primock.md"))
    assert set(v1_state) == set(v2_state) and len(v1_state) == 57

    v1_kept = json.load(open(M("fact_sheets_primock_v1.json")))
    v2_kept = json.load(open(M("fact_sheets_primock.json")))
    v1_raw = json.load(open(M("fact_sheets_raw_primock_v1.json")))
    v2_raw = json.load(open(M("fact_sheets_raw_primock.json")))
    assert all(r.get("extraction_provenance", {}).get("prompt_version") == "v2" for r in v2_raw)

    v1_drops = sorted(r for r, e in v1_state.items() if e["status"].startswith("dropped"))
    v2_drops = sorted(r for r, e in v2_state.items() if e["status"].startswith("dropped"))

    def fate(rid):
        s = v2_state[rid]["status"]
        if s in ("clean", "minor-only"):
            return "rescued-clean"
        if s == "revised-kept":
            return "rescued-after-revision"
        return "dropped-again"

    v1_drop_fates = {rid: {"v1_status": v1_state[rid]["status"], "v2_status": v2_state[rid]["status"],
                           "fate": fate(rid),
                           "v2_residual_cause": residual_cause(v2_state[rid]["residuals"])}
                     for rid in v1_drops}
    fate_counts = Counter(v["fate"] for v in v1_drop_fates.values())

    v2_drop_causes = {rid: {"cause": residual_cause(v2_state[rid]["residuals"]),
                            "residuals": v2_state[rid]["residuals"],
                            "v1_status": v1_state[rid]["status"]}
                      for rid in v2_drops}
    cause_counts = Counter(v["cause"] for v in v2_drop_causes.values())
    meta_only = sorted(r for r, v in v2_drop_causes.items() if v["cause"] == "metadata_only")
    newly_dropped = sorted(set(v2_drops) - set(v1_drops))

    n = 57
    report = {
        "spec": "w-d-master-dataset.md Amendment 2026-08-09 part P - gate report",
        "generated_by": "compare_primock_v1_v2.py",
        "instrument": {
            "v1": "master/extraction_prompt_v1.txt (spec 3.1 + Amendment 2026-07-29b H)",
            "v2": "master/extraction_prompt_v2.txt (v1 + part P's four fixes + part R "
                  "rubric-anchored importance)",
            "critic_panel": "identical both runs: support/materiality/leakage, claude-opus-5, "
                            "plan path, one revision cycle, re-audit = support critic"},
        "gate2": {
            "rule": "drop rate (still-material after one revision) < 10%",
            "v1": {"dropped": len(v1_drops), "of": n, "drop_rate": round(len(v1_drops) / n, 3)},
            "v2": {"dropped": len(v2_drops), "of": n, "drop_rate": round(len(v2_drops) / n, 3)},
            "v2_verdict": "PASS" if len(v2_drops) / n < 0.10 else "FAIL",
            "v2_drop_rate_excluding_metadata_only_drops": round(
                (len(v2_drops) - len(meta_only)) / n, 3),
            "note": "metadata_only drops died solely on the primock57_parsed.json "
                    "presenting_complaint header contradicting the transcript - outside the "
                    "extraction prompt's reach (the field is source data; the revise loop only "
                    "rewrites the fact_sheet). Even excluding them the gate does not pass."},
        "kept": {"v1": len(v1_kept), "v2": len(v2_kept),
                 "v2_statuses": dict(Counter(e["status"] for e in v2_state.values())),
                 "v1_statuses": dict(Counter(e["status"] for e in v1_state.values()))},
        "size_stats": {
            "raw_all_sheets": {"v1": size_stats(v1_raw), "v2": size_stats(v2_raw)},
            "kept_sheets": {"v1": size_stats(v1_kept), "v2": size_stats(v2_kept)}},
        "flags_by_critic": {"v1": v1_flags, "v2": v2_flags},
        "v1_dropped_ids_under_v2": {
            "counts": dict(fate_counts), "detail": v1_drop_fates},
        "v2_drops": {
            "ids": v2_drops, "cause_counts": dict(cause_counts),
            "metadata_only_ids": meta_only,
            "newly_dropped_ids_kept_under_v1": newly_dropped,
            "detail": v2_drop_causes},
        "per_id_status_map": {rid: {"v1": v1_state[rid]["status"], "v2": v2_state[rid]["status"]}
                              for rid in sorted(v1_state)},
    }
    out = M("primock_v1_v2_comparison.json")
    json.dump(report, open(out, "w"), ensure_ascii=False, indent=1)
    g = report["gate2"]
    print(f"v1 drop {g['v1']['dropped']}/{n} ({g['v1']['drop_rate']:.1%}) -> "
          f"v2 drop {g['v2']['dropped']}/{n} ({g['v2']['drop_rate']:.1%}) - gate 2 {g['v2_verdict']}")
    print(f"  excluding metadata-only drops ({len(meta_only)}: {', '.join(meta_only)}): "
          f"{g['v2_drop_rate_excluding_metadata_only_drops']:.1%}")
    print(f"  fate of the {len(v1_drops)} v1 drops: {dict(fate_counts)}")
    print(f"  newly dropped under v2 (kept under v1): {len(newly_dropped)}: {newly_dropped}")
    print(f"-> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
