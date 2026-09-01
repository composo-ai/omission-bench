"""relabel_partial_seed.py - the 34 excluded pairs, formally relabelled omit-partial.

Part of the graded-omission design correction. Free: no model calls, no new construction,
nothing rebuilt.

The ground-truth gate excluded 34 omit pairs because the corrected `truly_absent` check found
a partial or paraphrased residual of the removed fact still in the errored note. Under the
graded design those are not failures - they are the phenomenon. Each is
a valid PARTIAL omission that the old single-class instrument had no label for. This script
gives them the label, keeps every receipt that produced them, and extracts what residual
metadata the auditor's own reasons will honestly yield.

Residual parsing is deliberately conservative. A residual is recorded ONLY when a seed's
reason quotes a span and that span is verbatim-present in the errored note. Reasons that
describe the residual without quoting it ("still appears in the Depression medical reasoning
section") are recorded as `parse_status: "described_not_quoted"` with the verbatim reasons
kept - no guessing. The unparsed ones are filled at full scale by the residual-present check:

    python3 benchmark/verify_omissions_v2.py --seed-set

which is one auditor pass over these pairs and returns quoted, graded residuals directly.

Output: master/partial_omission_seed_set.json

Usage:
  python3 benchmark/relabel_partial_seed.py
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import json, os, re, sys, unicodedata
from collections import Counter

from common import HERE
import omission_sites as OS

SPEC = OS.SPEC
MASTER = OS.MASTER
OUT_FILE = os.path.join(MASTER, "partial_omission_seed_set.json")
WA_STATE = os.path.join(HERE, "results", "w-a-state", "wa_state.json")

QUOTE_CHARS = "‘’“”\"'"
QUOTED = re.compile("[%s]([^%s]{5,240})[%s]" % (QUOTE_CHARS, QUOTE_CHARS, QUOTE_CHARS))


def normlow(t):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", t or "")).strip().lower()


def parse_residuals(runs, errored):
    """Quoted spans from the auditor's reasons that are verbatim-present in the errored
    note. Returns (residuals, parse_status)."""
    lo = normlow(errored)
    found = {}
    for seed, r in runs.items():
        for m in QUOTED.finditer(r.get("reason") or ""):
            q = m.group(1).strip().strip(".,;:")
            if len(normlow(q).split()) < 2 or normlow(q) not in lo:
                continue
            e = found.setdefault(normlow(q), {"quote": q, "seeds": []})
            e["seeds"].append(int(seed))
    if found:
        out = sorted(found.values(), key=lambda e: -len(e["quote"]))
        return out, "quoted_and_verified"
    any_quote = any(QUOTED.search(r.get("reason") or "") for r in runs.values())
    return [], ("quoted_but_not_verbatim" if any_quote else "described_not_quoted")


def fold_completions(pairs):
    """Fold in the two completion passes, if they have run. Both are optional: without them
    this script produces exactly what it produced on 2026-08-10, and the `gaps` block still
    says what is missing.

      master/omission_verification_v2_seed_set.json  (verify_omissions_v2.py --seed-set)
        the auditor's residual-present check, which returns residuals QUOTED, sectioned and
        graded - the metadata the reason-text parser could not honestly recover. Its quotes
        supersede the parsed ones (same evidence, asked directly instead of scraped), and the
        parsed ones are kept beside them as the trail.

      master/partial_seed_severity_grades.json       (grade_partial_seed_severity.py)
        two-arm rubric severity for the pairs that carried none.

    Returns a dict of what was folded, for the file's own provenance block.
    """
    by_id = {p["pair_id"]: p for p in pairs}
    folded = {"residual": None, "severity": None}

    vf = os.path.join(MASTER, "omission_verification_v2_seed_set.json")
    if os.path.exists(vf):
        doc = json.load(open(vf))
        n_quoted = n_none = 0
        for rec in doc["pairs"]:
            p = by_id.get(rec["pair_id"])
            if not p:
                continue
            rv = rec.get("residual_verified") or {}
            sites = rv.get("sites") or []
            r = p["residual"]
            r["sites_parsed_from_reason"] = r["sites"]
            r["parse_status_from_reasons"] = r["parse_status"]
            if sites:
                r["sites"] = [{"quote": s["quote"], "section": s.get("section"),
                               "strength": s.get("strength"),
                               "seeds_naming_it": s.get("seeds_naming_it"),
                               "source": "auditor_residual_present_check"} for s in sites]
                r["n_surviving"] = rv.get("n_verified")
                r["max_strength"] = rv.get("strongest")
                r["parse_status"] = "quoted_and_verified"
                n_quoted += 1
            else:
                # the auditor was asked directly and returned no quotable surviving mention.
                # That is a real, recorded outcome - not a parsing limit - and it is the one
                # state in which this pair's residual is honestly UNQUOTABLE.
                r["n_surviving"] = 0
                r["max_strength"] = None
                r["parse_status"] = "explicitly_unquotable"
                n_none += 1
            r["note"] = ("residual quotes, sections and grades come from the auditor's "
                         "residual-present check (verify_omissions_v2.py --seed-set), which "
                         "asks for them directly instead of scraping them out of free-text "
                         "reasons. `explicitly_unquotable` means the auditor was asked and "
                         "returned no quotable surviving mention - it does not undo the "
                         "3-seed truly_absent majority that put the pair here, and the "
                         "disagreement is recorded in `residual_verification`.")
            p["residual_verification"] = {
                "check": "PARTIAL_CHECK (primary_site_removed, residual_present, "
                         "no_collateral_loss, reads_naturally)",
                "instrument": doc["models"]["auditor"]["resolved"],
                "reasoning_effort": doc["models"]["auditor"].get("reasoning_effort"),
                "lever": doc["lever"],
                "sem_verdict": rec["sem_verdict"], "sem_fails": rec["sem_fails"],
                "n_seeds": rec["n_seeds"], "verified": rec["verified"],
                "verified_reason": rec.get("verified_reason"),
                "algo_gated": rec["algo"].get("gated"),
                "reads_as": "a fail here is a note ABOUT the pair, not a rejection of it: "
                            "these 34 carry their own ground-truth-gate receipts and the "
                            "seed-set pass was run for the residual metadata (algorithmic "
                            "checks ungated). "
                            "primary_site_removed / reads_naturally failures are exactly the "
                            "cases worth a human look.",
            }
        folded["residual"] = {"source": os.path.relpath(vf, HERE), "n": len(doc["pairs"]),
                              "quoted": n_quoted, "explicitly_unquotable": n_none,
                              "spend_usd": doc.get("spend_usd"), "calls": doc.get("calls")}

    sf = os.path.join(MASTER, "partial_seed_severity_grades.json")
    if os.path.exists(sf):
        doc = json.load(open(sf))
        n = 0
        for pid, rec in doc["grades"].items():
            p, con = by_id.get(pid), rec.get("consensus") or {}
            if not p or p.get("severity") or not con.get("grade"):
                continue
            p["severity"] = con["grade"]
            p["severity_source"] = ("rubric_two_arm_consensus"
                                    if not con.get("flagged") else
                                    "rubric_two_arm_lower_grade_flagged")
            p["severity_grades"] = {
                "arms": {a: {k: v for k, v in r.items()
                             if k in ("role", "importance", "test_fired", "rationale")}
                         for a, r in rec["arms"].items()},
                "consensus": con,
                "rubric": "the severity rubric (instrument v1), prompt "
                          "hard_negatives_master.RUBRIC_GRADE verbatim",
                "source": os.path.relpath(sf, HERE),
            }
            n += 1
        folded["severity"] = {"source": os.path.relpath(sf, HERE), "n_graded": n,
                              "flagged": doc.get("counts", {}).get("flagged"),
                              "spend_usd": doc.get("spend", {}).get("cost_usd"),
                              "calls": doc.get("spend", {}).get("calls")}
    return folded


def main():
    rev = json.load(open(os.path.join(MASTER, "truly_absent_reverification.json")))
    post = rev["post_correction"]
    ids = [k for k in rev["group_A_truly_absent"] if post[k]["status_after"] == "excluded"]
    state = json.load(open(WA_STATE))
    hn = {p["pair_id"]: p for p in
          json.load(open(os.path.join(MASTER, "hard_negatives_master.json")))}
    dev = json.load(open(os.path.join(MASTER, "gepa_dev_pool.json")))
    dev_keys = {f"{s}|{i}" for s, lst in dev["ids"].items() for i in lst}

    pairs = []
    for pid in ids:
        pe = state["pairs"][pid]
        stratum, cid, _ = pid.split("|")
        key = f"{stratum}|{cid}"
        note = state["notes"][key]
        src = hn.get(pid, {})
        runs = post[pid]["sem_runs_after"]
        residuals, parse_status = parse_residuals(runs, pe["current"]["errored"])
        pairs.append({
            "pair_id": pid, "class": "omit-partial", "type": "omit",
            "stratum": stratum, "id": cid, "key": key,
            "split": "gepa_dev" if key in dev_keys else "eval",
            "origin": "w_a_excluded_2026-08-10",
            "clean": note["text"], "clean_sha": OS.WA.sha16(note["text"]),
            "errored": pe["current"]["errored"],
            "change": pe["current"].get("change"), "what": pe["current"].get("what"),
            "target": src.get("target"),
            "fact": (src.get("target") or {}).get("fact")
                    or (src.get("target") or {}).get("trap")
                    or (pe["current"].get("what") if isinstance(pe["current"].get("what"), str)
                        else json.dumps(pe["current"].get("what"))),
            "fact_key": (f"trap|{src['target']['index']}" if (src.get("target") or {}).get("kind") == "trap"
                         else f"mc|{src['target']['index']}" if (src.get("target") or {}).get("kind") == "must_contain"
                         else None),
            "severity": src.get("severity"), "severity_source": src.get("severity_source"),
            "mc_idx": post[pid]["mc_idx_after"],
            "residual": {
                "n_surviving": None,
                "parse_status": parse_status,
                "sites": [{"quote": r["quote"], "section": None, "strength": None,
                           "seeds_naming_it": sorted(r["seeds"]), "source": "parsed_from_reason"}
                          for r in residuals],
                "max_strength": None,
                "note": "residual COUNT, section and strength are not recoverable from the "
                        "reason text; verify_omissions_v2.py --seed-set fills them with one "
                        "auditor pass. Absence of a parsed quote is a parsing limit, not "
                        "evidence that no residual exists - all 34 failed truly_absent by "
                        "3-seed majority, which IS the evidence a residual exists.",
            },
            "construction": {
                "method": pe["current"].get("kind"),
                "builder": "hard_negatives_master.py / w_a_master.py (v1 single-span injector)",
                "note": "built as a v1 single-span omission; it is a PARTIAL omission by "
                        "outcome, not by design - which is exactly what the correction found",
            },
            "verification_receipts": {
                "instrument": rev["outcome"]["instrument"],
                "check": "corrected EDIT_CHECK truly_absent (whole-note residual clause)",
                "verdict": post[pid]["verdict_after"],
                "sem_fails": post[pid]["sem_fails_after"],
                "n_seeds": post[pid]["n_seeds_after"],
                "seed_runs": runs,
                "ground_truth_gate_excluded_reason": pe.get("excluded_reason"),
                "reads_as": "truly_absent FALSE by 3-seed majority = a residual mention of "
                            "the removed fact survives = a verified PARTIAL omission",
            },
            "provenance": {
                "relabelled_by": "relabel_partial_seed.py", "relabelled_utc": OS.now_utc(),
                "sources": ["master/truly_absent_reverification.json",
                            "results/w-a-state/wa_state.json",
                            "master/hard_negatives_master.json"],
                "frozen_artefact_touched": False,
            },
        })

    pairs.sort(key=lambda p: p["pair_id"])
    folded = fold_completions(pairs)
    out = {
        "generated_utc": OS.now_utc(), "spec": SPEC,
        "completions_folded_in": folded,
        "what": "the 34 omit pairs the ground-truth gate excluded for residual presence, "
                "formally relabelled "
                "omit-partial. These are the graded-omission design's seed partial set: "
                "already built, already verified to contain a residual, costing nothing to "
                "reuse. They are NOT in master/pairs_master_frozen.json and this file does "
                "not put them there - the frozen 281 and its pre-registered analysis are "
                "untouched.",
        "counts": {
            "pairs": len(pairs),
            "by_stratum": dict(Counter(p["stratum"] for p in pairs)),
            "by_split": dict(Counter(p["split"] for p in pairs)),
            "residual_parse_status": dict(Counter(p["residual"]["parse_status"] for p in pairs)),
            "with_verified_residual_quote": sum(1 for p in pairs if p["residual"]["sites"]),
            "residual_max_strength": dict(Counter(p["residual"].get("max_strength")
                                                  for p in pairs)),
            "residual_count": dict(Counter(p["residual"].get("n_surviving") for p in pairs)),
            "severity": dict(Counter(p.get("severity") for p in pairs)),
            "severity_source": dict(Counter(p.get("severity_source") for p in pairs)),
            "target_kind": dict(Counter((p.get("target") or {}).get("kind") for p in pairs)),
            "construction_method": dict(Counter(p["construction"]["method"] for p in pairs)),
            "schema_complete": all(
                p.get("severity") and p["residual"].get("parse_status") in
                ("quoted_and_verified", "explicitly_unquotable") for p in pairs),
        },
        "gaps": {
            "residual_metadata_incomplete": {
                "n": sum(1 for p in pairs
                         if p["residual"]["parse_status"] not in
                         ("quoted_and_verified", "explicitly_unquotable")),
                "which": [p["pair_id"] for p in pairs
                          if p["residual"]["parse_status"] not in
                          ("quoted_and_verified", "explicitly_unquotable")],
                "why": "the auditor's reason describes the residual without quoting it, or "
                       "quotes a paraphrase that is not verbatim in the note",
                "fix": "verify_omissions_v2.py --seed-set (one auditor pass; the "
                       "residual-present check returns quoted, graded, sectioned residuals)",
            },
            "residual_contested": {
                "n": sum(1 for p in pairs
                         if p["residual"].get("parse_status") == "explicitly_unquotable"),
                "which": [p["pair_id"] for p in pairs
                          if p["residual"].get("parse_status") == "explicitly_unquotable"],
                "why": "the seed-set pass was asked directly and returned no quotable "
                       "surviving mention, against a 3-seed truly_absent majority that says "
                       "one exists. Two instruments disagree about the same note; neither is "
                       "overruled here. Not a gap in the metadata - a gap in the agreement",
                "fix": "human adjudication (queued in the sitting pack)",
            },
            "severity_missing": {
                "n": sum(1 for p in pairs if not p.get("severity")),
                "which": [p["pair_id"] for p in pairs if not p.get("severity")],
                "why": "authored-stratum pairs were preserved verbatim from the pre-master "
                       "build, so they carry no injector target, and the ground-truth gate's "
                       "severity backfill ran only over PASSING pairs - these were excluded "
                       "before it",
                "fix": "one rubric grade each (hard_negatives_master.RUBRIC_GRADE on the "
                       "constructor role), ~10 calls, before any severity-stratified reading",
            },
            "severity_contested": {
                "n": sum(1 for p in pairs
                         if p.get("severity_source") == "rubric_two_arm_lower_grade_flagged"),
                "which": [p["pair_id"] for p in pairs
                          if p.get("severity_source") == "rubric_two_arm_lower_grade_flagged"],
                "why": "the two rubric arms disagreed; the rubric's conservative tie-break "
                       "(lower grade) decided it and the pair is flagged rather than settled",
                "fix": "human adjudication (queued in the sitting pack)",
            },
            "out_of_scope": {
                "group_B_single_edit_only": rev["group_B_single_edit_only"],
                "why": "4 authored pairs excluded on single_edit alone - each deletes a whole "
                       "Impression/assessment section, which the semantic clause calls several "
                       "edits and the v2 algorithmic rule calls one. That tension is a "
                       "separate open question and is NOT "
                       "resolved by relabelling; the graded design's omit-complete class does "
                       "redefine single-edit for omissions, so they are worth revisiting under "
                       "it, but not by this script",
            },
        },
        "pairs": pairs,
    }
    tmp = OUT_FILE + ".tmp"
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, OUT_FILE)
    c = out["counts"]
    print(f"master/partial_omission_seed_set.json - {c['pairs']} pairs relabelled omit-partial")
    print(f"  strata {c['by_stratum']} | split {c['by_split']}")
    print(f"  residual parse: {c['residual_parse_status']} | with a verified quote: "
          f"{c['with_verified_residual_quote']}/{c['pairs']} | strongest "
          f"{c['residual_max_strength']}")
    print(f"  severity {c['severity']} | source {c['severity_source']}")
    print(f"  target kind {c['target_kind']} | built as {c['construction_method']}")
    print(f"  folded in: {json.dumps(folded)}")
    print(f"  SCHEMA COMPLETE: {c['schema_complete']} | open gaps "
          f"{ {k: v['n'] for k, v in out['gaps'].items() if isinstance(v, dict) and 'n' in v} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
