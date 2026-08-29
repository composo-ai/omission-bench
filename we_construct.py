#!/usr/bin/env python3
"""W-E steps 4-6 (spec Section 5) - fact-sheet-analog construction over the RoSE sample.

  step 4  we_not_contain  - WE_NOT_CONTAIN per document        -> rose_master/not_contain_raw.json
  step 5  we_salience     - WE_SALIENCE per document's ACU list -> rose_master/salience_raw.json
  step 6  we_critic       - WE_CRITIC_SUPPORT + WE_CRITIC_PLAUSIBILITY panel, one revision cycle,
                            re-audit                            -> rose_master/not_contain.json,
                            rose_master/salience.json, rose_master/we_critique_report.md, and the
                            reformatted rose_master/fact_sheets.json (schema-compatible with the
                            clinical fact_sheet: must_contain / must_not_contain / salience_traps)

The spec keeps these as three scripts (we_not_contain.py / we_salience.py / we_critic.py); they
share a document loop, a resumable state file and one prompt gate, so they live here behind
--stage. Every stage is resumable and idempotent: state is keyed by (stage, doc_id).

All calls run on the Claude plan path (spec section 4: claude-opus-4-8, effort medium, $0
marginal). NO judge-arm call is made anywhere in this file.

Usage
  python3 we_construct.py --stage all [--limit N] [--workers 4]
  python3 we_construct.py --stage not_contain --limit 3     # shape check
  python3 we_construct.py --stage all --limit 3 --dry-run   # render prompts only, zero calls

Open points this script deliberately does NOT decide (see docs / the ingest report):
  * spec 3.4 says the critic panel "audits WE_NOT_CONTAIN and WE_SALIENCE output", but both
    pre-registered critic prompts take only {article} and {claims} - there is no salience slot.
    The panel therefore audits the must_not_contain claims only, and salience grades pass through
    unaudited. Recorded in the critique report as an UNAUDITED gap rather than patched with an
    invented third critic prompt.
  * spec 2.1 defines the salience_traps `trap` field as "the ACU text framed as what a summarizer
    could drop or distort". The ACU text is used VERBATIM here - no rewording - because any
    reframing template would be new instrument text that the pre-registration does not contain.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import we_common as W

STATE = os.path.join(W.ROSE_MASTER, "construct_state.json")
STAGES = ("not_contain", "salience", "critic")


# ---------------------------------------------------------------- prompt rendering
def p_not_contain(prompts, doc):
    return W.render(prompts["we_not_contain"], article=doc["source"], summary=doc["reference"])


def p_salience(prompts, doc):
    return W.render(prompts["we_salience"], article=doc["source"], summary=doc["reference"],
                    acu_list=W.acu_list_block(doc["reference_acus"]))


def claims_block(claims):
    return "\n".join(f"{i}. {c.get('claim', '')}" for i, c in enumerate(claims))


def p_critic(prompts, which, doc, claims):
    return W.render(prompts[f"we_critic_{which}"], article=doc["source"],
                    claims=claims_block(claims))


# ---------------------------------------------------------------- stages
def run_not_contain(prompts, doc, dry):
    prompt = p_not_contain(prompts, doc)
    if dry:
        return {"rendered_chars": len(prompt), "preview": prompt[:400]}
    out = W.construct(prompt)
    claims = (out or {}).get("claims") or []
    return {"claims": claims, "n_claims": len(claims), "raw_ok": bool(out)}


def run_salience(prompts, doc, dry):
    prompt = p_salience(prompts, doc)
    if dry:
        return {"rendered_chars": len(prompt), "preview": prompt[:400]}
    out = W.construct(prompt)
    grades = (out or {}).get("acu_grades") or []
    n = len(doc["reference_acus"])
    clean, problems = [], []
    seen = set()
    for g in grades:
        idx = g.get("acu_idx")
        if not isinstance(idx, int) or not 0 <= idx < n:
            problems.append(f"acu_idx {idx!r} out of range 0..{n - 1}")
            continue
        if idx in seen:
            problems.append(f"duplicate acu_idx {idx}")
            continue
        if g.get("importance") not in W.IMPORTANCE:
            problems.append(f"acu_idx {idx}: importance {g.get('importance')!r} not in "
                            f"{W.IMPORTANCE}")
            continue
        seen.add(idx)
        clean.append({"acu_idx": idx, "acu": doc["reference_acus"][idx],
                      "importance": g["importance"], "mode": g.get("mode"),
                      "rationale": g.get("rationale")})
    missing = sorted(set(range(n)) - seen)
    if missing:
        problems.append(f"no grade returned for acu_idx {missing}")
    return {"acu_grades": sorted(clean, key=lambda g: g["acu_idx"]),
            "n_acus": n, "n_graded": len(clean), "problems": problems, "raw_ok": bool(out)}


def audit_claims(prompts, doc, claims, dry):
    """Two-critic panel over one document's must_not_contain claims (spec 3.4)."""
    if dry:
        return {"support": None, "plausibility": None,
                "rendered_chars": {w: len(p_critic(prompts, w, doc, claims))
                                   for w in ("support", "plausibility")}}
    def call(which):
        # one retry: the plan path degrades to None on a timeout / transient refusal rather than
        # raising, and a missing critic verdict must never be silently read as a rejection
        for _ in range(2):
            out = W.construct(p_critic(prompts, which, doc, claims))
            if out and out.get("items"):
                return out
        return {}

    sup, pla = call("support"), call("plausibility")
    verdicts = {}
    for i in range(len(claims)):
        s = next((x for x in sup.get("items", []) if x.get("idx") == i), None)
        p = next((x for x in pla.get("items", []) if x.get("idx") == i), None)
        verdicts[i] = {
            "genuinely_unsupported": None if s is None else bool(s.get("genuinely_unsupported")),
            "support_reason": None if s is None else s.get("reason"),
            "plausible": None if p is None else bool(p.get("plausible")),
            "plausibility_reason": None if p is None else p.get("reason"),
        }
        v = verdicts[i]
        # three states, deliberately distinct: a claim the panel positively rejected is dropped;
        # a claim the panel never returned a verdict on is INCOMPLETE, not rejected
        v["rejected"] = (v["genuinely_unsupported"] is False) or (v["plausible"] is False)
        v["incomplete"] = (v["genuinely_unsupported"] is None) or (v["plausible"] is None)
        v["flagged"] = v["rejected"] or v["incomplete"]
    return {"verdicts": verdicts, "support_raw_ok": bool(sup), "plausibility_raw_ok": bool(pla)}


def run_critic(prompts, doc, nc, dry):
    """Panel -> one regeneration of the flagged claims -> re-audit -> accept or drop
    (spec 3.4 revision loop, the critique_scenarios.revise() pattern)."""
    claims = list(nc.get("claims") or [])
    if not claims:
        return {"accepted": [], "dropped": [], "rounds": 0, "note": "no claims to audit"}
    a1 = audit_claims(prompts, doc, claims, dry)
    if dry:
        return {"round1": a1, "rounds": 1}
    flagged = [i for i, v in a1["verdicts"].items() if v["flagged"]]
    regenerated, a2 = [], None
    if flagged:
        # single regeneration of the flagged claims only, then one re-audit
        out = W.construct(p_not_contain(prompts, doc))
        regenerated = (out or {}).get("claims") or []
        if regenerated:
            merged = [c for i, c in enumerate(claims) if i not in flagged] + regenerated
            a2 = audit_claims(prompts, doc, merged, dry)
            claims = merged
            a1 = a2
            flagged = [i for i, v in a1["verdicts"].items() if v["flagged"]]
    accepted = [{**claims[i], "audit": a1["verdicts"][i]}
                for i in range(len(claims)) if i not in flagged]
    dropped = [{**claims[i], "audit": a1["verdicts"][i]}
               for i in flagged if a1["verdicts"][i]["rejected"]]
    incomplete = [{**claims[i], "audit": a1["verdicts"][i]}
                  for i in flagged if not a1["verdicts"][i]["rejected"]]
    return {"accepted": accepted, "dropped": dropped, "audit_incomplete": incomplete,
            "rounds": 2 if a2 else 1, "n_regenerated": len(regenerated),
            "n_accepted": len(accepted), "n_dropped": len(dropped),
            "n_incomplete": len(incomplete)}


# ---------------------------------------------------------------- fact sheets
def build_fact_sheets(docs, not_contain, salience):
    """Reformat into the clinical 3-key fact_sheet shape (spec 2.1) so the severity-correlation
    code works unmodified across domains."""
    sheets = []
    for d in docs:
        did = d["doc_id"]
        nc = (not_contain.get(did) or {}).get("accepted") or []
        grades = (salience.get(did) or {}).get("acu_grades") or []
        traps = []
        for g in grades:
            if g["importance"] not in ("critical", "supporting"):
                continue      # spec 2.1: every ACU graded critical or supporting
            traps.append({
                "trap": g["acu"],          # ACU text VERBATIM - see the module docstring
                "correct_handling": (f"state this fact accurately, preserving its "
                                     f"{g['mode']}-relevant detail"),
                "mode": g["mode"],
                "acu_idx": g["acu_idx"],
                "importance": g["importance"],
            })
        sheets.append({
            "doc_id": did,
            "corpus": d["corpus"],
            "must_contain": d["reference_acus"],          # given directly by RoSE (spec 2.1)
            "must_not_contain": [c.get("claim") for c in nc],
            "must_not_contain_detail": nc,
            "salience_traps": traps,
            "acu_importance": {str(g["acu_idx"]): g["importance"] for g in grades},
            "acu_modes": {str(g["acu_idx"]): g["mode"] for g in grades},
        })
    return sheets


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("all",) + STAGES)
    ap.add_argument("--limit", type=int, default=0, help="first N documents only (shape check)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true",
                    help="render every prompt and validate the plumbing, make ZERO LLM calls")
    ap.add_argument("--out-suffix", default="", help="suffix for output files (e.g. _shapecheck)")
    args = ap.parse_args()

    prompts = W.load_prompts()          # hash gate; refuses to run on a single changed byte
    docs = W.load_sample()
    if args.limit:
        docs = docs[:args.limit]
    stages = STAGES if args.stage == "all" else (args.stage,)
    state = W.load_state(STATE, {"not_contain": {}, "salience": {}, "critic": {}})
    sfx = args.out_suffix

    def path(base):
        root, ext = os.path.splitext(base)
        return root + sfx + ext

    for stage in stages:
        todo = [d for d in docs if d["doc_id"] not in state.setdefault(stage, {})]
        print(f"[{stage}] {len(todo)} of {len(docs)} documents to do"
              + (" (dry-run: no LLM calls)" if args.dry_run else ""))
        if not todo:
            continue

        def work(d):
            if stage == "not_contain":
                return d["doc_id"], run_not_contain(prompts, d, args.dry_run)
            if stage == "salience":
                return d["doc_id"], run_salience(prompts, d, args.dry_run)
            nc = state["not_contain"].get(d["doc_id"]) or {}
            return d["doc_id"], run_critic(prompts, d, nc, args.dry_run)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for did, res in ex.map(work, todo):
                state[stage][did] = res
                print(f"  {stage} {did}: "
                      + json.dumps({k: v for k, v in res.items()
                                    if k in ("n_claims", "n_graded", "rounds", "rendered_chars",
                                             "raw_ok")})[:160])
        if not args.dry_run:
            W.save_json(STATE, state)

    if args.dry_run:
        print("dry-run complete - prompts rendered, no state written, no calls made")
        return

    if "not_contain" in stages:
        W.save_json(path(W.NOT_CONTAIN_RAW), {"generated": date.today().isoformat(),
                                              "n": len(state["not_contain"]),
                                              "documents": state["not_contain"]})
    if "salience" in stages:
        W.save_json(path(W.SALIENCE_RAW), {"generated": date.today().isoformat(),
                                           "n": len(state["salience"]),
                                           "documents": state["salience"]})
    if "critic" in stages:
        W.save_json(path(W.NOT_CONTAIN), {"generated": date.today().isoformat(),
                                          "n": len(state["critic"]),
                                          "documents": state["critic"]})
        W.save_json(path(W.SALIENCE), {"generated": date.today().isoformat(),
                                       "audited": False,
                                       "note": ("spec 3.4's critic prompts carry no salience slot, "
                                                "so these grades are NOT panel-audited - flagged, "
                                                "not patched"),
                                       "n": len(state["salience"]), "documents": state["salience"]})
        sheets = build_fact_sheets(docs, state["critic"], state["salience"])
        W.save_json(path(W.FACT_SHEETS), {"generated": date.today().isoformat(),
                                          "spec": "w-e-cross-domain-generalization.md 2.1, 5.6",
                                          "n": len(sheets), "fact_sheets": sheets})
        n_acc = sum(len(v.get("accepted") or []) for v in state["critic"].values())
        n_drop = sum(len(v.get("dropped") or []) for v in state["critic"].values())
        n_traps = sum(len(s["salience_traps"]) for s in sheets)
        with open(path(W.CRITIQUE_REPORT), "w", encoding="utf-8") as f:
            f.write(f"# W-E critic panel report ({date.today().isoformat()})\n\n"
                    f"- documents audited: {len(state['critic'])}\n"
                    f"- must_not_contain claims accepted: {n_acc}\n"
                    f"- must_not_contain claims dropped after one revision cycle: {n_drop}\n"
                    f"- salience_traps built (critical + supporting ACUs): {n_traps}\n\n"
                    "## Unaudited gap\n\n"
                    "spec 3.4 states the panel audits WE_NOT_CONTAIN **and** WE_SALIENCE output, "
                    "but both pre-registered critic prompts accept only `{article}` and "
                    "`{claims}`. The salience grades are therefore passed through unaudited "
                    "rather than checked by an invented third critic. Flagged for review.\n")
        print(f"fact sheets: {len(sheets)}  claims accepted {n_acc} / dropped {n_drop}  "
              f"traps {n_traps}")


if __name__ == "__main__":
    main()
