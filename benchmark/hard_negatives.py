"""Hard negatives: the manufactured-failure recipes the pair set is built from.

First of three. `hard_negatives_balanced.py` balances these recipes across the three
error types and `hard_negatives_master.py` runs the balanced set over the whole corpus;
each imports the one before it, so all three are live rather than superseded.

Two steps, both grounded in the authored fact-sheets (verified-clean by construction):
  1. IDEAL reference note per authored scenario - a correct SOAP note that includes every
     must_contain, asserts none of the must_not_contain, handles each salience trap right.
     (Verified-clean base = the thing the measurement-wall round lacked.)
  2. INJECT one known error per ideal note across the 2x2 grid (blatant<->subtle x
     trivial<->critical), which labels every injected error by construction.

Output: ideal_notes.json + hard_negatives.json
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import json, os
from concurrent.futures import ThreadPoolExecutor
from common import claude_json, HERE

def scenarios():
    """The authored scenarios and their answer keys.

    Not carried by this repository: the authored consultations are released as the dataset
    repository's `transcripts/authored/` and `fact_sheets/`, which hold the same material
    in the released schema. Assemble them into this file's shape to re-run this stage.
    """
    path = os.path.join(HERE, "authored_scenarios.json")
    if not os.path.exists(path):
        raise SystemExit(
            "cannot find %s - the authored scenarios are released as the dataset "
            "repository's transcripts/authored/ and fact_sheets/" % path)
    return json.load(open(path))
MODEL, EFFORT, WORKERS = "claude-opus-4-8", "medium", 6

IDEAL = """Write the CORRECT clinical note for this primary-care consultation - a concise, faithful SOAP-style
note (Presenting complaint / History / Examination / Impression / Plan) that:
- includes EVERY must_contain fact, accurately;
- asserts NONE of the must_not_contain items;
- handles each salience trap correctly (e.g. records uncertainty where the clinician was uncertain).
Normal UK GP documentation style. Return ONLY JSON {{"note":"<the note text>"}}.

MUST_CONTAIN:
{mc}
MUST_NOT_CONTAIN (never assert these):
{mnc}
SALIENCE TRAPS (handle each correctly):
{traps}
TRANSCRIPT:
{t}"""

CELLS = {
    "blatant.critical": "Flip a negation on a SERIOUS finding so the meaning reverses (e.g. a documented red-flag 'denies X' -> 'reports X', or 'no evidence of Y' -> 'evidence of Y'). Unmistakable on inspection.",
    "subtle.critical":  "Make ONE small but clinically dangerous ASSERTION change: harden a tentative plan/finding into a definite one ('considered antibiotics'/'wait-and-see' -> 'started antibiotics'; '?UTI' -> 'UTI'), OR change a medication dose/frequency to a wrong value. Inconspicuous.",
    "blatant.trivial":  "Add ONE obviously fabricated but clinically harmless detail to the social history (e.g. 'keen golfer', 'owns a cat') - never mentioned. Obvious.",
    "subtle.trivial":   "Make ONE small low-stakes ASSERTION change: a temporal/value tweak ('today'->'yesterday', '3 days'->'2 days') that barely alters clinical meaning.",
}
INJECT = """Consultation TRANSCRIPT and a correct clinical NOTE.

TRANSCRIPT:
{t}

NOTE:
{note}

Introduce EXACTLY ONE error of this kind: {cell_instr}
Change as little else as possible; keep it plausible. Return ONLY JSON
{{"note":"<modified note>","change":"<one-line description of the single change>","mode":"<negation|dose_value|modality_hardening|fabrication|temporal>"}}."""


def build_ideal(s):
    fs = s["fact_sheet"]
    res = claude_json(IDEAL.format(
        mc="\n".join("- " + x for x in fs["must_contain"]),
        mnc="\n".join("- " + x for x in fs["must_not_contain"]),
        traps="\n".join(f"- {t['trap']} -> {t['correct_handling']}" for t in fs["salience_traps"]),
        t=s["transcript"][:40000]), model=MODEL, effort=EFFORT)
    note = res.get("note", "") if isinstance(res, dict) else ""
    return {"id": s["id"], "note": note, "ok": len(note) > 150}


def inject(scen_by_id, ideal_rec, cell):
    s = scen_by_id[ideal_rec["id"]]
    res = claude_json(INJECT.format(t=s["transcript"][:40000], note=ideal_rec["note"],
                                    cell_instr=CELLS[cell]), model=MODEL, effort=EFFORT)
    if not (isinstance(res, dict) and res.get("note")):
        return None
    return {"id": ideal_rec["id"], "cell": cell, "clean": ideal_rec["note"],
            "errored": res["note"], "change": res.get("change"), "mode": res.get("mode")}


def main():
    scen = scenarios()
    scen_by_id = {s["id"]: s for s in scen}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        ideals = list(ex.map(build_ideal, scen))
    json.dump(ideals, open(os.path.join(HERE, "ideal_notes.json"), "w"), indent=1)
    ok_ideals = [i for i in ideals if i["ok"]]
    print(f"ideal notes: {len(ok_ideals)}/{len(ideals)} built")

    # assign cells round-robin so all four are represented across the 10 scenarios
    cells = list(CELLS)
    jobs = [(ok_ideals[i], cells[i % len(cells)]) for i in range(len(ok_ideals))]
    # plus a guaranteed blatant.critical for the M0 demo (first scenario, if not already)
    if jobs and jobs[0][1] != "blatant.critical":
        jobs.append((ok_ideals[0], "blatant.critical"))

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        hns = [h for h in ex.map(lambda j: inject(scen_by_id, j[0], j[1]), jobs) if h]
    json.dump(hns, open(os.path.join(HERE, "hard_negatives.json"), "w"), indent=1)
    from collections import Counter
    print(f"hard negatives: {len(hns)} | by cell: {dict(Counter(h['cell'] for h in hns))}")
    print("saved -> ideal_notes.json + hard_negatives.json")
    # show the M0 candidate
    m0 = next((h for h in hns if h["cell"] == "blatant.critical"), None)
    if m0:
        print(f"\nM0 demo candidate ({m0['id']}): change = {m0['change']}")


if __name__ == "__main__":
    main()
