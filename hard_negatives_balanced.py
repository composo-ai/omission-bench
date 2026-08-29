"""Balanced hard negatives for the HONEST crossover (P2.1).

Per authored scenario: build the verified-clean IDEAL note, then inject ONE critical error of each
TYPE - add / change / omit. The OMIT cell is the key addition the prior set (composo_arm/M2) lacked:
a faithfulness judge is structurally blind to omission ("omissions are NOT errors"), so it's exactly
where the steelman collapses and Composo (which penalises omission) separates. add/change are where
the steelman is strong - so the three types together give the honest, on-thesis crossover.

Reuses hard_negatives.build_ideal (the same verified-clean ideal-note builder).
Output: ideal_notes_balanced.json, hard_negatives_balanced.json
"""
import json, os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from common import claude_json, HERE
from hard_negatives import build_ideal

MODEL, EFFORT, WORKERS = "claude-opus-4-8", "medium", 6

TYPES = {
 "add":    "Introduce EXACTLY ONE clinically-significant FABRICATION: assert a finding, diagnosis, "
           "symptom, examination result, or medication that is NOT supported anywhere in the transcript "
           "and that would change clinical understanding (e.g. a physical exam that never happened, a "
           "diagnosis never made, a drug never discussed). Keep it plausible; change nothing else.",
 "change": "Introduce EXACTLY ONE clinically-dangerous CHANGE to an existing assertion: flip a negation "
           "on a serious finding ('denies'->'reports', 'no X'->'X'), OR change a medication dose/frequency "
           "to a wrong value, OR harden a tentative/deferred serious decision into a firm one "
           "('wait-and-see'->'started', '?diagnosis'->confirmed). Change nothing else.",
 "omit":   "REMOVE EXACTLY ONE clinically load-bearing fact so the note has a DANGEROUS OMISSION: drop "
           "the working diagnosis/impression, OR a red-flag safety-net instruction, OR a stated drug "
           "allergy, OR a key pertinent negative, OR a critical examination finding. Keep the rest of the "
           "note intact and natural; do NOT flag that anything was removed.",
}

INJECT = """Consultation TRANSCRIPT and a correct clinical NOTE.

TRANSCRIPT:
{t}

NOTE:
{note}

{instr}

Return ONLY JSON {{"note":"<modified note>","change":"<one-line description of the single edit>",
"what":"<the specific fact added / changed / removed>"}}."""


def inject(scen_by_id, ideal_rec, typ):
    s = scen_by_id[ideal_rec["id"]]
    res = claude_json(INJECT.format(t=s["transcript"][:40000], note=ideal_rec["note"], instr=TYPES[typ]),
                      model=MODEL, effort=EFFORT, retries=2)
    if not (isinstance(res, dict) and res.get("note")):
        return None
    return {"id": ideal_rec["id"], "type": typ, "clean": ideal_rec["note"], "errored": res["note"],
            "change": res.get("change"), "what": res.get("what")}


def main():
    scen = json.load(open(os.path.join(HERE, "authored_scenarios.json")))   # 30
    scen_by_id = {s["id"]: s for s in scen}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        ideals = list(ex.map(build_ideal, scen))
    json.dump(ideals, open(os.path.join(HERE, "ideal_notes_balanced.json"), "w"), indent=1)
    ok = [i for i in ideals if i["ok"]]
    print(f"ideal notes: {len(ok)}/{len(ideals)} built")

    jobs = [(i, t) for i in ok for t in TYPES]   # one of each type per scenario
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        hns = [h for h in ex.map(lambda j: inject(scen_by_id, j[0], j[1]), jobs) if h]
    json.dump(hns, open(os.path.join(HERE, "hard_negatives_balanced.json"), "w"), indent=1)
    print(f"hard negatives: {len(hns)} | by type: {dict(Counter(h['type'] for h in hns))}")
    for h in hns[:3]:
        print(f"  {h['id']}/{h['type']}: {h.get('change')}")


if __name__ == "__main__":
    main()
