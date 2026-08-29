"""One-off: revise the 4 minor-flagged scenarios to the same clean bar as the auto-revised ones.
Reuses critique_scenarios.revise() + run_critic(). Updates critiqued_scenarios.json in place."""
import json, os
from concurrent.futures import ThreadPoolExecutor
from critique_scenarios import revise, run_critic, HERE

FIXES = {
 "cp_anxiety_panic": {"factsheet": [{"severity": "material",
    "what": "must_contain references the patient's age ('in their 20s') which is never stated in the transcript; remove or rephrase any must_contain item that supplies an age/demographic not actually said in the dialogue, so a faithful note is not wrongly penalised."}]},
 "ha_migraine_aura_pill": {"factsheet": [{"severity": "material",
    "what": "must_contain references the patient's age ('woman in her 30s') which is not stated in the transcript; remove/rephrase any must_contain that supplies an age not actually said. Keep the gender only if the dialogue makes it clear."}]},
 "asthma_review": {"realism": [{"severity": "material",
    "what": "Management leaps to a combination ICS/LABA at the same visit despite the problem being non-adherence + poor technique. Make it guideline-consistent: first optimise the existing low-dose ICS (confirm adherence, correct technique, add a spacer) and frame any step-up as conditional on review if control remains poor on a properly-taken preventer. Keep the dose_value / symptom-burden / plan-advice traps intact."}]},
 "polypharmacy_elderly": {"contamination": [{"severity": "material",
    "what": "The doctor over-repeats the 'straight swap, not adding one on top' / 'nothing changes today' point (~4x) and uses faintly documentation-flavoured phrasing ('get this right on the record'); the patient's closing read-backs map one-to-one to the fact-sheet. Make the dialogue more natural: state the switch-vs-add and decision-status points once or twice, drop the meta phrasing, and soften the tidy read-backs - without removing the underlying traps."}]},
}


def main():
    scns = {s["id"]: s for s in json.load(open(os.path.join(HERE, "critiqued_scenarios.json")))}

    def one(sid):
        r = revise(scns[sid], FIXES[sid])
        if r:
            rc = run_critic(r, "factsheet")
            return sid, r, rc
        return sid, None, None

    with ThreadPoolExecutor(max_workers=4) as ex:
        for sid, r, rc in ex.map(one, list(FIXES)):
            if r:
                scns[sid] = r
                print(f"{sid}: revised; re-audit factsheet -> {rc.get('verdict')} ({len(rc.get('issues',[]))} residual)")
            else:
                print(f"{sid}: REVISE FAILED (kept original)")

    json.dump(list(scns.values()), open(os.path.join(HERE, "critiqued_scenarios.json"), "w"),
              ensure_ascii=False, indent=1)
    print("saved -> critiqued_scenarios.json")


if __name__ == "__main__":
    main()
