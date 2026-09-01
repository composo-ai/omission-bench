"""Author the trap-blind scenarios: the same procedure, with no planted error.

Fork of author_scenarios.py with the three pre-registered deletions from build_prompt():
  (i)   the brief specs carry NO "TRAPS:" sentences - clinical content only;
  (ii)  requirement 2 ("Every salience trap named in the brief MUST genuinely occur...") deleted;
  (iii) requirement 3 replaced by "Do NOT produce a fact_sheet - output keys are exactly:
        id, presenting_complaint, transcript" and requirement 4 deleted.
The exemplar scenario is passed with its `fact_sheet` key STRIPPED, so the author model never
sees a traps schema. Fact sheets for these 10 come exclusively from the blind extraction
pipeline in extract_fact_sheets.py - that symmetry is the whole point of the guard.

Clusters are deliberately drawn from modes/clusters NOT already saturated by the authored 30:
med-review (AF/anticoagulation), MSK (shoulder), paediatric fever, dermatology (psoriasis,
distinct from the existing eczema), results-review (lipids/QRISK), plus vestibular, menopause,
GI (chronic/IBS), new-hypertension workup, and acute gout.

Usage: python corpus/author_trapblind.py [--only id1,id2] [--workers 5]
Output: master/trapblind_scenarios_raw.json (+ master/trapblind_author_failures/ raw dumps).
Idempotent: ids already present with a transcript in the output file are skipped on rerun.
Then: critique_scenarios.py --in master/trapblind_scenarios_raw.json --critics realism,contamination
      --out master/trapblind_scenarios_critiqued.json --report master/trapblind_critique_report.md
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
from common import claude_json, claude, HERE

EXEMPLAR_ID = "sore_throat_anchoring"   # same exemplar as author_scenarios.py, fact_sheet stripped
MODEL = "claude-opus-4-8"
EFFORT = "medium"
TIMEOUT = 540
OUTFILE = os.path.join(HERE, "master", "trapblind_scenarios_raw.json")


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


WORKERS = int(_arg("--workers", "5"))
ONLY = set((_arg("--only") or "").split(",")) - {""}

# Each spec: id, presenting_complaint, brief. CLINICAL CONTENT ONLY - no traps section. The
# brief states presentation, history, findings, working diagnosis and
# plan (incl. doses and safety-netting) exactly as an authored consult needs, and nothing about
# what a scribe could get wrong.
SPECS = [
 {"id": "tb_af_anticoag_review", "presenting_complaint": "atrial fibrillation medication review",
  "brief": "Patient in their mid-70s attends an annual medication review for atrial fibrillation, in person. Current meds: apixaban 5mg twice daily and bisoprolol 2.5mg once daily. Feels well; no palpitations, no breathlessness, no chest pain, no blackouts. Admits to occasionally forgetting the evening apixaban dose, maybe once or twice a month; the doctor explains why consistent dosing matters for stroke prevention. On asking about bleeding: occasional gum bleeding when brushing, but no nosebleeds, no blood in urine or stool, no black stools, no easy bruising. Takes the odd ibuprofen for a stiff knee; the doctor advises paracetamol instead because ibuprofen with an anticoagulant raises bleeding risk. Pulse irregularly irregular at around 72, BP 128/76. Annual bloods (kidney function, full blood count) are due and get booked for next week - apixaban dosing depends on kidney function so the doctor will confirm the dose is still right once results are back. Alcohol modest, within guidance. Plan: continue both medications unchanged for now, bloods next week, telephone follow-up once results are back, and a safety-net: any significant bleeding, black or tarry stools, blood in urine, or a sudden severe headache means urgent review."},
 {"id": "tb_shoulder_pain", "presenting_complaint": "shoulder pain",
  "brief": "Painter-decorator in their early 50s, about 6 weeks of RIGHT shoulder pain, worse reaching overhead and lying on that side at night; no injury or fall, came on gradually during a big exterior job. No neck pain, no pins and needles or numbness in the arm or hand, no weakness dropping things, no weight loss, no fever or night sweats. Exam: painful arc on abduction roughly between 60 and 120 degrees, pain on resisted external rotation, full passive range once guided through, no wasting around the shoulder, neck movements full and pain-free. Working diagnosis: rotator-cuff-related shoulder pain (subacromial pain syndrome). Plan: relative rest from sustained overhead work where possible, regular simple analgesia (paracetamol, plus a topical NSAID gel over the shoulder), referral to physiotherapy with a simple pendulum-and-wall-slide home exercise routine to start meanwhile, review in about 6 weeks, and if it has not settled then a steroid injection would be the next option to discuss. Safety-net: sudden loss of power in the arm, constant unremitting night pain, or feeling unwell would need earlier review."},
 {"id": "tb_child_fever_ear", "presenting_complaint": "fever and ear pain in a toddler",
  "brief": "A parent brings their 18-month-old in with 2 days of fever, up to 38.9 at home on a forehead thermometer, and pulling at the LEFT ear, more unsettled overnight. Drinking a bit less than usual but still taking bottles and water, wet nappies as normal, eating less. No rash, no vomiting, no diarrhoea, no cough to speak of, no known contact with anyone seriously unwell. In the room the child is alert, clingy but consolable, playing with a toy by the end. Exam: temperature 38.2, left eardrum red and bulging, right eardrum normal, throat slightly red, chest clear, no rash anywhere on undressing, hands and feet warm, capillary refill under 2 seconds. Working diagnosis: acute otitis media of the left ear. Plan: regular paracetamol and ibuprofen syrup for fever and pain at weight-appropriate doses, keep fluids going little and often, most cases settle without antibiotics in 2 to 3 days; a back-up prescription for amoxicillin syrup three times a day for 5 days is given to the parent to start ONLY if things are no better after 48 hours or get worse sooner. Safety-net: a rash that does not fade with a glass pressed on it, the child becoming drowsy or floppy, breathing difficulty, refusing fluids or markedly fewer wet nappies, or fever persisting beyond 5 days all mean urgent same-day review or 999 as appropriate."},
 {"id": "tb_psoriasis_new", "presenting_complaint": "scaly rash on elbows and knees",
  "brief": "Adult in their early 30s, in person, with about 3 months of well-defined red patches with silvery scale over both elbows and both knees, and a flaky patch at the hairline behind one ear. Itchy but mostly they hate how it looks; a new patch appeared over a scratch on the forearm a few weeks ago. Dad has 'something similar'. Generally well: no joint pain or stiffness, no swollen fingers or toes, nails show a couple of tiny pits but no lifting. Not on any regular medication; started no new products. Exam confirms symmetrical plaques with silvery scale at elbows, knees and the scalp margin, a small linear plaque on the forearm, a few nail pits, no nail lifting, joints normal. Working diagnosis: chronic plaque psoriasis. Plan: explanation that it is common, not contagious, and controllable rather than curable; liberal emollient; a once-daily combined calcipotriol with betamethasone dipropionate ointment for the body plaques for up to 4 weeks then review; a scalp application of the same for the hairline patch; asked to come back in 4 to 6 weeks to check response, and told to mention any future joint pain, swelling or morning stiffness early because of the link with psoriatic arthritis. Safety-net: rapidly worsening widespread redness or feeling systemically unwell needs urgent review."},
 {"id": "tb_lipids_results_review", "presenting_complaint": "cholesterol results review",
  "brief": "Booked follow-up, in person, to go through blood results from an NHS health check. Patient mid-50s, smokes about 10 a day, works a desk job. Results: total cholesterol 6.2, non-HDL 5.0, HDL 1.1, and the computed 10-year cardiovascular risk (QRISK) comes out at around 14 percent. Blood pressure today 134/82. No chest pain, no breathlessness, well in themselves; father had a heart attack in his late 60s. The doctor walks through what the numbers mean, that this is about future risk rather than current illness, and offers atorvastatin 20mg once daily at night as primary prevention alongside lifestyle change, explaining common concerns: muscle aches affect a minority and are checkable, and liver blood tests will be done before starting and repeated at 3 months. The patient decides to start the statin. Bigger single win discussed is smoking: patient is open to quitting and accepts a referral to the local stop-smoking service, plus advice on diet (less saturated fat, more oily fish and fibre) and building walking into the commute. Plan: baseline liver tests on the recent bloods were fine, start atorvastatin 20mg nightly, repeat cholesterol and liver tests in 3 months with a review appointment, stop-smoking referral made today. Safety-net: unexplained widespread muscle pain, weakness or dark urine on the statin means stop and call."},
 {"id": "tb_bppv_dizziness", "presenting_complaint": "episodes of spinning dizziness",
  "brief": "Patient in their mid-60s, in person, with about 2 weeks of brief spinning episodes lasting well under a minute, set off by rolling over in bed, lying back at the hairdresser's, and looking up to a high shelf. Between episodes they feel largely fine, maybe slightly unsteady. No hearing change, no ringing in the ears, no ear pain or discharge. No headache, no double vision, no slurred speech, no weakness or numbness, no falls or blackouts, not on new medication. Sitting and standing blood pressures show no significant drop. Dix-Hallpike test to the right reproduces the spinning with a brief burst of rotatory nystagmus; to the left it is negative. Neurological examination otherwise normal. Working diagnosis: benign paroxysmal positional vertigo of the right posterior canal. Plan: the doctor performs an Epley manoeuvre in clinic, which helps, warns it can take more than one go, and teaches Brandt-Daroff exercises to do at home twice daily for the next week or two; advice to get out of bed in stages and avoid driving during acute spells; no medication needed, and no scan indicated. Review in about 4 weeks if still symptomatic for a repeat Epley. Safety-net: new deafness, continuous rather than positional vertigo, severe headache, or any weakness, speech or vision problem needs urgent assessment."},
 {"id": "tb_menopause_hrt", "presenting_complaint": "hot flushes and disturbed sleep",
  "brief": "49-year-old woman, in person, with around 8 months of hot flushes several times a day, drenching night sweats waking her most nights, poor sleep, irritability and difficulty concentrating at work; periods have spaced out to every 2 to 3 months and are lighter. She asks directly about HRT. No personal or family history of breast cancer, no history of blood clots, no migraine with aura, doesn't smoke, blood pressure today 122/78, BMI in the healthy range, uterus intact (never had a hysterectomy). The doctor explains the perimenopause, and that for her - under 60, within 10 years of the menopause, no contraindications - HRT is a reasonable and effective option: discusses the small increase in breast cancer risk with combined HRT, that through-the-skin oestrogen avoids the clot risk of tablets, and that because she still has a uterus she needs a progestogen alongside oestrogen, cyclically while she is still having periods. Shared decision to start a transdermal oestradiol patch twice weekly with cyclical oral micronised progesterone 12 days per cycle, and to also keep up exercise and cut back evening alcohol which worsens flushes. Contraception is still needed until 2 years after the last period at her age, briefly discussed. Review at 3 months to assess response and side effects. Safety-net: any unscheduled heavy or persistent bleeding on HRT, a painful swollen calf, chest pain or sudden breathlessness needs prompt review."},
 {"id": "tb_ibs_workup", "presenting_complaint": "recurrent abdominal pain and bloating",
  "brief": "Patient in their late 20s, in person, with about 6 months of crampy lower abdominal pain and bloating, worse through the afternoon and after big or rushed meals, eased by opening their bowels; stools alternate between loose and pellety with occasional mucus. Stress at work makes weeks worse. No blood in the stool ever, no black stools, weight steady, appetite fine, never woken from sleep by pain or needing the toilet at night, no fever. No family history of bowel or ovarian cancer or inflammatory bowel disease. Periods regular, no chance of pregnancy at present. Diet heavy on coffee (4 to 5 cups a day) and irregular meal times. Exam: abdomen soft, mild tenderness low down on both sides, no masses, no distension right now. Impression: symptoms fit irritable bowel syndrome, but bloods and a stool test are needed first to be safe - coeliac serology, full blood count, inflammatory markers, and a faecal calprotectin sample to drop off; results expected in about a week. Meanwhile: regular unhurried meals, cut coffee down to 1 to 2 cups, sensible fibre adjustment, and mebeverine 135mg three times a day before meals as needed for the cramps. Review booked once results are back to confirm the diagnosis and discuss longer-term options if needed. Safety-net: any blood in the stool, unintended weight loss, or symptoms waking her at night should prompt an earlier appointment."},
 {"id": "tb_htn_new_diagnosis", "presenting_complaint": "high blood pressure reading",
  "brief": "Patient in their mid-50s, in person, sent in after a pharmacy blood-pressure check found 162/98. Feels completely well: no headaches, no visual disturbance, no chest pain, no breathlessness, no ankle swelling. Doesn't smoke, drinks a couple of beers at weekends, desk job, admits to little exercise and being a bit overweight; their father was on blood-pressure tablets from his 50s. In clinic today, after 5 minutes seated rest, readings are 156/94 and then 154/92 in the other arm. The doctor explains that clinic and pharmacy one-off readings are not enough to diagnose high blood pressure, so the next step is 24-hour ambulatory monitoring - the cuff-and-box fitted appointment gets booked for next week - together with baseline tests to look for causes and end-organ effects: blood tests for kidney function, HbA1c and cholesterol, a urine sample for protein, and an ECG. No medication is started today; whether treatment is needed depends on the ambulatory average. Lifestyle work starts now regardless: cutting salt, more movement (a brisk 30-minute walk most days), modest weight loss, keeping alcohol where it is or lower. Follow-up appointment in 2 to 3 weeks to go through the monitor results and bloods together. Safety-net: chest pain, sudden severe headache, new visual problems or breathlessness in the meantime means being seen urgently, and a home reading persistently above 180/120 means same-day contact."},
 {"id": "tb_gout_first_mtp", "presenting_complaint": "acute painful big toe",
  "brief": "Man in his late 40s, in person, woke 2 days ago with excruciating pain, redness and swelling at the base of the RIGHT big toe; even the duvet resting on it is unbearable and he limped into the surgery in a sandal. Had a milder episode of the same thing about a year ago that settled by itself in a week; never formally diagnosed. No injury. Systemically well: no fever, no shivers, no other joints involved, no skin break or puncture near the toe. Enjoys a few pints of lager most weekends and a fair amount of red meat; on no regular medication, no diuretics. Exam: right first metatarsophalangeal joint hot, red, swollen and exquisitely tender, overlying skin intact and shiny, no spreading redness up the foot, temperature normal in clinic. Working diagnosis: acute gout, clinically typical; the doctor explains a joint infection looks different (fever, unwell, often after a wound) but is the thing to watch for. Plan: naproxen 500mg twice daily with food for up to a week with omeprazole 20mg daily as stomach cover while on it, rest and elevate the foot, ice wrapped in a towel, keep well hydrated, ease off the beer this week. The blood test for uric acid is deliberately deferred until 4 to 6 weeks after the attack settles because levels mislead during a flare; it gets booked for then. If attacks keep recurring, a daily preventer (allopurinol) would be worth discussing at that review. Safety-net: fever, feeling unwell, redness spreading up the foot, or pain not starting to settle within 2 to 3 days on treatment needs same-day review."},
]


def build_prompt(spec, exemplar_json):
    return f"""You are a UK GP and a careful clinical writer, authoring a realistic mock primary-care
consultation for an AI-evaluation research study. The output is used to test whether AI scribe notes
faithfully capture a consultation, so REALISM and INTERNAL CONSISTENCY are paramount.

Here is one EXISTING scenario in the exact schema and style to match (study the transcript voice and
the length):

{exemplar_json}

Now author ONE NEW scenario from this brief:

- id: "{spec['id']}"
- presenting_complaint: "{spec['presenting_complaint']}"
- clinical brief (expand this into a full, natural consultation): {spec['brief']}

REQUIREMENTS:
1. transcript: a natural, realistic GP consultation as alternating turns tagged [doctor] / [patient]
   (and [relative] if relevant), ~1000-1600 words. UK English, UK GP idiom. It must read like a real
   recorded consult: hesitations, the doctor taking a history, examining (or explicitly NOT examining
   if remote), explaining, and a clear plan with safety-netting. NO patient real-world identifiers.
2. Do NOT produce a fact_sheet - output keys are exactly: id, presenting_complaint, transcript.

Output ONLY a single valid JSON object with keys exactly: id, presenting_complaint, transcript.
Escape all newlines inside strings as \\n. No commentary, no markdown fences."""


def main():
    existing = json.load(open(os.path.join(HERE, "authored_scenarios.json")))
    exemplar = dict(next(s for s in existing if s["id"] == EXEMPLAR_ID))
    exemplar.pop("fact_sheet", None)      # the author model never sees a traps schema
    exemplar.pop("_cluster", None)
    exemplar_json = json.dumps(exemplar, ensure_ascii=False, indent=1)

    done = {}
    if os.path.exists(OUTFILE):
        done = {s["id"]: s for s in json.load(open(OUTFILE)) if s.get("transcript")}

    specs = [s for s in SPECS if (not ONLY or s["id"] in ONLY) and s["id"] not in done]
    print(f"Authoring {len(specs)} trap-blind scenarios on {MODEL} (effort={EFFORT}, {WORKERS} workers). "
          f"Exemplar={EXEMPLAR_ID} (fact_sheet stripped). Already done: {len(done)}.")

    faildir = os.path.join(HERE, "master", "trapblind_author_failures")
    os.makedirs(faildir, exist_ok=True)

    def one(spec):
        prompt = build_prompt(spec, exemplar_json)
        obj = claude_json(prompt, model=MODEL, effort=EFFORT, timeout=TIMEOUT, retries=2)
        if not obj or "transcript" not in obj:
            raw = claude(prompt, model=MODEL, effort=EFFORT, timeout=TIMEOUT)
            open(os.path.join(faildir, f"{spec['id']}.txt"), "w").write(raw)
            return {"id": spec["id"], "_error": "parse/shape failure"}
        obj.pop("fact_sheet", None)       # belt and braces: trap-blind scenarios carry NO fact sheet
        obj["id"] = spec["id"]
        obj.setdefault("presenting_complaint", spec["presenting_complaint"])
        return obj

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        out = list(ex.map(one, specs))

    merged = list(done.values()) + out
    order = {s["id"]: i for i, s in enumerate(SPECS)}
    merged.sort(key=lambda s: order.get(s["id"], 99))
    ok = [o for o in merged if "_error" not in o]
    bad = [o for o in merged if "_error" in o]
    json.dump(merged, open(OUTFILE, "w"), ensure_ascii=False, indent=1)
    print(f"OK {len(ok)} | failed {len(bad)} -> {os.path.relpath(OUTFILE, HERE)}")
    for o in ok:
        print(f"  {o['id']:26} transcript {len(o.get('transcript','').split()):4d}w")
    if bad:
        print("FAILED (raw dumped to master/trapblind_author_failures/):", [o["id"] for o in bad])


if __name__ == "__main__":
    main()
