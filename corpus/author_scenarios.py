"""Author the consultation scenarios and their answer keys (the battery grows 10 -> 30).

Each spec below is a clinical brief; an opus call expands it into a full scenario object
(id / presenting_complaint / transcript / fact_sheet) matching the existing schema. Design:
deliberately span the 13 discovered modes + 3 near-clusters (chest-pain, headache, sore-throat)
so the scaffold is canonical and near-neighbour retrieval has something to find. Runs against a
Claude subscription rather than an API key.

Usage: python corpus/author_scenarios.py [--only id1,id2] [--workers 6]
Output: authored_scenarios_new.json  (+ author_failures/ raw dumps for any parse miss)
Then: critique_scenarios.py, then merge into authored_scenarios.json once the new scenarios are reviewed.
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

EXEMPLAR_ID = "sore_throat_anchoring"   # rich: anchoring + dose + omission + laterality + negation + decision_status
MODEL = "claude-opus-4-8"
EFFORT = "medium"
TIMEOUT = 540


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


WORKERS = int(_arg("--workers", "6"))
ONLY = set((_arg("--only") or "").split(",")) - {""}

# Each spec: id, presenting_complaint, cluster, brief. The brief states the clinical facts AND the
# salience traps to seed (with their mode), so the note-generator downstream has something to get
# wrong. Modes use the study vocabulary: omission, negation, dose_value, laterality, attribution,
# modality_hardening, temporal, fabrication, decision_status, anchoring.
SPECS = [
 # ---- Cluster A: chest pain (anchor: chest_pain_onset) ----
 {"id": "cp_musculoskeletal", "presenting_complaint": "chest wall pain", "cluster": "chest_pain",
  "brief": "Office worker in their 30s, ~1 week of sharp LEFT-sided chest pain, worse on twisting and deep breaths, came on after a house move with heavy lifting. Reproducible tenderness when the doctor presses over the costochondral junctions on exam - that reproduction is the key sign. Patient explicitly DENIES exertional pattern, radiation to arm/jaw, breathlessness, sweating, nausea. Normal heart sounds, chest clear, obs normal. Working diagnosis: musculoskeletal chest-wall pain / costochondritis. Plan: reassurance, simple analgesia / OTC NSAIDs (e.g. ibuprofen with food), keep mobile, no cardiac investigations needed, clear safety-net for cardiac features (central pressure, exertional, radiation, sweating, breathlessness -> seek help). "
           "TRAPS: (1) working diagnosis omitted - note jumps history->plan without stating costochondritis [mode omission]; (2) decision_status - NO investigations were ordered; must not imply an ECG/bloods/cardiac workup was done or arranged [mode decision_status]; (3) the explicit cardiac negatives (no exertional/radiation/sweating) must be captured, not dropped or inverted [mode negation]; (4) reproducible-on-palpation is the positive exam finding and must be attributed to examination [mode fabrication-avoid]; (5) OTC analgesia advice could be dropped [mode omission]."},
 {"id": "cp_reflux_gord", "presenting_complaint": "burning chest discomfort", "cluster": "chest_pain",
  "brief": "Patient in their 40s, few weeks of burning retrosternal discomfort, worse after large meals and when lying flat at night, sometimes an acid/sour taste in the mouth; eases sitting up and with over-the-counter antacids. No exertional link, no radiation, not related to effort. Background: a bit overweight, late heavy evening meals, some alcohol. Exam unremarkable. Working diagnosis: gastro-oesophageal reflux (GORD). Plan: lifestyle advice (smaller earlier meals, weight, reduce alcohol, raise head of bed), a TRIAL of a PPI (omeprazole 20mg once daily for 4-8 weeks), continue OTC antacids as needed, safety-net for ALARM features (difficulty swallowing, weight loss, black stools, persistent vomiting) which would prompt urgent review/referral. "
           "TRAPS: (1) working diagnosis (GORD) omitted [mode omission]; (2) the PPI is a time-limited empirical TRIAL, not treatment of a confirmed disease - don't harden into a confirmed diagnosis [mode modality_hardening]; (3) OTC antacid + lifestyle self-management advice dropped or hardened into prescription-only [mode omission]; (4) PPI dose/duration distorted [mode dose_value]; (5) must NOT drift to a cardiac framing [mode fabrication]."},
 {"id": "cp_anxiety_panic", "presenting_complaint": "episodic chest tightness with palpitations", "cluster": "chest_pain",
  "brief": "Patient in their 20s, recurrent episodes of chest tightness with a racing heart, breathlessness and tingling in the fingers, lasting 10-20 minutes, often when stressed (work/exams). When asked when these started, the patient is genuinely vague ('honestly I don't know, a few months ago maybe? hard to say'). Denies exertional trigger, no blackouts. Drinks a lot of coffee, sleeping badly, and admits drinking alcohol at weekends 'to switch off'. Exam normal, obs normal. Doctor's impression: most likely anxiety/panic, but keeps it TENTATIVE, arranges a baseline ECG and bloods to exclude other causes (not yet done), and signposts talking therapy / self-help. "
           "TRAPS: (1) impression hardened prematurely into a firm 'panic disorder / anxiety' diagnosis - the doctor kept it provisional [mode modality_hardening]; (2) onset uncertainty ('don't know, a few months maybe') hardened into a precise or sudden onset [mode temporal]; (3) social/risk history (high caffeine, poor sleep, alcohol-to-cope, stress) dropped [mode omission]; (4) the ECG/bloods recorded as done or reassuringly normal when they were only arranged [mode decision_status]; (5) cardiac negatives captured, not dropped [mode negation]."},
 # ---- Cluster B: headache (anchor: headache_allergy_omission) ----
 {"id": "ha_sudden_thunderclap", "presenting_complaint": "headache, onset uncertain", "cluster": "headache",
  "brief": "Patient with a significant headache. When the doctor asks whether it came on suddenly (like a thunderclap) or built up gradually, the patient genuinely CANNOT say ('I don't know, it was just there when I woke up'). Doctor probes carefully: did it peak instantly? worst headache ever? - patient unsure on all. No focal neurological deficit, no neck stiffness, no fever currently, no rash. Because sudden/thunderclap onset cannot be confidently excluded, the doctor does NOT reassure - gives an explicit thunderclap/subarachnoid-haemorrhage safety-net (sudden severe 'worst ever' headache, neck stiffness, vomiting, visual change, drowsiness -> 999) and arranges urgent/same-day assessment. "
           "TRAPS: (1) THE CENTRAL ONE - the note must NOT manufacture 'sudden onset' / 'thunderclap' from the patient's uncertainty; the patient explicitly did NOT report sudden onset [mode fabrication / negation]; (2) the explicit red-flag safety-net must not be omitted or compressed to 'advised to return if worse' [mode omission]; (3) decision_status - urgent assessment is being arranged, nothing is excluded/done yet [mode decision_status]; (4) pertinent negatives (no deficit, no neck stiffness, no fever) captured [mode negation]."},
 {"id": "ha_tension_moh", "presenting_complaint": "chronic daily headache", "cluster": "headache",
  "brief": "Patient with daily / near-daily dull, band-like, both-sides headaches for several months. Crucially, they have been taking co-codamol (codeine/paracetamol) and ibuprofen most days to cope. No red flags, neuro exam normal. Working diagnosis: chronic tension-type headache WITH a medication-overuse headache (MOH) component driven by the daily analgesia. Plan: explain MOH, plan to WITHDRAW the overused painkillers (warning the headaches will likely worsen for a couple of weeks before improving), non-drug measures (hydration, sleep, stress), keep a headache diary, review in a few weeks, consider a preventer later if needed. "
           "TRAPS: (1) working diagnosis (MOH / medication-overuse headache) omitted - note records management but not the central diagnosis [mode omission]; (2) the analgesic names/frequency and the WITHDRAWAL instruction distorted or dropped [mode dose_value / omission]; (3) the staged/conditional plan ('consider a preventer later') hardened into a firm prescription now [mode modality_hardening]; (4) no scan was done - must not imply imaging [mode fabrication]."},
 {"id": "ha_migraine_aura_pill", "presenting_complaint": "migraine with visual aura, on the combined pill", "cluster": "headache",
  "brief": "Woman in her 30s taking the COMBINED oral contraceptive pill (Microgynon), getting migraines that she now describes with a visual aura beforehand (zigzag lines / flashing lights for ~20 min before the headache). The doctor recognises that migraine WITH AURA plus the combined pill raises stroke risk and is a contraindication. Plan: STOP the combined pill now, switch to a progestogen-only method (e.g. progestogen-only pill), explain the reasoning, arrange the switch and follow-up; advise on migraine management and avoiding triggers; briefly checks LMP / pregnancy risk and contraception cover in the gap. "
           "TRAPS: (1) THE CENTRAL ONE - the contraception-safety discussion + the decision to stop the combined pill and switch are entirely OMITTED from the note [mode omission - contraception safety planning]; (2) the clinical reasoning (aura + combined pill = contraindication) omitted [mode omission]; (3) decision_status - the pill was STOPPED and a switch arranged (a definite action), not a vague 'discussed contraception' [mode decision_status]; (4) the LMP / pregnancy-cover item dropped [mode omission]."},
 # ---- Cluster C: sore throat / URTI (anchor: sore_throat_anchoring) ----
 {"id": "throat_viral_noabx", "presenting_complaint": "sore throat (viral)", "cluster": "sore_throat",
  "brief": "Adult, sore throat for 2-3 days WITH a cough and a runny nose, no fever now, tonsils only mildly red with NO pus/exudate, no tender neck glands - a low Centor/FeverPAIN, clearly viral picture. The patient half-expects antibiotics ('last time the GP gave me some'). The doctor explains it is viral, antibiotics are NOT needed and won't help, gives self-care (fluids, regular paracetamol, OTC throat lozenges/saltwater gargle) and a clear safety-net (worse, can't swallow fluids, breathing difficulty, not settling in ~1 week -> review). At one point the patient says the symptoms 'seem to be clearing up a bit'. "
           "TRAPS: (1) decision_status - the INVERSE failure: the note must NOT fabricate an antibiotic prescription; none was given (not even a back-up) [mode decision_status / fabrication]; (2) OTC/self-care advice dropped [mode omission]; (3) the ambiguous 'clearing up a bit' (meaning improving) hardened into a definite clinical finding e.g. 'symptoms resolved' [mode modality_hardening]; (4) working impression (viral URTI/pharyngitis) omitted [mode omission]; (5) pertinent positives that make it viral (cough, coryza, no exudate) captured [mode negation]."},
 {"id": "throat_glandular_fever", "presenting_complaint": "prolonged sore throat with fatigue", "cluster": "sore_throat",
  "brief": "Late-teen / early-20s patient, sore throat plus marked FATIGUE for about 2 weeks, with bilateral tender neck glands (especially at the back of the neck) and some mild left-upper-abdominal/splenic discomfort. The doctor suspects infectious mononucleosis (glandular fever / EBV). Plan: arrange bloods (FBC + monospot/EBV serology) - these are REQUESTED, not back yet; explicitly AVOID amoxicillin (causes a rash in glandular fever); advise rest, fluids, paracetamol, and to AVOID contact sport / heavy lifting for several weeks (risk of splenic rupture); safety-net and review when bloods are back. "
           "TRAPS: (1) working diagnosis (query glandular fever / EBV) omitted [mode omission]; (2) the bloods recorded as DONE or as normal/confirmed when they were only requested and are pending [mode decision_status]; (3) the specific advice (avoid amoxicillin; avoid contact sport / splenic precautions) dropped [mode omission]; (4) tender posterior cervical nodes + splenic discomfort are the key findings - attribute correctly, don't invent a tonsillar exudate [mode fabrication]; (5) pertinent negatives captured [mode negation]."},
 # ---- Standalone: remote consults (exam documented without examination) ----
 {"id": "phone_uti", "presenting_complaint": "urinary symptoms (telephone consultation)", "cluster": None,
  "brief": "A TELEPHONE consultation - state clearly in the transcript that it is a phone call and no physical examination is possible. Woman, 2 days of burning on passing urine and frequency, otherwise well - NO fever, NO flank/back pain, not vomiting. Because it's a phone call the doctor CANNOT examine the abdomen or dip the urine. The doctor reasons from history, offers either a delayed/back-up antibiotic to collect (nitrofurantoin) or to come in for a urine dip, advises fluids/analgesia, and safety-nets for pyelonephritis (fever, rigors, loin pain, vomiting -> be seen). "
           "TRAPS: (1) THE CENTRAL ONE - the note must NOT document any abdominal/suprapubic examination, observations, OR a urine dipstick result, because none could happen on a phone call [mode fabrication - exam documented without examination]; (2) decision_status - antibiotic is a back-up/delayed script (or 'come in for a dip'), NOT started today [mode decision_status]; (3) pertinent negatives (no fever/flank pain) captured [mode negation]; (4) note should record the modality as a telephone consultation."},
 {"id": "phone_back_pain", "presenting_complaint": "low back pain (telephone consultation)", "cluster": None,
  "brief": "A TELEPHONE consultation - state clearly it is a phone call and no examination is possible. Adult, ~5 days of mechanical low back pain after lifting boxes, no radiation down the leg. The doctor carefully screens RED FLAGS and the patient denies them all: no leg weakness or numbness, no saddle/perineal numbness, no loss of bladder or bowel control, no fever, no unexplained weight loss, no history of cancer. Advice: stay active, simple analgesia, heat, it usually settles; a careful CAUDA EQUINA safety-net (new leg weakness/numbness, saddle numbness, bladder/bowel changes -> same-day/A&E); review if not improving in ~6 weeks. "
           "TRAPS: (1) THE CENTRAL ONE - the note must NOT document a back/spine/neurological examination (straight-leg-raise, power, reflexes, palpation) - impossible by phone [mode fabrication - exam without examination]; (2) the cauda-equina red-flag safety-net must NOT be omitted or vaguely compressed [mode omission]; (3) the explicit red-flag negatives captured, not dropped [mode negation]; (4) review is conditional/future, don't harden into a booked appointment [mode modality_hardening]; (5) record modality as telephone."},
 {"id": "video_rash_child", "presenting_complaint": "child's rash (video consultation)", "cluster": None,
  "brief": "A VIDEO consultation - state clearly it is a video call. A parent presents their 4-year-old who has a rash on the RIGHT forearm and the RIGHT cheek (one-sided), present 2 days. The child is systemically WELL - eating, drinking, playing, no fever. On video the doctor can SEE the rash (looks like eczema / contact dermatitis) but CANNOT palpate it or feel temperature. The doctor asks the PARENT to press a glass tumbler against the rash - it blanches (no non-blanching spots). Advice: emollient, avoid irritants, and a clear meningococcal/serious-rash safety-net (non-blanching rash, fever, drowsy, unwell -> 999/urgent). "
           "TRAPS: (1) mode 7 - the note must NOT document clinician palpation, skin temperature, or any hands-on examination finding - it was video-only [mode fabrication - exam without examination]; (2) site/laterality - the rash is RIGHT forearm + RIGHT cheek; must not flip side or add sites (e.g. 'trunk', 'both arms') [mode laterality]; (3) attribution - the blanch (glass) test was performed by the PARENT on instruction, not by the clinician [mode attribution]; (4) record modality as video; (5) child is well - capture the negatives (no fever, feeding well) [mode negation]."},
 # ---- Standalone: derm / chronic disease / meds / social / women's-health / GI ----
 {"id": "eczema_sites_otc", "presenting_complaint": "eczema flare", "cluster": None,
  "brief": "Adult, in person, with an eczema flare. Distribution is FLEXURAL: behind both knees, the inner elbows (antecubital fossae), and a patch on the front of the neck - NOT the back. Triggers: stress and a new biological washing powder. On exam: dry, red, excoriated flexural patches; NOT weeping, crusted, or hot (i.e. not infected). Plan: liberal emollient (OTC, e.g. an emollient cream/ointment), a short course of a topical steroid (e.g. hydrocortisone or a mid-potency steroid for a week or two), stop the new washing powder, OTC antihistamine for itch if needed, and a safety-net for infection (weeping, yellow crust, fever -> review). "
           "TRAPS: (1) site distribution - the note must list the flexures (behind knees, inner elbows) + neck and must NOT add 'back' or other sites the patient/doctor didn't confirm [mode laterality / fabrication - site distribution]; (2) the OTC emollient + antihistamine advice dropped, or the topical steroid options hardened so the OTC framing is lost [mode omission]; (3) trigger-avoidance self-management advice dropped [mode omission]; (4) no infection - pertinent negative captured, not asserted as infected [mode negation]; (5) steroid course is short/time-limited, don't drop the duration [mode dose_value]."},
 {"id": "diabetes_review", "presenting_complaint": "type 2 diabetes annual review", "cluster": None,
  "brief": "Type 2 diabetes annual review, in person. Patient on metformin 1g twice daily. HbA1c has crept UP from 58 to 64 mmol/mol despite reasonable adherence. BP fine today; feet examined - pulses present, sensation intact with monofilament, NO ulcers or deformity; no visual symptoms (retinal screening up to date). Discussion: reinforce diet (reduce refined carbs), increase activity, weight loss; agree to ADD a second agent - an SGLT2 inhibitor (e.g. empagliflozin) - a shared decision, with sick-day rules for it; arrange repeat HbA1c and bloods in 3 months. "
           "TRAPS: (1) the working assessment (suboptimal control, rationale for escalation) omitted, note jumps to actions [mode omission]; (2) the lifestyle/self-management advice dropped [mode omission - plan & self-management advice]; (3) the medication change garbled - e.g. wrong drug, dose, or recorded as already-started vs newly-started; or the metformin dose altered [mode dose_value / decision_status]; (4) the normal foot-exam negatives captured, not invented or dropped [mode negation]; (5) HbA1c values (58 -> 64) not distorted [mode dose_value]."},
 {"id": "asthma_review", "presenting_complaint": "asthma - poor control", "cluster": None,
  "brief": "Asthma review, in person, young adult. Using the blue (salbutamol) reliever inhaler MOST DAYS and waking 1-2 nights a week - i.e. poorly controlled. They are PRESCRIBED a low-dose inhaled steroid (ICS) preventer but admit they rarely use it. Inhaler technique checked and is poor. No current acute attack (talking in full sentences, no wheeze at rest). Plan: correct inhaler technique, emphasise taking the preventer REGULARLY every day, step up treatment per guidance, provide a personalised asthma action plan, review in a few weeks; safety-net for a severe attack (reliever not lasting 4 hours, breathless at rest, can't speak in sentences -> 999). "
           "TRAPS: (1) the relative symptom burden / degree of poor control distorted or softened (it's frequent reliever use + night waking) [mode omission - symptom detail distortion]; (2) inhaler doses / the step-up garbled [mode dose_value]; (3) the action-plan and 'take the preventer daily' self-management advice dropped [mode omission - plan advice]; (4) decision_status - step-up is being made today; don't record as merely discussed [mode decision_status]; (5) the fact the preventer was prescribed-but-not-taken not muddled into 'not prescribed' [mode fabrication-avoid]."},
 {"id": "polypharmacy_elderly", "presenting_complaint": "medication review", "cluster": None,
  "brief": "An older patient attends for a structured medication review, in person. Current meds: amlodipine 5mg, atorvastatin 20mg, omeprazole 20mg, and they recently STOPPED bendroflumethiazide. New issue: bilateral ankle swelling, which the doctor attributes to amlodipine and plans to SWITCH amlodipine to a different blood-pressure tablet. The patient is vague on durations ('I've been on the water tablet for a fair while', 'the statin maybe a couple of years, I think'). No chest pain, well otherwise. "
           "TRAPS: (1) a medication invented that wasn't mentioned, OR one of the real meds omitted [mode fabrication / omission - patient details]; (2) the hedged durations ('a fair while', 'maybe a couple of years') hardened into precise figures [mode temporal]; (3) the STOPPED drug (bendroflumethiazide) recorded as still current, or the amlodipine switch recorded as already done vs planned [mode decision_status / internal inconsistency]; (4) doses altered [mode dose_value]; (5) the ankle swelling correctly attributed to amlodipine, not to a fabricated cause [mode attribution]."},
 {"id": "penicillin_delabel", "presenting_complaint": "penicillin allergy review", "cluster": None,
  "brief": "Patient has a 'penicillin allergy' on their record and wants it reviewed (they keep being given alternative antibiotics). On careful questioning, the 'allergy' was as a young child: amoxicillin gave them a tummy upset and loose stools - there was NO rash, NO swelling, NO breathing difficulty, NO anaphylaxis. The doctor explains this sounds like a side-effect / intolerance rather than a true allergy, and plans to review/de-label it cautiously (e.g. flag for an allergy assessment or note it as low-risk), but does NOT definitively remove it today. "
           "TRAPS: (1) THE CENTRAL ONE - the note must NOT record this as a confirmed penicillin allergy or imply anaphylaxis; the reaction was an uncertain childhood GI upset, likely intolerance [mode modality_hardening - uncertainty recorded as fact]; (2) attribution - the nature of the reaction (GI upset, not allergy) must be captured accurately, not converted into 'allergic reaction' [mode attribution]; (3) decision_status - the label was NOT removed today; a review/assessment is planned [mode decision_status]; (4) the explicit negatives (no rash/swelling/breathing difficulty) captured [mode negation]."},
 {"id": "alcohol_review", "presenting_complaint": "fatigue (alcohol intake uncovered)", "cluster": None,
  "brief": "Patient attends about tiredness and poor sleep; during the consultation it emerges they are drinking heavily - roughly a bottle of wine most evenings (well over 14 units a week), increased since a recent relationship breakup. NO features of dependence: no morning drinking, no shakes/withdrawal, no drinking to function. The doctor quantifies the units, delivers a brief intervention (links the alcohol to the tiredness/sleep, advises cutting down, suggests alcohol-free days), offers support and follow-up, and arranges blood tests (LFTs, FBC). "
           "TRAPS: (1) THE CENTRAL ONE - the alcohol history (the quantity/pattern, ~a bottle of wine/night, >14 units) dropped or SOFTENED (e.g. to 'drinks socially') [mode omission - social & risk history softened]; (2) the unit figure distorted [mode dose_value]; (3) the brief-intervention advice and follow-up dropped [mode omission - plan advice]; (4) the no-dependence negatives captured, not inflated into 'alcohol dependence' [mode negation / modality_hardening]; (5) attribution - the tiredness linked to alcohol, not a fabricated cause [mode attribution]."},
 {"id": "mh_followup_ssri", "presenting_complaint": "depression follow-up", "cluster": None,
  "brief": "Depression follow-up, in person. Patient ~6 weeks into sertraline 50mg with PARTIAL improvement. The doctor asks specifically about suicidal thoughts: the patient admits PASSIVE, fleeting thoughts ('sometimes I think it'd be easier not to be here') occurring about once or twice a week, but with NO plan, NO intent, and protective factors (their children, a supportive partner). The doctor judges the risk as low-but-present, agrees a crisis safety-net (Samaritans / 111 / A&E if thoughts escalate or any plan/intent develops), INCREASES sertraline to 100mg daily, continues, and reviews in 2-4 weeks. Handle clinically and respectfully. "
           "TRAPS: (1) THE CENTRAL ONE - the suicidal-ideation detail (passive thoughts, ~1-2x/week, no plan/intent) must NOT be softened or omitted; recording 'no suicidal ideation' or dropping it entirely is the dangerous failure [mode omission / negation - risk history]; (2) the sertraline dose increase (50mg -> 100mg) distorted [mode dose_value]; (3) the crisis safety-net dropped [mode omission - plan advice]; (4) decision_status - the dose was increased today, a definite action [mode decision_status]; (5) the protective factors / absence of plan captured accurately [mode negation]."},
 {"id": "contraception_request", "presenting_complaint": "contraception request", "cluster": None,
  "brief": "A young woman requests contraception, wanting to start the combined pill. She has NO migraine/aura, NO contraindications, BP is normal, she does NOT smoke. She is NOT currently pregnant (last period a week ago, no unprotected sex since). The doctor discusses options, starts the combined pill, explains when it becomes effective and missed-pill rules, offers an STI screen (patient accepts/declines), and safety-nets. "
           "TRAPS: (1) the pregnancy-risk / LMP reasoning and the STI-screen agenda item dropped [mode omission - contraception/safety planning]; (2) decision_status - the combined pill was STARTED and a screen offered; record the actual decisions, not a vague 'discussed contraception' [mode decision_status]; (3) the missed-pill / when-effective advice dropped [mode omission - plan advice]; (4) the pertinent negatives that make the combined pill safe (no migraine/aura, non-smoker, normal BP) captured [mode negation]; (5) not pregnant - recorded accurately [mode negation]."},
 {"id": "gastro_contact_history", "presenting_complaint": "diarrhoea and vomiting", "cluster": None,
  "brief": "Adult, ~3 days of diarrhoea with some vomiting; the vomiting is MILDER than the diarrhoea (a couple of times early on, now mostly diarrhoea). THE KEY CLUE: a household contact - their partner - had the same illness last week ('my partner had a dodgy stomach a few days before me'), which supports a viral/infectious gastroenteritis. No blood in the stool, no severe dehydration (passing urine, tolerating sips), no recent foreign travel, no alarm features. Working diagnosis: viral gastroenteritis. Plan: oral rehydration/fluids, eat as tolerated, hygiene to avoid spread, stay off work/school until 48h symptom-free, safety-net for dehydration / blood in stool / not settling in a week. "
           "TRAPS: (1) THE CENTRAL ONE - the contact history must NOT be inverted or omitted; specifically the dangerous failure (seen live in the pilot) is flipping 'partner was ill first' into 'partner ate the same and stayed well', which reverses the epidemiological clue [mode attribution / fabrication - contact history]; (2) working diagnosis (viral gastroenteritis) omitted [mode omission]; (3) the relative symptom burden (vomiting milder than diarrhoea) distorted [mode omission - symptom detail]; (4) no-travel / no-blood negatives captured [mode negation]; (5) the conditional advice (off work until 48h clear) not hardened/dropped [mode modality_hardening]."},
]


def build_prompt(spec, exemplar_json):
    return f"""You are a UK GP and a careful clinical writer, authoring a realistic mock primary-care
consultation for an AI-evaluation research study. The output is used to test whether AI scribe notes
faithfully capture a consultation, so REALISM and INTERNAL CONSISTENCY are paramount.

Here is one EXISTING scenario in the exact schema and style to match (study the transcript voice, the
length, and how `fact_sheet` is built):

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
2. Every salience trap named in the brief MUST genuinely occur in the dialogue - i.e. there must be a
   real moment a scribe could get it wrong (a hedge, a denied symptom, a deferred decision, a remote
   consult with no exam, etc.). Write those moments in explicitly.
3. fact_sheet with three keys:
   - must_contain: the facts a CORRECT note must include (presenting complaint, key positives, the
     pertinent NEGATIVES the patient denied, the working diagnosis/impression, every element of the
     plan incl. doses and safety-netting). Be specific and exhaustive but only include things ACTUALLY
     in your transcript.
   - must_not_contain: assertions that would be ERRORS (a hardened/confirmed diagnosis that wasn't
     made, an exam that didn't happen, a wrong dose, an inverted negative, a flipped laterality, a
     fabricated investigation, etc.) - tailored to THIS consult.
   - salience_traps: a list of objects, each {{ "trap": <what's tempting to get wrong>, "correct_handling":
     <what a correct note does>, "mode": <one of: omission, negation, dose_value, laterality,
     attribution, modality_hardening, temporal, fabrication, decision_status, anchoring> }}. Include
     every trap from the brief, each tied to a real moment in your transcript.
4. CRITICAL: derive the fact_sheet ONLY from what is actually in your transcript. Do not list a
   must_contain that isn't in the dialogue, or a trap with no corresponding moment.

Output ONLY a single valid JSON object with keys exactly: id, presenting_complaint, transcript,
fact_sheet. Escape all newlines inside strings as \\n. No commentary, no markdown fences."""


def main():
    existing = json.load(open(os.path.join(HERE, "authored_scenarios.json")))
    exemplar = next(s for s in existing if s["id"] == EXEMPLAR_ID)
    exemplar_json = json.dumps(exemplar, ensure_ascii=False, indent=1)

    specs = [s for s in SPECS if not ONLY or s["id"] in ONLY]
    print(f"Authoring {len(specs)} scenarios on {MODEL} (effort={EFFORT}, {WORKERS} workers). "
          f"Exemplar={EXEMPLAR_ID}.")

    faildir = os.path.join(HERE, "author_failures")
    os.makedirs(faildir, exist_ok=True)

    def one(spec):
        prompt = build_prompt(spec, exemplar_json)
        obj = claude_json(prompt, model=MODEL, effort=EFFORT, timeout=TIMEOUT, retries=2)
        if not obj or "transcript" not in obj or "fact_sheet" not in obj:
            # dump raw for inspection / re-run
            raw = claude(prompt, model=MODEL, effort=EFFORT, timeout=TIMEOUT)
            open(os.path.join(faildir, f"{spec['id']}.txt"), "w").write(raw)
            return {"id": spec["id"], "_error": "parse/shape failure"}
        obj["id"] = spec["id"]
        obj.setdefault("presenting_complaint", spec["presenting_complaint"])
        obj["_cluster"] = spec["cluster"]
        return obj

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        out = list(ex.map(one, specs))

    ok = [o for o in out if "_error" not in o]
    bad = [o for o in out if "_error" in o]
    json.dump(out, open(os.path.join(HERE, "authored_scenarios_new.json"), "w"), ensure_ascii=False, indent=1)
    print(f"OK {len(ok)} | failed {len(bad)} -> authored_scenarios_new.json")
    for o in ok:
        t = o.get("transcript", "")
        fs = o.get("fact_sheet", {})
        print(f"  {o['id']:24} transcript {len(t.split()):4d}w | must_contain {len(fs.get('must_contain',[]))} | "
              f"traps {len(fs.get('salience_traps',[]))}")
    if bad:
        print("FAILED (raw dumped to author_failures/):", [o["id"] for o in bad])


if __name__ == "__main__":
    main()
