# T2 - the 17 subcategories

Markdown twin of the paper's `t2_clusters.tex` float. Both were generated from `t2_clusters.json` beside this file by `extract_t1_t2.py`, which is not part of this release - see `SOURCES.md`.

| Subcategory | n | A | B | C | consultations | high importance | frame placement | example |
|---|---|---|---|---|---|---|---|---|
| Allergy status and medication list omissions | 111 | 19 | 40 | 52 | 33 | 71 | omission | Allergy status was asked and answered in the consultation, and the note documents none while recording a prescription. [PARAPHRASED, Scribe B] |
| Invented patient identity: name and sex | 93 | 10 | 45 | 38 | 34 | 49 | addition / fabrication | The patient's name is inaudible in the transcript, and the note supplies one that is a chemotherapy brand name. [PARAPHRASED, Scribe C] |
| Stated working diagnosis dropped from note | 53 | 53 | 0 | 0 | 22 | 52 | omission | The clinician tells the patient the symptoms suggest gastroenteritis; the note carries the advice that follows from it and no impression at all. [PARAPHRASED, Scribe A] |
| Relative timing converted to invented calendar dates | 44 | 0 | 4 | 40 | 17 | 2 | wrong output / timing | The transcript says 'yesterday' and the note prints a specific calendar date. [PARAPHRASED, Scribe C] |
| Remote consult history written as objective exam | 42 | 0 | 11 | 31 | 10 | 33 | addition / examination provenance | A telephone consultation in which no examination took place, written up with examination findings as observed. [PARAPHRASED, Scribe C] |
| Planned investigations documented as already done | 32 | 4 | 15 | 13 | 11 | 21 | wrong output / modality hardening | The transcript has an ECG still to be arranged with reception; the note records it as performed in clinic today. [PARAPHRASED, Scribe C] |
| Fabricated/flipped symptom qualifiers (sleep, timing, location) | 28 | 0 | 14 | 14 | 12 | 11 | addition / fabrication | The note reports the symptoms as affecting sleep, which the consultation never mentions. [PARAPHRASED, Scribe B] |
| Diabetes lab values distorted and re-provenanced | 21 | 6 | 11 | 4 | 4 | 6 | wrong output / dose value | The transcript gives a recalled HbA1c of about 7.7 per cent and the note records 7.0 per cent. [PARAPHRASED, Scribe B] |
| Fabricated calendar year for vague onset date | 21 | 0 | 12 | 9 | 2 | 0 | wrong output / timing | 'Last December' becomes December 2025, a year the consultation never states and the patient's stated age contradicts. [PARAPHRASED, Scribe C] |
| Phantom counselling and unasked negative history | 20 | 0 | 11 | 9 | 8 | 4 | addition / fabrication | The clinician asks whether a previous doctor discussed the risks of aspirin; the note records that the advice was given in this consultation. [PARAPHRASED, Scribe C] |
| Family-history relative and lineage misassignment | 18 | 0 | 0 | 18 | 1 | 4 | wrong output / attribution | A cancer the transcript places in the maternal grandmother is recorded against the patient's mother. [PARAPHRASED, Scribe C] |
| Dropped work-absence and sick-note plan elements | 16 | 11 | 5 | 0 | 8 | 3 | omission | The clinician tells the patient not to go into work until this is sorted out, and the plan omits it. [PARAPHRASED, Scribe A] |
| False denials of swelling/redness | 16 | 0 | 6 | 10 | 4 | 7 | wrong output / negation | The patient reports inflamed knees and the examination finds swelling and redness; the note records both as denied. [PARAPHRASED, Scribe C] |
| Alternative options recorded as issued prescriptions/orders | 14 | 6 | 0 | 8 | 4 | 13 | wrong output / modality hardening | A choice offered between two antibiotics is listed as two issued prescriptions. [PARAPHRASED, Scribe C] |
| Unmeasured or corrupted values entered as objective vitals | 13 | 0 | 1 | 12 | 4 | 7 | addition / examination provenance | A temperature the patient measured at home, 37°C, is entered in the note's vitals as an implausible 13.7°C. [PARAPHRASED, Scribe C] |
| Retracted device captured as delivered care | 11 | 7 | 4 | 0 | 1 | 5 | no published category | The clinician retracts the thumb spica in the next breath and substitutes a wrist brace; the note records the spica as applied. [PARAPHRASED, Scribe A] |
| Red-flag safety-netting downgraded to routine follow-up | 10 | 6 | 1 | 3 | 4 | 7 | wrong output / negation | Sudden weight gain is given as a red flag warranting urgent contact, and the note attaches it to the routine booking instruction. [PARAPHRASED, Scribe A] |

**Caption.** The first two tiers of the taxonomy were fixed before the findings were sorted; the third is discovered from the data. We embed each verified finding's description, project it with UMAP and cluster by density with HDBSCAN, then label each cluster in one call (full recipe and parameters in Appendix B). 563 of the 618 verified findings fall in a cluster and 55 are left unassigned. The count carries measured instability: re-running the same recipe under UMAP seeds 42, 43 and 44 gives 17, 20 and 21 clusters with largely the same boundaries (pairwise adjusted Rand index 0.667, 0.707 and 0.808, mean 0.727, where 1.0 is identical groupings), so the count is stable only to within a few. Columns A, B and C count findings rather than notes; high importance is the verification panel's own rating, out of the cluster's n. Frame placement is one call for the whole cluster, so cluster sizes do not sum to the top-level class totals in Section 3; 16 of the 17 clusters place under the fixed frame and one has no category in our scheme, whose classes come from the published taxonomies. Examples are paraphrased from our own finding descriptions rather than quoted from the notes.

- A finding's own frame category and its cluster's placement are different measurements and must not be read as one column beside the other: omission is the placement of 3 of the 17 clusters, holding 180 findings between them, against the 143 findings whose own category is omission.
- The subcategory names are the labels the clustering step wrote, kept verbatim.
- Each example carries the product letter and the finding's reference in the released judgements, so it can be traced; the wording is ours, not the note's.

Sources: FINDINGS 16.4 (lines 448-462), 16.1 (lines 397-407) for the panel's importance ratings.

Artifacts: `master/findings_clusters.json`, `master/findings_cluster_sweep.json`, `master/findings_verified_master.json`.
