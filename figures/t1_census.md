# T1 - the census

Markdown twin of the paper's `t1_census.tex` float. Both were generated from `t1_census.json` beside this file by `extract_t1_t2.py`, which is not part of this release - see `SOURCES.md`.

| Product | notes | consultations | verified findings | findings per note (sd) | notes with at least one |
|---|---|---|---|---|---|
| Scribe A | 282 | 141 | 128 | 0.454 (1.05) | 60 = 21.3% [16.9, 26.4] |
| Scribe B | 141 | 141 | 215 | 1.525 (2.64) | 57 = 40.4% [32.7, 48.7] |
| Scribe C | 142 | 142 | 275 | 1.937 (3.26) | 60 = 42.3% [34.4, 50.5] |
| Pooled | 565 | 142 | 618 | 1.094 (2.32) | 177 = 31.3% [27.6, 35.3] |

**Caption.** Three commercial ambient scribes wrote notes from the same consultations. Every note went through 12 independent discovery passes, which produced 13,678 candidate findings; the ones the discovering model rated high or medium importance (5,898) went to a panel of two skeptics from different model families with a third family breaking ties, which verified 618, a survival rate of 10.48% [9.72, 11.29]. Findings per note is verified findings divided by that product's notes, with the standard deviation beside it; over all 565 notes the median is 0 and the maximum 18. Intervals are Wilson 95%. Scribe A is generated through an API at two note templates per consultation, so it contributes two notes where the other two contribute one, and its two templates land almost identically (0.447 and 0.461 findings per note), so the gap to the other products is not a property of one template. Notes span four strata: 228 PriMock57, 180 ACI-Bench, 120 authored, 37 trap-blind.

- Scribe A's 282 notes cover 141 consultations, two templates each; the other two products write one note per consultation.
- One trap-blind consultation has no note from Scribe A or Scribe B, so their denominators are 141 consultations rather than 142. FINDINGS 16.5 (line 470) records this gap for Scribe A only.
- Seven notes on disk are excluded from the census and named in master/findings_rates.json: four audio-variant capture probes and three notes whose consultation has no transcript in the master corpus.
- Panel coverage is 1.0, so every rate here is over panel-covered notes.

Sources: FINDINGS 16.1 (lines 397-407), 16.5 (lines 464-483).

Artifacts: `master/findings_master.json`, `master/findings_verified_master.json`, `master/findings_rates.json`.
