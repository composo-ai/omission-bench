# T4 - the eight judge designs

Markdown twin of the paper's `t4_grid.tex` float. Both were generated from `t4_grid.json` beside this file by `extract_t4_t5.py`, which is not part of this release - see `SOURCES.md`. Lands at Section 6, "Presence is caught at near-ceiling, absence at close to chance".

| Design | paired commissions | paired omissions | det. at own rule | false alarms | z | AUC, omissions | swept det. at FA <= 10% |
|---|---|---|---|---|---|---|---|
| Faithfulness only, yes/no, one sample (`F-bin-k1`) | 0.875 | 0.500 | 9.6% | 8.9% | 0.34 | 0.503 | - |
| Faithfulness only, yes/no, eight samples (`F-bin-k8`) | 0.904 | 0.509 | 9.6% | 7.4% | 1.15 | 0.512 | - |
| Faithfulness and completeness, yes/no, one sample (`FC-bin-k1`) | 0.792 | 0.539 | 40.3% | 31.8% | 2.71 | 0.542 | - |
| Faithfulness and completeness, yes/no, eight samples (`FC-bin-k8`) | 0.869 | 0.591 | 42.3% | 32.1% | 3.25 | 0.562 | - |
| Faithfulness only, score, one sample (`F-score-k1`) | 0.911 | 0.549 | 1.0% | 1.5% | -0.68 | 0.549 | 1.0% @ 1.5% |
| Faithfulness only, score, eight samples (`F-score-k8`) | 0.944 | 0.582 | 2.4% | 2.1% | 0.32 | 0.549 | 2.4% @ 2.1% |
| Faithfulness and completeness, score, one sample (`FC-score-k1`) | 0.895 | 0.585 | 4.7% | 5.7% | -0.71 | 0.573 | 4.7% @ 5.7% |
| Faithfulness and completeness, score, eight samples (`FC-score-k8`) | 0.939 | 0.634 | 8.3% | 6.5% | 1.02 | 0.575 | 8.3% @ 6.5% |

Caption: **The eight judge designs, on the 495-pair evaluation set.** Each design crosses what the judge is told to check (faithfulness only, F; faithfulness and completeness, FC), how it answers (a yes/no verdict, bin; a 0-to-10 score, score) and how many times it is asked (once, k1; eight times with the answers combined by a winsorised mean, k8). Paired discrimination is the tie-adjusted probability that an errored note scores below its own verified-clean twin, where 0.500 is a coin flip; it needs the twin, so it is a ceiling rather than a deployment measure. Detection and false alarms are single-note flag rates at each design's own rule, fixed before the run (a FAIL verdict for the yes/no designs, a score of 7 or below for the single-sample scored designs, a winsorised mean below 8.0 for the eight-sample ones), over 879 omission records and 336 clean records: three replicates of 293 omission pairs and 112 clean twins, with 202 commission pairs carrying the commission column. z is the two-proportion statistic for a design's detection against its own false alarms, pooled over the three replicates; pooling counts three passes over the same notes as independent observations and inflates z, so the per-note-majority reading in Section 5 (1.2 to 1.7 standard errors for the two yes/no completeness designs) is the honest strength of that gap. Where the flag line was drawn is not what limits these designs. The best-separating cut the data itself suggests - the threshold that maximises Youden's J (detection minus false alarms) on the mean-over-runs aggregate - lands at 8.17 (F-score-k1), 8.40 (F-score-k8), 8.50 (FC-score-k1) and 8.19 (FC-score-k8), every one of them above the 7 and 8.0 fixed before the run rather than below; and threshold-free, the area under the curve stays between 0.503 and 0.575 across all eight.

Notes:

- **F-bin-k1**: a single yes/no verdict takes two values, so there is no threshold to sweep
- **F-bin-k8**: the eight verdicts aggregate to a vote fraction rather than a score, so it is not swept with the scored designs; read as one anyway, the most sensitive cut inside the 10% budget gives 11.4% detection (100/879) at 8.9% false alarms (30/336)
- **FC-bin-k1**: a single yes/no verdict takes two values, so there is no threshold to sweep
- **FC-bin-k8**: the eight verdicts aggregate to a vote fraction rather than a score, and no cut on it keeps false alarms within 10%
- Detection counts, at each design's own rule: F-bin-k1 84/879 errored and 30/336 clean; F-bin-k8 84/879 errored and 25/336 clean; FC-bin-k1 354/879 errored and 107/336 clean; FC-bin-k8 372/879 errored and 108/336 clean; F-score-k1 9/879 errored and 5/336 clean; F-score-k8 21/879 errored and 7/336 clean; FC-score-k1 41/879 errored and 19/336 clean; FC-score-k8 73/879 errored and 22/336 clean

Sources: `results/w2-ablation/_state/grid-main2.jsonl`, `paper/figures/_recomputed.json (grid block)`, `w2_results_main.json (youden_thresholds)`
