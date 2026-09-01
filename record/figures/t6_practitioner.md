# T6 - what each rung delivers

Markdown twin of the paper's `t6_practitioner.tex` float. Both were generated from `t6_practitioner.json` beside this file by `render_tables.py`, which is not part of this release - see `SOURCES.md`. The price and flag columns live in `t6b_cost.md`, over the same rows in the same order.

| Approach | paired omissions | paired commissions | det. at own rule | false alarms | det. at FA <= 10% |
|---|---|---|---|---|---|
| **One call at standard settings, asked an open question** | | | | | |
| Best of the eight grid designs (FC / score / k=8) | 0.634 | 0.939 | 8.3% | 6.5% | 8.3% |
| Cheapest of the eight (F / score / k=1) | 0.549 | 0.911 | 1.0% | 1.5% | 1.0% |
| Best yes/no design (FC / binary / k=8) | 0.591 | 0.869 | 42.3% | 32.1% | - |
| Engineered omission judge (hand-written, 8 few-shots) | 0.639 | 0.754 | 8.4% | 2.7% | 8.4% |
| Deployed faithfulness judge, as shipped | 0.518 | 0.886 | 16.4% | 12.1% | - |
| Optimiser's exploratory winner, standard settings | 0.549 | 0.861 | 23.0% | 21.9% | 7.7% |
| G-Eval (decomposes into dimensions, never enumerates) | 0.568 | 0.890 | 26.8% | 12.5% | - |
| **Enumerate the transcript, read the result as a score** | | | | | |
| Instance checklist (arXiv 2507.17717 style) | 0.647 | 0.744 | 67.7% | 50.9% | 16.0% |
| Naive RAGAS-style recipe (ours) | 0.809 | 0.712 | 99.0% | 98.7% | 20.9% |
| RAGAS-style coverage score alone | 0.817 | 0.665 | 99.0% | 98.7% | 20.9% |
| Enumerate + check (pipeline B2) † | 0.786 | 0.650 | - | - | 16.0% |
| Enumerate + audit + check (pipeline B3) † | 0.762 | 0.617 | - | - | 18.3% |
| **The deployable operating points, each at its own rule** | | | | | |
| Route one - B3 verdicts read as a predicate: any critical fact absent | 0.795 | 0.661 | 24.6% | 2.7% | 24.6% |
| Route two - the optimiser's winner with a reasoning budget (one call) | 0.670 | 0.978 | 36.9% | 6.2% | 36.9% |
| Route two on the second judge family (gemini-3.1-pro, same prompt) † | 0.659 | 0.975 | 34.4% | 6.4% | 34.4% |
| Free-text rival: the engineered judge's own critical-omission list | 0.639 | 0.754 | 9.6% | 0.9% | - |

† held-out confirmation subset (131 omission pairs, 20 commission pairs, 47 clean twins); every other row is the evaluation set (293 omission pairs, 202 commission pairs, 112 clean twins; the eight designs 3 replicates, reference judges 2, route rows at the rules stated per row). The two pipeline-tier rows print paired means over three replicates; their sweep column is replicate 1. One denominator per set of items; rows measured on different sets are never ranked against each other.

The two route rows close the table at their own deployment rules, both already inside the 10% false-alarm budget, so the sweep column repeats their detection. Neither dominates: the budgeted call is the higher-detection point at 2.3x the predicate's false alarms and a tenth of its price, resting on a per-deployment calibration; the predicate is the quiet point whose flag names the missing fact with a severity grade (a grade a blinded clinician regrade reads about one grade hot on routine content).

A dash means the quantity is not expressible for that design: a binary or 5-point aggregate admits no threshold sweep, and the pipeline tiers write no flag.

- **Best yes/no design (FC / binary / k=8)**: binary aggregate: no usable sweep under a 10% cap
- **Deployed faithfulness judge, as shipped**: binary verdict: not expressible
- **G-Eval (decomposes into dimensions, never enumerates)**: 1-5 integers: not expressible
- **RAGAS-style coverage score alone**: detection and false alarms are the full recipe's; the component is a score, not a detector, and carries no severities so no per-fact predicate is expressible on it
- **Enumerate + check (pipeline B2)**: paired cells are means over 3 replicates: omissions 0.8015/0.771/0.7863, commissions 0.65/0.60/0.70; the swept column is replicate 1
- **Enumerate + audit + check (pipeline B3)**: paired cells are means over 3 replicates: omissions 0.7519/0.7672/0.7672, commissions 0.50/0.65/0.70; the swept column is replicate 1
- **Route one - B3 verdicts read as a predicate: any critical fact absent**: critical-severity pairs: 39.7% (60/151). Held out (47 twins) the same rule read 20.6/22.9/21.4% across 3 replicates at 2.1% false alarms in every one (the same clean note); the evaluation set tightens the false-alarm interval from [0.4, 11.1]% to [0.9, 7.6]%. For contrast, held out, the best AGGREGATE-score threshold on these same verdicts at the predicate's own false-alarm level detects only 7.6% - the predicate delivers 2.7x more from identical verdicts
- **Route two - the optimiser's winner with a reasoning budget (one call)**: critical-severity pairs: 58.9% (89/151). The rule rests on clean notes parking at exactly 10 (90-94% of the 112 twins per replicate) - a calibration to re-establish per deployment; tightened to flag-below-9 it detects 14.3% at 0.9%. Single replicates read 39.7/34.4/38.2% at 4.3/2.1/4.3% on the held-out 47 twins
- **Route two on the second judge family (gemini-3.1-pro, same prompt)**: per replicate: 34.4/36.6/34.4% detection at 4.3/4.3/8.5% false alarms; clean-at-10 45/45/43 of 47 - the operating point and its calibration both transfer
- **Free-text rival: the engineered judge's own critical-omission list**: a predicate on free text, so no threshold sweep exists

Sources: `results/w2-ablation/_state/grid-main2.jsonl`, `results/w2-strong/_state/arms-main.jsonl`, `results/w2-v14/_state/arms-main.jsonl`, `results/w2-baselines/_state/arms-main.jsonl`, `results/w2-pipeline/_state/confirm-B2.jsonl`, `results/w2-pipeline/_state/confirm-B3.jsonl`, `results/w2-pipeline/_state/confirm-stages.jsonl`, `results/w2-pipeline/_state/confirmr{2,3}-B{2,3}.jsonl`, `results/w2-pipeline-replicates/analysis.json`, `results/w2-power/_state/power.jsonl`, `results/w2-power/_state/power-gemini.jsonl`, `results/w2-evalfull/analysis.json`
