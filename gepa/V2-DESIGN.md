# The second campaign (v2): the best-judge experiment (GEPA on the strongest architecture + checklist decomposition)

Rewritten 2026-08-13, after the ablation grid completed and its reading settled. Written before this campaign ran, and kept as the design of record.

## Why this experiment exists

The first campaign optimised a handicapped student: the k=1 scored judge, on the grid's own
settings. The completed grid shows k=8 is the better architecture on the paired measure, so
the obvious objection to the first campaign's negative result is that the search was never
given the strongest judge to work on.

The handicap does not explain the wall. Tie-adjusted, ensembling buys +0.03-0.05 paired AUC
(0.585 -> 0.634 for the completeness-scored judge; the larger apparent jump in strict
win-rates is score granularity splitting ties). On the absolute task - one note, no
reference, which is what the first campaign's objective measured and what production faces -
k=8 moves detection from 4.7% to 8.3% against a 6.5% false-alarm rate, which is
statistically nothing (z=1.0). Doubling a signal that equals noise is still noise.

So the wall is almost certainly the task rather than the wording or the sample count. But
"almost certainly" is not a paper-grade sentence while the strongest architecture has never
been optimised directly. This campaign removes that loophole, and adds the one approach
that changes the task itself.

## What the grid established (inputs to this design)

| fact | number | consequence for v2 |
|---|---|---|
| Absolute omission detection ≈ FA rate in all 8 designs | AUC 0.50-0.58; det-FA z ≤ 1.0 for scored cells | the objective must be absolute detection at matched FA, or we optimise noise |
| Paired signal exists but is weak | best 0.634 (chance 0.500), stable 0.601-0.655 across reps | paired ranking is the secondary term, not the target |
| Ensembling's real gain | +0.03-0.05 pAUC | k=8 student is right, but do not expect rescue |
| Signal concentrates on complete removals of critical facts | 0.690 there; 0.526 with a strong residual | report v2 results on the same conditioning, or improvements will hide in the easy cells |
| Commissions are near-ceiling everywhere | 0.87-0.94 paired | closed lookups work; open "anything missing?" does not - the checklist rationale |

## Arm A - GEPA v2 on the strongest architecture

- **Student**: the grid's FC-score-k8 judge - completeness criterion slot, 0-10 score, 8 samples, winsorized mean - with GEPA mutating the instruction text only. Every candidate evaluation runs the full k=8 ensemble (this is what makes v2 cost more than v1's \$25).
- **Objective (primary)**: absolute detection at FA <= 10%, computed against the dev pool's clean twins - the deployment measure. **Secondary**: paired AUC (tie-adjusted), so candidates that improve ranking without moving the operating point still register. A candidate that merely relocates its threshold scores no better than its parent on either term.
- **Dev pool**: `master/gepa_dev_factorial.json` - 176 judgements over the 22 dev consultations (omit-partial 65, omit-complete 51, add 16, change 22, clean 22). Built 12 Aug for \$16; eval-disjoint by assertion. Clean twins are the FA denominators; 22 is thin (the FA estimate carries \~±9pp at n=22), so feasibility uses a soft band (FA <= 10% target, <= 15% hard) rather than a knife-edge.
- **Frontier scoring on the full pool**: minibatches only pre-filter; accept/reject and the Pareto frontier use full-pool scores.
- **Reflection feedback**: names the missed class per failure - severity stratum, residual level, residual strength, and whether the surviving mention was quoted - so the reflector aims at the conditions that matter.
- **Seeds (4)**: the grid's FC-score prompt (the architecture's own baseline), v1's winner gepa-04, the engineered-completeness prompt, and the high-recall seed_v14_incl with the explicit brief to drive FA down at retained recall.
- **Iterations**: 15 with a stall-stop after 5 consecutive rejects (v1's winner appeared at iteration 4; its remaining 36 iterations bought nothing).
- **Cost**: \~2,800-4,000 k=8 judgements ≈ \$95-140 at measured grid rates (heavy prefix caching), + reflection \~\$15.

## Arm B - the pipeline judge

The one approach that changes the *task* rather than the judge - and the realisation that upgrades it: **our ground-truth construction machinery, run at inference time, IS the strongest judge we know how to build.** The transcript is available in production (that is what an ambient scribe transcribes); "reference-free" means no verified answer key, not no source. Every stage of the construction pipeline has an inference-time analogue, and each analogue's quality is already measured: blind extraction recovers 99.4% of authored facts; the critic audit strips what does not survive scrutiny; per-fact presence checks are the commission side of the asymmetry (0.87-0.94); the refute pass is what killed present-elsewhere false positives in our own omission audit (25% of cut omissions were "it IS in the note, worded differently" - exactly what a refute step catches); the severity rubric (kappa 0.66) weights what matters. Same inputs as every grid judge saw - transcript + note - so any win is attributable purely to task restructuring.

Three tiers, so the paper can show WHERE the gain comes from:

- **B1 - single-prompt checklist** (already smoked): extract-and-check in one call. Baseline: caught 50-100% of omissions but at 44% FA, at an arbitrary threshold.
- **B2 - pipeline-lite**: stage 1 extracts the fact list from the TRANSCRIPT ONLY (one call per consultation, cached and shared across that consultation's notes); stage 2 checks all facts against the note in one keyed call (verdicts keyed by fact id - the RAGAS-fix pattern, so misalignment is structurally impossible). Score = share of facts present; threshold swept.
- **B3 - full pipeline**: extraction -> critic audit (drop unsupported/duplicate facts, grade severity by the rubric) -> per-fact **full / partial / absent** trichotomy checks (aimed directly at the partial-omission collapse: the judge is asked "is this fact fully captured, partially captured, or absent?", never "is anything missing?") -> **refute pass on flagged facts only** (a skeptic tries to find the fact in the note before an absence is accepted; cost scales with flags, not facts) -> severity-weighted aggregate + a legible missing-facts list.

Design rules: extraction NEVER sees the note under judgement (transcript only - else the checklist could be biased by the note); extraction and checking use different model families (constructor extracts, judge-primary checks - the existing role structure); one instrument per batch.

**Smoke first**: all three tiers on a 38-pair stratified subset with its clean twins. Report per tier: detection at matched FA <= 10%, the detection/FA curve, per-class and per-residual-level breakdown (partials are the point), paired discrimination for comparability with the grid, and measured cost per note. The full 495-pair run runs only if the smoke justifies it.

- **Cost intuition**: B3 \~\$0.50-1.00/note vs \~\$0.02-0.05 for a monolithic judge - a 20x price for evaluation that works is a finding in either direction.
- **Prediction to test** (stated before running, as discipline not ceremony): B3 beats every grid cell on absolute detection at matched FA, with the largest gain on partial omissions via the trichotomy. If it does NOT, the wall claim extends to decomposition - bleaker and equally publishable.

## Read-out and what each outcome means

- **A improves absolute detection at matched FA**: the wall was partly prompt-reachable after all - report the recipe; the paper's advice changes.
- **A improves paired but not absolute** (expected): confirms every lever relocates rather than creates; the wall claim stands on the strongest architecture, optimised directly, and no reviewer can call it a strawman.
- **B beats the grid at matched FA** (the expectation going in): the constructive result - omission detection is possible, but only by restructuring the task around enumerable facts; connects directly to the paper's ground-truth protocol (fact sheets ARE checklists).
- **Neither helps**: the strongest possible negative, stated with the strongest possible baseline.

## Staging

Minimal first, so that each stage is decided on the previous stage's evidence rather than
all at once.

- **Arm B**: the smoke on the 38-pair subset answers "does it work". If yes, one
  confirmation batch on a stratified 100-150 pairs, one replicate - the subsample analysis
  on the grid shows half-corpus estimates are stable to about +/-0.02-0.03. No
  full-corpus run. If the smoke is ambiguous, the confirmation batch targets the ambiguous
  cells only.
- **Arm A (GEPA v2)**: k=4 student during the search, which captures most of the ensemble's
  variance reduction at half the cost, with the winner confirmed at k=8 on the full dev
  pool; 15 iterations, stall-stop after 5 consecutive rejects.
- **Anchors**: MEDEC only. MedVAL is dropped as a primary anchor because its
  dialogue2note split is entirely ACI-derived.

Eval-set disjointness is asserted at load; one instrument change per batch; nothing edits
mid-run.

---

## Arm A result, 2026-08-14: the wall stands on the strongest architecture, optimised directly

Run in 18 units under experiment `gepa-omission-v2`.
Runner `gepa/gepa_v2.py`; artifacts `gepa/v2_results.json`,
`gepa/v2_lineage.jsonl`, `gepa/v2_state.json`. Student `openai/gpt-5.4` at temp 1.0,
`reasoning_effort` none, `max_tokens` 1024 (the grid's own settings); reflection
`openai/gpt-5.5` at reasoning high. Dev pool `master/gepa_dev_factorial.json` via
`gepa_data.load_examples`: 176 judgements over 22 consultations (65 omit-partial, 51
omit-complete, 22 clean, 16 add, 22 change), `eval_overlap: 0` asserted at every load.
**No evaluation pair was judged at any point.**

### What the objective measures, and why it cannot be gamed

Primary is absolute detection on the 116 omissions at the most sensitive operating point whose
false-alarm rate on the 22 clean twins is still at or below 10%, with the threshold chosen after
the fact from the pool. That number is invariant to adding a constant to every score, which is
exactly the loophole v1's fixed flag line left open: v1 rewarded any prompt that pushed the score
mass down past a threshold of 7. Secondary is paired tie-adjusted discrimination (errored note
below its own clean twin, ties count half). A candidate had to improve one and not lose the other
to be accepted, and accept/reject ran on full-pool scores, never on the minibatch.

### The k=8 confirms, full dev pool

| candidate | what it is | det@FA<=10% | thr | FA | paired disc (omissions) | complete | partial | det@FA<=15% |
|---|---|---|---|---|---|---|---|---|
| `seed_v1_winner` | v1's winner gepa-04, as a seed | **13.8%** | 7.50 | 9.1% | 0.547 | 9.8% | 16.9% | 14.7% |
| `seed_fc_score` | **the architecture's own baseline** | 10.3% | 7.75 | 9.1% | **0.685** | 11.8% | 9.2% | 10.3% |
| `gepa2-04` | **the only thing v2's search produced** | 8.6% | 8.50 | 9.1% | 0.418 | 9.8% | 7.7% | 8.6% |
| `seed_engineered` | best hand-written prompt | 7.8% | 7.62 | 4.5% | 0.547 | 11.8% | 4.6% | 20.7% |

`seed_v14_incl_lowfa` was evaluated at k=4 (det 6.0%, disc 0.582) and not confirmed at k=8: the
budget ran out first, and it was the least informative of the five.

**Did prompt search beat the architecture's own baseline? No.** `gepa2-04`, the single mutation
the search accepted, lands at 8.6% against the baseline's 10.3%, which is 2 fewer omissions
caught out of 116 (exact McNemar p = 0.75, cluster bootstrap over the 22 consultations
[-6.9, +3.7] pp). Its paired discrimination is 0.418 against the baseline's 0.685, which is below
chance: it is worse at the one thing the grid says the best design is mildly good at.

**The one candidate above the baseline is not a v2 finding.** `seed_v1_winner` is v1's existing
prompt, seeded in as a comparator; its instruction block is byte-identical to
`gepa/judge_prompt.txt`. It edges the baseline by +3.4pp at the same 9.1% false-alarm rate, and
that difference is not significant (10 discordant notes its way, 6 the other, exact McNemar
p = 0.45, cluster bootstrap [-1.9, +10.5] pp) while its paired discrimination is *worse* than the
baseline's (0.547 against 0.685). A prompt that catches slightly more at a matched operating
point while ranking notes against their own twins less well is relocating on the same curve, not
separating better. **`gepa/judge_prompt_v2.txt` was deliberately not written**: emitting a
byte-identical copy of v1's artifact under a v2 name would falsely imply that v2 discovered
something. It did not.

### Trajectory

4 seeds, 11 iterations, **1 accepted mutation** (iteration 7), 4 rejected on full-pool scores,
6 pre-filtered out at the minibatch. The pre-registered stall-stop (5 consecutive rejects) fired
at **iteration 5** and is recorded in `v2_state.json` under `prereg_stall_stop`. The search was
then extended past it on the reserved budget, deliberately and recorded as a deviation, because
this arm exists to remove the objection that the best architecture was never searched hard
enough; nothing else changed. The extension bought one accept, which then lost at k=8. The search
stopped at iteration 11 of 15 to keep enough budget for the confirms.

One check worth keeping: at 4 consecutive rejects the minibatch pre-filter was re-scored from
cache on omissions alone, in case mixing the near-ceiling commission classes into the pre-filter
statistic was throwing away good children. It was not. All three pre-filtered children lost on
omission-only separation too, by equal or larger margins.

### Calls

**6,731 calls at k=4** over 14 units (4 seed evaluations, 11 iterations) and **5,632 calls
at k=8** over 4 units, one per confirm, 176 judgements x 8 samples each. 12,363 calls in
total, with zero parse failures anywhere.

### The commission control, on the same k=8 runs

At the identical operating point, detection on fabrications and alterations is **39.5% to 57.9%**
while detection on omissions is **7.8% to 13.8%**; paired, 0.776-0.855 against 0.418-0.685. The
asymmetry seen across 21 optimisation candidates and across the 8 grid cells holds on the best
design after direct optimisation. It is the most robust thing in this line of work.

### The honest read

The wall stands. Optimising the grid's best judge directly, on an objective a threshold shift
cannot game, with four seeds including the best hand-written prompt and v1's own winner, produced
one accepted mutation that is worse than the prompt it started from. Nothing in the k=8 table is
distinguishable from the baseline: the widest gap is +3.4pp on 116 omissions at p = 0.45, and it
belongs to a seed rather than to anything the search found. The best design in the grid detects
about one omission in ten at a deployable false-alarm rate, and no amount of instruction rewriting
moved that. The loophole is closed: it is no longer sayable that the wall claim rests on a
handicapped k=1 student or on prompts nobody optimised.

Two things stop this being the last word. The dev pool is **176 judgements over 22 consultations
with only 22 clean twins**, so the false-alarm rate that sets every operating point carries about
±9pp, and a 3-4pp difference in detection is inside the noise in both directions - nothing here
should be quoted as a number about the world until it is re-measured on the evaluation set. And
the search itself was small: 11 iterations against the 15 designed, one accept. The
correct claim is "prompt optimisation of the strongest monolithic judge did not beat its own
baseline here", not "prompt optimisation cannot help".

**Deviations from the design above, all deliberate and all in the artifacts.** (1) The search ran
at k=4 and only the confirms at k=8, per the staging above. (2) The student uses
`gepa_data.build_prompt`'s shape - fixed preamble, evolved instruction, fixed JSON contract -
rather than the grid's literal `Score:` tail, so all four seeds are expressible in one harness and
the parse path is v1's; the criterion text, the 0-10 score, the winsorized mean and the k are the
grid's. Every comparison in the table is inside that one harness. (3) k=4 aggregates use the same
winsorizing operator on 4 draws (asserted equal to `common.winsorize8` at k=8 in `--selftest`).
(4) The stall-stop extension, above. (5) One seed unconfirmed at k=8.

---

## The third campaign (v3), 2026-08-14: the fair-split rerun - both legs improve on the validator and neither survives the held-out set

Run under experiment tag `gepa-v3` in 14 units. Runner `gepa/gepa_v3.py`, partition builder `gepa/gepa_v3_data.py`, artifacts
`gepa/v3_results.json`, `gepa/v3_lineage.jsonl`, `gepa/v3_state_{pipeline,mono}.json`,
`gepa/v3_partition.json`. 4,417 API calls, **zero parse failures**, zero killed runs.

### Why v3 exists

v1 and v2 shared one flaw: a single small pool served as BOTH the reflection source and the
acceptance validator - the overspecialisation the DSPy GEPA documentation warns about in as many
words. It cashed out exactly as predicted: v1's winner read 0.685-equivalent on its dev pool and
**0.549 paired on the evaluation set**. v3 fixes the inner split at real scale, runs the
mechanics the GEPA paper (arXiv 2507.19457) actually specifies, and points the search at a
compound system - GEPA's claimed home turf - as well as at a monolithic prompt.

**The partition** (`gepa/v3_partition.json`, committed before any optimisation, disjointness
asserted at every load against the test subset AND against `gepa/eval_exclusions.json`'s 154
optimisation-trained pair ids and 22 consultations, at both pair and consultation level):

| split | consultations | pairs | notes | role |
|---|---|---|---|---|
| TEST | 47 | 151 | 198 | `master/arms_confirm_subset.json`. Touched once, by one candidate per leg, at the end. Its baselines were already bought - the pipeline-judge runs for B2, the ablation grid for the grid cells. |
| REFLECT | 45 | 180 | 225 | minibatches and reflection traces. The reflector reads nothing else. |
| VALID | 20 | 85 | 105 | acceptance, per-instance Pareto rewards, winner choice. The reflector never sees it. |

Everything is split at CONSULTATION level, because two pairs from one consultation share a
transcript, a clean twin and a fact list.

**Canonical mechanics honoured**: per-instance Pareto frontier with parents sampled in proportion
to instances led; PROMISCUOUS acceptance (a child enters the pool on a minibatch improvement, and
the frontier prunes afterwards); merge enabled once 6 candidates exist, capped at 4 invocations;
rich per-failure textual feedback; reflection LM `openai/gpt-5.5` at reasoning **medium** (not
high - cost). Minibatches are drawn consultation-first, 12 notes over 4 consultations, so every
errored note's own clean twin is in the batch - which is what lets leg A's traces show the
reflector the DIFF of per-fact verdicts between a note and its twin.

**Objective on VALID**: primary is paired tie-adjusted discrimination on omissions; secondary is
absolute detection at the best threshold holding false alarms <= 10% on VALID's 20 clean twins.
Primary is the paired measure this time because 20 clean notes move the false-alarm axis in 5pp
steps - too coarse to steer a search with. Both are reported everywhere.

### Leg A - the pipeline judge's check module (primary)

Student: the pipeline judge's B2 tier, unchanged except for one paragraph. Extraction stays
frozen and is read from `results/w2-pipeline/_cache` (43 TRAIN consultations extracted once for
\$4.03 at \$0.094 each; all 47 TEST consultations were already cached from those same runs, so
the TEST run reuses the identical fact lists the baseline was scored on). The check stage runs
at `reasoning_effort` none, the setting the pipeline judge uses, through `w2_pipeline.judge_note`
itself. GEPA rewrites only the instruction paragraph of
`w2_prompts/pipeline_check_binary.txt`.

**Trajectory**: 15 iterations, 8 accepted (7 mutations, 1 merge), 7 rejected at the minibatch,
3 merge invocations. The winner is `v3p-01`, from **iteration 1**.

| | paired omissions | complete | partial-weak | partial-strong | det@FA<=10% | FA | commissions |
|---|---|---|---|---|---|---|---|
| **VALID** seed (the shipped prompt) | 0.792 | 0.907 | 0.692 | 0.562 | 33.3% | 10.0% | 0.581 |
| **VALID** `v3p-01` | **0.844** | 0.944 | 0.885 | 0.438 | 33.3% | **5.0%** | 0.649 |
| **TEST** `v3p-01` (once, 198 notes) | 0.779 | 0.911 | 0.792 | 0.515 | 16.8% (22/131) | 8.5% | 0.725 |
| **TEST** B2 baseline (free, `confirm-B2.jsonl`) | **0.801** | 0.935 | 0.792 | 0.561 | 16.0% (21/131) | 8.5% | 0.650 |

On VALID the mutation gains +5.2pp paired and halves the false-alarm rate at equal detection. On
the held-out 131 omissions it is **-2.3pp paired** (cluster bootstrap over the 47 consultations
[-0.074, +0.027]; sign test on the 21 discordant notes 8 its way against 13, p = 0.38) and
**+0.8pp detection**, which is one extra omission out of 131 (exact McNemar on the flag decisions
3 against 2, p = 1.0). Neither prompt's detection is significantly above its own false-alarm rate
(z = 1.38 and z = 1.27).

**The VALID gain was never significant either.** Winner against seed on VALID: sign test 8 wins
against 3, p = 0.23, cluster bootstrap [-4.2, +16.2] pp. A 20-consultation validator with 48
omissions cannot distinguish a 5pp difference from nothing, and the held-out set says it was
nothing.

`w2_prompts/pipeline_check_v3.txt` was **deliberately not written**: the rule set before the run
was that a prompt file gets written only when the winner beats its baseline on the primary
measure on held-out data. It does not.

### Leg B - the monolithic FC-score judge at k=1 (secondary, gated)

Seeds: the grid's own FC-score criterion (the harness rebuilds `w2_prompts/grid_FC_score.txt`
byte for byte from it - asserted in `--selftest`) and the engineered completeness prompt. 15
iterations, 7 accepted (4 mutations, 3 merges), 8 rejected, 3 merge invocations. Winner `v3m-08`
at **iteration 14**, itself a merge of two descendants of the engineered seed.

The pre-registered gate - confirm on TEST only if the winner beats the FC-score seed on VALID by
more than 2pp on BOTH measures - **fired**: +7.3pp paired (0.688 against 0.615) and +22.9pp
detection (25.0% against 2.1%). It clears the engineered seed too (+4.2pp and +6.2pp). So the
\$10.69 confirmation at k=8 was bought.

| TEST, 198 notes | paired omissions | complete | partial-weak | partial-strong | det@FA<=10% | FA | commissions |
|---|---|---|---|---|---|---|---|
| `v3m-08` at k=8 | 0.626 | 0.613 | 0.681 | 0.591 | 8.4% (11/131) | 6.4% | 0.950 |
| `FC-score-k8` replicate 1 (free, grid records) | 0.622 | 0.677 | 0.722 | 0.409 | 13.0% (17/131) | 6.4% | 0.950 |
| `FC-score-k8` replicate 2 | 0.649 | 0.677 | 0.681 | 0.561 | 10.7% | 8.5% | 0.950 |
| `FC-score-k8` replicate 3 | 0.531 | 0.589 | 0.514 | 0.439 | 7.6% | 6.4% | 0.975 |

Against the matched single-run baseline the winner is **+0.004 paired** (sign test 40 against 36,
p = 0.73; cluster bootstrap [-0.126, +0.133]) and **-4.6pp detection** (McNemar 2 against 8,
p = 0.11). Set that beside the baseline's own replicate spread - paired 0.531 to 0.649, detection
7.6% to 13.0% on the same 131 pairs - and the honest reading is that a single run of this
architecture on 131 omissions carries about ±0.06 of paired noise, which is fifteen times the
difference the search produced.

### What the search actually wrote

Both legs produced longer, more specific, entirely sensible-looking prompts. Leg A's seed is 564
characters; its winner is 3,183 and its most elaborate candidate 15,117. Leg A's winner names
load-bearing content, polarity, certainty, dose and attribution, and tells the checker to
preserve hedging - all of it good advice, all of it plausible, none of it worth anything on
unseen data. The mechanism is visible in the score distributions on TEST: the winner's mean clean
score is 0.9587 against the baseline's 0.9612, its mean omission score 0.9315 against 0.9323 - so
the gap between a clean note and an errored one is **0.0272 against 0.0289**. The prompt moved
the whole distribution down a quarter of a percentage point and left the separation alone. That
is the grid's own finding - wording relocates the operating point rather than creating signal -
reproduced one level down, at the per-fact lookup.

### Measured cost per note

Pipeline check \$0.0075, monolithic k=1 \$0.0073, monolithic k=8 \$0.054.

### The honest read

The wall stands, and the loophole that v3 existed to close is now closed properly. The objection
after v2 was that GEPA had only ever been run in a way that could overfit - one pool for
reflection and acceptance - and only ever against a monolithic judge. v3 gave it a real inner
split, the paper's own selection and acceptance mechanics, merge, rich per-fact feedback, and the
compound system it is supposed to be good at. Both legs improved on the validator. Neither
improvement survived contact with 47 held-out consultations, and in both cases the validator gain
was itself inside its own noise before the held-out set was ever touched.

The one thing worth flagging as a signal rather than a null: leg A's winner has better commission
calibration on TEST than the B2 baseline (0.725 against 0.650, and 0.700 against 0.600 on
the 10 `add` pairs, where B3 sat below chance at 0.300). At n=20 that is not a finding, but it
points the same way the pipeline judge's commission control did - a coverage judge's blindness
to fabrication is a wording-reachable property in a way its blindness to omission is not.

**Deviations from canonical GEPA, all deliberate.** (1) Merge is implemented as a
reflector-mediated crossover of two frontier candidates that lead on disjoint instances, not as
the paper's module swap: both legs evolve exactly one module (leg A freezes extraction on
purpose), so module swapping is undefined. (2) Leg B's engineered seed is the engineered
completeness prompt's content re-expressed in the grid's frame - its transcript and note move to
the front - so it is not byte-identical to that prompt as the study ran it; the FC-score seed is
byte-identical to the grid's prompt. (3) VALID's primary is the paired measure rather than
absolute detection, for the sample-size reason above. (4) The search ran at k=1 on both
legs; only leg B's TEST confirm ran at k=8.
