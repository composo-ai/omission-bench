# W-F - the GEPA-optimized omission judge

Spec: `specs/w-f-gepa-omission-judge.md`. Arm slot: `gepa-optimized` in `w2_arms.py`.

This directory holds an optimizer, the prompt it found, and the full record of how it got
there. The prompt is the deliverable: `gepa/judge_prompt.txt` is the file `w2_arms.py`
loads for the `gepa-optimized` arm, and it is a first-class entrant in the comparison, not
a courtesy row.

---

## Why the arm exists

The paper compares prompt-based judges against scored-and-ensembled ones, and a fair
comparison has to answer an obvious objection: *maybe your prompt-based arms are just
badly written*. Hand-written prompts are a weak field. So one arm is not hand-written at
all - it is whatever a prompt optimizer can find when the objective actually rewards the
thing we care about.

There is a second reason, specific to this project. The June pilot already ran GEPA
(`experiments/clinical-hallucination-detection/methods/v23b_gepa-judge-fulltrain/`), and
it lost to hand-tuning - kappa 0.46 against v14's 0.77. But that run was optimized against
ACI-Bench's hallucination-annotation scheme, **which has no omission category at all**. It
could not have been rewarded for catching a missing fact, and it kept v14's explicit
"omissions are NOT errors" exclusion rule all the way through optimization. Quoting the
pilot as evidence that "GEPA cannot fix omission blindness" would be quoting an optimizer
that was never asked. This run asks.

---

## The data rule

The optimizer sees the **GEPA dev pool and nothing else**: 22 consultations carved out at
the W-A freeze precisely so this could happen (`master/gepa_dev_pool.json`,
`master/gepa_dev_pool_ids.json`), plus the 5 `gepa_dev`-tagged partial-omission pairs from
`master/partial_omission_seed_set.json`. It never touches an evaluation pair, and
`gepa_data.load_examples()` re-proves that on every call by intersecting the dev pool's
consultation ids against every evaluation set on disk (`master/pairs_master_frozen.json`,
and `master/dataset_v2.json` when the factorial build lands it) and refusing to run if the
intersection is non-empty. The winning prompt reaches the grid **untested on evaluation
data**. No eval-pair number for this arm exists yet, and none should until the grid buys
it in the ordinary way.

The pool becomes 79 judgements:

| | n | what it is |
|---|---|---|
| clean | 22 | one ideal note per consultation - the false-alarm side |
| add | 16 | fabricated content |
| change | 22 | altered content: flipped negation, wrong value, hardened decision |
| omit-complete | 14 | the removed fact survives nowhere in the note |
| omit-partial | 5 | the topic is still mentioned; the load-bearing part is gone |

The partial class is the hard one and the reason this workstream is interesting: a judge
can see the word "referral" in the note, tick the topic off, and never notice that the
urgency, the suspicion and the red-flag features behind it have all vanished.

---

## What the judge looks like

Same interface as every other single-prompt arm in `w2_arms.py`, so it is comparable to
the grid cells by construction rather than by argument: transcript and note in, one call
at `k=1`, `{"score": 0-10, "omissions": [...]}` out, flagged at `score <= 7`, run on
`gpt-5.4` (`judge-primary`) at temperature 1.0, `reasoning_effort: none`, `max_tokens
1024` - the grid's own settings, through the grid's own `common.llm()` transport.

The prompt file has three parts and the optimizer only ever rewrites the middle one:

```
CONSULTATION TRANSCRIPT: {transcript}         <- fixed preamble, FIRST
CLINICAL NOTE:           {note}                  (so every candidate shares a cache prefix)
---
<the evolving instruction block>              <- this is the artifact
Return ONLY a JSON object ...                 <- fixed output contract
```

Two deliberate boundaries:

- **The output contract is fixed** - JSON shape and score direction (10 = clean, 0 =
  severe). A candidate can never evolve its way out of being parseable, and the metric can
  never invert underneath the search. This mirrors how the pilot's DSPy program let a typed
  `OutputField` own the format rather than the prompt text.
- **Scoring policy is NOT fixed.** How many points a critical omission costs, where the
  judge puts its own flag line, what counts as normal clinical compression - all of that
  lives inside the evolvable block, because that calibration is exactly what we want to
  find out about.

Putting the transcript first is not cosmetic: it means all \~40 candidates judging the same
note share a byte-identical prompt prefix, so the provider's cache absorbs most of the
search's cost (`docs/COSTS.md` lever 1).

---

## The objective

The pilot's failure was an objective problem, so this one is written out explicitly. Per
example, reward is

```
0.7 x (did the flag decision match ground truth) + 0.3 x (magnitude-aware soft credit)
```

The soft term gives the search a gradient instead of a 0/1 cliff - an 8 on a note with a
missing red flag is a nearer miss than a 10, even though both sit the same side of the
flag line.

Aggregated, the objective is a **vector, not a scalar**: detection on omit-complete,
detection on omit-partial, detection on add, detection on change, false-alarm rate on
clean notes, and a severity-weighted omission rate that uses the freeze's own rubric grades
(`critical` 1.0, `supporting` 0.5, `peripheral` 0.25) so missing a red flag costs more than
missing a detail. Parent selection runs on the Pareto frontier over individual examples;
the single scalar exists only to pick a final winner:

```
J = 0.45 x omission detection (severity-weighted, both classes)
  + 0.20 x add/change detection (severity-weighted)
  + 0.35 x (1 - false-alarm rate)
subject to a HARD constraint: false alarms <= 10% of clean notes
```

A candidate that breaks the false-alarm ceiling cannot win at any J. Flagging everything
is not a solution, and the constraint is what stops the search from finding that out the
expensive way.

**Detection here means the arm's own flag rule** (`score <= 7`), not W2's discrimination
metric (errored note scored below its clean twin). Discrimination is computed and recorded
alongside, but it is not what the search optimizes: a judge you can only use by comparing
a note against a counterfactual clean version of itself is not a judge anyone can deploy.

---

## The search

Standard GEPA, structurally the same loop as the pilot's `optimize.py`:

1. Evaluate every seed on the full dev pool.
2. Pick a parent **from the Pareto frontier over examples** - a candidate stays in
   contention if it is the best anyone has managed on at least one example, sampled in
   proportion to how many examples it leads. A candidate that cracks two brutal partial
   omissions and loses on average therefore survives to be a parent. This is why GEPA uses
   a frontier rather than a leaderboard.
3. Draw a **stratified minibatch of 8**. Not uniform: the pool is 72% errored, so a uniform
   draw routinely contains no clean note at all and the false-alarm side of the objective
   gets no gradient that round. Every minibatch carries both sides and both omission
   classes.
4. Run the parent on it and build natural-language traces. On a missed omission the trace
   **names the fact that went missing**, quotes the freeze's severity rationale, and - for
   partial omissions - quotes the residual mention that survives and probably caused the
   miss. This is the one thing the pilot's feedback function structurally could not do.
5. A reflection model (`gpt-5.5`, `reasoning_effort: high`) reads the traces and the current
   instruction and proposes a rewrite. It is told what it may not touch (the output
   contract), what it must preserve (whatever produced the true positives and true
   negatives, which the traces mark explicitly), and that it must not encode the specific
   examples it was just shown - those notes are never seen again.
6. Run the child on the same minibatch. Better than its parent -> evaluate on the full dev
   pool and admit it to the population. Worse -> discard, but keep it in the lineage.
7. Stop at the iteration count or the credit cap, whichever comes first, and take the best
   feasible candidate.

Seeds (`gepa/seeds/`, lifted from `w2_prompts/` by `--init-seeds` so they are exactly the
grid's own text):

| seed | source | what it is |
|---|---|---|
| `seed_fc_score` | `grid_FC_score.txt` | W2's `FC-score-k1` cell - the completeness-aware criterion as a minimal pair. The pivotal contrast. |
| `seed_v14_incl` | `v14_incl.txt` | a hand-tuned production judge with its omission exclusions removed and an affirmative omission bullet added |
| `seed_engineered` | `engineered_completeness.txt` | the engineered completeness judge: criterion + severity decision procedure + 8 salience few-shots. The strongest prompt a human wrote here. |

`v14_incl` is a binary YES/NO judge and this arm is a scored one, so its decision idiom is
ported onto the score - "flag YES" becomes "flag-worthy", the `Verdict:` lines in its
few-shots become `Judgement:` lines. Its rules, few-shots and reasoning are verbatim;
anything more would stop it being a v14 seed.

### Why not DSPy

The pilot ran GEPA through `dspy.GEPA`, and reusing it was the default plan. It was not the
right call here, for three reasons:

1. **The artifact is a raw prompt file.** `w2_arms.py` had already registered
   `gepa/judge_prompt.txt` as the arm's slot, renders `{transcript}`/`{note}` into it, and
   parses the result with `parse_score_json`. Optimizing inside DSPy would have tuned the
   instruction against DSPy's own field-formatting wrapper, then handed it to a harness that
   does not use that wrapper - the prompt would be evaluated in a shape it was never
   optimized in.
2. **Transport comparability.** Going through `common.llm()` means this arm's calls are the
   same pinned model, temperature, seed formula, retry policy, receipts and cost accounting
   as every grid cell. That is the whole basis for putting a GEPA prompt and a grid cell in
   the same table.
3. `dspy` is not installed in this harness's venv, and installing it plus LiteLLM into a
   tree three other agents are writing to, for a wrapper we would then be working around,
   was a poor trade.

So the loop is reimplemented - Pareto-over-instances selection, reflective mutation from
failure traces, minibatch accept test, full-set promotion - over this harness's own
transport. `gepa_optimize.py` is \~450 lines and the whole trajectory is in `lineage.jsonl`.

---

## Cost and budget

Hard cap \$120 of OpenRouter credits, enforced in-process (`Budget`), on top of a
project-cumulative guard in `w2_common.SpendGuard`, fed live rather than at the end. Judge responses are cached by `(prompt sha, seed)` in
`gepa/cache/judge_cache.jsonl`, which matters more than it sounds: a surviving parent is
re-scored every time it is selected, and without the cache the search would re-buy the same
answers dozens of times.

The optimization step writes its own manifest to `results/gepa-omission/<run_id>/` and its
own cost-ledger line, per the spec's documented manifest divergence (section 4) - a
GEPA-shaped loop cannot be wrapped in `Run()` per call. The subsequent *evaluation* of the
winning prompt is an ordinary `w2_arms.py` cell and goes through `llm()` + `Run()` exactly
like every other arm.

---

## What happened

Run `20260812-153734-46ad9f4`: 40 iterations, 3 seeds + 18 accepted mutations = 21
candidates, 22 mutations rejected at the minibatch test, 1,867 judge calls, 110 minutes,
**\$25.24** of credits (\$18.10 student / \$7.14 reflection) against the \$120 cap. Nine of
the 21 candidates met the false-alarm ceiling. Regenerate any number below with
`gepa_report.py`.

### Seeds vs winner, on the dev pool

Detection is the arm's flag rule (`score <= 7`). All numbers are dev-pool only.

| | omit-complete (14) | omit-partial (5) | add (16) | change (22) | FA, clean (22) | omit sev-wt (19) | J |
|---|---|---|---|---|---|---|---|
| `seed_fc_score` (W2's FC-score-k1) | 14.3% | 0.0% | 56.2% | 36.4% | 9.1% | 11.4% | 0.459 |
| `seed_engineered` (best human prompt) | 14.3% | 0.0% | 50.0% | 36.4% | **0.0%** | 11.4% | 0.486 |
| `seed_v14_incl` (production judge) | 71.4% | 40.0% | 93.8% | 68.2% | **54.5% - unusable** | 62.9% | 0.600 |
| **`gepa-04` (winner)** | **28.6%** | **40.0%** | 56.2% | 45.5% | 9.1% | **28.6%** | 0.547 |

Against the two seeds anyone could actually deploy, the optimizer bought a real
improvement: severity-weighted omission detection **11.4% -> 28.6%** (+17.1pp), complete
omissions +14.3pp, partial omissions **0% -> 40%**, while holding false alarms at 9.1%
(2 of 22 clean notes) - inside the 10% ceiling. It is the best feasible candidate on both
J and omission detection, so the winner selection is not a close call.

Three things have to be said plainly alongside that.

**1. The absolute numbers are poor.** 28.6% severity-weighted omission detection means the
judge, operating inside its false-alarm budget, still misses roughly seven in ten
load-bearing omissions. Nothing here is a solved problem. And n is small - 14 complete
omissions, 5 partial, 22 clean notes, where FA 9.1% means literally two notes. One example
moves these percentages by 5-7pp. No significance testing was run and none should be
quoted from this table; the eval-set numbers with Wilson CIs, McNemar and cluster bootstrap
are what the grid will buy.

**2. The v14 seed's apparent recall is a threshold artifact, not detection.** It looks like
the strongest omission catcher in the table at 62.9%. It is not. Its mean score on clean
notes is **5.86** and on omitted notes **5.32** - a gap of 0.54. It scores nearly everything
below the flag line, so it "catches" most errors and flags 54.5% of clean notes for the same
reason. On threshold-free discrimination (did it score the errored note below its own clean
twin) it gets complete omissions right 35.7% of the time, *worse* than the plain FC-score
cell's 42.9%. Anyone reading the detection column alone would draw the wrong conclusion.

**3. GEPA moved the operating point more than the discriminability.** The same
threshold-free check, run on the winner:

| | mean clean | mean omit | gap | discrimination: omit-complete | omit-partial |
|---|---|---|---|---|---|
| `seed_fc_score` | 8.95 | 8.58 | 0.38 | 42.9% | 20.0% |
| `seed_engineered` | 9.14 | 8.63 | 0.50 | 35.7% | 40.0% |
| `seed_v14_incl` | 5.86 | 5.32 | 0.54 | 35.7% | 20.0% |
| `gepa-04` (winner) | 8.73 | 8.37 | 0.36 | 28.6% | 40.0% |

The clean/omit score gap is 0.36-0.54 for every prompt in the study, seeds and winner
alike. On complete omissions the winner's discrimination is the *lowest* of the four. Most
of its flag-rate gain came from shifting its whole score distribution down a little so
more errored notes cross the line - which costs two clean notes and buys fourteen errored
ones, a good trade at this threshold but not new information about the notes. The one
place it looks like genuine capability is the partial class, where it matches the
engineered seed's 40% discrimination while the simple criterion gets 20%.

### The frontier, which is the actual result

Across the whole 21-candidate population there is a wall. **Every candidate that reached
30% or better severity-weighted omission detection had a false-alarm rate of at least
22.7%.** Not one landed in the usable region:

| candidate | omit sev-wt | FA |
|---|---|---|
| `seed_v14_incl` | 62.9% | 54.5% |
| `gepa-09` | 48.6% | 36.4% |
| `gepa-03` | 45.7% | 27.3% |
| `gepa-13` | 45.7% | 45.5% |
| `gepa-14` / `gepa-15` / `gepa-16` | 34.3% | 27.3% / 22.7% / 31.8% |
| `gepa-19` / `gepa-20` | 31.4% | 22.7% |
| **best feasible (`gepa-04`)** | **28.6%** | **9.1%** |

And the search plateaued almost immediately: the winner was found at **iteration 4**, the
second accepted mutation. Thirty-six further iterations, seventeen more candidates and
\~\$21 of additional search produced nothing feasible that beat it. Twelve of the 21
candidates were infeasible, and every one bought its recall with false alarms.

In the spec's pre-registered language this sits between **GEPA-2 (partial / format-bound)**
and **GEPA-3 (architectural)**, and it is a bounded, honest result rather than a flattering
one. Reflective search with a correct objective, a feedback signal that names the exact
missing fact, three strong seeds and forty rounds could not find prompt wording that gets
omission recall without paying for it in false alarms. That is a stronger form of evidence
than the seeds' mere existence: the optimizer had every chance and the frontier held.
Which of GEPA-2 or GEPA-3 applies is not this file's call - it depends on `OD_best` from
the grid, on eval-set data this run has never seen.

### What GEPA actually wrote

The winner descends from `seed_engineered` and grew from 6,039 to 8,503 characters. Nine
substantive changes, and the interesting thing is how few of them are about the criterion:

1. **It made the audit two-directional and put the transcript first.** The seed says "look
   above all for CONTENT THAT IS MISSING". The winner replaces that with an ordered method:
   *build a mental inventory of the clinically important facts the consultation
   established, then check whether the note carries them*, and only then audit the note's
   assertions back against the transcript. No seed told the judge to enumerate from the
   transcript before reading the note.
2. **It generated the inventory's contents** - a nine-category checklist (chronology and
   progression; pertinent positives and negatives, especially red-flag questions and
   answers; examination findings *including important normal findings used to justify
   reassurance*; working diagnosis, differential and level of uncertainty; medication dose,
   route, changes, allergies; investigations with urgency, target service and reason;
   safety-netting with trigger, route and timeframe; safeguarding, capacity, driving
   advice). This converts an open-ended "is anything missing?" into a search over named
   slots.
3. **It promoted the partial-omission failure mode into its own section with seven worked
   shells**: "bloods requested" without the suspected condition that makes the result
   actionable; "referral made" without the urgency or cancer suspicion; "safety-net given"
   without the trigger, route or timeframe; a symptom area without the severity or
   exertional nature; a medication without the dose; a diagnosis without the uncertainty
   that drove management; a symptom region without the specific red-flag answer. The seed
   had a single sentence about this. This is where the winner's only real capability gain
   shows up.
4. **It built an altered-content taxonomy the completeness-focused seed did not have** -
   flipped polarity with worked examples ("denies vomiting" -> "vomiting"), wrong value,
   wrong laterality, wrong subject (patient vs partner vs family history), hardened
   uncertainty, fabrication, and one category none of the seeds contain: **over-reassurance**,
   "no red flags" or "normal examination" asserted when the transcript has a positive,
   equivocal or simply unasked red-flag point. That is a fabrication that presents itself
   as an absence, and it is a genuinely novel bit of prompt content.
5. **It discovered the flag threshold and wrote a triage policy instead of arithmetic.**
   The seed carried a deduction rule ("start at 10 and deduct 3 for each critical"). The
   winner throws the arithmetic away and states the decision directly: *any critical issue
   must receive a score low enough to be flagged for human review*; *one critical omission,
   inversion, fabrication or clinically material alteration is enough to fail the
   human-review threshold*; *supporting issues usually should not be flagged unless several
   together materially undermine the note*. The optimizer learned that a downstream
   threshold existed - the feedback traces state "flags at <= 7" - and encoded the triage
   rule rather than the score. This is the single most interesting thing it did, and it is
   consistent with finding 3 above: it optimized the operating point.
6. **It split the tie-break in both directions.** The seed had one rule: torn between two
   grades, take the lower. The winner has two: torn between supporting and peripheral,
   choose peripheral and do not penalise; torn between critical and supporting, choose
   supporting only when care would probably be unchanged. That asymmetry is a deliberate
   false-alarm brake and is very likely what held FA at 9.1% while the flag rate rose.
7. **It expanded "what not to penalise" from two bullets to nine** - paraphrase,
   abbreviations, combining repeated statements, recording a fact once, dropping
   conversational colour, omitting exhaustive symptom detail, ordering and formatting,
   minor wording changes in safety-netting where trigger and route survive, and "a concise
   note that a careful clinician would still regard as faithful and sufficient". The
   false-alarm constraint is visible in the prompt text.
8. **It deleted all eight hand-authored few-shots.** The engineered seed's distinguishing
   feature was worked examples; the winner has none, having traded demonstrations for
   explicit enumerable rules. Worth noting given how much of the judge literature leans on
   few-shots.
9. **It added a five-question final check**, of which question 4 is "did a surviving vague
   mention hide the loss of the load-bearing detail?" - the partial-omission test promoted
   to a mandatory last step before scoring.

Read together: the optimizer spent its effort on **procedure and calibration**, not on the
criterion. Every seed already said "faithful AND complete". What GEPA added was a method
for enumerating what completeness means, a taxonomy of the ways a note can be wrong, a
named list of things that are not violations, and an explicit triage policy at the flag
line. That it produced a better operating point and barely-changed discriminability is a
coherent story rather than a contradiction - and it is the reason this arm belongs in the
comparison rather than being waved at.

### Lineage highlights

- **it01** took the engineered seed from 14.3% to 57.1% on complete omissions in a single
  reflection round - and took false alarms from 0% to 27.3% with it. The trade was visible
  from the first mutation and never went away.
- **it04** produced the winner: the same jump, smaller, with the tie-break asymmetry and
  the nine-item do-not-penalise list attached. Nothing later beat it.
- **it12, it14, it18, it35** all selected `seed_v14_incl` as parent - it leads on 50-53 of
  79 examples, so the Pareto frontier keeps offering it - and all four rewrites were
  rejected at the minibatch test. The optimizer could not pull v14's false-alarm rate down
  without losing what made it fire.
- **12 of 21 candidates were infeasible.** The failure mode of this search is uniform:
  every route to better recall went through flagging more clean notes.
- The full trajectory, including all 22 rejected rewrites with their complete instruction
  text, is in `lineage.jsonl`.

### Eval hygiene

`eval_overlap: 0` at run time, asserted at load and recorded in `results.json`,
`lineage.jsonl` and the run manifest. The optimizer's 79 judgements come from 22
consultations; `master/pairs_master_frozen.json` holds 281 pairs over 112 consultations;
the intersection is empty. **No evaluation pair was judged at any point in this run, and no
eval-set number for this arm exists.**

#### A contamination flag, raised here and since resolved

> **Resolved before release.** The global drop recommended at the end of this
> section is what happened. In the released pair set the five pairs below carry
> `split: "gepa_dev"` and `eval_set: false`, every judge is scored on the same
> 495 evaluation pairs, and no released number is computed over them. The rest of
> this section is kept as the record of how the decision was reached.

`master/dataset_v2.json` landed while this run was in flight, and the same guard fires
against it. It carries **324 pairs = 319 `split: "eval"` + 5 `split: "gepa_dev"`**. Those
5 are the partial-omission pairs this optimizer trained on:

```
authored|med_review_dose|omit      authored|penicillin_delabel|omit
authored|video_rash_child|omit     primock|day3_consultation02|omit
primock|day4_consultation02|omit
```

The builder carried the `split` field through faithfully and simply does not filter on it,
so the fix is one predicate: `split != "gepa_dev"`. This does not affect the run above -
those pairs were training data by design, tagged as such at the freeze, and
`dataset_v2.json` did not exist when the optimization started. It affects what happens
next:

- It is **1.5% of the whole eval set but 13% of its omit-partial class (5 of 39)** - and
  the partial class is exactly where this arm shows its only real gain, so leaving it
  uncorrected would flatter precisely the number most worth doubting.
- It is contamination **for the `gepa-optimized` arm only**. No other arm has ever seen
  those pairs, so for every other arm they are legitimate eval data.

That last point makes it an analysis decision rather than a pure bug, and it is not this
file's to make. Two clean options: drop the 5 globally from `dataset_v2.json` (loses 5 of
39 partial pairs but keeps every arm on an identical item set, which the comparison table
needs), or keep them and score the `gepa-optimized` arm on the 34 uncontaminated partial
pairs with a stated footnote. **The global drop is the recommendation** - comparability is
worth more here than five pairs.

`gepa/eval_exclusions.json` lists all 57 trained pair ids and the 22 consultation ids in
machine-readable form so any consumer can filter without re-deriving them.

The next thing that should happen to `judge_prompt.txt` is
`w2_arms.py --arms gepa-optimized --runs 3` on a dataset with those 5 pairs handled, which
will be the first time it meets evaluation data.

---

## Files

| file | what |
|---|---|
| `judge_prompt.txt` | **the artifact** - the winning prompt, the file `w2_arms.py` loads |
| `gepa_optimize.py` | the optimizer: seeds, search loop, reflection, budget, manifest |
| `gepa_data.py` | dev pool + eval-disjointness guard, prompt shape, reward function, objective vector |
| `gepa_report.py` | regenerates every number quoted above from `results.json` |
| `lineage.jsonl` | every candidate, accepted and rejected, with its full instruction text, scores and spend |
| `results.json` | final scores, winner, per-example detail, provenance |
| `eval_exclusions.json` | the 57 pair ids and 22 consultation ids this optimizer trained on - filter any eval set against it |
| `seeds/*.txt` | the three seed instruction blocks, with their source hashes |
| `cache/judge_cache.jsonl` | every raw judge response, `(prompt sha, seed)` keyed - the evidence under every number here, and it makes a re-run free |
| `run.log` | console transcript - written locally, **not committed** (`*.log` is gitignored repo-wide); everything in it is in `lineage.jsonl` in structured form |

Re-run:

```
.venv/bin/python gepa/gepa_optimize.py --init-seeds        # refresh seeds from w2_prompts
.venv/bin/python gepa/gepa_optimize.py --smoke -y          # 1 seed, 8 examples, 1 iteration
.venv/bin/python gepa/gepa_optimize.py --iterations 40 -y  # the real run
.venv/bin/python gepa/gepa_optimize.py --iterations 25 --resume -y   # extend the population
.venv/bin/python gepa/gepa_report.py --all --misses --tree # the tables above
```

`--resume` rebuilds the population from `lineage.jsonl` instead of restarting from the
seeds; re-scoring recovered candidates is free because the judge cache is keyed on
`(prompt sha, seed)`. Worth using only if there is reason to think the frontier above is a
search failure rather than a real one - 36 iterations of evidence say it is not.

The grid consumes it with no further ceremony - the arm was registered before this
directory existed:

```
.venv/bin/python w2_arms.py --list                         # gepa-optimized: available
.venv/bin/python w2_arms.py --arms gepa-optimized --runs 3 -y
```
