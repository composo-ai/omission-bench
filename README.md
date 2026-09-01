# OmissionBench - harness

This repository is the code behind two companion papers released alongside it, and it
generated every number they report. The judge paper, *LLM Judges Verify Presence, Not
Absence: Omission Blindness in AI Clinical Notes and What Recovers It*, introduces the
benchmark, finds that judges verify what a note contains while sitting near chance on what
it omits, and reports what recovers omission detection. The census paper, *One note in
three: a verified census of three deployed AI scribes, and the instrument that counted it*,
reports the audit of the three products.

## What this repository is

This is the instrument, published so that someone else can use it. Every prompt, the
pipeline that builds the benchmark and verifies it, the runners for all twenty judge
designs, the three prompt-optimisation campaigns with each candidate's full instruction
text, the census instrument that audited three deployed products, the model pins, and the
analysis behind the published numbers. **Repeating the study** below is the shortest path
from a clone to a number.

It is an archival research release rather than a maintained project: versioned and
archived rather than developed, no roadmap, no issue triage. Questions and corrections are
welcome by email to the corresponding author, seb@composo.ai.

It is not a complete record of how the study got here. Superseded approaches, abandoned
arms and the order in which things were tried are not in it, with two deliberate
exceptions the papers report as results: the optimisation lineages are here in full,
because both papers make claims about what the search found, and the deployed judge's
superseded prompt wordings are here because a paper compares them. Every other file was
kept because someone re-running this work would need it, and everything else was left
out because it was working residue - one-offs, superseded drafts, and records of the
authors' own process rather than of the study.


## Repeating the study

This is the shortest complete path from a clone to a number, and it is the thing the
repository is for. Download the dataset, tell the runners where its two halves are, and
run an arm:

    pip install -r requirements.txt
    export OPENROUTER_API_KEY=...
    export OMISSIONBENCH_TRANSCRIPTS=<dataset>/transcripts

    export PAIRS=<dataset>/pairs/dataset_v2.json

    python3 judges/w2_arms.py --list                              # the 20 judge designs
    python3 judges/w2_arms.py --plan --arms strong --dataset $PAIRS   # size a run: judgements and calls, no calls made
    python3 judges/w2_grid.py --smoke --dataset $PAIRS            # 2 pairs x 8 designs, confirms before spending
    python3 judges/w2_grid.py --cells all --runs 3 --dataset $PAIRS -y   # the full eight-design ablation
    python3 judges/w2_analyze.py                                  # metrics and intervals off the finished run

Two paths are the reader's to supply and neither needs the code edited. `--dataset` is the
pair set: the runners look for it at `master/dataset_v2.json`, and the dataset repository
ships it as `pairs/dataset_v2.json`. `OMISSIONBENCH_TRANSCRIPTS` is the transcripts each
note is scored against: the runners' own sources for those are the study's working-tree
files, which are not released, so without this variable a run stops at the first
consultation with `no transcript for ...`. Everything else resolves inside the clone.

`--plan` sizes a run without making a call: how many judgements, how many model calls, and
which store each arm would write to. `--smoke` runs two representative pairs through every
cell, and `--limit N` caps a run to the first N pairs. Every runner counts its calls and
asks before spending, unless you pass `-y`; `--warn-usd` and `--stop-usd` set spending
rails it enforces between blocks.

The same two paths drive `judges/w2_pipeline.py` (the pipeline judge), `judges/w2_baselines.py` (the
published baselines), `judges/w2_v14_arms.py` (the deployed judge's three wordings) and
`judges/w2_power.py` (the reasoning-budget ladder). `judges/w2_analyze.py` reads a finished run's store
and prints the metrics with their intervals.

One input this release genuinely cannot carry is PriMock57's parsed transcripts in the
shape the *construction* scripts expect, because they belong to PriMock57's own release.
Judging does not need them - the released `transcripts/` covers every consultation the
released pairs use - but rebuilding the corpus from scratch does: get them from PriMock57
and set `PRIMOCK_PARSED` to the file. The modules that need them read it through
`common.PRIMOCK`, so that one variable covers all of them.

## What is where

A few filenames carry the study's own stage prefixes - `w1` the reproducibility regime,
`w2` the judge benchmark, `w_a` the pair-set ground-truth gate, `wb` the external
anchors, `wd` the corpus-construction gates. They are kept because both papers cite
these scripts by filename. Where a comment compares against "June", it means the June
2026 pilot capture and its discovery pass, the reference point the census instrument was
built to improve on.


**Start here**

| file | what it does |
|---|---|
| `README.md` | what this repository is, how to install and run it, and the map below |
| `LICENSE` | MIT, for the code; the data at the dataset repository is CC BY 4.0 |
| `CITATION.cff` | how to cite the two papers and this repository |
| `record/VERIFICATION.md` | what was checked over both repositories before publication, and how |
| `requirements.txt` | the dependencies the code imports, at the versions the study ran |
| `models.lock.json` | every model pin: role, provider, slug, sampling settings |
| `common.py` | the shared client: the two model transports, the corpus loaders, the failure taxonomy |
| `w2_common.py` | the judge-run scaffolding: pinned calls, replicates, the per-run manifest |
| `taxonomy_common.py` | the census substrate: routing, parsing, and the reporting frame's checks |
| `taxonomy_frame.json` | the published reporting frame: the two-tier taxonomy and all twelve discovery passes |
| `record/check_manifests.py` | the validator every released run manifest was written to satisfy |

**Capturing the products' notes**

| file | what it does |
|---|---|
| `capture/scribe_A_generate.py` | generates Scribe A's notes through its documented API, two templates per consultation |
| `capture/build_scribe_B_queue.py` | the ordered upload queue Scribe B's capture works through |
| `capture/scribe_B_batch.py` | the record of Scribe B's first capture: browser automation, not a runnable script |
| `capture/scribe_B_overnight.py` | the maintained form of that capture, which verifies by content what it captured |
| `capture/scribe_B_match.py` | attributes a captured note to its consultation by content rather than by upload order |
| `capture/build_scribe_C_queue.py` | the ordered capture queue for Scribe C, which had to be dictated to in real time |
| `capture/scribe_C_batch.py` | the record of Scribe C's capture: browser automation, not a runnable script |
| `capture/ingest_scribe_notes.py` | turns the captured notes into the corpus format the census reads |
| `capture/build_audio_manifest.py` | the committed record of every capture audio file and its checksum |

**Rendering the audio**

| file | what it does |
|---|---|
| `audio/tts_consult.py` | renders an authored consultation as two-voice audio; the model and voice pins are at its top |
| `audio/tts_aci_adapter.py` | the same for ACI-Bench encounters |
| `audio/tts_batch.py` | drives both over a whole stratum |
| `audio/make_silence_variant.py` | the silence-compressed variant built for the speed test the study then rejected |

**Building the corpus and its answer keys**

| file | what it does |
|---|---|
| `corpus/fetch_acibench.py` | fetches and normalises the ACI-Bench release |
| `corpus/subsample_acibench.py` | draws the seeded subsample, excluding encounters a prior study had used |
| `corpus/author_scenarios.py` | authors the original consultation scenarios and their answer keys |
| `corpus/author_trapblind.py` | authors the trap-blind scenarios: same procedure, no planted error |
| `corpus/critique_scenarios.py` | the adversarial critic panel over newly authored scenarios |
| `corpus/extract_fact_sheets.py` | blind fact-sheet extraction from a transcript, with no note in view |
| `corpus/critique_extracted.py` | the three-critic support, materiality and leakage audit over those sheets |
| `corpus/consolidate_sheets.py` | consolidates a sheet down to its core assertions |
| `corpus/make_audit_sample.py` | the seeded draw for the human audit of the critiqued sheets |
| `corpus/audit_reference_notes.py` | audits each corpus's own reference notes against their transcripts, and repairs them |
| `corpus/compare_primock_v1_v2.py` | the two extraction instruments compared on the same transcripts |
| `corpus/primock_trajectory.py` | the full record of how the extraction instrument moved from a 33% drop rate to 7% |
| `corpus/wd_gate_metrics.py` | the scoreboard for the construction gates, per corpus |
| `corpus/wd_provenance_compare.py` | the two blind-recovery readings of where each fact sheet's facts came from |

**Building and grading the pairs**

| file | what it does |
|---|---|
| `benchmark/omission_sites.py` | the fact-site map: where in a note each fact is actually stated |
| `benchmark/inject_omissions_v2.py` | builds complete and partial omission pairs by deleting from the site map |
| `benchmark/verify_omissions_v2.py` | the cross-family verification panel each graded pair had to pass |
| `benchmark/hard_negatives.py` | the manufactured-failure recipes: what a fabrication, an alteration and a removal look like |
| `benchmark/hard_negatives_balanced.py` | balances those recipes across the three error types |
| `benchmark/hard_negatives_master.py` | builds the master ideal notes and the balanced pair set over the whole corpus |
| `benchmark/relabel_partial_seed.py` | formally relabels the earlier seed pairs whose omission was partial |
| `benchmark/grade_salience_importance.py` | the salience and importance grades over the authored scenarios |
| `benchmark/grade_partial_seed_severity.py` | backfills rubric severity onto the seed set's ungraded pairs |
| `benchmark/regrade_severity.py` | the rubric-anchored regrade: two independent graders, conservative tie-break |
| `benchmark/omission_factorial.py` | per-fact severity, then the severity-by-residual allocation the released set is built on |
| `benchmark/residual_location.py` | where the surviving mention sits when an omission is only partial |
| `benchmark/build_dataset_v2.py` | assembles the one file holding every pair the papers evaluate on |
| `benchmark/w_a_master.py` | the ground-truth gate: the audit a pair set had to pass before any judge scored it |

**The judge benchmark**

| file | what it does |
|---|---|
| `record/w1_smoke.py` | the reproducibility regime's own validation run |
| `judges/w2_arms.py` | the registry of every judge design compared, and the runner for them |
| `judges/w2_grid.py` | the eight-cell ablation: criterion scope, output format, ensembling |
| `judges/w2_baselines.py` | the published baselines, re-implemented openly rather than called as libraries |
| `judges/w2_v14_arms.py` | the deployed faithfulness judge's three wordings, run as arms |
| `judges/v14_judge.py` | that judge as a standalone yardstick, against the optimiser's winner |
| `judges/w2_factlist.py` | the fact-list-in-prompt control: the missing cell of the mechanism triad |
| `judges/w2_pipeline.py` | the pipeline judge: extract the facts, check each one, refute the flags |
| `judges/w2_power.py` | the reasoning-budget ladder: does test-time compute close the gap |
| `judges/w2_secondfamily_smoke.py` | the candidate-judge smoke for the replication on a second model family |
| `judges/w2_analyze.py` | the shared metrics, intervals and handoff every read-out below uses |
| `judges/w2_power_stats.py` | the statistical treatment the reasoning-budget numbers need |
| `judges/w2_evalfull_analyze.py` | read-out: the extended evaluation set |
| `judges/w2_factlist_analyze.py` | read-out: the fact-list control |
| `judges/w2_pipeline_analyze.py` | read-out: the pipeline judge |
| `judges/w2_pipeline_replicates_analyze.py` | read-out: the pipeline judge's replicate round |
| `judges/w2_power_analyze.py` | read-out: the reasoning-budget ladder |
| `judges/w2_robustness_slice.py` | two robustness re-reads of records already stored, with no new model calls |
| `judges/w2_secondfamily_analyze.py` | read-out: the second-family replication |
| `judges/w2_sf_pipeline_analyze.py` | read-out: the pipeline judge with a second-family checker |

**External anchors**

| file | what it does |
|---|---|
| `anchors/wb_adapters.py` | feeds third-party corpora to the same judges through the same prompts |
| `anchors/wb_medec_ingest.py` | ingests the MEDEC corpus and builds its clean twins |
| `anchors/wb_run_medec.py` | runs two judge arms against MEDEC's physician-labelled texts |
| `anchors/wb_medec_analyze.py` | read-out: the MEDEC anchor |
| `anchors/wb_medval_ingest.py` | ingests MedVAL-Bench and maps its taxonomy onto this study's |
| `anchors/wb_overlap_check.py` | measures how far MedVAL-Bench's dialogue2note split overlaps ACI-Bench |
| `anchors/wb_medval_coanalysis_check.py` | both sides of the rule for handling that overlap, in one place |

**The census of deployed products**

| file | what it does |
|---|---|
| `census/discover.py` | the broad discovery pass over a note and its transcript |
| `census/verify_findings.py` | the adversarial panel that tries to refute every candidate finding |
| `census/taxonomy_discover.py` | the twelve targeted discovery passes over the whole corpus |
| `census/taxonomy_verify.py` | the two-family refutation panel over what they found |
| `census/taxonomy_cluster.py` | clusters the verified findings into subcategories, with a stability sweep |
| `census/taxonomy_analyze.py` | the census rates, their replication and the published tables |
| `census/taxonomy_omission_audit.py` | why the panel refused the omissions it refused |
| `census/second_panel.py` | the same candidates read again under four review standards |
| `census/second_panel_analyze.py` | read-out: what survives each standard |
| `census/second_panel_classmix.py` | the class mix each standard verifies |
| `census/census_note_ci.py` | consultation-clustered intervals for the note-level rates |
| `census/census_sensitivity_noprefill.py` | the census net of the classes an electronic record would have prefilled |
| `census/build_census_realerror.py` | turns the census subset into pairs the judges can be scored on |
| `census/build_census_realerror_subset.py` | draws that subset |
| `census/census_realerror_analyze.py` | read-out: the judges on real product errors |
| `census/dedup_estimate.py` | how many distinct errors the verified findings represent |
| `census/position_bias_ci.py` | uncertainty for the verification position-bias check |
| `pilot/scripts/cross_scribe_matches.py` | the failure-family map four of the modules above load by path |

**Clinician validation**

| file | what it does |
|---|---|
| `sittings/build_sitting_pack.py` | assembles the author-clinician's blinded sitting |
| `sittings/build_severity_pack.py` | assembles the blinded severity-rubric sitting |
| `sittings/precision_sitting.py` | assembles the blinded adjudication of sampled verified findings |
| `sittings/precision_sitting_analyze.py` | unblinds and scores it |
| `sittings/second_clinician_sitting.py` | assembles the offline adjudication app an independent clinician used |
| `sittings/second_clinician_analyze.py` | unblinds and scores that sitting |

Four directories are not in that table, because they hold prompts and records rather than
code you run. `w2_prompts/` and `prompts/` are the prompts, with their hashes. `gepa/` is
the three optimisation campaigns: the optimisers, their seeds, every candidate's full
instruction text, the winning prompts, and the design notes for the second and third
campaigns. `provenance/` holds the construction records the appendices cite. Those four
keep the paths the appendices print for them, which is why the layout has a flat `gepa/`
and `prompts/` beside the topic directories rather than one tidy scheme. The scripts were
free to move because the papers name every script by filename alone.

`record/` holds the release's own bookkeeping: the data behind every measured figure and
table, one file per float, in `record/figures/`; the manifest validator; the
reproducibility regime's own validation run; and `record/VERIFICATION.md`. The schematics and
the hand-authored rosters carry no data file there, because they carry no measurement of
their own. And `_modulepath.py`, the one root file not in the map above, is what makes
the grouped layout work: it puts the topic directories on the import path, so modules
keep importing each other by bare name.

The data lives separately, at `huggingface.co/datasets/ComposoAI/OmissionBench` under CC BY 4.0.
This repository is the code, under MIT: capture scripts, every prompt used to build,
verify or judge any layer, the prompt-optimisation campaigns' lineage records with every
candidate's full instruction text, judge configurations and `models.lock.json`. The per-run
manifests are in the dataset repository rather than here, one per judge run at
`judgements/judges/<family>/<run>/manifest.json`, each with that run's raw completions
beside it. `record/figures/` holds the data behind every figure and table in the two papers, one
file per float; `record/figures/SOURCES.md` records which artifacts each of those numbers came
from, and says which of the code that drew them is here and which is not. `pilot/` holds a
single file, the failure-family matcher that four modules here load by path.

`record/check_manifests.py` is the validator for those manifests. Point it at them and it checks
every run's identity, status, model pins, call counts and cost accounting:

    python3 record/check_manifests.py --results <dataset>/judgements/judges --no-inputs

`--no-inputs` is needed because each manifest also records the sha256 of the construction
files that run read, and those files are not part of the release. On the working-tree gate,
the validator checks what the runs actually promised: from 12 August 2026 the study stopped
requiring a clean tree, because analysis code changed between runs, and required instead
that every run record its commit together with each uncommitted path. All 240 released
manifests satisfy that; the 204 that ran dirty carry `git_dirty_file_count` rather than
the filenames, because those named notes this release withholds, and the count is what the
validator checks.

How far a replicator can get depends on which product. Scribe A is driven through a
documented API, and `capture/scribe_A_generate.py` regenerates its notes end to end for anyone with
their own account, once the addresses and template identifiers the anonymisation removed
are restored. Scribes B and C had no API for this, so their notes were captured by driving
each product's web interface - one uploading audio, one playing it into the product in real
time through a virtual audio device. Those capture scripts are published as the record of
how the capture ran, not as a route anyone can re-run: they depend on a browser-automation
tool that is not part of this release, on a macOS audio setup, and on interface selectors
that were current in August 2026. What no replicator can get from us in any case is the
products' note text: it is withheld for licence reasons.

## Installing and running

There is no packaging here. Clone the repository and run the scripts from its root.
`requirements.txt` lists the ten packages this code imports, at the versions the study's
own environment recorded, against Python 3.14: `pip install -r requirements.txt`. It is a
dependency list rather than a snapshot of that environment - the environment was a working
tree with other projects in it - so nothing here claims these are the only versions that
work, only that they are the ones the runs were made under.

Keys are read from a `secrets.env` file beside the scripts, or from the environment. Every
paper-bound model call routes through OpenRouter, so `OPENROUTER_API_KEY` is the one you
always need. The one exception is `judges/v14_judge.py`, the standalone yardstick, which
takes the subscription path below; no published benchmark number depends on it. `OPENAI_API_KEY` covers the OpenAI batch route and the text-to-speech scripts,
and `COHERE_API_KEY` the embeddings behind clustering and matching.

### The `claude` binary

Some of these scripts need more than a key. About twenty of them do not use an API at
all: they run a one-shot prompt by shelling out to `claude`, the Claude command-line
tool, which bills a Claude subscription rather than an API key. `common.claude()` and
`common.claude_json()` are that path, and everything that imports either of them takes it -
fact-sheet extraction, scenario authoring, the critique panels, the severity and salience
graders, and the census instruments in `census/discover.py` and `census/verify_findings.py`. Without that
binary on `PATH` they raise `FileNotFoundError`, and no environment variable fixes it.

Two things are worth knowing before you go looking for it. None of the benchmarking the
papers report runs on that path: every judge call goes through `common.llm()`, which routes
to OpenRouter on a role pinned in `models.lock.json` and asserts that provider rather than
accepting another. And the census stack already prefers OpenRouter: `taxonomy_common.py`
runs the discovery and verification passes through `llm()` by default and reaches the
binary only under `--route plan`, which is why the taxonomy runners work with the key alone.

If you do not have the binary, set `BATCH_LLM=openai` and an `OPENAI_API_KEY`. That sends
every `claude()` and `claude_json()` call to a strong OpenAI model instead, which is the
same switch the study used when it hit a subscription limit. The prompts, the pass counts,
the panel sizes and the parsing are identical either way; only the transport moves.

`common.py` is the shared client and the pinned-call wrapper, `w2_common.py` the judge-run
scaffolding, `models.lock.json` the model pins. Start at `judges/w2_grid.py` for the eight
ablation designs, `judges/w2_arms.py` for the reference judges and `judges/w2_pipeline.py` for the
pipeline judge.

## Anonymisation

The three scribe systems are Scribe A, B and C throughout. Some capture scripts originally
carried product names in their file names and configuration keys, and the packaging step
rewrote them to the letters. A released capture script therefore names its driving mode - a
documented API, or audio played to the product in real time - and carries no product name.

The rewrite reached further than names. In `common.py` and `capture/scribe_B_overnight.py` the
letters also stand inside the web and API addresses those two clients call, an OAuth token
endpoint, an HTTP request header and the environment variables they read (`scribe_A_ENV`,
`scribe_A_TENANT`, `scribe_B_WAV` and the rest), so none of those addresses resolve as
shipped. One vendor's API takes two live template identifiers, and those were replaced with
environment lookups the same way (`SCRIBE_A_TEMPLATE_SHORT`, `SCRIBE_A_TEMPLATE_DETAILED`).
Running a capture client against your own account therefore means restoring its addresses,
its OAuth token endpoint, its two custom header names, its documents path and its template
identifiers from that vendor's own documentation first. Each is read from an environment
variable named in `common.py`, so nothing needs editing: supply `SCRIBE_A_API_BASE`,
`SCRIBE_A_TOKEN_URL`, `SCRIBE_A_TENANT_HEADER`, `SCRIBE_A_RETENTION_HEADER`,
`SCRIBE_A_DOCUMENTS_PATH` and the two `SCRIBE_A_TEMPLATE_*` ids alongside the credentials. What the study did through that API is
unchanged and fully described: two note templates per consultation, one request per note.
Everywhere else in the repository the letters are labels and directory names only, and
nothing downstream of capture needs repair. The letter-to-product mapping is not in this
repository in any form.

## Audio

The synthesised consultation audio is not redistributed, but it is regenerable from the
released transcripts. `audio/tts_consult.py` renders authored and trap-blind consultations,
`audio/tts_aci_adapter.py` does the same for ACI-Bench, and `audio/tts_batch.py` drives both; the model
and per-speaker voice pins are constants at the top of each. One adaptation step comes
first: all three read a scenario JSON that this repository does not carry, rather than the
transcript files themselves, so wrap the dataset's `transcripts/<stratum>/*.txt` into a JSON
list of `{"id": <filename without .txt>, "transcript": <file contents>}`, then either pass
that file to the single-consultation scripts with `--from` or save it at the paths
`audio/tts_batch.py`'s `SOURCES` map names. PriMock57's own recordings were never redistributed -
they belong to PriMock57's release.

## Verification

`record/VERIFICATION.md` records what was checked over both repositories before
publication, and how. The checking code itself stays private, for the reason given there.

## Prompts

Every prompt used to build, verify or judge any layer is here.

- `w2_prompts/` - the clinical prompts: the eight ablation designs, the three wordings of
  the deployed judge, the engineered completeness judge, the published baselines this study
  re-implements, the fact-list control and the five pipeline stages.
- `gepa/seeds/seed_*.txt`, `gepa/judge_prompt.txt` and `gepa/*lineage.jsonl` - the optimiser's
  seed prompts, its winners, and every candidate's full instruction text across all three
  campaigns.
- `prompts/extraction_prompt_v1.txt` and `_v2.txt` - both fact-sheet extraction prompts,
  the superseded one included.
- The census instruments - the eleven targeted discovery passes and the open one, the
  two-family refutation panel, the salience rater, the rubric-anchored severity grader and
  the completeness critic, plus the lenient single-call re-read the census paper measures
  the strict standard against - are inline in `census/discover.py`, `census/verify_findings.py`,
  `census/taxonomy_verify.py` and `census/second_panel.py`, next to the parsing they depend on. The twelve
  discovery passes are declared in `taxonomy_frame.json`, which is the single definition of
  which passes a discovery run makes and carries the instructions of the two passes added
  for the published classes; the other nine targeted instructions are pulled verbatim from
  `census/discover.py`'s MODE_PASSES at import, so the frame cannot silently reword them.
- The construction and audit instruments - the four audit sections, the repair and
  regeneration prompts, the site-mapping instruments - are inline in the scripts that
  issue them, next to the parsing they depend on.

The prompt directory carries its own `w2_prompts/PROMPTS.sha256`, and every prompt in it hashes to what that
file records.

The severity rubric travels with the data rather than the code: it is embedded verbatim in
`validation/sitting_severity_items.json` in the dataset repository, which is the artefact
that shows what the graders were actually given.

Model pins, seeds, transports and prompt hashes for each stage are in the judge paper's
Appendix D (reproducibility) and in `models.lock.json`.

## What is not here

Two things a reader might look for are deliberately absent. The **severity rubric** - the
written instrument two graders applied to every trap - travels with the data rather than
the code: it is embedded verbatim, with its SHA-256, in the dataset repository's
`validation/sitting_severity_items.json`, which is the artefact that shows what the graders
were actually given. The scripts that grade against it look for it at
`master/severity-rubric.md`, and `benchmark/hard_negatives_master.py`,
`benchmark/regrade_severity.py` and `sittings/build_severity_pack.py` also take
`SEVERITY_RUBRIC` as an override. Saving it at that path satisfies all of them.

The **checking code** that scanned this bundle before publication is not here either, and
`record/VERIFICATION.md` gives the reason: a checker that searches for forbidden strings
has to contain them, so publishing it would publish exactly what it exists to keep out.
What ships instead is a description of what was checked and how.

## Licence

MIT. See `LICENSE`. The data at `ComposoAI/OmissionBench` is CC BY 4.0 and carries its own
attribution chain.
