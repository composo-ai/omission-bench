# OmissionBench - harness

This repository is the code behind two companion papers released alongside it, and it
generated every number they report. The judge paper, *LLM Judges Verify Presence, Not
Absence: Omission Blindness in AI Clinical Notes and What Recovers It*, introduces the
benchmark, finds that judges verify what a note contains while sitting near chance on what
it omits, and reports what recovers omission detection. The census paper, *One note in
three: a verified census of three deployed AI scribes, and the instrument that counted it*,
reports the audit of the three products.

The data lives separately, at `huggingface.co/datasets/ComposoAI/OmissionBench` under CC BY 4.0.
This repository is the code, under MIT: capture scripts, every prompt used to build,
verify or judge any layer, the prompt-optimisation campaigns' lineage records with every
candidate's full instruction text, judge configurations and `models.lock.json`. The per-run
manifests are in the dataset repository rather than here, one per judge run at
`judgements/judges/<family>/<run>/manifest.json`, each with that run's raw completions
beside it. `figures/` holds the data behind every figure and table in the two papers, one
file per float; `figures/SOURCES.md` records which artifacts each of those numbers came
from, and says which of the code that drew them is here and which is not. `pilot/` holds a
single file, the failure-family matcher that four modules here load by path.

`check_manifests.py` is the validator for those manifests. Point it at them and it checks
every run's identity, status, model pins, call counts and cost accounting:

    python3 check_manifests.py --results <dataset>/judgements/judges --no-inputs

`--no-inputs` is needed because each manifest also records the sha256 of the construction
files that run read, and those files are not part of the release. One of the validator's
gates is stricter than the regime the benchmark runs actually ran under: it requires a
clean working tree, and `w2_common.py` records that gate as deliberately not enforced from
12 August 2026, with each manifest disclosing its exact uncommitted paths instead. So most
benchmark runs report that gate as a failure, and every one of them carries the disclosure
the relaxation asks for. No other gate is affected.

How far a replicator can get depends on which product. Scribe A is driven through a
documented API, and `scribe_A_generate.py` regenerates its notes end to end for anyone with
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
`requirements.txt` is the pinned dependency set the study ran on, resolved against Python
3.14: `pip install -r requirements.txt`.

Keys are read from a `secrets.env` file beside the scripts, or from the environment. Every
paper-bound model call routes through OpenRouter, so `OPENROUTER_API_KEY` is the one you
always need. `OPENAI_API_KEY` covers the OpenAI batch route and the text-to-speech scripts,
`COHERE_API_KEY` the embeddings behind clustering and matching, and `COMPOSO_API_KEY` the
judge called through our own evaluation API in `crossover_balanced.py`.

### The `claude` binary

Some of these scripts need more than a key. About twenty of them do not use an API at
all: they run a one-shot prompt by shelling out to `claude`, the Claude command-line
tool, which bills a Claude subscription rather than an API key. `common.claude()` and
`common.claude_json()` are that path, and everything that imports either of them takes it -
fact-sheet extraction, scenario authoring, the critique panels, the severity and salience
graders, and the census instruments in `discover.py` and `verify_findings.py`. Without that
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
scaffolding, `models.lock.json` the model pins. Start at `w2_grid.py` for the eight
ablation designs, `w2_arms.py` for the reference judges and `w2_pipeline.py` for the
pipeline judge.

Two things you have to supply. The runners look for their pairs at
`master/dataset_v2.json`, and the dataset repository ships that file as
`pairs/dataset_v2.json`, so pass `--dataset` with its path. And two paths still point where
our own working copy kept them rather than into this tree: PriMock57's parsed transcripts in
`common.py`, which you rebuild from PriMock57's own release, and the deployed judge's prompt
in `w2_arms.py`, which is released here as `w2_prompts/v14_as_shipped.txt`.

## Anonymisation

The three scribe systems are Scribe A, B and C throughout. Some capture scripts originally
carried product names in their file names and configuration keys, and the packaging step
rewrote them to the letters. A released capture script therefore names its driving mode - a
documented API, or audio played to the product in real time - and carries no product name.

The rewrite reached further than names. In `common.py` and `scribe_B_overnight.py` the
letters also stand inside the web and API addresses those two clients call, an OAuth token
endpoint, an HTTP request header and the environment variables they read (`scribe_A_ENV`,
`scribe_A_TENANT`, `scribe_B_WAV` and the rest), so none of those addresses resolve as
shipped. One vendor's API takes two live template identifiers, and those were replaced with
environment lookups the same way (`SCRIBE_A_TEMPLATE_SHORT`, `SCRIBE_A_TEMPLATE_DETAILED`).
Running a capture client against your own account therefore means restoring its addresses,
header names and template identifiers from that vendor's own documentation first.
Everywhere else in the repository the letters are labels and directory names only, and
nothing downstream of capture needs repair. The letter-to-product mapping is not in this
repository in any form.

## Audio

The synthesised consultation audio is not redistributed, but it is regenerable from the
released transcripts. `tts_consult.py` renders authored and trap-blind consultations,
`tts_aci_adapter.py` does the same for ACI-Bench, and `tts_batch.py` drives both; the model
and per-speaker voice pins are constants at the top of each. One adaptation step comes
first: all three read a scenario JSON that this repository does not carry, rather than the
transcript files themselves, so wrap the dataset's `transcripts/<stratum>/*.txt` into a JSON
list of `{"id": <filename without .txt>, "transcript": <file contents>}`, then either pass
that file to the single-consultation scripts with `--from` or save it at the paths
`tts_batch.py`'s `SOURCES` map names. PriMock57's own recordings were never redistributed -
they belong to PriMock57's release.

## Verification

`release/VERIFICATION.md` records what was checked over both repositories before
publication, and how. The checking code itself stays private, for the reason given there.

## Prompts

Every prompt used to build, verify or judge any layer is here.

- `w2_prompts/` - the clinical prompts: the eight ablation designs, the three wordings of
  the deployed judge, the engineered completeness judge, the published baselines this study
  re-implements, the fact-list control and the five pipeline stages.
- `we_prompts/` - the frozen pre-registered prompts for a planned second-domain replication
  on news summarisation, which reads a source article and a summary rather than a transcript
  and a note. Same four `grid_*` file names as the clinical set, plus that workstream's own
  construction and verification instruments. Neither paper reports a result from it, and the
  dataset repository carries no data for it.
- `gepa/seeds/seed_*.txt`, `gepa/judge_prompt.txt` and `gepa/*lineage.jsonl` - the optimiser's
  seed prompts, its winners, and every candidate's full instruction text across all three
  campaigns.
- `prompts/extraction_prompt_v1.txt` and `_v2.txt` - both fact-sheet extraction prompts,
  the superseded one included.
- The census instruments - the eleven targeted discovery passes and the open one, the
  two-family refutation panel, the salience rater, the rubric-anchored severity grader and
  the completeness critic, plus the lenient single-call re-read the census paper measures
  the strict standard against - are inline in `discover.py`, `verify_findings.py`,
  `taxonomy_verify.py` and `second_panel.py`, next to the parsing they depend on. The twelve
  discovery passes are declared in `taxonomy_frame.json`, which also carries each pass's instruction and is the
  single definition of which passes a discovery run makes.
- The construction and audit instruments - the four audit sections, the repair and
  regeneration prompts, the site-mapping instruments - are inline in the scripts that
  issue them, next to the parsing they depend on.

`w2_prompts/` and `we_prompts/` each carry their own `PROMPTS.sha256`.

The severity rubric travels with the data rather than the code: it is embedded verbatim in
`validation/sitting_severity_items.json` in the dataset repository, which is the artefact
that shows what the graders were actually given.

Model pins, seeds, transports and prompt hashes for each stage are in the judge paper's
Appendix D (reproducibility) and in `models.lock.json`.

## Documents this repository cites but does not contain

Scripts and docstrings here cite the study's own working documents - `docs/FINDINGS.md`
(the internal results log, cited as `FINDINGS section NN`), the pre-registered `specs/`,
and the project's planning notes. None of them is part of this release: they are working
documents, not deliverables, and several name the three scribe products directly. Where a
comment cites `FINDINGS section NN`, the citable form of the same number is the
corresponding section of one of the two papers, which carry every result with its
denominator and its interval. Nothing in this repository depends on those files at
runtime.

Those documents also leave their labels behind in file names, docstrings and manifests.
This is the key to them.

| Code | What it was |
|---|---|
| **W1** | The reproducibility regime, not an experiment: pinned models, deterministic per-call seeds, a manifest per run, and the gates a paper-bound run has to pass (`G1`, that every number in a paper resolves to valid runs; `G3`, the per-run structural checks). Implemented in `common.llm()`, `w2_common.Run` and `check_manifests.py`. |
| **W2** | The judge benchmark. The eight-design ablation grid, the reference judges and the published baselines, the fact-list control, the two-stage pipeline, the second judge family and the reasoning-budget power runs - every `w2_*.py` file, and every judge family in the dataset repository. |
| **W-A** | Ground-truth verification: the audit a pair set had to pass before any judge was allowed to score it. `w_a_master.py`, and its gate record at `provenance/wa_gate_report.json`. |
| **W-B** | External anchors. The same judges run against third-party corpora that clinicians outside this study labelled, so at least one measurement does not rest on ground truth we built ourselves. The `wb_*.py` files. |
| **W-D** | The master dataset and the census over it: note capture, fact-sheet extraction and critique, reference-note repair, the twelve discovery passes and the verification panel. `wd_*.py`, `taxonomy_*.py`, `discover.py`, `verify_findings.py`. |
| **WD-R2** to **WD-R5** | Numbered readings inside W-D, one question each. R2 compares trapped against trap-blind consultations; R3 is the reference-note audit and repair, recorded at `provenance/wd_r3_report.json`; R4 and R5 are the two blind-recovery readings of fact-sheet provenance. |
| **W-E** | A pre-registered second-domain replication on news summarisation. Its instruments are frozen in `we_prompts/` and its construction code is in the `we_*.py` files; it was never run, neither paper reports a result from it, and the dataset repository carries no data for it. |
| **W-F** | The prompt-optimisation campaigns, all three of them. The `gepa/` directory. |
| **P1**, **P2** | The two papers: P1 is the judge paper, P2 the census paper. |

Two more citation forms appear in comments. **"Amendment YYYY-MM-DD A3"** and its
relatives are dated amendments to those pre-registered specs, lettered within each date;
each one records a decision taken after a spec was written and before the runs it governs -
which provider a role routes through, what a manifest must carry, which substrate a run may
load. And a **`v14`** prefix on a file, arm or results directory names the deployed
faithfulness judge the study benchmarks its own designs against; the papers call it by that
name and print no version.

## Licence

MIT. See `LICENSE`. The data at `ComposoAI/OmissionBench` is CC BY 4.0 and carries its own
attribution chain.
