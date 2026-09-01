"""Validate a tree of run manifests: identity, status, model pins, call counts and cost
accounting, one manifest per run.

Modes:
    python record/check_manifests.py [--results DIR] [--inputs-root DIR | --no-inputs]
        Walk a tree of <family>/<run>/manifest.json (default: results/ beside this
        script; the released manifests are the dataset repository's judgements/judges/),
        and validate every run manifest: status complete, git_dirty false,
        models non-empty with resolved IDs (+ v2: provider/route/k_impl), input sha256s
        still match the files on disk, cost_usd present, calls.errors == 0.

    python record/check_manifests.py --paper paper_numbers.json
        The pre-submission check: paper_numbers.json maps every figure/table/inline
        stat to its evidence. Entry forms:
            "fig_w2_grid": {"runs": ["<run_id>", ...], "experiment": "w2-ablation",
                            "stochastic": true}
            "wa_audit_m1": {"plan_path": true, "file": "w_a_report.json"}
        Fails if any run_id is missing or invalid, a stochastic entry has <3 complete
        replicates, a paper-bound API run's inputs lack pairs_master_frozen.json at the
        recorded freeze hash (freeze hashes live in freeze_hashes.json at the harness
        root - {filename: sha256}, appended when the pair set was frozen), or a plan-path
        entry's output file lacks an embedded manifest block (the relaxed check).

Each manifest also records the input files that run read, with their sha256, and by
default those are resolved against this script's directory and re-hashed. The released
manifests name the study's own construction corpora, which are not part of the release, so
validating the released manifests means turning that check off:

    python record/check_manifests.py --results <dataset>/judgements/judges --no-inputs

Exit code 0 = all pass, 1 = any failure. manifest_version 2 required; v1 accepted for
the earliest w1-smoke runs only.
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import argparse, glob, hashlib, json, os, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repository root
#: Root of the manifest tree, one level above <run>/. Overridden by --results.
RESULTS = os.path.join(HERE, "results")
#: What a manifest's recorded input paths are resolved against. Separate from RESULTS: a
#: manifest names its inputs relative to the harness root, not relative to the run.
INPUTS_ROOT = HERE
#: Set by --no-inputs, for validating manifests whose input files are not to hand.
CHECK_INPUTS = True
FREEZE_FILE = os.path.join(HERE, "freeze_hashes.json")
MASTER_PAIRS = "pairs_master_frozen.json"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: Experiments whose runner writes its own manifest instead of going through `Run()`,
#: a deliberate divergence, not an oversight. A search loop that scores hundreds of
#: candidates cannot open a run per model call, so these manifests carry the run's identity,
#: models, status and totals but not the per-call fields `Run()` fills in. Their gaps are
#: reported as warnings rather than failures - visible, named, and not counted as defects.
SELF_MANIFESTING = {"gepa-omission"}


def validate_manifest(path, problems, warnings=None):
    """Validate one manifest.json. Appends 'run_id: problem' strings; returns the manifest.
    warnings collects non-fatal notes (currently: input-hash drift on w1-smoke runs, which
    judge pre-freeze construction files that legitimately mutate - smoke runs are allowed
    on pre-audit files; every OTHER experiment hard-fails on drift)."""
    if warnings is None:
        warnings = []
    rel = os.path.relpath(path, RESULTS)
    try:
        m = json.load(open(path))
    except Exception as ex:
        problems.append(f"{rel}: unreadable manifest ({ex})")
        return None
    rid = m.get("run_id", rel)
    ver = m.get("manifest_version")
    if ver == 1 and m.get("experiment") != "w1-smoke":
        problems.append(f"{rid}: manifest_version 1 only grandfathered for w1-smoke, got experiment {m.get('experiment')!r}")
    elif ver not in (1, 2):
        problems.append(f"{rid}: bad manifest_version {ver!r}")
    run_dir = os.path.dirname(path)
    if os.path.basename(run_dir) != m.get("run_id"):
        problems.append(f"{rid}: run_id does not match its directory name {os.path.basename(run_dir)!r}")
    if m.get("experiment") != os.path.basename(os.path.dirname(run_dir)):
        problems.append(f"{rid}: experiment {m.get('experiment')!r} does not match parent dir")
    if m.get("status") != "complete":
        problems.append(f"{rid}: status {m.get('status')!r} != complete")
    if m.get("git_dirty") is not False:
        # From 12 August 2026 the study stopped requiring a clean tree - analysis code
        # changed between runs - and required instead that every run record its commit
        # together with each uncommitted path. Released manifests carry the count rather
        # than the filenames (the paths named withheld material); the count is checked.
        if m.get("git_dirty_file_count") is None:
            problems.append(f"{rid}: git_dirty is {m.get('git_dirty')!r} (must be false; "
                            "a run that discloses how many files were uncommitted is exempt)")
    if not m.get("git_commit"):
        problems.append(f"{rid}: missing git_commit")
    models = m.get("models") or []
    if not models:
        problems.append(f"{rid}: models is empty")
    for mm in models:
        if not mm.get("resolved"):
            problems.append(f"{rid}: model role {mm.get('role')!r} has no resolved ID")
        if ver == 2:
            if mm.get("provider") != "openrouter":
                problems.append(f"{rid}: role {mm.get('role')!r} provider {mm.get('provider')!r} != openrouter")
            # The optimisation campaigns write their own manifests, so the two
            # fields `Run()` adds are absent by design rather than by omission.
            where = warnings if m.get("experiment") in SELF_MANIFESTING else problems
            if "route" not in mm:
                where.append(f"{rid}: role {mm.get('role')!r} missing route (a manifest_version 2 field)")
            if not mm.get("k_impl"):
                where.append(f"{rid}: role {mm.get('role')!r} missing k_impl (a manifest_version 2 field)")
    for inp in (m.get("inputs") or []) if CHECK_INPUTS else []:
        f = os.path.join(INPUTS_ROOT, inp["file"])
        if not os.path.exists(f):
            problems.append(f"{rid}: input {inp['file']} missing from disk")
        elif _sha256(f) != inp.get("sha256"):
            msg = f"{rid}: input {inp['file']} sha256 changed since the run"
            (warnings if m.get("experiment") == "w1-smoke" else problems).append(msg)
    if m.get("cost_usd") is None:
        problems.append(f"{rid}: cost_usd missing/null (multi-model runs need per-role attribution)")
    if (m.get("calls") or {}).get("errors", 1) != 0:
        (warnings if m.get("experiment") in SELF_MANIFESTING else problems).append(
            f"{rid}: calls.errors = {(m.get('calls') or {}).get('errors')!r} (must be 0)")
    return m


def walk_results():
    """-> {run_id: manifest}, problems, warnings; validates every manifest under results/."""
    problems, warnings, by_id = [], [], {}
    paths = sorted(glob.glob(os.path.join(RESULTS, "*", "*", "manifest.json")))
    if not paths:
        problems.append(f"no <family>/<run>/manifest.json found under {RESULTS}")
    seen = {}
    for p in paths:
        m = validate_manifest(p, problems, warnings)
        if m and m.get("run_id"):
            rid = m["run_id"]
            if rid in seen:
                warnings.append(
                    f"{rid}: run_id used by more than one run ({seen[rid]}, "
                    f"{m.get('experiment')}) - the id is a timestamp and a commit, and "
                    f"these runs started in the same second off the same commit")
            seen.setdefault(rid, m.get("experiment"))
            by_id[rid] = m
    return by_id, problems, warnings, len(paths)


def check_paper(paper_file, by_id, valid_ids, problems):
    """Every paper number resolves to valid manifests (or an embedded plan-path block)."""
    try:
        paper = json.load(open(paper_file))
    except Exception as ex:
        problems.append(f"--paper: cannot read {paper_file} ({ex})")
        return
    freeze = json.load(open(FREEZE_FILE)) if os.path.exists(FREEZE_FILE) else {}
    for stat, entry in paper.items():
        if isinstance(entry, list):  # bare run_id list shorthand
            entry = {"runs": entry}
        if entry.get("plan_path"):
            f = os.path.join(INPUTS_ROOT, entry.get("file", ""))
            if not entry.get("file") or not os.path.exists(f):
                problems.append(f"{stat}: plan-path file {entry.get('file')!r} missing")
                continue
            try:
                blob = json.load(open(f))
            except Exception as ex:
                problems.append(f"{stat}: plan-path file unreadable ({ex})")
                continue
            block = blob.get("manifest") if isinstance(blob, dict) else None
            if not isinstance(block, dict) or not block.get("git_commit"):
                problems.append(f"{stat}: {entry['file']} lacks an embedded manifest block "
                                "with git_commit (the relaxed check)")
            continue
        runs = entry.get("runs") or []
        if not runs:
            problems.append(f"{stat}: no run_ids listed")
        complete_reps = set()
        for rid in runs:
            if rid not in by_id:
                problems.append(f"{stat}: run {rid} not found under results/")
                continue
            m = by_id[rid]
            if rid not in valid_ids:
                problems.append(f"{stat}: run {rid} exists but failed validation")
            if m.get("status") == "complete":
                complete_reps.add(m.get("replicate"))
            inputs = {i["file"]: i.get("sha256") for i in m.get("inputs") or []}
            if MASTER_PAIRS not in inputs:
                problems.append(f"{stat}: run {rid} inputs lack {MASTER_PAIRS} (the only "
                                "pair file paper-bound runs may load)")
            elif not freeze:
                problems.append(f"{stat}: no freeze_hashes.json to check {MASTER_PAIRS} against "
                                "(record the freeze hash when the pair set is frozen)")
            elif inputs[MASTER_PAIRS] not in freeze.values():
                problems.append(f"{stat}: run {rid} loaded {MASTER_PAIRS} at an unrecorded hash "
                                f"{inputs[MASTER_PAIRS][:12]}... (not in freeze_hashes.json)")
        if entry.get("stochastic") and len(complete_reps) < 3:
            problems.append(f"{stat}: stochastic cell has {len(complete_reps)} complete "
                            "replicates (<3)")


def main():
    global RESULTS, INPUTS_ROOT, CHECK_INPUTS, FREEZE_FILE
    ap = argparse.ArgumentParser(description="Validate run manifests and the paper's evidence map")
    ap.add_argument("--paper", metavar="PAPER_NUMBERS_JSON",
                    help="paper mode: validate every run behind every paper number")
    ap.add_argument("--results", metavar="DIR", default=RESULTS,
                    help="tree of <family>/<run>/manifest.json to validate. Default: "
                         "results/ beside this script. The released manifests are at "
                         "judgements/judges/ in the dataset repository.")
    ap.add_argument("--inputs-root", metavar="DIR",
                    help="root the manifests' recorded input paths resolve against, for the "
                         "sha256 re-check. Default: this script's directory.")
    ap.add_argument("--no-inputs", action="store_true",
                    help="skip the input-file checks. Use this on the released manifests: "
                         "they name construction corpora the release does not ship, and "
                         "without it every run reports its inputs as missing.")
    args = ap.parse_args()
    RESULTS = os.path.abspath(args.results)
    if args.inputs_root:
        INPUTS_ROOT = os.path.abspath(args.inputs_root)
    CHECK_INPUTS = not args.no_inputs
    FREEZE_FILE = os.path.join(INPUTS_ROOT, "freeze_hashes.json")

    by_id, problems, warnings, n_manifests = walk_results()
    base_problem_ids = {p.split(":")[0] for p in problems}
    valid_ids = {rid for rid in by_id if rid not in base_problem_ids}
    if args.paper:
        check_paper(args.paper, by_id, valid_ids, problems)

    print(f"checked {n_manifests} manifest(s) under {RESULTS}"
          + ("" if CHECK_INPUTS else " (input files not checked)")
          + (f" + paper map {args.paper}" if args.paper else ""))
    for w in warnings:
        print(f"  WARN {w}")
    if problems:
        for p in problems:
            print(f"  FAIL {p}")
        print(f"{len(problems)} problem(s)")
        sys.exit(1)
    print("all manifests valid" + (f" ({len(warnings)} warning(s))" if warnings else ""))
    sys.exit(0)


if __name__ == "__main__":
    main()
