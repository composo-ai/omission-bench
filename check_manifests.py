"""check_manifests.py - W1 gates G1/G3 (specs/w1-reproducibility.md section 6, amended 2026-07-29).

Modes:
    python check_manifests.py
        Walk results/, validate every run manifest: status complete, git_dirty false,
        models non-empty with resolved IDs (+ v2: provider/route/k_impl), input sha256s
        still match the files on disk, cost_usd present, calls.errors == 0.

    python check_manifests.py --paper paper_numbers.json
        Gate G1, the pre-submission step: paper_numbers.json maps every figure/table/inline
        stat to its evidence. Entry forms:
            "fig_w2_grid": {"runs": ["<run_id>", ...], "experiment": "w2-ablation",
                            "stochastic": true}
            "wa_audit_m1": {"plan_path": true, "file": "w_a_report.json"}
        Fails if any run_id is missing or invalid, a stochastic entry has <3 complete
        replicates, a paper-bound API run's inputs lack pairs_master_frozen.json at the
        recorded freeze hash (Amendment A1; freeze hashes live in freeze_hashes.json at the
        harness root - {filename: sha256}, appended at the W-A freeze), or a plan-path
        entry's output file lacks an embedded manifest block (relaxed mode, R3).

Exit code 0 = all pass, 1 = any failure. manifest_version 2 required; v1 accepted for
pre-amendment w1-smoke runs only.
"""
import argparse, glob, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FREEZE_FILE = os.path.join(HERE, "freeze_hashes.json")
MASTER_PAIRS = "pairs_master_frozen.json"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_manifest(path, problems, warnings=None):
    """Validate one manifest.json. Appends 'run_id: problem' strings; returns the manifest.
    warnings collects non-fatal notes (currently: input-hash drift on w1-smoke runs, which
    judge pre-freeze construction files that legitimately mutate - W2 Amendment A.7 permits
    smoke on pre-audit files; every OTHER experiment hard-fails on drift)."""
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
        # G3: paper-bound runs must come from a clean tree. w1-smoke (machinery validation,
        # never behind a paper number) may run dirty IF the manifest discloses the exact
        # uncommitted paths - the shared-working-tree concession, decided 2026-07-30.
        if not (m.get("experiment") == "w1-smoke" and m.get("git_dirty_paths")):
            problems.append(f"{rid}: git_dirty is {m.get('git_dirty')!r} (G3 requires false; "
                            "only w1-smoke with git_dirty_paths disclosure is exempt)")
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
                problems.append(f"{rid}: role {mm.get('role')!r} provider {mm.get('provider')!r} != openrouter (A3)")
            if "route" not in mm:
                problems.append(f"{rid}: role {mm.get('role')!r} missing route (A3 manifest addition)")
            if not mm.get("k_impl"):
                problems.append(f"{rid}: role {mm.get('role')!r} missing k_impl (A3 manifest addition)")
    for inp in m.get("inputs") or []:
        f = os.path.join(HERE, inp["file"])
        if not os.path.exists(f):
            problems.append(f"{rid}: input {inp['file']} missing from disk")
        elif _sha256(f) != inp.get("sha256"):
            msg = f"{rid}: input {inp['file']} sha256 changed since the run"
            (warnings if m.get("experiment") == "w1-smoke" else problems).append(msg)
    if m.get("cost_usd") is None:
        problems.append(f"{rid}: cost_usd missing/null (multi-model runs need per-role attribution)")
    if (m.get("calls") or {}).get("errors", 1) != 0:
        problems.append(f"{rid}: calls.errors = {(m.get('calls') or {}).get('errors')!r} (must be 0)")
    return m


def walk_results():
    """-> {run_id: manifest}, problems, warnings; validates every manifest under results/."""
    problems, warnings, by_id = [], [], {}
    paths = sorted(glob.glob(os.path.join(RESULTS, "*", "*", "manifest.json")))
    if not paths:
        problems.append("no manifests found under results/")
    for p in paths:
        m = validate_manifest(p, problems, warnings)
        if m and m.get("run_id"):
            by_id[m["run_id"]] = m
    return by_id, problems, warnings


def check_paper(paper_file, by_id, valid_ids, problems):
    """Gate G1: every paper number resolves to valid manifests (or an embedded plan-path block)."""
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
            f = os.path.join(HERE, entry.get("file", ""))
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
                                "with git_commit (R3 relaxed mode)")
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
                problems.append(f"{stat}: run {rid} inputs lack {MASTER_PAIRS} (Amendment A1 - "
                                "the only pair file paper-bound runs may load)")
            elif not freeze:
                problems.append(f"{stat}: no freeze_hashes.json to check {MASTER_PAIRS} against "
                                "(record the freeze hash at the W-A freeze)")
            elif inputs[MASTER_PAIRS] not in freeze.values():
                problems.append(f"{stat}: run {rid} loaded {MASTER_PAIRS} at an unrecorded hash "
                                f"{inputs[MASTER_PAIRS][:12]}... (not in freeze_hashes.json)")
        if entry.get("stochastic") and len(complete_reps) < 3:
            problems.append(f"{stat}: stochastic cell has {len(complete_reps)} complete "
                            "replicates (<3)")


def main():
    ap = argparse.ArgumentParser(description="Validate results/ manifests (W1 gates G1/G3)")
    ap.add_argument("--paper", metavar="PAPER_NUMBERS_JSON",
                    help="gate-G1 mode: validate every run behind every paper number")
    args = ap.parse_args()

    by_id, problems, warnings = walk_results()
    base_problem_ids = {p.split(":")[0] for p in problems}
    valid_ids = {rid for rid in by_id if rid not in base_problem_ids}
    if args.paper:
        check_paper(args.paper, by_id, valid_ids, problems)

    print(f"checked {len(by_id)} manifest(s) under results/"
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
