"""w2_common.py - shared substrate for the W2 runners (specs/w2-ablation-grid.md).

Holds only what all four W2 runners need and none of them should own alone:
prompt loading + change detection, the dataset/transcript substrate, the
resumable record store, the spend tripwire, and the cache-ordered executor that
implements the pair-major "cache harder" mandate (docs/COSTS.md lever 1).

TWO GATES ARE DELIBERATELY NOT ENFORCED, and each records what it does not enforce:

  * Prompts are improvable. `PROMPTS.sha256` is a CHANGE DETECTOR, not a gate: a
    drifted prompt prints a warning and is recorded in the manifest, it does not
    refuse the run. The discipline that replaces the freeze is "one instrument
    per batch, never mid-run" - change a prompt, re-hash it, then launch.
  * The dataset is not frozen ahead of a run. The runner takes whatever file it is
    pointed at, records its `dataset_version` + sha256 into every manifest and
    every record, and refuses only if the file is missing or CHANGES UNDER IT
    mid-run (`assert_dataset_stable`). A clean working tree is not required,
    because analysis code changed between runs; instead each manifest records the
    commit it ran from together with every uncommitted path.

It adds NO chat helper of its own - every paper-bound call goes through W1's
`llm()` + `Run()` in common.py (W2 spec section 5 item 2).

Ordering contract (the cache-harder mandate). Work is grouped into BLOCKS that
share a byte-identical prompt prefix, and blocks run one at a time:

    for each replicate:                      # Run() carries exactly one replicate
      for each consultation block:           # everything below shares one transcript
        judge the first note alone           # warms the provider prefix cache
        fan the rest out across `workers`    # ~all of them hit the warm prefix

Because every grid prompt places {transcript} first (W2 3.1), one consultation's
clean note, its 3 errored notes, all 8 cells and all k samples share that prefix,
so a block of ~144 calls warms once and hits thereafter. Cross-replicate reuse is
deliberately not chased: it would need one Run spanning 3 replicates (against W1
3.5) to buy a rounding error on top of an already ~99%-of-block hit rate.
"""
import gzip, hashlib, json, os, re, sys, threading, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from common import HERE, RESULTS, llm

PROMPT_DIR = os.path.join(HERE, "w2_prompts")
SPEC = "specs/w2-ablation-grid.md"
CACHE_DIR = os.path.join(HERE, "w2_cache")

# Dataset candidates, in resolution order. dataset_v2.json is the combined set the
# factorial construction produces (add/change controls + the complete/partial
# omission factorial with severity + residual metadata); pairs_master_frozen.json
# is the pre-factorial 281-pair set and stays as the fallback until v2 lands.
DATASET_V2 = "master/dataset_v2.json"
FROZEN_PAIRS = "master/pairs_master_frozen.json"
UNFROZEN_PAIRS = "master/hard_negatives_master.json"
DATASET_CANDIDATES = [DATASET_V2, FROZEN_PAIRS]

# Replicate seeds (W2 section 2 "N and runs").
RUN_SEEDS = {1: 11, 2: 22, 3: 33}

# Model roles (models.lock.json). reasoning_effort is sent only where the pinned
# route supports it: gpt-5.4 honours it (W1 smoke probe), the Qwen non-thinking
# variant and the Anthropic route take no reasoning params (W2 section 4).
# gemini (added 2026-08-17, second-family replication): the Gemini 3.1 Pro endpoint
# REFUSES effort "none" outright ("Reasoning is mandatory for this endpoint"), so the
# nearest analogue "minimal" is pinned and the deviation is disclosed in the lock.
ROLES = {
    "gpt54": {"role": "judge-primary", "reasoning_effort": "none"},
    "qwen": {"role": "judge-qwen", "reasoning_effort": None},
    "opus": {"role": "judge-opus", "reasoning_effort": None},
    "gemini": {"role": "judge-gemini", "reasoning_effort": "minimal"},
}

CELLS = ["F-bin-k1", "F-bin-k8", "F-score-k1", "F-score-k8",
         "FC-bin-k1", "FC-bin-k8", "FC-score-k1", "FC-score-k8"]
# Model-generality rows: the 4 prompts at k=1 plus the recipe cell (W2 section 2).
GENERALITY_CELLS = ["F-bin-k1", "F-score-k1", "FC-bin-k1", "FC-score-k1", "FC-score-k8"]

MAX_TOKENS = 1024


def parse_cell(cell):
    """'FC-score-k8' -> ('FC', 'score', 8)."""
    crit, fmt, kk = cell.split("-")
    assert crit in ("F", "FC") and fmt in ("bin", "score") and kk in ("k1", "k8"), cell
    return crit, fmt, int(kk[1:])


# ---------------------------------------------------------------- prompts
HASH_FILE = "PROMPTS.sha256"
# Populated by every load_prompts() call: {name: {"recorded": sha, "on_disk": sha}} for
# prompts whose bytes differ from PROMPTS.sha256, plus untracked/missing files. Runners
# copy this into their manifest params so a drifted prompt is disclosed, not blocked.
PROMPT_DRIFT = {}


def prompt_hashes():
    """{filename: sha256} recorded in w2_prompts/PROMPTS.sha256 (empty if absent)."""
    path = os.path.join(PROMPT_DIR, HASH_FILE)
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            if line.strip():
                digest, name = line.split()
                out[name] = digest
    return out


def prompt_files():
    return sorted(f for f in os.listdir(PROMPT_DIR) if f.endswith(".txt"))


def load_prompts(names=None, strict=False, quiet=False):
    """Load w2_prompts/*.txt and DETECT drift against PROMPTS.sha256.

    Post-review (2026-08-12): prompts may be improved at any time between batches,
    so a changed byte no longer refuses the run. Drift is printed, recorded in
    PROMPT_DRIFT, and carried into the manifest; `strict=True` restores the old
    refusal for anyone who wants it. The discipline is procedural now: change one
    instrument per batch, re-hash (`python w2_common.py --rehash`), then launch -
    never mid-run.
    """
    recorded = prompt_hashes()
    out, drift = {}, {}
    for name in prompt_files():
        raw = open(os.path.join(PROMPT_DIR, name), "rb").read()
        got = hashlib.sha256(raw).hexdigest()
        want = recorded.get(name)
        if want is None:
            drift[name] = {"recorded": None, "on_disk": got, "state": "untracked"}
        elif want != got:
            drift[name] = {"recorded": want, "on_disk": got, "state": "changed"}
        out[name[:-4]] = raw.decode()
    for name in recorded:
        if not os.path.exists(os.path.join(PROMPT_DIR, name)):
            drift[name] = {"recorded": recorded[name], "on_disk": None, "state": "missing"}
    PROMPT_DRIFT.clear()
    PROMPT_DRIFT.update(drift)
    if drift and not quiet:
        print(f"  prompt change-detector: {len(drift)} file(s) differ from {HASH_FILE} "
              "(recorded in the manifest, not a gate)")
        for name, d in sorted(drift.items()):
            print(f"    - {name}: {d['state']}")
        print(f"    re-hash with: .venv/bin/python w2_common.py --rehash")
    if drift and strict:
        raise RuntimeError("prompt check FAILED (strict): " + ", ".join(sorted(drift)))
    if names:
        missing = [n for n in names if n not in out]
        if missing:
            raise KeyError(f"prompts not found in w2_prompts/: {missing}")
        out = {n: out[n] for n in names}
    return out


def rehash_prompts(write=True):
    """Rewrite w2_prompts/PROMPTS.sha256 from the files on disk. Returns the diff."""
    before, after = prompt_hashes(), {}
    lines = []
    for name in prompt_files():
        d = sha256_file(os.path.join(PROMPT_DIR, name))
        after[name] = d
        lines.append(f"{d}  {name}")
    if write:
        with open(os.path.join(PROMPT_DIR, HASH_FILE), "w") as f:
            f.write("\n".join(lines) + "\n")
    return {"added": sorted(set(after) - set(before)),
            "removed": sorted(set(before) - set(after)),
            "changed": sorted(n for n in set(after) & set(before) if after[n] != before[n]),
            "n_files": len(after)}


def render(template, **slots):
    """Fill {slot} placeholders by literal replacement, never str.format().

    Several frozen prompts (ragas_*, checklist_*) end with a literal JSON schema
    such as {"facts": ["<fact 1>"]}, which str.format reads as a replacement
    field and blows up on. The prompts are pre-registration artifacts and cannot
    be re-escaped, so substitution has to be slot-exact instead.
    """
    out = template
    for name, value in slots.items():
        token = "{" + name + "}"
        if token not in out:
            raise KeyError(f"prompt has no {token} slot")
        out = out.replace(token, value)
    return out


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- the substrate
SEVERITY_LEVELS = ["critical", "supporting", "peripheral"]
# The residual axis, ordered by HOW MUCH OF THE FACT SURVIVES - which is the order the
# surface is plotted along. `omission_factorial.py` builds complete / partial-strong /
# partial-weak (one surviving site in each partial class, so residual STRENGTH is the
# manipulated variable rather than the count); "partial" is the ungraded label the 34
# rescued seed pairs and any older artefact carry.
RESIDUAL_LEVELS = ["complete", "partial-weak", "partial", "partial-strong"]
CONTROL_LEVELS = ["add", "change"]                 # non-omission controls
STRENGTH_ORDINAL = {"explicit": 3, "paraphrase": 2, "partial": 1}
# A surviving EXPLICIT or PARAPHRASE mention leaves the fact recoverable (strong); only a
# fragment surviving is weak. Used to place a pair that records a residual but no class.
STRENGTH_TO_LEVEL = {"explicit": "partial-strong", "paraphrase": "partial-strong",
                     "partial": "partial-weak"}


def dataset_version_of(blob, path, digest):
    """The recorded version string of a dataset file.

    Prefers what the file says about itself (`dataset_version`, then `version`,
    then a build stamp); falls back to `<basename>@<sha16>`, which is still a
    stable, checkable identity. Never guesses semantics - it is an identity, and
    the sha256 sits beside it in every manifest.
    """
    if isinstance(blob, dict):
        for key in ("dataset_version", "version", "dataset_id", "build_id"):
            v = blob.get(key)
            if isinstance(v, (str, int, float)) and str(v).strip():
                return str(v).strip()
    return f"{os.path.basename(path)}@{digest[:8]}"


def _classify(pair):
    """Factorial coordinates of one pair: (class, residual_level, severity, residual).

    Reads what the dataset says and only infers where it is silent:
      * `class` ("omit-complete"/"omit-partial"/"add"/"change") is taken verbatim
        when present - the construction agent owns that label.
      * a type=="omit" pair with no class is COMPLETE unless it carries residual
        sites, in which case it is PARTIAL (that is exactly the distinction the
        residual metadata exists to record).
      * severity comes from `severity`, else the target trap's `importance`, else
        "ungraded" - never silently defaulted to a real grade, because a fake
        "supporting" would flatten the severity axis the surface measures.
    """
    typ = pair.get("type")
    res = pair.get("residual") if isinstance(pair.get("residual"), dict) else {}
    sites = res.get("sites") if isinstance(res.get("sites"), list) else []
    n_surviving = res.get("n_surviving")
    if n_surviving is None:  # the factorial builder records it flat, one surviving site
        n_surviving = pair.get("n_surviving")
    if n_surviving is None:
        n_surviving = len(sites) if sites else None

    # strength of the strongest surviving mention: from the residual block, else the flat
    # field the plan-driven builder writes (`keep_site_strength`), else the graded sites.
    strength = (res.get("residual_max_strength") or res.get("max_strength")
                or pair.get("residual_strength") or pair.get("keep_site_strength"))
    if not strength and sites:
        graded = [s.get("strength") for s in sites if isinstance(s, dict) and s.get("strength")]
        if graded:
            strength = max(graded, key=lambda s: STRENGTH_ORDINAL.get(s, 0))
    if strength and n_surviving is None:
        n_surviving = 1

    raw = (pair.get("class") or pair.get("omission_class") or
           pair.get("residual_level") or "").strip().lower().replace("_", "-")
    if raw.startswith("omit-"):
        raw = raw[len("omit-"):]
    if raw.startswith("control-"):
        raw = raw[len("control-"):]
    if raw in ("complete",) or raw.startswith("partial"):
        level = raw
    elif raw in CONTROL_LEVELS:
        level = raw
    elif raw:
        level = raw  # an unrecognised label passes through rather than being coerced
    elif typ == "omit":
        # no class recorded: infer it from the residual the dataset DID record. A
        # surviving explicit/paraphrase mention leaves the fact recoverable; only a
        # fragment surviving is weak; nothing surviving is a complete omission.
        level = (STRENGTH_TO_LEVEL.get(strength, "partial") if (n_surviving or 0) > 0
                 else "complete")
    else:
        level = typ or "unknown"

    # `class` is the coarse label ("omit-partial") and it is read first above, which
    # before 2026-08-12 silently flattened the residual axis: dataset_v2 grades every
    # partial as partial-weak / partial-strong, and all of that detail collapsed to a
    # bare "partial". The surface's whole point is the gradient, so refine the coarse
    # label - preferring what the dataset says outright, else deriving it from the
    # surviving mention's strength, which is the same rule the dataset itself applied
    # (explicit/paraphrase -> strong, fragment -> weak; verified to reproduce all 148
    # graded partials exactly). Records written before this fix keep the coarse label
    # but carry residual_strength, so the analysis can normalise them identically.
    cls = f"omit-{level}" if level not in CONTROL_LEVELS and typ == "omit" else level

    if level == "partial":
        stated = (pair.get("residual_level") or "").strip().lower().replace("_", "-")
        if stated.startswith("partial-"):
            level = stated
        elif strength:
            level = STRENGTH_TO_LEVEL.get(strength, "partial")

    sev = (pair.get("severity") or (pair.get("target") or {}).get("importance") or "").strip().lower()
    if sev not in SEVERITY_LEVELS:
        sev = "ungraded"
    residual = {"n_surviving": n_surviving, "max_strength": strength or None,
                "n_sites_recorded": len(sites) or None}
    return cls, level, sev, residual


def factorial_of(pair):
    """The per-pair factorial block copied onto every note record and read by
    w2_analyze.py's surface. Flat and self-describing so a record is analysable
    without the dataset file beside it.

    `matched_key` identifies the FACT, so the complete and partial pairs built from
    the same fact in the same note share it - that is what the matched McNemar keys
    off. It is None when the dataset names no fact, because two pairs that merely
    lack a fact id are not a matched pair and must not be treated as one.
    """
    cls, level, sev, residual = _classify(pair)
    fact_key = pair.get("fact_key") or pair.get("fact")
    ident = fact_key if fact_key else (pair.get("mc_idx") if pair.get("mc_idx") is not None
                                       else None)
    return {"class": cls, "residual_level": level, "severity": sev,
            "residual_n_surviving": residual["n_surviving"],
            "residual_strength": residual["max_strength"],
            "fact_key": fact_key or None,
            "split": pair.get("split") or "eval",
            "matched_key": (None if ident is None else
                            f"{pair.get('stratum')}|{pair.get('id')}|{ident}")}


def resolve_dataset(path=None, allow_unfrozen=False):
    """Which dataset file this run will read, in resolution order."""
    if path:
        cand = path if os.path.isabs(path) else os.path.join(HERE, path)
        if not os.path.exists(cand):
            raise RuntimeError(f"--dataset {path} does not exist")
        return os.path.relpath(cand, HERE)
    for rel in DATASET_CANDIDATES:
        if os.path.exists(os.path.join(HERE, rel)):
            return rel
    if allow_unfrozen and os.path.exists(os.path.join(HERE, UNFROZEN_PAIRS)):
        return UNFROZEN_PAIRS
    raise RuntimeError(
        "no dataset file found. Looked for " + ", ".join(DATASET_CANDIDATES)
        + f" (and {UNFROZEN_PAIRS} with --allow-unfrozen). Pass --dataset PATH.")


def load_dataset(path=None, allow_unfrozen=False):
    """Return (pairs, info) for whichever dataset file resolves.

    Replaces the frozen-substrate gate with a version check (the lead author, 2026-08-12): the
    runner records `dataset_version` + sha256 and refuses only if the file is
    missing, malformed, or changes mid-run (see assert_dataset_stable). Nothing
    about a freeze record, a pending status or a clean tree blocks a launch.
    """
    rel = resolve_dataset(path, allow_unfrozen)
    full = os.path.join(HERE, rel)
    digest = sha256_file(full)
    blob = json.load(open(full))
    info = {"pairs_file": rel, "sha256": digest,
            "dataset_version": dataset_version_of(blob, full, digest),
            "dataset_kind": ("factorial_v2" if rel == DATASET_V2 else
                             "frozen_281" if rel == FROZEN_PAIRS else
                             "pre_audit" if rel == UNFROZEN_PAIRS else "custom")}
    # A pre-audit file is still labelled: its numbers must never read as paper numbers.
    info["frozen"] = info["dataset_kind"] != "pre_audit"
    info["paper_bound"] = info["frozen"]
    if not info["frozen"]:
        info["warning"] = ("PRE-AUDIT construction file - smoke only, never paper-bound")

    pairs = blob
    if isinstance(pairs, dict):
        for key in ("pairs", "items", "data"):
            if isinstance(pairs.get(key), list):
                info["wrapper_keys"] = sorted(k for k in pairs if k != key)
                pairs = pairs[key]
                break
        else:
            raise RuntimeError(f"{rel}: object with no pairs/items/data list inside")
    if not isinstance(pairs, list) or not pairs:
        raise RuntimeError(f"{rel}: no pairs found")

    for i, p in enumerate(pairs):
        for f in ("id", "type", "clean", "errored"):
            if not p.get(f):
                raise RuntimeError(f"{rel}[{i}]: missing field {f!r}")
        p.setdefault("stratum", "authored")
        p.setdefault("pair_id", f"{p['stratum']}|{p['id']}|{p['type']}")
        p["_fac"] = factorial_of(p)

    # W-F contamination guard. dataset_v2 marks pairs the GEPA optimizer trained on with
    # eval_set:false; the build asserts the split, but the assertion lives in the build,
    # not here, so before 2026-08-12 this loader happily returned all 500 and the gepa arm
    # would have been scored on its own training data. Dropping them at load is the only
    # place that protects EVERY caller (grid, arms, baselines, analysis) at once.
    held = [p for p in pairs if p.get("eval_set") is False]
    if held:
        pairs = [p for p in pairs if p.get("eval_set") is not False]
        info["held_out_of_eval"] = {
            "n": len(held),
            "reasons": dict(Counter(p.get("held_out_reason") or "unspecified" for p in held)),
            "pair_ids": sorted(p["pair_id"] for p in held)}

    # pair_ids must be unique - the record store keys off them, and a v2 file that
    # holds a complete AND a partial pair for the same consultation+type would
    # otherwise silently collapse into one judged note.
    dupes = [k for k, n in Counter(p["pair_id"] for p in pairs).items() if n > 1]
    if dupes:
        raise RuntimeError(f"{rel}: {len(dupes)} duplicate pair_id(s), e.g. {dupes[:3]} - "
                           "the record store keys off pair_id; disambiguate in the dataset "
                           "(e.g. append the class) before running")
    info["n_pairs"] = len(pairs)
    info["by_type"] = dict(Counter(p["type"] for p in pairs))
    info["by_stratum"] = dict(Counter(p["stratum"] for p in pairs))
    info["by_class"] = dict(Counter(p["_fac"]["class"] for p in pairs))
    info["by_severity"] = dict(Counter(p["_fac"]["severity"] for p in pairs))
    info["by_residual_x_severity"] = {
        f"{lvl}|{sev}": n for (lvl, sev), n in sorted(Counter(
            (p["_fac"]["residual_level"], p["_fac"]["severity"]) for p in pairs).items())}
    info["by_split"] = dict(Counter(p["_fac"]["split"] for p in pairs))
    return pairs, info


def load_pairs(allow_unfrozen=False, path=None):
    """Back-compatible alias for load_dataset (older call sites)."""
    return load_dataset(path=path, allow_unfrozen=allow_unfrozen)


def assert_dataset_stable(info):
    """The only substrate gate left: the file we are reading must not change under us.

    Called at every chunk boundary. A dataset rewritten mid-run would mean records
    bought at two different versions sharing one store, which no downstream analysis
    could untangle - so that is a hard stop, and the only one.
    """
    full = os.path.join(HERE, info["pairs_file"])
    if not os.path.exists(full):
        raise SystemExit(f"\nDATASET GONE: {info['pairs_file']} disappeared mid-run "
                         f"(version {info['dataset_version']}). Stopping.")
    now = sha256_file(full)
    if now != info["sha256"]:
        raise SystemExit(
            f"\nDATASET CHANGED MID-RUN: {info['pairs_file']} was {info['sha256'][:12]} at "
            f"start, is {now[:12]} now (version {info['dataset_version']}).\n"
            "Records in this store would span two dataset versions. Stopping - re-launch "
            "with a fresh --tag against the new version.")


TRANSCRIPT_SOURCES = [
    # (stratum, path) - authored_scenarios.json is W2 3.5's named source; the master
    # strata carry their transcripts in W-D's core fact-sheet files. If W-A fixes a
    # single companion file at the freeze (Amendment F item 2) it is picked up first.
    (None, "master/transcripts_master.json"),
    ("authored", "authored_scenarios.json"),
    ("trapblind", "master/fact_sheets_trapblind_core.json"),
    ("primock", "master/fact_sheets_primock_core.json"),
    ("aci", "master/fact_sheets_aci_core.json"),
    ("authored", "master/fact_sheets_authored_extracted_core.json"),
]


def transcript_index():
    """(stratum, id) -> transcript, plus a {file: n} provenance map."""
    idx, provenance = {}, {}
    for stratum, rel in TRANSCRIPT_SOURCES:
        path = os.path.join(HERE, rel)
        if not os.path.exists(path):
            continue
        data = json.load(open(path))
        rows = data if isinstance(data, list) else [
            (dict(v, id=k) if isinstance(v, dict) else {"id": k, "transcript": v})
            for k, v in data.items()]
        n = 0
        for r in rows:
            if not isinstance(r, dict):
                continue
            t = r.get("transcript") or r.get("transcript_text") or ""
            rid, st = r.get("id"), r.get("stratum", stratum)
            if not (t and rid):
                continue
            for key in ({(st, rid)} if st else {(s, rid) for s in
                        ("authored", "trapblind", "primock", "aci")}):
                if key not in idx:
                    idx[key] = t
                    n += 1
        if n:
            provenance[rel] = n
    return idx, provenance


def note_units(pairs, transcripts):
    """Turn pairs into the unique NOTE judgements W2 actually buys, grouped into
    per-consultation blocks (W2 section 2 dedup: a consultation's clean twin is
    judged once and reused across its pairs).

    Returns (blocks, notes). blocks = [{consultation, stratum, transcript, notes:[...]}]
    in a stable order; notes carry a global item_index used by the seed formula.
    """
    by_consult = defaultdict(list)
    for p in pairs:
        by_consult[(p["stratum"], p["id"])].append(p)

    blocks = []
    for (stratum, cid), ps in sorted(by_consult.items()):
        transcript = transcripts.get((stratum, cid))
        if transcript is None:
            raise RuntimeError(f"no transcript for {stratum}/{cid} - check TRANSCRIPT_SOURCES "
                               "(W2 Amendment F item 2: the companion file path is fixed at "
                               "the W-A freeze)")
        ps.sort(key=lambda p: (p["type"], p["pair_id"]))
        notes, clean_of = [], {}
        for p in ps:  # one clean note per DISTINCT clean text (usually 1 per consultation)
            h = sha256_text(p["clean"])[:12]
            if h not in clean_of:
                key = f"{stratum}|{cid}|clean|{h}"
                clean_of[h] = key
                notes.append({"note_key": key, "note_role": "clean", "text": p["clean"],
                              "stratum": stratum, "consultation": cid, "pair_type": None,
                              "pair_id": None, "clean_key": key})
        for p in ps:
            notes.append({"note_key": f"{p['pair_id']}|err", "note_role": "errored",
                          "text": p["errored"], "stratum": stratum, "consultation": cid,
                          "pair_type": p["type"], "pair_id": p["pair_id"],
                          "clean_key": clean_of[sha256_text(p["clean"])[:12]],
                          "importance": importance_of(p),
                          "factorial": p.get("_fac") or factorial_of(p)})
        blocks.append({"consultation": cid, "stratum": stratum, "transcript": transcript,
                       "n_pairs": len(ps), "notes": notes})

    flat = [n for b in blocks for n in b["notes"]]
    for i, n in enumerate(sorted(flat, key=lambda n: n["note_key"])):
        n["item_index"] = i  # stable across runs: derived from the sorted note keys
    return blocks, flat


def load_subset_spec(path):
    """Opt-in spec for judging a CHOSEN SUBSET of a corpus's notes. Returns None when unused.

    Added 2026-08-25 for the real-error arm, which points the existing judge designs at the
    census's real vendor notes. Two things differ from a benchmark run and neither is a
    property of the judges, so both are handled here rather than by forking a runner:

      transcripts  the benchmark's transcript index covers 137 of the census's 142
                   consultations, and note_units RAISES on a gap rather than skipping. The
                   spec carries the missing transcripts, verified byte-identical to the
                   benchmark's on the 137 they share.
      note_keys    census notes have no clean twin to judge, but note_units mints one per
                   distinct clean text. Without an allowlist the arm buys ~140 note
                   judgements it never scores.

    Off unless a runner is passed --subset-spec, so no existing run changes in any way.
    Keys are "stratum|consultation" because JSON has no tuple keys.
    """
    if not path:
        return None
    spec = json.load(open(path))
    return {"note_keys": set(spec.get("note_keys") or []),
            "transcripts": {tuple(k.split("|", 1)): v
                            for k, v in (spec.get("transcripts") or {}).items()},
            "path": path}


def apply_subset_spec(spec, blocks, notes):
    """Restrict blocks/notes to the spec's allowlist, dropping consultations left empty.

    Fails loudly if the allowlist names a note the dataset did not produce - a silently
    short run would read as a real measurement on fewer notes.
    """
    if not spec or not spec["note_keys"]:
        return blocks, notes
    keep = spec["note_keys"]
    blocks = [dict(b, notes=[n for n in b["notes"] if n["note_key"] in keep]) for b in blocks]
    blocks = [b for b in blocks if b["notes"]]
    notes = [n for b in blocks for n in b["notes"]]
    absent = keep - {n["note_key"] for n in notes}
    if absent:
        raise RuntimeError(
            f"subset spec {spec['path']} names {len(absent)} note(s) the dataset does not "
            f"produce, e.g. {sorted(absent)[:3]} - refusing to run a silently short arm")
    return blocks, notes


def importance_of(pair):
    """Ordinal salience grade of the injected item (W2 Amendment 2026-07-29b M).

    Prefers the target trap's own `importance`, falls back to the pair's severity
    grade, then to the amendment's stated default of "supporting".
    """
    tgt = pair.get("target") or {}
    return (tgt.get("importance") or pair.get("severity") or "supporting")


IMPORTANCE_ORDINAL = {"critical": 3, "supporting": 2, "peripheral": 1}


# ---------------------------------------------------------------- parsing
def parse_verdict(text):
    m = re.findall(r"Verdict:\s*(PASS|FAIL)", text or "", re.I)
    return m[-1].upper() if m else None


def parse_yesno(text):
    """v14's verdict line (YES = flagged)."""
    m = re.findall(r"Verdict:\s*(YES|NO)", text or "", re.I)
    return m[-1].upper() if m else None


def parse_score(text, lo=0, hi=10):
    m = re.findall(r"Score:\s*(\d+)", text or "")
    if not m:
        return None
    v = int(m[-1])
    return v if lo <= v <= hi else None


def parse_json_blob(text):
    m = re.search(r"(\{.*\}|\[.*\])", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


# ---------------------------------------------------------------- record store
class RecordStore:
    """Append-only JSONL of completed judgements: the resume index AND the input
    w2_analyze.py reads. Lives under results/ so analysis stays recomputable from
    results/ alone (W1 3.5 rule ii); every record carries the run_id that bought
    it, so each one still traces to a manifest.
    """

    def __init__(self, experiment, tag):
        self.dir = os.path.join(RESULTS, experiment, "_state")
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, f"{tag}.jsonl")
        self._lock = threading.Lock()
        self.records = {}
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # torn final line from a killed process
                    self.records[r["key"]] = r
        self._fh = None  # opened on first write, so planning a run leaves no litter

    def has(self, key):
        return key in self.records

    def put(self, rec):
        with self._lock:
            if self._fh is None:
                self._fh = open(self.path, "a")
            self.records[rec["key"]] = rec
            self._fh.write(json.dumps(rec) + "\n")
            self._fh.flush()

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def since(self, keys):
        return [self.records[k] for k in keys if k in self.records]


# ---------------------------------------------------------------- spend tripwire
def ledger_total(spend="openrouter_credits"):
    path = os.path.join(RESULTS, "cost_ledger.jsonl")
    if not os.path.exists(path):
        return 0.0
    tot = 0.0
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if spend is None or d.get("spend", "openrouter_credits") == spend:
                tot += d.get("cost_usd") or 0.0
    return tot


class SpendGuard:
    r"""Credit tripwires, checked at start and after every block.

    Defaults are the runner-level guards this build was commissioned with
    (warn \$700 / stop \$900); the project-wide tripwires in W1 Amendment A4 are
    looser (review \$1,500 / stop \$2,500), so these bite first by design.

    SCOPE (added 2026-08-17). The original guard is CUMULATIVE: its baseline is the
    whole ledger, so a tripwire is a statement about total project spend. Once the
    ledger passed \$1,500 that made small tripwires unusable - a \$40 cap on a new
    run would fire before the first call. `scope="run"` (or an explicit `baseline`)
    makes the tripwire a statement about THIS RUN's spend instead, which is what a
    per-experiment cap means. Default is unchanged, so every existing caller keeps
    the cumulative behaviour byte for byte.
    """

    def __init__(self, warn=700.0, stop=900.0, authorized=False, baseline=None,
                 scope="cumulative"):
        self.warn, self.stop, self.authorized = warn, stop, authorized
        self.scope = "run" if (scope == "run" or baseline is not None) else "cumulative"
        self.ledger_at_start = ledger_total()
        self.baseline = (0.0 if scope == "run" else self.ledger_at_start) \
            if baseline is None else float(baseline)
        self.spent_here = 0.0
        self._warned = self.baseline >= warn
        if self._warned:
            print(f"  ! SPEND WARNING: {self.baseline:.2f} USD of credits already spent "
                  f"(warn tripwire {warn:.0f})")
        self._check()

    @property
    def total(self):
        return self.baseline + self.spent_here

    def add(self, usd):
        self.spent_here += usd or 0.0
        self._check()

    def _check(self):
        if self.total >= self.warn and not self._warned:
            self._warned = True
            print(f"  ! SPEND WARNING: cumulative credits {self.total:.2f} USD "
                  f">= {self.warn:.0f} tripwire")
        if self.total >= self.stop and not self.authorized:
            raise SystemExit(
                f"\nSPEND STOP: {self.scope} OpenRouter credits {self.total:.2f} USD >= "
                f"{self.stop:.0f} hard tripwire.\nStopping without fresh authorization. "
                "Re-run with --authorize-spend after checking with the lead author.")


# ---------------------------------------------------------------- judging
def judge(prompt, role, cell_k, base_seed, temperature=1.0, reasoning_effort="none",
          parser=parse_score, max_tokens=None):
    """One note-judgement = one llm() call of k samples + the pre-registered
    single retry of any unparseable sample.

    Seeds follow W2 section 4: base_seed = run_seed*100000 + item_index*10, and
    llm() offsets sample i by +i, so samples occupy base_seed+0..+7 and the two
    retry slots +8/+9 sit inside the same item's decade (the spec's "retry at
    seed+1" made collision-free at k=8).

    `max_tokens` defaults to MAX_TOKENS (1024), which is right for a grid cell's
    2-6 sentence assessment plus a verdict line. The decomposition-style baselines
    emit a list with one entry per claim or per fact and genuinely need more: at
    1024 the RAGAS faithfulness step truncated mid-JSON on every long note in the
    2026-08-12 smoke, which read as a parse failure and killed the whole record.
    Callers that emit a list pass their own budget rather than losing the arm.
    """
    mt = max_tokens or MAX_TOKENS
    texts, meta = llm(prompt, role, temperature=temperature, k=cell_k, seed=base_seed,
                      reasoning_effort=reasoning_effort, max_tokens=mt)
    vals = [parser(t) for t in texts]
    metas, retried = [meta], []
    for i, v in enumerate(vals):
        if v is None:
            rt, rm = llm(prompt, role, temperature=temperature, k=1, seed=base_seed + 8 + (i % 2),
                         reasoning_effort=reasoning_effort, max_tokens=mt)
            metas.append(rm)
            vals[i] = parser(rt[0])
            texts[i] = rt[0]
            retried.append(i)
    return vals, texts, metas, retried


def receipts(metas):
    """Per-call receipts: generation id, served provider, usage incl. cached
    tokens, OpenRouter-reported credit cost (W2 Amendment D)."""
    out = []
    for m in metas:
        for r in m.get("requests", []):
            u = r.get("usage", {})
            out.append({"gen": r.get("generation_id"), "prov": r.get("provider"),
                        "model": r.get("returned_model"), "seed": r.get("seed"),
                        "pt": u.get("prompt_tokens", 0), "ct": u.get("completion_tokens", 0),
                        "rt": u.get("reasoning_tokens", 0), "cached": u.get("cached_tokens", 0),
                        "cost": r.get("cost")})
    return out


def receipt_totals(recs):
    """Roll receipts up: calls, tokens, cached tokens, cost."""
    t = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
         "cached_tokens": 0, "cost_usd": 0.0}
    for r in recs:
        t["calls"] += 1
        t["prompt_tokens"] += r.get("pt") or 0
        t["completion_tokens"] += r.get("ct") or 0
        t["reasoning_tokens"] += r.get("rt") or 0
        t["cached_tokens"] += r.get("cached") or 0
        t["cost_usd"] += r.get("cost") or 0.0
    t["cost_usd"] = round(t["cost_usd"], 8)
    return t


def aggregate_cell(fmt, k, vals):
    """W2 3.2's aggregation + flag rules. Returns (aggregate, verdict, flagged).

    Score cells: k=1 -> the integer; k=8 -> winsorize8. Binary cells: PASS=10/
    FAIL=0, the identical operator, FAIL iff winsorized mean <= 5.0 (an exact
    4-4 tie counts as a flag). Flag rules: bin -> FAIL; score k=1 -> <=7;
    score k=8 -> winsorized mean < 8.0 (W2 section 6).
    """
    from common import winsorize8
    if any(v is None for v in vals) or len(vals) != k:
        return None, None, None
    if fmt == "score":
        agg = float(vals[0]) if k == 1 else winsorize8([float(v) for v in vals])
        flagged = agg <= 7 if k == 1 else agg < 8.0
        return agg, ("FAIL" if flagged else "PASS"), flagged
    enc = [10.0 if v == "PASS" else 0.0 for v in vals]
    agg = enc[0] if k == 1 else winsorize8(enc)
    verdict = "FAIL" if agg <= 5.0 else "PASS"
    return agg, verdict, verdict == "FAIL"


# ---------------------------------------------------------------- executor
def run_block(jobs, worker, workers, warmup=True):
    """Run one prefix-sharing block: warm the cache with the first job alone,
    then fan the rest out. Returns the number of jobs executed.

    The serial warm-up is the whole point - firing N cold requests at an unseen
    prefix concurrently makes all N of them miss, which is exactly the failure
    mode docs/COSTS.md lever 1 is trying to avoid.
    """
    if not jobs:
        return 0
    if warmup:
        worker(jobs[0])
        jobs = jobs[1:]
    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            list(ex.map(worker, jobs))
    return len(jobs) + (1 if warmup else 0)


def sub_blocks(block, cells, order, done):
    """Split one consultation block into the prefix-sharing units to run in turn.

    order="note-major" (default): one sub-block per NOTE, so every call for a note
    shares transcript + note + scaffold-up-to-the-criterion. That longer prefix is
    what gets a short transcript over the provider's 1,024-token cache minimum -
    measured at the W2 smoke, transcripts under ~4.1k chars cache 0% when only the
    transcript is shared, because the transcript prefix alone is below the floor.

    order="block-major": the whole consultation as one unit (all notes interleaved).
    Kept so the ordering effect stays measurable rather than asserted.
    """
    def jobs_for(notes):
        js = [(c, n, block) for n in notes for c in cells if not done(c, n)]
        # k=1 first so the serial warm-up call is a single request, then by cell
        js.sort(key=lambda j: (parse_cell(j[0].split("@")[0])[2],
                               j[1]["note_role"] != "clean", j[0]))
        return js

    if order == "block-major":
        return [jobs_for(block["notes"])]
    # clean note first: its prefix is the one the errored twins extend
    notes = sorted(block["notes"], key=lambda n: (n["note_role"] != "clean", n["note_key"]))
    return [j for j in (jobs_for([n]) for n in notes) if j]


def note_fields(note):
    """The note-identity + factorial block every runner copies onto every record, so
    w2_analyze.py can draw the surface from results/ alone."""
    fac = note.get("factorial") or {}
    return {"note_key": note["note_key"], "note_role": note["note_role"],
            "consultation": note["consultation"], "stratum": note["stratum"],
            "pair_id": note["pair_id"], "pair_type": note["pair_type"],
            "clean_key": note["clean_key"], "item_index": note["item_index"],
            "importance": note.get("importance"),
            "pair_class": fac.get("class"),
            "residual_level": fac.get("residual_level"),
            "severity": fac.get("severity"),
            "residual_n_surviving": fac.get("residual_n_surviving"),
            "residual_strength": fac.get("residual_strength"),
            "fact_key": fac.get("fact_key"), "matched_key": fac.get("matched_key"),
            "split": fac.get("split")}


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def confirm(msg, yes):
    if yes:
        return
    try:
        answer = input(f"{msg} [y/N] ").strip().lower()
    except EOFError:  # non-interactive invocation is never an approval
        answer = ""
    if answer != "y":
        sys.exit("aborted")


def git_state():
    """(commit, dirty_paths) for the harness subtree - the record that replaced the
    clean-tree gate. Siblings write this tree constantly, so a run discloses the
    commit plus exactly what was uncommitted rather than refusing to start."""
    from common import _git_state
    return _git_state()


def run_params(base, info, extra=None):
    """Common manifest params block: dataset identity, prompt drift, tree state."""
    p = dict(base)
    commit, dirty = git_state()
    p.update({"pairs_file": info["pairs_file"], "pairs_sha256": info["sha256"],
              "dataset_version": info["dataset_version"],
              "dataset_kind": info.get("dataset_kind"),
              "n_pairs": info["n_pairs"],
              "substrate_frozen": info["frozen"],
              "git_commit": commit,
              "git_dirty_paths": dirty,
              "gate_regime": "dataset-version check + mid-run stability (2026-08-12); "
                             "prompt hashes are a change detector, not a gate"})
    for key in ("by_class", "by_severity", "by_residual_x_severity", "by_split",
                "consultation_half"):
        if info.get(key):
            p[key] = info[key]
    if PROMPT_DRIFT:
        p["prompt_drift_vs_hashfile"] = PROMPT_DRIFT
    if not info["frozen"]:
        p["NOT_PAPER_BOUND"] = "pre-audit substrate: smoke only"
    if extra:
        p.update(extra)
    return p


# ---------------------------------------------------------------- tiny CLI
def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="W2 substrate utilities")
    ap.add_argument("--rehash", action="store_true",
                    help="rewrite w2_prompts/PROMPTS.sha256 from the files on disk")
    ap.add_argument("--check", action="store_true", help="report prompt drift and exit")
    ap.add_argument("--dataset", default=None, help="report what a run would load")
    args = ap.parse_args()
    if args.rehash:
        d = rehash_prompts()
        print(f"rehashed {d['n_files']} prompt file(s) -> w2_prompts/{HASH_FILE}")
        for k in ("added", "changed", "removed"):
            if d[k]:
                print(f"  {k}: {', '.join(d[k])}")
    if args.check:
        load_prompts()
        print("prompt drift:", json.dumps(PROMPT_DRIFT, indent=1) if PROMPT_DRIFT else "none")
    if args.dataset is not None or (not args.rehash and not args.check):
        pairs, info = load_dataset(path=args.dataset or None, allow_unfrozen=True)
        print(f"dataset: {info['pairs_file']} version={info['dataset_version']} "
              f"sha {info['sha256'][:12]} kind={info['dataset_kind']}")
        print(f"  {info['n_pairs']} pairs | by class {info['by_class']} | "
              f"by severity {info['by_severity']}")
        print(f"  residual x severity: {info['by_residual_x_severity']}")


if __name__ == "__main__":
    _cli()
