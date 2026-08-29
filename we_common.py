#!/usr/bin/env python3
"""Shared primitives for the W-E cross-domain (RoSE) workstream.

specs/w-e-cross-domain-generalization.md section 2.4 requires W-E to REUSE W2/W-A machinery by
name rather than reimplement it, so this module is deliberately thin:

  render()           imported from w2_common - literal {slot} replacement, never str.format()
                     (the construction prompts contain literal JSON braces)
  winsorize8()       imported from common - W2's aggregation operator, for the grid runner
  units_of(),        imported from w_a_master - W-A's sentence-unit tokenizer, word-diff stats,
  word_diff_stats(),   unified unit diff, and BOTH algorithmic single-edit rules
  algo_check_v1(),
  algo_check(),
  unit_diff_text()

What is W-E's own: the prompt-hash gate over we_prompts/, the rose_master/ paths, the pair-id
convention, and the construction-call wrapper.

ALGORITHMIC-RULE DIVERGENCE (flagged, not resolved here). W-E spec 2.2/3.5 pre-registers "W-A's
exact acceptance constants (MAX_UNITS=2, MAX_REGIONS=2, MAX_CHANGED_WORDS=30)" - that is W-A's v1
rule. W-A itself replaced those on 2026-08-10 with its section-6 single revision (v2:
PURE_BLOCK_MAX_UNITS=6, REPLACE_MAX_UNITS=3, word-op purity and MAX_REGIONS dropped) after v1
rejected 51% of clinical pairs. So "W-A's constants" now names two different rules depending on
when you read it. we_verify_pairs.py therefore computes BOTH and gates on the one selected by
--algo-rule (default: v1, the text W-E actually pre-registers). Which rule W-E should gate on is
a decision for the review, and it interacts with spec Q9 (whether XSum's single-sentence
summaries need tighter constants, not looser ones).
"""

import hashlib
import json
import os

from common import HERE, claude_json, winsorize8              # noqa: F401  (re-export)
from w2_common import render                                   # noqa: F401  (re-export)
from w_a_master import (algo_check, algo_check_v1, unit_diff_text,   # noqa: F401  (re-export)
                        units_of, word_diff_stats)

PROMPT_DIR = os.path.join(HERE, "we_prompts")
ROSE_MASTER = os.path.join(HERE, "rose_master")

SAMPLE = os.path.join(ROSE_MASTER, "sample.json")
RAW = os.path.join(ROSE_MASTER, "rose_raw.json")
NOT_CONTAIN_RAW = os.path.join(ROSE_MASTER, "not_contain_raw.json")
SALIENCE_RAW = os.path.join(ROSE_MASTER, "salience_raw.json")
NOT_CONTAIN = os.path.join(ROSE_MASTER, "not_contain.json")
SALIENCE = os.path.join(ROSE_MASTER, "salience.json")
FACT_SHEETS = os.path.join(ROSE_MASTER, "fact_sheets.json")
CRITIQUE_REPORT = os.path.join(ROSE_MASTER, "we_critique_report.md")
PAIRS = os.path.join(ROSE_MASTER, "hard_negatives_rose.json")
PAIR_VERIFICATION = os.path.join(ROSE_MASTER, "pair_verification.json")
FROZEN = os.path.join(ROSE_MASTER, "pairs_rose_frozen.json")

# Construction model + settings, spec section 4 table row 2: Claude plan path, effort medium,
# $0 marginal. NOT the judge role - no paper-bound judge call is made by any construction script.
CONSTRUCT_MODEL = "claude-opus-4-8"
CONSTRUCT_EFFORT = "medium"
CONSTRUCT_TIMEOUT = 420

TYPES = ("add", "change", "omit")          # spec 2.2, W-A's taxonomy by name
IMPORTANCE = ("critical", "supporting", "peripheral")   # spec 3.3 WE_SALIENCE
SEED_INJECT = 20260730                     # spec 2.2 base seed for construction-side sampling


def load_prompts(names=None, strict=True):
    """Load we_prompts/*.txt and verify every byte against PROMPTS.sha256.

    Same discipline as w2_common.load_prompts (spec section 5 item 1: every runner checks the
    manifest and refuses to run on a mismatch). Returns {stem: text}.
    """
    manifest_path = os.path.join(PROMPT_DIR, "PROMPTS.sha256")
    if not os.path.exists(manifest_path):
        raise RuntimeError(f"{manifest_path} missing - run we_prompts/_extract_from_spec.py")
    manifest = {}
    with open(manifest_path) as f:
        for line in f:
            if line.strip():
                digest, name = line.split()
                manifest[name] = digest
    out, problems = {}, []
    for name, want in sorted(manifest.items()):
        path = os.path.join(PROMPT_DIR, name)
        if not os.path.exists(path):
            problems.append(f"{name}: listed in PROMPTS.sha256 but missing on disk")
            continue
        raw = open(path, "rb").read()
        got = hashlib.sha256(raw).hexdigest()
        if got != want:
            problems.append(f"{name}: sha256 {got[:16]} != frozen {want[:16]}")
        out[name[:-4] if name.endswith(".txt") else name] = raw.decode()
    if problems and strict:
        raise RuntimeError("frozen prompt check FAILED - refusing to run:\n  "
                           + "\n  ".join(problems))
    if names:
        missing = [n for n in names if n not in out]
        if missing:
            raise KeyError(f"prompts not found in we_prompts/: {missing}")
        out = {n: out[n] for n in names}
    return out


def prompt_hashes():
    """{filename: sha256} for the run manifest (Run.register_prompts equivalent)."""
    out = {}
    for name in sorted(os.listdir(PROMPT_DIR)):
        if name.endswith(".txt"):
            out[name] = hashlib.sha256(open(os.path.join(PROMPT_DIR, name), "rb").read()).hexdigest()
    return out


def load_sample():
    if not os.path.exists(SAMPLE):
        raise FileNotFoundError(f"{SAMPLE} missing - run we_fetch_rose.py then we_sample.py")
    return json.load(open(SAMPLE))["documents"]


def acu_list_block(acus):
    """The {acu_list} slot: 0-based indices, matching WE_SALIENCE's `acu_idx` and
    WE_EDIT_CHECK's `acu_idx` output fields."""
    return "\n".join(f"{i}. {a}" for i, a in enumerate(acus))


def pair_id(doc_id, typ):
    return f"{doc_id}__{typ}"


def construct(prompt, model=CONSTRUCT_MODEL, effort=CONSTRUCT_EFFORT, timeout=CONSTRUCT_TIMEOUT):
    """One construction/critic/verification call on the Claude plan path ($0 marginal, spec 2.4).
    Returns parsed JSON or None. Never used for a judge arm."""
    return claude_json(prompt, timeout=timeout, model=model, effort=effort)


def load_state(path, default=None):
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            pass
    return default if default is not None else {}


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, path)
