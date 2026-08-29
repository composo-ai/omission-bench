#!/usr/bin/env python3
"""Extract W2's 12 (13, counting the optional v14-incl arm) pre-registration
prompt files programmatically from specs/w2-ablation-grid.md.

Why this exists: these are frozen pre-registered instruments. Hand-retyping
them into files risks transcription drift, which would corrupt the
experiment. Instead this script PARSES the fenced code blocks straight out
of the spec (single source of truth) and writes them verbatim to
w2_prompts/*.txt, then verifies byte-equality against the spec text it read
them from. The two v14 arms are not verbatim blocks in the spec - they are
exact diffs against the real v14 source prompt - so this script also parses
those diff hunks out of the spec and applies them programmatically to the
source file, rather than hand-editing a copy.

Re-run this script any time the spec or the v14 source changes before
pre-registration freezes; after freeze, do not re-run against a changed
spec without a new dated Amendment (see sandbox/scribe_eval/CLAUDE.md).

Usage: python3 w2_prompts/_extract_from_spec.py
"""

from __future__ import annotations

import difflib
import hashlib
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent          # .../sandbox/scribe_eval/w2_prompts
SCRIBE_EVAL_DIR = SCRIPT_DIR.parent                    # .../sandbox/scribe_eval
REPO_ROOT = SCRIBE_EVAL_DIR.parent.parent

SPEC_PATH = SCRIBE_EVAL_DIR / "specs" / "w2-ablation-grid.md"
V14_SOURCE_PATH = (
    COMPOREPO_ROOT
    / "experiments"
    / "clinical-hallucination-detection"
    / "methods"
    / "v14_v9-none"
    / "prompt.txt"
)
OUT_DIR = SCRIPT_DIR
MANIFEST_PATH = OUT_DIR / "PROMPTS.sha256"

# Section 3.2 (grid prompts) + section 3.4 (external baselines): verbatim
# fenced blocks, each immediately preceded in the spec by a backtick-quoted
# `w2_prompts/<name>.txt` filename marker.
VERBATIM_FILES = [
    "grid_F_bin.txt",
    "grid_F_score.txt",
    "grid_FC_bin.txt",
    "grid_FC_score.txt",
    "geval_consistency.txt",
    "geval_completeness.txt",
    "ragas_faithfulness.txt",
    "ragas_extract.txt",
    "ragas_coverage.txt",
    "checklist_gen.txt",
    "checklist_answer.txt",
]


def extract_block_after_marker(text: str, marker: str, fence_lang: str = "") -> str:
    """Find `marker` in `text`, then return the content of the next fenced
    code block that follows it (without the ``` fences)."""
    idx = text.index(marker)  # raises ValueError (loudly) if marker absent
    pattern = re.compile(r"```" + re.escape(fence_lang) + r"\n(.*?)\n```", re.S)
    m = pattern.search(text, idx)
    if not m:
        raise RuntimeError(f"No fenced block found after marker: {marker!r}")
    return m.group(1)


def write_verbatim_file(name: str, content: str) -> None:
    path = OUT_DIR / name
    path.write_text(content + "\n", encoding="utf-8")


def verify_verbatim_file(name: str, expected_content: str) -> None:
    path = OUT_DIR / name
    written = path.read_bytes()
    expected = (expected_content + "\n").encode("utf-8")
    if written != expected:
        raise RuntimeError(f"BYTE MISMATCH writing {name} - aborting.")
    # Anti-slicing-bug guard: the exact block we wrote must appear verbatim
    # as a literal substring of the spec text (not just "close").
    spec_text = SPEC_PATH.read_text(encoding="utf-8")
    if expected_content not in spec_text:
        raise RuntimeError(
            f"{name}: extracted content not found verbatim in spec text - "
            "extraction logic bug, aborting."
        )
    print(f"  OK  {name}  ({len(written)} bytes, byte-exact vs spec block)")


# ---------------------------------------------------------------------------
# v14 diff arms
# ---------------------------------------------------------------------------


def parse_diff_hunks(diff_text: str) -> list[tuple[str, list[str]]]:
    """Parse the spec's simplified diff notation into a flat list of
    (sign, content_lines) runs, in order, across all hunks. `@@ ... @@`
    hunk-label lines are dropped. Each content line keeps everything after
    its single leading '-'/'+' diff-marker character (so a removed bullet
    line like '-- **Omissions.**...' correctly keeps its own leading '-').
    """
    runs: list[tuple[str, list[str]]] = []
    cur_sign = None
    cur_lines: list[str] = []

    def flush():
        if cur_lines:
            runs.append((cur_sign, cur_lines[:]))

    for line in diff_text.split("\n"):
        if line.startswith("@@"):
            flush()
            cur_lines.clear()
            cur_sign = None
            continue
        if not line:
            # A blank line inside a hunk body shouldn't occur in this spec's
            # diffs (blank *content* lines are represented as a lone '-' or
            # '+' marker with nothing after it) - guard against silent data
            # loss if the format ever changes.
            raise RuntimeError("Unexpected blank line inside a diff hunk body")
        sign, rest = line[0], line[1:]
        if sign not in ("-", "+"):
            raise RuntimeError(f"Unexpected diff line (no +/- marker): {line!r}")
        if sign != cur_sign:
            flush()
            cur_sign = sign
            cur_lines = [rest]
        else:
            cur_lines.append(rest)
    flush()
    return runs


def run_text(lines: list[str]) -> str:
    """Each parsed diff line is one raw source line; reconstruct with an
    explicit trailing '\\n' after every line (including a trailing blank
    line, which shows up as an empty string), so the block is byte-exact
    against how it sits in the source file."""
    return "".join(line + "\n" for line in lines)


def apply_replace_or_delete(source: str, removal: str, addition: str, label: str) -> str:
    count = source.count(removal)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly 1 occurrence of the removal block in "
            f"the source, found {count}. Refusing to guess - stop."
        )
    return source.replace(removal, addition)


def build_v14_noexcl(source_text: str) -> str:
    diff_block = extract_block_after_marker(
        SPEC_PATH.read_text(encoding="utf-8"), "`w2_prompts/v14_noexcl.txt`", "diff"
    )
    runs = parse_diff_hunks(diff_block)
    # Expect exactly: [-, +, -]  -> hunk1 pure delete (4 lines incl. trailing
    # blank), hunk2 delete+add (3+3 lines, the omission clause removed).
    # Reconstruct hunk boundaries by pairing consecutive runs: a '-' run
    # followed immediately by a '+' run is a replace; a lone '-' run with no
    # following '+' (before the next '-' or end) is a pure delete.
    result = source_text
    i = 0
    applied = []
    while i < len(runs):
        sign, lines = runs[i]
        if sign != "-":
            raise RuntimeError("v14_noexcl diff: expected a '-' run to start each hunk")
        removal = run_text(lines)
        if i + 1 < len(runs) and runs[i + 1][0] == "+":
            addition = run_text(runs[i + 1][1])
            i += 2
        else:
            addition = ""
            i += 1
        result = apply_replace_or_delete(result, removal, addition, "v14_noexcl")
        applied.append((removal, addition))
    if len(applied) != 2:
        raise RuntimeError(f"v14_noexcl: expected 2 hunks, parsed {len(applied)} - stop.")
    return result


def build_v14_incl(noexcl_text: str) -> str:
    diff_block = extract_block_after_marker(
        SPEC_PATH.read_text(encoding="utf-8"), "`w2_prompts/v14_incl.txt`", "diff"
    )
    runs = parse_diff_hunks(diff_block)
    if len(runs) != 1 or runs[0][0] != "+":
        raise RuntimeError(f"v14_incl: expected exactly 1 pure '+' run, got {runs}")
    addition_lines = runs[0][1]
    addition_text = run_text(addition_lines)  # ends with a single trailing \n

    # The spec's diff gives no insertion anchor (it's a pure-addition hunk
    # under the "What COUNTS as flag-worthy" section) - unlike the two
    # v14_noexcl hunks, which are fully explicit deletions. Judgment call,
    # flagged here and in the run report: insert item 4 as a new list item
    # at the end of that section, following the existing convention in the
    # source (each numbered item separated by exactly one blank line) -
    # i.e. right before the blank line + next section heading.
    anchor = (
        "   or shifts clinical interpretation.\n"
        "\n"
        "### What does NOT count as flag-worthy"
    )
    if noexcl_text.count(anchor) != 1:
        raise RuntimeError("v14_incl: insertion anchor not found exactly once - stop.")
    replacement = (
        "   or shifts clinical interpretation.\n"
        "\n" + addition_text + "\n"
        "### What does NOT count as flag-worthy"
    )
    return noexcl_text.replace(anchor, replacement)


def verify_spec_quotes_match_source(source_text: str) -> None:
    """Section 3.3 quotes v14's omission-exclusion text inline (not as a
    diff). Cross-check those quotes against the real source file before
    trusting the diff-derived arms at all - per the task's explicit
    instruction, a mismatch here is a pre-registration issue for humans,
    not something to paper over."""
    spec_text = SPEC_PATH.read_text(encoding="utf-8")
    quoted_omission = extract_block_after_marker(
        spec_text, 'Under "What does NOT count as flag-worthy":'
    )
    quoted_step3 = extract_block_after_marker(spec_text, "And in Instructions step 3:")

    if quoted_omission not in source_text:
        print("DISCREPANCY: spec's quoted 'Omissions' bullet does NOT match "
              "the real v14 source file verbatim. STOPPING - do not guess; "
              "this is a pre-registration issue for humans to resolve.",
              file=sys.stderr)
        sys.exit(2)
    if quoted_step3 not in source_text:
        print("DISCREPANCY: spec's quoted Instructions step 3 does NOT match "
              "the real v14 source file verbatim. STOPPING - do not guess; "
              "this is a pre-registration issue for humans to resolve.",
              file=sys.stderr)
        sys.exit(2)
    print("  OK  spec's quoted v14 omission-exclusion text matches the real "
          "source file verbatim (both hunks' anchors verified).")


def print_diff(label: str, before: str, after: str) -> None:
    diff = list(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"{label} (source)",
            tofile=f"{label} (result)",
        )
    )
    print(f"\n--- unified diff: {label} ---")
    print("".join(diff) if diff else "(no difference - unexpected)")
    print(f"--- end diff: {label} ---\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec_text = SPEC_PATH.read_text(encoding="utf-8")

    print(f"Spec: {SPEC_PATH}")
    print(f"v14 source: {V14_SOURCE_PATH}")
    print()

    print("== Verbatim blocks (section 3.2 grid prompts + section 3.4 baselines) ==")
    for name in VERBATIM_FILES:
        marker = f"`w2_prompts/{name}`"
        content = extract_block_after_marker(spec_text, marker)
        write_verbatim_file(name, content)
        verify_verbatim_file(name, content)

    print("\n== v14 diff arms (section 3.3) ==")
    if not V14_SOURCE_PATH.exists():
        raise RuntimeError(f"v14 source prompt not found: {V14_SOURCE_PATH}")
    source_text = V14_SOURCE_PATH.read_text(encoding="utf-8")

    verify_spec_quotes_match_source(source_text)

    noexcl_text = build_v14_noexcl(source_text)
    (OUT_DIR / "v14_noexcl.txt").write_text(noexcl_text, encoding="utf-8")
    print_diff("v14_noexcl vs source", source_text, noexcl_text)

    incl_text = build_v14_incl(noexcl_text)
    (OUT_DIR / "v14_incl.txt").write_text(incl_text, encoding="utf-8")
    print_diff("v14_incl vs v14_noexcl", noexcl_text, incl_text)
    print_diff("v14_incl vs source (both hunks combined)", source_text, incl_text)

    # ------------------------------------------------------------------
    # PROMPTS.sha256 manifest: every .txt file in this directory.
    # ------------------------------------------------------------------
    print("\n== PROMPTS.sha256 manifest ==")
    txt_files = sorted(p.name for p in OUT_DIR.glob("*.txt"))
    lines = []
    for name in txt_files:
        digest = hashlib.sha256((OUT_DIR / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}")
        print(f"  {digest}  {name}")
    MANIFEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {MANIFEST_PATH} ({len(txt_files)} files).")
    print(f"\nTotal named prompt files: {len(txt_files)} "
          f"({len(VERBATIM_FILES)} verbatim + 2 v14 diff-derived arms, "
          "one of which - v14_incl - is optional per section 3.3).")


if __name__ == "__main__":
    main()
