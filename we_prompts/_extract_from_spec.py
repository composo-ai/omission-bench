#!/usr/bin/env python3
"""Extract W-E's pre-registration prompt files programmatically from
specs/w-e-cross-domain-generalization.md.

Same reasoning as w2_prompts/_extract_from_spec.py, which this mirrors: these are frozen
pre-registered instruments, and hand-retyping them into files risks transcription drift that
would corrupt the experiment. This script PARSES the fenced code blocks straight out of the spec
(single source of truth), writes them verbatim to we_prompts/*.txt, asserts each written block
appears verbatim as a literal substring of the spec, and writes PROMPTS.sha256.

Spec section 3: "All prompts below are new ... frozen verbatim at pre-registration, stored in
sandbox/scribe_eval/we_prompts/, hashed into a PROMPTS.sha256 manifest exactly as W2 section 3
does." Spec section 5 item 1 requires every W-E runner to check that manifest and refuse to run
on a mismatch (we_common.load_prompts does this).

File names: sections 3.2's four grid prompts name their own files in the spec. The construction,
critic, injection and verification prompts (3.3-3.5) are named in the spec only by prompt id
(WE_NOT_CONTAIN etc.), so the file name is the lower-cased id - recorded here so the mapping is
explicit rather than folklore.

Re-run this script any time the spec changes before the pre-registration freeze; after freeze,
do not re-run against a changed spec without a new dated Amendment (see ../CLAUDE.md).

Usage: python3 we_prompts/_extract_from_spec.py
"""

import hashlib
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIBE_EVAL_DIR = SCRIPT_DIR.parent
SPEC_PATH = SCRIBE_EVAL_DIR / "specs" / "w-e-cross-domain-generalization.md"
MANIFEST_PATH = SCRIPT_DIR / "PROMPTS.sha256"

# (output file, marker text that immediately precedes the fenced block in the spec, spec section)
BLOCKS = [
    # 3.1 - the criterion minimal pair (the paragraphs the grid prompts embed)
    ("crit_FC.txt", "**WE-CRIT-FC** - completeness-aware:", "3.1"),
    ("crit_F.txt", "**WE-CRIT-F** - faithfulness-only, the minimal pair:", "3.1"),
    # 3.2 - the four grid prompts (files named by the spec itself)
    ("grid_F_bin.txt", "**WE-P-F-bin** (`we_prompts/grid_F_bin.txt`):", "3.2"),
    ("grid_F_score.txt", "**WE-P-F-score** (`we_prompts/grid_F_score.txt`):", "3.2"),
    ("grid_FC_bin.txt", "**WE-P-FC-bin** (`we_prompts/grid_FC_bin.txt`):", "3.2"),
    ("grid_FC_score.txt", "**WE-P-FC-score** (`we_prompts/grid_FC_score.txt`)", "3.2"),
    # 3.3 - fact-sheet-analog construction
    ("we_not_contain.txt", "**`WE_NOT_CONTAIN`** - generates the `must_not_contain` pool", "3.3"),
    ("we_salience.txt", "**`WE_SALIENCE`** - grades every ACU's importance", "3.3"),
    # 3.4 - critic panel
    ("we_critic_support.txt", "**`WE_CRITIC_SUPPORT`**:", "3.4"),
    ("we_critic_plausibility.txt", "**`WE_CRITIC_PLAUSIBILITY`**:", "3.4"),
    # 3.5 - injection + semantic verification
    ("we_inject.txt", "**`WE_INJECT`** (one call per pair;", "3.5"),
    ("we_inject_instr.txt", "Per-type `{instr}` (the `TYPES` analog):", "3.5"),
    ("we_edit_check.txt", "**`WE_EDIT_CHECK`** (semantic verification,", "3.5"),
]

FENCE = re.compile(r"```[a-z]*\n(.*?)\n```", re.S)


def extract_block_after_marker(text, marker):
    idx = text.index(marker)          # raises ValueError (loudly) if the marker moved
    m = FENCE.search(text, idx)
    if not m:
        raise RuntimeError(f"no fenced block found after marker: {marker!r}")
    return m.group(1)


def main():
    spec = SPEC_PATH.read_text(encoding="utf-8")
    written = []
    for name, marker, section in BLOCKS:
        content = extract_block_after_marker(spec, marker)
        if content not in spec:
            raise RuntimeError(f"{name}: extracted content not a verbatim substring of the spec")
        path = SCRIPT_DIR / name
        path.write_text(content + "\n", encoding="utf-8")
        blob = path.read_bytes()
        if blob != (content + "\n").encode("utf-8"):
            raise RuntimeError(f"BYTE MISMATCH writing {name}")
        digest = hashlib.sha256(blob).hexdigest()
        written.append((name, digest, len(blob), section))
        print(f"  OK  {name:28s} {len(blob):5d} bytes  (spec {section}, byte-exact)")

    MANIFEST_PATH.write_text(
        "".join(f"{d}  {n}\n" for n, d, _, _ in sorted(written)), encoding="utf-8")
    print(f"wrote {MANIFEST_PATH} ({len(written)} prompts)")

    # sanity: the F/FC criterion pair must be the exact two-deletion minimal pair (spec 3.1)
    fc = (SCRIPT_DIR / "crit_FC.txt").read_text(encoding="utf-8")
    f = (SCRIPT_DIR / "crit_F.txt").read_text(encoding="utf-8")
    rebuilt = fc.replace(" AND complete", "").replace(
        ", and no omission of salient information that a careful reader would expect the summary "
        "to cover", "")
    if rebuilt != f:
        print("WARNING: crit_F.txt is not crit_FC.txt minus exactly the two pre-registered "
              "deletions - check spec 3.1 before freezing", file=sys.stderr)
        sys.exit(1)
    print("  criterion minimal-pair check: PASS (F == FC minus exactly the two deletions)")


if __name__ == "__main__":
    main()
