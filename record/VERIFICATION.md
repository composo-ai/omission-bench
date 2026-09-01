# How this release was verified

Every file in both halves of this release - the dataset repository and this code
repository - was scanned before publication by a step that fails the build rather than
warning. Three conditions, no exemptions, no waivers, and a bundle that trips any of them
does not ship.

**A. No scribe product name.** In a file's contents or in its path, matched
case-insensitively, both as a whole word and as a fragment inside a longer token. Clinical
vocabulary that legitimately contains one of those fragments is allowed through by an
explicit list, which is not reproduced here for the same reason the checking code is not:
naming its members would name what it exists to keep out. The list can only explain away
the part of a token that actually carries the fragment; it cannot be used to launder a
name glued to a real word by a hyphen or an underscore.

**B. No withheld note text.** The three scribe systems' own note text is withheld for
licence reasons and is not in this release in any form. The check measures how much of any
one withheld note a released file reproduces, using overlapping twelve-word runs, and
fails on fifteen distinct matching runs or twenty percent of a note - whichever comes
first. It runs twice over the same evidence: once per released string, which catches a note
pasted into a field, and once per file with every string's matches accumulated, which
catches a note chopped into fragments that are individually harmless. Runs that also appear
verbatim in a consultation transcript this release publishes are discounted, because the
scribe copied the transcript and the transcript is here by commitment. There is no separate
span-coverage check any more: no note-side character offsets or spans ship, and note text
appears only as the short quoted phrases inside ten of the 618 finding descriptions -
internal contradictions that cannot be stated without them, disclosed in the findings file
itself - which the run-length measure above is sized to catch if they ever grew.

**C. Neither banned string.** Two strings this project's own integrity rules forbid in
released material.

**What gets scanned.** Every file. JSON and JSONL are parsed and walked to every string
value; gzipped stores are decompressed and parsed line by line; markdown, plain text,
prompts and source are scanned whole. A file that cannot be decoded as UTF-8, or that is
named as JSON and does not parse, is itself reported as a finding - silence is never taken
for a pass. Only genuine binaries are skipped, and their paths are still checked.

**How it ran.** The packaging step builds the two trees, then starts the checker in a
separate process against the finished trees on disk, and fails if that process reports
anything. The checker can be pointed at an unpacked copy of this release at any time and
will reproduce the same verdict; it assumes nothing about how the tree was made.

The checking code itself is not in this bundle, and that is deliberate. A checker that
searches for forbidden strings must contain those strings, and its tests must contain
fixtures that trip it - so publishing it would publish, in this bundle, exactly what it
exists to keep out.
