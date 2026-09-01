"""The pilot study's failure-family matcher, reduced to the part this release uses.

Findings come out of the discovery passes carrying free-text mode labels that vary from run
to run, so before anything could ask whether the same failure recurred across products on
the same consultation, those labels had to be mapped onto a fixed set of families.
`family()` is that map and `FAMILIES` is the set: eleven mechanism families, matched by
regular expression against a finding's mode and description, with a generic omission family
last as the catch-all.

Several modules in this repository import this file by path rather than copying the
expressions, so the mapping has one definition and cannot drift between the analysis stages
that use it. `taxonomy_common.assert_family_keys_known` checks these keys against the
published reporting frame in `taxonomy_frame.json`.

Only `FAMILIES` and `family` are used downstream. The pilot's own cross-product analysis
around them read a working file this release does not carry and wrote a markdown summary to
a local folder; it has been removed rather than shipped dead.
"""
import re

# map the varied emergent labels onto a few canonical families so cross-scribe matching works
FAMILIES = [
    ("exam_not_done", r"exam|remote|physical|palpat|history framed"),
    ("demographics", r"demograph|sex|gender|age|patient (attribute|detail)"),
    ("onset_timing", r"onset|temporal|timing|sudden|thunderclap|duration"),
    ("diagnosis_omitted", r"diagnos|impression|assessment|working|rationale"),
    ("safety_net", r"safety|safety-net|red flag|red-flag"),
    ("laterality_site", r"laterality|site|location|body site|distribution"),
    ("fabrication", r"fabricat|invent|unsupported|hallucin|assert"),
    ("hardening", r"harden|modality|definite|conditional|made definite|overstat"),
    ("attribution", r"attribution|subject|contact history|epidemiolog"),
    ("negation", r"negat|denied|pertinent negative"),
    ("omission", r"omit|omission|dropped|missing"),   # generic omission last (catch-all)
]


def family(f):
    blob = f"{f.get('mode','')} {f.get('description','')}".lower()
    for name, rx in FAMILIES:
        if re.search(rx, blob):
            return name
    return f.get("mode", "other")


