#!/usr/bin/env python3
"""Build the blinded audit-stage severity validation pack.

WHY THIS EXISTS
---------------
The paper's best deployable result is a per-fact decision rule: flag the note iff
any fact the pipeline's AUDIT STAGE graded `critical` is verdicted `absent`
(20.6% detection at 2.1% false alarms on the held-out set). The earlier
clinician sitting validated the BENCHMARK's severity grades (70% exact,
kappa 0.63). It did not touch the audit stage's OWN grades, which
are the ones the rule actually fires on. This pack closes that gap: a blinded
physician pass over 24 audit-stage-graded facts.

WHAT IT READS (read-only; in the study these sat behind a symlink, and in this
repository they resolve from the root - nothing is ever
written back into the study's working tree, and no harness module is
imported, so no __pycache__ is created there either):

  provenance/confirm-B3.jsonl (in this repository)   the 47-consultation
        confirmation run: which consultations are in it, and (via
        detail.missing_facts) which facts the per-fact rule fired on.
  results/w2-pipeline/_cache/audit_<stratum>__<id>.json (not part of this release)   the audit
        stage's own output per consultation: every kept fact with its text,
        its severity grade and the auditor's reason. THE SEVERITY GRADES UNDER
        TEST LIVE HERE.
  the transcript stores listed in TRANSCRIPT_SOURCES (copied verbatim from
        w2_common.py) - the same resolution order the pipeline used.
        Cross-checked against each B3 record's own `transcript_chars`.
  master/severity-rubric.md (save the rubric there; see the README)   reproduced verbatim in the pack.

WHAT IT WRITES (both in this folder):

  severity-pack.html        the self-contained sitting tool. Blind: it carries
        the fact text and the full transcript and nothing else. No machine
        grade, no note, no verdict, no consultation id, no fact id, no hint
        that the rule fired.
  severity_pack_keys.json   the answer key: machine grade + full provenance per
        item id. Not to be opened before the sitting.

DETERMINISM
-----------
Seeded at SEED with every candidate list sorted before it is sampled, and no
build timestamp anywhere in either output, so two runs are byte-identical.

    python3 sittings/build_severity_pack.py            build + self-check
    python3 sittings/build_severity_pack.py --check    self-check only, write nothing
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path

import hashlib
import html
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # the repository root
HARNESS = HERE  # the study reached these paths through a symlink; here, the root
B3_PATH = HARNESS / "provenance" / "confirm-B3.jsonl"
AUDIT_DIR = HARNESS / "results" / "w2-pipeline" / "_cache"
import os as _os  # noqa: E402 - for the SEVERITY_RUBRIC override below
RUBRIC_PATH = Path(_os.environ.get("SEVERITY_RUBRIC",
                                   str(HARNESS / "master" / "severity-rubric.md")))
OUT_HTML = HERE / "severity-pack.html"
OUT_KEYS = HERE / "severity_pack_keys.json"

# ---- the sample -----------------------------------------------------------
SEED = 20260817
N_ITEMS = 24
N_FIRED = 10                       # stratum A ceiling: facts the rule fired on
RANDOM_QUOTA = [("critical", 5), ("supporting", 5), ("peripheral", 4)]
# The one clean-note false alarm on the held-out set: the whole 2.1% FA rate is
# this single fact, so the sitting must include it.
FORCED_FIRED = ("aci", "D2N199", "f27")
MAX_PER_CONSULT = 2                # coverage cap across the whole pack

# ---- transcript resolution, copied verbatim from w2_common.py -------------
# (stratum, path) - first source that carries a (stratum, id) wins, exactly as
# transcript_index() resolves it. A None stratum means the file's rows may key
# any stratum.
TRANSCRIPT_SOURCES = [
    (None, "master/transcripts_master.json"),
    ("authored", "authored_scenarios.json"),
    ("trapblind", "master/fact_sheets_trapblind_core.json"),
    ("primock", "master/fact_sheets_primock_core.json"),
    ("aci", "master/fact_sheets_aci_core.json"),
    ("authored", "master/fact_sheets_authored_extracted_core.json"),
]
STRATA = ("authored", "trapblind", "primock", "aci")

# ---- the answer scheme ----------------------------------------------------
OPTS = [
    {"c": "critical", "l": "Critical", "k": "c"},
    {"c": "supporting", "l": "Supporting", "k": "s"},
    {"c": "peripheral", "l": "Peripheral", "k": "p"},
    {"c": "?", "l": "Can't judge", "k": "u"},
]

QUESTION = ("Grade this fact's clinical importance per the rubric: if the note "
            "carried an error on this fact (omitting, fabricating or altering "
            "it), how much would it matter?")


# ==========================================================================
# sources
# ==========================================================================

def transcript_index():
    """(stratum, id) -> transcript, plus a (stratum, id) -> source file map.

    Same resolution order as w2_common.py transcript_index(); the file
    map is extra, so the keys can record where each transcript came from."""
    idx, src = {}, {}
    for stratum, rel in TRANSCRIPT_SOURCES:
        path = HARNESS / rel
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else [
            (dict(v, id=k) if isinstance(v, dict) else {"id": k, "transcript": v})
            for k, v in data.items()]
        for r in rows:
            if not isinstance(r, dict):
                continue
            t = r.get("transcript") or r.get("transcript_text") or ""
            rid, st = r.get("id"), r.get("stratum", stratum)
            if not (t and rid):
                continue
            for key in ({(st, rid)} if st else {(s, rid) for s in STRATA}):
                if key not in idx:
                    idx[key] = t
                    src[key] = rel
    return idx, src


def load_b3():
    """The confirmation run's records, its consultation set, and the per-fact
    rule's firings.

    A firing = an audit-graded `critical` fact verdicted `absent` on some note.
    Keyed by (stratum, consultation, fact_id) because the pack's unit is a FACT,
    while the rule's unit is a note - one fact can fire on several notes."""
    records = [json.loads(l) for l in B3_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    consults = sorted({(r["stratum"], r["consultation"]) for r in records})
    fired = {}
    for r in records:
        for mf in r["detail"]["missing_facts"]:
            if mf.get("severity") == "critical" and mf.get("verdict") == "absent":
                key = (r["stratum"], r["consultation"], mf["id"])
                e = fired.setdefault(key, {"fact": mf["fact"], "notes": []})
                e["notes"].append({"note_key": r["note_key"], "note_role": r["note_role"],
                                   "pair_type": r["pair_type"], "verdict": mf["verdict"]})
    for e in fired.values():
        e["notes"].sort(key=lambda n: n["note_key"])
    return records, consults, fired


def load_audits(consults):
    """(stratum, consultation) -> the audit stage's own output for it."""
    audits = {}
    for st, cid in consults:
        path = AUDIT_DIR / ("audit_%s__%s.json" % (st, cid))
        audits[(st, cid)] = json.loads(path.read_text(encoding="utf-8"))
    return audits


# ==========================================================================
# sampling (deterministic)
# ==========================================================================

def pick(rng, pool, k):
    """rng.sample over a sorted pool - order in, order out, every time."""
    pool = sorted(pool)
    if k >= len(pool):
        return pool
    return sorted(rng.sample(pool, k))


NEAR_DUPLICATE = 0.5           # Jaccard over lowercased word sets


def separate_near_duplicates(chosen, audits):
    """Keep two near-identical facts off consecutive screens.

    The corpus contains the same clinical statement in more than one
    consultation ("dorsalis pedis and posterior tibial pulses were palpable").
    Back to back they read as a bug and the second answer anchors on the first,
    which is exactly the independence the sitting needs. This touches the ORDER
    only - the 24 facts drawn are not affected - and order was already an
    arbitrary choice made for blinding. Greedy and deterministic: keep the
    shuffled order except where it would place a near-duplicate next to its twin,
    and then take the next item that is not one. On this pack the overlap
    distribution is bimodal (one pair at 0.78, everything else <= 0.2), so the
    0.5 threshold cannot misfire."""
    def toks(k):
        fact = next(f["fact"] for f in audits[(k[0], k[1])]["facts"] if f["id"] == k[2])
        return set(re.findall(r"[a-z0-9]+", fact.lower()))

    def near(a, b):
        A, B = toks(a), toks(b)
        return bool(A and B) and len(A & B) / len(A | B) >= NEAR_DUPLICATE

    rest, out, moved = list(chosen), [chosen[0]], []
    rest.pop(0)
    while rest:
        i = next((j for j, x in enumerate(rest) if not near(out[-1], x)), 0)
        if i:
            moved.append(("|".join(out[-1][:3]), "|".join(rest[0][:3])))
        out.append(rest.pop(i))
    return out, moved


def sample_items(fired, audits):
    """Stratum A: up to N_FIRED facts the rule fired on, one per consultation,
    with the clean-note false alarm forced in.
    Stratum B: the remainder, quota'd across the three audit grades, drawn from
    facts the rule did NOT fire on, preferring consultations the pack has not
    used yet and never exceeding MAX_PER_CONSULT.

    Returns (items, notes, order_notes, used). `notes` records anything the
    stores could not supply, so a shortfall is stated rather than silently
    absorbed; `order_notes` records presentation-only reorderings."""
    rng = random.Random(SEED)
    notes = []
    used = Counter()
    chosen = []          # (stratum, consultation, fact_id, draw_stratum)

    # ---- stratum A ---------------------------------------------------------
    by_consult = defaultdict(list)
    for st, cid, fid in fired:
        by_consult[(st, cid)].append(fid)

    forced_key = FORCED_FIRED
    if forced_key in fired:
        chosen.append(forced_key + ("rule-fired",))
        used[forced_key[:2]] += 1
    else:
        notes.append("the named clean-note false alarm %s is NOT in the store; "
                     "stratum A was drawn without it" % ("|".join(forced_key),))

    want_a = min(N_FIRED, len(fired)) - len(chosen)
    pool_a = [c for c in by_consult if used[c] == 0]
    if len(pool_a) < want_a:
        notes.append("stratum A wanted %d further consultations with a rule-fired fact and the "
                     "store holds %d" % (want_a, len(pool_a)))
        want_a = len(pool_a)
    for c in pick(rng, pool_a, want_a):
        fid = rng.choice(sorted(by_consult[c]))
        chosen.append((c[0], c[1], fid, "rule-fired"))
        used[c] += 1

    n_fired_chosen = len(chosen)

    # ---- stratum B ---------------------------------------------------------
    remaining = N_ITEMS - n_fired_chosen
    quota = [(g, n) for g, n in RANDOM_QUOTA]
    short = remaining - sum(n for _, n in quota)
    if short:            # keep the pack at N_ITEMS if stratum A came up short
        quota = [(g, n + (short if i == 0 else 0)) for i, (g, n) in enumerate(quota)]
        notes.append("stratum A supplied %d of %d, so the random-critical quota absorbs the "
                     "shortfall (+%d)" % (n_fired_chosen, N_FIRED, short))

    cand = defaultdict(lambda: defaultdict(list))    # grade -> consult -> [fact_id]
    for c, a in audits.items():
        for f in a["facts"]:
            if (c[0], c[1], f["id"]) in fired:
                continue                              # keep the strata disjoint
            cand[f["severity"]][c].append(f["id"])

    for grade, k in quota:
        if k <= 0:
            continue
        fresh = [c for c in cand[grade] if used[c] == 0]
        take = pick(rng, fresh, k)
        if len(take) < k:                             # fall back to the cap of 2
            reuse = [c for c in cand[grade] if 0 < used[c] < MAX_PER_CONSULT and c not in take]
            take = sorted(take + pick(rng, reuse, k - len(take)))
        if len(take) < k:
            notes.append("the random-%s quota wanted %d consultations and the store supplied %d"
                         % (grade, k, len(take)))
        for c in take:
            fid = rng.choice(sorted(cand[grade][c]))
            chosen.append((c[0], c[1], fid, "random-" + grade))
            used[c] += 1

    # ---- shuffle, so neither the order nor the id encodes the stratum -------
    chosen.sort()
    rng.shuffle(chosen)
    chosen, moved = separate_near_duplicates(chosen, audits)
    order_notes = ["near-duplicate facts %s and %s were pulled apart in the running order "
                   "(presentation only - the sample is unchanged)" % (a, b) for a, b in moved]

    tags, items = {}, []
    for i, (st, cid, fid, draw) in enumerate(chosen):
        c = (st, cid)
        if c not in tags:
            tags[c] = "T-%02d" % (len(tags) + 1)
        fact = next(f for f in audits[c]["facts"] if f["id"] == fid)
        items.append({
            "id": "S%02d" % (i + 1),
            "tag": tags[c],
            "stratum": st,
            "consultation": cid,
            "fact_id": fid,
            "fact": fact["fact"],
            "machine_severity": fact["severity"],
            "machine_why": fact["why"],
            "draw": draw,
        })
    return items, notes, order_notes, used


# ==========================================================================
# rendering
# ==========================================================================

def esc(s):
    return html.escape(s, quote=False)


def inline_md(t):
    """`code`, **bold**, *italics* - all the rubric uses."""
    codes = []

    def grab(m):
        codes.append(m.group(1))
        return "\x00%d\x00" % (len(codes) - 1)

    t = re.sub(r"`([^`]+)`", grab, t)
    t = esc(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"<em>\1</em>", t)
    return re.sub(r"\x00(\d+)\x00", lambda m: "<code>%s</code>" % esc(codes[int(m.group(1))]), t)


def render_md(text):
    """Markdown -> html for the rubric: headings, bullets, numbered items with
    continuation lines, paragraphs. Reflows soft-wrapped source lines the way
    markdown does; the words are untouched."""
    out, para, lst, ordered = [], [], [], False

    def flush_para():
        if para:
            out.append("<p>%s</p>" % inline_md(" ".join(para)))
            para.clear()

    def flush_list():
        if lst:
            tag = "ol" if ordered else "ul"
            out.append("<%s>%s</%s>" % (tag, "".join("<li>%s</li>" % inline_md(" ".join(i))
                                                     for i in lst), tag))
            lst.clear()

    for raw in text.split("\n"):
        line = raw.rstrip()
        s = line.strip()
        if not s:
            flush_para()
            flush_list()
            continue
        m_h = re.match(r"^(#{1,4})\s+(.*)$", s)
        m_b = re.match(r"^[-*]\s+(.*)$", s)
        m_n = re.match(r"^(\d+)\.\s+(.*)$", s)
        if m_h:
            flush_para()
            flush_list()
            lvl = min(len(m_h.group(1)) + 2, 5)
            out.append("<h%d>%s</h%d>" % (lvl, inline_md(m_h.group(2)), lvl))
        elif m_b or m_n:
            flush_para()
            new_ordered = bool(m_n)
            if lst and new_ordered != ordered:
                flush_list()
            ordered = new_ordered
            lst.append([m_n.group(2) if m_n else m_b.group(1)])
        elif lst and line[:1] in (" ", "\t"):
            lst[-1].append(s)                      # continuation of a list item
        else:
            flush_list()
            para.append(s)
    flush_para()
    flush_list()
    return "\n".join(out)


SPEAKER_RE = re.compile(r"^\[([a-z_]+)\]\s*(.*)$")
SPEAKER_RE2 = re.compile(r"^([A-Z][a-z]+(?: [A-Z]?[a-z]+)?):\s*(.*)$")


def render_transcript(text):
    """One paragraph per source line, speaker label pulled out for legibility.
    Every word of every line survives; only the leading/trailing whitespace and
    the blank lines go."""
    out = []
    for raw in text.split("\n"):
        s = raw.strip()
        if not s:
            continue
        m = SPEAKER_RE.match(s) or SPEAKER_RE2.match(s)
        if m and m.group(2):
            out.append('<p class="turn"><span class="who">%s</span>%s</p>'
                       % (esc(m.group(1)), esc(m.group(2))))
        else:
            out.append('<p class="turn cont">%s</p>' % esc(s))
    return "\n".join(out)


def build(items, transcripts, tsrc, rubric_text):
    seen = {}
    cards, meta_items = [], []
    for pos, it in enumerate(items):
        c = (it["stratum"], it["consultation"])
        t = transcripts[c]
        first = seen.get(it["tag"])
        again = ("" if first is None else
                 '<p class="again">You have already read <b>%s</b> - it was item %s. '
                 'Grade this fact on its own merits.</p>' % (it["tag"], first))
        seen.setdefault(it["tag"], it["id"])
        opts_html = "".join(
            '<button class="opt" type="button" data-code="%s">%s<span class="kbd">%s</span></button>'
            % (o["c"], esc(o["l"]), o["k"]) for o in OPTS)
        cards.append(
            '<article class="card" id="card-%s" data-id="%s" hidden>'
            '<div class="crumb"><span class="pill">%s</span>'
            '<span class="cid">%s</span>'
            '<span class="cpos">item %d of %d</span></div>'
            '<p class="lbl">A fact drawn from this consultation:</p>'
            '<blockquote class="fact">%s</blockquote>'
            '%s'
            '<p class="q">%s</p>'
            '<details class="brief"><summary>Full transcript (%s)</summary>'
            '<div class="brief-body tx">%s</div></details>'
            '<div class="answer">%s</div>'
            '<div class="notewrap">'
            '<button class="addnote" type="button">+ add comment</button>'
            '<textarea class="note" rows="3" hidden '
            'placeholder="Anything the three grades do not carry - what tipped it, what you '
            'would need to know, where the fact is wrong about the consultation."></textarea>'
            "</div></article>"
            % (it["id"], it["id"], it["tag"], it["id"], pos + 1, len(items),
               esc(it["fact"]), again, esc(QUESTION), it["tag"], render_transcript(t),
               opts_html))
        meta_items.append({"id": it["id"], "tag": it["tag"], "fact": it["fact"],
                           "opts": OPTS})

    pack_sha = hashlib.sha256("\x1e".join(
        "\x1f".join([it["id"], it["tag"], it["fact"],
                     hashlib.sha256(transcripts[(it["stratum"], it["consultation"])]
                                    .encode("utf-8")).hexdigest()[:16],
                     ",".join(o["c"] for o in OPTS)])
        for it in items).encode("utf-8")).hexdigest()[:12]

    meta = {
        # no build timestamp on purpose: an unchanged source rebuilds byte-identical
        # pack_sha keys browser storage off the pack's CONTENT (ids, tags, fact
        # text, transcript hashes, answer codes), so answers can never survive
        # into a pack where the same id means a different fact
        "pack": "audit-stage severity validation",
        "pack_sha": pack_sha,
        "total": len(items),
        "mins": 25,
        "items": meta_items,
    }
    doc = (HTML_SHELL
           .replace("__TOTAL__", str(len(items)))
           .replace("__RUBRIC__", render_md(rubric_text))
           .replace("__RUBRIC_RAW__", esc(rubric_text))
           .replace("__CARDS__", "\n".join(cards))
           .replace("__META__", json.dumps(meta, ensure_ascii=False).replace("<", "\\u003c")))
    return doc, meta


def build_keys(items, fired, audits, transcripts, tsrc, notes, order_notes, used,
               records, consults):
    """Machine grades + provenance per item. Every field here is a lookup into a
    real record; nothing is computed for presentation."""
    key_items = []
    for it in items:
        c = (it["stratum"], it["consultation"])
        fk = (it["stratum"], it["consultation"], it["fact_id"])
        f = fired.get(fk)
        a = audits[c]
        key_items.append({
            "id": it["id"],
            "consultation_tag_in_pack": it["tag"],
            "machine_severity": it["machine_severity"],
            "machine_why": it["machine_why"],
            "draw_stratum": it["draw"],
            "rule_fired": bool(f),
            "rule_fired_on_notes": ([n["note_key"] for n in f["notes"]] if f else []),
            "rule_fired_on_clean_note": bool(f and any(n["note_role"] == "clean" for n in f["notes"])),
            "provenance": {
                "stratum": it["stratum"],
                "consultation": it["consultation"],
                "fact_id": it["fact_id"],
                "fact": it["fact"],
                "audit_record": "results/w2-pipeline/_cache/audit_%s__%s.json"
                                % (it["stratum"], it["consultation"]),
                "audit_prompt_sha256": a["prompt_sha256"],
                "audit_generated_utc": a["generated_utc"],
                "audit_n_kept": a["n_kept"],
                "transcript_source": tsrc[c],
                "transcript_sha256": hashlib.sha256(transcripts[c].encode("utf-8")).hexdigest(),
                "transcript_chars": len(transcripts[c]),
            },
        })

    sev_universe = Counter()
    for a in audits.values():
        sev_universe.update(f["severity"] for f in a["facts"])

    return {
        "_": "KEYS - do not open before sitting",
        "pack": "audit-stage severity validation",
        "purpose": ("Blinded physician pass over the AUDIT STAGE's own severity grades - the "
                    "grades the pipeline's per-fact decision rule fires on. The earlier "
                    "clinician sitting validated the benchmark's grades, not these."),
        "build": {
            "builder": "build_severity_pack.py",
            "seed": SEED,
            "n_items": len(items),
            "target_rule_fired": N_FIRED,
            "random_quota": {g: n for g, n in RANDOM_QUOTA},
            "max_items_per_consultation": MAX_PER_CONSULT,
            "forced_item": "|".join(FORCED_FIRED) + " (the single clean-note false alarm)",
        },
        "sources": {
            "confirmation_run": "results/w2-pipeline/_state/confirm-B3.jsonl",
            "confirmation_run_sha256": hashlib.sha256(B3_PATH.read_bytes()).hexdigest(),
            "confirmation_records": len(records),
            "confirmation_consultations": len(consults),
            "audit_stage_records": "results/w2-pipeline/_cache/audit_<stratum>__<id>.json",
            "rubric": "master/severity-rubric.md",
            "rubric_sha256": hashlib.sha256(RUBRIC_PATH.read_bytes()).hexdigest(),
        },
        "universe": {
            "audit_graded_facts_over_47_consultations": sum(sev_universe.values()),
            "by_severity": dict(sorted(sev_universe.items())),
            "distinct_rule_fired_facts": len(fired),
            "consultations_with_a_rule_fired_fact": len({(s, c) for s, c, _ in fired}),
        },
        "composition": {
            "rule_fired": sum(1 for i in items if i["draw"] == "rule-fired"),
            **{"random_" + g: sum(1 for i in items if i["draw"] == "random-" + g)
               for g, _ in RANDOM_QUOTA},
            "distinct_consultations": len({(i["stratum"], i["consultation"]) for i in items}),
            "max_items_from_one_consultation": max(used.values()),
            "machine_grades_in_pack": dict(sorted(Counter(i["machine_severity"]
                                                          for i in items).items())),
        },
        "shortfalls": notes or ["none - every quota was filled from the stores"],
        "order_adjustments": order_notes or ["none"],
        "answer_scheme": [o["c"] for o in OPTS],
        "items": key_items,
    }


# ==========================================================================
# self-checks
# ==========================================================================

def self_check(doc, keys, items, audits, transcripts, records):
    """Every check returns (name, ok, detail). Run on every build."""
    out = []

    # 1. fact text verbatim from the audit-stage record, and in the html
    bad = []
    for it in items:
        a = audits[(it["stratum"], it["consultation"])]
        src = next((f for f in a["facts"] if f["id"] == it["fact_id"]), None)
        if src is None or src["fact"] != it["fact"] or src["severity"] != it["machine_severity"]:
            bad.append(it["id"] + " not matched in its audit record")
        elif esc(it["fact"]) not in doc:
            bad.append(it["id"] + " fact text missing from the html")
    out.append(("fact text verbatim in audit record and in the page", not bad, bad))

    # 2. the transcript shown is the fact's own consultation, and is the one the
    #    pipeline judged (each B3 record carries its own transcript_chars)
    chars = {(r["stratum"], r["consultation"]): r["transcript_chars"] for r in records}
    bad = []
    for it in items:
        c = (it["stratum"], it["consultation"])
        t = transcripts[c]
        if len(t) != chars[c]:
            bad.append("%s transcript length %d != run's %d" % (it["id"], len(t), chars[c]))
        block = _tx_block(doc, it["id"])
        if block is None:
            bad.append(it["id"] + " has no transcript block")
            continue
        for line in [l.strip() for l in t.split("\n") if l.strip()]:
            payload = line
            m = SPEAKER_RE.match(line) or SPEAKER_RE2.match(line)
            if m and m.group(2):
                payload = m.group(2)
            if esc(payload) not in block:
                bad.append("%s transcript line missing: %r" % (it["id"], line[:60]))
                break
    out.append(("transcript matches the fact's consultation, verbatim and complete",
                not bad, bad))

    # 3. no key material in the page
    leaks = []
    for it in items:
        for probe, what in ((it["consultation"], "consultation id"),
                            (it["machine_why"], "auditor's reason")):
            if probe and probe in doc:
                leaks.append("%s: %s leaked" % (it["id"], what))
        if re.search(r"\b%s\b" % re.escape(it["fact_id"]), doc):
            leaks.append("%s: fact id leaked" % it["id"])
    for word in ("missing_facts", "note_key", "verdict", "absent",
                 "rule fired", "rule-fired", "false alarm", "machine_severity", "D2N"):
        if word in doc:
            leaks.append("page contains %r" % word)
    # The three grade words are unavoidable - they are the answer buttons. What
    # would leak is a grade word that varies WITH the item. So strip each card's
    # two item-specific regions (the fact and the transcript, both source text
    # that may legitimately contain the words) and require the remaining
    # furniture to mention each grade exactly the same number of times on every
    # card. Any per-item grade, in text or in an attribute, breaks that.
    shapes = {}
    for it in items:
        card = _card_block(doc, it["id"])
        if card is None:
            leaks.append("%s: card not found" % it["id"])
            continue
        stripped = card.replace(_tx_block(doc, it["id"]) or "", "")
        stripped = re.sub(r"<blockquote class=\"fact\">.*?</blockquote>", "", stripped, flags=re.S)
        shape = tuple(len(re.findall(g, stripped, re.I))
                      for g in ("critical", "supporting", "peripheral"))
        shapes.setdefault(shape, []).append(it["id"])
    if len(shapes) > 1:
        leaks.append("cards differ in how they mention the grades: %s"
                     % {str(k): v for k, v in shapes.items()})
    out.append(("no key material in the page (grade wording identical on all %d cards: %s)"
                % (len(items), list(shapes)[0] if len(shapes) == 1 else "varies"),
                not leaks, leaks))

    # 4. blindness of the ordering: the draw strata must not run in blocks
    seq = [i["draw"] for i in items]
    runs = 1 + sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    out.append(("item order does not group the strata (%d runs over %d items)" % (runs, len(seq)),
                runs >= len(seq) * 0.5, seq))

    # 5. the pack is what it claims
    comp = keys["composition"]
    ok = (len(items) == N_ITEMS
          and comp["max_items_from_one_consultation"] <= MAX_PER_CONSULT
          and len({i["id"] for i in items}) == len(items))
    out.append(("24 unique items, <=%d per consultation" % MAX_PER_CONSULT, ok, comp))

    # 6. the forced false-alarm fact is in
    forced_in = any((i["stratum"], i["consultation"], i["fact_id"]) == FORCED_FIRED for i in items)
    out.append(("the clean-note false alarm is in the pack", forced_in, "|".join(FORCED_FIRED)))
    return out


def _card_block(doc, item_id):
    a = doc.find('<article class="card" id="card-%s"' % item_id)
    if a < 0:
        return None
    b = doc.find("</article>", a)
    return doc[a:b] if b >= 0 else None


def _tx_block(doc, item_id):
    card = _card_block(doc, item_id)
    if card is None:
        return None
    a = card.find('<div class="brief-body tx">')
    b = card.find("</details>", a)
    return card[a:b] if a >= 0 and b >= 0 else None


# ==========================================================================
# main
# ==========================================================================

def main():
    check_only = "--check" in sys.argv
    for p in (B3_PATH, RUBRIC_PATH, AUDIT_DIR):
        if not p.exists():
            sys.exit("source not found: %s" % p)

    records, consults, fired = load_b3()
    audits = load_audits(consults)
    transcripts, tsrc = transcript_index()
    missing = [c for c in consults if c not in transcripts]
    if missing:
        sys.exit("no transcript for: %s" % missing)

    # the stores must agree with themselves before anything is sampled
    for r in records:
        a = audits[(r["stratum"], r["consultation"])]
        byid = {f["id"]: f for f in a["facts"]}
        for mf in r["detail"]["missing_facts"]:
            f = byid.get(mf["id"])
            if not f or f["fact"] != mf["fact"] or f["severity"] != mf["severity"]:
                sys.exit("store disagreement on %s/%s %s" % (r["stratum"], r["consultation"],
                                                             mf["id"]))

    items, notes, order_notes, used = sample_items(fired, audits)
    rubric_text = RUBRIC_PATH.read_text(encoding="utf-8")
    doc, meta = build(items, transcripts, tsrc, rubric_text)
    keys = build_keys(items, fired, audits, transcripts, tsrc, notes, order_notes, used,
                      records, consults)

    checks = self_check(doc, keys, items, audits, transcripts, records)

    print("audit-stage severity pack")
    print("  seed %d   pack_sha %s   %d items" % (SEED, meta["pack_sha"], len(items)))
    print("  universe: %d audit-graded facts over %d consultations (%s)"
          % (keys["universe"]["audit_graded_facts_over_47_consultations"], len(consults),
             ", ".join("%s %d" % kv for kv in keys["universe"]["by_severity"].items())))
    print("  rule-fired facts in the store: %d over %d consultations"
          % (keys["universe"]["distinct_rule_fired_facts"],
             keys["universe"]["consultations_with_a_rule_fired_fact"]))
    print("  composition: " + ", ".join("%s %s" % (k, v) for k, v in keys["composition"].items()
                                        if isinstance(v, int)))
    print("  machine grades in pack: %s" % keys["composition"]["machine_grades_in_pack"])
    for n in notes:
        print("  SHORTFALL: %s" % n)
    for n in order_notes:
        print("  ORDER: %s" % n)
    print()
    for name, ok, detail in checks:
        print("  [%s] %s" % ("ok" if ok else "FAIL", name))
        if not ok:
            for d in (detail if isinstance(detail, list) else [detail])[:8]:
                print("        %s" % (d,))
    if not all(ok for _, ok, _ in checks):
        sys.exit("self-check failed - nothing written")

    if check_only:
        print("\n--check: nothing written")
        return
    OUT_HTML.write_text(doc, encoding="utf-8")
    OUT_KEYS.write_text(json.dumps(keys, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n  wrote %s (%.0f KB)" % (OUT_HTML.name, OUT_HTML.stat().st_size / 1024))
    print("  wrote %s" % OUT_KEYS.name)


# ==========================================================================
# the page: css + markup + js, no external anything
# ==========================================================================

HTML_SHELL = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Severity validation - clinician sitting</title>
<style>
:root{
  --bg:#f4f6f7; --panel:#ffffff; --panel2:#fafbfc;
  --ink:#171a1f; --ink2:#3c444f; --muted:#6b7480; --line:#e3e7eb; --line2:#eef1f4;
  --quote:#f0f5f4; --code:#f4f6f8; --code-ink:#1f242b;
  --sel:#0f766e; --sec-d:#5eead4; --ok:#15803d;
  --shadow:0 1px 2px rgba(16,24,40,.05), 0 8px 24px rgba(16,24,40,.06);
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#101317; --panel:#171b21; --panel2:#1c2128;
    --ink:#e8ecf1; --ink2:#c3cbd5; --muted:#8f9aa7; --line:#2a313a; --line2:#222831;
    --quote:#16211f; --code:#11151a; --code-ink:#d7dee7; --ok:#4ade80;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.35);
  }
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font:18px/1.62 -apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  text-rendering:optimizeLegibility;
}
header{
  position:sticky; top:0; z-index:20; background:color-mix(in srgb, var(--panel) 92%, transparent);
  backdrop-filter:saturate(1.4) blur(10px); border-bottom:1px solid var(--line);
}
.hrow{max-width:1080px; margin:0 auto; padding:10px 20px; display:flex; align-items:center; gap:14px; flex-wrap:wrap}
.brand{font-size:15px; font-weight:640; color:var(--ink); white-space:nowrap}
.brand small{display:block; font-weight:450; color:var(--muted); font-size:12.5px}
.spacer{flex:1 1 auto}
.count{font-size:14.5px; color:var(--muted); font-variant-numeric:tabular-nums; white-space:nowrap}
.count b{color:var(--ink); font-weight:640}
.hbtn{
  font:inherit; font-size:14.5px; padding:7px 13px; border-radius:9px; cursor:pointer;
  background:var(--panel2); color:var(--ink2); border:1px solid var(--line); white-space:nowrap;
}
.hbtn:hover{border-color:var(--muted); color:var(--ink)}
.hbtn.primary{background:var(--sel); border-color:var(--sel); color:#fff; font-weight:600}
.hbtn.ready{box-shadow:0 0 0 3px color-mix(in srgb, var(--sel) 25%, transparent)}
.saved{font-size:13px; color:var(--ok); opacity:0; transition:opacity .25s; white-space:nowrap}
.saved.on{opacity:1}
.bar{height:3px; background:var(--line2)}
.bar > i{display:block; height:100%; width:0; background:var(--sel); transition:width .3s ease}
main{max-width:860px; margin:0 auto; padding:30px 20px 120px}
.wide{max-width:1000px}
h1{font-size:33px; line-height:1.22; letter-spacing:-.4px; margin:.1em 0 .35em; font-weight:660}
h3{font-size:22px; margin:1.7em 0 .5em; font-weight:660}
h4{font-size:17.5px; margin:1.5em 0 .4em; font-weight:660}
h5{font-size:16px; margin:1.3em 0 .4em; font-weight:660; color:var(--ink2)}
p{margin:0 0 1.05em}
ul,ol{margin:0 0 1.05em; padding-left:1.35em}
li{margin:.32em 0}
strong{font-weight:660}
code{
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.86em;
  background:var(--code); border:1px solid var(--line); border-radius:5px; padding:1px 5px; color:var(--code-ink);
}
pre.raw{
  background:var(--code); border:1px solid var(--line); border-radius:10px;
  padding:15px 17px; overflow-x:auto; font-size:13.5px; line-height:1.5; white-space:pre-wrap;
  color:var(--code-ink); margin:0;
}
.card{
  background:var(--panel); border:1px solid var(--line); border-radius:16px;
  padding:26px 30px 24px; box-shadow:var(--shadow); border-top:3px solid var(--sel);
}
.crumb{display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; margin-bottom:16px}
.pill{font-size:12.5px; font-weight:650; color:#fff; background:var(--sel); padding:3px 10px; border-radius:999px}
.cid{font-size:19px; font-weight:700; letter-spacing:.3px}
.cpos{font-size:13px; color:var(--muted); margin-left:auto; font-variant-numeric:tabular-nums}
@media (prefers-color-scheme: dark){ .pill{color:#0b0e12; background:var(--sec-d)} }
p.lbl{margin-bottom:.45em; color:var(--muted); font-size:14.5px}
blockquote.fact{
  margin:0 0 18px; padding:18px 22px; background:var(--quote);
  border-left:4px solid var(--sel); border-radius:0 10px 10px 0;
  font-size:20px; line-height:1.5; color:var(--ink);
}
p.again{font-size:14.5px; color:var(--muted); margin:-6px 0 16px}
.brief{margin:0 0 20px; border:1px solid var(--line); border-radius:11px; background:var(--panel2)}
.brief > summary{
  cursor:pointer; padding:11px 15px; font-size:14.5px; font-weight:600; color:var(--ink2);
  list-style:none; display:flex; align-items:center; gap:8px;
}
.brief > summary::-webkit-details-marker{display:none}
.brief > summary::before{content:"\25b8"; color:var(--muted); font-size:12px; transition:transform .15s}
.brief[open] > summary::before{transform:rotate(90deg)}
.brief[open] > summary{border-bottom:1px solid var(--line)}
.brief-body{padding:16px 20px 6px; color:var(--ink2); line-height:1.58}
.brief-body.tx{max-height:none}
.tx p.turn{margin:0 0 .62em; font-size:16.5px; line-height:1.56}
.tx p.turn.cont{color:var(--ink2)}
.tx .who{
  display:inline-block; min-width:74px; font-size:12px; letter-spacing:.06em; text-transform:uppercase;
  font-weight:700; color:var(--sel); vertical-align:baseline; padding-right:8px;
}
@media (prefers-color-scheme: dark){ .tx .who{color:var(--sec-d)} }
p.q{font-size:17px; color:var(--ink2); margin:0 0 18px}
.answer{display:flex; gap:11px; flex-wrap:wrap; margin:6px 0; padding-top:22px; border-top:1px dashed var(--line)}
.opt{
  flex:1 1 150px; min-height:62px; font:inherit; font-size:17.5px; font-weight:600;
  border:1.5px solid var(--line); background:var(--panel2); color:var(--ink);
  border-radius:12px; cursor:pointer; padding:10px 14px;
  display:flex; align-items:center; justify-content:center; gap:10px;
  transition:border-color .12s, background .12s, transform .06s;
}
.opt:hover{border-color:var(--sel); background:color-mix(in srgb, var(--sel) 7%, var(--panel2))}
.opt:active{transform:translateY(1px)}
.opt .kbd{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px; font-weight:600;
  border:1px solid var(--line); border-radius:5px; padding:1px 6px; color:var(--muted); background:var(--panel);
}
.opt.on{background:var(--sel); border-color:var(--sel); color:#fff}
.opt.on .kbd{background:rgba(255,255,255,.18); border-color:rgba(255,255,255,.32); color:#fff}
@media (prefers-color-scheme: dark){
  .opt.on{background:var(--sec-d); border-color:var(--sec-d); color:#0b0e12}
  .opt.on .kbd{background:rgba(0,0,0,.18); border-color:rgba(0,0,0,.28); color:#0b0e12}
}
textarea{
  width:100%; font:inherit; font-size:17px; line-height:1.55; color:var(--ink);
  background:var(--panel2); border:1.5px solid var(--line); border-radius:11px; padding:12px 14px; resize:vertical;
}
textarea:focus{outline:none; border-color:var(--sel)}
.notewrap{margin-top:14px}
.addnote{
  font:inherit; font-size:14.5px; color:var(--muted); background:none; border:0; padding:4px 0;
  cursor:pointer; text-decoration:underline; text-underline-offset:3px; text-decoration-color:var(--line);
}
.addnote:hover, .addnote.has{color:var(--sel)}
.note{margin-top:9px; font-size:16.5px}
.itemnav{display:flex; gap:10px; align-items:center; margin:20px 2px 0}
.nbtn{
  font:inherit; font-size:15px; padding:9px 15px; border-radius:10px; cursor:pointer;
  background:var(--panel); border:1px solid var(--line); color:var(--ink2);
}
.nbtn:hover{border-color:var(--muted); color:var(--ink)}
.nbtn[disabled]{opacity:.4; cursor:default}
.navhint{margin-left:auto; font-size:13px; color:var(--muted)}
.row{
  display:flex; gap:13px; align-items:center; width:100%; text-align:left;
  font:inherit; font-size:16.5px; background:var(--panel); border:1px solid var(--line);
  border-left:3px solid var(--sel); border-radius:10px; padding:11px 14px; margin-bottom:7px; cursor:pointer; color:var(--ink);
}
.row:hover{border-color:var(--muted)}
.row .rid{font-weight:700; min-width:44px; font-variant-numeric:tabular-nums}
.row .rt{flex:1 1 auto; color:var(--ink2); font-size:15.5px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
.row .ra{font-size:13.5px; font-weight:640; padding:3px 10px; border-radius:999px; background:var(--sel); color:#fff; white-space:nowrap}
.row .ra.none{background:var(--line2); color:var(--muted); font-weight:500}
.row .rn{font-size:13px; color:var(--muted)}
@media (prefers-color-scheme: dark){ .row .ra{background:var(--sec-d); color:#0b0e12} }
.callout{
  background:var(--panel); border:1px solid var(--line); border-left:4px solid var(--sel);
  border-radius:0 12px 12px 0; padding:16px 20px; margin:0 0 22px; font-size:17px; color:var(--ink2);
}
.callout b{color:var(--ink)}
.resume{border-left-color:var(--ok); background:color-mix(in srgb, var(--ok) 8%, var(--panel))}
.startrow{display:flex; gap:12px; flex-wrap:wrap; margin:26px 0 10px}
.big{
  font:inherit; font-size:18px; font-weight:640; padding:14px 26px; border-radius:12px; cursor:pointer;
  background:var(--sel); color:#fff; border:1px solid var(--sel);
}
.big.ghost{background:var(--panel); color:var(--ink2); border-color:var(--line)}
.big.ghost:hover{border-color:var(--muted); color:var(--ink)}
.foot{margin-top:34px; font-size:14.5px; color:var(--muted)}
hr{border:0; border-top:1px solid var(--line); margin:1.6em 0}
.toast{
  position:fixed; left:50%; bottom:28px; transform:translateX(-50%) translateY(12px);
  background:var(--ink); color:var(--bg); font-size:15px; font-weight:560; padding:10px 18px; border-radius:999px;
  opacity:0; pointer-events:none; transition:opacity .2s, transform .2s; z-index:60;
}
.toast.on{opacity:.96; transform:translateX(-50%) translateY(0)}
.sheet{position:fixed; inset:0; background:rgba(10,12,15,.45); display:none; z-index:70; align-items:center; justify-content:center; padding:24px}
.sheet.on{display:flex}
.sheetbox{
  background:var(--panel); border:1px solid var(--line); border-radius:16px; max-width:560px; width:100%;
  padding:24px 26px; box-shadow:var(--shadow); max-height:84vh; overflow:auto;
}
.sheetbox h3{margin:0 0 14px; font-size:19px}
.keys{width:100%; font-size:16px; border-collapse:collapse}
.keys td{padding:6px 8px; border-bottom:1px solid var(--line2)}
.keys td:first-child{width:130px}
kbd{
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13px; border:1px solid var(--line);
  border-bottom-width:2px; border-radius:6px; padding:2px 7px; background:var(--panel2); color:var(--ink2);
}
.danger{color:#be123c; background:none; border:0; font:inherit; font-size:14.5px; cursor:pointer; text-decoration:underline; padding:0}
.hide{display:none !important}
@media (max-width:640px){
  body{font-size:17px}
  main{padding:20px 13px 100px}
  .card{padding:20px 17px; border-radius:13px}
  .opt{flex:1 1 100%}
  .cpos{margin-left:0}
  .tx .who{display:block; min-width:0}
}
</style>
</head>
<body>

<header>
  <div class="hrow">
    <div class="brand">Severity validation<small>audit-stage grades &nbsp;&middot;&nbsp; __TOTAL__ items &nbsp;&middot;&nbsp; ~20-25 min</small></div>
    <span class="spacer"></span>
    <span class="count"><b id="c-ans">0</b>/__TOTAL__ answered</span>
    <span class="saved" id="saved">saved &#10003;</span>
    <button class="hbtn" id="b-view" type="button">List view</button>
    <button class="hbtn" id="b-keys" type="button">Keys</button>
    <button class="hbtn primary" id="b-export" type="button">Export JSON</button>
  </div>
  <div class="bar"><i id="bar"></i></div>
</header>

<main>
  <section id="v-intro" class="wide">
    <div id="resume" class="callout resume hide"></div>
    <h1>Grading the machine's own severity grades</h1>
    <p><b>__TOTAL__ items, about 20-25 minutes.</b> Each item is one fact drawn from a
      consultation in the confirmation run, with that consultation's full transcript. Grade how
      much it would matter if the note got that fact wrong. That is the whole task.</p>
    <p>This is blind on purpose. You are not shown the machine's grade for the fact, the note
      that was written, whether anything was found missing, or why the fact was picked. Several
      items come from the same consultation, and the transcript is repeated in full each time
      so you never have to grade from memory.</p>
    <div class="callout">
      <b>What this is validating.</b> The last sitting graded the severity labels on the
      benchmark's own trap items. These are different labels: the ones the pipeline's audit
      stage writes for itself, before it ever sees a note. A decision rule reads those labels
      directly, so their agreement with a physician is load-bearing in a way the benchmark's
      labels are not.
    </div>
    <h3>The rubric</h3>
    <p class="foot" style="margin:-.5em 0 1.2em">Reproduced exactly as the study holds it. This is
      the same rubric the machine was given.</p>
    __RUBRIC__
    <details class="brief"><summary>The rubric file, byte for byte</summary>
      <div class="brief-body"><pre class="raw">__RUBRIC_RAW__</pre></div></details>
    <h3>Answering</h3>
    <p>Four buttons: <b>critical</b>, <b>supporting</b>, <b>peripheral</b>, and
      <b>can't judge</b>. Use <b>can't judge</b> when the transcript genuinely does not let you
      decide - it is a real answer, not a failure, and it is analysed separately from a skip.
      The comment box is optional and worth more than agonising over a borderline grade: if a
      fact is garbled, or states something the consultation did not actually contain, say so
      there.</p>
    <p>The transcript sits folded under each fact - click it, or press <kbd>t</kbd>, and the whole
      consultation is there. Keys: <kbd>c</kbd> <kbd>s</kbd> <kbd>p</kbd> <kbd>u</kbd> to answer,
      <kbd>1</kbd>-<kbd>4</kbd> the same, <kbd>t</kbd> for the transcript, <kbd>m</kbd> for a
      comment, <kbd>&larr;</kbd> <kbd>&rarr;</kbd> to move.</p>
    <div class="callout">
      <b>When you are done, hit Export JSON.</b> It downloads
      <code>severity_answers.json</code> - move that file into this same folder,
      the folder this pack came from, and return it to the study author
      from there. Answers save to this browser as you go, so you can close the tab and come
      back.
    </div>
    <div class="startrow">
      <button class="big" id="b-start" type="button">Start</button>
      <button class="big ghost" id="b-list2" type="button">See all items</button>
    </div>
    <p class="foot">Nothing leaves this machine: no network calls, no analytics. Answers live in
      this browser's local storage until you export them.</p>
  </section>

  <section id="v-focus" class="hide">
    <div id="cards">__CARDS__</div>
    <div class="itemnav">
      <button class="nbtn" id="b-prev" type="button">&larr; Previous</button>
      <button class="nbtn" id="b-next" type="button">Next &rarr;</button>
      <span class="navhint" id="navhint"></span>
    </div>
  </section>

  <section id="v-list" class="hide wide">
    <h1 style="font-size:26px">All items</h1>
    <p class="foot" style="margin:0 0 24px">Click any row to jump to it.</p>
    <div id="listbody"></div>
  </section>

  <section id="v-done" class="hide">
    <h1>That is the sitting.</h1>
    <div id="donesum"></div>
    <div class="callout">
      <b>Last step.</b> Hit <b>Export JSON</b> and move <code>severity_answers.json</code> into
      the folder this pack came from. Or copy it to the clipboard and send it to the study author.
    </div>
    <div class="startrow">
      <button class="big" id="b-export2" type="button">Export severity_answers.json</button>
      <button class="big ghost" id="b-copy" type="button">Copy JSON</button>
      <button class="big ghost" id="b-back" type="button">Back to items</button>
    </div>
  </section>
</main>

<div class="toast" id="toast"></div>

<div class="sheet" id="keysheet">
  <div class="sheetbox">
    <h3>Keyboard</h3>
    <table class="keys"><tbody>
      <tr><td><kbd>c</kbd> <kbd>s</kbd> <kbd>p</kbd> <kbd>u</kbd></td><td>critical / supporting / peripheral / can't judge</td></tr>
      <tr><td><kbd>1</kbd>&hellip;<kbd>4</kbd></td><td>the same four, in order</td></tr>
      <tr><td><kbd>&rarr;</kbd> <kbd>j</kbd></td><td>next item</td></tr>
      <tr><td><kbd>&larr;</kbd></td><td>previous item</td></tr>
      <tr><td><kbd>t</kbd></td><td>open / close the transcript</td></tr>
      <tr><td><kbd>m</kbd></td><td>add a comment</td></tr>
      <tr><td><kbd>l</kbd></td><td>list view / back</td></tr>
      <tr><td><kbd>e</kbd></td><td>export JSON</td></tr>
      <tr><td><kbd>esc</kbd></td><td>leave a text box / close this</td></tr>
    </tbody></table>
    <p class="foot">Answering jumps you to the next unanswered item. Skipping is fine - a blank
      is recorded as a skip, which is not the same as <b>can't judge</b>.</p>
    <div class="startrow" style="margin-bottom:0">
      <button class="big ghost" id="b-closekeys" type="button">Close</button>
      <button class="danger" id="b-reset" type="button">Clear all saved answers</button>
    </div>
  </div>
</div>

<script id="pack-meta" type="application/json">__META__</script>
<script>
(function(){
"use strict";
var META = JSON.parse(document.getElementById('pack-meta').textContent);
var ITEMS = META.items, N = ITEMS.length;
// Keyed to the pack's CONTENT hash (ids, tags, fact text, transcript hashes,
// answer codes). If the pack is ever re-cut, S07 may mean a different fact, and
// a content-keyed store guarantees old answers cannot silently re-attach to it.
var KEYNAME = 'scribe-severity-' + META.pack_sha;
var IDX = {}; ITEMS.forEach(function(it,i){ IDX[it.id] = i; });
var CODES = ITEMS.length ? ITEMS[0].opts.map(function(o){ return o.c; }) : [];
var OPTKEYS = ITEMS.length ? ITEMS[0].opts.map(function(o){ return o.k; }) : [];

var state = { v:1, answers:{}, current:0, started:null, updated:null };
var view = 'intro';
var $ = function(id){ return document.getElementById(id); };
var cards = Array.prototype.slice.call(document.querySelectorAll('.card'));

/* ---------- storage ---------- */
function load(){
  try{
    var raw = localStorage.getItem(KEYNAME);
    if(!raw) return false;
    var s = JSON.parse(raw);
    if(s && s.answers){
      state.answers = s.answers;
      state.current = (typeof s.current === 'number' && s.current >= 0 && s.current < N) ? s.current : 0;
      state.started = s.started || null;
      state.updated = s.updated || null;
      return true;
    }
  }catch(e){}
  return false;
}
function persist(){
  state.updated = new Date().toISOString();
  if(!state.started) state.started = state.updated;
  try{ localStorage.setItem(KEYNAME, JSON.stringify(state)); }catch(e){}
}
function rec(id){
  if(!state.answers[id]) state.answers[id] = { answer:null, comment:'', answered_at:null };
  return state.answers[id];
}
function answered(id){
  var r = state.answers[id];
  return !!(r && r.answer !== null && r.answer !== undefined && String(r.answer).trim() !== '');
}
function nAnswered(){ var k=0; ITEMS.forEach(function(it){ if(answered(it.id)) k++; }); return k; }

/* ---------- progress ---------- */
function paint(){
  var a = nAnswered();
  $('c-ans').textContent = a;
  $('bar').style.width = (100 * a / N) + '%';
  var ex = $('b-export');
  ex.classList.toggle('ready', a >= N);
  ex.textContent = (a >= N) ? 'Export JSON ✓' : 'Export JSON';
}

/* ---------- views ---------- */
function setView(v){
  view = v;
  $('v-intro').classList.toggle('hide', v !== 'intro');
  $('v-focus').classList.toggle('hide', v !== 'focus');
  $('v-list').classList.toggle('hide', v !== 'list');
  $('v-done').classList.toggle('hide', v !== 'done');
  $('b-view').textContent = (v === 'list') ? 'Back to item' : 'List view';
  if(v === 'list') buildList();
  if(v === 'done') buildDone();
  paint();
  window.scrollTo(0, 0);
}
function show(i, noscroll){
  if(i < 0) i = 0;
  if(i > N - 1) i = N - 1;
  state.current = i;
  var it = ITEMS[i];
  cards.forEach(function(c){ c.hidden = (c.dataset.id !== it.id); });
  syncCard(it.id);
  $('b-prev').disabled = (i === 0);
  $('b-next').disabled = (i === N - 1);
  $('navhint').innerHTML = 'keys: ' + OPTKEYS.map(function(k){ return '<kbd>' + k + '</kbd>'; }).join(' ') +
      ' &nbsp;&middot;&nbsp; <kbd>t</kbd> transcript &nbsp;&middot;&nbsp; <kbd>m</kbd> comment' +
      ' &nbsp;&middot;&nbsp; <kbd>&larr;</kbd> <kbd>&rarr;</kbd> move';
  setView('focus');
  if(!noscroll) window.scrollTo(0, 0);
  persist();
}
function nextUnanswered(from){
  for(var k = 1; k <= N; k++){
    var i = (from + k) % N;
    if(!answered(ITEMS[i].id)) return i;
  }
  return -1;
}
function syncCard(id){
  var card = $('card-' + id), r = state.answers[id] || {};
  if(!card) return;
  Array.prototype.forEach.call(card.querySelectorAll('.opt'), function(b){
    b.classList.toggle('on', r.answer === b.dataset.code);
  });
  var note = card.querySelector('.note'), btn = card.querySelector('.addnote');
  if(note){
    if(note.value !== (r.comment || '')) note.value = r.comment || '';
    var has = !!(r.comment && r.comment.trim());
    btn.classList.toggle('has', has);
    btn.textContent = has ? 'comment ▾' : '+ add comment';
    if(has) note.hidden = false;
  }
}
function setAnswer(id, code){
  var r = rec(id);
  if(r.answer === code){ r.answer = null; r.answered_at = null; }   /* click again to clear */
  else { r.answer = code; r.answered_at = new Date().toISOString(); }
  persist(); syncCard(id); paint(); flash('saved ✓');
  if(r.answer !== null){
    var nx = nextUnanswered(IDX[id]);
    setTimeout(function(){
      if(nAnswered() >= N){ setView('done'); }
      else if(nx >= 0 && view === 'focus'){ show(nx); }
    }, 180);
  }
}

/* ---------- list ---------- */
function buildList(){
  var out = '';
  ITEMS.forEach(function(it){
    var r = state.answers[it.id] || {};
    var a = (r.answer === null || r.answer === undefined) ? '' : String(r.answer);
    var o = it.opts.filter(function(x){ return x.c === a; })[0];
    var lab = o ? o.l : '—';
    out += '<button class="row" data-id="' + it.id + '"><span class="rid">' + it.id + '</span>' +
           '<span class="rt">' + esc(it.fact) + '</span>' +
           (r.comment && r.comment.trim() ? '<span class="rn">comment</span>' : '') +
           '<span class="ra' + (o ? '' : ' none') + '">' + esc(lab) + '</span></button>';
  });
  $('listbody').innerHTML = out;
}
function esc(s){
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/* ---------- done ---------- */
function buildDone(){
  var a = nAnswered();
  var out = '<p>' + a + ' of ' + N + ' answered' +
            (a < N ? ' - ' + (N - a) + ' skipped, which is recorded as a skip, not a guess.' : '.') + '</p>';
  var bits = ITEMS.map(function(it){
    var r = state.answers[it.id] || {};
    var v = (r.answer === null || r.answer === undefined || r.answer === '') ? '·' : String(r.answer).charAt(0);
    return it.id + ' ' + v;
  });
  out += '<p style="font-family:ui-monospace,Menlo,monospace;font-size:15px;color:var(--ink2)">' +
         esc(bits.join('   ')) + '</p>';
  $('donesum').innerHTML = out;
}

/* ---------- export ---------- */
function payload(){
  return {
    schema_version: 1,
    pack: META.pack,
    pack_sha: META.pack_sha,
    exported_at: new Date().toISOString(),
    items: ITEMS.map(function(it){
      var r = state.answers[it.id] || {};
      var a = (r.answer === null || r.answer === undefined || String(r.answer).trim() === '') ? null : r.answer;
      return { id: it.id, answer: a, comment: (r.comment || '').trim(),
               answered_at: a === null ? null : (r.answered_at || null) };
    })
  };
}
function jsonText(){ return JSON.stringify(payload(), null, 2); }
function download(){
  var blob = new Blob([jsonText()], {type:'application/json'});
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url; a.download = 'severity_answers.json';
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  setTimeout(function(){ URL.revokeObjectURL(url); }, 1500);
  flash('severity_answers.json downloaded - move it into the folder this pack came from');
}
function copyJSON(){
  var t = jsonText();
  var done = function(){ flash('JSON copied to clipboard'); };
  if(navigator.clipboard && navigator.clipboard.writeText){
    navigator.clipboard.writeText(t).then(done, function(){ fallbackCopy(t, done); });
  } else { fallbackCopy(t, done); }
}
function fallbackCopy(t, done){
  var ta = document.createElement('textarea');
  ta.value = t; ta.style.position = 'fixed'; ta.style.opacity = '0';
  document.body.appendChild(ta); ta.select();
  var ok = false;
  try{ ok = document.execCommand('copy'); }catch(e){}
  document.body.removeChild(ta);
  if(ok) done(); else showRaw(t);
}
function showRaw(t){
  var box = document.createElement('div');
  box.className = 'sheet on';
  box.innerHTML = '<div class="sheetbox"><h3>Copy this, or use Export JSON</h3>' +
    '<textarea rows="14" readonly></textarea><div class="startrow" style="margin-bottom:0">' +
    '<button class="big ghost" type="button">Close</button></div></div>';
  box.querySelector('textarea').value = t;
  box.querySelector('button').onclick = function(){ document.body.removeChild(box); };
  document.body.appendChild(box);
  box.querySelector('textarea').select();
}

/* ---------- toast ---------- */
var toastT = null;
function flash(msg){
  var el = $('toast');
  el.textContent = msg; el.classList.add('on');
  var s = $('saved'); s.classList.add('on');
  clearTimeout(toastT);
  toastT = setTimeout(function(){ el.classList.remove('on'); s.classList.remove('on'); }, 1600);
}

/* ---------- events ---------- */
$('cards').addEventListener('click', function(e){
  var card = e.target.closest ? e.target.closest('.card') : null;
  if(!card) return;
  var id = card.dataset.id;
  var opt = e.target.closest('.opt');
  if(opt){ setAnswer(id, opt.dataset.code); return; }
  var add = e.target.closest('.addnote');
  if(add){
    var note = card.querySelector('.note');
    note.hidden = !note.hidden;
    if(!note.hidden) note.focus();
  }
});
$('cards').addEventListener('input', function(e){
  var card = e.target.closest ? e.target.closest('.card') : null;
  if(!card) return;
  var r = rec(card.dataset.id);
  if(e.target.classList.contains('note')){
    r.comment = e.target.value;
    card.querySelector('.addnote').classList.toggle('has', !!r.comment.trim());
  }
  persist();
  clearTimeout(toastT); flash('saved ✓');
});
$('listbody').addEventListener('click', function(e){
  var row = e.target.closest ? e.target.closest('.row') : null;
  if(row) show(IDX[row.dataset.id]);
});
$('b-prev').onclick = function(){ show(state.current - 1); };
$('b-next').onclick = function(){ show(state.current + 1); };
$('b-view').onclick = function(){ setView(view === 'list' ? 'focus' : 'list'); };
$('b-list2').onclick = function(){ setView('list'); };
$('b-start').onclick = function(){ var u = answered(ITEMS[0].id) ? nextUnanswered(-1) : 0; show(u < 0 ? 0 : u); };
$('b-export').onclick = download;
$('b-export2').onclick = download;
$('b-copy').onclick = copyJSON;
$('b-back').onclick = function(){ show(state.current); };
$('b-keys').onclick = function(){ $('keysheet').classList.add('on'); };
$('b-closekeys').onclick = function(){ $('keysheet').classList.remove('on'); };
$('keysheet').addEventListener('click', function(e){ if(e.target === $('keysheet')) $('keysheet').classList.remove('on'); });
$('b-reset').onclick = function(){
  if(!confirm('Clear every saved answer and comment on this machine? This cannot be undone.')) return;
  try{ localStorage.removeItem(KEYNAME); }catch(e){}
  state = { v:1, answers:{}, current:0, started:null, updated:null };
  cards.forEach(function(c){ syncCard(c.dataset.id); });
  $('keysheet').classList.remove('on');
  $('resume').classList.add('hide');
  setView('intro');
};

document.addEventListener('keydown', function(e){
  var t = e.target;
  if(t && (t.tagName === 'TEXTAREA' || t.tagName === 'INPUT')){
    if(e.key === 'Escape') t.blur();
    return;
  }
  if(e.metaKey || e.ctrlKey || e.altKey) return;
  var k = e.key;
  if(k === 'Escape'){
    if($('keysheet').classList.contains('on')) $('keysheet').classList.remove('on');
    else if(view === 'list') setView('focus');
    return;
  }
  if(k === 'l' || k === 'L'){ setView(view === 'list' ? 'focus' : 'list'); e.preventDefault(); return; }
  if(k === 'e' || k === 'E'){ download(); e.preventDefault(); return; }
  if(k === 'h' || k === 'H' || k === '?'){ $('keysheet').classList.toggle('on'); e.preventDefault(); return; }
  if(view !== 'focus') return;
  var it = ITEMS[state.current];
  var card = $('card-' + it.id);
  if(k === 'ArrowRight' || k === 'j' || k === 'J'){ show(state.current + 1); e.preventDefault(); return; }
  if(k === 'ArrowLeft'){ show(state.current - 1); e.preventDefault(); return; }
  if(k === 't' || k === 'T'){
    var d = card.querySelector('details.brief');
    if(d){ d.open = !d.open; e.preventDefault(); }
    return;
  }
  if(k === 'm' || k === 'M'){
    var note = card.querySelector('.note');
    if(note){ note.hidden = false; note.focus(); e.preventDefault(); }
    return;
  }
  var ki = OPTKEYS.indexOf(k.toLowerCase());
  if(ki >= 0){ setAnswer(it.id, CODES[ki]); e.preventDefault(); return; }
  if(/^[1-9]$/.test(k)){
    var n = parseInt(k, 10) - 1;
    if(n < CODES.length){ setAnswer(it.id, CODES[n]); e.preventDefault(); }
  }
});

/* ---------- boot ---------- */
var had = load();
cards.forEach(function(c){ syncCard(c.dataset.id); });
var a0 = nAnswered();
if(had && a0 > 0){
  var r = $('resume');
  var cur = ITEMS[state.current] ? ITEMS[state.current].id : ITEMS[0].id;
  r.innerHTML = '<b>' + a0 + '/' + N + ' answered</b> - continue where you left off (' + cur + ').' +
    ' <button class="hbtn primary" id="b-resume" type="button" style="margin-left:10px">Continue</button>';
  r.classList.remove('hide');
  $('b-resume').onclick = function(){ show(state.current); };
  $('b-start').textContent = 'Start from the top';
}
setView('intro');
paint();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
