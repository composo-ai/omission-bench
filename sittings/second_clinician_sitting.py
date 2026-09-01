#!/usr/bin/env python3
"""second_clinician_sitting.py - builds the offline adjudication app for an independent clinician.

The census's 618 verified findings have been adjudicated by exactly one human, who is an
author, the designer of the instrument and the writer of the rubric he graded
against. That is the one item the TRIPOD-LLM checklist marks not met (7d), and it is
the first thing an external reader will challenge. The objection is about
INDEPENDENCE, not about agreement: no human who is not an author has assessed a random
sample of the 618.

So the clinician sees findings NOBODY has adjudicated. Two lots, each a self-contained
offline HTML file, each about half an hour. The first stands on its own; the second is
built and held.

Why fresh items rather than the author's. At a dozen shared items, with the first rater
having answered genuine on 29 of 30, Cohen's kappa is undefined, exactly zero, or negative
- uninformative by construction - and re-adjudicating done items adds no coverage. Fresh
items take the number of findings a human has looked at from 21 to 28, or to 35 across
both lots. TRIPOD-LLM 7d therefore stays not-met, with one honest line saying why.

The draw
  - N_VERIFIED verified findings drawn uniformly at random from the 618, and N_REFUSED
    from the high-salience refused stratum - the same stratum the existing pack used, so
    the two sittings' refusal results are comparable.
  - Every item in the existing pack is excluded, and the exclusion is asserted rather
    than trusted. The exclusion covers all 45 drawn items, not only the 30 adjudicated:
    the extra 15 cost 2.4% of the sampling frame and remove any chance of a collision
    with the first sitting's key if the remainder is ever sat.
  - Split into two lots, STRATIFIED so each lot carries the existing pack's mix (roughly
    seven verified to three refused). A plain shuffle-then-cut cannot guarantee the mix in
    each lot, and the mix is what keeps the task from degenerating into rubber-stamping:
    the first rater answered genuine on 29 of 30 items, so acquiescence is the live risk.
    Each lot is still a uniform random sample of each stratum, so lot one alone is a clean
    random sample and the analysis holds if lot two never goes out.
  - Order shuffled within each lot; no status label anywhere; seed, item ids and their
    hash recorded in the manifest.

The app
  - One self-contained HTML file per lot. Opens by double-click, works with no network,
    no server, no account. Everything inline: no CDN, no external font, no remote image.
  - Autosave to localStorage on every answer, wrapped in try/catch.
  - Nothing pre-selected; an item is not complete until the clinician actively chooses.
  - Two ways to return the answers: a downloaded JSON file and a copy-to-clipboard block
    of the same content. Neither carries note or transcript text.
  - Per-item active reading time, which is what tells us whether the size below is right.

Blinding is enforced over the emitted file by assert_blind(), not claimed in a docstring:
no vendor name, no finding id, no status vocabulary, and every item's HTML skeleton
byte-identical once its text content is stripped.

    python3 sittings/second_clinician_sitting.py
    python3 sittings/second_clinician_sitting.py --items 15 --verified 11   # the size originally proposed
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import argparse
import hashlib
import html
import json
import os
import random
import re
import statistics
import sys

from common import HERE, RESULTS
import taxonomy_common as T

SEED = 20260826
VERIFIED = "master/findings_verified_master.json"
RUBRIC = "master/severity-rubric.md"   # the written severity rubric, a study working file that is not in this release; the same text ships with the dataset
PRIOR_KEY = "results/precision-sitting/key.json"
PRIOR_PACK = "results/precision-sitting/release/adjudication-pack.md"
OUTDIR = os.path.join(RESULTS, "second-clinician-sitting")
TRANSCRIPT_CHARS = 40000        # what the panel saw, and what the first sitting showed
N_LOTS = 2

# Size. See size_note() - this is a measurement, not a guess, and the measurement says
# fifteen items is a forty-to-ninety-minute job rather than a thirty-minute one.
DEFAULT_ITEMS_PER_LOT = 8
DEFAULT_VERIFIED_PER_LOT = 6

SCRIBE_LETTER = {"scribe_A": "A", "scribe_B": "B", "scribe_C": "C"}   # the released convention

# ---------------------------------------------------------------- reading time

# Measured from the existing pack (results/precision-sitting/release/adjudication-pack.md,
# 30 items, read end to end): 56 words of claim and quotes, 320 of note, 1,652 of
# transcript, 2,028 in all, 12.0 KB per item.
#
# The reading volume is a measurement. Turning it into minutes is not, so the model below
# is behavioural and explicit rather than one global words-per-minute, and it produces a
# band. Argue with the components, not with the arithmetic.
#
#   1. read the claim and its two evidence quotes, carefully          ~110 words
#   2. read the whole note, as you would a colleague's                 320 words
#   3. read ~200 words of transcript around the highlighted quote,
#      then scan the remaining ~1,450 for anything that contradicts
#   4. decide, then grade against a three-line rubric
#
# The one empirical anchor for the whole task: the author's own sitting was budgeted at
# "1-2 hours" for 45 items and he completed 30, so the pace that actually happened was
# about four minutes an item - for the person who built the corpus, the taxonomy and the
# rubric. A cold clinician is not faster than that. Our band brackets it.
READ = {                       # (claim wpm, note wpm, local wpm, scan wpm, decide s)
    "fast":    (200, 280, 300, 1200, 30),
    "central": (160, 240, 250,  900, 45),
    "slow":    (130, 200, 200,  700, 60),
}
W_CLAIM, W_NOTE, W_LOCAL, W_SCAN = 110, 320, 200, 1452
TARGET_MINUTES = 30

# ------------------------------------------------------------------- quote find

_WS = re.compile(r"\s+")
_SPEAKER = re.compile(r"^(doctor|patient|clinician|dr)\s*:\s*", re.I)


def _norm_index(s):
    """Whitespace-collapsed lowercase text, plus a map from its offsets back to s."""
    out, idx, prev_ws = [], [], True
    for i, ch in enumerate(s):
        if ch.isspace():
            if prev_ws:
                continue
            out.append(" ")
            idx.append(i)
            prev_ws = True
        else:
            out.append(ch.lower())
            idx.append(i)
            prev_ws = False
    return "".join(out), idx


def _needles(quote):
    """Progressively shorter needles from an evidence quote, longest first."""
    quote = (quote or "").strip()
    if not quote or quote == "-":
        return []
    seen, out = set(), []

    def add(x):
        x = _WS.sub(" ", _SPEAKER.sub("", x.strip())).strip().lower()
        if len(x) >= 12 and x not in seen:
            seen.add(x)
            out.append(x)

    add(quote)
    for part in re.split(r"\.\.\.|…|\n", quote):
        add(part)
        words = part.split()
        for n in (12, 10, 8, 6, 5, 4):
            for i in range(0, max(0, len(words) - n + 1)):
                add(" ".join(words[i:i + n]))
    return out


def find_span(haystack, quote):
    """(start, end) of the evidence quote in haystack, or None."""
    norm, idx = _norm_index(haystack)
    for needle in _needles(quote):
        p = norm.find(needle)
        if p >= 0:
            return idx[p], idx[min(p + len(needle) - 1, len(idx) - 1)] + 1
    return None


def mark(text, span):
    """Escaped HTML with the span wrapped in <mark>, or plain escaped HTML."""
    if not span:
        return html.escape(text)
    a, b = span
    return "%s<mark>%s</mark>%s" % (
        html.escape(text[:a]), html.escape(text[a:b]), html.escape(text[b:]))


# ----------------------------------------------------------------------- sizing

def minutes_per_item(band="central"):
    claim, note, local, scan, decide = READ[band]
    return (W_CLAIM / claim + W_NOTE / note + W_LOCAL / local + W_SCAN / scan
            + decide / 60)


def size_note(items, verified_per_lot):
    """The arithmetic behind the item count, for the manifest and for the write-up."""
    band = {b: round(minutes_per_item(b), 2) for b in READ}
    lot = {b: round(items * m, 0) for b, m in band.items()}
    return {
        "measured_from": "results/precision-sitting/release/adjudication-pack.md, 30 items",
        "words_per_item": {"claim_and_quotes": 56, "note": 320, "transcript": 1652,
                           "total": 2028},
        "kb_per_item": 12.0,
        "model": "claim %dw, note %dw, transcript %dw local plus %dw scanned, then decide "
                 "and grade; rates in READ" % (W_CLAIM, W_NOTE, W_LOCAL, W_SCAN),
        "minutes_per_item": band,
        "items_per_lot": items,
        "verified_per_lot": verified_per_lot,
        "estimated_minutes_per_lot": lot,
        "target_minutes": TARGET_MINUTES,
        "anchor": "the author's own sitting: 45 items budgeted at 1-2 hours, 30 completed, "
                  "so about four minutes an item for the person who built the instrument",
        "finding": "Fifteen items inside thirty minutes was the size first proposed. On this "
                   "measurement fifteen items is %.0f to %.0f minutes, and thirty minutes "
                   "buys about %d items. The default lot is %d items at %.0f to %.0f "
                   "minutes - deliberately a little over the half hour, because at six "
                   "items the sitting stops being worth a doctor's goodwill. This is the "
                   "one place the build knowingly exceeds that proposal; --items resets it "
                   "either way."
                   % (15 * band["fast"], 15 * band["slow"],
                      int(TARGET_MINUTES / band["central"]),
                      items, lot["fast"], lot["slow"]),
        "measure_it_for_real": "the app records per-item active seconds, so lot one turns "
                               "this estimate into an observation and lot two can be "
                               "resized before it goes out",
    }


# -------------------------------------------------------------------- the build

def build_lot(lot, items, rubric, units, marks_ok, pack_sha):  # noqa: C901
    """One self-contained HTML file. Returns (html_text, key_rows)."""
    cards, key_rows = [], []
    for n, (f, _status) in enumerate(items, 1):
        u = units[f["note_key"]]
        note = u["note"].strip()
        transcript = u["transcript"][:TRANSCRIPT_CHARS].strip()
        note_html = mark(note, find_span(note, f.get("note_quote")) if marks_ok else None)
        tr_html = mark(transcript,
                       find_span(transcript, f.get("source_quote")) if marks_ok else None)
        cards.append(ITEM_TMPL.format(
            n=n,
            mode=html.escape(str(f.get("mode") or "-")),
            desc=html.escape(str(f.get("description") or "-")),
            nq=html.escape(str(f.get("note_quote") or "-")),
            tq=html.escape(str(f.get("source_quote") or "-")),
            letter=SCRIBE_LETTER[f["scribe"]],
            rubric=html.escape(rubric),
            note=note_html,
            transcript=tr_html))
        key_rows.append({
            "lot": lot, "item": n, "finding_id": f["finding_id"], "status": _status,
            "note_key": f["note_key"], "consultation": f["consultation"],
            "scribe": f["scribe"], "pass": f.get("pass"), "mode": f.get("mode"),
            "frame_tier1": f.get("frame_tier1"), "frame_tier2": f.get("frame_tier2"),
            "panel_salience": (f.get("verdict") or {}).get("salience"),
            "severity_rubric": f.get("severity_rubric"),
            "panel_decided_by": "tiebreak" if (f.get("verdict") or {}).get("split")
                                else "unanimous"})
    doc = PAGE_TMPL.format(
        lot=lot, n_items=len(items), rubric=html.escape(rubric),
        pack_sha=pack_sha, cards="\n".join(cards),
        minutes=int(round(len(items) * minutes_per_item("central"))),
        per_item=int(round(minutes_per_item("central"))))
    return doc, key_rows


# ------------------------------------------------------------------- assertions

# Checked over the CHROME only - everything except the note and transcript panes. Those
# two are verbatim source material written before this study existed, and they contain
# ordinary clinical English: a note says "corticosteroid", a transcript says "the patient
# refused". Banning those words across the whole file would be a false alarm, and banning
# them only in the chrome is the check that actually means something.
BANNED = ["verified", "refused", "foil", "survivor", "adjudicated by",
          "is_real", "ground truth", "key.json", "precision-sitting", "answers.txt",
          "second_clinician", "findings_verified_master", "panel"]

# Checked over the WHOLE file, on word boundaries, because a product name appearing
# anywhere - including inside a note the product wrote about itself - breaks the blind.
VENDORS = re.compile(r"\b(scribe_A|scribe_B|scribe_C)\b", re.I)

OFFLINE_BANNED = ["http://", "https://", "<script src", "<link ", "@import",
                  "fetch(", "XMLHttpRequest", "navigator.sendBeacon", "srcset",
                  "WebSocket", "import(", "//cdn."]

_TEXT = re.compile(r">[^<]*<")
_ITEM = re.compile(r'<section class="item"[^>]*>.*?</section>', re.S)


def assert_blind(doc, key_rows, prior_answers):
    """Everything the clinician must not be able to recover from the file itself."""
    low = doc.lower()
    chrome = re.sub(r'(<pre class="doc (?:note|tx)">).*?(</pre>)', r"\1\2", low, flags=re.S)
    for word in BANNED:
        if word in chrome:
            raise SystemExit("BLINDING: the emitted file's own text contains %r" % word)
    hit = VENDORS.search(doc)
    if hit:
        raise SystemExit("BLINDING: emitted file names a product (%r at %d)"
                         % (hit.group(0), hit.start()))
    # Per-item identifiers and status vocabulary. The rubric grade and the panel's
    # salience are NOT checked here - they are the ordinary words "critical",
    # "supporting", "peripheral", "high", every one of which must appear on every item as
    # a button label. That they cannot be read off an item is what the structural-identity
    # check below establishes, not a substring search.
    for row in key_rows:
        for leak in (row["finding_id"], row["note_key"], row["consultation"],
                     row["status"], row["panel_decided_by"]):
            if leak and leak.lower() in low:
                raise SystemExit("BLINDING: emitted file contains %r" % leak)
    if prior_answers and prior_answers.strip() and prior_answers.strip().lower() in low:
        raise SystemExit("BLINDING: emitted file contains the first rater's answers")

    # Structural identity: strip every item's text content and every item block must be
    # byte-identical to the first. Nothing about an item may vary except its content.
    blocks = _ITEM.findall(doc)
    if len(blocks) != len(key_rows):
        raise SystemExit("BLINDING: %d item blocks for %d items" % (len(blocks), len(key_rows)))
    def skeleton(b):
        b = re.sub(r'data-i="\d+"', 'data-i=""', b)
        b = re.sub(r"</?mark>", "", b)      # the highlight sits inside the text, not the shape
        return _TEXT.sub("><", b)

    skeletons = {skeleton(b) for b in blocks}
    if len(skeletons) != 1:
        raise SystemExit("BLINDING: item blocks are not structurally identical (%d shapes)"
                         % len(skeletons))

    for word in OFFLINE_BANNED:
        if word in low:
            raise SystemExit("OFFLINE: emitted file contains %r" % word)
    # Highlighting reveals nothing new only if its presence is a deterministic function of
    # the evidence quote already printed at the top of the item. Check that item by item:
    # a highlighted note exactly when the note-side quote is not "-", same for transcript.
    for b in blocks:
        quotes = re.findall(r"<span>(Note|Transcript)-side evidence quote</span>\s*([^<]*)<", b)
        panes = re.findall(r'<pre class="doc (note|tx)">(.*?)</pre>', b, re.S)
        if len(quotes) != 2 or len(panes) != 2:
            raise SystemExit("BLINDING: cannot parse an item block for the highlight check")
        for (_side, q), (_pane, body) in zip(quotes, panes):
            if (q.strip() not in ("", "-")) != ("<mark>" in body):
                raise SystemExit(
                    "BLINDING: highlight presence does not track the printed quote on "
                    "item %s - it would leak" % re.search(r'data-i="(\d+)"', b).group(1))


def assert_wording(doc, pack_text):
    """The two questions and the rubric are copied from the existing pack, not rewritten."""
    for line in [
        "Reading the note against the transcript, is this",
        "a genuine documentation failure a reviewing clinician should accept as real?",
        "a real failure; the note asserts, omits or distorts something the",
        "transcript does not support, in a way a reviewing clinician would accept as a defect",
        "not a real failure (the claim misreads the note or transcript, the",
        "content is actually captured, the inference is clinically justified, etc.)",
        "the materials are insufficient to decide",
        "the severity grade",
    ]:
        if line not in pack_text:
            raise SystemExit("WORDING: %r is not in the existing pack - re-copy it" % line[:40])
        if html.escape(line) not in doc and line not in doc:
            raise SystemExit("WORDING: %r is not in the emitted file" % line[:40])


# ------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=DEFAULT_ITEMS_PER_LOT,
                    help="items per lot (default %d - see size_note)" % DEFAULT_ITEMS_PER_LOT)
    ap.add_argument("--verified", type=int, default=DEFAULT_VERIFIED_PER_LOT,
                    help="of those, how many drawn from the 618 (default %d)"
                         % DEFAULT_VERIFIED_PER_LOT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if not 0 < args.verified < args.items:
        raise SystemExit("--verified must be between 1 and --items - 1")
    n_ver, n_ref = args.verified * N_LOTS, (args.items - args.verified) * N_LOTS

    d = json.load(open(os.path.join(HERE, VERIFIED)))
    ai = d["all_issues"]
    if len(ai) != 5898:
        raise SystemExit("master holds %d candidates, expected 5,898" % len(ai))
    verified = [f for f in ai if (f.get("verdict") or {}).get("is_real")]
    if len(verified) != 618:
        raise SystemExit("%d survived the panel, expected 618" % len(verified))
    refused_high = [f for f in ai if not (f.get("verdict") or {}).get("is_real")
                    and (f.get("verdict") or {}).get("salience") == "high"]
    if len(refused_high) != 1697:
        raise SystemExit("%d in the high-salience refused stratum, expected 1,697"
                         % len(refused_high))

    # Fresh means fresh. Assert it; do not trust yourself to have done it.
    prior = {k["finding_id"] for k in json.load(open(os.path.join(HERE, PRIOR_KEY)))}
    if len(prior) != 45:
        raise SystemExit("the existing pack's key holds %d ids, expected 45" % len(prior))
    pool_v = [f for f in verified if f["finding_id"] not in prior]
    pool_r = [f for f in refused_high if f["finding_id"] not in prior]
    if len(pool_v) != 618 - 30 or len(pool_r) != 1697 - 15:
        raise SystemExit("exclusion arithmetic: %d / %d fresh" % (len(pool_v), len(pool_r)))

    rng = random.Random(args.seed)
    draw_v = rng.sample(sorted(pool_v, key=lambda f: f["finding_id"]), n_ver)
    draw_r = rng.sample(sorted(pool_r, key=lambda f: f["finding_id"]), n_ref)
    drawn = {f["finding_id"] for f in draw_v + draw_r}
    if drawn & prior:
        raise SystemExit("EXCLUSION FAILED: %d drawn ids are in the existing pack"
                         % len(drawn & prior))
    if len(drawn) != n_ver + n_ref:
        raise SystemExit("the draw is not distinct")

    # Stratified split, so each lot carries the mix.
    rng.shuffle(draw_v)
    rng.shuffle(draw_r)
    lots = []
    for i in range(N_LOTS):
        items = ([(f, "verified_finding") for f in draw_v[i * args.verified:(i + 1) * args.verified]]
                 + [(f, "panel_refused") for f in
                    draw_r[i * (args.items - args.verified):(i + 1) * (args.items - args.verified)]])
        rng.shuffle(items)
        lots.append(items)

    all_units, _skipped = T.note_units()
    units = {u["note_key"]: u for u in all_units}
    for lot in lots:
        for f, _ in lot:
            if f["note_key"] not in units:
                raise SystemExit("no note unit for %s" % f["note_key"])

    rubric = open(os.path.join(HERE, RUBRIC)).read().strip()
    pack_text = open(os.path.join(HERE, PRIOR_PACK)).read()
    prior_answers = ""
    ans_path = os.path.join(HERE, "results/precision-sitting/release/answers.txt")
    if os.path.exists(ans_path):
        prior_answers = open(ans_path).read()

    # Highlighting is a lot-level property or nothing: if any item's evidence quote cannot
    # be located, no item in that lot is highlighted, so highlight presence can never vary
    # with an item's status. Where it is on, it is a function of the quote already printed
    # above the note, so it reveals nothing the clinician cannot already see.
    marks_ok = []
    for lot in lots:
        ok = True
        for f, _ in lot:
            u = units[f["note_key"]]
            nq, tq = (f.get("note_quote") or "").strip(), (f.get("source_quote") or "").strip()
            if nq and nq != "-" and not find_span(u["note"].strip(), nq):
                ok = False
            if tq and tq != "-" and not find_span(u["transcript"][:TRANSCRIPT_CHARS].strip(), tq):
                ok = False
        marks_ok.append(ok)

    os.makedirs(OUTDIR, exist_ok=True)
    key_all, written = [], []
    for i, items in enumerate(lots, 1):
        ids = [f["finding_id"] for f, _ in items]
        pack_sha = hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()[:16]
        doc, rows = build_lot(i, items, rubric, units, marks_ok[i - 1], pack_sha)
        assert_blind(doc, rows, prior_answers)
        assert_wording(doc, pack_text)
        path = os.path.join(OUTDIR, "lot-%d.html" % i)
        open(path, "w").write(doc)
        key_all += rows
        written.append((path, pack_sha, len(items)))

    key_path = os.path.join(OUTDIR, "key.json")
    json.dump(key_all, open(key_path, "w"), indent=1)

    words = []
    for items in lots:
        for f, _ in items:
            u = units[f["note_key"]]
            words.append(len(u["note"].split())
                         + len(u["transcript"][:TRANSCRIPT_CHARS].split()))
    man = {
        "generated_by": "second_clinician_sitting.py",
        "seed": args.seed,
        "n_lots": N_LOTS, "items_per_lot": args.items,
        "verified_per_lot": args.verified, "refused_per_lot": args.items - args.verified,
        "n_verified_pool": len(verified), "n_refused_high_pool": len(refused_high),
        "n_fresh_verified_pool": len(pool_v), "n_fresh_refused_pool": len(pool_r),
        "excluded_prior_ids": len(prior),
        "exclusion": "every id in the existing pack's key.json - all 45 drawn, not only the "
                     "30 adjudicated - asserted absent from the draw",
        "design": "verified findings drawn uniformly at random from the 618; refused "
                  "candidates drawn uniformly from the high-salience refused stratum, the "
                  "same stratum the existing pack used; split stratified so each lot "
                  "carries the mix; order shuffled within each lot; no status label "
                  "anywhere in either file",
        "lots": [{"lot": i + 1, "file": os.path.basename(p), "n_items": n,
                  "pack_sha256_16": sha, "highlighting": marks_ok[i],
                  "ids_sha256": hashlib.sha256(json.dumps(sorted(
                      [r["finding_id"] for r in key_all if r["lot"] == i + 1])).encode()).hexdigest()}
                 for i, (p, sha, n) in enumerate(written)],
        "transcript_chars": TRANSCRIPT_CHARS,
        "scribe_letters": "the released A/B/C convention; the file names no product",
        "size": size_note(args.items, args.verified),
        "drawn_words_per_item": {"mean": round(statistics.mean(words)),
                                 "median": round(statistics.median(words)),
                                 "min": min(words), "max": max(words)},
        "reporting_plan": REPORTING_PLAN,
    }
    json.dump(man, open(os.path.join(OUTDIR, "manifest.json"), "w"), indent=1)

    print("wrote %s" % OUTDIR)
    for path, sha, n in written:
        print("  %-14s %2d items  %4d KB  pack %s  highlights %s"
              % (os.path.basename(path), n, os.path.getsize(path) // 1024, sha,
                 "on" if marks_ok[written.index((path, sha, n))] else "OFF (a quote would not locate)"))
    print("  key.json      %d rows (NOT for the clinician)" % len(key_all))
    s = man["size"]
    print("\nsize: %.1f-%.1f min/item x %d items = %.0f-%.0f min per lot (target %d)"
          % (s["minutes_per_item"]["fast"], s["minutes_per_item"]["slow"], args.items,
             s["estimated_minutes_per_lot"]["fast"], s["estimated_minutes_per_lot"]["slow"],
             TARGET_MINUTES))
    print(s["finding"])


REPORTING_PLAN = [
    "An independent precision estimate on the verified subset, with a Wilson interval, "
    "reported BESIDE the author's 20 of 21 and never pooled with it: two raters, "
    "disjoint samples.",
    "An independent severity read against the rubric grades on the same items: exact, "
    "adjacent, and the DIRECTION of the disagreements. Direction is the point - two prior "
    "samples disagree about which way the rubric leans.",
    "The refused items reported separately. The author judged all nine of his refusals "
    "genuine; an independent clinician replicating that strengthens the claim and breaking "
    "it is the most interesting thing this sitting can produce.",
    "Every disagreement described individually in plain words: what the finding claimed, "
    "what the clinician said, what it turns on.",
    "No kappa, and say why: two raters on disjoint samples produce no interassessor "
    "agreement by construction. TRIPOD-LLM 7d stays not-met, with one line saying the "
    "second rater was given fresh items to maximise the number of findings a human has "
    "assessed rather than to compute a coefficient a dozen shared items could not support.",
    "Be straight about the width. The value of this sitting is categorical rather than "
    "statistical - a non-author looked at a random sample - and the write-up should say "
    "so rather than dressing a wide interval as a precise one.",
]

# ------------------------------------------------------------------- templates
# Note for anyone editing these: assert_blind() strips every item block's text content and
# requires the remainder to be byte-identical across items, so nothing in ITEM_TMPL may
# vary by item except the values substituted in below.

ITEM_TMPL = """<section class="item" data-i="{n}" hidden>
 <p class="crumb">Item <span class="ino">{n}</span></p>
 <div class="claim">
  <p class="kv"><span>Claimed failure mode</span> {mode}</p>
  <p class="lead">{desc}</p>
  <p class="kv"><span>Note-side evidence quote</span> {nq}</p>
  <p class="kv"><span>Transcript-side evidence quote</span> {tq}</p>
 </div>
 <h3>The full note (Scribe {letter})</h3>
 <pre class="doc note">{note}</pre>
 <h3>The full transcript</h3>
 <pre class="doc tx">{transcript}</pre>
 <div class="qbox">
  <p class="q">Reading the note against the transcript, is this
a genuine documentation failure a reviewing clinician should accept as real?</p>
  <div class="btns" role="group">
   <button type="button" class="v" data-v="genuine">genuine</button>
   <button type="button" class="v" data-v="not-genuine">not-genuine</button>
   <button type="button" class="v" data-v="cannot-judge">cannot-judge</button>
  </div>
  <ul class="defs">
   <li><code>genuine</code> - a real failure; the note asserts, omits or distorts something the
transcript does not support, in a way a reviewing clinician would accept as a defect</li>
   <li><code>not-genuine</code> - not a real failure (the claim misreads the note or transcript, the
content is actually captured, the inference is clinically justified, etc.)</li>
   <li><code>cannot-judge</code> - the materials are insufficient to decide</li>
  </ul>
  <div class="sev" hidden>
   <p class="q">And the severity grade, against the rubric below: </p>
   <div class="btns" role="group">
    <button type="button" class="s" data-s="critical">critical</button>
    <button type="button" class="s" data-s="supporting">supporting</button>
    <button type="button" class="s" data-s="peripheral">peripheral</button>
   </div>
   <details class="rubwrap"><summary>the severity rubric</summary>
<pre class="rubric">{rubric}</pre>
   </details>
  </div>
 </div>
</section>"""

PAGE_TMPL = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Note review - set {lot}</title>
<style>
:root{{--bg:#fbfbf9;--fg:#1b1b19;--mut:#5d5d57;--line:#dedcd4;--card:#fff;--accent:#1f5f4f;--mark:#fdf0b8}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}}
.wrap{{max-width:820px;margin:0 auto;padding:24px 20px 96px}}
h1{{font-size:26px;line-height:1.25;margin:0 0 16px}}
h2{{font-size:19px;margin:28px 0 8px}}
h3{{font-size:15px;margin:26px 0 6px;color:var(--mut);text-transform:uppercase;letter-spacing:.06em}}
p{{margin:0 0 12px}}
.mut{{color:var(--mut)}}
.crumb{{color:var(--mut);font-size:14px;margin-bottom:14px}}
.claim{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-bottom:18px}}
.lead{{font-size:18px;line-height:1.5;margin:10px 0}}
.lede{{font-size:17px;line-height:1.55;margin-bottom:16px}}
.brief{{padding-left:20px;margin-bottom:18px}}
.brief li{{margin-bottom:9px;line-height:1.5}}
.kv{{font-size:14px;color:var(--mut);margin:6px 0}}
.kv span{{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:#8a8a80}}
.qbox{{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:10px;padding:16px 18px;margin-bottom:8px}}
.q{{font-weight:600;margin-bottom:10px}}
.btns{{display:flex;flex-wrap:wrap;gap:8px}}
button{{font:inherit;cursor:pointer}}
.v,.s{{background:#fff;border:1.5px solid var(--line);border-radius:999px;padding:9px 18px;color:var(--fg)}}
.v:hover,.s:hover{{border-color:var(--accent)}}
.v[aria-pressed=true],.s[aria-pressed=true]{{background:var(--accent);border-color:var(--accent);color:#fff}}
.defs{{margin:12px 0 0;padding-left:20px;font-size:13.5px;color:var(--mut);line-height:1.5}}
.defs li{{margin-bottom:5px}}
.defs code{{font:12.5px ui-monospace,Menlo,Consolas,monospace;color:var(--fg)}}
.sev{{margin-top:16px;padding-top:14px;border-top:1px dashed var(--line)}}
.link{{background:none;border:none;color:var(--accent);text-decoration:underline;padding:0;font-size:14px}}
.doc{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;white-space:pre-wrap;word-wrap:break-word;font:14px/1.65 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;max-height:52vh;overflow:auto;margin:0}}
mark{{background:var(--mark);padding:1px 0}}
.bar{{position:fixed;left:0;right:0;bottom:0;background:rgba(251,251,249,.97);border-top:1px solid var(--line);padding:10px 20px}}
.bar .in{{max-width:820px;margin:0 auto;display:flex;align-items:center;gap:12px}}
.bar .prog{{flex:1;font-size:14px;color:var(--mut)}}
.nav{{background:var(--accent);color:#fff;border:none;border-radius:8px;padding:10px 20px}}
.nav.ghost{{background:#fff;color:var(--fg);border:1.5px solid var(--line)}}
.nav[disabled]{{opacity:.4;cursor:default}}
.rubwrap{{margin-top:14px}}
.rubwrap summary{{font-size:13.5px;font-weight:400;color:var(--mut)}}
.dots{{display:flex;flex-wrap:wrap;gap:6px;margin:18px 0}}
.dots button{{width:34px;height:34px;border-radius:8px;border:1.5px solid var(--line);background:#fff;font-size:13px;color:var(--mut)}}
.dots button.done{{background:var(--accent);border-color:var(--accent);color:#fff}}
.dots button.here{{outline:2px solid var(--accent);outline-offset:2px}}
.box{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin:16px 0}}
ul{{margin:0 0 12px;padding-left:22px}} li{{margin-bottom:7px}}
textarea{{width:100%;height:150px;font:13px/1.5 ui-monospace,Menlo,Consolas,monospace;padding:12px;border:1.5px solid var(--line);border-radius:8px;background:#fff;color:var(--fg)}}
.ok{{color:var(--accent);font-weight:600}}
details summary{{cursor:pointer;font-weight:600;margin-bottom:8px}}
pre.rubric{{white-space:pre-wrap;font:13px/1.6 ui-monospace,Menlo,Consolas,monospace;background:#f4f3ee;border-radius:8px;padding:14px;max-height:60vh;overflow:auto}}
@media (prefers-color-scheme:dark){{
:root{{--bg:#16171a;--fg:#e9e8e3;--mut:#9d9d96;--line:#33353a;--card:#1e2024;--accent:#6fd0b4;--mark:#5a5220}}
.v[aria-pressed=true],.s[aria-pressed=true],.nav{{color:#12130f}}
.dots button.done{{color:#12130f}}
pre.rubric{{background:#1a1c1f}}
}}
</style></head><body><div class="wrap">

<section id="intro">
<h1>Reviewing AI-written clinical notes</h1>
<p class="lede">Three commercial AI scribe products were each given the same consultations, and each wrote a clinical note from the recording. An automated system then flagged things it thought the notes had got wrong. <strong>Your job is to say, for each flag, whether it is a real documentation problem - and if it is, how much it would matter.</strong></p>
<ul class="brief">
<li><strong>No real patients anywhere in this.</strong> The consultations are recordings of clinicians with actors, simulated encounters released for research, and scenarios written for the study.</li>
<li><strong>Your disagreement is the valuable part.</strong> The only person who has done this so far is one of the study's authors, who also built the thing being checked. An independent doctor who marks things differently is exactly what the study needs, and nobody is hoping for a particular answer.</li>
<li><strong>You are not being asked</strong> to rate or rank the products (you are not told which wrote which note), to say whether you would have caught the error at sign-off, or to look anything up. Judge the note against the transcript, as you would a colleague's note.</li>
<li id="savenote"><strong>Answers save in this browser as you go</strong>, so you can stop and reopen the file later. Nothing is sent anywhere - this file has no internet connection of any kind.</li>
<li><strong>{n_items} items.</strong> Take as long over them as you like - it saves as you go, so one sitting or several is fine. At the end a button downloads your answers as a small file - send that back.</li>
<li>The notes came from commercial products whose terms do not let us republish them, so please do not forward or post this file.</li>
</ul>
<details><summary>The severity rubric, for the second question - it is one click away on every item too</summary>
<pre class="rubric">{rubric}</pre>
</details>
<p><button type="button" class="nav" id="start">Start</button></p>
</section>

<section id="items" hidden>
<p id="resume" class="mut" hidden><span id="resumed"></span></p>
{cards}
<div class="dots" id="dots"></div>
<p class="mut" id="tail"></p>
</section>

<section id="done" hidden>
<h1>Finished - thank you</h1>
<p id="summary" class="mut"></p>
<div class="box">
<p><strong>1. Download your answers</strong> and email the file back.</p>
<p><button type="button" class="nav" id="dl">Download my answers</button> <span id="dlok" class="ok"></span></p>
</div>
<div class="box">
<p><strong>2. Or, if the download does not work</strong>, copy the text below into an email instead. It is exactly the same information.</p>
<p><button type="button" class="nav ghost" id="cp">Copy to clipboard</button> <span id="cpok" class="ok"></span></p>
<textarea id="out" readonly></textarea>
</div>
<p class="mut">The file records only your verdicts, your grades and how long each item took. It contains none of the note or transcript text, so it is small and safe to email.</p>
<p><button type="button" class="nav ghost" id="back">Back to the items</button></p>
</section>

</div>
<div class="bar" id="bar" hidden><div class="in">
<button type="button" class="nav ghost" id="prev">Back</button>
<span class="prog" id="prog"></span>
<button type="button" class="nav" id="next">Next</button>
</div></div>

<script>
(function(){{
"use strict";
var LOT={lot}, PACK="{pack_sha}", N={n_items};
var KEY="note-review-"+LOT+"-"+PACK;
var state={{answers:{{}},ms:{{}},started:null}};

// Saving can fail for reasons that have nothing to do with the clinician: a browser
// with site data switched off, a private window, or a local file opened where the
// browser refuses storage. If it does, the promise on the intro screen would be a lie,
// so we test it once and rewrite that promise rather than silently losing the work.
var STORAGE=true;
function load(){{
  try{{
    window.localStorage.setItem(KEY+"-probe","1");
    window.localStorage.removeItem(KEY+"-probe");
    var raw=window.localStorage.getItem(KEY);
    if(raw){{var p=JSON.parse(raw); if(p&&p.answers){{state=p;}}}}
  }}catch(e){{ STORAGE=false; }}
  if(!state.started){{state.started=new Date().toISOString();}}
  if(!STORAGE){{
    var w=document.getElementById("savenote");
    if(w) w.innerHTML="<strong>This browser will not let the file save your answers</strong>, "+
      "so please keep this tab open until you have downloaded them at the end. Nothing is "+
      "sent anywhere - this file has no internet connection of any kind.";
  }}
}}
function save(){{
  try{{window.localStorage.setItem(KEY,JSON.stringify(state));}}catch(e){{ STORAGE=false; }}
}}

var items=[].slice.call(document.querySelectorAll("section.item"));
var cur=0, tick=null, last=0;

function complete(i){{
  var a=state.answers[i];
  if(!a||!a.verdict) return false;
  return a.verdict!=="genuine" || !!a.severity;
}}
function nDone(){{var n=0;for(var i=1;i<=N;i++){{if(complete(i))n++;}}return n;}}

function startClock(){{last=Date.now(); if(tick) clearInterval(tick);
  tick=setInterval(function(){{
    if(document.hidden){{last=Date.now();return;}}
    var i=cur+1, now=Date.now();
    state.ms[i]=(state.ms[i]||0)+(now-last); last=now; save();
  }},1000);
}}
function stopClock(){{if(tick){{clearInterval(tick);tick=null;}}}}

function paint(){{
  items.forEach(function(el,k){{el.hidden = k!==cur;}});
  var a=state.answers[cur+1]||{{}};
  var el=items[cur];
  [].forEach.call(el.querySelectorAll("button.v"),function(b){{
    b.setAttribute("aria-pressed", a.verdict===b.getAttribute("data-v") ? "true":"false");
  }});
  el.querySelector(".sev").hidden = a.verdict!=="genuine";
  [].forEach.call(el.querySelectorAll("button.s"),function(b){{
    b.setAttribute("aria-pressed", a.severity===b.getAttribute("data-s") ? "true":"false");
  }});
  document.getElementById("prog").textContent =
    "Item "+(cur+1)+" of "+N+"  -  "+nDone()+" answered"+(complete(cur+1)?"":"  -  no answer yet");
  document.getElementById("prev").disabled = cur===0;
  document.getElementById("next").textContent = cur===N-1 ? "Finish" : "Next";
  paintDots();
  document.getElementById("tail").textContent =
    "Jump to any item above. You can change an answer at any time before you download.";
  window.scrollTo(0,0);
}}

function paintDots(){{
  var d=document.getElementById("dots");
  if(!d.childNodes.length){{
    for(var i=1;i<=N;i++){{
      var b=document.createElement("button");
      b.type="button"; b.textContent=String(i); b.setAttribute("data-go",String(i));
      d.appendChild(b);
    }}
    d.addEventListener("click",function(ev){{
      var b=ev.target.closest("button[data-go]");
      if(b) go(parseInt(b.getAttribute("data-go"),10)-1);
    }});
  }}
  [].forEach.call(d.querySelectorAll("button"),function(b,k){{
    b.className=(complete(k+1)?"done":"")+(k===cur?" here":"");
  }});
}}

function go(k){{
  if(k<0) return;
  if(k>=N){{ finish(); return; }}
  cur=k; paint(); startClock();
}}

document.getElementById("items").addEventListener("click",function(ev){{
  var b=ev.target.closest("button"); if(!b) return;
  var i=cur+1;
  if(b.classList.contains("v")){{
    var a=state.answers[i]||{{}};
    a.verdict=b.getAttribute("data-v");
    if(a.verdict!=="genuine") delete a.severity;
    state.answers[i]=a; save(); paint();
  }} else if(b.classList.contains("s")){{
    var a2=state.answers[i]||{{}};
    a2.severity=b.getAttribute("data-s"); state.answers[i]=a2; save(); paint();
  }}
}});

document.getElementById("start").addEventListener("click",function(){{
  document.getElementById("intro").hidden=true;
  document.getElementById("items").hidden=false;
  document.getElementById("bar").hidden=false;
  go(cur);
}});
document.getElementById("prev").addEventListener("click",function(){{go(cur-1);}});
document.getElementById("next").addEventListener("click",function(){{go(cur+1);}});
document.getElementById("back").addEventListener("click",function(){{
  document.getElementById("done").hidden=true;
  document.getElementById("items").hidden=false;
  document.getElementById("bar").hidden=false;
  go(cur);
}});

function payload(){{
  var rows=[];
  for(var i=1;i<=N;i++){{
    var a=state.answers[i]||{{}};
    rows.push({{item:i, verdict:a.verdict||null, severity:a.severity||null,
                seconds:Math.round((state.ms[i]||0)/1000)}});
  }}
  return {{set:LOT, pack:PACK, n_items:N, started:state.started,
           finished:new Date().toISOString(),
           total_minutes:Math.round(rows.reduce(function(s,r){{return s+r.seconds;}},0)/60),
           answers:rows}};
}}

function finish(){{
  stopClock();
  document.getElementById("items").hidden=true;
  document.getElementById("bar").hidden=true;
  document.getElementById("done").hidden=false;
  var p=payload();
  var miss=p.answers.filter(function(r){{return !r.verdict;}}).length;
  document.getElementById("summary").textContent =
    (N-miss)+" of "+N+" items answered"+(miss?", "+miss+" left blank - that is fine, and you can go back":"")+
    ". About "+p.total_minutes+" minutes of reading.";
  document.getElementById("out").value=JSON.stringify(p,null,1);
  window.scrollTo(0,0);
}}

document.getElementById("dl").addEventListener("click",function(){{
  try{{
    var blob=new Blob([JSON.stringify(payload(),null,1)],{{type:"application/json"}});
    var u=URL.createObjectURL(blob);
    var a=document.createElement("a");
    a.href=u; a.download="note-review-set-"+LOT+".json";
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    setTimeout(function(){{URL.revokeObjectURL(u);}},2000);
    document.getElementById("dlok").textContent="saved";
  }}catch(e){{
    document.getElementById("dlok").textContent="that did not work - please use the copy button below";
  }}
}});
document.getElementById("cp").addEventListener("click",function(){{
  var t=document.getElementById("out");
  t.select(); t.setSelectionRange(0,999999);
  var ok=false;
  try{{ok=document.execCommand("copy");}}catch(e){{}}
  document.getElementById("cpok").textContent = ok ? "copied" : "select the text and copy it by hand";
}});

document.addEventListener("visibilitychange",function(){{ last=Date.now(); }});
window.addEventListener("beforeunload",save);

load();
// Come back to where you stopped rather than to the front page.
var answered=0;
for(var i=1;i<=N;i++){{ if(state.answers[i]&&state.answers[i].verdict) answered++; }}
if(answered>0){{
  var first=N;
  for(var j=1;j<=N;j++){{ if(!complete(j)){{ first=j; break; }} }}
  document.getElementById("resume").hidden=false;
  document.getElementById("resumed").textContent=
    "Welcome back - "+answered+" of "+N+" answered. Picking up where you stopped; "+
    "the Back button and the item strip below let you revisit anything.";
  document.getElementById("intro").hidden=true;
  document.getElementById("items").hidden=false;
  document.getElementById("bar").hidden=false;
  go(first-1);
}} else {{
  paint();
}}
}})();
</script>
</body></html>"""


if __name__ == "__main__":
    main()
