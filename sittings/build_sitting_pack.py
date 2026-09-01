"""build_sitting_pack.py - assembles the author-clinician's single sitting into one file.

One structured sitting by an author who is also a clinician, disclosed as such: extraction
audit, spot-checks of the instruments that gate the corpus,
~20 rubric-anchored severity grades and a few realism ratings, as ONE pack, phone/laptop
readable, target 60-90 minutes.

RE-CUT 2026-08-14. The 11 August pack (kept verbatim at
`master/archive/sitting_pack_2026-08-11.md`) was assembled before the factorial dataset,
the skeptic panel and the pipeline judge existed, so half of it sampled artifacts the study
has since superseded. Shape unchanged; sources rebased on the final ones:

  A  now samples the PIPELINE's inference-time extractions (results/w2-pipeline/_cache),
     so agreement validates the deployed instrument, not a construction-time sheet.
  B  unchanged (the ground-truth gate's repair-loop artifact is unchanged), trimmed 10 -> 8.
  C  now the factorial pairs the verification panel REJECTED (the 44% complete-class
     rejection) rather than the older ground-truth-gate exclusions.
  D  new: findings the two-skeptic panel CUT, stratified over the omission
     audit's buckets. These verdicts are what the 10.5% survival rate rests on.
  E  now drawn from the factorial severity strata (master/factorial_severity.json), still
     blinded, still c/s/p.
  F  new: notes where the pipeline judge's per-fact verdict and the monolithic grid judge
     disagree. The study's central contrast, adjudicated on real cases.
  G  realism/fidelity, trimmed 6 -> 3.
  Q  rewritten for the final design.

The queues behind it hold thousands of items. This script curates them DOWN to a
sitting-sized pack by seeded stratified draw, and inlines everything each item needs - the
artefact under judgment, the trimmed excerpt, the specific question - so the pack is read
start to finish with nothing else open. Every draw is seeded (SEED below) so the pack
regenerates identically, and the coverage note at the end says what was left in the queues
and where it lives, so nothing is hidden by the curation.

Blinding: section E never prints a model's grade or whether the two rubric arms split on
the item. Section F never says which way the two judges disagreed, or whether the note in
front of you is an original or an edited one.

Sources:
  results/w2-pipeline/_cache/facts_*.json     the deployed extractor's fact lists (A)
  master/sitting_wa_items.json                ground-truth gate adjudication items (B)
  master/wa_audit_flags.json                  the auditor flags behind each repair (B)
  results/w-a-state/wa_state.json             note texts + per-pair seed verdicts (B)
  master/dataset_v2.json                      factorial_pairs_dropped_unverified (C), pairs (E)
  master/omission_pairs_v2.json               the rejected pairs as built (C)
  master/omission_verification_v2.json        the panel's per-field verdicts (C)
  master/omission_audit_omission-audit-full-v2.json   40 audited panel refusals (D)
  master/factorial_severity.json              880 rubric-graded facts (E)
  master/severity-rubric.md                    the instrument, printed verbatim (E)
  results/w2-pipeline/_state/confirm-B3.jsonl per-fact verdicts (F)
  results/w2-ablation/_state/grid-main2.jsonl the monolithic judge's flags (F)
  master/arms_confirm_subset.json             the notes both judged (F)
  master/trapblind_scenarios_critiqued.json   trap-blind transcripts (G)
  master/fact_sheets_<src>_core.json          transcripts + extracted sheets (E, G)

Output: master/sitting_pack.md

Usage:
  python3 sittings/build_sitting_pack.py [--out master/sitting_pack.md]
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import argparse, glob, json, os, random, re
from collections import Counter, defaultdict

from common import HERE
import w2_common as W

MASTER = os.path.join(HERE, "master")
SEED = 20260809                      # the sitting seed, unchanged so B redraws identically
BUILT = "2026-08-14"                 # the re-cut date, stamped in the pack header
OUT_DEFAULT = os.path.join(MASTER, "sitting_pack.md")
ARCHIVED = "master/archive/sitting_pack_2026-08-11.md"

# per-section item counts and the minutes each is budgeted (stated in the pack)
BUDGET = {"A": ("Fact-list extraction audit", 10, 8),
          "B": ("Repaired-note spot checks", 8, 9),
          "C": ("Rejected-pair adjudications", 6, 9),
          "D": ("Panel-killed findings", 10, 11),
          "E": ("Severity grades", 20, 18),
          "F": ("Pipeline-judge disagreements", 10, 13),
          "G": ("Realism and fidelity", 3, 4),
          "Q": ("Standing methodology questions", 3, 3)}

# the answer codes each section takes, and the labels the review tool puts on its buttons
CODES = {"A": ("`y` quote supports it / `n` it does not / `?` unsure", "y / n / ?"),
         "B": ("`y` good repair / `n` not / `?` unsure", "y / n / ?"),
         "C": ("`y` right to reject / `n` should have kept it / `?` unsure", "y / n / ?"),
         "D": ("`y` real omission / `n` fair cut / `?` unsure", "y / n / ?"),
         "E": ("`c` critical / `s` supporting / `p` peripheral", "c / s / p"),
         "F": ("`y` captured / `n` not captured / `?` unsure", "y / n / ?"),
         "G": ("`1`-`5` (5 = indistinguishable from real)", "1 / 2 / 3 / 4 / 5"),
         "Q": ("free text, a sentence each", "free text")}

WHAT = {"A": "does the quote support the fact",
        "B": "did the repair fix a real defect",
        "C": "was the pair rightly rejected",
        "D": "was a cut finding really not a finding",
        "E": "grade the severity",
        "F": "is the fact in the note or not",
        "G": "rate realism / fidelity",
        "Q": "three open questions"}


def J(path):
    return json.load(open(os.path.join(HERE, path)))


def JL(path):
    with open(os.path.join(HERE, path)) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def rng(*tag):
    return random.Random("|".join([str(SEED), *map(str, tag)]))


def squash(t, n=None):
    t = re.sub(r"[ \t]+", " ", (t or "").strip())
    t = t.replace("\n", " ")
    return t if not n or len(t) <= n else t[:n].rsplit(" ", 1)[0] + " ..."


def keywords(text, n=12):
    """Content words for locating a subject inside a note or transcript."""
    stop = {"the", "and", "that", "with", "this", "from", "have", "has", "was", "were",
            "not", "for", "are", "but", "any", "its", "which", "note", "patient", "record",
            "recorded", "records", "does", "did", "been", "than", "then", "there", "their",
            "omits", "omitted", "note's", "doctor", "clinician"}
    seen, out = set(), []
    for w in re.findall(r"[A-Za-z][A-Za-z\-']{3,}", (text or "").lower()):
        if w in stop or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out[:n]


def excerpt_around(text, subject, chars=900):
    """The window of `text` that best covers `subject`'s content words, cut on line
    boundaries. Falls back to the head of the text when nothing matches."""
    lines = [l for l in (text or "").splitlines()]
    if not lines:
        return ""
    kw = keywords(subject)
    scores = [sum(1 for w in kw if w in l.lower()) for l in lines]
    if not any(scores):
        return squash_block("\n".join(lines), chars)
    best, span = -1, (0, 1)
    for i in range(len(lines)):
        acc, j, used = 0, i, 0
        while j < len(lines) and used < chars:
            acc += scores[j]
            used += len(lines[j]) + 1
            j += 1
        if acc > best:
            best, span = acc, (i, j)
    i, j = span
    out = "\n".join(lines[i:j]).strip()
    pre = "[...]\n" if i > 0 else ""
    post = "\n[...]" if j < len(lines) else ""
    return pre + out + post


def squash_block(text, chars):
    t = (text or "").strip()
    if len(t) <= chars:
        return t
    return t[:chars].rsplit("\n", 1)[0].rstrip() + "\n[...]"


def middle_window(text, chars):
    """A window from the clinical middle of a long transcript - past the greeting, before
    the sign-off - which is where realism either holds up or does not."""
    lines = [l for l in (text or "").splitlines() if l.strip()]
    if not lines:
        return ""
    start = max(0, int(len(lines) * 0.18))
    out, used = [], 0
    for l in lines[start:]:
        if used + len(l) > chars:
            break
        out.append(l)
        used += len(l) + 1
    return ("[...]\n" if start else "") + "\n".join(out) + "\n[...]"


def word_diff(a, b):
    """(removed, added) word runs between two versions of one line. A line-level diff of a
    note buries a deleted clause inside 400 characters of unchanged text; this pulls out
    exactly what left and what arrived, which is the only part the reader is judging."""
    import difflib
    wa, wb = a.split(), b.split()
    rem, add = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, wa, wb).get_opcodes():
        if tag in ("delete", "replace") and wa[i1:i2]:
            rem.append(" ".join(wa[i1:i2]))
        if tag in ("insert", "replace") and wb[j1:j2]:
            add.append(" ".join(wb[j1:j2]))
    return rem, add


def diff_lines(clean, errored):
    """The clean->errored diff. Whole lines that vanished are shown whole; a line that was
    edited in place is reduced to the words that changed."""
    import difflib
    a, b = clean.splitlines(), errored.splitlines()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and (i2 - i1) == (j2 - j1):
            for x, y in zip(a[i1:i2], b[j1:j2]):
                rem, add = word_diff(x, y)
                for r in rem:
                    out.append(f"- REMOVED: {squash(r, 400)}")
                for ad in add:
                    out.append(f"+ left in its place: {squash(ad, 200)}")
            continue
        for l in a[i1:i2]:
            if l.strip():
                out.append(f"- REMOVED: {squash(l, 400)}")
        for l in b[j1:j2]:
            if l.strip():
                out.append(f"+ ADDED: {squash(l, 400)}")
    return out


def fence(text):
    """Content for a fenced block. Two structural characters would break the pack's own
    markdown if a note or transcript ever carried them: a line opening with `#` (the
    parser splits sections on those) and a stray fence. Neither occurs in the corpus
    today; this keeps it that way if one ever does."""
    out = []
    for l in (text or "").splitlines():
        if l.lstrip().startswith("#"):
            l = " " + l
        out.append(l.replace("```", "'''"))
    return "\n".join(out)


def escape_tildes(text):
    """`~` is strikethrough syntax in GFM and in most phone markdown viewers, so two
    approximation tildes in one paragraph strike out everything between them. Escape them
    in prose; leave fenced blocks (transcripts, notes, the answer sheet) exactly as they
    are."""
    out, fenced = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        out.append(line if fenced or line.startswith("```")
                   else line.replace("~", "\\~"))
    return "\n".join(out)


def ans(item_id, codes):
    return f"\n**{item_id}** -> `{codes}`\n"


def spread(pool, n, keyfns, r):
    """Seeded draw of n from pool, spread across one or more key functions in priority
    order (the first key is balanced hardest) as evenly as the pool allows."""
    if callable(keyfns):
        keyfns = [keyfns]
    pool = list(pool)
    r.shuffle(pool)
    used, out = [Counter() for _ in keyfns], []
    while pool and len(out) < n:
        pick = min(pool, key=lambda x: tuple(u[k(x)] for u, k in zip(used, keyfns))
                   + (pool.index(x),))
        out.append(pick)
        for u, k in zip(used, keyfns):
            u[k(pick)] += 1
        pool.remove(pick)
    return out


def sentences(text, n=2, chars=320):
    """First n sentences of a model's rationale - enough to see the reasoning, short
    enough to read at speed."""
    t = squash(text)
    parts = re.split(r"(?<=[.!?])\s+", t)
    return squash(" ".join(parts[:n]), chars)


# ------------------------------------------------------------------ sections
def section_a(pool_consults):
    """10 facts drawn from the pipeline judge's own inference-time extractions: does the
    evidence quote it cites actually support the fact it wrote?

    The pool is fixed to the consultations of the held-out CONFIRMATION run - the one the
    published pipeline-judge figures are computed on - rather than to whatever is in the
    cache directory, which later
    runs keep adding to and which would make this draw non-reproducible."""
    by_st = defaultdict(list)
    tot_facts = 0
    for p in sorted(glob.glob(os.path.join(HERE, "results/w2-pipeline/_cache/facts_*.json"))):
        d = json.load(open(p))
        if (d["stratum"], d["consultation"]) not in pool_consults:
            continue
        tot_facts += len(d.get("facts") or [])
        good = [f for f in (d.get("facts") or [])
                if (f.get("fact") or "").strip() and len((f.get("evidence") or "").strip()) >= 15]
        if good:
            by_st[d["stratum"]].append((d, good))
    picks = []
    for st, n in (("aci", 4), ("primock", 3), ("authored", 2), ("trapblind", 1)):
        pool = sorted(by_st[st], key=lambda x: x[0]["consultation"])
        r = rng("A2", st)
        r.shuffle(pool)
        for d, good in pool[:n]:
            f = rng("A2", st, d["consultation"]).choice(good)
            ids = [x["id"] for x in d["facts"]]
            picks.append({"stratum": st, "cid": d["consultation"], "f": f,
                          "pos": ids.index(f["id"]) + 1, "n_facts": len(d["facts"])})
    rng("A2", "order").shuffle(picks)
    stats = {"consultations": sum(len(v) for v in by_st.values()), "facts": tot_facts}
    return picks[:BUDGET["A"][1]], stats


def section_b(items, notes, flags_by_note):
    """8 repaired notes, stratified by stratum and by how heavily flagged they were. The
    same draw as the 11 August pack, two items shorter."""
    pool = [i for i in items if i["kind"] == "repaired_note_review"]

    def band(i):
        n = i["n_confirmed_flags_round0"]
        return "heavy" if n >= 4 else ("several" if n >= 2 else "single")

    quota = {("primock", "heavy"): 1, ("primock", "several"): 2, ("primock", "single"): 1,
             ("aci", "heavy"): 1, ("aci", "several"): 1, ("aci", "single"): 1,
             ("trapblind", "single"): 1}
    picks = []
    for k, n in sorted(quota.items()):
        cand = sorted([i for i in pool
                       if (i["note"].split("|")[0], band(i)) == k], key=lambda i: i["note"])
        r = rng("B", *k)
        r.shuffle(cand)
        picks.extend(cand[:n])
    # top up with the multi-round repairs - the ones the instrument found hardest
    rest = sorted([i for i in pool if i not in picks and i["rounds"] >= 2],
                  key=lambda i: (-i["rounds"], -i["n_confirmed_flags_round0"], i["note"]))
    r = rng("B", "topup")
    r.shuffle(rest)
    picks.extend(rest[:BUDGET["B"][1] - len(picks)])
    out = []
    for i in picks[:BUDGET["B"][1]]:
        fl = [f for f in flags_by_note.get(i["note"], []) if f["round"] == 0]
        fl.sort(key=lambda f: (-f["n_fail"], f["section"]))
        out.append({"item": i, "flags": fl, "note_text": (notes.get(i["note"]) or {}).get("text", "")})
    return out


def section_c(dropped, verif, built):
    """6 of the 92 factorial pairs the verification panel rejected, one per failure
    signature: was the rejection right?"""
    sig = {}
    for pid in dropped:
        v = verif.get(pid)
        if not v or pid not in built:
            continue
        fails = tuple(sorted(v.get("sem_fails") or []))
        algo = tuple(sorted((v.get("algo") or {}).get("failed") or []))
        sig[pid] = (fails, algo)
    want = [(("reads_naturally",), (), 2),
            (("truly_absent",), (), 1),
            (("reads_naturally", "truly_absent"), (), 1),
            (("no_collateral_loss", "spans_all_state_fact"), (), 1)]
    picks, taken = [], set()
    for fails, algo, n in want:
        cand = [p for p, (f, a) in sig.items() if f == fails and not a and p not in taken]
        r = rng("C2", "|".join(fails))
        for p in spread(cand, n, lambda p: p.split("|")[1], r):
            picks.append(p)
            taken.add(p)
    # one long combination, preferring one that also failed an algorithmic check
    longs = [p for p, (f, a) in sig.items()
             if p not in taken and (len(f) >= 3 or a)]
    longs.sort(key=lambda p: (not sig[p][1], -len(sig[p][0]), p))
    if longs:
        picks.append(longs[0])
    rng("C2", "order").shuffle(picks)
    return [{"pair_id": p, "v": verif[p], "b": built[p], "reason": dropped[p]}
            for p in picks[:BUDGET["C"][1]]]


def section_d(audit, notes_by_key):
    """10 of the 40 audited panel refusals, stratified over the audit's own buckets - all
    3 wrongly-cut suspects, then present-elsewhere, not-material and not-required."""
    by_bucket = defaultdict(list)
    for rec in audit["records"]:
        if rec.get("note_key") in notes_by_key:
            by_bucket[rec["bucket"]].append(rec)
    picks = []
    for bucket, n in (("wrongly_cut", 3), ("present_elsewhere", 3),
                      ("not_material", 2), ("not_required", 2)):
        cand = sorted(by_bucket.get(bucket, []), key=lambda r: r["finding_id"])
        r = rng("D2", bucket)
        picks.extend(spread(cand, n, lambda x: x["scribe"], r))
    rng("D2", "order").shuffle(picks)          # buckets interleaved: the label never leaks
    return picks[:BUDGET["D"][1]]


def section_e(sev, used_uids, transcripts, complaints):
    """20 facts from the factorial's own severity strata, balanced across the three grades
    and salted with the cases the two rubric arms split on. Blinded."""
    pool = []
    for uid, f in sev["facts"].items():
        if uid not in used_uids or f.get("split") != "eval":
            continue
        if not transcripts.get((f["stratum"], f["id"])):
            continue
        pool.append(f)
    # 7/7/6 over the three grades; within supporting and peripheral, four each come from
    # the cases the two rubric arms split on (critical has none by construction - a split
    # resolves downward). One greedy pass over all six buckets with a shared usage count,
    # so no consultation carries three items just because it is fact-rich.
    plan = [("critical", False, 7), ("supporting", True, 4), ("supporting", False, 3),
            ("peripheral", True, 4), ("peripheral", False, 2)]
    picks, used_c, used_s = [], Counter(), Counter()
    for grade, want_flag, k in plan:
        cand = [f for f in pool
                if f["consensus"]["grade"] == grade
                and bool(f["consensus"]["flagged"]) == want_flag
                and f not in picks]
        cand.sort(key=lambda f: f["fact_uid"])
        rng("E2", grade, str(want_flag)).shuffle(cand)
        for _ in range(k):
            if not cand:
                break
            pick = min(cand, key=lambda f: (used_c[(f["stratum"], f["id"])],
                                            used_s[f["stratum"]], cand.index(f)))
            picks.append(pick)
            used_c[(pick["stratum"], pick["id"])] += 1
            used_s[pick["stratum"]] += 1
            cand.remove(pick)
    rng("E2", "order").shuffle(picks)
    out = []
    for f in picks[:BUDGET["E"][1]]:
        fact, correct = f["fact"], ""
        m = re.search(r"\(correct handling:\s*(.+?)\)\s*$", fact, re.S)
        if m:
            fact, correct = fact[:m.start()].strip(), m.group(1).strip()
        out.append({"f": f, "fact": fact, "correct": correct,
                    "complaint": complaints.get((f["stratum"], f["id"]), ""),
                    "excerpt": excerpt_around(transcripts[(f["stratum"], f["id"])],
                                              fact, 850)})
    return out


def section_f(b3, grid_flags, note_text):
    """10 notes where the pipeline judge's per-fact verdict and the monolithic grid judge
    part company. Blinded: the item never says which way."""
    caught, reverse, false_alarm = [], [], []
    for r in b3:
        miss = r["detail"]["missing_facts"]
        absent = [m for m in miss if m["verdict"] == "absent"]
        crit = [m for m in absent if m.get("severity") == "critical"]
        flagged = grid_flags.get(r["note_key"], [])
        if crit and not any(flagged):
            (false_alarm if r["note_role"] == "clean" else caught).append(
                {"r": r, "m": sorted(crit, key=lambda m: int(m["id"][1:]))[0]})
        elif not absent and any(flagged):
            part = [m for m in miss if m["verdict"] == "partial"]
            part.sort(key=lambda m: (m.get("severity") != "critical", int(m["id"][1:])))
            if part:
                reverse.append({"r": r, "m": part[0]})
    # one consultation appears at most once in the section: two items off one note is
    # near-duplicate reading, and the reader would spot the pairing.
    out = false_alarm[:1]                      # the per-fact rule's single false alarm
    seen = {c["r"]["consultation"] for c in out}
    r1 = rng("F2", "caught")
    for c in spread(caught, len(caught),
                    lambda c: (c["r"]["stratum"], c["r"].get("residual_level")), r1):
        if len(out) >= 7 or c["r"]["consultation"] in seen:
            continue
        out.append(c)
        seen.add(c["r"]["consultation"])
    r2 = rng("F2", "reverse")
    for c in spread(reverse, len(reverse), lambda c: c["r"]["stratum"], r2):
        if len(out) >= 10 or c["r"]["consultation"] in seen:
            continue
        out.append(c)
        seen.add(c["r"]["consultation"])
    for c in out:
        c["note"] = note_text.get(c["r"]["note_key"], "")
    rng("F2", "order").shuffle(out)
    stats = {"caught": len(caught), "reverse": len(reverse), "false_alarm": len(false_alarm),
             "total": len(caught) + len(reverse) + len(false_alarm)}
    return out[:BUDGET["F"][1]], stats


def section_g(sheets_by_src, tb_scenarios):
    """2 trap-blind scenarios rated for realism, 1 extracted sheet rated for fidelity -
    the first three of the 11 August draw, unchanged."""
    r = rng("F", "trapblind")
    tb = sorted(tb_scenarios, key=lambda s: s["id"])
    r.shuffle(tb)
    rr = rng("F", "primock")
    pool = sorted(sheets_by_src["primock"].values(), key=lambda rec: rec["id"])
    rr.shuffle(pool)
    return tb[:2], pool[:1]


# ------------------------------------------------------------------ writer
def build(out_path):
    items = J("master/sitting_wa_items.json")["items"]
    wa = J("results/w-a-state/wa_state.json")
    flags = J("master/wa_audit_flags.json")["flags"]
    ds = J("master/dataset_v2.json")
    built = {p["pair_id"]: p for p in J("master/omission_pairs_v2.json")["pairs"]}
    verif = {p["pair_id"]: p for p in J("master/omission_verification_v2.json")["pairs"]}
    audit = J("master/omission_audit_omission-audit-full-v2.json")
    sev = J("master/factorial_severity.json")
    rubric = open(os.path.join(HERE, "master/severity-rubric.md")).read()
    tb = J("master/trapblind_scenarios_critiqued.json")
    sheets_by_src = {src: {r["id"]: r for r in J(f"master/fact_sheets_{src}_core.json")}
                     for src in ("primock", "aci", "trapblind")}
    flags_by_note = defaultdict(list)
    for f in flags:
        flags_by_note[f["note"]].append(f)

    transcripts, _ = W.transcript_index()
    complaints = {}
    for src in ("trapblind", "authored_extracted", "primock", "aci"):
        for rec in J(f"master/fact_sheets_{src}_core.json"):
            st = "authored" if src == "authored_extracted" else src
            pc = rec.get("presenting_complaint")
            if not pc:
                mc = (rec.get("fact_sheet") or {}).get("must_contain") or []
                pc = mc[0].get("fact") if mc and isinstance(mc[0], dict) else None
            if pc:
                complaints[(st, rec["id"])] = squash(pc, 220)

    import taxonomy_common as T
    units, _skipped = T.note_units()
    notes_by_key = {u["note_key"]: u for u in units}

    confirm = J("master/arms_confirm_subset.json")["pairs"]
    note_text = {}
    for p in confirm:
        h = W.sha256_text(p["clean"])[:12]
        note_text[f"{p['stratum']}|{p['id']}|clean|{h}"] = p["clean"]
        note_text[f"{p['pair_id']}|err"] = p["errored"]
    b3 = JL("results/w2-pipeline/_state/confirm-B3.jsonl")
    keys = {r["note_key"] for r in b3}
    grid_flags = defaultdict(list)
    with open(os.path.join(HERE, "results/w2-ablation/_state/grid-main2.jsonl")) as fh:
        for line in fh:
            if '"FC-score-k8|' not in line:
                continue
            r = json.loads(line)
            if r["note_key"] in keys:
                grid_flags[r["note_key"]].append(r["flagged"])

    used_uids = {p["fact_uid"] for p in ds["pairs"] if p.get("fact_uid")}

    A, astats = section_a({(p["stratum"], p["id"]) for p in confirm})
    B = section_b(items, wa["notes"], flags_by_note)
    C = section_c(ds["factorial_pairs_dropped_unverified"], verif, built)
    D = section_d(audit, notes_by_key)
    E = section_e(sev, used_uids, transcripts, complaints)
    F, fstats = section_f(b3, grid_flags, note_text)
    Grealism, Gfid = section_g(sheets_by_src, tb)

    ids = {k: [] for k in BUDGET}
    ids["Q"] = ["Q1", "Q2", "Q3"]
    L = []                                   # the document, line by line
    w = L.append

    total_min = sum(m for _, _, m in BUDGET.values())
    total_items = sum(n for _, n, _ in BUDGET.values())
    w("# The sitting pack")
    w("")
    w(f"*Regenerated {BUILT} by `build_sitting_pack.py` from the study's final artifacts, "
      f"seed {SEED}. One sitting, {total_min} minutes, {total_items} items. Supersedes the "
      f"11 August pack, which is kept verbatim at `{ARCHIVED}` - same shape, but half its "
      f"sections sampled artifacts the study has since replaced.*")
    w("")
    w("This is the human layer of the paper. Everything else in the study is a model "
      "judging a model: the fact lists were written by one model and audited by another, "
      "the errored notes were built by a third and verified by a panel of two more, the "
      "failure taxonomy was found by one model and cut by two more, and the severity "
      "grades come from two rubric arms that disagree with each other one time in six. "
      "That stack "
      "is defensible only if a clinician has looked at a sample of it and said whether it "
      "is right. You are that clinician, disclosed in the paper as an author with the "
      "conflict of interest that implies - which is exactly why the sample has to be "
      "honest rather than flattering.")
    w("")
    w("Six things hang on your answers. Whether the fact list the deployed judge extracts "
      "at inference time is supported by the transcript it cites (A). Whether the auditor "
      "that gated every reference note called real defects and got them fixed (B). "
      "Whether the panel that threw out 44% of the complete-omission pairs was right to "
      "(C). Whether the two skeptics that cut 89.5% of the taxonomy's candidate findings "
      "were killing noise or killing findings (D) - that one decides whether the headline "
      "survival rate is quotable at all. Whether the severity axis the whole error surface "
      "is drawn on means anything to a doctor (E). And, on real cases, which of the two "
      "judge designs is right when they disagree about whether a fact is in a note (F).")
    w("")
    w("**Nothing here is trick-checked.** There is no hidden right answer and no scoring "
      "of you. Where the models disagreed with each other, that is said plainly. Where "
      "knowing would contaminate the answer you are not told: section E never shows a "
      "model's grade, and section F never says which judge said what, or whether the note "
      "in front of you is an original or an edited one.")
    w("")
    w("## How to answer")
    w("")
    w("Every item ends with its own answer line - **A1** followed by an arrow and the codes "
      "it accepts. Type your answer straight after the arrow as you go, or ignore the "
      "inline lines entirely and work down the **answer sheet at the end** - that block is "
      "the thing that gets read back, and it is the only part you have to fill in. One "
      "character is a complete answer. Add a dash and anything you want to say after it "
      "when the character is not enough; free comments are more useful than agonising over "
      "a borderline code.")
    w("")
    w("Skipping is allowed and is recorded as a skip, not a guess: leave it blank or put "
      "`?`. An item you cannot judge from what is printed is itself a finding about the "
      "pack - say so in the comment.")
    w("")
    w("| section | what you are doing | items | codes | mins |")
    w("|---|---|---|---|---|")
    for k, (name, n, mins) in BUDGET.items():
        w(f"| **{k}** {name} | {WHAT[k]} | {n} | {CODES[k][0]} | {mins} |")
    w("")
    w("---")
    w("")

    # ---------------------------------------------------------------- A
    name, n, mins = BUDGET["A"]
    w(f"## A. {name} - {n} items, ~{mins} min")
    w("")
    w(f"The pipeline judge (the strongest design in the study) begins by reading the "
      f"transcript alone and writing out the facts a note of that consultation should "
      f"contain, each with a quote from the transcript as its evidence. Everything "
      f"downstream - which facts are checked, which are graded critical, which absences "
      f"raise a flag - is built on that list, and unlike the fact sheets the corpus was "
      f"built from, this list is written at inference time by the thing we would actually "
      f"deploy. On the held-out confirmation run it wrote {astats['facts']:,} facts over "
      f"{astats['consultations']} consultations. These {n} are a seeded draw from them: "
      f"one fact per consultation, {n} consultations across the four strata, no filtering "
      f"for interestingness.")
    w("")
    w("**The question each time: does the quote, on its own, support the fact as stated?** "
      "Not whether the fact is true of the consultation, and not whether it is well "
      "worded - whether *this quote* carries *this claim*. An extra detail the quote does "
      "not contain (a laterality, a number, a duration, a denial turned into a positive) "
      "is the usual way this fails. The quotes are spans of the transcript or joins of two "
      "spans, lightly normalised, so a paraphrase that reads oddly is not by itself a "
      "failure.")
    w("")
    for i, p in enumerate(A, 1):
        iid = f"A{i}"
        ids["A"].append(iid)
        w(f"### {iid} - {p['stratum']} / {p['cid']} - extracted fact {p['pos']} of "
          f"{p['n_facts']}")
        w("")
        w(f"**Fact claimed:** {squash(p['f']['fact'])}")
        w("")
        w("**Evidence quote:**")
        w("")
        w(f"> {squash(p['f']['evidence'], 700)}")
        w(ans(iid, CODES["A"][1]))
    w("---")
    w("")

    # ---------------------------------------------------------------- B
    name, n, mins = BUDGET["B"]
    w(f"## B. {name} - {n} items, ~{mins} min")
    w("")
    w("Every reference note in the corpus was audited by the cross-family auditor (gpt-5.5, "
      "3 seeds) against its fact sheet. Where a flag carried a majority the note was sent "
      "back to be repaired, then re-audited until clean. 85 notes went through that loop, "
      "and those repaired notes are what every pair in the dataset is built from - an "
      "over-firing auditor and a bad repair both end up baked into the substrate. These 8 "
      "are stratified by stratum and by how heavily flagged they were, from single-flag "
      "notes up through the ones that needed two or three rounds.")
    w("")
    w("You are shown the flag the auditor raised, its reason, and the part of the "
      "**repaired** note that now covers it. **Two things to judge, one answer:** was the "
      "flag a real defect, and does the note now handle it properly? `y` if both, `n` if "
      "either fails - and say which in the comment, because they point at different fixes "
      "(a bad flag means the auditor is over-firing; a bad repair means the repair prompt "
      "is).")
    w("")
    for i, rec in enumerate(B, 1):
        iid = f"B{i}"
        ids["B"].append(iid)
        it, fl = rec["item"], rec["flags"]
        top = fl[0] if fl else None
        w(f"### {iid} - {it['note']} - {it['n_confirmed_flags_round0']} confirmed flag"
          f"{'s' if it['n_confirmed_flags_round0'] != 1 else ''}, "
          f"{it['rounds']} repair round{'s' if it['rounds'] != 1 else ''}")
        w("")
        if top:
            w(f"**Flag ({top['kind']}, {top['n_fail']}/{top['n_runs']} seeds):** "
              f"{squash(top['item'], 400)}")
            w("")
            w(f"**Auditor's reason:** {squash(top['reasons'][0], 320)}")
            if top.get("evidence"):
                w("")
                w(f"**Note text it pointed at:** {squash(top['evidence'], 240)}")
            if len(fl) > 1:
                others = "; ".join(squash(f["item"], 90) for f in fl[1:4])
                w("")
                w(f"**Also flagged on this note:** {others}"
                  f"{f' (+{len(fl) - 4} more)' if len(fl) > 4 else ''}")
        w("")
        w("**The repaired note, where it covers this:**")
        w("")
        w("```")
        subject = " ".join(filter(None, [(top or {}).get("item"), (top or {}).get("evidence")]))
        w(fence(excerpt_around(rec["note_text"], subject, 1000)))
        w("```")
        w(ans(iid, CODES["B"][1]))
    w("---")
    w("")

    # ---------------------------------------------------------------- C
    name, n, mins = BUDGET["C"]
    w(f"## C. {name} - {n} items, ~{mins} min")
    w("")
    w("The factorial dataset removes one mapped fact from a real note, either every "
      "mention of it (complete) or its home statement while a weaker trace survives "
      "(partial). A cross-family panel of two then checks each built pair on four "
      "semantic tests: does the deleted text really state the fact, is the fact now truly "
      "absent, did anything else go with it, and does the note still read naturally. **92 "
      "of the 277 built pairs were rejected** - 44% of the complete class - and that "
      "rejection rate is itself a result the paper reports: on a real clinical note, "
      "removing every trace of a fact often cannot be done without leaving a visible wound "
      "or taking a neighbour with it.")
    w("")
    w("Which makes the panel's standard load-bearing. If it is too strict the study is "
      "throwing away usable pairs and overstating how infeasible complete omission is; if "
      "it is too lax the dataset contains pairs that do not test what they claim. These 6 "
      "are drawn one per failure signature, across the five commonest ways a pair died - "
      "including one that failed several checks at once.")
    w("")
    w("**The question: was the panel right to reject this pair?** `y` = right, this is not "
      "a usable test item as built. `n` = it should have gone into the dataset. Judge the "
      "check it actually failed, and say in the comment if you would keep the pair under a "
      "different class (a complete omission that is really a partial one, say).")
    w("")
    gloss = {"truly_absent": "the fact is still recoverable from the note",
             "reads_naturally": "the note now reads as visibly cut about",
             "no_collateral_loss": "something else went missing with it",
             "spans_all_state_fact": "the deleted text did not state the whole fact",
             "primary_site_removed": "the deleted statement was not the fact's home statement",
             "residual_present": "no surviving mention where one was expected"}
    for i, rec in enumerate(C, 1):
        iid = f"C{i}"
        ids["C"].append(iid)
        v, b = rec["v"], rec["b"]
        fails = v.get("sem_fails") or []
        algo = (v.get("algo") or {}).get("failed") or []
        w(f"### {iid} - {rec['pair_id']} - {b['class']}, severity {b['severity']} - "
          f"rejected on {', '.join('`%s`' % f for f in fails + list(algo))}")
        w("")
        w(f"**Fact the edit was meant to remove:** {squash(b['fact'], 400)}")
        w("")
        w(f"**Edit made:** {squash(b.get('change') or '', 300)}")
        w("")
        w("**What left the note:**")
        w("")
        w("```")
        for l in diff_lines(b["clean"], b["errored"])[:8]:
            w(fence(l))
        w("```")
        if (b.get("residual") or {}).get("sites"):
            w("")
            w("**What construction knew still survived:**")
            w("")
            for s in b["residual"]["sites"][:3]:
                w(f"- *{s.get('strength', 'residual')}*, in {s.get('section') or 'the note'}: "
                  f"{squash(s.get('span') or s.get('quote') or '', 220)}")
        w("")
        w("**The note as it now reads:**")
        w("")
        w("```")
        w(fence(excerpt_around(b["errored"], b["fact"], 850)))
        w("```")
        w("")
        w("**Why the panel rejected it** (" + "; ".join(
            f"`{f}`: {gloss.get(f, 'see the verification record')}" for f in fails) + "):")
        w("")
        for role, run in sorted((v.get("sem_runs") or {}).items()):
            if not run.get("reason"):
                continue
            w(f"- *{role}*: {sentences(run['reason'], 2, 300)}")
        if algo:
            w(f"- *algorithmic pre-check*: failed {', '.join(algo)}")
        w(ans(iid, CODES["C"][1]))
    w("---")
    w("")

    # ---------------------------------------------------------------- D
    name, n, mins = BUDGET["D"]
    w(f"## D. {name} - {n} items, ~{mins} min")
    w("")
    w("The failure taxonomy ran a discovery pass over all 565 scribe notes and found "
      "13,678 candidate errors. Two skeptics from different model families then took each "
      "candidate with the note and the transcript in front of them and were told to refute "
      "it if refutation was at all defensible: both keep means verified, both refute means "
      "cut, a split goes to a tiebreak. **5,898 candidates reached them and 618 survived - "
      "10.5%.** Omission candidates survived at 6.1%, which is far below every published "
      "scribe-error study, and the whole question is whether that is scribes being better "
      "than the literature says or our panel being harsher than the literature's human "
      "reviewers.")
    w("")
    w("An impartial classifier already audited a stratified sample of 40 cut omission "
      "candidates and put a quarter of them in the bucket 'the fact is actually stated "
      "somewhere else in the note'. That is a model auditing models again. **These 10 are "
      "from that same sample and you are the check on it.** They are ordered randomly and "
      "you are not told which bucket the classifier put each one in.")
    w("")
    w("**The question: reading the note, is this a real omission the study should have "
      "counted?** `y` = a reportable omission, the panel was wrong to cut it. `n` = the "
      "cut was fair, whether because the note does say it elsewhere, or because it is not "
      "the sort of thing a note has to carry. Use your own standard for a note that has to "
      "be safe to hand over - not the study's fact sheets, which are not the instrument "
      "here. Where a note is too long to print whole you get the part covering the point "
      "and its full length; if you think the fact could be sitting in the part not shown, "
      "`?` and say so, because 'you cannot tell without the whole note' is exactly the "
      "problem the panel had.")
    w("")
    for i, rec in enumerate(D, 1):
        iid = f"D{i}"
        ids["D"].append(iid)
        u = notes_by_key[rec["note_key"]]
        w(f"### {iid} - {u['scribe']} note on {u['source']}/{u['id']}"
          f"{' (' + u['template'] + ' template)' if u['scribe'] == 'scribe_A' else ''}")
        w("")
        w(f"**What the finding claimed the note left out:** {squash(rec['description'], 500)}")
        w("")
        if rec.get("source_quote") and rec["source_quote"] != "-":
            w("**The transcript line behind it:**")
            w("")
            w(f"> {squash(rec['source_quote'], 500)}")
            w("")
        for part in re.split(r";\s*(?=constructor:|auditor:)", rec.get("panel_reasons") or ""):
            if not part.strip():
                continue
            role, _, body = part.partition(":")
            w(f"**Why the panel cut it ({role.strip()}):** {sentences(body, 2, 340)}")
            w("")
        body = fence(excerpt_around(
            u["note"], rec["description"] + " " + (rec.get("note_quote") or ""), 1200))
        whole = len(body) >= len(u["note"].strip()) - 2
        how = "printed in full" if whole else "the part covering this point; the rest is not printed"
        w(f"**The note ({len(u['note']):,} characters, {how}):**")
        w("")
        w("```")
        w(body)
        w("```")
        w(ans(iid, CODES["D"][1]))
    w("---")
    w("")

    # ---------------------------------------------------------------- E
    name, n, mins = BUDGET["E"]
    w(f"## E. {name} - {n} items, ~{mins} min")
    w("")
    w("Every omission pair in the dataset carries a severity grade, and that grade is the "
      "axis the whole error surface is drawn on: detection is reported by severity, the "
      "one decision rule in the study that separates detection from false alarms fires "
      "only on **critical** facts, and 'a critical fact is absent' is what a deployed "
      "version of this would page a human about. Two rubric arms from different model "
      "families graded all 880 facts against the rubric below; they split on 137 of them "
      "(15.6%) and the lower grade won. These 20 are drawn from the facts that actually "
      "back a pair in the evaluation set, balanced across the three grades.")
    w("")
    w("**The model grades are deliberately not shown**, and neither is which items the two "
      "arms split on: if you knew, your grade would not be independent, and independence "
      "is the only thing that makes this number worth reporting.")
    w("")
    w("The rubric is printed once, here, and applies to all 20.")
    w("")
    w("<details><summary><strong>The rubric (tap to open; it is the instrument, verbatim)"
      "</strong></summary>")
    w("")
    for line in rubric.strip().splitlines():
        # demote the rubric's own headings so they sit under section E rather than beside it
        w("###" + line if line.startswith("#") else line)
    w("")
    w("</details>")
    w("")
    w("For each item: read the excerpt, then grade **how much it would matter if a note of "
      "this consultation left this fact out**. Walk the decision procedure in order "
      "(action, safety, record quality, otherwise peripheral) and take the lower grade "
      "when genuinely torn.")
    w("")
    for i, rec in enumerate(E, 1):
        iid = f"E{i}"
        ids["E"].append(iid)
        f = rec["f"]
        w(f"### {iid} - {f['stratum']} / {f['id']}")
        w("")
        if rec["complaint"]:
            w(f"**The consultation:** {rec['complaint']}")
            w("")
        w(f"**The fact at risk:** {squash(rec['fact'], 600)}")
        w("")
        if rec["correct"]:
            w(f"**What a correct note would do:** {squash(rec['correct'], 500)}")
            w("")
        w("**Transcript:**")
        w("")
        w("```")
        w(fence(rec["excerpt"].strip()))
        w("```")
        w(ans(iid, CODES["E"][1]))
    w("---")
    w("")

    # ---------------------------------------------------------------- F
    name, n, mins = BUDGET["F"]
    w(f"## F. {name} - {n} items, ~{mins} min")
    w("")
    w("This is the study's central contrast, put in front of a doctor. One judge design "
      "reads the note as a whole and scores it - the standard LLM-as-judge shape, eight "
      "variants of it in our grid. The other extracts a fact list from the transcript "
      "first and then answers, per fact, whether the note carries it. On the same notes "
      "the two designs reach opposite conclusions often enough that the paper's main claim "
      "rests on which of them is right when they part.")
    w("")
    w("Each item is one fact and one note. **Is that fact captured in that note?** `y` "
      "captured (even loosely, even in different words, even implied strongly enough that "
      "the next clinician would not miss it), `n` not captured, `?` genuinely ambiguous - "
      "and `?` is a real answer here, because how much of this disagreement is genuine "
      "ambiguity is itself the finding.")
    w("")
    w("You are not told which judge said what, and you are not told whether the note is an "
      "original or one we edited. Some of these notes had a fact deliberately removed and "
      "some did not. The fact is quoted in the words the extractor used, which is part of "
      "what is under test: a fact phrased too specifically is a way to manufacture an "
      "omission that is not there.")
    w("")
    for i, rec in enumerate(F, 1):
        iid = f"F{i}"
        ids["F"].append(iid)
        r = rec["r"]
        w(f"### {iid} - {r['stratum']} / {r['consultation']} - fact {rec['m']['id']} of "
          f"{r['n_facts']}")
        w("")
        w(f"**The fact:** {squash(rec['m']['fact'], 400)}")
        w("")
        w(f"**The note ({len(rec['note']):,} characters, printed in full):**")
        w("")
        w("```")
        w(fence(rec["note"].strip()))
        w("```")
        w(ans(iid, CODES["F"][1]))
    w("---")
    w("")

    # ---------------------------------------------------------------- G
    name, n, mins = BUDGET["G"]
    w(f"## G. {name} - {n} items, ~{mins} min")
    w("")
    w("Two different worries, three items, and the section is deliberately thin because it "
      "is the one whose answers change least about the study. **G1-G2** are trap-blind "
      "scenarios - consultations written by a model to contain a specific difficulty, "
      "without the writer being told what the difficulty was. If they read as synthetic, "
      "every result measured on them is measured on a stylised object and the paper has to "
      "say so. **G3** is a real consultation (PriMock57) with the fact sheet a model "
      "extracted from it, and the worry is the opposite one: whether the extraction is a "
      "fair account of what was actually said.")
    w("")
    w("Excerpts are trimmed to a couple of minutes' reading from the clinical middle of "
      "each consultation - past the greeting, before the sign-off.")
    w("")
    w("**G1-G2, realism, 1-5:** would this pass as a transcript of a real UK primary care "
      "consultation? 5 = you would not know. 3 = plausible but a bit tidy. 1 = obviously "
      "written.")
    w("")
    for i, s in enumerate(Grealism, 1):
        iid = f"G{i}"
        ids["G"].append(iid)
        w(f"### {iid} - trap-blind: {s['id']}")
        w("")
        w(f"**Presenting complaint:** {squash(s.get('presenting_complaint') or '', 300)}")
        w("")
        w("```")
        w(fence(middle_window(s["transcript"], 2000)))
        w("```")
        w("")
        w("*Full transcript: `master/trapblind_scenarios_critiqued.json`*")
        w(ans(iid, CODES["G"][1]))
    w("")
    w("**G3, fidelity, 1-5:** is the extracted sheet a fair account of this consultation? "
      "5 = nothing important missed or invented. 3 = broadly right with a gap or a stretch. "
      "1 = it is describing a different consultation. Judge the sheet against the excerpt "
      "shown; a sheet item that covers a part of the consultation not printed here is not "
      "a mark against it.")
    w("")
    for i, rec in enumerate(Gfid, len(Grealism) + 1):
        iid = f"G{i}"
        ids["G"].append(iid)
        fs = rec["fact_sheet"]
        w(f"### {iid} - {rec.get('source', '?')}: {rec['id']}")
        w("")
        w("```")
        w(fence(middle_window(rec["transcript"], 1800)))
        w("```")
        w("")
        core = [m for m in fs["must_contain"] if m.get("scope") == "core"] or fs["must_contain"]
        w(f"**The extracted sheet** ({len(fs['must_contain'])} must-contain items, "
          f"{len(fs.get('salience_traps') or [])} salience traps; the core items and the "
          f"first traps shown):")
        w("")
        for m in core[:8]:
            w(f"- {squash(m['fact'], 200)}")
        for t in (fs.get("salience_traps") or [])[:3]:
            w(f"- *trap:* {squash(t['trap'], 200)}")
        w("")
        w(f"*Full sheet: `master/fact_sheets_{rec.get('source', '?')}_core.json` -> "
          f"`{rec['id']}`*")
        w(ans(iid, CODES["G"][1]))
    w("---")
    w("")

    # ---------------------------------------------------------------- Q
    name, n, mins = BUDGET["Q"]
    w(f"## Q. {name} - {n} questions, ~{mins} min")
    w("")
    w("Three things the finished study cannot settle by itself. A sentence each is plenty; "
      "a shrug is also an answer, and more useful than a manufactured position.")
    w("")
    w("### Q1 - the flag rule: is one missing critical fact the right alarm")
    w("")
    w("The only decision rule in the study that separates omission detection from false "
      "alarms is a predicate, not a score: **flag the note if any fact the pipeline graded "
      "critical comes back absent.** It detects 20.6% of omissions at a 2.1% false-alarm "
      "rate, where every threshold on an aggregate score sits at or near its own noise. "
      "Aggregation is what kills them - one absent fact out of 38 moves a weighted score "
      "by two or three percent and disappears into the between-note spread. **Is 'any "
      "single critical fact absent' the rule you would actually want a scribe monitor to "
      "fire on?** It says a note with one missing allergy is as flagworthy as a note "
      "missing five, and it says nothing at all about a note that drops six supporting "
      "facts. If you would rather it fired on a count, on particular classes of fact, or "
      "on severity-weighted mass instead, say so - the rule is one line of analysis code "
      "and every alternative can be scored on the same purchased verdicts.")
    w(ans("Q1", CODES["Q"][1]))
    w("### Q2 - the partial-strong blind spot, and whether to publish it as unsolved")
    w("")
    w("When a fact's home statement is deleted but an explicit trace of it survives "
      "elsewhere in the note - the clinically deceptive case, a note that looks complete "
      "and is not - **nothing in the study detects it.** Not the grid's best cell, not "
      "RAGAS, not the strongest hand-written prompt, not the pipeline, not any decision "
      "rule applied to any of them: paired scores of 0.47 to 0.56 against a chance of "
      "0.50, and 0 of 33 caught under the critical-fact rule. The proposal is to publish "
      "that as a named open problem with the cell reported rather than pooled away. **Is "
      "that the right call clinically, or is partial-strong the case you would most want a "
      "monitor to catch,** in which case a negative result on it is the paper's most "
      "important sentence and should be in the abstract rather than the limitations?")
    w(ans("Q2", CODES["Q"][1]))
    w("### Q3 - the rubric's tie-break-down rule")
    w("")
    w("The severity rubric says: when genuinely torn between two grades, take the lower "
      "one. It was written to stop grade inflation, and it does - but it also means every "
      "case the two model arms split on (137 of 880 facts) was resolved downward by "
      "construction, and it is part of why only 54 of 613 facts came out peripheral while "
      "the middle grade absorbed everything else. **Does taking the lower grade when torn "
      "match how you would grade a note in practice,** or does a doctor's instinct on a "
      "borderline case run the other way - if it might change what the next clinician "
      "does, treat it as though it would? The rule is the instrument, so if it is wrong "
      "the severity conditioning in every result moves with it.")
    w(ans("Q3", CODES["Q"][1]))
    w("---")
    w("")

    # ---------------------------------------------------------------- answer sheet
    w("## The answer sheet")
    w("")
    w("Copy this block, fill it in, send it back. Blank = skipped.")
    w("")
    w("```")
    w(f"sitting pack - {BUILT} - seed {SEED}")
    w("")
    for k in ("A", "B", "C", "D", "E", "F", "G"):
        nm, _, _ = BUDGET[k]
        w(f"# {k} - {nm}   [{CODES[k][0].replace('`', '')}]")
        for iid in ids[k]:
            w(f"{iid}: ")
        w("")
    w("# Q - open questions (a sentence each)")
    for iid in ids["Q"]:
        w(f"{iid}: ")
    w("```")
    w("")
    w("---")
    w("")

    # ---------------------------------------------------------------- coverage
    w("## What this pack leaves out")
    w("")
    w("The curation is a time budget, not a filter on inconvenient items. The full queues, "
      "and what was left in them:")
    w("")
    counts = Counter(i["kind"] for i in items)
    n_dropped = len(ds["factorial_pairs_dropped_unverified"])
    w("| queue | full size | in this pack | what is left there |")
    w("|---|---|---|---|")
    w(f"| Pipeline extractions (`results/w2-pipeline/_cache/facts_*.json`) | "
      f"{astats['facts']:,} facts over {astats['consultations']} consultations | 10 | "
      f"everything else, including the audit stage's own verdicts on the same facts "
      f"(`audit_*.json`), which drops the unsupported ones and grades the rest |")
    w(f"| Repaired notes (`master/sitting_wa_items.json`, `repaired_note_review`) | "
      f"{counts['repaired_note_review']} | 8 | {counts['repaired_note_review'] - 8} more, "
      f"all already re-audited clean; the flags behind each are in "
      f"`master/wa_audit_flags.json` |")
    w(f"| Rejected factorial pairs (`dataset_v2.json` -> "
      f"`factorial_pairs_dropped_unverified`) | {n_dropped} | 6, one per failure "
      f"signature | {n_dropped - 6} more, each with its failing checks and both panel "
      f"reasons in `master/omission_verification_v2.json` |")
    w(f"| Cut findings (`master/findings_verified_master.json`) | 5,280 refuted, of which "
      f"1,206 omissions; 40 audited in "
      f"`master/omission_audit_omission-audit-full-v2.json` | 10 of the 40 | the other 30 "
      f"audited refusals and the 5,240 that were never sampled |")
    w(f"| Severity grades (`master/factorial_severity.json`) | 880 graded facts, 137 with "
      f"the two arms split | 20 | the rest, including the 158 traps that carry both a "
      f"sheet importance and a fresh rubric grade |")
    w(f"| Pipeline-vs-grid disagreements (`confirm-B3.jsonl` x `grid-main2.jsonl`) | "
      f"{fstats['total']} notes where the two designs' own flag rules reach opposite "
      f"conclusions ({fstats['caught']} the pipeline flags and the grid does not, "
      f"{fstats['reverse']} the reverse, {fstats['false_alarm']} clean note the pipeline "
      f"rule fires on) | 10 | the rest, reproducible "
      f"from the two stores "
      f"with `w2_pipeline_analyze.py` |")
    w(f"| Realism / fidelity | 10 planned for the sitting (6 authored + 4 extracted) | 3 | "
      f"the other 7; this is the section the study leans on least |")
    w("")
    w("Three things were left out on purpose rather than for time. **The MEDEC anchor is "
      "not in here**: its labels are already physician-written, so a second clinician "
      "rating them measures nothing the external anchor does not already give. **Nothing "
      "asks you to score a scribe's note against a reference** - that is the benchmark's "
      "job, and one human doing it once measures nothing. And **the 11 August pack's "
      "question about the ground-truth gate's pre-check is retired**: that pre-check has "
      "since been superseded by the factorial "
      "builder's own "
      "algorithmic gate, and the decision no longer changes anything downstream.")
    w("")
    w(f"*Regenerate with `python3 sittings/build_sitting_pack.py` (seed {SEED}, "
      f"deterministic). The previous cut is at `{ARCHIVED}`.*")

    text = escape_tildes("\n".join(L)) + "\n"
    tmp = out_path + ".tmp"
    open(tmp, "w").write(text)
    os.replace(tmp, out_path)
    return text, ids


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()
    text, ids = build(args.out)
    n = sum(len(v) for v in ids.values())
    print(f"{os.path.relpath(args.out, HERE)} - {n} items, {len(text):,} chars, "
          f"~{sum(m for _, _, m in BUDGET.values())} min")
    for k, (name, want, mins) in BUDGET.items():
        got = len(ids[k])
        print(f"  {k} {name:34} {got:2}/{want:2} items  ~{mins:2} min"
              f"{'   <-- SHORT' if got < want else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
