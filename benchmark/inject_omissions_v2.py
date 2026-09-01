"""inject_omissions_v2.py - build COMPLETE and PARTIAL omission pairs from the site map.

Part of the graded-omission design correction. The v1 injector removed ONE span and
called the result an
omission. On a redundant clinical note that is often a PARTIAL omission: the fact
survives as a paraphrase somewhere else. This builder makes the distinction explicit
and constructs both sides of it from `master/fact_sites.json`:

  omit-complete   remove EVERY mapped site of one fact, and nothing else.
                  Single edit = a single FACT (which may live in several places),
                  not a single span. This class does not exist in the frozen corpus:
                  every frozen omit pair is a single-span deletion that happened to
                  leave no residual, i.e. a single-SITE fact.
  omit-partial    remove the PRIMARY site only; the residual mentions stay exactly
                  where they were, and their count, sections and strengths are
                  recorded as the pair's residual metadata.

Both are built from the SAME fact in the SAME consultation wherever the fact has >= 2
sites, giving a MATCHED contrast: identical note, identical fact, identical severity,
one difference - whether a residual survives. That pairing is what makes the residual
prediction testable by exact McNemar rather than by eyeball.

Construction is deterministic where it can be. A site whose span is unit-aligned (a
whole sentence or a whole bullet line - most of them) is deleted by exact character
span: no model call, no drift, byte-reproducible. Only a sub-sentence span needs the
constructor (models.lock `constructor` = anthropic/claude-opus-5 over OpenRouter) to
repair the grammar of the sentence it leaves behind, and that repair is accepted only
if it changed nothing but the wound. Severity is graded ONCE per fact and shared by the
matched pair (the fact is the same; what differs is the residual, which is recorded
separately, not folded into severity).

Nothing here touches master/pairs_master_frozen.json or any frozen artefact.

Usage:
  python3 benchmark/inject_omissions_v2.py --smoke --allow-dirty -y
  python3 benchmark/inject_omissions_v2.py --full --workers 4 -y
  python3 benchmark/inject_omissions_v2.py --report          # free: rebuild the artefact
Exit: 0 complete | 2 work left (rerun) | 3 budget stop.
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import argparse, json, os, random, re, sys, threading, unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from common import HERE, Run, resolve_model
import hard_negatives_master as HNM
import omission_sites as OS
import w_a_master as WA

SPEC = OS.SPEC
EXPERIMENT = "wd-omission-inject"
MASTER = OS.MASTER
PAIRS_FILE = os.path.join(MASTER, "omission_pairs_v2.json")
STATE_FILE = os.path.join(MASTER, "omission_v2_state.json")

SEED = OS.SEED
CLASSES = ("omit-complete", "omit-partial")
MAX_TIDY_WORDS = 12          # a grammar repair may not move more than this many word tokens
MAX_CLEANUP_WORDS = 30       # ...and may not DROP more than this many beyond the deletion
MAX_CLEANUP_UNITS = 3        # ...spread over at most this many sentences / bullet lines
DATASET_VERSION = "factorial-v1"
ALLOC_FILE = os.path.join(MASTER, "factorial_allocation.json")
LEVEL_TAG = {"complete": "omitc", "partial-strong": "omitps", "partial-weak": "omitpw"}


# ---------------------------------------------------------------- prompts (driver additions)
# TYPES["omit"] and the severity rubric prompt are imported VERBATIM from the v1 instrument
# (hard_negatives_balanced / hard_negatives_master) so the new pairs are graded on the same
# scale as the frozen ones. TIDY is the only new construction prompt, and it is a copy-edit
# instruction, not an injection instruction - it is never allowed to decide WHAT is removed.
TIDY = """A clinical note had one or more spans deleted from it deliberately, as part of a research
dataset. The deletion cut mid-sentence in at least one place, so the text now reads
awkwardly. Repair ONLY that: make the affected sentence(s) grammatical again.

Absolute constraints:
- Do NOT restore, reword or hint at anything that was deleted. The deleted content must stay
  gone.
- Do NOT add any new clinical content, and do NOT remove anything that is still there.
- Every sentence that reads correctly already must be reproduced VERBATIM, character for
  character, including headings, line breaks, punctuation and capitalisation.
- The only permitted changes are the minimal ones that fix the grammar of the wound: dropping
  a stranded conjunction or comma, closing a sentence, fixing a list separator.

SPANS THAT WERE DELETED (they must not come back):
{removed}

NOTE AFTER DELETION (repair this):
{note}

Return ONLY JSON {"note":"<the repaired note>"}."""

# CLEANUP replaces TIDY for the factorial build. The difference is one permission: TIDY could
# only re-punctuate, so a deletion that stranded a fragment of the SAME fact - a dose with no
# drug left in front of it, "which is why he stopped it" with nothing to refer back to - had
# to be rejected whole (the one smoke failure was exactly that: plan item 6 left reading
# "135 mg three times daily before meals"). Those fragments are part of the fact being
# removed, so removing them too is the same single edit, not a second one. Every such removal
# is logged as a collateral edit and the pair still has to pass `reads_naturally` and
# `no_collateral_loss` afterwards.
CLEANUP = """A clinical note had one or more spans deleted from it deliberately, as part of a research
dataset: every place the note stated ONE clinical fact was removed. The deletion cut
mid-sentence in at least one place, so the text now reads awkwardly, and it may have left
behind a FRAGMENT of the removed fact that no longer makes sense on its own - a dose with no
drug in front of it, a laterality with no finding, a clause referring back to something that
is now gone.

Repair the note so it reads as a note that simply never recorded that fact. You may:
- fix the grammar of the wound: drop a stranded conjunction, comma or connective, close a
  sentence, fix a list separator;
- DELETE a leftover fragment that belonged to the removed fact and is now meaningless or
  misleading on its own.

Absolute constraints:
- Do NOT restore, reword or hint at anything that was deleted. The deleted content must stay
  gone.
- Do NOT add ANY new words or clinical content. You may only delete and re-punctuate.
- Do NOT delete anything that stands on its own as a clinical statement - another finding,
  another drug, another plan item, a different piece of history. Those are other facts and
  they must survive intact. If you are unsure whether a fragment belongs to the removed fact,
  LEAVE IT.
{keep}- Every sentence that reads correctly already must be reproduced VERBATIM, character
  for character, including headings, line breaks, punctuation and capitalisation.

SPANS THAT WERE DELETED (they must not come back):
{removed}

NOTE AFTER DELETION (repair this):
{note}

Return ONLY JSON {"note":"<the repaired note>"}."""

KEEP_CLAUSE = """- This sentence MUST survive exactly as it is, character for character - it is a
  deliberate part of the dataset:
{keep_spans}
"""

PROMPTS = {"TIDY": TIDY, "CLEANUP": CLEANUP,
           "TYPE_omit (verbatim import)": HNM.TYPES["omit"],
           "RUBRIC_GRADE (verbatim import)": HNM.RUBRIC_GRADE}


def prompts():
    """`PROMPTS` with the rubric text included. Needs the rubric file - see
    `hard_negatives_master.read_rubric` for where it is released."""
    return dict(PROMPTS, **{"severity_rubric_md (verbatim import)": HNM.RUBRIC})


# ---------------------------------------------------------------- targeting
def slug(fact_key):
    return fact_key.replace("|", "")


def eligible_facts(cons):
    """Facts in this consultation that can carry a construction, in the map's own rank
    order (the map is written in `omission_sites.rank_facts` order and preserves it).

    A fact is eligible when every one of its sites was located exactly once in the note
    (`span_found`), the sites do not overlap each other, and it has a primary site.
    """
    out = []
    for f in cons["facts"]:
        if f["n_sites"] < 1 or f.get("primary_site_id") is None:
            continue
        if any(not s["span_found"] for s in f["sites"]):
            continue
        spans = sorted((s["char_span"] for s in f["sites"]), key=lambda c: c[0])
        if any(spans[i][1] > spans[i + 1][0] for i in range(len(spans) - 1)):
            continue   # overlapping sites: the deletion arithmetic is not well defined
        out.append(f)
    return out


def pick_targets(cons, per_consult, seeded=True):
    """Which facts this consultation contributes. Multi-site facts only: a single-site
    fact produces exactly the pair class the frozen corpus already has 93 of, and no
    partial pair at all. Consultations with no multi-site eligible fact contribute
    nothing and are counted - that count IS the redundancy finding."""
    multi = [f for f in eligible_facts(cons) if f["n_sites"] >= 2]
    if not multi:
        return []
    if seeded and per_consult < len(multi):
        # rank order already encodes the injector's preference; the seeded shuffle only
        # breaks ties among facts the ranking cannot separate
        return multi[:per_consult]
    return multi[:per_consult] if per_consult else multi


# ---------------------------------------------------------------- deletion
def delete_spans(note, spans):
    """Remove exact character spans. Deterministic, and LOCAL: every line the deletion does
    not touch comes out byte-identical.

    Two refinements the first cut needed (caught in the smoke):
      - line-aware. When a span IS the whole content of its line, the line's newline goes
        with it. Otherwise deleting "No breathlessness" leaves a blank line exactly where the
        content was - a visible scar that tells a judge something was cut.
      - no global tidy. Whole-text whitespace regexes rewrote untouched lines (they stripped a
        trailing space off a heading three paragraphs away), which shows up in the diff as an
        edit we never made. Repair is confined to the wound.
    `spans` = [[start, end], ...] non-overlapping, in any order.
    """
    text = unicodedata.normalize("NFC", note)
    expanded = []
    for st, en in spans:
        ls = text.rfind("\n", 0, st) + 1
        le = text.find("\n", en)
        le = len(text) if le == -1 else le
        if not text[ls:st].strip() and not text[en:le].strip():
            expanded.append([ls, min(le + 1, len(text))])     # the line and its newline
        else:
            expanded.append([st, en])
    expanded.sort()
    merged = []
    for s, e in expanded:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    for s, e in reversed(merged):
        head, tail = text[:s], text[e:]
        if head.endswith(" ") and tail[:1] in (" ", ".", ",", ";"):
            head = head.rstrip(" ")                            # close a mid-sentence wound
        text = head + tail
    return text.strip()


NUM_LINE = re.compile(r"^(\s*)(\d+)([.)])(\s)")


def renumber_lists(clean, errored):
    """Close the numbering gap a deleted list item leaves behind.

    A plan that runs 1, 2, 3, 4, 5, 7 tells the reader something was taken out, and the smoke
    caught a judge-side check calling exactly that ("the errored plan has a visible numbering
    gap from 5 to 7"). Renumbering is the one repair allowed to touch a line the deletion did
    not, and only ever the number at its start. Applied only when the clean note's numbering
    was itself a clean 1..n, so a note that was already odd is left alone.
    """
    clean_nums = [int(m.group(2)) for m in
                  (NUM_LINE.match(l) for l in clean.split("\n")) if m]
    if not clean_nums or clean_nums != list(range(1, len(clean_nums) + 1)):
        return errored, False
    lines, changed = errored.split("\n"), False
    n = 0
    for i, line in enumerate(lines):
        m = NUM_LINE.match(line)
        if not m:
            continue
        n += 1
        if int(m.group(2)) != n:
            lines[i] = f"{m.group(1)}{n}{m.group(3)}{m.group(4)}{line[m.end():]}"
            changed = True
    return "\n".join(lines), changed


def wound_is_clean(sites):
    """True when every removed span is unit-aligned, so plain deletion leaves grammatical
    text and no model call is needed."""
    return all(s.get("unit_aligned") for s in sites)


def tidy_ok(rough, tidied, removed_spans):
    """Accept a constructor grammar repair only if it (a) reproduces everything else
    verbatim - at most MAX_TIDY_WORDS word tokens moved in total - and (b) did not smuggle
    any deleted content back in."""
    if not tidied or not tidied.strip():
        return False, "empty"
    w = WA.word_diff_stats(rough, tidied)
    if w["changed_words"] > MAX_TIDY_WORDS:
        return False, f"moved {w['changed_words']} word tokens (max {MAX_TIDY_WORDS})"
    low = OS.normlow(tidied)
    for sp in removed_spans:
        toks = [t for t in OS.normlow(sp).split() if len(t) > 3]
        if toks and sum(1 for t in toks if t in low) / len(toks) > 0.8:
            return False, "a deleted span came back"
    return True, None


def _content_words(text):
    return [w for w in (t.strip(".,;:()[]\"'!?-") for t in OS.normlow(text).split()) if w]


def cleanup_ok(rough, cleaned, removed_spans, keep_spans):
    """Accept a collateral-cleanup repair only if it is a pure SUBTRACTION.

    Deliberately looser than tidy_ok in one direction and tighter in another: the repair may
    drop up to MAX_CLEANUP_WORDS words across at most MAX_CLEANUP_UNITS sentences (that is the
    stranded-fragment allowance), but it may not introduce a single word that was not already
    there, and it may not touch a span the design needs to survive. Returns
    (ok, reason, collateral) where collateral describes exactly what the cleanup removed, for
    the log."""
    if not cleaned or not cleaned.strip():
        return False, "empty", None
    low = OS.normlow(cleaned)
    for sp in removed_spans:
        toks = [t for t in OS.normlow(sp).split() if len(t) > 3]
        if toks and sum(1 for t in toks if t in low) / len(toks) > 0.8:
            return False, "a deleted span came back", None
    for sp in keep_spans or []:
        if OS.normlow(sp) not in low:
            return False, "the surviving residual span was damaged", None
    before, after = Counter(_content_words(rough)), Counter(_content_words(cleaned))
    added = after - before
    if added:
        return False, f"introduced words not in the note: {sorted(added)[:6]}", None
    dropped = sum((before - after).values())
    if dropped > MAX_CLEANUP_WORDS:
        return False, f"dropped {dropped} words (max {MAX_CLEANUP_WORDS})", None
    import difflib
    uc, ue = WA.units_of(rough), WA.units_of(cleaned)
    gone, changed = [], 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, uc, ue).get_opcodes():
        if tag == "equal":
            continue
        changed += max(i2 - i1, j2 - j1)
        for k in range(i1, i2):
            gone.append(uc[k])
    if changed > MAX_CLEANUP_UNITS:
        return False, f"touched {changed} units (max {MAX_CLEANUP_UNITS})", None
    return True, None, {"words_dropped": dropped, "units_touched": changed,
                        "units_before": [u[:200] for u in gone],
                        "units_after": [u[:200] for u in ue if u not in uc]}


# ---------------------------------------------------------------- construction
def build_variant(cons, fact, level, budget, keep_site_ids=None):
    """One pair record (unverified). Returns (rec, error).

    `level` is the residual level - complete / partial-strong / partial-weak (or the legacy
    `partial`, which keeps every non-primary site). `keep_site_ids` names the sites that must
    SURVIVE; everything else mapped for this fact is deleted. The primary is never in it, so
    every class removes the note's home statement of the fact and only the residual differs.
    """
    sites = fact["sites"]
    pid_primary = fact["primary_site_id"]
    klass = "omit-complete" if level == "complete" else "omit-partial"
    if level == "complete":
        keep = []
    elif keep_site_ids is None:                       # legacy: primary only
        keep = [i for i in range(len(sites)) if i != pid_primary]
    else:
        keep = [i for i in range(len(sites)) if sites[i]["site_id"] in set(keep_site_ids)]
    if level != "complete" and not keep:
        return None, "no surviving site for a partial pair"
    if pid_primary in keep:
        return None, "the primary site cannot be the survivor"
    removed = [i for i in range(len(sites)) if i not in keep]
    rm_sites = [sites[i] for i in removed]
    keep_spans = [sites[i]["span"] for i in keep]
    rough = delete_spans(cons["note"], [s["char_span"] for s in rm_sites])
    rough, renumbered = renumber_lists(cons["note"], rough)
    method, tidy_note, collateral = "programmatic_span_delete", None, None
    if not wound_is_clean(rm_sites):
        keep_clause = (KEEP_CLAUSE.replace(
            "{keep_spans}", "\n".join(f"    {s}" for s in keep_spans)) if keep_spans else "")
        obj, meta = OS.call_json(
            budget, "tidy",
            CLEANUP.replace("{removed}", "\n".join(f"- {s['span']}" for s in rm_sites))
                   .replace("{keep}", keep_clause).replace("{note}", rough),
            OS.MAPPER_ROLE, OS.MAPPER_EFFORT)
        cand = (obj or {}).get("note") if isinstance(obj, dict) else None
        ok, why, coll = (cleanup_ok(rough, cand, [s["span"] for s in rm_sites], keep_spans)
                         if cand else (False, "no output", None))
        if ok:
            rough, collateral = cand.strip(), coll
            method = ("constructor_cleanup" if coll and coll["words_dropped"]
                      else "constructor_tidy")
        else:
            method, tidy_note = "programmatic_span_delete_cleanup_rejected", why
    if OS.normlow(rough) == OS.normlow(cons["note"]):
        return None, "deletion produced an identical note"

    sections = [s["section"] for s in rm_sites]
    if level == "complete":
        change = (f"Removed every mention of one fact from the note - {len(rm_sites)} site(s) "
                  f"in {', '.join(sorted(set(sections)))}. Nothing else changed.")
    else:
        change = (f"Removed the note's primary statement of one fact (section "
                  f"{sites[pid_primary]['section']}), and any other mention of it except one; "
                  f"{len(keep)} residual mention(s) of the same fact left untouched.")
    residual = [{"site_id": sites[i]["site_id"], "span": sites[i]["span"],
                 "section": sites[i]["section"], "strength": sites[i]["strength"]}
                for i in keep]
    strength_rank = {g: r for r, g in enumerate(OS.STRENGTHS)}
    rec = {
        "pair_id": f"{cons['key']}|{LEVEL_TAG.get(level, 'omitp')}|{slug(fact['fact_key'])}",
        "class": klass, "residual_level": level, "type": "omit",
        "dataset_version": DATASET_VERSION,
        "stratum": cons["stratum"], "id": cons["id"], "key": cons["key"],
        "split": cons["split"],
        "fact_uid": f"{cons['key']}|{fact['fact_key']}", "fact_key": fact["fact_key"],
        "fact_kind": fact["kind"], "fact": fact["fact"],
        "clean": cons["note"], "clean_sha": cons["clean_sha"], "errored": rough,
        "change": change,
        "what": fact.get("trap") or fact["fact"],
        "n_sites_total": len(sites),
        "removed_site_ids": [sites[i]["site_id"] for i in removed],
        "removed_spans": [s["span"] for s in rm_sites],
        "removed_char_spans": [s["char_span"] for s in rm_sites],
        "removed_sections": sections,
        "primary_site_id": pid_primary,
        "residual": {"n_surviving": len(residual), "sites": residual,
                     "max_strength": min((r["strength"] for r in residual),
                                         key=lambda g: strength_rank.get(g, 9), default=None),
                     "sections": sorted({r["section"] for r in residual})},
        "target": ({"kind": "trap", "index": fact["index"], "trap": fact.get("trap"),
                    "correct_handling": fact.get("correct_handling"),
                    "mode": fact.get("mode"), "importance": fact.get("importance")}
                   if fact["kind"] == "trap" else
                   {"kind": "must_contain", "index": fact["index"], "fact": fact["fact"],
                    "load_bearing": fact.get("load_bearing")}),
        "kept_site_ids": [sites[i]["site_id"] for i in keep],
        "construction": {"method": method, "tidy_reject_reason": tidy_note,
                         "list_renumbered": renumbered,
                         "collateral_cleanup": collateral,
                         "unit_aligned_sites": sum(1 for s in rm_sites if s.get("unit_aligned")),
                         "spec": SPEC},
        "provenance": {
            "builder": "inject_omissions_v2.py",
            "site_map": "master/fact_sites.json",
            "constructor": (resolve_model(OS.MAPPER_ROLE)["resolved"]
                            if method.startswith("constructor_") else None),
            "constructor_effort": (OS.MAPPER_EFFORT if method.startswith("constructor_")
                                   else None),
            "deterministic": not method.startswith("constructor_")},
    }
    return rec, None


def grade_severity(cons, fact, budget):
    """One severity grade per FACT, shared by its matched complete/partial pair.

    Trap targets take the trap's own importance grade (as v1 did). must_contain targets are
    rubric-graded by one constructor call on the verbatim RUBRIC_GRADE prompt. The grade
    describes the FACT going missing; the residual is recorded separately and deliberately
    NOT folded in - how much a surviving paraphrase rescues the reader is the thing the
    experiment measures, so baking it into severity would beg the question.
    """
    if fact["kind"] == "trap" and fact.get("importance"):
        return {"severity": fact["importance"], "severity_source": "trap_importance"}
    prompt = HNM.RUBRIC_GRADE.format(
        rubric=HNM.RUBRIC, t=cons["transcript"][:40000], typ="omit",
        change=f"Removed the fact from the note entirely: {fact['fact']}",
        what=fact["fact"])
    obj, meta = OS.call_json(budget, "severity", prompt, OS.MAPPER_ROLE, OS.MAPPER_EFFORT)
    if isinstance(obj, dict) and obj.get("importance") in ("critical", "supporting", "peripheral"):
        return {"severity": obj["importance"], "severity_source": "rubric_graded",
                "severity_rationale": obj.get("rationale"),
                "severity_test_fired": obj.get("test_fired")}
    return {"severity": None, "severity_source": "grade_failed", "severity_error": meta}


# ---------------------------------------------------------------- driver
_lock = threading.RLock()   # see omission_sites._lock - save_state() re-enters it


def load_state():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {"created": OS.now_utc(), "spend": {"cost_usd": 0.0, "calls": 0, "by_stage": {}},
            "pairs": {}, "severity": {}, "skipped": {}}


def save_state(state):
    with _lock:
        tmp = STATE_FILE + ".tmp"
        json.dump(state, open(tmp, "w"), ensure_ascii=False, indent=1)
        os.replace(tmp, STATE_FILE)


def load_sites(keys=None):
    """The site map, with each consultation's transcript joined back on (the map carries the
    NOTE, because every span offset points into it; transcripts stay in the fact sheets and
    are only needed for the severity grade)."""
    if not os.path.exists(OS.SITES_FILE):
        sys.exit(f"no site map at {OS.SITES_FILE} - run omission_sites.py first")
    doc = json.load(open(OS.SITES_FILE))
    cons = doc["consultations"]
    if keys:
        cons = [c for c in cons if c["key"] in set(keys)]
    tx = {r["key"]: r["transcript"] for r in OS.consultations(("eval", "gepa_dev"))}
    for c in cons:
        assert WA.sha16(c["note"]) == c["clean_sha"], \
            f"{c['key']}: site map note does not match its recorded sha - remap"
        c["transcript"] = tx.get(c["key"], "")
    return doc, cons


def plan_jobs(cons_rows, state):
    """The build queue from master/factorial_allocation.json: one job per planned pair, with
    the severity the allocator already graded and the site that has to survive."""
    if not os.path.exists(ALLOC_FILE):
        sys.exit(f"no allocation at {ALLOC_FILE} - run omission_factorial.py --allocate first")
    alloc = json.load(open(ALLOC_FILE))
    by_key = {c["key"]: c for c in cons_rows}
    jobs = []
    for row in alloc["plan"]:
        cons = by_key.get(row["key"])
        if cons is None:
            continue
        fact = next((f for f in cons["facts"] if f["fact_key"] == row["fact_key"]), None)
        if fact is None:
            state["skipped"][row["pair_id"]] = "fact not in the site map"
            continue
        if row["pair_id"] in state["pairs"]:
            continue
        jobs.append((cons, fact, row))
    return jobs, alloc


def legacy_jobs(cons_rows, state, args):
    """The pre-factorial queue: every multi-site target fact, both classes. Kept so the smoke
    path still runs; the factorial build uses --plan."""
    jobs = []
    for cons in cons_rows:
        targets = pick_targets(cons, args.targets_per_consult)
        if not targets:
            state["skipped"][cons["key"]] = "no multi-site eligible fact"
            continue
        for fact in targets:
            for klass in (CLASSES if not args.classes else tuple(args.classes.split(","))):
                level = "complete" if klass == "omit-complete" else "partial"
                pid = f"{cons['key']}|{LEVEL_TAG.get(level, 'omitp')}|{slug(fact['fact_key'])}"
                if pid in state["pairs"]:
                    continue
                jobs.append((cons, fact, {"pair_id": pid, "residual_level": level,
                                          "keep_site_id": None}))
    return jobs


def build(cons_rows, state, budget, args):
    alloc = None
    if args.plan:
        jobs, alloc = plan_jobs(cons_rows, state)
    else:
        jobs = legacy_jobs(cons_rows, state, args)
    print(f"pairs to build: {len(jobs)} over {len({c['key'] for c, _, _ in jobs})} "
          f"consultations | already built: {len(state['pairs'])} | budget "
          f"${budget.spent:.2f}/${budget.cap:.0f}", flush=True)
    if args.chunk:
        jobs = jobs[:args.chunk]

    def one(job):
        cons, fact, row = job
        if not budget.ok():
            return
        level = row["residual_level"]
        keep = ([row["keep_site_id"]] if row.get("keep_site_id") is not None
                else (None if level in ("partial", "complete") else []))
        rec, err = build_variant(cons, fact, level, budget, keep)
        if rec is None:
            with _lock:
                state["skipped"][row["pair_id"]] = err
            print(f"  {row['pair_id']:52} {level:15} SKIPPED: {err}", flush=True)
            return
        uid = rec["fact_uid"]
        if alloc is not None:
            rec.update({k: row[k] for k in
                        ("severity", "severity_source", "severity_flagged", "cell",
                         "matched_levels", "bucket")
                        if k in row})
            for k in ("severity_arms", "severity_rationale", "severity_test_fired",
                      "audit_rubric_grade", "audit_agrees"):
                if k in row:
                    rec[k] = row[k]
        else:
            with _lock:
                sev = state["severity"].get(uid)
            if sev is None:
                sev = grade_severity(cons, fact, budget)
                with _lock:
                    state["severity"][uid] = sev
            rec.update(sev)
        with _lock:
            state["pairs"][rec["pair_id"]] = rec
            save_state(state)
        print(f"  {rec['pair_id']:52} {level:15} sites {len(rec['removed_site_ids'])}/"
              f"{rec['n_sites_total']} resid {rec['residual']['n_surviving']}"
              f"/{rec['residual']['max_strength']} [{rec['construction']['method']}] "
              f"{rec.get('severity')} ${budget.spent:.2f}", flush=True)

    if jobs:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(one, jobs))
    save_state(state)
    if args.plan:
        left, _ = plan_jobs(cons_rows, state)
        return len(left)
    return len(legacy_jobs(cons_rows, state, args))


def write_pairs_file(doc, cons_rows, state, args):
    pairs = [p for p in state["pairs"].values()
             if p["key"] in {c["key"] for c in cons_rows}]
    pairs.sort(key=lambda p: p["pair_id"])
    # matched set = every pair built from the SAME fact at a different residual level. Same
    # note, same fact, same severity; the residual is the only thing that moved, which is what
    # makes the contrast an exact within-fact one.
    by_uid = {}
    for p in pairs:
        by_uid.setdefault(p["fact_uid"], []).append(p["pair_id"])
    for p in pairs:
        mates = [q for q in by_uid[p["fact_uid"]] if q != p["pair_id"]]
        p["matched_pair_ids"] = mates
        p["matched_pair_id"] = mates[0] if len(mates) == 1 else None
    comp = [p for p in pairs if p["class"] == "omit-complete"]
    part = [p for p in pairs if p["class"] == "omit-partial"]
    out = {
        "generated_utc": OS.now_utc(), "spec": SPEC,
        "what": "graded-omission pairs: omit-complete (every site of a fact removed) and "
                "omit-partial (primary site removed, residuals left) built from "
                "master/fact_sites.json over the FROZEN clean notes. ADDITIVE - the frozen "
                "281-pair artefact is untouched and its pre-registered analysis is unaffected.",
        "site_map": {"file": "master/fact_sites.json",
                     "generated_utc": doc["generated_utc"],
                     "prompt_sha256_16": doc["prompt_sha256_16"]},
        "models": {"constructor": {"role": OS.MAPPER_ROLE, **resolve_model(OS.MAPPER_ROLE),
                                   "reasoning_effort": OS.MAPPER_EFFORT,
                                   "used_for": "sub-sentence wound repair + must_contain "
                                               "severity grades only; unit-aligned deletions "
                                               "are programmatic and need no call"}},
        "prompt_sha256_16": {k: WA.sha16(v) for k, v in prompts().items()},
        "params": {"seed": SEED, "targets_per_consult": args.targets_per_consult,
                   "smoke": bool(args.smoke)},
        "dataset_version": DATASET_VERSION,
        "counts": {
            "pairs": len(pairs), "by_class": dict(Counter(p["class"] for p in pairs)),
            "by_residual_level": dict(Counter(p.get("residual_level") for p in pairs)),
            "by_cell": dict(Counter(p.get("cell") for p in pairs)),
            "collateral_cleanups": sum(1 for p in pairs
                                       if (p["construction"].get("collateral_cleanup") or {})
                                       .get("words_dropped")),
            "collateral_words_dropped": sum(
                (p["construction"].get("collateral_cleanup") or {}).get("words_dropped", 0)
                for p in pairs),
            "by_stratum": dict(Counter(p["stratum"] for p in pairs)),
            "by_split": dict(Counter(p["split"] for p in pairs)),
            "matched_pairs": sum(1 for p in comp if p.get("matched_pair_id")),
            "consultations": len({p["key"] for p in pairs}),
            "consultations_without_a_multi_site_fact": len(state["skipped"]),
            "construction_method": dict(Counter(p["construction"]["method"] for p in pairs)),
            "severity": dict(Counter(p.get("severity") for p in pairs)),
            "severity_source": dict(Counter(p.get("severity_source") for p in pairs)),
            "complete_sites_removed": dict(Counter(len(p["removed_site_ids"]) for p in comp)),
            "partial_residual_count": dict(Counter(p["residual"]["n_surviving"] for p in part)),
            "partial_residual_max_strength": dict(Counter(p["residual"]["max_strength"]
                                                          for p in part)),
        },
        "spend_usd": state["spend"]["cost_usd"], "calls": state["spend"]["calls"],
        "skipped": state["skipped"],
        "pairs": pairs,
    }
    tmp = PAIRS_FILE + ".tmp"
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, PAIRS_FILE)
    return out


def report(out):
    c = out["counts"]
    print(f"\nmaster/omission_pairs_v2.json - {c['pairs']} pairs {c['by_class']} over "
          f"{c['consultations']} consultations {c['by_stratum']}")
    print(f"  levels {c.get('by_residual_level')} | cells {c.get('by_cell')}")
    print(f"  matched sets (same fact, >1 level): {c['matched_pairs']}")
    print(f"  collateral cleanups: {c.get('collateral_cleanups')} pairs, "
          f"{c.get('collateral_words_dropped')} words dropped")
    print(f"  consultations with NO multi-site fact (contribute nothing): "
          f"{c['consultations_without_a_multi_site_fact']}")
    print(f"  construction: {c['construction_method']}")
    print(f"  complete pairs by sites removed: {c['complete_sites_removed']}")
    print(f"  partial pairs by residual count: {c['partial_residual_count']} | strongest "
          f"residual: {c['partial_residual_max_strength']}")
    print(f"  severity: {c['severity']} via {c['severity_source']}")
    print(f"  spend ${out['spend_usd']:.2f} over {out['calls']} calls")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true", help="whatever the site map holds (smoke map)")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--consults", help="comma-separated 'stratum|id' keys")
    ap.add_argument("--classes", help="omit-complete,omit-partial (default both)")
    ap.add_argument("--plan", action="store_true",
                    help="build from master/factorial_allocation.json (the factorial build)")
    ap.add_argument("--dev-pool", action="store_true",
                    help="build the GEPA dev-split plan into its own artefacts, so the "
                         "evaluation build's files are never touched")
    ap.add_argument("--targets-per-consult", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--budget", type=float, default=OS.DEFAULT_BUDGET)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    if args.dev_pool:
        globals()["ALLOC_FILE"] = os.path.join(MASTER, "factorial_allocation_dev.json")
        globals()["PAIRS_FILE"] = os.path.join(MASTER, "omission_pairs_dev.json")
        globals()["STATE_FILE"] = os.path.join(MASTER, "omission_v2_dev_state.json")
    keys = [k.strip() for k in args.consults.split(",")] if args.consults else None
    doc, cons_rows = load_sites(keys)
    state = load_state()
    if args.report:
        report(write_pairs_file(doc, cons_rows, state, args))
        return 0
    if not (args.smoke or args.full or args.plan or keys):
        sys.exit("pick a scope: --plan, --smoke, --full, or --consults")

    budget = OS.Budget(state, args.budget)
    if args.plan:
        n_plan = len(json.load(open(ALLOC_FILE))["plan"]) if os.path.exists(ALLOC_FILE) else 0
        print(f"site map: {len(cons_rows)} consultations | allocation plan: {n_plan} pairs")
    else:
        n_targets = sum(len(pick_targets(c, args.targets_per_consult)) for c in cons_rows)
        print(f"site map: {len(cons_rows)} consultations, {n_targets} target facts -> up to "
              f"{2 * n_targets} pairs")
    if not args.yes:
        try:
            if input(f"build (cap ${args.budget:.0f})? [y/N] ").strip().lower() != "y":
                sys.exit("aborted")
        except EOFError:
            sys.exit("aborted (non-interactive)")

    params = {"scope": "plan" if args.plan else ("smoke" if args.smoke else "full"),
              "n_consultations": len(cons_rows), "dataset_version": DATASET_VERSION,
              "targets_per_consult": args.targets_per_consult, "budget_cap_usd": args.budget}
    if args.smoke:
        params["NOT_PAPER_BOUND"] = "smoke: instrument shakedown, not a corpus build"
    with Run(EXPERIMENT, params=params, seed=SEED, inputs=["master/fact_sites.json"],
             spec=SPEC, allow_dirty=args.allow_dirty) as run:
        run.register_prompts(prompts())
        left = build(cons_rows, state, budget, args)
        out = write_pairs_file(doc, cons_rows, state, args)
        run.save("omission_pairs_summary.json", out["counts"])
    report(out)
    if budget.hit.is_set():
        print(f"\nBUDGET STOP at ${budget.spent:.2f} (cap ${budget.cap:.0f}) - state saved.")
        return 3
    return 0 if left == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
