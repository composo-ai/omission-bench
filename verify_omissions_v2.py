"""verify_omissions_v2.py - class-appropriate verification for graded omission pairs.

Deliverable 4 of the graded-omission design correction (specs/w-d-master-dataset.md
Amendment 2026-08-10). The v1 EDIT_CHECK asks one question of every omit pair - did the
fact survive anywhere? - which is the right question for a COMPLETE omission and exactly
the wrong one for a deliberate PARTIAL. Each class gets the check it deserves:

  omit-complete
    truly_absent          the removed fact was present in the clean note and survives
                          NOWHERE in the errored note (no partial or paraphrased residual
                          mention). VERBATIM the corrected W-A clause - same standard, so
                          the new complete pairs and the frozen 93 are directly comparable.
    spans_all_state_fact  NEW CHECK. Every removed span states the target fact. This is the
                          other half of "single edit = single FACT": a multi-site deletion
                          is legitimate only if all of it was that fact, so nothing else
                          left the note by accident.
    no_collateral_loss    every OTHER fact in the clean note is still in the errored note.
    reads_naturally       the result reads as a note that simply never recorded the fact,
                          not as a mutilated one (a visibly damaged note would be detected
                          for the wrong reason, which would inflate the headline result).

  omit-partial
    primary_site_removed  the note's home statement of the fact is gone.
    residual_present      the INVERSE of truly_absent: at least one mention of the fact
                          survives - quoted and graded, which is where the residual
                          metadata comes from.
    no_collateral_loss    as above.
    reads_naturally       as above.

Free algorithmic checks run first and cost nothing: deletion-only diff, no new content,
removed spans really gone, residual spans really intact, and - using the site map - every
mapped site of every OTHER fact still present. Semantic checks then run on the auditor
(models.lock `auditor` = openai/gpt-5.5, reasoning_effort high) under the SAME A5 lever
W-A used: seed 21 screens, any failing field escalates to seeds 22 and 23, and the
per-field majority of the three decides.

Also verifies the relabelled seed set (master/partial_omission_seed_set.json) when asked
with --seed-set: those 34 pairs already have their W-A receipts, and one residual-present
pass fills the residual metadata the auditor's free-text reasons did not yield.

Usage:
  .venv/bin/python verify_omissions_v2.py --smoke --allow-dirty -y
  .venv/bin/python verify_omissions_v2.py --full --workers 6 -y
  .venv/bin/python verify_omissions_v2.py --seed-set --workers 4 -y
  .venv/bin/python verify_omissions_v2.py --report            # free
Exit: 0 complete | 2 work left (rerun) | 3 budget stop.
"""
import argparse, json, os, sys, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from common import HERE, Run, resolve_model
import inject_omissions_v2 as INJ
import omission_sites as OS
import w_a_master as WA

SPEC = OS.SPEC
EXPERIMENT = "wd-omission-verify"
MASTER = OS.MASTER
VERIFY_FILE = os.path.join(MASTER, "omission_verification_v2.json")
STATE_FILE = os.path.join(MASTER, "omission_verify_state.json")
SEED_SET_FILE = os.path.join(MASTER, "partial_omission_seed_set.json")

EDIT_SEEDS = WA.EDIT_SEEDS          # (21, 22, 23) - the W-A discipline, unchanged

# The factorial build verifies on a CROSS-FAMILY PANEL rather than three seeds of one model.
# Re-seeding a model mostly re-samples its own blind spots; a second family has different
# ones, which is the whole point of a check. Two panelists run on every pair; a third call
# fires only when they disagree on a field, and the per-field majority of the three decides.
#
# The asymmetry is deliberate and worth stating plainly: with two of the three calls coming
# from gpt-5.5, a lone opus dissent cannot by itself fail a pair - it forces a re-check. So
# this is "the W-A auditor, challenged by a second family", not a neutral 3-way vote. Every
# pair records whether the panel was unanimous and which fields the families split on, and
# the report gives both the majority pass rate and the stricter unanimous one.
PANEL = (("auditor", "high", 21),        # openai/gpt-5.5 - W-A's own verifier, unchanged
         ("judge-opus", "high", 21))     # anthropic/claude-opus-4.8 - the second family
TIEBREAK = ("auditor", "high", 22)       # W-A's own escalation seed
FIELDS = {
    "omit-complete": ("spans_all_state_fact", "truly_absent", "no_collateral_loss",
                      "reads_naturally"),
    "omit-partial": ("primary_site_removed", "residual_present", "no_collateral_loss",
                     "reads_naturally"),
}


# ---------------------------------------------------------------- prompts
# Item 2 of COMPLETE_CHECK is the corrected W-A truly_absent clause, word for word, so a
# complete pair verified here meets exactly the standard the frozen 93 met.
COMPLETE_CHECK = """A clean clinical note had ONE clinical FACT removed from it - every place the note stated
that fact was deleted, and nothing else was touched. You are given the fact, the exact spans
that were deleted, the diff, and the full errored note. Verify:

1. spans_all_state_fact: EVERY deleted span states the target fact, in whole or in part.
   False if any deleted span carries something else as well - a different finding, a
   different plan item - that has now been lost along with the target.
2. truly_absent: the removed fact was present in the clean note and survives NOWHERE in the
   errored note (no partial or paraphrased residual mention). Judge this over the ERRORED
   NOTE ONLY. The fact obviously still exists in the consultation itself - that is not what
   is being asked. The question is whether the NOTE still records it anywhere.
3. no_collateral_loss: every OTHER clinical fact the clean note recorded is still recorded in
   the errored note. False if anything besides the target fact went missing.
4. reads_naturally: the errored note reads as a clinical note that simply never recorded this
   fact - not as a note visibly cut about (dangling clauses, stranded conjunctions, an empty
   heading where content obviously used to be).

Return ONLY JSON:
{"spans_all_state_fact":true|false,"truly_absent":true|false,"no_collateral_loss":true|false,
"reads_naturally":true|false,"reason":"<one line covering any false>"}

THE TARGET FACT:
{fact}

SPANS DELETED (all of them):
{removed}

DIFF (unified, clean -> errored):
{diff}

ERRORED NOTE (full, for the residual-mention check):
{errored}"""

PARTIAL_CHECK = """A clean clinical note had a clinical FACT partly removed: the note's home statement of it
was deleted, and so was every other mention EXCEPT one, which was left in place
deliberately. This is a PARTIAL omission and the surviving mention is the point of it, not a
defect. You are given the fact, the spans that were deleted, the diff, and the full errored
note. Verify:

1. primary_site_removed: the note's home statement of the fact - the place a reader would
   have gone for it, carrying it most completely and specifically - is among the deleted
   spans and is gone from the errored note.
2. residual_present: at least one mention of the SAME fact still survives in the ERRORED
   NOTE, in whole or in part (the note is the whole universe here - what the consultation
   itself contained is irrelevant). This is the inverse of the usual check: here a surviving
   paraphrase is REQUIRED. Quote every surviving mention and grade each one:
     "explicit"   - states the fact plainly, in the note's own clinical terms
     "paraphrase" - states the same fact in different words
     "partial"    - states only part of it, losing a defining detail (a value, a laterality,
                    a hedge, or the half that drives action)
   A passage that merely lets the fact be inferred, or that touches on something related, is
   NOT a residual mention - do not list it.
3. no_collateral_loss: every OTHER clinical fact the clean note recorded is still recorded in
   the errored note. False if anything besides the target fact went missing.
4. reads_naturally: the errored note reads as a clinical note that simply recorded this fact
   in fewer places - not as a note visibly cut about (dangling clauses, stranded
   conjunctions, an empty heading where content obviously used to be).

Return ONLY JSON:
{"primary_site_removed":true|false,"residual_present":true|false,
"residuals":[{"quote":"<exact text from the errored note>","section":"<heading>",
"strength":"explicit|paraphrase|partial"}],
"no_collateral_loss":true|false,"reads_naturally":true|false,
"reason":"<one line covering any false>"}

THE TARGET FACT:
{fact}

SPANS DELETED:
{removed}

DIFF (unified, clean -> errored):
{diff}

ERRORED NOTE (full, for the residual check):
{errored}"""

PROMPTS = {"COMPLETE_CHECK": COMPLETE_CHECK, "PARTIAL_CHECK": PARTIAL_CHECK}


# ---------------------------------------------------------------- algorithmic checks (free)
def deleted_units(clean, errored):
    """The unit texts present in `clean` and gone from `errored`. Used for pairs that predate
    the site map (the 34 relabelled partials), so the check prompt can still name what left."""
    import difflib
    uc, ue = WA.units_of(clean), WA.units_of(errored)
    out = []
    for tag, i1, i2, _, _ in difflib.SequenceMatcher(None, uc, ue).get_opcodes():
        if tag in ("delete", "replace"):
            out.extend(uc[i1:i2])
    return out


def algo_checks(pair, cons):
    """Everything checkable without a model. Cheap, exact, and it catches the failure modes
    a semantic checker is worst at: a span that did not actually get deleted, a residual that
    got deleted by accident, content appearing from nowhere."""
    clean, errored = pair["clean"], pair["errored"]
    lo_err, lo_clean = OS.normlow(errored), OS.normlow(clean)
    v = {}

    uc, ue = WA.units_of(clean), WA.units_of(errored)
    import difflib
    blocks = [op for op in difflib.SequenceMatcher(None, uc, ue).get_opcodes()
              if op[0] != "equal"]
    v["n_blocks"] = len(blocks)
    v["block_tags"] = sorted({op[0] for op in blocks})
    # shape guard: every block either deletes outright, or replaces a span with something no
    # longer than it was plus the grammar-repair allowance (a tidied wound, or a renumbered
    # list item). Whether any CONTENT arrived is `no_new_content`'s job, below.
    v["deletions_only"] = bool(blocks) and all(
        op[0] == "delete"
        or (op[0] == "replace"
            and len(" ".join(ue[op[3]:op[4]]).split())
            <= len(" ".join(uc[op[1]:op[2]]).split()) + INJ.MAX_TIDY_WORDS)
        for op in blocks)

    # word tokens stripped of attached punctuation before comparison: the first smoke failed
    # two good pairs on "above." and "after)." - words that ARE in the clean note, carrying
    # different punctuation after a clause was cut out from beside them
    def content_words(text):
        return {w for w in (t.strip(".,;:()[]\"'!?") for t in OS.normlow(text).split())
                if len(w) > 4}
    new_words = sorted(content_words(errored) - content_words(clean))
    v["no_new_content"] = not new_words
    v["new_words"] = new_words[:10]

    v["removed_spans_gone"] = all(OS.normlow(s) not in lo_err for s in pair["removed_spans"])
    # the seed set's residual entries are parsed quotes, not mapped spans - key on either
    v["residual_spans_intact"] = all(
        OS.normlow(r.get("span") or r.get("quote") or "") in lo_err
        for r in pair["residual"]["sites"])

    # Collateral, from the site map: every mapped site of every OTHER fact should survive.
    #
    # With one exception that has to be separated out, or the check is wrong more often than
    # it is right. Fact sheets overlap: the same clinical content is frequently both a
    # must_contain item and a salience trap, so "Ibuprofen and Motrin have not helped" is a
    # site of both, and removing one necessarily removes the other. That is not collateral
    # loss - it is one fact wearing two keys. Sites whose characters OVERLAP the removed
    # spans are therefore recorded as co-located and left to the auditor's semantic
    # no_collateral_loss judgement; only a site somewhere ELSE in the note vanishing is a
    # deterministic failure.
    rm_ranges = [c for c in (pair.get("removed_char_spans") or []) if c]
    removed_txt = {OS.normlow(s) for s in pair["removed_spans"]}

    def overlaps(cs):
        return cs and any(cs[0] < b and a < cs[1] for a, b in rm_ranges)

    lost, colocated = [], []
    for f in (cons or {}).get("facts", []):
        if f["fact_key"] == pair["fact_key"]:
            continue
        for s in f["sites"]:
            if not s["span_found"] or OS.normlow(s["span"]) in removed_txt:
                continue
            if OS.normlow(s["span"]) in lo_err:
                continue
            rec = {"fact_key": f["fact_key"], "span": s["span"][:120]}
            (colocated if overlaps(s.get("char_span")) else lost).append(rec)
    v["other_fact_sites_intact"] = not lost
    v["other_fact_sites_lost"] = lost[:8]
    v["co_located_sites_removed"] = colocated[:8]

    gates = ["deletions_only", "no_new_content", "removed_spans_gone",
             "other_fact_sites_intact"]
    if pair["class"] == "omit-partial":
        gates.append("residual_spans_intact")
    if pair.get("origin") == "w_a_excluded_2026-08-10":
        # the 34 relabelled partials were built and verified by the v1 instrument and carry
        # their own W-A receipts. Re-gating them on checks designed for site-map construction
        # would fail them for not having been built that way. Recorded, not gated - the only
        # thing --seed-set is asking of them is their residual metadata.
        v["gated"] = False
        v["pass"], v["failed"] = True, []
        return v
    v["gated"] = True
    v["pass"] = all(v[g] for g in gates)
    v["failed"] = [g for g in gates if not v[g]]
    return v


# ---------------------------------------------------------------- semantic checks
def render_removed(pair):
    return "\n".join(f"- [{sec}] {sp}" for sp, sec in
                     zip(pair["removed_spans"], pair["removed_sections"]))


def sem_prompt(pair, transcript):
    tmpl = COMPLETE_CHECK if pair["class"] == "omit-complete" else PARTIAL_CHECK
    # No transcript. Every omit check is note-internal (was it deleted, does it survive, did
    # anything else go), and supplying the transcript made the auditor fail truly_absent
    # because the fact "still appears in the transcript" - it always does; the transcript is
    # the source, not the note. Dropping it removed that failure mode and ~10k tokens a call.
    return (tmpl.replace("{fact}", pair["fact"])
            .replace("{removed}", render_removed(pair))
            .replace("{diff}", WA.unit_diff_text(pair["clean"], pair["errored"]))
            .replace("{errored}", pair["errored"]))


def parse_sem(obj, klass, errored):
    if not isinstance(obj, dict):
        return None
    out = {}
    for f in FIELDS[klass]:
        v = obj.get(f)
        if not isinstance(v, bool):
            return None
        out[f] = v
    out["reason"] = obj.get("reason")
    if klass == "omit-partial":
        res, lo = [], OS.normlow(errored)
        for r in (obj.get("residuals") or []):
            if not isinstance(r, dict) or not r.get("quote"):
                continue
            q = OS.norm(r["quote"])
            res.append({"quote": q, "section": OS.norm(r.get("section")) or "(none)",
                        "strength": r.get("strength") if r.get("strength") in OS.STRENGTHS
                        else None,
                        "verbatim_in_errored": OS.normlow(q) in lo})
        out["residuals"] = res
    return out


def run_panel(pair, prompt, klass, budget, args):
    """The cross-family panel: two panelists, plus a tiebreak call on any field they split
    on. Returns (runs, errors) keyed by a readable call label."""
    runs, errs = {}, []
    for role, effort, seed in PANEL:
        obj, meta = OS.call_json(budget, "editcheck", prompt, role, effort, seed=seed)
        parsed = parse_sem(obj, klass, pair["errored"]) if obj is not None else None
        if parsed is None:
            errs.append({"panelist": role, "seed": seed, **(meta or {})})
            continue
        parsed.update({"gen": (meta or {}).get("generation_ids"), "role": role,
                       "reasoning_effort": effort, "seed": seed})
        runs[f"{role}|s{seed}"] = parsed
    split = [f for f in FIELDS[klass]
             if len({r[f] for r in runs.values() if r.get(f) is not None}) > 1]
    if split and len(runs) == len(PANEL):
        role, effort, seed = TIEBREAK
        obj, meta = OS.call_json(budget, "editcheck", prompt, role, effort, seed=seed)
        parsed = parse_sem(obj, klass, pair["errored"]) if obj is not None else None
        if parsed is None:
            errs.append({"panelist": role, "seed": seed, "tiebreak": True, **(meta or {})})
        else:
            parsed.update({"gen": (meta or {}).get("generation_ids"), "role": role,
                           "reasoning_effort": effort, "seed": seed, "tiebreak": True})
            runs[f"{role}|s{seed}"] = parsed
    return runs, errs, split


def sem_majority(runs, klass):
    """Per-field majority over the seeds that returned. Single-seed screens stand on their
    own (the A5 lever's first-pass acceptance rule) - identical to W-A's sem_majority."""
    verdict, fails = {}, []
    n = len(runs)
    for f in FIELDS[klass]:
        vals = [r[f] for r in runs.values() if r.get(f) is not None]
        if not vals:
            verdict[f], _ = None, fails.append(f)
            continue
        maj = (sum(1 for v in vals if v) > len(vals) / 2) if n > 1 else bool(vals[0])
        verdict[f] = maj
        if not maj:
            fails.append(f)
    return verdict, all(verdict.get(f) is True for f in FIELDS[klass]), fails


def fold_residuals(runs):
    """Residual metadata from the seeds that saw one: a residual counts when its quote is
    verbatim-present in the errored note; strength is the modal grade across seeds; the
    pair's residual_strength is the STRONGEST surviving mention (the hardest case for the
    prediction, so the conservative one to record)."""
    by_quote = {}
    for r in runs.values():
        for res in r.get("residuals", []):
            if not res["verbatim_in_errored"]:
                continue
            e = by_quote.setdefault(OS.normlow(res["quote"]),
                                    {"quote": res["quote"], "section": res["section"],
                                     "grades": [], "seeds": 0})
            e["seeds"] += 1
            if res["strength"]:
                e["grades"].append(res["strength"])
    out = []
    for e in by_quote.values():
        grade = Counter(e["grades"]).most_common(1)[0][0] if e["grades"] else None
        out.append({"quote": e["quote"], "section": e["section"], "strength": grade,
                    "seeds_naming_it": e["seeds"]})
    rank = {g: i for i, g in enumerate(OS.STRENGTHS)}
    strongest = min((o["strength"] for o in out if o["strength"]),
                    key=lambda g: rank[g], default=None)
    return {"n_verified": len(out), "sites": out, "strongest": strongest}


# ---------------------------------------------------------------- driver
_lock = threading.RLock()   # see omission_sites._lock - save_state() re-enters it


def load_state():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {"created": OS.now_utc(), "spend": {"cost_usd": 0.0, "calls": 0, "by_stage": {}},
            "pairs": {}}


def save_state(state):
    with _lock:
        tmp = STATE_FILE + ".tmp"
        json.dump(state, open(tmp, "w"), ensure_ascii=False, indent=1)
        os.replace(tmp, STATE_FILE)


def load_inputs(seed_set=False):
    """(pairs, consultations-by-key, transcripts-by-key)."""
    sites = json.load(open(OS.SITES_FILE)) if os.path.exists(OS.SITES_FILE) else {"consultations": []}
    cons = {c["key"]: c for c in sites["consultations"]}
    tx = {r["key"]: r["transcript"] for r in OS.consultations(("eval", "gepa_dev"))}
    if seed_set:
        if not os.path.exists(SEED_SET_FILE):
            sys.exit(f"no seed set at {SEED_SET_FILE} - run relabel_partial_seed.py first")
        doc = json.load(open(SEED_SET_FILE))
        pairs = [{**p, "class": "omit-partial",
                  # these 34 predate the site map, so what was removed is recovered from the
                  # clean->errored diff rather than from a span list
                  "removed_spans": deleted_units(p["clean"], p["errored"]),
                  "removed_sections": ["(from the diff)"] * len(deleted_units(p["clean"],
                                                                              p["errored"])),
                  "residual": p.get("residual") or {"n_surviving": None, "sites": []},
                  "fact_key": p.get("fact_key") or "unmapped"}
                 for p in doc["pairs"]]
        return pairs, cons, tx
    if not os.path.exists(INJ.PAIRS_FILE):
        sys.exit(f"no pairs at {INJ.PAIRS_FILE} - run inject_omissions_v2.py first")
    return json.load(open(INJ.PAIRS_FILE))["pairs"], cons, tx


def verify(pairs, cons, tx, state, budget, args):
    todo = [p for p in pairs if p["pair_id"] not in state["pairs"]]
    print(f"pairs: {len(pairs)} in scope, {len(todo)} to verify | auditor "
          f"{resolve_model('auditor')['resolved']} effort {WA.REASONING} | seeds "
          f"{EDIT_SEEDS} (screen {EDIT_SEEDS[0]}, escalate on any false) | budget "
          f"${budget.spent:.2f}/${budget.cap:.0f}", flush=True)
    if args.chunk:
        todo = todo[:args.chunk]

    def one(pair):
        if not budget.ok():
            return
        klass = pair["class"]
        algo = algo_checks(pair, cons.get(pair.get("key")))
        prompt = sem_prompt(pair, tx.get(pair.get("key")))
        split = []
        if args.panel:
            runs, errs, split = run_panel(pair, prompt, klass, budget, args)
        else:
            runs, errs = {}, []
            for i, seed in enumerate(EDIT_SEEDS[:max(1, args.seeds)]):
                obj, meta = OS.call_json(budget, "editcheck", prompt, "auditor", WA.REASONING,
                                         seed=seed)
                parsed = parse_sem(obj, klass, pair["errored"]) if obj is not None else None
                if parsed is None:
                    errs.append({"seed": seed, **(meta or {})})
                    continue
                parsed["gen"] = (meta or {}).get("generation_ids")
                runs[str(seed)] = parsed
                if i == 0 and args.seeds > 1:
                    _, ok0, _ = sem_majority({str(seed): parsed}, klass)
                    if ok0:
                        break        # A5: a clean screen is accepted on one seed
        verdict, sem_ok, fails = sem_majority(runs, klass) if runs else ({}, False, ["no_runs"])
        rec = {"pair_id": pair["pair_id"], "class": klass, "stratum": pair.get("stratum"),
               "id": pair.get("id"), "key": pair.get("key"), "split": pair.get("split"),
               "residual_level": pair.get("residual_level"), "cell": pair.get("cell"),
               "severity": pair.get("severity"),
               "fact_uid": pair.get("fact_uid"), "algo": algo, "sem_verdict": verdict,
               "sem_fails": fails, "n_seeds": len(runs), "sem_runs": runs,
               "panel": bool(args.panel), "cross_family_split": split,
               "unanimous": bool(runs) and not split,
               "sem_errors": errs, "utc": OS.now_utc(),
               "verified": bool(algo["pass"] and sem_ok),
               "verified_reason": None if (algo["pass"] and sem_ok) else
               "; ".join(filter(None, [
                   ("algorithmic: " + ", ".join(algo["failed"])) if not algo["pass"] else "",
                   ("semantic: " + ", ".join(fails)) if not sem_ok else ""]))}
        if klass == "omit-partial":
            rec["residual_verified"] = fold_residuals(runs)
        with _lock:
            state["pairs"][pair["pair_id"]] = rec
            save_state(state)
        mark = "PASS" if rec["verified"] else "fail"
        extra = ""
        if klass == "omit-partial" and rec.get("residual_verified"):
            extra = (f" resid {rec['residual_verified']['n_verified']}"
                     f"/{rec['residual_verified']['strongest']}")
        print(f"  {pair['pair_id']:52} {klass:14} {mark} seeds {len(runs)}"
              f"{extra} {rec['verified_reason'] or ''} ${budget.spent:.2f}", flush=True)

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(one, todo))
    save_state(state)
    return sum(1 for p in pairs if p["pair_id"] not in state["pairs"])


def write_report(pairs, state, args):
    recs = [state["pairs"][p["pair_id"]] for p in pairs if p["pair_id"] in state["pairs"]]
    by_class = {}
    for k in ("omit-complete", "omit-partial"):
        sub = [r for r in recs if r["class"] == k]
        if not sub:
            continue
        field_fail = Counter()
        for r in sub:
            for f in r["sem_fails"]:
                field_fail[f] += 1
        by_class[k] = {
            "n": len(sub), "verified": sum(1 for r in sub if r["verified"]),
            "verified_rate": round(sum(1 for r in sub if r["verified"]) / len(sub), 4),
            "algo_pass": sum(1 for r in sub if r["algo"]["pass"]),
            "algo_failed_checks": dict(Counter(c for r in sub for c in r["algo"]["failed"])),
            "semantic_field_failures": dict(field_fail),
            "seeds_used": dict(Counter(r["n_seeds"] for r in sub)),
            "unanimous": sum(1 for r in sub if r.get("unanimous")),
            "cross_family_split_fields": dict(Counter(
                f for r in sub for f in (r.get("cross_family_split") or []))),
            "verified_and_unanimous": sum(1 for r in sub
                                          if r["verified"] and r.get("unanimous")),
            "by_cell": dict(Counter(r.get("cell") for r in sub)),
            "verified_by_cell": dict(Counter(r.get("cell") for r in sub if r["verified"])),
        }
        if k == "omit-partial":
            by_class[k]["residual_strength"] = dict(Counter(
                (r.get("residual_verified") or {}).get("strongest") for r in sub))
            by_class[k]["residual_count"] = dict(Counter(
                (r.get("residual_verified") or {}).get("n_verified") for r in sub))
    out = {"generated_utc": OS.now_utc(), "spec": SPEC,
           "what": "class-appropriate verification of the graded-omission pairs. "
                   "omit-complete = corrected truly_absent + the new same-fact-spans check; "
                   "omit-partial = residual-present + primary-site-removed. Same auditor, "
                   "same seeds, same majority rule as W-A.",
           "models": ({"panel": [{"role": r, "reasoning_effort": e, "seed": sd,
                                 **resolve_model(r)} for r, e, sd in PANEL],
                       "tiebreak": {"role": TIEBREAK[0], "reasoning_effort": TIEBREAK[1],
                                    "seed": TIEBREAK[2], **resolve_model(TIEBREAK[0])}}
                      if args.panel else
                      {"auditor": {"role": "auditor", **resolve_model("auditor"),
                                   "reasoning_effort": WA.REASONING,
                                   "seeds": list(EDIT_SEEDS[:max(1, args.seeds)])}}),
           "prompt_sha256_16": {k: WA.sha16(v) for k, v in PROMPTS.items()},
           "lever": ("cross-family panel of 2 (gpt-5.5 + claude-opus-4.8), third call from "
                     "the auditor at seed 22 on any field they split on, per-field majority"
                     if args.panel else
                     "A5: 1-seed screen + 3-seed majority confirmation on any failing field"),
           "params": {"seeds": args.seeds, "smoke": bool(args.smoke),
                      "seed_set": bool(args.seed_set)},
           "counts": {"n": len(recs), "by_class": by_class,
                      "verified_total": sum(1 for r in recs if r["verified"])},
           "spend_usd": state["spend"]["cost_usd"], "calls": state["spend"]["calls"],
           "pairs": recs}
    path = VERIFY_FILE if not args.seed_set else VERIFY_FILE.replace(".json", "_seed_set.json")
    tmp = path + ".tmp"
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return out, path


def report(out, path):
    print(f"\n{os.path.relpath(path, HERE)} - {out['counts']['n']} pairs, "
          f"{out['counts']['verified_total']} verified")
    for k, c in out["counts"]["by_class"].items():
        print(f"  {k}: {c['verified']}/{c['n']} verified ({100 * c['verified_rate']:.0f}%) | "
              f"algo pass {c['algo_pass']}/{c['n']} {c['algo_failed_checks']}")
        print(f"    semantic field failures: {c['semantic_field_failures']} | calls used "
              f"{c['seeds_used']}")
        if c.get("unanimous") is not None:
            print(f"    panel unanimous {c['unanimous']}/{c['n']} | verified+unanimous "
                  f"{c['verified_and_unanimous']} | family splits {c['cross_family_split_fields']}")
        if "residual_strength" in c:
            print(f"    residual strength: {c['residual_strength']} | residual count: "
                  f"{c['residual_count']}")
    print(f"  spend ${out['spend_usd']:.2f} over {out['calls']} calls")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--only-class", help="omit-complete | omit-partial - verify one class "
                                        "only, so the two can run at different seed budgets")
    ap.add_argument("--dev-pool", action="store_true",
                    help="verify the GEPA dev-split build into its own artefacts")
    ap.add_argument("--seed-set", action="store_true",
                    help="verify the 34 relabelled partials instead of the built pairs")
    ap.add_argument("--seeds", type=int, default=3, help="1 = screen only; 3 = the A5 lever")
    ap.add_argument("--panel", action="store_true", default=True,
                    help="cross-family panel of 2 + tiebreak (the factorial default)")
    ap.add_argument("--no-panel", dest="panel", action="store_false",
                    help="fall back to the single-model A5 seed lever")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--budget", type=float, default=OS.DEFAULT_BUDGET)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    if args.dev_pool:
        globals()["VERIFY_FILE"] = os.path.join(MASTER, "omission_verification_dev.json")
        globals()["STATE_FILE"] = os.path.join(MASTER, "omission_verify_dev_state.json")
        INJ.PAIRS_FILE = os.path.join(MASTER, "omission_pairs_dev.json")
    pairs, cons, tx = load_inputs(args.seed_set)
    if args.only_class:
        pairs = [p for p in pairs if p["class"] == args.only_class]
    state = load_state()
    if args.report:
        out, path = write_report(pairs, state, args)
        report(out, path)
        return 0
    if not (args.smoke or args.full or args.seed_set):
        sys.exit("pick a scope: --smoke, --full or --seed-set")

    budget = OS.Budget(state, args.budget)
    print(f"{len(pairs)} pairs in scope "
          f"({dict(Counter(p['class'] for p in pairs))}) | "
          + ("panel: " + " + ".join(f"{r}@{resolve_model(r)['resolved']}" for r, _, _ in PANEL)
             if args.panel else f"A5 lever, {args.seeds} seeds"))
    if not args.yes:
        try:
            if input(f"verify (cap ${args.budget:.0f})? [y/N] ").strip().lower() != "y":
                sys.exit("aborted")
        except EOFError:
            sys.exit("aborted (non-interactive)")

    params = {"scope": "seed_set" if args.seed_set else ("smoke" if args.smoke else "full"),
              "n_pairs": len(pairs), "seeds": args.seeds, "budget_cap_usd": args.budget}
    if args.smoke:
        params["NOT_PAPER_BOUND"] = "smoke: instrument shakedown"
    with Run(EXPERIMENT, params=params, seed=EDIT_SEEDS[0],
             inputs=[], spec=SPEC, allow_dirty=args.allow_dirty) as run:
        run.register_prompts(PROMPTS)
        left = verify(pairs, cons, tx, state, budget, args)
        out, path = write_report(pairs, state, args)
        run.save("omission_verification_summary.json", out["counts"])
    report(out, path)
    if budget.hit.is_set():
        print(f"\nBUDGET STOP at ${budget.spent:.2f} (cap ${budget.cap:.0f}) - state saved.")
        return 3
    return 0 if left == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
