"""The census, stage 2 of 4: the adversarial panel that tries to refute every candidate.

Four instruments over the candidate findings from `taxonomy_discover.py`:

  1. REFUTE panel - **2 skeptics from DIFFERENT model families**: the constructor
     (anthropic/claude-opus-5, June's own model on the OpenRouter transport) and the
     auditor (openai/gpt-5.5). Each is shown the FULL
     note and the FULL transcript and told to refute if at all defensible. Both keep
     -> verified; both refute -> cut; **they disagree -> one tiebreak call on a third
     model (judge-primary, openai/gpt-5.4, reasoning_effort high) whose verdict is
     final**. An unparseable or failed reply still counts as REFUTED (June's
     convention), which means a transport failure produces a split rather than a
     silent cut, and the tiebreak resolves it on evidence.

     Why this replaces June's 3-of-one-model majority: the 2026-08-10 smoke found 45
     of 48 verdicts unanimous with near-identical reasoning text, i.e. one opinion
     counted three times. Two families cost less AND disagree for real reasons; a
     shared blind spot now has to cross a family boundary to survive.

  2. SALIENCE rater - `verify_findings.py`'s high/medium/low rater, verbatim, on
     every candidate; it overrides the differ's own guess exactly as it did in June.
  3. SEVERITY grader - graded against the written rubric (embedded verbatim in the
     prompt and hashed into every manifest) on critical / supporting / peripheral. A
     different axis from the
     differ's own critical/moderate/trivial guess, so it lands in `severity_rubric`
     and nothing is overwritten. Graded only on panel survivors.
  4. COMPLETENESS critic (--complete) - per note, hunt what the targeted passes
     missed; new candidates go through the same panel. Verbatim prompt.

**Prompt-cache order.** The panel prompt is the June REFUTE text with its blocks
reordered transcript-first, and panel work is executed one note at a time with a serial
warm-up call before the fan-out, so every call for a note shares a hot prefix. Measured
2026-08-12: OpenAI routes cache implicitly (71% of input tokens on a 10.6k-char prefix),
Anthropic routes return 0 cached tokens without cache_control breakpoints - so this buys
the gpt-5.5 and gpt-5.4 legs, which is where the reorder was worth its comparability
cost. The wording is June's; only block order moved, and both members see the identical
prompt so the cross-family comparison stays clean.

**Salience filter is ON by default** (`--salience-filter high,medium`) - which MATCHES
June, whose run verified high+medium only. Pass `--salience-filter all` to verify every
candidate.

Idempotent per CALL (finding x panel member), so a killed run resumes without
re-buying a single skeptic.

    python3 census/taxonomy_verify.py --in master/findings_master_discover-smoke.json \\
        --tag verify-smoke --allow-dirty -y
    python3 census/taxonomy_verify.py -y                 # the full pass
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import argparse, hashlib, json, os, time
from collections import Counter, defaultdict

from common import HERE, Run
import taxonomy_common as T

REFUTE, SALIENCE, COMPLETE = T.import_verbatim("verify_findings", ["REFUTE", "SALIENCE", "COMPLETE"])
TRANSCRIPT_CHARS = 40000          # verify_findings.py's own truncation, kept
RUBRIC_PATH = "master/severity-rubric.md"

# June's REFUTE, reordered transcript-first so every finding on one note shares a
# cacheable prefix. Every sentence is verbatim from verify_findings.REFUTE; only the
# order of the blocks changed, and `assert_refute_is_a_reorder()` fails the run if the
# two ever stop being the same sentences.
REFUTE_TF = """FULL SOURCE ({skind}):
{source}

FULL NOTE:
{note}

A discovery tool flagged a possible discrepancy in an AI-generated clinical NOTE versus its SOURCE. Your job is to REFUTE it if at all defensible - be a hard skeptic. It is NOT a real, reportable discrepancy if it is any of: paraphrase, clinical synonymy, a standard abbreviation, a normal documentation convention, justified clinical inference, a quote that does not actually appear, a mischaracterisation by the tool, or clinically immaterial. Default to REFUTED if you are uncertain.

FLAGGED MODE: {mode}
claim: {description}
note says: {note_quote}
source says: {source_quote}

Is this a REAL, correctly-characterised discrepancy of the stated kind? Return ONLY JSON {{"refuted": true|false, "reason": "<short>"}}  (refuted=true means it is NOT a genuine issue)."""


def assert_refute_is_a_reorder():
    """The reordered panel prompt must be June's sentences in a different order and
    nothing else. Compares the two as multisets of non-empty lines."""
    a = sorted(l.strip() for l in REFUTE.splitlines() if l.strip())
    b = sorted(l.strip() for l in REFUTE_TF.splitlines() if l.strip())
    if a != b:
        only_new = [x for x in b if x not in a]
        only_old = [x for x in a if x not in b]
        raise SystemExit("REFUTE_TF is no longer a pure reordering of verify_findings.REFUTE - "
                         f"the panel instrument has drifted.\n  added: {only_new}\n  lost: {only_old}")

# The rubric-anchored severity grader. The rubric text
# is inlined verbatim so the published instrument and the prompt cannot drift apart.
SEVERITY = """{rubric}

---
Apply the rubric above to ONE confirmed discrepancy in an AI-generated clinical note.

mode {mode}: {description}
note says: {note_quote}
source says: {source_quote}

Return ONLY JSON {{"severity":"critical"|"supporting"|"peripheral","reason":"<short>"}}."""


def load_rubric():
    return open(os.path.join(HERE, RUBRIC_PATH)).read().strip()


def note_index(units):
    return {u["note_key"]: u for u in units}


def consultation_order(findings):
    """Consultations in the order the panel works through them: sorted within substrate,
    then one substrate at a time in turn. The same order `_cache_order` executes in, so
    a `--consultations N` subsample and a cap-truncated run cover the same prefix."""
    by_src = defaultdict(list)
    for c in sorted({(f["source"], f["id"]) for f in findings}):
        by_src[c[0]].append(c)
    order, i = [], 0
    while any(len(v) > i for v in by_src.values()):
        for s in sorted(by_src):
            if len(by_src[s]) > i:
                order.append(by_src[s][i])
        i += 1
    return order


def main():
    ap = argparse.ArgumentParser(description="census stage 2: refute panel + salience + severity")
    ap.add_argument("--in", dest="infile", default="master/findings_master.json")
    ap.add_argument("--out", default=None, help="default: master/findings_verified_master.json")
    ap.add_argument("--route", default="openrouter", choices=["openrouter", "plan"])
    ap.add_argument("--panel-roles", default=",".join(T.PANEL_ROLES),
                    help="the cross-family skeptics (default: constructor,auditor)")
    ap.add_argument("--tiebreak-role", default=T.TIEBREAK_ROLE,
                    help="third opinion when the two skeptics disagree (default: judge-primary)")
    ap.add_argument("--tiebreak-effort", default="high")
    ap.add_argument("--no-tiebreak", action="store_true",
                    help="leave splits unresolved (they count as NOT verified) instead of buying "
                         "a third opinion")
    ap.add_argument("--salience-filter", default="high,medium",
                    help="verify only these differ-guessed saliences (default high,medium - the "
                         "June behaviour; 'all' verifies every candidate)")
    ap.add_argument("--limit", type=int, default=None,
                    help="verify only N findings, round-robin across notes (smoke)")
    ap.add_argument("--consultations", type=int, default=None,
                    help="verify only the first N CONSULTATIONS of the substrate round-robin "
                         "order - a declared stratified subsample. Use this instead of letting "
                         "the spend cap truncate the run: the cap stops mid-consultation, which "
                         "silently breaks cross-vendor matching, while this stops on whole "
                         "consultations with every vendor verified.")
    ap.add_argument("--complete", action="store_true", help="run the completeness critic")
    ap.add_argument("--rounds", type=int, default=1, help="completeness rounds (loop until dry)")
    ap.add_argument("--no-severity", action="store_true", help="skip the rubric severity grade")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=60, help="findings per Run() manifest")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--budget-usd", type=float, default=100.0)
    ap.add_argument("--stage-cap-usd", type=float, default=T.STAGE_CAP_DEFAULT,
                    help="cap on the WHOLE taxonomy stage across every script and resume")
    ap.add_argument("--warn-usd", type=float, default=200.0)
    ap.add_argument("--stop-usd", type=float, default=400.0)
    ap.add_argument("--authorize-spend", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    assert_refute_is_a_reorder()
    roles = [r.strip() for r in args.panel_roles.split(",") if r.strip()]
    if len(roles) != len(set(roles)):
        raise SystemExit(f"panel roles must be distinct (cross-family is the point): {roles}")
    if args.route == "plan" and any(r != T.CONSTRUCTOR_ROLE for r in roles):
        raise SystemExit("--route plan only reaches the constructor model, so it cannot run a "
                         "cross-family panel. Use --route openrouter, or --panel-roles constructor "
                         "to run a deliberately single-family panel.")

    inpath = os.path.join(HERE, args.infile)
    if not os.path.exists(inpath):
        raise SystemExit(f"{args.infile} does not exist - run taxonomy_discover.py first "
                         "(it writes master/findings_master.json on the canonical pass).")
    data = json.load(open(inpath))
    findings = list(data["all_issues"])
    if args.salience_filter and args.salience_filter != "all":
        keep = set(args.salience_filter.split(","))
        before = len(findings)
        findings = [f for f in findings if f.get("salience") in keep]
        print(f"salience filter {sorted(keep)}: {before} -> {len(findings)} candidates "
              f"(June verified high+medium only)")
    covered = None                     # None = every consultation in the input file
    if args.consultations:
        order = consultation_order(findings)
        keep = covered = set(order[: args.consultations])
        before, nb = len(findings), len({(f["source"], f["id"]) for f in findings})
        findings = [f for f in findings if (f["source"], f["id"]) in keep]
        by_src = Counter(s for s, _ in keep)
        print(f"consultation subsample: {len(keep)} of {nb} consultations "
              f"({before} -> {len(findings)} findings)")
        print(f"  stratified by substrate: {dict(sorted(by_src.items()))}")
        print("  frame: consultations sorted within substrate, taken one substrate at a time in "
              "turn - deterministic, so the covered set is a declared sample rather than "
              "wherever a spend cap happened to bite")
    if args.limit and args.limit < len(findings):
        # round-robin across notes so a smoke sample spans every vendor, not one note
        buckets = defaultdict(list)
        for f in findings:
            buckets[f["note_key"]].append(f)
        picked, i = [], 0
        while len(picked) < args.limit:
            row = [b[i] for b in (buckets[k] for k in sorted(buckets)) if len(b) > i]
            if not row:
                break
            picked += row[: args.limit - len(picked)]
            i += 1
        print(f"--limit {args.limit}: {len(findings)} -> {len(picked)} findings "
              f"(round-robin over {len(buckets)} notes)")
        findings = picked
    for f in findings:
        f.setdefault("discovered_by", "discover")

    units, _ = T.note_units()
    notes = note_index(units)
    # The completeness critic must see every note DISCOVERY covered, including the ones
    # that produced no candidate - a note with zero findings is exactly where a missed
    # discrepancy is most interesting. discovery's per_note block is that list.
    in_scope = sorted({p["note_key"] for p in data.get("per_note", [])}
                      | {f["note_key"] for f in findings})
    missing = [nk for nk in in_scope if nk not in notes]
    if missing:
        raise SystemExit(f"{len(missing)} audited notes are not in the corpus: {missing[:3]}")

    tag = args.tag or "verify-r1"
    store = T.RecordStore(T.EXPERIMENT, tag)
    rubric = load_rubric()
    prompts = {"refute_june_verbatim": REFUTE, "refute_transcript_first": REFUTE_TF,
               "salience": SALIENCE, "severity_rubric": SEVERITY,
               "severity_rubric_text": rubric, "completeness": COMPLETE}

    print(f"verify | route={args.route} | panel={roles} | tiebreak="
          f"{'off' if args.no_tiebreak else args.tiebreak_role} | tag={tag}")
    print(f"  {len(findings)} candidate findings over {len(in_scope)} notes "
          f"(complete={args.complete}, rounds={args.rounds}, severity={not args.no_severity})")

    guard = T.BudgetGuard(budget=args.budget_usd, warn=args.warn_usd, stop=args.stop_usd,
                          authorized=args.authorize_spend, route=args.route,
                          stage_cap=args.stage_cap_usd)
    usage = T.new_usage()
    state = {"store": store, "notes": notes, "route": args.route, "roles": roles,
             "tiebreak_role": None if args.no_tiebreak else args.tiebreak_role,
             "tiebreak_effort": args.tiebreak_effort,
             "workers": args.workers, "chunk": args.chunk, "guard": guard, "usage": usage,
             "rubric": rubric, "prompts": prompts, "allow_dirty": args.allow_dirty,
             "yes": args.yes, "no_severity": args.no_severity, "units": units,
             "covered": covered}

    run_panel(findings, state, stage="panel")

    if args.complete:
        for rnd in range(1, args.rounds + 1):
            real = [f for f in findings if f["verdict"]["is_real"]]
            new = completeness(real, in_scope, state, rnd)
            print(f"  completeness round {rnd}: {len(new)} new candidates")
            if not new:
                break
            run_panel(new, state, stage=f"complete{rnd}")
            findings += new
            if not any(f["verdict"]["is_real"] for f in new):
                break

    store.close()
    write_out(findings, data, args, state)


# ---------------------------------------------------------------- the panel
def run_panel(findings, S, stage):
    """The cross-family skeptics + salience rater, then tiebreaks on splits, then
    rubric severity on survivors. Three buying rounds, because the tiebreak set is not
    knowable until the skeptics have voted."""
    store, roles = S["store"], S["roles"]
    todo = []
    for f in findings:
        fid = f["finding_id"]
        for r in roles:
            if not store.has(f"{fid}|ref:{r}"):
                todo.append(("ref", f, r))
        if not store.has(f"{fid}|sal"):
            todo.append(("sal", f, None))
    if todo:
        print(f"  [{stage}] {len(todo)} panel calls to buy "
              f"({len(findings)} findings x {len(roles)} skeptics + 1 salience)")
        T.confirm(f"buy {len(todo)} panel calls?", S["yes"])
        _execute(todo, S, stage=f"{stage}-panel")

    for f in findings:
        attach_verdict(f, store, roles, S["tiebreak_role"])

    # round 2: one third opinion per finding the two families disagreed on
    if S["tiebreak_role"]:
        split = [f for f in findings if f["verdict"].get("split")
                 and not store.has(f"{f['finding_id']}|tie")]
        if split:
            print(f"  [{stage}] {len(split)} split verdicts -> tiebreak on "
                  f"{S['tiebreak_role']} ({len(split) / max(len(findings), 1):.1%} of findings)")
            T.confirm(f"buy {len(split)} tiebreak calls?", S["yes"])
            _execute([("tie", f, None) for f in split], S, stage=f"{stage}-tiebreak")
            for f in findings:
                attach_verdict(f, store, roles, S["tiebreak_role"])

    if not S["no_severity"]:
        survivors = [f for f in findings if f["verdict"]["is_real"]]
        sev_todo = [("sev", f, None) for f in survivors if not store.has(f"{f['finding_id']}|sev")]
        if sev_todo:
            print(f"  [{stage}] {len(sev_todo)} rubric-severity calls on panel survivors")
            T.confirm(f"buy {len(sev_todo)} severity calls?", S["yes"])
            _execute(sev_todo, S, stage=f"{stage}-severity")
        for f in survivors:
            r = store.records.get(f"{f['finding_id']}|sev")
            if r:
                f["severity_rubric"] = r.get("value")
                f["severity_rubric_reason"] = r.get("reason")
    return findings


def attach_verdict(f, store, roles, tiebreak_role):
    """The 2-of-2 cross-family rule.

    Both keep -> real. Both refute -> cut. Split -> the tiebreak's verdict decides, and
    until that call is bought the finding counts as NOT real. A missing skeptic call is
    not a vote at all: the finding stays unverified and is flagged `incomplete` so a
    re-run fills it, rather than a half-panel silently deciding.

    An unparseable or failed skeptic reply is still recorded as REFUTED (June's
    default-refute convention), which under a 2-panel means one transport failure
    produces a split rather than a silent cut - and the tiebreak then resolves it on
    evidence. That is strictly safer than June's behaviour, not looser.
    """
    fid = f["finding_id"]
    votes = {r: store.records.get(f"{fid}|ref:{r}") for r in roles}
    have = {r: v for r, v in votes.items() if v is not None}
    refuted = {r: (v.get("value") is True) for r, v in have.items()}
    nref = sum(1 for x in refuted.values() if x)
    sal = (store.records.get(f"{fid}|sal") or {}).get("value")
    tie = store.records.get(f"{fid}|tie")

    v = {"panel_roles": list(roles), "n_votes": len(have), "n_refuted": nref,
         "votes": {r: ("refute" if refuted[r] else "keep") for r in sorted(refuted)},
         "salience": sal,
         "reasons": {r: (have[r].get("reason") if have[r] else None) for r in sorted(have)}}
    if len(have) < len(roles):
        v.update({"is_real": False, "incomplete": True, "split": False,
                  "decided_by": "incomplete panel"})
    elif nref == 0:
        v.update({"is_real": True, "split": False, "decided_by": "unanimous keep"})
    elif nref == len(roles):
        v.update({"is_real": False, "split": False, "decided_by": "unanimous refute"})
    else:
        v["split"] = True
        v["dissenter"] = sorted(r for r in refuted if not refuted[r])   # who wanted to keep
        if tie is not None:
            v.update({"is_real": not bool(tie.get("value")), "decided_by": "tiebreak",
                      "tiebreak_role": tie.get("role"), "tiebreak_reason": tie.get("reason"),
                      "tiebreak_defaulted": bool(tie.get("defaulted"))})
        else:
            v.update({"is_real": False,
                      "decided_by": "unresolved split (tiebreak not bought)"})
    f["verdict"] = v
    if sal:
        f["salience"] = sal            # panel salience overrides the differ's guess (June behaviour)
    return f


def _cache_order(todo):
    """Order panel work: whole consultations, interleaved across substrates, note-blocked.

    Three properties at once, and the third is the one that would be easy to miss.

    (a) **Cache.** Every call for one note is adjacent, so the shared transcript+note
        prefix stays hot for the OpenAI legs - the only reason the panel prompt is
        transcript-first at all.
    (b) **Balance.** Consultations are round-robined across substrates, so a run the
        stage cap stops part-way has covered ACI, PriMock, authored and trap-blind
        alike rather than everything alphabetically before the letter M.
    (c) **Whole consultations.** The unit of ordering is the CONSULTATION, not the
        note, so a truncated run leaves every vendor's notes for a covered
        consultation verified together. Cross-vendor matching asks whether >=2 vendors
        independently failed on the SAME consultation; if truncation cut consultations
        in half - scribe_A verified, scribe_C not - that comparison would quietly become
        unanswerable on exactly the notes it needs. Whole consultations keep a partial
        run fully analysable, just on fewer consultations.

    Returns a flat list of per-note blocks, in consultation order.
    """
    by_note, note_consult = defaultdict(list), {}
    for job in todo:
        nk = job[1]["note_key"]
        by_note[nk].append(job)
        note_consult[nk] = (job[1]["source"], job[1]["id"])
    by_consult = defaultdict(list)
    for nk in sorted(by_note):
        by_consult[note_consult[nk]].append(nk)
    buckets = defaultdict(list)
    for c in sorted(by_consult):
        buckets[c[0]].append(c)                       # bucket by substrate
    order, i = [], 0
    while any(len(v) > i for v in buckets.values()):
        for s in sorted(buckets):
            if len(buckets[s]) > i:
                order.append(buckets[s][i])
        i += 1
    # inside a note: skeptics first (they carry the big shared prefix), then the cheap
    # prefix-less calls
    rank = {"ref": 0, "tie": 1, "sal": 2, "sev": 3}
    return [sorted(by_note[nk], key=lambda j: (rank.get(j[0], 9), j[1]["finding_id"], j[2] or ""))
            for c in order for nk in by_consult[c]]


def progress_survival(store, roles, top=8):
    """Running survival rate by frame category and by vendor, read straight off the
    record store at a chunk boundary.

    Built in because the smoke saw ZERO of 15 filtered `omission` candidates survive the
    panel, and omission is the category the
    whole thesis rests on. If that holds at scale it is either a real result or a broken
    refute standard, and the difference has to be visible while the run is happening -
    not discovered at the analysis stage with the money already spent.
    """
    by_f = defaultdict(dict)
    meta = {}
    for r in store.records.values():
        if r.get("kind") != "ref":
            continue
        by_f[r["finding_id"]][r.get("role")] = r.get("value") is True
        meta[r["finding_id"]] = (r.get("frame_tier2") or r.get("mode"), r.get("scribe"))
    cat, ven = defaultdict(lambda: [0, 0, 0]), defaultdict(lambda: [0, 0, 0])  # kept, split, total
    for fid, votes in by_f.items():
        if len(votes) < len(roles):
            continue
        n_ref = sum(1 for v in votes.values() if v)
        c, v = meta[fid]
        for d, k in ((cat, c), (ven, v)):
            d[k][2] += 1
            if n_ref == 0:
                d[k][0] += 1
            elif n_ref < len(roles):
                d[k][1] += 1
    if not cat:
        return
    print("    survival so far (both skeptics in; split = tiebreak pending):")
    rows = sorted(cat.items(), key=lambda kv: -kv[1][2])[:top]
    for k, (kept, split, tot) in rows:
        flag = "  <-- ZERO" if kept == 0 and tot >= 20 else ""
        print(f"      {k:22} {kept:4}/{tot:<4} kept ({kept / tot:5.1%})  +{split} split{flag}")
    print("      " + " | ".join(f"{k}: {a}/{c} ({a / c:.0%})" for k, (a, s, c) in sorted(ven.items())))


def _execute(todo, S, stage):
    """Buy a list of (kind, finding, role) calls, note-blocked and chunked into Run()
    manifests. Each note block runs one serial warm-up call before fanning out, so the
    provider cache is populated rather than missed N ways at once."""
    store, notes, guard, usage = S["store"], S["notes"], S["guard"], S["usage"]
    blocks = _cache_order(todo)
    t0 = time.time()
    for ci, chunk in enumerate(T.chunked(blocks, max(1, S["chunk"] // 4)), 1):
        n_calls = sum(len(b) for b in chunk)
        params = T.manifest_params(
            {"stage": stage, "panel_roles": S["roles"], "tiebreak_role": S["tiebreak_role"],
             "chunk": ci, "workers": S["workers"], "n_note_blocks": len(chunk),
             "record_store": os.path.relpath(store.path, HERE),
             "panel_prompt": "verify_findings.REFUTE, blocks reordered transcript-first "
                             "(2026-08-12) for provider prompt caching; wording unchanged",
             "severity_rubric_sha256": hashlib.sha256(S["rubric"].encode()).hexdigest()},
            S["route"], S["units"])
        with Run(T.EXPERIMENT, params=params, inputs=T.run_inputs(), spec=T.SPEC,
                 allow_dirty=S["allow_dirty"], spend=T.spend_label(S["route"])) as run:
            run.register_prompts(S["prompts"])

            def worker(job):
                kind, f, role = job
                u = notes[f["note_key"]]
                fid = f["finding_id"]
                effort = None
                if kind in ("ref", "tie"):
                    key = f"{fid}|ref:{role}" if kind == "ref" else f"{fid}|tie"
                    call_role = role if kind == "ref" else S["tiebreak_role"]
                    if kind == "tie":
                        effort = S["tiebreak_effort"]
                    prompt = REFUTE_TF.format(
                        mode=f.get("mode"), description=f.get("description"),
                        note_quote=f.get("note_quote", "-"), source_quote=f.get("source_quote", "-"),
                        note=u["note"], source=u["transcript"][:TRANSCRIPT_CHARS], skind="transcript")
                elif kind == "sal":
                    key, call_role = f"{fid}|sal", T.CONSTRUCTOR_ROLE
                    prompt = SALIENCE.format(
                        mode=f.get("mode"), description=f.get("description"),
                        note_quote=f.get("note_quote", "-"), source_quote=f.get("source_quote", "-"))
                else:
                    key, call_role = f"{fid}|sev", T.CONSTRUCTOR_ROLE
                    prompt = SEVERITY.format(
                        rubric=S["rubric"], mode=f.get("mode"), description=f.get("description"),
                        note_quote=f.get("note_quote", "-"), source_quote=f.get("source_quote", "-"))
                obj, meta = T.route_call(prompt, key, S["route"], kind="panel",
                                         role=call_role, effort=effort)
                if kind in ("ref", "tie"):
                    # default-refute on anything unparseable or failed (pre-registered, as in June)
                    value = bool(obj.get("refuted")) if isinstance(obj, dict) else True
                elif kind == "sal":
                    value = obj.get("salience") if isinstance(obj, dict) else None
                    if value not in ("high", "medium", "low"):
                        value = None
                else:
                    value = obj.get("severity") if isinstance(obj, dict) else None
                    if value not in ("critical", "supporting", "peripheral"):
                        value = None
                rec = {"key": key, "kind": kind, "role": call_role, "finding_id": fid,
                       "note_key": f["note_key"], "scribe": f["scribe"], "source": f["source"],
                       "id": f["id"], "template": f["template"], "mode": f.get("mode"),
                       "pass": f.get("pass"),
                       "frame_tier2": f.get("frame_tier2") or f.get("pass"),
                       "value": value, "reason": (obj or {}).get("reason") if isinstance(obj, dict) else None,
                       "defaulted": bool(kind in ("ref", "tie") and not isinstance(obj, dict)),
                       "route": meta.get("route"), "model": meta.get("model"),
                       "seed": meta.get("seed"), "error": meta.get("error"),
                       "usage": meta.get("usage"), "cost_usd": meta.get("cost_usd"),
                       "wall_s": meta.get("wall_s"), "run_id": run.run_id, "t": round(time.time(), 3)}
                store.put(rec)
                T.usage_add(usage, meta)
                guard.add(meta.get("cost_usd"))

            for block in chunk:
                T.run_block(block, worker, S["workers"], warmup=len(block) > 1)
            print(f"    {stage} chunk {ci}: {n_calls} calls over {len(chunk)} notes | "
                  f"{usage['errors']} errors | ${usage['cost_usd']:.2f} | "
                  f"{time.time() - t0:.0f}s | {T.cache_line(usage)}", flush=True)
            if stage.endswith("-panel"):
                progress_survival(store, S["roles"])
                print(f"      stage ${guard.stage_total:.2f} of ${guard.stage_cap:.0f}", flush=True)
            run.save("verify_chunk.json", {"stage": stage, "chunk": ci, "params": params,
                                           "n_calls": n_calls})


# ---------------------------------------------------------------- completeness critic
def completeness(real_so_far, in_scope, S, rnd):
    """Per note, hunt discrepancies the 10 passes missed. Verbatim COMPLETE prompt.

    Two ports away from verify_findings.py, both forced by scale:
      - it iterated the WHOLE loaded corpus rather than the notes under audit; here it
        iterates exactly the notes in scope, so --limit/--smoke mean what they say;
      - its per-note key was (source, id, template), which collides scribe_B and scribe_C
        (both template="audio") into one bucket; here it is the full note_key.
    """
    store, notes, usage, guard = S["store"], S["notes"], S["usage"], S["guard"]
    by_note = defaultdict(list)
    for f in real_so_far:
        by_note[f["note_key"]].append(f)
    todo = [nk for nk in in_scope if not store.has(f"{nk}|complete|rd{rnd}")]
    if not todo:
        return _harvest_completeness(store, in_scope, rnd)
    print(f"  completeness round {rnd}: {len(todo)} note calls to buy")
    T.confirm(f"buy {len(todo)} completeness calls?", S["yes"])

    for ci, chunk in enumerate(T.chunked(todo, S["chunk"]), 1):
        params = T.manifest_params({"stage": f"completeness-r{rnd}", "chunk": ci,
                                    "workers": S["workers"],
                                    "record_store": os.path.relpath(store.path, HERE)},
                                   S["route"], S["units"])
        with Run(T.EXPERIMENT, params=params, inputs=T.run_inputs(), spec=T.SPEC,
                 allow_dirty=S["allow_dirty"], spend=T.spend_label(S["route"])) as run:
            run.register_prompts(S["prompts"])

            def worker(nk):
                u = notes[nk]
                known = "; ".join(f"{f.get('mode')}: {f.get('description', '')}"
                                  for f in by_note.get(nk, [])) or "(none)"
                prompt = COMPLETE.format(known=known[:6000], note=u["note"],
                                         source=u["transcript"][:TRANSCRIPT_CHARS], skind="transcript")
                obj, meta = T.route_call(prompt, f"{nk}|complete|rd{rnd}", S["route"], kind="complete")
                miss = (obj or {}).get("missed", []) if isinstance(obj, dict) else []
                out = []
                for i, m in enumerate(miss if isinstance(miss, list) else []):
                    if not isinstance(m, dict):
                        continue
                    out.append({**m, "finding_id": f"{nk}#complete{rnd}#{i}",
                                "pass": f"completeness{rnd}", "check": "completeness",
                                "discovered_by": "completeness"})
                store.put({"key": f"{nk}|complete|rd{rnd}", "kind": "complete", "note_key": nk,
                           "scribe": u["scribe"], "source": u["source"], "id": u["id"],
                           "template": u["template"], "consultation": u["consultation"],
                           "round": rnd, "issues": out, "n_issues": len(out),
                           "route": meta.get("route"), "model": meta.get("model"),
                           "error": meta.get("error"), "usage": meta.get("usage"),
                           "cost_usd": meta.get("cost_usd"), "run_id": run.run_id,
                           "t": round(time.time(), 3)})
                T.usage_add(usage, meta)
                guard.add(meta.get("cost_usd"))

            T.run_block(chunk, worker, S["workers"], warmup=False)
            print(f"    completeness r{rnd} chunk {ci}: {len(chunk)} notes "
                  f"| ${usage['cost_usd']:.2f}", flush=True)
            run.save("completeness_chunk.json", {"round": rnd, "chunk": ci, "params": params})
    return _harvest_completeness(store, in_scope, rnd)


def _harvest_completeness(store, in_scope, rnd):
    out = []
    for nk in in_scope:
        r = store.records.get(f"{nk}|complete|rd{rnd}")
        if not r:
            continue
        for it in r["issues"]:
            out.append({"note_key": nk, "scribe": r["scribe"], "source": r["source"],
                        "id": r["id"], "template": r["template"],
                        "consultation": r["consultation"], **it})
    return out


# ---------------------------------------------------------------- output
def panel_stats(findings, S):
    """What the cross-family panel actually did."""
    n = len(findings)
    if not n:
        return {"n_findings": 0}
    split = [f for f in findings if f["verdict"].get("split")]
    dec = Counter(f["verdict"].get("decided_by") for f in findings)
    # per-role refute rate: does one family carry the skepticism?
    by_role = {}
    for r in S["roles"]:
        votes = [f["verdict"]["votes"].get(r) for f in findings if f["verdict"].get("votes")]
        votes = [v for v in votes if v]
        by_role[r] = {"n_votes": len(votes),
                      "refute_rate": round(sum(v == "refute" for v in votes) / len(votes), 4)
                      if votes else None}
    # when they split, whose side did the tiebreak take?
    tie_for = Counter()
    for f in split:
        if f["verdict"].get("decided_by") != "tiebreak":
            continue
        keeper = (f["verdict"].get("dissenter") or ["?"])[0]
        refuser = next((r for r in S["roles"] if r != keeper), "?")
        tie_for[keeper if f["verdict"]["is_real"] else refuser] += 1
    # survival by frame category x vendor - the diagnostic the omission watch needs
    surv = {}
    for key, get in (("by_frame_tier2", lambda f: f.get("frame_tier2") or f.get("pass")),
                     ("by_vendor", lambda f: f.get("scribe")),
                     ("by_tier1", lambda f: f.get("frame_tier1") or "(open)")):
        d = defaultdict(lambda: {"n": 0, "verified": 0})
        for f in findings:
            b = d[get(f)]
            b["n"] += 1
            b["verified"] += 1 if f["verdict"]["is_real"] else 0
        surv[key] = {k: {**v, "survival_rate": round(v["verified"] / v["n"], 4)}
                     for k, v in sorted(d.items(), key=lambda kv: -kv[1]["n"])}
    cross = defaultdict(lambda: {"n": 0, "verified": 0})
    for f in findings:
        b = cross[f"{f.get('frame_tier2') or f.get('pass')}|{f.get('scribe')}"]
        b["n"] += 1
        b["verified"] += 1 if f["verdict"]["is_real"] else 0
    surv["by_frame_tier2_x_vendor"] = {k: {**v, "survival_rate": round(v["verified"] / v["n"], 4)}
                                       for k, v in sorted(cross.items())}
    return {"n_findings": n,
            "survival": surv,
            "agreement_rate": round(1 - len(split) / n, 4),
            "n_split": len(split),
            "split_rate": round(len(split) / n, 4),
            "decided_by": dict(dec),
            "per_role": by_role,
            "tiebreak_sided_with": dict(tie_for),
            "tiebreak_role": S["tiebreak_role"],
            "n_defaulted_votes": sum(1 for r in S["store"].records.values()
                                     if r.get("kind") in ("ref", "tie") and r.get("defaulted")),
            "reading": "agreement_rate is how often the two families reached the same verdict "
                       "unaided; tiebreak_sided_with shows whether the third model systematically "
                       "backs the family it shares a lab with (it shares one with the auditor)."}


def write_out(findings, data, args, S):
    real = [f for f in findings if f["verdict"]["is_real"]]
    cut = [f for f in findings if not f["verdict"]["is_real"]]
    incomplete = [f for f in findings if f["verdict"].get("incomplete")]
    # Which consultations did the panel actually cover? A note in a covered consultation
    # that produced no filtered candidate is a genuine zero and belongs in the rate
    # denominator; a note in an UNCOVERED consultation was never judged and must not be,
    # or its zero would be read as "verified, found nothing". Both look identical in the
    # per_note rows unless coverage is written down here.
    covered = S["covered"]
    if covered is None:
        covered = {(f["source"], f["id"]) for f in findings} | \
                  {(p["source"], p["id"]) for p in data.get("per_note", [])}
    per_note = {}
    for p in data.get("per_note", []):
        per_note[p["note_key"]] = {**p, "n_verified": 0, "n_cut": 0,
                                   "panel_covered": (p["source"], p["id"]) in covered}
    for f in findings:
        p = per_note.setdefault(f["note_key"], {
            "note_key": f["note_key"], "scribe": f["scribe"], "source": f["source"],
            "id": f["id"], "template": f["template"], "consultation": f["consultation"],
            "n_issues": 0, "n_verified": 0, "n_cut": 0, "panel_covered": True})
        p["panel_covered"] = True
        p["n_verified" if f["verdict"]["is_real"] else "n_cut"] += 1

    out = {"stage": "verify", "tag": args.tag or "verify-r1", "source_file": args.infile,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "panel": len(S["roles"]), "panel_roles": S["roles"],
           "tiebreak_role": S["tiebreak_role"], "route": args.route,
           "salience_filter": args.salience_filter,
           "consultation_subsample": args.consultations,
           "n_consultations_covered": len(covered),
           "n_notes_covered": sum(1 for p in per_note.values() if p["panel_covered"]),
           "coverage_note": "rates must be computed over panel_covered notes only; an "
                            "uncovered note has 0 verified findings because it was never "
                            "judged, not because it was clean",
           "panel_stats": panel_stats(findings, S),
           "prompt_cache": {"cached_tokens": S["usage"].get("cached_tokens", 0),
                            "prompt_tokens": S["usage"].get("prompt_tokens", 0)},
           "severity_rubric": RUBRIC_PATH,
           "severity_rubric_sha256": hashlib.sha256(S["rubric"].encode()).hexdigest(),
           "completeness_rounds": args.rounds if args.complete else 0,
           "n_candidates": len(findings), "n_verified": len(real), "n_cut": len(cut),
           "n_incomplete_panels": len(incomplete),
           "cost_usd": round(S["usage"]["cost_usd"], 6),
           "per_note": sorted(per_note.values(), key=lambda p: p["note_key"]),
           "all_issues": findings}
    path = os.path.join(HERE, args.out) if args.out else \
        os.path.join(T.MASTER, "findings_verified_master.json" if not args.tag or args.tag == "verify-r1"
                     else f"findings_verified_{args.tag}.json")
    json.dump(out, open(path, "w"), indent=1)

    rate = len(cut) / len(findings) if findings else 0
    ps = out["panel_stats"]
    print(f"\n=== verified: {len(real)}/{len(findings)} findings survived "
          f"(refute rate {rate:.1%}) ===")
    print(f"panel: {ps.get('agreement_rate', 0):.1%} cross-family agreement, "
          f"{ps.get('n_split', 0)} splits ({ps.get('split_rate', 0):.1%}) -> tiebreak")
    for r, b in (ps.get("per_role") or {}).items():
        print(f"  {r:14} refuted {b['refute_rate']:.1%} of {b['n_votes']} votes"
              if b["refute_rate"] is not None else f"  {r:14} no votes")
    if ps.get("tiebreak_sided_with"):
        print(f"  tiebreak ({ps['tiebreak_role']}) sided with: {ps['tiebreak_sided_with']}")
    if ps.get("n_defaulted_votes"):
        print(f"  ! {ps['n_defaulted_votes']} default-refute votes (unparseable/failed calls)")
    print(f"  {T.cache_line(S['usage'])}")
    print("\nsurvival by frame category:")
    for k, b in ps["survival"]["by_frame_tier2"].items():
        flag = ""
        if b["n"] >= 20 and b["survival_rate"] <= 0.02:
            flag = "   <-- NEAR-ZERO: run taxonomy_omission_audit.py before trusting this rate"
        print(f"  {k:22} {b['verified']:5}/{b['n']:<5} = {b['survival_rate']:6.1%}{flag}")
    if findings and (rate == 0 or rate == 1):
        print("  ! RED FLAG: the panel refuted everything or nothing - check the prompts/route "
              "before trusting these verdicts (June reference: 62.7% refuted).")
    if incomplete:
        print(f"  ! {len(incomplete)} findings have an incomplete panel (<{len(S['roles'])} votes) "
              "- re-run to fill them; they are counted as NOT verified for now.")
    print("survived by mode:     ", dict(Counter(f.get("mode") for f in real).most_common(12)))
    print("survived by salience: ", dict(Counter(f.get("salience") for f in real)))
    print("severity (rubric):    ", dict(Counter(f.get("severity_rubric") for f in real)))
    print("survived by vendor:   ", dict(Counter(f["scribe"] for f in real)))
    print("cut by mode:          ", dict(Counter(f.get("mode") for f in cut).most_common(8)))
    print("survived by tier1:    ", dict(Counter(f.get("frame_tier1") or "(open)"
                                                 for f in real).most_common()))
    print(f"cost this session: ${S['usage']['cost_usd']:.4f} over {S['usage']['calls']} calls "
          f"| stage ${S['guard'].stage_total:.2f} of ${S['guard'].stage_cap:.0f}")
    print(f"saved -> {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    main()
