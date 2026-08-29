"""W-D step 5 - support/materiality/leakage critic panel over EXTRACTED fact sheets.

Fork of critique_scenarios.py per spec 3.2 (the extracted-sheet panel) + Amendment 2026-07-29b
part H (materiality gains check (d) on `importance` grades). Differences from the authored panel:
  - support     = the existing `factsheet` critic VERBATIM (internal consistency).
  - materiality = new: clinical materiality of the sheet (padding / missing load-bearing facts /
                  wrong load_bearing grades / wrong importance grades).
  - leakage     = new: anything in the sheet not derivable from the transcript alone.
  - revise loop = critique_scenarios.revise() with ONE named change: "The TRANSCRIPT is immutable
                  and must be returned byte-identical; fix the fact_sheet only." Byte-identity is
                  also ENFORCED in code: the original transcript is always carried forward, and a
                  model that returned a drifted transcript is logged (`transcript_drift`).
  - re-audit    = support critic on the revised sheet (the existing recheck pattern).
  - gate 2      = consults still carrying material issues after ONE revision cycle (revise failed,
                  or the re-audit finds material issues) are DROPPED and logged. Drop-rate >=10%
                  per source means systematic extraction-prompt failure -> STOP, human review.

Amendment 2026-08-09b additions:
  - TRANSCRIPT-ONLY VIEWS (part T1, PriMock): the primock57_parsed.json presenting_complaint
    header is actor case-card metadata, not ground truth. For --source primock every critic,
    revise and recheck view carries id + transcript + fact_sheet ONLY - the header line is
    struck from _scn_text and from the revise output schema. (The v2.1 extraction records no
    longer carry the field at all; the strip here is belt-and-braces and also covers any stale
    record.) Other sources' views are unchanged.
  - EXCISE RULE (part T2, all extracted strata): after the one revision cycle, if a sheet's
    ONLY residual material issues are contested must_not_contain items (the critic holds the
    transcript partly/actually supports the assertion, i.e. a faithful note would be wrongly
    penalised), those items are EXCISED and the sheet is KEPT (status `revised-kept-excised`).
    Deleting a contested must_not_contain item can never make a sheet WRONG - it only stops the
    item penalising anyone. Any residual touching must_contain or salience_traps still drops
    the sheet, as does a revise that never produced a sheet. A dedicated classifier call maps
    each residual to the section it touches + the contested item indices; the excision is
    applied mechanically and logged per item to master/excised_mnc_items.json.

Model: `claude-opus-5` per Amendment 2026-07-30 part M.

TRANSPORT ROUTE (`--route plan|openrouter`, default plan). The critic panel AUDITS the artifact
rather than producing it, so it may ride a different transport from the extraction layer without
touching artifact provenance - the extraction layer stays wholly on the plan path (`claude -p`)
so the ground-truth annotation layer is one instrument end to end. `--route openrouter` calls
https://openrouter.ai/api/v1 with model `anthropic/claude-opus-5` (same pinned model, different
transport), which buys ~3x the parallelism of the plan path for the highest-volume layer. The
route actually used is recorded per consult in the state file, per kept record in the output
(`critique_provenance`), and in the report header; OpenRouter usage/credit cost is appended to
master/critique_or_spend.jsonl per invocation and enforced against --spend-cap (cumulative).

Usage: python critique_extracted.py --source primock|aci|trapblind|authored
         [--route plan|openrouter] [--workers N] [--no-revise] [--max-seconds S]
         [--spend-cap USD]
Input : master/fact_sheets_raw_<stem>.json   (from extract_fact_sheets.py)
Output: master/fact_sheets_<stem>.json       (kept consults, revised where flagged)
        master/critique_extracted_<stem>.md  (audit-trail report)
        master/critique_state_<stem>.json    (resumable per-consult state - idempotent reruns)
        master/excised_mnc_items.json        (per-item excision log, all sources merged)
        master/critique_or_spend.jsonl       (OpenRouter usage ledger, --route openrouter only)
stem = source, except authored -> `authored_extracted` (spec Amendment 2026-07-30 part N).

--max-seconds S bounds one invocation: critic/revise/recheck jobs still pending when the budget
is spent are left for the next run and the output stages are SKIPPED (so a partial run can never
write a partial fact_sheets_<stem>.json or mis-label a consult critic-error). Rerun to continue.
"""
import hashlib, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import common
from common import claude_json, validate_fact_sheet, HERE

MODEL = "claude-opus-5"   # Amendment 2026-07-30 part M
OR_MODEL = "anthropic/claude-opus-5"   # same pinned model, OpenRouter transport
EFFORT = "medium"
CRITIC_ATTEMPTS = 3   # per critic call, within one run (claude_json itself retries once more)
OR_ATTEMPTS = 4       # HTTP/parse retries inside one OpenRouter call, exponential backoff
T0 = time.time()


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


SOURCE = _arg("--source")
ROUTE = _arg("--route", "plan")
assert ROUTE in ("plan", "openrouter"), "usage: --route plan|openrouter"
# the openrouter pool takes 16 workers (3x the plan path); the plan path stays at 5 unless told
WORKERS = int(_arg("--workers", "16" if ROUTE == "openrouter" else "5"))
MAX_SECONDS = float(_arg("--max-seconds", "0"))
TIMEOUT = float(_arg("--timeout", "420"))   # per-call wall clock; lower it to bound a chunk's drain
REVISE_TIMEOUT = float(_arg("--revise-timeout", "540"))   # revise echoes the transcript - slower
REVISE_TRIES = int(_arg("--revise-tries", "3"))   # TRANSPORT retries, not extra revision cycles
SPEND_CAP = float(_arg("--spend-cap", "80"))
REVISE = "--no-revise" not in sys.argv
ROUTE_MODEL = OR_MODEL if ROUTE == "openrouter" else MODEL
STEMS = {"primock": "primock", "aci": "aci", "trapblind": "trapblind",
         "authored": "authored_extracted"}
assert SOURCE in STEMS, "usage: --source primock|aci|trapblind|authored"
STEM = STEMS[SOURCE]
# Amendment 2026-08-09b T1: PriMock's presenting_complaint header is actor case-card metadata -
# struck from every critic/revise view (and absent from v2.1 extraction records to begin with).
TRANSCRIPT_ONLY = SOURCE == "primock"
INFILE = os.path.join(HERE, "master", f"fact_sheets_raw_{STEM}.json")
OUTFILE = os.path.join(HERE, "master", f"fact_sheets_{STEM}.json")
REPORT = os.path.join(HERE, "master", f"critique_extracted_{STEM}.md")
STATE = os.path.join(HERE, "master", f"critique_state_{STEM}.json")
EXCISED_LOG = os.path.join(HERE, "master", "excised_mnc_items.json")
SPEND_LEDGER = os.path.join(HERE, "master", "critique_or_spend.jsonl")
BUDGET_HIT = [0]


def out_of_budget():
    if MAX_SECONDS and time.time() - T0 > MAX_SECONDS:
        BUDGET_HIT[0] += 1
        return True
    return False


# ---------------------------------------------------------------- OpenRouter transport
# Reuses common._openrouter_call (the W1 client: raises on HTTP/API error and on an all-empty
# completion, so an empty string can never look like a verdict). Additive - common.py untouched.
_OR_LOCK = threading.Lock()
OR_USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
            "cost_usd": 0.0, "retries": 0, "providers": {}, "returned_models": {}}
PRIOR_SPEND = 0.0          # cumulative OpenRouter credits already spent by earlier invocations
SPEND_TRIPPED = [0]


def _prior_spend():
    """Cumulative OpenRouter cost from this script's ledger, so --spend-cap holds across the
    several blocking chunks a big source is run in (each chunk is a fresh process)."""
    tot = 0.0
    if os.path.exists(SPEND_LEDGER):
        for line in open(SPEND_LEDGER):
            try:
                tot += json.loads(line).get("cost_usd") or 0.0
            except Exception:
                pass
    return tot


def _spent():
    return PRIOR_SPEND + OR_USAGE["cost_usd"]


def _or_json(prompt, effort, timeout, max_tokens):
    """One critic/revise/recheck call on `anthropic/claude-opus-5` via OpenRouter; returns the
    first JSON blob in the completion, or None (caller treats None as a failed call and retries
    on a later invocation - the same failure mode as a plan-path miss). `effort` maps the spec's
    plan-path --effort onto OpenRouter's unified reasoning block."""
    body = {"model": OR_MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "reasoning": {"effort": effort},
            "usage": {"include": True}}
    for attempt in range(OR_ATTEMPTS):
        with _OR_LOCK:
            if _spent() >= SPEND_CAP:
                SPEND_TRIPPED[0] += 1
                return None                      # hard stop: cost model wrong, report don't burn
        try:
            j = common._openrouter_call(body, timeout)
        except Exception:
            with _OR_LOCK:
                OR_USAGE["retries"] += 1
            time.sleep(min(2 ** attempt, 30))
            continue
        u = j.get("usage") or {}
        with _OR_LOCK:
            OR_USAGE["calls"] += 1
            OR_USAGE["prompt_tokens"] += u.get("prompt_tokens") or 0
            OR_USAGE["completion_tokens"] += u.get("completion_tokens") or 0
            OR_USAGE["reasoning_tokens"] += (u.get("completion_tokens_details") or {}
                                             ).get("reasoning_tokens") or 0
            OR_USAGE["cost_usd"] += u.get("cost") or 0.0
            for field, val in (("providers", j.get("provider")),
                               ("returned_models", j.get("model"))):
                if val:
                    OR_USAGE[field][val] = OR_USAGE[field].get(val, 0) + 1
        txt = (j["choices"][0]["message"]["content"] or "").strip()
        m = re.search(r"(\{.*\}|\[.*\])", txt, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        with _OR_LOCK:
            OR_USAGE["retries"] += 1
    return None


def call_json(prompt, effort, timeout, max_tokens, retries=1):
    """Route-dispatching JSON call. plan -> common.claude_json (`claude -p`, the construction
    transport); openrouter -> _or_json. Same pinned model either way."""
    if ROUTE == "openrouter":
        return _or_json(prompt, effort, timeout, max_tokens)
    return claude_json(prompt, model=MODEL, effort=effort, timeout=timeout, retries=retries)


def log_spend():
    if ROUTE != "openrouter" or not OR_USAGE["calls"]:
        return
    with open(SPEND_LEDGER, "a") as f:
        f.write(json.dumps({
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": SOURCE, "model": OR_MODEL, "route": "openrouter",
            "wall_s": round(time.time() - T0), **OR_USAGE}) + "\n")
    print(f"\n[openrouter] {OR_USAGE['calls']} calls | in {OR_USAGE['prompt_tokens']:,} tok | "
          f"out {OR_USAGE['completion_tokens']:,} tok (reasoning {OR_USAGE['reasoning_tokens']:,}) | "
          f"cost this run ${OR_USAGE['cost_usd']:.4f} | cumulative ${_spent():.4f} "
          f"(cap ${SPEND_CAP:.0f}) | retries {OR_USAGE['retries']} | "
          f"providers {OR_USAGE['providers']}", flush=True)
    if SPEND_TRIPPED[0]:
        print(f"\n*** SPEND CAP ${SPEND_CAP:.0f} HIT - {SPEND_TRIPPED[0]} calls refused. "
              f"STOP and review the cost model before rerunning. ***", flush=True)

# Spec 3.2, verbatim. `support` is critique_scenarios.py CRITICS["factsheet"] unchanged;
# `materiality` carries Amendment 2026-07-29b H's check (d).
CRITICS = {
 "support": """You are auditing INTERNAL CONSISTENCY between the transcript and its fact_sheet.
Check three things and flag any failure: (a) every `must_contain` item is genuinely supported by the
transcript (flag any that isn't actually said); (b) every `salience_trap` corresponds to a REAL moment
in the dialogue where a scribe could plausibly get it wrong (flag any trap with no such moment); (c)
every `must_not_contain` is a genuine ERROR for this consult (flag any that the transcript actually
supports, i.e. would be wrongly penalised). Also flag any internal contradiction.""",
 "materiality": """You are a senior UK GP reviewing a fact sheet extracted from a consultation transcript for use in
scoring AI scribe notes. Judge CLINICAL MATERIALITY, not support (support is audited separately).
Flag: (a) any `must_contain` item that is trivia a competent clinician would NOT expect in the note
(padding inflates completeness scoring and punishes good notes); (b) any clinically LOAD-BEARING fact
that is plainly present in the transcript but MISSING from `must_contain` (a working diagnosis, a red
flag safety-net, a drug dose, a stated allergy, a decisive negative); (c) any `load_bearing: "high"`
grade that is clearly wrong in either direction; (d) any `importance` grade that is clearly wrong in
either direction - in particular any trap graded `peripheral` that a careful clinician would treat as
a missed diagnosis, drug, or safety-net. Be strict but fair - style is not materiality.""",
 "leakage": """You are auditing a fact sheet extracted from a consultation TRANSCRIPT. Flag any item (in
must_contain, must_not_contain, or salience_traps) that relies on information NOT present in the
transcript: outside clinical knowledge asserted as fact, content that could only come from a separate
reference note or from knowing how downstream scribes typically fail, meta/test-artifact language
("the trap here", "the scribe should"), or paraphrase drift where the `evidence` quote does not
actually appear in the transcript or does not support the `fact`. A clean fact sheet is fully
derivable from the transcript alone by a careful clinician.""",
}


def _fs_sha(rec):
    return hashlib.sha256(json.dumps(rec["fact_sheet"], sort_keys=True).encode()).hexdigest()[:16]


def _scn_text(rec, fact_sheet=None):
    fs = fact_sheet if fact_sheet is not None else rec["fact_sheet"]
    # T1: the presenting_complaint line appears ONLY when the source's record view carries the
    # field legitimately - never for primock (case-card metadata), never for a record without it.
    pc = ("" if TRANSCRIPT_ONLY or "presenting_complaint" not in rec
          else f"presenting_complaint: {rec.get('presenting_complaint', '')}\n")
    return (f"id: {rec['id']}\n{pc}\n"
            f"TRANSCRIPT:\n{rec['transcript']}\n\n"
            f"FACT_SHEET:\n{json.dumps(fs, ensure_ascii=False, indent=1)}")


def run_critic(rec, name, fact_sheet=None):
    prompt = (f"{CRITICS[name]}\n\n{_scn_text(rec, fact_sheet)}\n\n"
              'Output ONLY JSON: {"verdict": "ok" | "issues", "issues": [ {"what": <concise issue>, '
              '"severity": "minor"|"material"} ]}. Empty issues list if verdict is ok.')
    for attempt in range(CRITIC_ATTEMPTS):
        r = call_json(prompt, EFFORT, TIMEOUT, max_tokens=8000, retries=1)
        if r and r.get("verdict") in ("ok", "issues"):
            r.setdefault("issues", [])
            r["route"] = ROUTE
            r["model"] = ROUTE_MODEL
            return r
        if SPEND_TRIPPED[0]:
            break
        # a retry chain must not blow the chunk's wall-clock budget: bail and let the next
        # invocation pick this call up (errored critics are dropped from state on reload)
        if attempt + 1 < CRITIC_ATTEMPTS and MAX_SECONDS and time.time() - T0 > MAX_SECONDS:
            break
    return {"verdict": "error", "issues": [], "route": ROUTE, "model": ROUTE_MODEL}


def revise(rec, all_issues):
    """critique_scenarios.revise() with the spec 3.2 named change: the immutable-transcript rule."""
    issues_txt = "\n".join(f"- [{i.get('severity','?')}] ({src}) {i.get('what','')}"
                           for src, items in all_issues.items() for i in items)
    prompt = (f"Here is a mock GP consultation scenario and the issues a review panel found. FIX ONLY "
              f"these issues; keep everything else (style, length, the deliberate salience traps) "
              f"intact. The TRANSCRIPT is immutable and must be returned byte-identical; fix the "
              f"fact_sheet only. "
              f"Maintain strict internal consistency: the fact_sheet must derive only from the "
              f"transcript.\n\n"
              f"{_scn_text(rec)}\n\nISSUES TO FIX:\n{issues_txt}\n\n"
              f"Output ONLY the full corrected JSON object with keys: id, "
              + ("" if TRANSCRIPT_ONLY or "presenting_complaint" not in rec
                 else "presenting_complaint, ")
              + f"transcript, fact_sheet. "
              f"Escape newlines inside strings as \\n.")
    # effort high on revise() = the existing harness behaviour (spec section 4); the revised object
    # echoes the whole transcript back, so max_tokens has to clear transcript + fact sheet
    obj = call_json(prompt, "high", REVISE_TIMEOUT, max_tokens=32000, retries=2)
    if obj and "transcript" in obj and "fact_sheet" in obj:
        # A revision must satisfy the SAME schema contract extraction is held to (shared
        # common.validate_fact_sheet). Observed drift when this was unguarded: revisers invent
        # out-of-vocabulary values - load_bearing "low" (schema is high|medium) and salience_trap
        # modes outside the 10-mode list - usually while acting on a materiality downgrade request.
        # An out-of-schema sheet is not a usable fact sheet, so it is not a successful revision.
        probs = validate_fact_sheet(obj["fact_sheet"])
        if probs:
            return None, False, probs
        drift = obj["transcript"] != rec["transcript"]
        final = dict(rec)                      # carries the ORIGINAL transcript + metadata forward,
        final["fact_sheet"] = obj["fact_sheet"]  # so transcript immutability holds by construction
        return final, drift, []
    return None, False, ["revise returned no usable JSON object"]


def _resid_sha(revised_fs, residuals):
    """Key an excise decision to the exact (revised sheet, residual set) it classified - a
    re-run recheck or a re-revision invalidates the stored decision."""
    blob = json.dumps({"fs": revised_fs, "resid": sorted(residuals)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


EXCISE_PROMPT = """You are auditing the residual findings of a clinical fact-sheet review panel, to apply one
narrow rescue rule (study spec, Amendment 2026-08-09b part T2):

  A fact sheet that still carries material issues after its one revision cycle is normally
  DROPPED. But if its ONLY residual material issues are CONTESTED must_not_contain items -
  the reviewer holds that the transcript partly or actually supports the forbidden assertion,
  so a faithful note would be wrongly penalised - those items can simply be DELETED and the
  sheet kept: removing a must_not_contain item can never make the sheet wrong, it only stops
  the item penalising anyone. Any residual whose substance is a defect in a must_contain item
  or a salience_trap (or anything deletion of must_not_contain items cannot fully resolve)
  still drops the sheet.

Below are the residual findings and the revised fact sheet they refer to. For EACH residual,
decide what it touches:

- "contested_mnc": the residual's substance is that one or more must_not_contain items are
  defective (partly supported by the transcript / would wrongly penalise a faithful note /
  stated too absolutely / in tension with a CORRECT must_contain item). Deleting the named
  must_not_contain item(s) fully resolves the residual. A residual that merely CITES a
  must_contain item or trap as being correct while contesting the must_not_contain item is
  still "contested_mnc".
- "touches_mc_or_traps": the residual asserts a defect in a must_contain item or a
  salience_trap themselves (wrong, unsupported, self-contradictory, needing rewording) - even
  if it also mentions must_not_contain items.
- "other": anything else (a defect no must_not_contain deletion can resolve).

RESIDUAL FINDINGS (numbered):
{residuals}

REVISED FACT SHEET must_not_contain (numbered):
{mnc}

REVISED FACT SHEET must_contain (numbered, for reference):
{mc}

REVISED FACT SHEET salience_traps (numbered, for reference):
{traps}

Output ONLY JSON:
{"classifications": [ {"residual": <residual number>, "verdict": "contested_mnc" |
"touches_mc_or_traps" | "other", "mnc_indices": [<numbers of the must_not_contain items this
residual contests; empty unless verdict is contested_mnc>], "reason": "<one line>"} ]}"""


def classify_residuals(rec, revised_fs, residuals):
    """One classifier call mapping each residual material issue to the sheet section it touches.
    Returns the classifications list, or None on transport/parse failure (caller defers)."""
    def numbered(items):
        return "\n".join(f"{i + 1}. {json.dumps(t, ensure_ascii=False)}"
                         for i, t in enumerate(items)) or "(none)"
    prompt = (EXCISE_PROMPT
              .replace("{residuals}", numbered(residuals))
              .replace("{mnc}", numbered([f"{x['assertion']} (why_wrong: {x['why_wrong']})"
                                          for x in revised_fs["must_not_contain"]]))
              .replace("{mc}", numbered([x["fact"] for x in revised_fs["must_contain"]]))
              .replace("{traps}", numbered([x["trap"] for x in revised_fs["salience_traps"]])))
    for attempt in range(CRITIC_ATTEMPTS):
        r = call_json(prompt, EFFORT, TIMEOUT, max_tokens=8000, retries=1)
        cls = (r or {}).get("classifications")
        if isinstance(cls, list) and len(cls) == len(residuals):
            ok = all(isinstance(c, dict)
                     and isinstance(c.get("residual"), int)
                     and c.get("verdict") in ("contested_mnc", "touches_mc_or_traps", "other")
                     and isinstance(c.get("mnc_indices", []), list)
                     and all(isinstance(i, int) and 1 <= i <= len(revised_fs["must_not_contain"])
                             for i in c.get("mnc_indices", []))
                     for c in cls) \
                 and {c["residual"] for c in cls} == set(range(1, len(residuals) + 1))
            if ok:
                return cls
        if SPEND_TRIPPED[0]:
            break
        if attempt + 1 < CRITIC_ATTEMPTS and MAX_SECONDS and time.time() - T0 > MAX_SECONDS:
            break
    return None


def _note_route(e):
    """Accumulate the (route, model) pairs a consult's critique actually used - a source run in
    several chunks, or resumed on the other transport, stays honestly labelled."""
    seen = e.setdefault("transports", [])
    tag = {"route": ROUTE, "model": ROUTE_MODEL}
    if tag not in seen:
        seen.append(tag)


def main():
    global PRIOR_SPEND
    PRIOR_SPEND = _prior_spend()
    recs = json.load(open(INFILE))
    by_id = {r["id"]: r for r in recs}
    print(f"[{SOURCE}] critiquing {len(recs)} extracted fact sheets on {ROUTE_MODEL} "
          f"(route={ROUTE}, {WORKERS} workers, revise={REVISE}, critics={list(CRITICS)})"
          + (f" | prior openrouter spend ${PRIOR_SPEND:.4f} of ${SPEND_CAP:.0f} cap"
             if ROUTE == "openrouter" else ""), flush=True)

    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    # discard state entries whose input fact_sheet changed (re-extraction) or whose critic errored
    for rid, rec in by_id.items():
        e = state.get(rid)
        if e and e.get("fs_sha") != _fs_sha(rec):
            state.pop(rid, None)
    for e in state.values():
        e["critics"] = {c: r for c, r in e.get("critics", {}).items() if r.get("verdict") != "error"}
        # an errored re-audit is a transport failure, not a verdict - drop it so the rerun retries
        # it (otherwise the consult is stuck at status critic-error forever and never scores)
        if e.get("recheck", {}).get("verdict") == "error":
            e.pop("recheck")

    lock = threading.Lock()

    def save():
        tmp = STATE + ".tmp"
        json.dump(state, open(tmp, "w"), ensure_ascii=False, indent=1)
        os.replace(tmp, STATE)

    # ---- stage 1: the three critics per consult (skip results already in state) ----
    jobs = [(r, c) for r in recs for c in CRITICS
            if c not in state.get(r["id"], {}).get("critics", {})]
    print(f"stage 1: {len(jobs)} critic calls to run", flush=True)
    n = [0]

    def crit_job(job):
        rec, cname = job
        if out_of_budget():
            return
        res = run_critic(rec, cname)
        with lock:
            e = state.setdefault(rec["id"], {"fs_sha": _fs_sha(rec), "critics": {}})
            e["critics"][cname] = res
            _note_route(e)
            n[0] += 1
            flags = sum(1 for i in res["issues"] if i.get("severity") == "material")
            print(f"  [{n[0]}/{len(jobs)}] {rec['id']:28} {cname:11} {res['verdict']}"
                  + (f" ({flags} material)" if flags else ""), flush=True)
            save()

    if jobs:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(crit_job, jobs))
    if BUDGET_HIT[0]:
        print(f"\nBUDGET HIT after {time.time() - T0:.0f}s in stage 1 ({BUDGET_HIT[0]} critic "
              f"calls deferred) - output stages SKIPPED; rerun to continue.", flush=True)
        return

    errored = [rid for rid, e in state.items()
               if any(r.get("verdict") == "error" for r in e["critics"].values())
               or len(e["critics"]) < len(CRITICS)]
    if errored:
        print(f"\nCRITIC CALLS STILL FAILING (excluded from output; rerun to retry): {errored}",
              flush=True)

    # ---- stage 2: one revision cycle for any consult with material issues ----
    to_revise = []
    for rec in recs:
        e = state.get(rec["id"])
        if not e or rec["id"] in errored:
            continue
        material = {c: [i for i in r.get("issues", []) if i.get("severity") == "material"]
                    for c, r in e["critics"].items()}
        e["material"] = {c: v for c, v in material.items() if v}
        # ONE revision cycle per spec 3.2 - but a revise call that never returned a usable object
        # is a TRANSPORT failure, not a failed revision cycle, so it is re-attempted (bounded by
        # --revise-tries) rather than counted as a gate-2 drop.
        rv = e.get("revise")
        if REVISE and e["material"] and (rv is None or
                                         (not rv.get("ok") and rv.get("tries", 1) < REVISE_TRIES)):
            to_revise.append(rec)
    print(f"\nstage 2: flagged for revision (material issues): "
          f"{[r['id'] for r in to_revise] or 'none'}", flush=True)

    def rev_job(rec):
        e = state[rec["id"]]
        if out_of_budget():
            return
        final, drift, probs = revise(rec, e["material"])
        with lock:
            _note_route(e)
            if final is None:
                tries = (e.get("revise") or {}).get("tries", 0) + 1
                e["revise"] = {"ok": False, "tries": tries, "route": ROUTE, "model": ROUTE_MODEL,
                               "schema_problems": probs}
                print(f"  {rec['id']}: REVISE REJECTED (try {tries}/{REVISE_TRIES}): "
                      f"{probs[0] if probs else '?'}", flush=True)
            else:
                e["revise"] = {"ok": True, "transcript_drift": drift, "route": ROUTE,
                               "model": ROUTE_MODEL, "fact_sheet": final["fact_sheet"]}
                if drift:
                    print(f"  {rec['id']}: revised (transcript drift RESTORED to original)", flush=True)
                else:
                    print(f"  {rec['id']}: revised", flush=True)
            save()

    if to_revise:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(rev_job, to_revise))
    if BUDGET_HIT[0]:
        print(f"\nBUDGET HIT after {time.time() - T0:.0f}s in stage 2 - output stages SKIPPED; "
              f"rerun to continue.", flush=True)
        return

    # ---- stage 3: re-audit revised sheets with the support critic ----
    to_recheck = [r for r in recs
                  if state.get(r["id"], {}).get("revise", {}).get("ok")
                  and "recheck" not in state[r["id"]]]
    print(f"stage 3: re-auditing {len(to_recheck)} revised sheets (support critic)", flush=True)

    def recheck_job(rec):
        e = state[rec["id"]]
        if out_of_budget():
            return
        res = run_critic(rec, "support", fact_sheet=e["revise"]["fact_sheet"])
        with lock:
            _note_route(e)
            e["recheck"] = res
            resid = sum(1 for i in res["issues"] if i.get("severity") == "material")
            print(f"  {rec['id']}: re-audit {res['verdict']} ({resid} residual material)", flush=True)
            save()

    if to_recheck:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(recheck_job, to_recheck))
    if BUDGET_HIT[0]:
        print(f"\nBUDGET HIT after {time.time() - T0:.0f}s in stage 3 - output stages SKIPPED; "
              f"rerun to continue.", flush=True)
        return

    # ---- stage 3b: excise pass (Amendment 2026-08-09b part T2) ----
    # For every consult whose re-audit left residual material issues: classify each residual by
    # the section it touches. Only if ALL residuals are contested must_not_contain items is the
    # sheet rescued (stage 4 deletes those items and keeps it); anything touching must_contain
    # or salience_traps keeps the drop. Idempotent: the stored decision is keyed to the exact
    # (revised sheet, residual set) via _resid_sha.
    def _residuals(e):
        return [i.get("what", "") for i in e.get("recheck", {}).get("issues", [])
                if i.get("severity") == "material"]

    to_excise = []
    for rec in recs:
        e = state.get(rec["id"], {})
        if not e.get("revise", {}).get("ok") or not e.get("recheck"):
            continue
        resid = _residuals(e)
        if not resid:
            continue
        sha = _resid_sha(e["revise"]["fact_sheet"], resid)
        if e.get("excise", {}).get("resid_sha") != sha:
            to_excise.append(rec)
    print(f"stage 3b: excise-classifying {len(to_excise)} sheets with residual material issues",
          flush=True)

    def excise_job(rec):
        e = state[rec["id"]]
        if out_of_budget():
            return
        revised_fs = e["revise"]["fact_sheet"]
        resid = _residuals(e)
        cls = classify_residuals(rec, revised_fs, resid)
        with lock:
            _note_route(e)
            if cls is None:
                e.pop("excise", None)   # transport/parse failure - rerun retries
                print(f"  {rec['id']}: excise classifier FAILED (rerun to retry)", flush=True)
                save()
                return
            all_mnc = all(c["verdict"] == "contested_mnc" and c.get("mnc_indices") for c in cls)
            idxs = sorted({i for c in cls for i in c.get("mnc_indices", [])})
            if all_mnc and idxs:
                fs2 = {k: (list(v) if isinstance(v, list) else v) for k, v in revised_fs.items()}
                fs2["must_not_contain"] = [x for j, x in enumerate(revised_fs["must_not_contain"], 1)
                                           if j not in idxs]
                probs = validate_fact_sheet(fs2)
                excised = [{"mnc_index": j,
                            "assertion": revised_fs["must_not_contain"][j - 1]["assertion"],
                            "why_wrong": revised_fs["must_not_contain"][j - 1]["why_wrong"],
                            "critic_reason": "; ".join(
                                resid[c["residual"] - 1] for c in cls
                                if j in c.get("mnc_indices", []) and 1 <= c["residual"] <= len(resid))}
                           for j in idxs]
                if probs:   # an excised sheet must still satisfy the shared schema contract
                    e["excise"] = {"resid_sha": _resid_sha(revised_fs, resid), "decision": "drop",
                                   "classifications": cls, "route": ROUTE, "model": ROUTE_MODEL,
                                   "note": f"excision would break schema: {probs[0]}"}
                    print(f"  {rec['id']}: excise would break schema -> drop stands", flush=True)
                else:
                    e["excise"] = {"resid_sha": _resid_sha(revised_fs, resid),
                                   "decision": "excise", "classifications": cls,
                                   "excised": excised, "fact_sheet": fs2,
                                   "route": ROUTE, "model": ROUTE_MODEL}
                    print(f"  {rec['id']}: EXCISED {len(idxs)} contested must_not_contain "
                          f"item(s) -> sheet kept", flush=True)
            else:
                e["excise"] = {"resid_sha": _resid_sha(revised_fs, resid), "decision": "drop",
                               "classifications": cls, "route": ROUTE, "model": ROUTE_MODEL}
                why = [c["verdict"] for c in cls if c["verdict"] != "contested_mnc"]
                print(f"  {rec['id']}: drop stands "
                      f"({', '.join(sorted(set(why))) or 'no contested items named'})", flush=True)
            save()

    if to_excise:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(excise_job, to_excise))
    if BUDGET_HIT[0]:
        print(f"\nBUDGET HIT after {time.time() - T0:.0f}s in stage 3b - output stages SKIPPED; "
              f"rerun to continue.", flush=True)
        return

    # ---- stage 4: statuses, gate 2, outputs ----
    def keep(rec, e, fact_sheet=None):
        """Kept record + its critique provenance (which transport(s) audited this sheet)."""
        out = dict(rec)
        if fact_sheet is not None:
            out["fact_sheet"] = fact_sheet
        out["critique_provenance"] = {
            "panel": list(CRITICS), "effort": EFFORT, "revise_effort": "high",
            "transports": e.get("transports", [{"route": ROUTE, "model": ROUTE_MODEL}]),
            "status": e.get("status"), "revised": bool(e.get("revise", {}).get("ok")),
            **({"excised_mnc": [x["assertion"] for x in e["excise"]["excised"]]}
               if e.get("status") == "revised-kept-excised" else {}),
            "spec": "w-d-master-dataset.md 3.2 + Amendment 2026-07-29b H + 2026-07-30 M"
                    + " + 2026-08-09b T (excise rule"
                    + (", transcript-only views" if TRANSCRIPT_ONLY else "") + ")"}
        return out

    kept, dropped = [], []
    pre_excise_dropped = []   # what gate 2 would have dropped WITHOUT the excise rule
    for rec in recs:
        rid = rec["id"]
        e = state.get(rid, {})
        if rid in errored or len(e.get("critics", {})) < len(CRITICS):
            e["status"] = "critic-error"
            continue
        if not e.get("material"):
            has_minor = any(r.get("issues") for r in e["critics"].values())
            e["status"] = "minor-only" if has_minor else "clean"
            kept.append(keep(rec, e))
            continue
        if not REVISE:
            e["status"] = "material-unrevised"
            continue
        rev = e.get("revise", {})
        if not rev.get("ok"):
            # only a genuinely exhausted revision (--revise-tries transport attempts) drops
            e["status"] = ("dropped-revise-failed" if rev.get("tries", 0) >= REVISE_TRIES
                           else "revise-pending")
            if e["status"] == "dropped-revise-failed":
                dropped.append(rid)
                pre_excise_dropped.append(rid)
            continue
        recheck = e.get("recheck", {"verdict": "error", "issues": []})
        residual = [i for i in recheck.get("issues", []) if i.get("severity") == "material"]
        if recheck.get("verdict") == "error":
            e["status"] = "critic-error"
            continue
        if residual:
            pre_excise_dropped.append(rid)
            # T2 excise rule: residuals that are ALL contested must_not_contain items delete
            # those items and keep the sheet; anything else (or a pending/failed classifier)
            # keeps/awaits the drop.
            ex = e.get("excise", {})
            if ex.get("resid_sha") != _resid_sha(rev["fact_sheet"], _residuals(e)):
                e["status"] = "excise-pending"   # classifier not yet run for THIS residual set
                pre_excise_dropped.pop()
                continue
            if ex.get("decision") == "excise":
                e["status"] = "revised-kept-excised"
                kept.append(keep(rec, e, fact_sheet=ex["fact_sheet"]))
            else:
                e["status"] = "dropped-still-material"
                dropped.append(rid)
        else:
            e["status"] = "revised-kept"
            kept.append(keep(rec, e, fact_sheet=rev["fact_sheet"]))
    save()

    # excision log: rebuild this source's entries from state, keep other sources' entries
    excision_entries = []
    for rec in recs:
        e = state.get(rec["id"], {})
        if e.get("status") == "revised-kept-excised":
            for x in e["excise"]["excised"]:
                excision_entries.append({
                    "source": SOURCE, "id": rec["id"], "mnc_index": x["mnc_index"],
                    "assertion": x["assertion"], "why_wrong": x["why_wrong"],
                    "critic_reason": x["critic_reason"],
                    "classifier": {"route": e["excise"].get("route"),
                                   "model": e["excise"].get("model")}})
    log_obj = {"spec": "w-d-master-dataset.md Amendment 2026-08-09b part T2 (excise rule)",
               "note": "must_not_contain items deleted after the one revision cycle because the "
                       "re-audit held the transcript partly/actually supports them (a faithful "
                       "note would be wrongly penalised); the sheets are KEPT with status "
                       "revised-kept-excised. Deleting a must_not_contain item only makes the "
                       "sheet less strict, never wrong.",
               "entries": []}
    if os.path.exists(EXCISED_LOG):
        try:
            old = json.load(open(EXCISED_LOG))
            log_obj["entries"] = [x for x in old.get("entries", []) if x.get("source") != SOURCE]
        except Exception:
            pass
    log_obj["entries"] += excision_entries
    log_obj["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = EXCISED_LOG + ".tmp"
    json.dump(log_obj, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, EXCISED_LOG)

    tmp = OUTFILE + ".tmp"
    json.dump(kept, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, OUTFILE)

    # report
    from collections import Counter
    statuses = Counter(state[r["id"]]["status"] for r in recs if r["id"] in state)
    flags_by_critic = Counter()
    material_by_critic = Counter()
    for e in state.values():
        for c, r in e.get("critics", {}).items():
            for i in r.get("issues", []):
                flags_by_critic[c] += 1
                if i.get("severity") == "material":
                    material_by_critic[c] += 1
    # incomplete work (a critic, revise or excise transport that still needs a rerun) is NOT
    # scored: gate 2's drop-rate must be computed over consults whose cycle actually completed
    incomplete = (statuses.get("critic-error", 0) + statuses.get("revise-pending", 0)
                  + statuses.get("excise-pending", 0))
    n_scored = len(recs) - incomplete
    drop_rate = (len(dropped) / n_scored) if n_scored else 0.0
    pre_excise_rate = (len(pre_excise_dropped) / n_scored) if n_scored else 0.0
    pass_rate = (len(kept) / n_scored) if n_scored else 0.0
    n_excised_sheets = statuses.get("revised-kept-excised", 0)
    n_excised_items = len(excision_entries)
    gate2 = "PASS" if drop_rate < 0.10 else "FAIL - systematic extraction-prompt failure, STOP: human review required (spec 6 gate 2)"

    transports = sorted({json.dumps(t, sort_keys=True)
                         for e in state.values() for t in e.get("transports", [])})
    lines = [f"# Critique report - extracted fact sheets, source `{SOURCE}`", "",
             f"Panel: support / materiality / leakage on `{ROUTE_MODEL}` (effort {EFFORT}); one "
             f"revision cycle (effort high) with the immutable-transcript rule; re-audit = support "
             f"critic.",
             f"Transport(s) actually used: "
             + "; ".join(f"{json.loads(t)['route']} -> `{json.loads(t)['model']}`" for t in transports),
             f"(The extraction layer this audits ran wholly on the plan path / `claude -p`; the "
             f"critique layer is a separate transport pool - it audits the artifact rather than "
             f"producing it. Per-record provenance in `critique_provenance`.)", "",
             "## Summary", "",
             f"- consults: {len(recs)} | scored: {n_scored} | kept: {len(kept)} "
             f"(clean {statuses.get('clean', 0)}, minor-only {statuses.get('minor-only', 0)}, "
             f"revised-kept {statuses.get('revised-kept', 0)}, "
             f"revised-kept-excised {n_excised_sheets})",
             f"- dropped: {len(dropped)} ({', '.join(dropped) if dropped else 'none'})",
             f"- excise rule (Amendment 2026-08-09b T2): {n_excised_items} contested "
             f"must_not_contain item(s) excised across {n_excised_sheets} sheet(s) -> "
             f"master/excised_mnc_items.json; drop rate before excise would have been "
             f"{pre_excise_rate:.1%}",
             f"- incomplete, excluded from scoring (rerun to retry): "
             f"{statuses.get('critic-error', 0)} critic-error, "
             f"{statuses.get('revise-pending', 0)} revise-pending, "
             f"{statuses.get('excise-pending', 0)} excise-pending",
             f"- flags by critic (total/material): "
             + " | ".join(f"{c} {flags_by_critic.get(c, 0)}/{material_by_critic.get(c, 0)}"
                          for c in CRITICS),
             f"- pass rate after one revise cycle: {pass_rate:.1%}",
             f"- drop rate: {drop_rate:.1%} -> **gate 2 (<10%): {gate2}**", ""]
    for rec in recs:
        rid = rec["id"]
        e = state.get(rid, {})
        lines.append(f"## {rid} - {e.get('status', 'MISSING')}")
        flagged = False
        for c in CRITICS:
            for i in e.get("critics", {}).get(c, {}).get("issues", []):
                lines.append(f"  - [{c}/{i.get('severity', '?')}] {i.get('what', '')}")
                flagged = True
        if not flagged:
            lines.append("  (no issues)")
        rev = e.get("revise")
        if rev is not None:
            lines.append(f"  revision: {'ok' if rev.get('ok') else 'FAILED'}"
                         + (" (transcript drift restored)" if rev.get("transcript_drift") else ""))
        if e.get("recheck"):
            rc = e["recheck"]
            resid = [i for i in rc.get("issues", []) if i.get("severity") == "material"]
            lines.append(f"  re-audit after revision: {rc.get('verdict')} "
                         f"({len(resid)} residual material)")
            for i in resid:
                lines.append(f"    - [residual] {i.get('what', '')}")
        if e.get("status") == "revised-kept-excised":
            for x in e["excise"]["excised"]:
                lines.append(f"  excised must_not_contain #{x['mnc_index']}: {x['assertion']}")
        elif e.get("excise", {}).get("decision") == "drop" and e.get("status") == "dropped-still-material":
            verdicts = sorted({c['verdict'] for c in e['excise'].get('classifications', [])})
            lines.append(f"  excise rule: not applicable ({', '.join(verdicts)})")
        lines.append("")
    open(REPORT, "w").write("\n".join(lines))

    print(f"\n[{SOURCE}] kept {len(kept)}/{n_scored} (pass {pass_rate:.1%}) | dropped {len(dropped)} "
          f"(drop {drop_rate:.1%}; pre-excise {pre_excise_rate:.1%}) | "
          f"excised {n_excised_items} item(s)/{n_excised_sheets} sheet(s) | gate 2: {gate2}")
    print(f"-> {os.path.relpath(OUTFILE, HERE)} + {os.path.relpath(REPORT, HERE)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        log_spend()   # every exit path (incl. a --max-seconds stage skip) ledgers its credits
