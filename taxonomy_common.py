"""Shared substrate for the census: routing, parsing, and the reporting frame's own checks.

Holds what the four taxonomy runners all need and none of them should own alone:
the three-vendor note corpus unified into one iterator, the published reporting
frame, the route switch (plan vs OpenRouter) with per-role model selection,
deterministic per-call seeds, the resumable record store, and the budget tripwires.

**The published frame.** The reporting taxonomy is no
longer purely emergent. `taxonomy_frame.json` holds a two-tier frame: Tier 1 is a
published spine (Biro et al. 2025's four error classes) that every finding is
assigned to exactly once so our shares sit beside published numbers, and Tier 2 is
the mechanism level - one category per targeted discovery pass, each declaring its
Tier-1 parent and its crosswalk into every published taxonomy that names it.
Clustering mints Tier-3 subcategories under Tier 2 and flags what it cannot place.
`load_frame()` is the single reader; nothing else may hardcode a category name.

**Route switch (a disclosed deviation from the pre-registered wording).** The
specification puts this stage on the subscription transport. It runs on the API instead: a subscription's session limits cap throughput and metered credits
do not, so heavy batches route via OpenRouter. `--route openrouter`
(default) sends every call through the pinned `llm()` client on the `constructor` role
(`anthropic/claude-opus-5`, the same model the plan path would use); `--route plan` sends the identical prompt through
`common.claude()` / `claude -p`. The INSTRUMENT is unchanged either way - prompts,
pass count, panel size and the majority rule are imported verbatim from
`discover.py` / `verify_findings.py`. Only the transport moves, and every record
carries the route that bought it.

The unified corpus (`note_units`):
  scribe_A  - master/notes_corpus_master.json, 2 templates per consultation (short,
           detailed), transcript in the row itself.
  scribe_B  - scribe_B_notes/<source>__<id>.txt, one note per consultation, template
           "audio", transcript joined from the master corpus by (source, id).
  scribe_C  - scribe_C_notes/<source>__<id>.txt, same shape.
Every unit gets a stable `note_key` = "<scribe>|<source>|<id>|<template>", which is
the idempotency key for discovery and the join key for the panel.
"""
import hashlib, json, os, random, sys, time

from common import HERE, RESULTS, claude, llm
from w2_common import RecordStore, SpendGuard, chunked, confirm, parse_json_blob, run_block  # noqa: F401

SPEC = "the pre-registered specification"
EXPERIMENT = "wd-taxonomy"
MASTER = os.path.join(HERE, "master")

CORPUS_FILE = "master/notes_corpus_master.json"
TRAPBLIND_FILE = "master/trapblind_scenarios_critiqued.json"   # transcripts scribe_A never generated for
NOTE_DIRS = {"scribe_B": "scribe_B_notes", "scribe_C": "scribe_C_notes"}
VENDORS = ("scribe_A", "scribe_B", "scribe_C")
SUBSTRATES = ("authored", "trapblind", "primock", "aci")

# Seeded stability probe: 15 consultations, seed 20260728, 3 runs.
STABILITY_SEED = 20260728
STABILITY_N = 15

# Transport settings. reasoning_effort "medium" mirrors the June instrument's
# `--effort medium`; max_tokens is generous because a discovery pass may legitimately
# return a long issue list and reasoning tokens count against the ceiling.
CONSTRUCTOR_ROLE = "constructor"           # models.lock.json -> anthropic/claude-opus-5
AUDITOR_ROLE = "auditor"                   # models.lock.json -> openai/gpt-5.5
TIEBREAK_ROLE = "judge-primary"            # models.lock.json -> openai/gpt-5.4
PLAN_MODEL = "claude-opus-5"
REASONING_EFFORT = "medium"
# Panel max_tokens raised from 4000 on 2026-08-12: measured, gpt-5.5 at
# reasoning_effort high returns finish_reason='length' on a short JSON verdict
# because reasoning tokens eat the ceiling, and llm() (correctly) raises rather
# than let an empty string look like a verdict.
MAX_TOKENS = {"discover": 16000, "complete": 16000, "panel": 6000, "label": 2000}

# The cross-family refute panel. Two skeptics from
# DIFFERENT model families, so a shared blind spot has to cross a family boundary to
# survive; disagreement goes to one tiebreak call on a third model, whose verdict is
# final. Ordered constructor-first so the Anthropic member always votes first.
PANEL_ROLES = (CONSTRUCTOR_ROLE, AUDITOR_ROLE)


# ---------------------------------------------------------------- the published frame
FRAME_FILE = "taxonomy_frame.json"
_FRAME = None


def load_frame():
    """The published reporting frame, read once. See the module docstring."""
    global _FRAME
    if _FRAME is None:
        _FRAME = json.load(open(os.path.join(HERE, FRAME_FILE)))
    return _FRAME


def frame_sha256():
    """Hash of the frame as it stood for this run, written into every manifest."""
    return hashlib.sha256(open(os.path.join(HERE, FRAME_FILE), "rb").read()).hexdigest()


def frame_passes():
    """The targeted passes the frame declares, in frame order, plus 'open' last.

    This is the ONLY definition of which passes a discovery run makes. Adding a pass
    means adding a Tier-2 entry with `hunted: true` to taxonomy_frame.json.
    """
    out = [c["pass"] for c in load_frame()["tier2"]
           if c.get("hunted") and c.get("pass") and c["pass"] != "open"]
    return out + ["open"]


def tier2_for_pass(pass_name):
    for c in load_frame()["tier2"]:
        if c.get("pass") == pass_name:
            return c["key"]
    return None


def tier1_of_tier2(tier2_key):
    for c in load_frame()["tier2"]:
        if c["key"] == tier2_key:
            return c.get("tier1")
    return None


# Emergent labels the open pass invents have to be placed in the frame somehow. The
# family() matcher from the June pilot is the study's existing free-text mode
# classifier, so it is reused rather than a second one written; this table only maps
# its output onto frame keys. Anything that lands nowhere stays "unmapped" and is
# reported as such - never silently absorbed.
# Keys are exactly the family names in pilot/scripts/cross_scribe_matches.py FAMILIES -
# `assert_family_keys_known()` fails loudly if that list ever changes underneath us,
# because a silently unrecognised family would show up as a large "unmapped" row rather
# than as an error.
_FAMILY_TO_TIER2 = {
    "omission": "omission", "diagnosis_omitted": "omission", "safety_net": "omission",
    "exam_not_done": "exam_provenance",
    "fabrication": "fabrication", "demographics": "fabrication",
    "hardening": "modality_hardening",
    "onset_timing": "temporal", "negation": "negation",
    "laterality_site": "laterality", "attribution": "attribution",
}


def assert_family_keys_known(families):
    """Every family the June matcher can emit must have a frame home."""
    names = {f[0] if isinstance(f, (list, tuple)) else f for f in families}
    missing = sorted(names - set(_FAMILY_TO_TIER2))
    if missing:
        raise SystemExit(
            f"the June family matcher emits families with no frame home: {missing}. Add them to "
            "_FAMILY_TO_TIER2 (or decide they are genuinely unmappable) - leaving them out would "
            "quietly inflate the 'unmapped' row instead of failing.")


def frame_place(finding, family_fn=None):
    """(tier2, tier1, how) for one finding.

    Targeted passes are frame-derived by construction, so their placement is exact. The
    OPEN pass is not a frame category - it is the escape hatch - so its findings are
    placed by the June family matcher on the emergent mode label the model chose. What
    the matcher cannot place comes back as ("unmapped", None, "unmapped") and is
    reported in its own row rather than absorbed.
    """
    p = finding.get("pass")
    t2 = tier2_for_pass(p) if p and p != "open" else None
    if t2 and t2 != "open":
        return t2, tier1_of_tier2(t2), "pass"
    if family_fn is not None:
        fam = family_fn(finding)
        t2 = _FAMILY_TO_TIER2.get(fam)
        if t2:
            return t2, tier1_of_tier2(t2), f"family:{fam}"
    return "unmapped", None, "unmapped"


# ---------------------------------------------------------------- the corpus
def _transcript_index():
    """(source, id) -> (transcript, provenance). The master corpus is the source of
    truth; trap-blind consultations scribe_A never generated a note for fall back to the
    critiqued scenario file so a scribe_C/scribe_B-only consultation is still auditable."""
    idx, prov = {}, {}
    for r in json.load(open(os.path.join(HERE, CORPUS_FILE))):
        k = (r["source"], r["id"])
        if r.get("transcript") and k not in idx:
            idx[k], prov[k] = r["transcript"], CORPUS_FILE
    tb = os.path.join(HERE, TRAPBLIND_FILE)
    if os.path.exists(tb):
        for r in json.load(open(tb)):
            k = ("trapblind", r["id"])
            if r.get("transcript") and k not in idx:
                idx[k], prov[k] = r["transcript"], TRAPBLIND_FILE
    return idx, prov


def _vendor_files(scribe):
    """(source, id, path) per note file, skipping the audio-variant probe captures.

    Filenames are '<source>__<id>.txt'. scribe_C's speed/silence A/B captures append a
    second '__<variant>' (e.g. 'primock__day1_consultation06__vsilence') - those are
    a capture-methodology probe (master/speed_ab_results.json), not corpus notes, so
    they are excluded here and reported by `corpus_report`.
    """
    d = os.path.join(HERE, NOTE_DIRS[scribe])
    out, variants = [], []
    if not os.path.isdir(d):
        return out, variants
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".txt"):
            continue
        parts = fn[:-4].split("__")
        if len(parts) == 2:
            out.append((parts[0], parts[1], os.path.join(d, fn)))
        else:
            variants.append(fn)
    return out, variants


def note_units(vendors=VENDORS, sources=None, consults=None, limit=None):
    """Every scribe note in scope, as one flat list of audit units.

    vendors:  subset of ('scribe_A','scribe_B','scribe_C')
    sources:  subset of the four substrates (None = all)
    consults: iterable of (source, id) to restrict to (the stability subsample)
    limit:    keep the first N CONSULTATIONS (all their notes, all vendors)
    """
    tidx, tprov = _transcript_index()
    want_src = set(sources) if sources else None
    want_con = set(consults) if consults else None
    units, skipped = [], []

    if "scribe_A" in vendors:
        for r in json.load(open(os.path.join(HERE, CORPUS_FILE))):
            if not r.get("note") or r.get("scribe", "scribe_A") != "scribe_A":
                continue
            units.append({"scribe": "scribe_A", "source": r["source"], "id": r["id"],
                          "template": r["template"], "note": r["note"],
                          "transcript": r["transcript"], "transcript_source": CORPUS_FILE})
    for scribe in ("scribe_B", "scribe_C"):
        if scribe not in vendors:
            continue
        files, _ = _vendor_files(scribe)
        for source, cid, path in files:
            t = tidx.get((source, cid))
            if not t:
                skipped.append({"scribe": scribe, "source": source, "id": cid,
                                "why": "no transcript in the master corpus"})
                continue
            note = open(path).read().strip()
            if not note:
                skipped.append({"scribe": scribe, "source": source, "id": cid, "why": "empty note file"})
                continue
            units.append({"scribe": scribe, "source": source, "id": cid, "template": "audio",
                          "note": note, "transcript": t,
                          "transcript_source": tprov[(source, cid)]})

    for u in units:
        u["consultation"] = f"{u['source']}/{u['id']}"
        u["note_key"] = f"{u['scribe']}|{u['source']}|{u['id']}|{u['template']}"
    units.sort(key=lambda u: (u["source"], u["id"], u["scribe"], u["template"]))

    if want_src:
        units = [u for u in units if u["source"] in want_src]
    if want_con:
        units = [u for u in units if (u["source"], u["id"]) in want_con]
    if limit:
        keep, seen = [], []
        for u in units:
            c = (u["source"], u["id"])
            if c not in seen:
                if len(seen) >= limit:
                    continue
                seen.append(c)
            keep.append(u)
        units = keep
    return units, skipped


def corpus_report(units, skipped):
    """One-screen census of what is in scope (printed before any call is bought)."""
    from collections import Counter
    by_v = Counter(u["scribe"] for u in units)
    by_s = Counter(u["source"] for u in units)
    cons = {(u["source"], u["id"]) for u in units}
    lines = [f"corpus: {len(units)} notes over {len(cons)} consultations",
             f"  by vendor:    {dict(by_v)}",
             f"  by substrate: {dict(by_s)}"]
    for scribe in ("scribe_B", "scribe_C"):
        _, variants = _vendor_files(scribe)
        if variants:
            lines.append(f"  {scribe}: {len(variants)} audio-variant probe captures excluded "
                         f"({', '.join(v[:-4] for v in variants[:3])}{'...' if len(variants) > 3 else ''})")
    for s in skipped:
        lines.append(f"  ! skipped {s['scribe']}/{s['source']}/{s['id']}: {s['why']}")
    return "\n".join(lines)


def corpus_census(units, skipped):
    """The paper's denominator, itemised: what is in, what is out, and why.

    565 auditable notes come from 572 files on disk. The seven-file gap is not a
    rounding difference and must not live only in a console line - it is written into
    the analysis output so the denominator can be audited from the artefact alone.
    """
    from collections import Counter
    variants = []
    for scribe in NOTE_DIRS:
        _, v = _vendor_files(scribe)
        variants += [{"scribe": scribe, "file": x,
                      "why": "audio-variant capture probe (speed/silence A/B), not a corpus note"}
                     for x in v]
    return {
        "n_notes_audited": len(units),
        "n_consultations": len({(u["source"], u["id"]) for u in units}),
        "by_vendor": dict(Counter(u["scribe"] for u in units)),
        "by_substrate": dict(Counter(u["source"] for u in units)),
        "by_vendor_template": dict(Counter(f"{u['scribe']}/{u['template']}" for u in units)),
        "excluded": {
            "n_total": len(variants) + len(skipped),
            "audio_variant_probes": variants,
            "no_transcript_in_master_corpus": [
                {"scribe": s["scribe"], "file": f"{s['source']}__{s['id']}.txt", "why": s["why"]}
                for s in skipped],
        },
        "note": "note files on disk = audited + excluded. scribe_A is API-generated at two "
                "templates per consultation, so it contributes two notes per consultation "
                "where scribe_B and scribe_C contribute one.",
        "known_gap": "trapblind/tb_af_anticoag_review has no scribe_A note (never generated), so "
                     "that consultation can never produce a cross-vendor match involving scribe_A; "
                     "its transcript is read from " + TRAPBLIND_FILE,
    }


def stability_consults(units, n=STABILITY_N, seed=STABILITY_SEED):
    """The seeded consultation subsample for the 3-run stability probe.

    Drawn from consultations that ALL vendors present in `units` cover, so the sd is
    measured on the same notes for every scribe rather than on a vendor-skewed draw.
    """
    from collections import defaultdict
    have = defaultdict(set)
    for u in units:
        have[(u["source"], u["id"])].add(u["scribe"])
    vendors = {u["scribe"] for u in units}
    full = sorted(c for c, vs in have.items() if vs == vendors)
    pool = full or sorted(have)
    return set(random.Random(seed).sample(pool, min(n, len(pool))))


# ---------------------------------------------------------------- the route switch
def call_seed(key):
    """Deterministic per-call seed, so a resumed or repeated run asks the same way."""
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") % (2 ** 31 - 1)


_EMPTY_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0}


def route_call(prompt, key, route, kind="panel", timeout=600, role=CONSTRUCTOR_ROLE,
               effort=None):
    """One JSON-returning instrument call on the chosen transport.

    `role` is a models.lock.json role - constructor (claude-opus-5) for everything
    June ran, auditor (gpt-5.5) for the cross-family panel member, judge-primary
    (gpt-5.4) for the tiebreak. `--route plan` can only serve the constructor role
    (the plan path is `claude -p`), and callers must refuse the combination rather
    than silently substituting a model.

    Returns (parsed_or_None, meta). NEVER raises: a transport failure comes back as
    (None, meta with 'error'), and each caller applies its own pre-registered
    fallback (discovery: no issues from that pass; refute: REFUTED).
    """
    mt = MAX_TOKENS.get(kind, MAX_TOKENS["panel"])
    eff = effort or REASONING_EFFORT
    t0 = time.time()
    if route == "plan" and role != CONSTRUCTOR_ROLE:
        raise SystemExit(
            f"--route plan cannot serve role {role!r}: the plan path is `claude -p` and only "
            "reaches the constructor model. The cross-family panel needs --route openrouter.")
    if route == "plan":
        txt = claude(prompt, timeout=timeout, model=PLAN_MODEL, effort=eff, retries=1)
        meta = {"route": "plan", "model": PLAN_MODEL, "role": role, "calls": 1, "cost_usd": 0.0,
                "usage": dict(_EMPTY_USAGE), "wall_s": round(time.time() - t0, 2)}
        if not txt:
            meta["error"] = "empty reply from claude -p"
            return None, meta
        obj = parse_json_blob(txt)
        if obj is None:
            meta["error"] = "unparseable reply"
        return obj, meta
    try:
        texts, m = llm(prompt, role, temperature=1.0, k=1, seed=call_seed(key),
                       reasoning_effort=eff, max_tokens=mt, timeout=timeout)
    except Exception as e:                       # llm() raises after its own retries
        return None, {"route": "openrouter", "model": None, "role": role, "calls": 0,
                      "cost_usd": 0.0, "usage": dict(_EMPTY_USAGE),
                      "wall_s": round(time.time() - t0, 2),
                      "error": f"{type(e).__name__}: {str(e)[:200]}"}
    meta = {"route": "openrouter", "model": m["model"], "role": role,
            "calls": len(m.get("requests", [])) or 1,
            "cost_usd": m.get("cost_usd_reported") or 0.0, "usage": m.get("usage", dict(_EMPTY_USAGE)),
            "seed": call_seed(key), "gen_ids": m.get("generation_ids"),
            "providers": m.get("providers"), "wall_s": round(time.time() - t0, 2)}
    obj = parse_json_blob(texts[0] if texts else "")
    if obj is None:
        meta["error"] = "unparseable reply"
    return obj, meta


def spend_label(route):
    return "claude_plan" if route == "plan" else "openrouter_credits"


# ---------------------------------------------------------------- budget
STAGE_LEDGER = os.path.join(MASTER, "taxonomy_stage_ledger.json")
# A ceiling for the whole stage, so a resumed run cannot quietly re-buy work. It is a
# default rather than a licence: every invocation reports what it has spent against
# it. Size it to your own budget before a full run; the shipped default is conservative.
STAGE_CAP_DEFAULT = 250.0


def stage_spent():
    """What THIS stage has spent, summed from the cost ledger's own experiment field.

    Sums only this stage's own lines: summing the whole ledger would count sibling
    experiments' spend against this stage's cap. Filtering on `experiment` keeps the
    number auditable - `wd-taxonomy` in results/cost_ledger.jsonl is exactly what this
    stage cost.
    """
    path = os.path.join(RESULTS, "cost_ledger.jsonl")
    if not os.path.exists(path):
        return 0.0
    tot = 0.0
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("experiment") == EXPERIMENT and \
                    d.get("spend", "openrouter_credits") == "openrouter_credits":
                tot += d.get("cost_usd") or 0.0
    return tot


def stage_baseline():
    """Kept for the manifest record: what the whole project had spent when this stage
    opened. No longer used for the cap - see stage_spent()."""
    from w2_common import ledger_total
    if os.path.exists(STAGE_LEDGER):
        return json.load(open(STAGE_LEDGER))["baseline_usd"]
    os.makedirs(MASTER, exist_ok=True)
    b = ledger_total()
    json.dump({"baseline_usd": b, "opened_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "what": "project-wide ledger total when the taxonomy stage opened (record only; "
                       "the cap is measured by stage_spent(), which filters on experiment)"},
              open(STAGE_LEDGER, "w"), indent=1)
    return b


class BudgetGuard:
    """Three tripwires, the tighter ones first.

    `budget`     - what THIS invocation may spend.
    `stage_cap`  - what the WHOLE taxonomy stage may spend, across all four scripts and
                   every resume. Measured off the project
                   cost ledger against the baseline recorded when the stage opened, so
                   it survives process restarts and cannot be reset by re-running.
    SpendGuard   - the project-wide cumulative backstop.

    None can be crossed without --authorize-spend. A stop is clean: everything bought
    is already in the record store, so a resume re-buys nothing.
    """

    def __init__(self, budget=100.0, warn=200.0, stop=400.0, authorized=False,
                 route="openrouter", stage_cap=STAGE_CAP_DEFAULT):
        self.budget, self.authorized, self.route = budget, authorized, route
        self.stage_cap = stage_cap
        self.spent_here = 0.0
        self._guard = SpendGuard(warn=warn, stop=stop, authorized=authorized) if route != "plan" else None
        self.baseline = self._guard.baseline if self._guard else 0.0
        self.stage_base = stage_baseline() if route != "plan" else 0.0
        # what THIS experiment has spent, not what the whole project has
        self.stage_before = stage_spent() if route != "plan" else 0.0
        self._warned = False
        self._stage_warned = False
        if self.stage_cap and route != "plan":
            print(f"  stage spend so far: ${self.stage_before:.2f} of the "
                  f"${self.stage_cap:.0f} taxonomy-stage cap")
            self._check_stage()

    def _check_stage(self):
        used = self.stage_before + self.spent_here
        if self.stage_cap and used >= self.stage_cap * 0.8 and not self._stage_warned:
            self._stage_warned = True
            print(f"  ! STAGE WARNING: ${used:.2f} of the ${self.stage_cap:.0f} taxonomy-stage cap")
        if self.stage_cap and used >= self.stage_cap and not self.authorized:
            raise SystemExit(
                f"\nSTAGE CAP: the taxonomy stage has now spent ${used:.2f} >= the "
                f"${self.stage_cap:.0f} stage cap.\nStopping cleanly. Everything "
                "bought is in the record store, so a resume re-buys nothing - but the remaining "
                "work needs --authorize-spend, or a higher --stage-cap-usd.")

    def add(self, usd):
        usd = usd or 0.0
        self.spent_here += usd
        if self._guard:
            self._guard.add(usd)
        if self.spent_here >= self.budget * 0.8 and not self._warned:
            self._warned = True
            print(f"  ! BUDGET WARNING: ${self.spent_here:.2f} of the ${self.budget:.0f} "
                  f"run budget spent")
        self._check_stage()
        if self.spent_here >= self.budget and not self.authorized:
            raise SystemExit(
                f"\nBUDGET STOP: this run has spent ${self.spent_here:.2f} >= the ${self.budget:.0f} "
                "cap.\nRe-run with a higher --budget-usd (and --authorize-spend) after checking "
                "the cap. Completed work is in the record store - nothing is re-bought on resume.")

    @property
    def total(self):
        return self.baseline + self.spent_here

    @property
    def stage_total(self):
        return self.stage_before + self.spent_here


# ---------------------------------------------------------------- misc
def usage_add(acc, meta):
    """Roll one call's receipts into a running total dict."""
    acc["calls"] += meta.get("calls", 0)
    acc["cost_usd"] += meta.get("cost_usd") or 0.0
    for f, v in (meta.get("usage") or {}).items():
        acc[f] = acc.get(f, 0) + (v or 0)
    if meta.get("error"):
        acc["errors"] += 1
    return acc


def new_usage():
    return {"calls": 0, "cost_usd": 0.0, "errors": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0}


def manifest_params(base, route, units, extra=None):
    p = dict(base)
    p.update({"routing": "openrouter - a disclosed deviation; the specification says plan-path"
                         if route == "openrouter" else "plan-path, as the specification says",
              "route": route,
              "model": PLAN_MODEL if route == "plan" else "anthropic/claude-opus-5 (role=constructor)",
              "reasoning_effort": REASONING_EFFORT,
              "n_notes_in_scope": len(units),
              "n_consultations_in_scope": len({(u["source"], u["id"]) for u in units}),
              "frame_file": FRAME_FILE,
              "frame_version": load_frame()["frame_version"],
              "frame_sha256": frame_sha256(),
              "instrument": "discover.py + verify_findings.py prompts, verbatim except the two "
                            "documented 2026-08-12 changes (frame-derived pass set; transcript-first "
                            "block order on the panel prompt)"})
    if extra:
        p.update(extra)
    return p


def run_inputs(extra=()):
    """Everything a taxonomy Run() should hash into its manifest.

    The four scripts plus the frame plus the corpus. Other stages write into this tree
    while a run is in flight, so a clean-tree guarantee is not always available at launch; hashing the exact code and data THIS stage executes and
    reads means provenance does not depend on one, and a reviewer can prove that the
    disclosed dirty paths were not part of this instrument.
    """
    base = ["taxonomy_common.py", "census/taxonomy_discover.py",
            "census/taxonomy_verify.py", "census/taxonomy_cluster.py",
            "census/taxonomy_analyze.py", FRAME_FILE, CORPUS_FILE]
    seen, out = set(), []
    for f in list(base) + list(extra):
        if f not in seen and os.path.exists(os.path.join(HERE, f)):
            seen.add(f)
            out.append(f)
    return out


def cache_line(usage):
    """One-line prompt-cache report. Written next to every stage's cost line, because
    whether caching engaged is the single biggest lever on this stage's bill and it is
    provider-dependent (measured 2026-08-12: OpenAI routes cache implicitly, Anthropic
    routes return 0 cached tokens without cache_control breakpoints, which live in
    `common.llm()`).
    """
    pt, ct = usage.get("prompt_tokens", 0), usage.get("cached_tokens", 0)
    return (f"prompt cache: {ct:,}/{pt:,} input tokens cached ({ct / pt:.1%})"
            if pt else "prompt cache: no input tokens recorded")


def import_verbatim(module, names):
    """Import instrument constants from a June script without its argv parsing firing.

    discover.py / verify_findings.py read sys.argv at import time (they are scripts,
    not libraries). Blanking argv around the import means our own CLI flags can never
    silently reconfigure the instrument we are supposed to be reusing verbatim.
    """
    import importlib
    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        m = importlib.import_module(module)
    finally:
        sys.argv = argv
    return [getattr(m, n) for n in names]
