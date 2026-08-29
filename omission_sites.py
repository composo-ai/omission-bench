"""omission_sites.py - the fact-site map: where does each fact actually LIVE in the note?

Deliverable 2 of the graded-omission design correction (specs/w-d-master-dataset.md
Amendment 2026-08-10). It exists because of what the W-A re-verification found: real
clinical notes are REDUNDANT. The same clinical fact is stated in the history, again
in the examination or impression, and again in the plan, in different words. Deleting
one span therefore does not remove the fact - it leaves a paraphrase behind. 34 pairs
the corpus called "omissions" are partial omissions, and nobody could tell until the
corrected truly_absent check said so (docs/FREEZE-DECISION.md, APPLIED section).

You cannot build a complete omission, or a deliberate partial one, without first
knowing every site. That is this file's whole job.

  per consultation
    -> the omission-eligible facts (pair-eligible omission-mode traps + core
       must_contain, ranked by the SAME preference the original injector used)
    -> for each fact, EVERY mention site in the clean note:
         span (verbatim), section, strength (explicit/paraphrase/partial),
         char offset, whether the span is unit-aligned, and which site is PRIMARY
    -> a second, cross-family call that checks the map: does each listed span really
       state the fact, was any site MISSED, is the primary nomination right
    -> master/fact_sites.json

Two models, two families, as everywhere else in W-A: the MAPPER is the models.lock
`constructor` role (anthropic/claude-opus-5 over OpenRouter); the VERIFIER is the
`auditor` role (openai/gpt-5.5, reasoning_effort high) at the W-A edit seeds. The
verifier's corrections are APPLIED, not just recorded - dropped sites, added sites and
a re-pointed primary all land in the artefact with the reason. No human review anywhere.

Substrate: the FROZEN clean notes. `master/pairs_master_frozen.json -> ideal_notes`
for the 112 evaluation consultations, `master/gepa_dev_pool.json` for the 22 carved
into W-F's dev pool. The frozen 281-pair artefact is READ ONLY and never written here.

Usage:
  .venv/bin/python omission_sites.py --smoke --allow-dirty -y      # 6 consults, ~1 min
  .venv/bin/python omission_sites.py --consults aci|D2N077,...     # named consults
  .venv/bin/python omission_sites.py --full --workers 6 -y         # the whole eval split
  .venv/bin/python omission_sites.py --report                      # free: print the map's stats
Exit: 0 complete | 2 work left (rerun) | 3 budget stop.
"""
import argparse, json, os, random, re, sys, threading, unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from common import HERE, Run, llm, resolve_model
from w2_common import parse_json_blob
import w_a_master as WA

SPEC = "specs/w-d-master-dataset.md Amendment 2026-08-10 (graded-omission design)"
EXPERIMENT = "wd-omission-sites"
MASTER = os.path.join(HERE, "master")
SITES_FILE = os.path.join(MASTER, "fact_sites.json")
STATE_FILE = os.path.join(MASTER, "fact_sites_state.json")

MAPPER_ROLE = "constructor"        # anthropic/claude-opus-5
MAPPER_EFFORT = "medium"           # the W-A construction row
AUDITOR_ROLE = "auditor"           # openai/gpt-5.5
AUDITOR_EFFORT = "high"            # models.lock auditor notes
MAX_TOKENS = 16000
VERIFY_SEEDS = (21, 22, 23)        # the W-A EDIT_SEEDS - same instrument, same discipline
SEED = 20260728                    # the study seed
# Three grades, strongest first. There is deliberately no "implied" grade: a span that only
# lets a reader INFER the fact is not a statement of it, and treating one as a site got real
# clinical content deleted in the first smoke (an exam finding that shared the word "pain"
# with a trap about self-applied ice). If it does not state the fact, it is not a site.
STRENGTHS = ("explicit", "paraphrase", "partial")
MAX_FACTS_DEFAULT = 14             # per consultation; see rank_facts()
DEFAULT_BUDGET = 40.0              # USD, this task's cap (the lead author's instruction, build+smoke)

# The prompts below are the v2 instrument, byte-identical to the one the smoke validated
# (docs/OMISSION-DESIGN-QA.md §5). What changed for the factorial run is the CANDIDATE SET
# the instrument is pointed at, not the instrument: see rank_facts().
INSTRUMENT_VERSION = "sitemap-v2"
CANDIDATE_SET_VERSION = "severity-stratified-v1"
# How the 14 slots per consultation are spent. The v1 candidate set preferred critical traps
# and load_bearing-high core facts, which is exactly what collapsed the frozen corpus to
# 255 critical / 26 supporting / 0 peripheral. These quotas spend the same number of slots
# deliberately across the severity range instead (the lead author's review decision 5, 2026-08-12).
QUOTAS = (("trap_critical", 2), ("trap_supporting", 2), ("trap_peripheral", 2),
          ("mc_core_high", 4), ("mc_core_medium", 2), ("mc_contextual", 6))
# Back-fill priority when a bucket is short. Deliberately the INVERSE of abundance: the
# corpus already has 65 complete-critical pairs and no peripheral ones at all, so a spare
# slot is worth most on a contextual fact and least on another load-bearing core fact.
BACKFILL_ORDER = ("mc_contextual", "mc_core_medium", "trap_supporting", "trap_peripheral",
                  "trap_critical", "mc_core_high")


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def norm(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()


def normlow(text):
    return norm(text).lower()


# ---------------------------------------------------------------- prompts
# Mapper: no transcript. Locating a fact's statements is a TEXTUAL task over the note,
# and the fact list already carries the ground truth - so the transcript would add
# ~10k tokens per call and nothing else. The verifier is deliberately given the same
# view, so it is checking the same evidence the mapper had.
MAP_SITES = """You are indexing a clinical note so a research tool can remove facts from it precisely.

For EACH numbered FACT below, find EVERY place in the NOTE that states it, in whole or in
part. Clinical notes repeat themselves: the same fact is often recorded in the history,
again in the examination or impression, and again in the plan, in different words. A fact
you list only once when it appears twice will break the tool. Miss nothing.

For each site give:
- span: the EXACT text from the note, copied character for character, that carries the
  statement. Quote the SMALLEST span that a reader could delete and thereby lose this
  statement, and NOTHING MORE. If the sentence also carries other clinical facts - another
  finding, a dose, a different piece of reasoning - quote only the clause that states THIS
  fact, not the whole sentence. Deleting a whole sentence to remove one fact inside it
  destroys the others, which is the single most common way this index goes wrong. Never
  normalise, correct or re-punctuate the note's wording.
- section: the section heading the span sits under, copied from the note ("(none)" if the
  note has no headings).
- strength: how fully THIS site states the fact -
    "explicit"   - states it plainly, in the note's own clinical terms
    "paraphrase" - states the same fact in different words
    "partial"    - states only part of it, losing a defining detail (a value, a laterality,
                   a hedge, or the half that drives action)
- primary: true for the ONE site that is the note's home statement of the fact - the site a
  reader would cite, carrying it most completely and specifically. EXACTLY ONE site per fact
  is primary; the rest are echoes, paraphrases or fragments. If a fact has one site, it is
  primary.

A site must STATE the fact. A span that merely touches on something related - the same body
part, the same symptom in a different context, a different finding that happens to share a
word - is NOT a site, however strongly it hints. Nor is a span from which the fact could only
be inferred. The tool deletes what you list, so a wrong site destroys real clinical content
that has nothing to do with the fact. When in doubt, leave it out.

If a fact is not in the note at all, return an empty sites list for it.

Return ONLY JSON:
{"facts":[{"idx":<the FACT number>,"sites":[{"span":"<exact text from the note>",
"section":"<heading>","strength":"explicit|paraphrase|partial","primary":true|false}]}]}

FACTS:
{facts}

NOTE:
{note}"""

VERIFY_SITES = """A research tool has indexed a clinical note: for each numbered FACT it lists every place in
the note it believes states that fact. Check the index, fact by fact. The tool deletes these
spans to build test data, so a MISSED site silently corrupts the experiment - the tool will
believe it removed a fact that is in truth still in the note.

For EACH fact:
1. sites: for each listed site, does the quoted span genuinely state THAT fact, in whole or
   in part? Give the correct strength grade for it (correct the tool's grade where it is
   wrong):
     "explicit"   - states the fact plainly, in the note's own clinical terms
     "paraphrase" - states the same fact in different words
     "partial"    - states only part of it, losing a defining detail (a value, a laterality,
                    a hedge, or the half that drives action)
   Answer states_fact FALSE for a span that only touches on something related - the same body
   part, the same symptom in a different context, a different finding sharing a word - or
   that merely lets the fact be inferred. Those are not sites, and the tool will delete them.
   Answer states_fact FALSE ALSO for a span that is WIDER than the fact - one that carries
   other clinical content (another finding, a dose, separate reasoning) alongside it. Say so
   in the reason, and give the narrower text in `missed` instead: deleting an over-wide span
   destroys clinical facts that were never the target.
2. missed: any OTHER place in the note that STATES this fact, in whole or in part, that the
   list does not contain. Quote each missed span exactly as it appears in the note. Clinical
   notes restate the same fact across sections, so this is the check that matters most - but
   add a span only if it genuinely states the fact. A wrong addition deletes unrelated
   clinical content; when in doubt, leave it out.
3. primary_i: the site that should be the note's home statement of the fact - the one
   carrying it most completely and specifically, the place a reader would go for it. Give its
   0-based index in that fact's SITES list, or -1 if none of the listed sites qualifies.

Cover every fact number, even the ones you agree with completely.

Return ONLY JSON:
{"facts":[{"idx":<the FACT number>,
"sites":[{"i":<0-based index into that fact's SITES>,"states_fact":true|false,
"strength":"explicit|paraphrase|partial",
"reason":"<one line, only when states_fact is false or you changed the strength>"}],
"missed":[{"span":"<exact text from the note>","section":"<heading>",
"strength":"explicit|paraphrase|partial"}],
"primary_i":<0-based index, or -1>,
"reason":"<one line covering any correction; empty string if the index is right>"}]}

FACTS AND THEIR INDEXED SITES:
{facts}

NOTE:
{note}"""

PROMPTS = {"MAP_SITES": MAP_SITES, "VERIFY_SITES": VERIFY_SITES}


# ---------------------------------------------------------------- substrate
def _frozen_notes():
    """The clean notes as FROZEN, keyed 'stratum|id', tagged with their split.

    eval     -> master/pairs_master_frozen.json ideal_notes (112 consultations)
    gepa_dev -> master/gepa_dev_pool.json consultations (22, carved out for W-F)
    Excluded notes never appear. The frozen artefact is opened read-only.
    """
    out = {}
    fz = json.load(open(os.path.join(MASTER, "pairs_master_frozen.json")))
    for n in fz["ideal_notes"]:
        if n.get("status") == "excluded" or not (n.get("note") or "").strip():
            continue
        out[f"{n['stratum']}|{n['id']}"] = {"note": n["note"], "split": "eval",
                                            "repaired": bool(n.get("repaired")),
                                            "repair_rounds": n.get("repair_rounds")}
    dev = json.load(open(os.path.join(MASTER, "gepa_dev_pool.json")))
    for c in dev["consultations"]:
        k = f"{c['stratum']}|{c['id']}"
        if k in out or not (c.get("ideal_note") or "").strip():
            continue
        out[k] = {"note": c["ideal_note"], "split": "gepa_dev", "repaired": None,
                  "repair_rounds": None}
    return out


def consultations(splits=("eval",)):
    """[{key, stratum, id, split, note, transcript, facts:[...]}] over the frozen substrate.

    Fact sheets, must_contain core view and trap metadata come from W-A's own
    `load_corpus()`, so this file cannot drift from the view W-A verified against.
    """
    corpus_notes, _ = WA.load_corpus()
    frozen = _frozen_notes()
    rows = []
    for key, fz in sorted(frozen.items()):
        if fz["split"] not in splits:
            continue
        cn = corpus_notes.get(key)
        if cn is None:
            continue
        facts = []
        for j, (fact, orig_idx) in enumerate(zip(cn["mc"], cn["mc_orig_idx"])):
            facts.append({"fact_key": f"mc|{orig_idx}", "kind": "must_contain",
                          "index": orig_idx, "core_index": j, "fact": fact,
                          "scope": "core",
                          "load_bearing": None, "importance": None, "mode": None})
        # load_bearing lives on the raw sheet rows; re-read it for the core items only
        lb = _load_bearing(cn)
        for f in facts:
            f["load_bearing"] = lb.get(f["index"])
        # CONTEXTUAL must_contain items - "legitimate content whose absence does not make the
        # note deficient" (consolidate_sheets' own definition). W-A's core view drops them, and
        # dropping them is most of why the corpus has no low-severity omissions at all: a
        # non-smoker line or a self-management tip is exactly what a PERIPHERAL omission looks
        # like. They are candidates here, and the rubric grades them like anything else.
        core_idx = set(cn["mc_orig_idx"])
        for i, m in enumerate(_sheets(cn["stratum"]).get(cn["id"], [])):
            if i in core_idx or not isinstance(m, dict) or m.get("scope") != "contextual":
                continue
            facts.append({"fact_key": f"mc|{i}", "kind": "must_contain", "index": i,
                          "core_index": None, "fact": m["fact"], "scope": "contextual",
                          "load_bearing": m.get("load_bearing"), "importance": None,
                          "mode": None})
        for i, t in enumerate(cn["trap_meta"]):
            # pair_eligible is Amendment Q metadata that only the re-extracted strata carry;
            # the authored fact sheets predate it, and its ABSENCE means "never graded", not
            # "excluded" (the original injector never ran on authored - those pairs were
            # preserved verbatim). So exclude only an explicit False.
            #
            # ...EXCEPT for peripheral traps. consolidate_sheets sets pair_eligible purely as
            # `importance != "peripheral"` - a severity judgement, not a feasibility one - so
            # that one line is why the frozen corpus has zero peripheral pairs. The factorial
            # needs them, so an omission-mode peripheral trap is a candidate again and carries
            # `pair_eligible_relaxed` to say so.
            if t.get("mode") != "omission":
                continue
            relaxed = (t.get("pair_eligible") is False
                       and t.get("importance") == "peripheral")
            if t.get("pair_eligible") is False and not relaxed:
                continue
            facts.append({"fact_key": f"trap|{i}", "kind": "trap", "index": i,
                          "core_index": None, "scope": "trap",
                          "fact": f"{t['trap']} (correct handling: {t['correct_handling']})",
                          "trap": t["trap"], "correct_handling": t["correct_handling"],
                          "load_bearing": None, "importance": t.get("importance"),
                          "mode": t.get("mode"),
                          "pair_eligible_relaxed": relaxed})
        rows.append({"key": key, "stratum": cn["stratum"], "id": cn["id"],
                     "split": fz["split"], "note": fz["note"],
                     "note_repaired": fz["repaired"], "transcript": cn["transcript"],
                     "facts": facts})
    return rows


_SHEET_CACHE = {}


def _sheets(stratum):
    """{id: must_contain rows} per stratum, read once (these files are 0.8-1.7MB each and
    consultations() would otherwise re-parse one per consultation)."""
    if stratum not in _SHEET_CACHE:
        if stratum == "authored":
            _SHEET_CACHE[stratum] = {
                s["id"]: s["fact_sheet"]["must_contain"]
                for s in json.load(open(os.path.join(HERE, "authored_scenarios.json")))}
        else:
            _SHEET_CACHE[stratum] = {
                r["id"]: r["fact_sheet"]["must_contain"] for r in
                json.load(open(os.path.join(MASTER, f"fact_sheets_{stratum}_core.json")))}
    return _SHEET_CACHE[stratum]


def _load_bearing(cn):
    """{orig_idx: load_bearing} for a consultation, from its source fact sheet. The authored
    sheets predate the field (their must_contain rows are plain strings) -> None."""
    mcs = _sheets(cn["stratum"]).get(cn["id"], [])
    return {i: (m.get("load_bearing") if isinstance(m, dict) else None)
            for i, m in enumerate(mcs)}


def bucket(f):
    """Which severity-stratum bucket a candidate fact belongs to. Buckets are PROXIES for
    severity, not severity itself - severity is graded on the rubric later, per fact
    (omission_factorial.py). Their only job is to make the mapped candidate set span the
    range instead of piling up on critical."""
    if f["kind"] == "trap":
        return f"trap_{f.get('importance') or 'critical'}"
    if f.get("scope") == "contextual":
        return "mc_contextual"
    return f"mc_core_high" if f.get("load_bearing") in (None, "high") else "mc_core_medium"


def rank_facts(row, max_facts=MAX_FACTS_DEFAULT, quotas=QUOTAS):
    """The omission-eligible candidate set for one consultation: `max_facts` facts spread
    deliberately across the severity range.

    The v1 rule (hard_negatives_master.pick_target, inherited by the smoke) ranked omission
    targets critical-trap-first, then load_bearing-high core facts. Pointed at 137
    consultations that produced 255 critical / 26 supporting / 0 peripheral pairs - the
    severity axis had no variance left to correlate anything with. Same slot count, spent
    differently: QUOTAS asks for 2 critical + 2 supporting + 2 peripheral omission traps,
    3 load-bearing-high core facts, 2 medium core facts and 3 contextual facts, then
    back-fills any short bucket from whatever else the consultation has (large buckets
    first, so a consultation with no peripheral trap still contributes 14 facts).

    Within a bucket the order is a seeded per-fact jitter on the study seed, so the set is
    deterministic and reproducible. The site-count distribution we report is over THIS
    candidate set, exactly as it was over the v1 one.
    """
    rng = random.Random(f"{SEED}|{row['key']}|omit-sites")
    pools = {}
    for f in row["facts"]:
        pools.setdefault(bucket(f), []).append((rng.random(), f))
    for v in pools.values():
        v.sort(key=lambda it: it[0])

    picked, taken = [], {k: 0 for k in pools}
    for name, want in quotas:
        pool = pools.get(name, [])
        take = min(want, len(pool))
        picked.extend(f for _, f in pool[:take])
        taken[name] = take
    # back-fill the shortfall in BACKFILL_ORDER, so every consultation contributes its full
    # quota of facts even when a bucket is empty - and a spare slot goes to the stratum the
    # dataset is short of, not to the one it already has hundreds of
    while len(picked) < max_facts:
        nxt = next((k for k in BACKFILL_ORDER
                    if len(pools.get(k, [])) - taken.get(k, 0) > 0), None)
        if nxt is None:
            break
        picked.append(pools[nxt][taken[nxt]][1])
        taken[nxt] += 1
    return picked[:max_facts]


# ---------------------------------------------------------------- span location
def find_span(note, span):
    """(start, end, n_matches) for a verbatim-ish span. Whitespace differences are
    tolerated (the models re-wrap); anything else must match exactly. n_matches != 1
    means the span cannot be used for construction (0 = hallucinated, >1 = ambiguous)."""
    s = norm(span)
    if not s:
        return None, None, 0
    pat = re.compile(r"\s+".join(re.escape(w) for w in s.split()), re.I)
    hits = list(pat.finditer(unicodedata.normalize("NFC", note)))
    if len(hits) != 1:
        return None, None, len(hits)
    return hits[0].start(), hits[0].end(), 1


def unit_alignment(note, start, end):
    """Is the located span exactly one or more whole units (sentences / bullet lines)?
    Unit-aligned spans delete cleanly with no LLM tidy pass - see inject_omissions_v2."""
    units = WA.units_of(note)
    target = normlow(note[start:end])
    covered, i = [], 0
    for idx, u in enumerate(units):
        if normlow(u) and normlow(u) in target:
            covered.append(idx)
    if not covered:
        return False, []
    joined = normlow(" ".join(units[covered[0]:covered[-1] + 1]))
    contiguous = covered == list(range(covered[0], covered[-1] + 1))
    return bool(contiguous and joined == target), covered


def site_record(note, raw, source):
    st, en, n = find_span(note, raw.get("span"))
    strength = raw.get("strength") if raw.get("strength") in STRENGTHS else "explicit"
    rec = {"span": norm(raw.get("span")), "section": norm(raw.get("section")) or "(none)",
           "strength": strength, "source": source, "span_found": n == 1,
           "n_matches": n, "char_span": [st, en] if n == 1 else None}
    if n == 1:
        aligned, units = unit_alignment(note, st, en)
        rec["unit_aligned"], rec["unit_indices"] = aligned, units
    else:
        rec["unit_aligned"], rec["unit_indices"] = False, []
    return rec


# ---------------------------------------------------------------- LLM plumbing
# RLock, not Lock: workers hold it while recording a result and save_state() takes it again
# to write the file atomically. A plain Lock self-deadlocks there (caught in the offline
# shakedown before any credit was spent).
_lock = threading.RLock()


class Budget:
    """This task's own cap, on top of the program tripwires. Persisted in the state file
    so reruns accumulate rather than resetting to zero."""

    def __init__(self, state, cap):
        self.state, self.cap = state, cap
        self.hit = threading.Event()

    @property
    def spent(self):
        return self.state.setdefault("spend", {}).get("cost_usd", 0.0)

    def ok(self):
        return self.spent < self.cap and not self.hit.is_set()

    def add(self, meta, stage):
        with _lock:
            sp = self.state.setdefault("spend", {"cost_usd": 0.0, "calls": 0, "by_stage": {}})
            c = meta.get("cost_usd_reported") or 0.0
            sp["cost_usd"] = round(sp.get("cost_usd", 0.0) + c, 6)
            sp["calls"] = sp.get("calls", 0) + len(meta.get("requests") or [1])
            bs = sp.setdefault("by_stage", {}).setdefault(stage, {"calls": 0, "cost_usd": 0.0})
            bs["calls"] += len(meta.get("requests") or [1])
            bs["cost_usd"] = round(bs["cost_usd"] + c, 6)
            if sp["cost_usd"] >= self.cap:
                self.hit.set()


CALL_TIMEOUT = 420      # seconds per HTTP attempt; --call-timeout overrides
CALL_RETRIES = 1        # attempts inside llm() beyond the first. Deliberately low: an
                        # auditor call over a whole-consultation map at effort high runs
                        # 2-8 minutes normally, and the first smoke had one consultation
                        # sit for 40+ minutes because a stuck call could burn
                        # 2 tries x 3 attempts x 420s before giving up. Worst case is now
                        # ~28 minutes, and --chunk keeps a stall from blocking a batch.


def call_json(budget, stage, prompt, role, effort, seed=None, tries=2, timeout=None):
    """One pinned call returning parsed JSON, or (None, error). Never returns '' as a
    verdict: llm() raises rather than yielding an empty completion (W1 R1)."""
    err = None
    for i in range(tries):
        if not budget.ok():
            return None, {"error": "budget_stop"}
        p = prompt if i == 0 else prompt + "\nReturn ONLY the JSON object, no prose."
        try:
            texts, meta = llm(p, role=role, temperature=1.0, k=1, seed=seed,
                              reasoning_effort=effort, max_tokens=MAX_TOKENS,
                              timeout=timeout or CALL_TIMEOUT, max_retries=CALL_RETRIES)
        except Exception as ex:
            err = str(ex)[:300]
            continue
        budget.add(meta, stage)
        obj = parse_json_blob(texts[0])
        if obj is not None:
            return obj, {"generation_ids": meta.get("generation_ids"), "seed": seed,
                         "cost": meta.get("cost_usd_reported"), "json_retry": i}
        err = "unparseable JSON"
    return None, {"error": err or "failed"}


# ---------------------------------------------------------------- stage 1: map
def render_facts(facts):
    return "\n".join(f"{i}. {f['fact']}" for i, f in enumerate(facts))


def map_one(row, facts, budget):
    prompt = (MAP_SITES.replace("{facts}", render_facts(facts))
              .replace("{note}", row["note"]))
    obj, meta = call_json(budget, "map", prompt, MAPPER_ROLE, MAPPER_EFFORT)
    if obj is None:
        return None, meta
    by_idx = {}
    for entry in (obj.get("facts") or []):
        try:
            i = int(entry.get("idx"))
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(facts):
            by_idx[i] = entry.get("sites") or []
    out = []
    for i, f in enumerate(facts):
        sites = [site_record(row["note"], s, "mapper") for s in by_idx.get(i, [])
                 if isinstance(s, dict)]
        primary_flags = [j for j, s in enumerate(by_idx.get(i, []))
                         if isinstance(s, dict) and s.get("primary")]
        out.append({**f, "bucket": bucket(f), "sites": sites,
                    "mapper_primary": primary_flags[0] if len(primary_flags) == 1 else None})
    return out, meta


# ---------------------------------------------------------------- stage 2: verify
def render_mapped(facts):
    """The whole consultation's map, as the verifier sees it. Verification is batched per
    NOTE rather than per fact: the note is the expensive shared prefix, and per-fact calls
    would cost ~14x more for the same judgement - 1,568 auditor calls at full scale instead
    of ~112. The escalation lever is per note for the same reason."""
    out = []
    for i, f in enumerate(facts):
        out.append(f"FACT {i}: {f['fact']}")
        if not f["sites"]:
            out.append("  SITES: (none found)")
        for j, s in enumerate(f["sites"]):
            out.append(f"  [{j}] section={s['section']} strength={s['strength']}"
                       f"{' PRIMARY' if f.get('mapper_primary') == j else ''}\n"
                       f"      {s['span']}")
        out.append("")
    return "\n".join(out)


def verify_batch(note, facts, budget, seed):
    """One auditor call over the whole consultation's map -> {fact_idx: per-fact verdict}."""
    prompt = (VERIFY_SITES.replace("{facts}", render_mapped(facts))
              .replace("{note}", note))
    obj, meta = call_json(budget, "verify", prompt, AUDITOR_ROLE, AUDITOR_EFFORT, seed=seed)
    if obj is None:
        return None, meta
    out = {}
    for entry in (obj.get("facts") or []):
        if not isinstance(entry, dict):
            continue
        try:
            fi = int(entry.get("idx"))
        except (TypeError, ValueError):
            continue
        per = {}
        for s in (entry.get("sites") or []):
            try:
                i = int(s.get("i"))
            except (TypeError, ValueError):
                continue
            per[i] = {"states_fact": bool(s.get("states_fact")),
                      "strength": s.get("strength") if s.get("strength") in STRENGTHS else None,
                      "reason": s.get("reason")}
        missed = [m for m in (entry.get("missed") or [])
                  if isinstance(m, dict) and m.get("span")]
        try:
            pi = int(entry.get("primary_i"))
        except (TypeError, ValueError):
            pi = -1
        out[fi] = {"per_site": per, "missed": missed, "primary_i": pi,
                   "reason": entry.get("reason"), "seed": seed}
    return out, meta


def apply_verification(note, fact, runs):
    """Fold the verifier run(s) into the map. Corrections are APPLIED and recorded.

    One run (the smoke lever): its verdict stands.
    Three runs (full scale, escalated when the screen amended anything): a site is
    dropped only on a 2/3 majority of states_fact=false; a missed span is added only when
    >=2 seeds name a span with the same normalised text; strength is the modal grade;
    primary is the modal nomination.
    """
    n = len(runs)

    def maj(vals, default=None):
        vals = [v for v in vals if v is not None]
        if not vals:
            return default
        top, ct = Counter(vals).most_common(1)[0]
        return top if (n == 1 or ct > n / 2) else default

    kept, dropped = [], []
    for i, s in enumerate(fact["sites"]):
        votes = [r["per_site"].get(i, {}).get("states_fact") for r in runs]
        states = maj(votes, default=True)
        if not states:
            dropped.append({**s, "dropped_reason": next(
                (r["per_site"].get(i, {}).get("reason") for r in runs
                 if r["per_site"].get(i, {}).get("reason")), "verifier: span does not state the fact"),
                "verifier_votes": votes})
            continue
        grade = maj([r["per_site"].get(i, {}).get("strength") for r in runs], default=s["strength"])
        kept.append({**s, "strength": grade or s["strength"],
                     "strength_changed": bool(grade and grade != s["strength"])})

    # missed spans: normalise, require the majority when escalated, and require the span to
    # be locatable in the note before it can be added (a hallucinated quote is not a site)
    seen = {normlow(s["span"]) for s in kept}
    counts, examples = Counter(), {}
    for r in runs:
        for m in r["missed"]:
            k = normlow(m.get("span"))
            if not k or k in seen:
                continue
            counts[k] += 1
            examples.setdefault(k, m)
    added, rejected = [], []
    for k, ct in counts.items():
        if n > 1 and ct < 2:
            rejected.append({"span": examples[k].get("span"), "votes": ct,
                             "reason": "below the 2/3 majority for an added site"})
            continue
        rec = site_record(note, examples[k], "verifier_added")
        if not rec["span_found"]:
            rejected.append({"span": rec["span"], "votes": ct,
                             "reason": f"span not locatable in the note (n_matches={rec['n_matches']})"})
            continue
        added.append(rec)
    sites = kept + added
    sites.sort(key=lambda s: (s["char_span"][0] if s["char_span"] else 10 ** 9))

    # primary: the verifier's nomination (mapped through the drop/keep shuffle), else the
    # mapper's, else the deterministic fallback - first EXPLICIT site in note order, then
    # first paraphrase, then the first site. The rule used is always recorded.
    primary, rule = None, None
    pv = maj([r["primary_i"] for r in runs if r["primary_i"] is not None and r["primary_i"] >= 0])
    if pv is not None and 0 <= pv < len(fact["sites"]):
        want = normlow(fact["sites"][pv]["span"])
        for j, s in enumerate(sites):
            if normlow(s["span"]) == want:
                primary, rule = j, "verifier_nominated"
                break
    if primary is None and fact.get("mapper_primary") is not None:
        mi = fact["mapper_primary"]
        if 0 <= mi < len(fact["sites"]):
            want = normlow(fact["sites"][mi]["span"])
            for j, s in enumerate(sites):
                if normlow(s["span"]) == want:
                    primary, rule = j, "mapper_nominated"
                    break
    if primary is None and sites:
        for grade in STRENGTHS:
            hit = [j for j, s in enumerate(sites) if s["strength"] == grade]
            if hit:
                primary, rule = hit[0], f"fallback_first_{grade}_site_in_note_order"
                break
    mp = fact.get("mapper_primary")
    repointed = (rule == "verifier_nominated" and mp is not None and primary is not None
                 and 0 <= mp < len(fact["sites"])
                 and normlow(sites[primary]["span"])
                 != normlow(fact["sites"][mp]["span"]))
    amended = bool(dropped or added or repointed
                   or any(s.get("strength_changed") for s in kept))
    verdict = "no_sites" if not sites else ("amended" if amended else "confirmed")
    for j, s in enumerate(sites):
        s["site_id"] = j
    return {"sites": sites, "primary_site_id": primary, "primary_rule": rule,
            "verification": {"verdict": verdict, "n_seeds": n,
                             "seeds": [r["seed"] for r in runs],
                             "dropped": dropped, "added": [a["span"] for a in added],
                             "rejected_missed": rejected,
                             "reasons": [r.get("reason") for r in runs if r.get("reason")]}}


# ---------------------------------------------------------------- driver
def load_state():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {"created": now_utc(), "spend": {"cost_usd": 0.0, "calls": 0, "by_stage": {}},
            "consults": {}}


def save_state(state):
    with _lock:
        tmp = STATE_FILE + ".tmp"
        json.dump(state, open(tmp, "w"), ensure_ascii=False, indent=1)
        os.replace(tmp, STATE_FILE)


def pick_smoke(rows, n=6):
    """A seeded stratified draw spanning all four strata (2 aci, 2 primock, 1 authored,
    1 trapblind), from the eval split only. Deterministic on the study seed."""
    want = {"aci": 2, "primock": 2, "authored": 1, "trapblind": 1}
    rng = random.Random(f"{SEED}|omission-sites-smoke")
    out = []
    for stratum, k in want.items():
        pool = sorted([r for r in rows if r["stratum"] == stratum], key=lambda r: r["key"])
        out.extend(rng.sample(pool, min(k, len(pool))))
    return out[:n]


def build(rows, state, budget, args):
    todo = [r for r in rows if state["consults"].get(r["key"], {}).get("status") != "done"]
    print(f"consultations: {len(rows)} in scope, {len(todo)} to map | mapper "
          f"{resolve_model(MAPPER_ROLE)['resolved']} | verifier "
          f"{resolve_model(AUDITOR_ROLE)['resolved']} | verify seeds "
          f"{VERIFY_SEEDS[:1] if args.verify_seeds == 1 else VERIFY_SEEDS} | budget cap "
          f"${budget.cap:.0f} (spent ${budget.spent:.2f})", flush=True)
    if not todo:
        return 0

    def one(row):
        if not budget.ok():
            return
        facts = rank_facts(row, args.max_facts)
        mapped, meta = map_one(row, facts, budget)
        if mapped is None:
            with _lock:
                state["consults"][row["key"]] = {"status": "map_failed", "error": meta,
                                                 "utc": now_utc()}
            print(f"  {row['key']:34} MAP FAILED ({meta.get('error')})", flush=True)
            return
        # one auditor call over the whole map; escalated (A5 lever) only when the screen
        # wants to change something anywhere in this consultation
        batches, errs = [], []
        for si, seed in enumerate(VERIFY_SEEDS[:max(1, args.verify_seeds)]):
            got, vmeta = verify_batch(row["note"], mapped, budget, seed)
            if got is None:
                errs.append(vmeta)
                continue
            batches.append(got)
            if si == 0 and args.verify_seeds > 1:
                quiet = all(
                    not v["missed"]
                    and all(x.get("states_fact") for x in v["per_site"].values())
                    and len(v["per_site"]) == len(mapped[fi]["sites"])
                    for fi, v in got.items() if 0 <= fi < len(mapped))
                if quiet and len(got) == len(mapped):
                    break
        out_facts = []
        for fi, f in enumerate(mapped):
            base = {k: v for k, v in f.items() if k not in ("sites", "mapper_primary")}
            if not f["sites"]:
                out_facts.append({**base, "sites": [], "n_sites": 0, "primary_site_id": None,
                                  "primary_rule": None,
                                  "verification": {"verdict": "not_in_note", "n_seeds": 0,
                                                   "seeds": [], "dropped": [], "added": [],
                                                   "rejected_missed": [], "reasons": []}})
                continue
            runs = [b[fi] for b in batches if fi in b]
            if not runs:
                out_facts.append({**base, "sites": f["sites"], "n_sites": len(f["sites"]),
                                  "primary_site_id": None, "primary_rule": None,
                                  "verification": {"verdict": "verify_failed", "n_seeds": 0,
                                                   "seeds": [], "dropped": [], "added": [],
                                                   "rejected_missed": [], "errors": errs,
                                                   "reasons": []}})
                continue
            folded = apply_verification(row["note"], f, runs)
            out_facts.append({**base, **folded, "n_sites": len(folded["sites"])})
        with _lock:
            state["consults"][row["key"]] = {
                "status": "done", "utc": now_utc(), "split": row["split"],
                "stratum": row["stratum"], "id": row["id"],
                "clean_sha": WA.sha16(row["note"]), "n_facts": len(out_facts),
                "facts": out_facts}
            save_state(state)
        dist = Counter(f["n_sites"] for f in out_facts)
        print(f"  {row['key']:34} {len(out_facts)} facts | sites/fact "
              f"{dict(sorted(dist.items()))} | ${budget.spent:.2f}", flush=True)

    if args.chunk:
        todo = todo[:args.chunk]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(one, todo))
    save_state(state)
    return sum(1 for r in rows if state["consults"].get(r["key"], {}).get("status") != "done")


def write_sites_file(rows, state, args):
    cons = []
    for r in rows:
        e = state["consults"].get(r["key"])
        if not e or e["status"] != "done":
            continue
        # the clean note travels WITH the map: every span offset in `facts` is an offset
        # into this exact text, so the two must never be able to drift apart
        cons.append({"key": r["key"], "stratum": r["stratum"], "id": r["id"],
                     "split": r["split"], "clean_sha": e["clean_sha"], "note": r["note"],
                     "note_repaired": r["note_repaired"], "facts": e["facts"]})
    dist = Counter()
    for c in cons:
        for f in c["facts"]:
            dist[f["n_sites"]] += 1
    present = [f for c in cons for f in c["facts"] if f["n_sites"] > 0]
    out = {
        "generated_utc": now_utc(), "spec": SPEC,
        "what": "every mention site of every omission-eligible fact, per consultation, in "
                "the FROZEN clean note - the substrate for complete vs partial omission "
                "construction (inject_omissions_v2.py)",
        "models": {"mapper": {"role": MAPPER_ROLE, **resolve_model(MAPPER_ROLE),
                              "reasoning_effort": MAPPER_EFFORT},
                   "verifier": {"role": AUDITOR_ROLE, **resolve_model(AUDITOR_ROLE),
                                "reasoning_effort": AUDITOR_EFFORT,
                                "seeds": list(VERIFY_SEEDS[:max(1, args.verify_seeds)])}},
        "prompt_sha256_16": {k: WA.sha16(v) for k, v in PROMPTS.items()},
        "instrument_version": INSTRUMENT_VERSION,
        "candidate_set_version": CANDIDATE_SET_VERSION,
        "params": {"seed": SEED, "max_facts_per_consult": args.max_facts,
                   "candidate_set": "severity-stratified quotas over omission-mode salience "
                                    "traps (critical/supporting/peripheral), core must_contain "
                                    "(load_bearing high then medium) and CONTEXTUAL "
                                    "must_contain, back-filled from the deepest pools, "
                                    "study-seed jitter within a bucket",
                   "quotas": dict(QUOTAS),
                   "verify_seeds": args.verify_seeds, "smoke": bool(args.smoke)},
        "counts": {"consultations": len(cons),
                   "by_stratum": dict(Counter(c["stratum"] for c in cons)),
                   "by_split": dict(Counter(c["split"] for c in cons)),
                   "facts_mapped": sum(len(c["facts"]) for c in cons),
                   "by_bucket": dict(Counter(f.get("bucket") for c in cons
                                             for f in c["facts"])),
                   "by_bucket_present": dict(Counter(f.get("bucket") for f in present)),
                   "multi_site_by_bucket": dict(Counter(f.get("bucket") for f in present
                                                        if f["n_sites"] >= 2)),
                   "facts_present_in_note": len(present),
                   "facts_absent_from_note": sum(1 for c in cons for f in c["facts"]
                                                 if f["n_sites"] == 0),
                   "sites_total": sum(f["n_sites"] for f in present),
                   "site_count_distribution": {str(k): v for k, v in sorted(dist.items())},
                   "multi_site_facts": sum(1 for f in present if f["n_sites"] >= 2),
                   "multi_site_share": round(sum(1 for f in present if f["n_sites"] >= 2)
                                             / max(len(present), 1), 4),
                   "mean_sites_per_present_fact": round(
                       sum(f["n_sites"] for f in present) / max(len(present), 1), 3),
                   "verification_verdicts": dict(Counter(
                       f["verification"]["verdict"] for c in cons for f in c["facts"])),
                   "primary_rule": dict(Counter(f.get("primary_rule") for f in present)),
                   "unusable_spans": sum(1 for f in present for s in f["sites"]
                                         if not s["span_found"]),
                   "unit_aligned_sites": sum(1 for f in present for s in f["sites"]
                                             if s.get("unit_aligned"))},
        "spend_usd": state["spend"]["cost_usd"], "calls": state["spend"]["calls"],
        "consultations": cons,
    }
    tmp = SITES_FILE + ".tmp"
    json.dump(out, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, SITES_FILE)
    return out


def report(out):
    c = out["counts"]
    print(f"\nmaster/fact_sites.json - {c['consultations']} consultations "
          f"{c['by_stratum']} split {c['by_split']}")
    print(f"  facts mapped {c['facts_mapped']} | present in the note "
          f"{c['facts_present_in_note']} | absent {c['facts_absent_from_note']}")
    print(f"  SITE-COUNT DISTRIBUTION (sites per present fact): {c['site_count_distribution']}")
    print(f"  multi-site facts {c['multi_site_facts']} ({100 * c['multi_site_share']:.1f}%) | "
          f"mean sites/fact {c['mean_sites_per_present_fact']}")
    print(f"  candidate buckets mapped {c.get('by_bucket')}")
    print(f"    present {c.get('by_bucket_present')}")
    print(f"    multi-site {c.get('multi_site_by_bucket')}")
    print(f"  verification {c['verification_verdicts']} | primary rule {c['primary_rule']}")
    print(f"  unit-aligned sites {c['unit_aligned_sites']}/{c['sites_total']} | "
          f"unlocatable spans {c['unusable_spans']}")
    print(f"  spend ${out['spend_usd']:.2f} over {out['calls']} calls")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true", help="6 consultations spanning the strata")
    ap.add_argument("--full", action="store_true", help="every eval-split consultation")
    ap.add_argument("--consults", help="comma-separated 'stratum|id' keys")
    ap.add_argument("--splits", default="eval", help="eval | gepa_dev | eval,gepa_dev")
    ap.add_argument("--max-facts", type=int, default=MAX_FACTS_DEFAULT)
    ap.add_argument("--verify-seeds", type=int, default=1,
                    help="1 = screen only (smoke); 3 = W-A A5 lever (escalate on amendment)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=0, help="consultations per invocation")
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET)
    ap.add_argument("--call-timeout", type=int, default=CALL_TIMEOUT,
                    help="seconds per HTTP attempt (see CALL_TIMEOUT)")
    ap.add_argument("--allow-dirty", action="store_true", help="smoke only: run on a dirty tree")
    ap.add_argument("--report", action="store_true", help="free: rewrite + print the artefact")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    globals()["CALL_TIMEOUT"] = args.call_timeout
    rows = consultations(tuple(s.strip() for s in args.splits.split(",") if s.strip()))
    if args.consults:
        want = {k.strip() for k in args.consults.split(",") if k.strip()}
        rows = [r for r in rows if r["key"] in want]
    elif args.smoke:
        rows = pick_smoke(rows)
    elif not args.full and not args.report:
        sys.exit("pick a scope: --smoke, --full, or --consults")

    state = load_state()
    if args.report:
        report(write_sites_file(rows, state, args))
        return 0

    budget = Budget(state, args.budget)
    print(f"scope: {len(rows)} consultations")
    for r in rows[:8]:
        n_facts = len(rank_facts(r, args.max_facts))
        print(f"    {r['key']:34} {r['split']:8} {len(r['note'])} note chars, "
              f"{n_facts} candidate facts")
    if not args.yes:
        try:
            if input(f"map {len(rows)} consultations (cap ${args.budget:.0f})? [y/N] ").strip().lower() != "y":
                sys.exit("aborted")
        except EOFError:
            sys.exit("aborted (non-interactive)")

    params = {"scope": "smoke" if args.smoke else ("full" if args.full else "named"),
              "n_consultations": len(rows), "max_facts": args.max_facts,
              "verify_seeds": args.verify_seeds, "splits": args.splits,
              "budget_cap_usd": args.budget}
    if args.smoke:
        params["NOT_PAPER_BOUND"] = "smoke: instrument shakedown, not a corpus build"
    left = 0
    with Run(EXPERIMENT, params=params, seed=SEED,
             inputs=["master/pairs_master_frozen.json"], spec=SPEC,
             allow_dirty=args.allow_dirty) as run:
        run.register_prompts(PROMPTS)
        left = build(rows, state, budget, args)
        out = write_sites_file(rows, state, args)
        run.save("fact_sites_summary.json", out["counts"])
    report(out)
    if budget.hit.is_set():
        print(f"\nBUDGET STOP at ${budget.spent:.2f} (cap ${budget.cap:.0f}) - state saved.")
        return 3
    return 0 if left == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
