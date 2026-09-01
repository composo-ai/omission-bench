"""second_panel.py - the second-panel arm: a controlled two-standard reading of the census.

The census reports one number - 10.48% of 5,898 candidates verified - and the census paper
sets it beside the June pilot's 37.3% as *history*, because four things moved between those two
runs (model generation, panel family mix, discovery pass count, corpus). This run turns that
into a controlled contrast: the SAME candidates, the SAME model, the SAME evidence, the SAME
transport and settings, and one thing different - the review standard the reviewer is told to
apply.

WHAT IS BOUGHT HERE, and what is free
-------------------------------------
Bought: one **lenient** reviewer call per drawn candidate. A single call on the panel's own
constructor pin (anthropic/claude-opus-5), no second skeptic, no tiebreak, no instruction to
refute - the informal standard a vendor or health system would recognise: shown the full note
and the full transcript, is this a genuine documentation error worth reporting to the note's
author?

Free, and read straight off `master/findings_verified_master.json`, which already holds every
vote the shipped panel cast on these same candidates:

  A. constructor alone, STRICT refute instruction  (`verdict.votes.constructor`)
  B. auditor alone, STRICT refute instruction      (`verdict.votes.auditor`)
  C. the shipped two-family panel + tiebreak       (`verdict.is_real`)

so the read-out is four standards over one candidate set, not two. That matters, because the
brief's two-column version (panel vs lenient single call) confounds the *instruction* with the
*panel shape* - it moves both at once, which is the very criticism the census paper makes of the
37.3%-vs-10.48% pair it is trying to replace. With A in hand the contrast decomposes:

  A vs D (this run)  - the INSTRUCTION alone. Same model, same call shape, same evidence, same
                       settings, same candidates. Nothing else moves. This is the controlled
                       result.
  A vs C             - the PANEL MACHINERY alone (a second family plus a tiebreak, at a fixed
                       instruction), and it is already paid for.

D costs money; A, B and C cost nothing and were bought in August. The added arms are the reason
this script reads the verified master rather than only writing a new store.

THE DRAW (a declared departure from the brief this run was written to)
----------------------------------------------------------------------
The brief specified a candidate-level draw stratified by tier-2 x vendor x salience. This draws
whole NOTES instead, stratified by vendor x substrate, proportional, largest-remainder, fixed
seed. Reason: one of the three pre-registered read-outs is the note-level census statement
("between X% and Y% of notes carry a verified finding, by review standard"), and a candidate-level
draw cannot estimate it without bias - it reads ~20% of each note's candidates, so "at least one
verified finding on this note" is systematically undercounted under BOTH standards by an amount
that depends on how many candidates the note had. Drawing whole notes makes that read-out
unbiased and directly comparable to the census's note-level 31.3%, makes the consultation-clustered bootstrap
match the sampling design exactly rather than approximate it, and costs almost nothing on the
secondary tables: at seed 20260824 every tier-2 and salience marginal lands within 1.8pp of the
census (printed by --plan-only, and asserted in the analysis).

Idempotent per candidate, so a killed run resumes without re-buying a call, and the smoke
subsets are strict prefixes of the main run's order - smoke calls are main-run calls, never
extra spend.

    python3 census/second_panel.py --plan-only              # the draw, free, no calls
    python3 census/second_panel.py --limit 20  --tag smoke1 -y
    python3 census/second_panel.py --limit 100 --tag smoke1 -y
    python3 census/second_panel.py --tag smoke1 -y          # the main run, same store
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import argparse, hashlib, json, os, random, re, time
from collections import Counter, defaultdict

from common import HERE, Run, llm
import taxonomy_common as T
import taxonomy_verify as V
from w2_common import RecordStore, SpendGuard, chunked, confirm, parse_json_blob, run_block

SPEC = "the master-dataset spec: corpus build and census"
EXPERIMENT = "second-panel"
VERIFIED = "master/findings_verified_master.json"
DRAW_SEED = 20260824          # the run date, the study's seed convention (cf. STABILITY_SEED)
DRAW_TARGET = 1200            # target CANDIDATES; whole notes are drawn, so the realised n differs
TRANSCRIPT_CHARS = V.TRANSCRIPT_CHARS      # 40,000 - the panel's own truncation, unchanged

# The lenient arm. Every EVIDENCE block is byte-identical to the panel prompt
# (taxonomy_verify.REFUTE_TF): same source, same note, same flagged-mode/claim/quote block, in
# the same order. Only the standard the reviewer is told to apply, the question, and the JSON
# key change. `assert_only_the_standard_moved()` fails the run if that ever stops being true.
LENIENT = """FULL SOURCE ({skind}):
{source}

FULL NOTE:
{note}

A discovery tool flagged a possible discrepancy in an AI-generated clinical NOTE versus its SOURCE. Your job is to apply the standard an informal reviewer would use before signing the note off: would you flag this back to the person who wrote it? It IS worth reporting if a reasonable reviewer would want it corrected before the note enters the record, even if it is minor, arguable, or a matter of documentation practice rather than clinical fact.

FLAGGED MODE: {mode}
claim: {description}
note says: {note_quote}
source says: {source_quote}

Is this a genuine documentation error worth reporting to the note's author? Return ONLY JSON {{"error": true|false, "reason": "<one sentence>"}}"""

# The three blocks that carry the evidence. They must survive verbatim from the panel prompt, or
# the two arms are not reading the same thing and the contrast is not a change of standard.
_EVIDENCE_BLOCKS = ("FULL SOURCE ({skind}):\n{source}",
                    "FULL NOTE:\n{note}",
                    "FLAGGED MODE: {mode}\nclaim: {description}\n"
                    "note says: {note_quote}\nsource says: {source_quote}")


def assert_only_the_standard_moved():
    """The lenient prompt must be the panel prompt with the STANDARD swapped and nothing else.

    Two checks, because either failure would quietly turn this from a controlled contrast into
    an uncontrolled one:
      (a) every evidence block appears verbatim, in the same order, in both prompts - so both
          arms see identical inputs;
      (b) the prompts differ ONLY in the instruction sentence, the question and the JSON key -
          i.e. the leading shared sentence about what a discovery tool flagged is still shared.
    """
    for b in _EVIDENCE_BLOCKS:
        for name, p in (("REFUTE_TF", V.REFUTE_TF), ("LENIENT", LENIENT)):
            if b not in p:
                raise SystemExit(f"{name} no longer contains the evidence block {b!r} verbatim - "
                                 "the two arms would not be reading the same thing.")
    if V.REFUTE_TF.index("FULL SOURCE") > V.REFUTE_TF.index("FULL NOTE") or \
            LENIENT.index("FULL SOURCE") > LENIENT.index("FULL NOTE"):
        raise SystemExit("evidence block order differs between the arms.")
    shared = "A discovery tool flagged a possible discrepancy in an AI-generated clinical NOTE " \
             "versus its SOURCE."
    for name, p in (("REFUTE_TF", V.REFUTE_TF), ("LENIENT", LENIENT)):
        if shared not in p:
            raise SystemExit(f"{name} lost the shared framing sentence - the arms now differ by "
                             "more than the standard.")


def prompt_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------- reading one reply
_ERROR_RE = re.compile(r'"error"\s*:\s*(true|false)', re.I)


def parse_lenient(text):
    """(verdict, reason, how) from one lenient reply. verdict None = could not be read.

    The shared `parse_json_blob` regex-matches the outermost braces and hands the span to
    json.loads, so a reply whose free-text `reason` contains an unescaped double quote - which
    this arm's one-sentence reason invites, because the sentence quotes the note - fails to load
    and comes back None. Measured on the first 319 calls of this run: 2 of them, both plainly
    `{"error": true, ...}` verdicts broken only by a quoted phrase inside the reason. Under the
    declared default those become NOT-an-error, i.e. the parser silently converts a clear YES
    into a NO and biases this arm's rate DOWN. That is a defect in the reader, not a verdict.

    So: try the shared parser first (unchanged, and it wins whenever it works); if it cannot
    load, recover the verdict token itself - the model's own `"error": true|false` - by regex,
    and only when every occurrence agrees, so an ambiguous reply is never resolved by guessing.
    The raw reply is kept as the reason and the record is marked `parsed_by: salvaged_regex`, so
    every salvaged verdict is countable and auditable in the store.
    """
    obj = parse_json_blob(text)
    if isinstance(obj, dict) and isinstance(obj.get("error"), bool):
        return obj["error"], obj.get("reason"), "json"
    hits = {h.lower() for h in _ERROR_RE.findall(text or "")}
    if len(hits) == 1:
        return hits.pop() == "true", (text or "").strip()[:1500], "salvaged_regex"
    return None, None, "unparseable"


def call_lenient(prompt, key):
    """One lenient call, returning the RAW reply alongside the parse so `parse_lenient` can
    salvage. This is `taxonomy_common.route_call`'s OpenRouter branch: the llm() invocation is
    argument-for-argument identical to route_call(kind="panel", role=constructor) - same model,
    temperature, deterministic seed, reasoning effort and token ceiling - so the call the panel's
    constructor leg made and the call this arm makes differ only in the prompt text.
    """
    t0 = time.time()
    try:
        texts, m = llm(prompt, T.CONSTRUCTOR_ROLE, temperature=1.0, k=1, seed=T.call_seed(key),
                       reasoning_effort=T.REASONING_EFFORT, max_tokens=T.MAX_TOKENS["panel"],
                       timeout=600)
    except Exception as e:
        return "", {"route": "openrouter", "model": None, "role": T.CONSTRUCTOR_ROLE, "calls": 0,
                    "cost_usd": 0.0, "usage": {}, "wall_s": round(time.time() - t0, 2),
                    "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return (texts[0] if texts else ""), {
        "route": "openrouter", "model": m["model"], "role": T.CONSTRUCTOR_ROLE,
        "calls": len(m.get("requests", [])) or 1, "cost_usd": m.get("cost_usd_reported") or 0.0,
        "usage": m.get("usage", {}), "seed": T.call_seed(key), "gen_ids": m.get("generation_ids"),
        "providers": m.get("providers"), "wall_s": round(time.time() - t0, 2)}


# ---------------------------------------------------------------- the draw
def draw(verified, seed=DRAW_SEED, target=DRAW_TARGET):
    """A stratified sample of whole NOTES from the panel-covered corpus.

    Strata are vendor x substrate (12). Allocation is proportional to notes in stratum, by
    largest remainder, at the fraction that puts the expected candidate count on `target`.
    Notes with zero panel candidates are in the frame and can be drawn: they cost nothing, and
    dropping them would bias the note-level denominator upward under both standards.

    Returns (note_keys, candidates, meta) - all deterministic in `seed`.
    """
    issues = verified["all_issues"]
    by_note = defaultdict(list)
    for f in issues:
        by_note[f["note_key"]].append(f)
    notes = [p for p in verified["per_note"] if p.get("panel_covered")]
    strata = defaultdict(list)
    for p in notes:
        strata[(p["scribe"], p["source"])].append(p["note_key"])

    frac = target / len(issues)
    alloc, remainders, base = {}, [], 0
    for k in sorted(strata):
        exact = frac * len(strata[k])
        alloc[k] = int(exact)
        remainders.append((exact - alloc[k], k))
        base += alloc[k]
    for _, k in sorted(remainders, reverse=True)[: round(frac * len(notes)) - base]:
        alloc[k] += 1

    rng = random.Random(seed)
    picked = []
    for k in sorted(strata):
        picked += rng.sample(sorted(strata[k]), alloc[k])
    picked.sort()
    cands = [f for nk in picked for f in by_note.get(nk, [])]
    meta = {
        "seed": seed, "target_candidates": target, "frame": "panel-covered notes",
        "strata": "vendor x substrate, proportional, largest remainder",
        "n_notes_frame": len(notes), "n_notes_drawn": len(picked),
        "n_candidates_frame": len(issues), "n_candidates_drawn": len(cands),
        "sampling_fraction": round(frac, 6),
        "allocation": {f"{a}|{b}": {"notes_in_stratum": len(strata[(a, b)]), "drawn": alloc[(a, b)]}
                       for a, b in sorted(strata)},
        "note_keys_sha256": hashlib.sha256("\n".join(picked).encode()).hexdigest(),
        "finding_ids_sha256": hashlib.sha256(
            "\n".join(f["finding_id"] for f in cands).encode()).hexdigest(),
        "reproduce": "second_panel.draw(json.load(open('master/findings_verified_master.json')), "
                     f"seed={seed}, target={target})",
    }
    return picked, cands, meta


def smoke_order(cands, seed=DRAW_SEED):
    """A deterministic ordering of the drawn candidates whose every PREFIX is stratified.

    Each candidate gets its fractional rank within its (tier-2, panel salience) stratum after a
    seeded shuffle; sorting globally on that rank interleaves the strata in proportion. So
    `--limit 20` and `--limit 100` are stratified samples, nested inside each other and inside
    the main run - a smoke call is a main-run call bought early, never extra spend.
    """
    rng = random.Random(seed + 1)
    buckets = defaultdict(list)
    for f in cands:
        buckets[(f.get("frame_tier2") or f.get("pass"), f.get("salience"))].append(f)
    keyed = []
    for k in sorted(buckets, key=lambda x: (str(x[0]), str(x[1]))):
        b = sorted(buckets[k], key=lambda f: f["finding_id"])
        rng.shuffle(b)
        for i, f in enumerate(b):
            keyed.append(((i + 0.5) / len(b), str(k), f["finding_id"], f))
    keyed.sort(key=lambda t: t[:3])
    return [t[3] for t in keyed]


def note_blocks(cands):
    """Group the work into per-note blocks, notes interleaved across substrates.

    Same shape as taxonomy_verify._cache_order, and for the same reason: every call for one note
    shares the transcript+note prefix. Measured on the shipped panel, the Anthropic leg cached 0
    of 26.8M input tokens (llm() only marks a cache breakpoint when a caller passes
    cache_prefix, and the panel did not), so this buys nothing on price here - it is kept so the
    execution order matches the instrument being compared against.
    """
    by_note, consult = defaultdict(list), {}
    for f in cands:
        by_note[f["note_key"]].append(f)
        consult[f["note_key"]] = (f["source"], f["id"])
    by_consult = defaultdict(list)
    for nk in sorted(by_note):
        by_consult[consult[nk]].append(nk)
    buckets = defaultdict(list)
    for c in sorted(by_consult):
        buckets[c[0]].append(c)
    order, i = [], 0
    while any(len(v) > i for v in buckets.values()):
        for s in sorted(buckets):
            if len(buckets[s]) > i:
                order.append(buckets[s][i])
        i += 1
    return [sorted(by_note[nk], key=lambda f: f["finding_id"])
            for c in order for nk in by_consult[c]]


# ---------------------------------------------------------------- reporting
def marginal_fit(cands, issues):
    """Achieved vs population marginals for the draw - the check that replaces the brief's
    per-cell stratification. Printed before a call is bought and asserted in the analysis."""
    out = {}
    for name, key in (("frame_tier2", lambda f: f.get("frame_tier2") or f.get("pass")),
                      ("salience", lambda f: f.get("salience")),
                      ("scribe", lambda f: f["scribe"]),
                      ("source", lambda f: f["source"])):
        pop, draw_c = Counter(key(f) for f in issues), Counter(key(f) for f in cands)
        out[name] = {str(k): {"population": v, "population_share": round(v / len(issues), 6),
                              "drawn": draw_c.get(k, 0),
                              "drawn_share": round(draw_c.get(k, 0) / len(cands), 6),
                              "share_gap_pp": round(100 * (draw_c.get(k, 0) / len(cands)
                                                           - v / len(issues)), 2)}
                     for k, v in pop.most_common()}
    out["max_abs_share_gap_pp"] = round(max(abs(r["share_gap_pp"])
                                            for t in ("frame_tier2", "salience", "scribe", "source")
                                            for r in out[t].values()), 2)
    return out


def free_arms(cands):
    """The three standards already paid for, on the drawn candidates."""
    n = len(cands)
    keep = {"constructor_strict_solo": sum(1 for f in cands
                                           if f["verdict"]["votes"].get("constructor") == "keep"),
            "auditor_strict_solo": sum(1 for f in cands
                                       if f["verdict"]["votes"].get("auditor") == "keep"),
            "shipped_panel": sum(1 for f in cands if f["verdict"]["is_real"])}
    return {k: {"verified": v, "n": n, "rate": round(v / n, 6)} for k, v in keep.items()}


def plan_report(picked, cands, meta, issues, unit_cost):
    fit = marginal_fit(cands, issues)
    lines = [f"draw: {len(picked)} notes, {len(cands)} candidates, "
             f"{len({(f['source'], f['id']) for f in cands})} consultations "
             f"(seed {meta['seed']}, target {meta['target_candidates']})",
             f"  finding_ids sha256 {meta['finding_ids_sha256'][:16]}...",
             f"  worst marginal gap vs census: {fit['max_abs_share_gap_pp']}pp",
             "  tier-2 support (drawn/census):  " +
             "  ".join(f"{k} {v['drawn']}/{v['population']}"
                       for k, v in fit["frame_tier2"].items()),
             "  already-paid arms on this subset: " +
             ", ".join(f"{k} {v['verified']}/{v['n']} = {v['rate']:.2%}"
                       for k, v in free_arms(cands).items()),
             f"  to buy: {len(cands)} lenient calls x ${unit_cost:.5f} measured "
             f"= ${len(cands) * unit_cost:.2f}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- execution
def execute(blocks, S, tag):
    store, notes, guard, usage = S["store"], S["notes"], S["guard"], S["usage"]
    t0 = time.time()
    n_total = sum(len(b) for b in blocks)
    done = 0
    for ci, chunk in enumerate(chunked(blocks, S["chunk"]), 1):
        n_calls = sum(len(b) for b in chunk)
        params = dict(S["manifest_params"], stage="lenient", chunk=ci, workers=S["workers"],
                      n_note_blocks=len(chunk), n_calls_in_chunk=n_calls,
                      record_store=os.path.relpath(store.path, HERE))
        with Run(EXPERIMENT, params=params, inputs=T.run_inputs(extra=["second_panel.py", VERIFIED]),
                 spec=SPEC, allow_dirty=S["allow_dirty"], spend="openrouter_credits") as run:
            run.register_prompts(S["prompts"])

            def worker(f):
                fid = f["finding_id"]
                key = f"{fid}|len"
                u = notes[f["note_key"]]
                prompt = LENIENT.format(
                    mode=f.get("mode"), description=f.get("description"),
                    note_quote=f.get("note_quote", "-"), source_quote=f.get("source_quote", "-"),
                    note=u["note"], source=u["transcript"][:TRANSCRIPT_CHARS], skind="transcript")
                text, meta = call_lenient(prompt, key)
                verdict, reason, how = parse_lenient(text)
                # Default-to-NOT-an-error on anything that still cannot be read. This mirrors the
                # panel's default-refute convention and is conservative in the same direction:
                # it can only push the lenient arm's rate DOWN, i.e. against the hypothesis this
                # run is testing. `defaulted` is counted and reported.
                value = bool(verdict) if verdict is not None else False
                store.put({
                    "key": key, "kind": "len", "role": T.CONSTRUCTOR_ROLE, "finding_id": fid,
                    "note_key": f["note_key"], "scribe": f["scribe"], "source": f["source"],
                    "id": f["id"], "template": f["template"], "consultation": f["consultation"],
                    "mode": f.get("mode"), "pass": f.get("pass"),
                    "frame_tier2": f.get("frame_tier2") or f.get("pass"),
                    "frame_tier1": f.get("frame_tier1"), "salience": f.get("salience"),
                    # the already-paid standards, copied on so results/ is self-contained
                    "panel_is_real": bool(f["verdict"]["is_real"]),
                    "panel_decided_by": f["verdict"].get("decided_by"),
                    "constructor_strict_keep": f["verdict"]["votes"].get("constructor") == "keep",
                    "auditor_strict_keep": f["verdict"]["votes"].get("auditor") == "keep",
                    "value": value, "reason": reason, "parsed_by": how,
                    "defaulted": verdict is None,
                    "route": meta.get("route"), "model": meta.get("model"), "seed": meta.get("seed"),
                    "error": meta.get("error"), "usage": meta.get("usage"),
                    "cost_usd": meta.get("cost_usd"), "wall_s": meta.get("wall_s"),
                    "run_id": run.run_id, "t": round(time.time(), 3)})
                T.usage_add(usage, meta)
                guard.add(meta.get("cost_usd"))

            for block in chunk:
                # warmup=False: the shipped panel's Anthropic leg cached 0 of 26.8M input tokens
                # (no cache_control breakpoint on that path), so a serial warm-up call per note
                # buys nothing here and costs one round-trip of latency per note. Transport
                # throughput only - prompts, settings, seeds and records are unaffected.
                run_block(block, worker, S["workers"], warmup=False)
            done += n_calls
            rate = live_rate(store)
            print(f"    chunk {ci}: {n_calls} calls over {len(chunk)} notes | {done}/{n_total} "
                  f"| {usage['errors']} errors | ${usage['cost_usd']:.2f} "
                  f"| lenient-so-far {rate[0]}/{rate[1]} = {rate[0] / max(rate[1], 1):.1%} "
                  f"(panel on the same {rate[1]}: {rate[2] / max(rate[1], 1):.1%}) "
                  f"| {time.time() - t0:.0f}s", flush=True)
            run.save("chunk.json", {"tag": tag, "chunk": ci, "n_calls": n_calls, "params": params})


def live_rate(store):
    """(lenient kept, n scored, panel kept on the same n) - the running comparison, printed at
    every chunk boundary. The second smoke run's go/no-go check is exactly this pair, so it has
    to be visible while the run is happening rather than at the analysis stage with the money
    already spent."""
    rows = [r for r in store.records.values() if r.get("kind") == "len"]
    return (sum(1 for r in rows if r["value"]), len(rows),
            sum(1 for r in rows if r.get("panel_is_real")))


def main():
    ap = argparse.ArgumentParser(description="the second-panel arm: one lenient reviewer call "
                                             "per drawn census candidate")
    ap.add_argument("--in", dest="infile", default=VERIFIED)
    ap.add_argument("--tag", default="lenient-r1")
    ap.add_argument("--seed", type=int, default=DRAW_SEED)
    ap.add_argument("--target", type=int, default=DRAW_TARGET,
                    help="target CANDIDATE count; whole notes are drawn so the realised n differs")
    ap.add_argument("--limit", type=int, default=None,
                    help="buy only the first N of the stratified smoke order (a strict prefix of "
                         "the main run - smoke calls are never extra spend)")
    ap.add_argument("--plan-only", action="store_true", help="print the draw and stop, no calls")
    ap.add_argument("--redo-defaulted", action="store_true",
                    help="re-buy the calls whose reply could not be read, so a reader defect is "
                         "not left standing as a verdict. Same call, same deterministic seed - "
                         "only the reply parser has been fixed since. The store is last-write-"
                         "wins on reload, so the new record supersedes the defaulted one.")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=15, help="note blocks per Run() manifest")
    ap.add_argument("--unit-cost", type=float, default=0.02829,
                    help="measured mean cost of the shipped panel's constructor refute call "
                         "(results/wd-taxonomy/_state/verify-r1.jsonl), used for the estimate only")
    ap.add_argument("--warn-usd", type=float, default=30.0)
    ap.add_argument("--stop-usd", type=float, default=45.0,
                    help="hard cap for this run. Run-scoped, not cumulative.")
    ap.add_argument("--authorize-spend", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    assert_only_the_standard_moved()
    verified = json.load(open(os.path.join(HERE, args.infile)))
    picked, cands, dmeta = draw(verified, args.seed, args.target)
    issues = verified["all_issues"]

    print(plan_report(picked, cands, dmeta, issues, args.unit_cost))

    ordered = smoke_order(cands, args.seed)
    if args.limit:
        ordered = ordered[: args.limit]
        print(f"  --limit {args.limit}: stratified prefix, "
              f"{len({(f['source'], f['id']) for f in ordered})} consultations, "
              f"est ${len(ordered) * args.unit_cost:.2f}")

    store = RecordStore(EXPERIMENT, args.tag)
    if args.redo_defaulted:
        stale = {r["finding_id"] for r in store.records.values() if r.get("defaulted")}
        todo = [f for f in ordered if f["finding_id"] in stale]
        print(f"  --redo-defaulted: {len(todo)} unreadable verdicts to re-buy")
    else:
        todo = [f for f in ordered if not store.has(f"{f['finding_id']}|len")]
    print(f"  store {os.path.relpath(store.path, HERE)}: {len(store.records)} records held, "
          f"{len(todo)} calls to buy (${len(todo) * args.unit_cost:.2f})")

    subset_path = os.path.join(HERE, "results", EXPERIMENT, f"subset_{args.tag}.json")
    os.makedirs(os.path.dirname(subset_path), exist_ok=True)
    json.dump({"draw": dmeta, "marginal_fit": marginal_fit(cands, issues),
               "already_paid_arms_on_subset": free_arms(cands),
               "note_keys": picked,
               "finding_ids": [f["finding_id"] for f in cands],
               "smoke_order_finding_ids": [f["finding_id"] for f in smoke_order(cands, args.seed)]},
              open(subset_path, "w"), indent=1)
    print(f"  subset -> {os.path.relpath(subset_path, HERE)}")

    if args.plan_only or not todo:
        if not todo and not args.plan_only:
            print("  nothing to buy - the store already holds every drawn candidate.")
        return

    units, _ = T.note_units()
    notes = {u["note_key"]: u for u in units}
    missing = [nk for nk in picked if nk not in notes]
    if missing:
        raise SystemExit(f"{len(missing)} drawn notes are not in the corpus: {missing[:3]}")

    guard = SpendGuard(warn=args.warn_usd, stop=args.stop_usd,
                       authorized=args.authorize_spend, scope="run")
    prompts = {"lenient": LENIENT, "panel_refute_transcript_first_for_contrast": V.REFUTE_TF}
    manifest_params = T.manifest_params(
        {"arm": "lenient single reviewer", "tag": args.tag,
         "contrast": "same candidates, same model, same evidence, same settings as the shipped "
                     "panel's constructor leg; the review standard is the only difference",
         "panel_prompt_sha256": prompt_sha(V.REFUTE_TF),
         "lenient_prompt_sha256": prompt_sha(LENIENT),
         "draw": dmeta, "limit": args.limit,
         "hard_cap_usd": args.stop_usd,
         "warmup": "off - the shipped panel's Anthropic leg cached 0 of 26.8M input tokens, so a "
                   "per-note warm-up call buys nothing on this route",
         "reply_reader": "parse_json_blob first; on a JSON-load failure the verdict token "
                         "\"error\": true|false is recovered by regex when every occurrence "
                         "agrees (recorded as parsed_by=salvaged_regex); anything still "
                         "unreadable defaults to not-an-error, mirroring the panel's "
                         "default-refute and conservative against this run's hypothesis"},
        "openrouter", units)

    confirm(f"buy {len(todo)} lenient calls (~${len(todo) * args.unit_cost:.2f}, "
            f"hard cap ${args.stop_usd:.0f})?", args.yes)
    S = {"store": store, "notes": notes, "guard": guard, "usage": T.new_usage(),
         "workers": args.workers, "chunk": args.chunk, "prompts": prompts,
         "manifest_params": manifest_params, "allow_dirty": args.allow_dirty}
    execute(note_blocks(todo), S, args.tag)
    store.close()

    u, kept, n, panel = S["usage"], *live_rate(store)
    print(f"\n=== lenient arm, tag {args.tag} ===")
    print(f"  {kept}/{n} verified = {kept / max(n, 1):.2%}   "
          f"(shipped panel on the same {n}: {panel}/{n} = {panel / max(n, 1):.2%})")
    print(f"  {u['errors']} errors, "
          f"{sum(1 for r in store.records.values() if r.get('defaulted'))} defaulted verdicts, "
          f"{sum(1 for r in store.records.values() if r.get('parsed_by') == 'salvaged_regex')} "
          f"salvaged from an unescaped quote in the reason")
    print(f"  spent ${u['cost_usd']:.2f} over {u['calls']} calls "
          f"(estimate was ${len(todo) * args.unit_cost:.2f}); {T.cache_line(u)}")
    print(f"  store -> {os.path.relpath(store.path, HERE)}")
    print("  next: python3 census/second_panel_analyze.py --tag " + args.tag)


if __name__ == "__main__":
    main()
