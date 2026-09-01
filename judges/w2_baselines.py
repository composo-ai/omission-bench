"""w2_baselines.py - the external baselines: G-Eval, RAGAS and an instance checklist.

  geval      G-Eval-style CoT scoring, 2 dimensions (consistency, completeness),
             1-5 integers, note score = min of the two (both reported)
  ragas      RAGAS-style decomposition: faithfulness = fraction of note claims
             supported, completeness = fraction of transcript facts covered,
             note score = min (both reported). Transcript-fact extraction runs
             ONCE per consultation at temp 0 and is cached, so extraction noise
             cancels within a pair.
  checklist  arXiv 2507.17717-style: an instance-specific checklist generated
             once per consultation (temp 0, seed 11, cached), answered per note;
             score = fraction of items answered yes.

Cached generation lands in w2_cache/{ragas_facts,checklists}.json and is released
with the paper. Everything else matches the grid runner: frozen prompts loaded and
sha-verified from w2_prompts/, consultation-major blocks with a warm-up call,
resumable record store, per-call receipts, spend tripwires.

Flag rule for the false-alarm column: thresholds were pre-registered for the ablation
grid's cells only, so these arms use the natural fixed analogue - flag iff the note
scores below full marks (min(c,comp) < 5 for G-Eval, < 1.0 for the two fractional
arms) - and w2_analyze.py additionally reports each arm threshold-free (AUC) and at
its Youden operating point. The fixed rule is a runner convention, labelled as such.

RAGAS COVERAGE: KEYED VERDICTS (replaces the lenient-parse hack, 2026-08-12).
The old positional contract asked for a bare boolean array, one entry per reference
fact, and the model mis-counted it on every call (e.g. 34 booleans for 36 facts).
Strict length checking made the arm 100% parse_failure; the lenient patch computed
the fraction from whatever array came back, which silently mis-ALIGNED verdicts to
facts - a wrong number that looks fine. Both are gone. Facts now carry IDs (f1..fN)
and the judge returns a verdict per ID, so a mis-count is a set difference rather
than an alignment error:

  1. missing IDs trigger ONE targeted retry asking only for those IDs;
  2. IDs still missing after it are recorded UNKNOWN (`unknown_ids`), excluded from
     the coverage denominator, and reported per arm by w2_analyze.py;
  3. `completeness_unknown_as_missing` is recorded alongside as the conservative
     sensitivity variant, so the choice of denominator is visible, not baked in;
  4. only a reply with no usable keyed verdict at all is a parse_failure.

The checklist arm keeps the positional contract - its items are answered as a short
`{"answers": [...]}` list whose length matched on every smoke call - but its
`length_mismatch` is still recorded per record.

    python3 judges/w2_baselines.py --smoke --allow-unfrozen --arms geval -y
    python3 judges/w2_baselines.py --runs 3 -y
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import argparse, json, os, re, threading, time

from common import HERE, Run
import w2_common as W

ARMS = ["geval", "ragas", "checklist"]
GEN_SEED = 11  # cached generation steps: temp 0, seed 11, shared across runs
COVERAGE_RETRY_SEED_OFFSET = 5  # targeted missing-ID retry: a distinct sample, same item decade

# Token budgets. The grid's 1024 fits a short assessment + verdict; these arms emit one
# JSON entry per claim / per fact / per checklist item, so they need room proportional to
# the list. Measured 2026-08-12: at 1024 the faithfulness step truncated mid-JSON on every
# note over ~40 claims - 5 of 9 smoke records died as parse failures for that reason alone.
FAITHFULNESS_MAX_TOKENS = 4096
CHECKLIST_MAX_TOKENS = 2048


def coverage_budget(n_facts):
    """~8 output tokens per keyed verdict, plus slack; capped so nothing runs away."""
    return int(min(4096, max(1024, 512 + 14 * n_facts)))


def parse_1to5(text):
    m = re.findall(r"(\d+)", text or "")
    for v in reversed(m):
        if 1 <= int(v) <= 5:
            return int(v)
    return None


# ---------------------------------------------------------------- keyed coverage
def fact_ids(facts):
    return [f"f{i + 1}" for i in range(len(facts))]


def facts_block(ids, facts):
    """The numbered fact list the keyed coverage prompt reads."""
    return "\n".join(f"{i}: {str(t).strip()}" for i, t in zip(ids, facts))


def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "y", "covered", "1"):
            return True
        if s in ("false", "no", "n", "missing", "not covered", "absent", "0"):
            return False
    return None


def parse_keyed_verdicts(text, want_ids):
    """{"verdicts": {"f1": true, ...}} -> {id: bool} for the IDs we asked about.

    Tolerates a bare keyed object (no `verdicts` wrapper) and yes/no strings, because
    those carry the same information keyed the same way. Returns None only when there
    is no usable keyed verdict at all - that is the one case the judge() retry exists
    for. Extra or unrecognised keys are dropped by the caller, which records them.
    """
    j = W.parse_json_blob(text)
    if not isinstance(j, dict):
        return None
    v = j.get("verdicts")
    if not isinstance(v, dict):
        v = j.get("covered") if isinstance(j.get("covered"), dict) else None
    if not isinstance(v, dict):
        v = j  # bare keyed object
    want = {str(i).lower() for i in want_ids}
    out = {}
    for k, val in v.items():
        key = str(k).strip().lower()
        b = _to_bool(val)
        if key in want and b is not None:
            out[key] = b
    return out or None


# ---------------------------------------------------------------- the arms
class ArmCtx:
    """What an arm callable is given: the prompts, the note under test, its transcript,
    a `call` that makes one judge call at this item's seed, and `gen_once` for the
    cached per-consultation artifacts. Arms return (aggregate, flagged, detail) and
    know nothing about stores, runs or receipts - that is the runner's job.
    """

    def __init__(self, prompts, transcript, note_text, call, gen_once):
        self.prompts, self.transcript, self.note = prompts, transcript, note_text
        self.call, self.gen_once = call, gen_once


def arm_geval(c):
    """G-Eval-style CoT scoring on two dimensions; note score = min of the two."""
    cons = c.call("geval_consistency",
                  W.render(c.prompts["geval_consistency"], transcript=c.transcript, note=c.note),
                  parse_1to5)
    comp = c.call("geval_completeness",
                  W.render(c.prompts["geval_completeness"], transcript=c.transcript, note=c.note),
                  parse_1to5)
    detail = {"consistency": cons, "completeness": comp, "scale": [1, 5]}
    agg = None if (cons is None or comp is None) else float(min(cons, comp))
    return agg, (None if agg is None else agg < 5), detail


GEN_FAILURES = []


def arm_ragas(c):
    """RAGAS-style decomposition. Faithfulness = fraction of note claims supported;
    completeness = fraction of reference facts covered, from KEYED verdicts."""
    facts = c.gen_once("facts")
    if facts is None:
        return None, None, {"gen_failure": "ragas_extract"}

    def pf(t):
        j = W.parse_json_blob(t)
        cl = (j or {}).get("claims")
        return cl if isinstance(cl, list) and cl else None

    claims = c.call("ragas_faithfulness",
                    W.render(c.prompts["ragas_faithfulness"], transcript=c.transcript,
                             note=c.note), pf, max_tokens=FAITHFULNESS_MAX_TOKENS)
    faith = (sum(1 for x in claims if x.get("supported")) / len(claims)) if claims else None

    ids = fact_ids(facts)

    def coverage_call(subset_ids, seed_offset):
        subset = [facts[ids.index(i)] for i in subset_ids]
        prompt = W.render(c.prompts["ragas_coverage"], note=c.note,
                          facts_block=facts_block(subset_ids, subset))
        return c.call("ragas_coverage", prompt,
                      lambda t: parse_keyed_verdicts(t, subset_ids), seed_offset=seed_offset,
                      max_tokens=coverage_budget(len(subset_ids)))

    verdicts = coverage_call(ids, 0) or {}
    missing = [i for i in ids if i not in verdicts]
    retried = bool(missing)
    if missing:  # ONE targeted retry, asking only about the IDs that came back missing
        extra = coverage_call(missing, COVERAGE_RETRY_SEED_OFFSET) or {}
        verdicts.update({k: v for k, v in extra.items() if k in set(missing)})
    unknown = [i for i in ids if i not in verdicts]
    resolved = {i: verdicts[i] for i in ids if i in verdicts}
    comp = (sum(1 for v in resolved.values() if v) / len(resolved)) if resolved else None
    comp_strict = (sum(1 for v in resolved.values() if v) / len(ids)) if ids else None
    detail = {"faithfulness": faith, "completeness": comp,
              "completeness_unknown_as_missing": comp_strict,
              "n_claims": len(claims) if claims else None,
              "n_facts": len(ids), "n_verdicts": len(resolved),
              "n_unknown": len(unknown), "unknown_ids": unknown,
              "missing_before_retry": len(ids) - len(verdicts) + len(unknown) if retried else 0,
              "retried_missing_ids": retried,
              "keyed_verdicts": True, "scale": [0, 1]}
    agg = None if (faith is None or comp is None) else min(faith, comp)
    return agg, (None if agg is None else agg < 1.0), detail


def arm_checklist(c):
    """arXiv 2507.17717-style instance checklist, generated once per consultation and
    answered per note; score = fraction of items answered yes."""
    items = c.gen_once("checklist")
    if items is None:
        return None, None, {"gen_failure": "checklist_gen"}

    def pa(t):
        j = W.parse_json_blob(t)
        a = (j or {}).get("answers")
        return a if isinstance(a, list) and a else None

    ans = c.call("checklist_answer",
                 W.render(c.prompts["checklist_answer"], transcript=c.transcript, note=c.note,
                          checklist_json=json.dumps(items, ensure_ascii=False)), pa,
                 max_tokens=CHECKLIST_MAX_TOKENS)
    yes = (sum(1 for a in ans if str(a).strip().lower().startswith("y")) / len(ans)) if ans else None
    detail = {"n_items": len(items), "yes_fraction": yes,
              "n_answers_returned": len(ans) if ans else None,
              "length_mismatch": ans is not None and len(ans) != len(items), "scale": [0, 1]}
    return yes, (None if yes is None else yes < 1.0), detail


ARM_FNS = {"geval": arm_geval, "ragas": arm_ragas, "checklist": arm_checklist}
ARM_PROMPTS = {"geval": ["geval_consistency", "geval_completeness"],
               "ragas": ["ragas_extract", "ragas_faithfulness", "ragas_coverage"],
               "checklist": ["checklist_gen", "checklist_answer"]}
ARM_GEN = {"ragas": "facts", "checklist": "checklist"}  # cached per-consultation artifact


class GenCache:
    """w2_cache/*.json - the once-per-consultation artifacts, shared across runs."""

    def __init__(self, name):
        os.makedirs(W.CACHE_DIR, exist_ok=True)
        self.path = os.path.join(W.CACHE_DIR, name)
        self.data = json.load(open(self.path)) if os.path.exists(self.path) else {}
        self._lock = threading.Lock()

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        with self._lock:
            self.data[key] = value
            json.dump(self.data, open(self.path, "w"), indent=1, sort_keys=True)


def main():
    ap = argparse.ArgumentParser(description="external baselines for the judge benchmark")
    ap.add_argument("--model", default="gpt54", choices=sorted(W.ROLES))
    ap.add_argument("--arms", default="all", help="'all' or a comma list of " + ",".join(ARMS))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--run", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dataset", default=None,
                    help=f"dataset file (default: first of {', '.join(W.DATASET_CANDIDATES)})")
    ap.add_argument("--allow-unfrozen", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--warn-usd", type=float, default=700.0)
    ap.add_argument("--stop-usd", type=float, default=900.0)
    ap.add_argument("--authorize-spend", action="store_true")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    if args.allow_unfrozen and not (args.smoke or args.limit):
        raise SystemExit("--allow-unfrozen is a smoke-only gate: pass --smoke or --limit too")

    arms = ARMS if args.arms == "all" else [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in ARM_FNS]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; known: {sorted(ARM_FNS)}")
    prompts = W.load_prompts(sorted({p for a in arms for p in ARM_PROMPTS[a]}))

    pairs, info = W.load_dataset(path=args.dataset, allow_unfrozen=args.allow_unfrozen)
    if args.smoke:
        from w2_grid import pick_smoke_pairs
        pairs = pick_smoke_pairs(pairs, n=2)
    elif args.limit:
        pairs = sorted(pairs, key=lambda p: p["pair_id"])[:args.limit]
    info["n_pairs_selected"] = len(pairs)
    transcripts, tprov = W.transcript_index()
    blocks, notes = W.note_units(pairs, transcripts)

    role_cfg = W.ROLES[args.model]
    replicates = [args.run] if args.run else list(range(1, args.runs + 1))
    tag = args.tag or (f"baselines-{args.model}-{info['sha256'][:8]}"
                       + ("-smoke" if (args.smoke or args.limit) else ""))
    store = W.RecordStore("w2-baselines", tag)
    facts_cache, check_cache = GenCache("ragas_facts.json"), GenCache("checklists.json")

    per_note = {"geval": 2, "ragas": 2, "checklist": 1}
    todo = [1 for rep in replicates for b in blocks for a in arms for n in b["notes"]
            if not store.has(f"{a}|r{rep}|{n['note_key']}")]
    print(f"w2 baselines | arms={arms} x {len(replicates)} replicates x {len(notes)} notes; "
          f"{len(todo)} judgements to run "
          f"(~{sum(per_note[a] for a in arms) * len(replicates) * len(notes)} calls + cached gen)")
    print(f"  dataset: {info['pairs_file']} version={info['dataset_version']} "
          f"sha {info['sha256'][:12]}")
    if not info["frozen"]:
        print("  ! PRE-AUDIT SUBSTRATE - smoke only")
    if not todo:
        print("nothing to do")
        return
    W.confirm("proceed?", args.yes)
    guard = W.SpendGuard(args.warn_usd, args.stop_usd, args.authorize_spend)

    t0 = time.time()
    for rep in replicates:
        seed = W.RUN_SEEDS.get(rep, 11 * rep)
        pending = [b for b in blocks if any(not store.has(f"{a}|r{rep}|{n['note_key']}")
                                            for a in arms for n in b["notes"])]
        for ci, chunk in enumerate(W.chunked(pending, args.chunk), 1):
            params = W.run_params(
                {"arms": arms, "model_key": args.model, "role": role_cfg["role"],
                 "reasoning_effort": role_cfg["reasoning_effort"], "chunk": ci,
                 "gen_seed": GEN_SEED, "gen_temperature": 0.0,
                 "flag_rule": "fixed: below full marks (a runner convention, not a pre-registered rule)",
                 "ragas_coverage_contract": "keyed verdicts per fact ID; one targeted retry for "
                                            "missing IDs, then unknown (excluded from the "
                                            "denominator, recorded per record)",
                 "max_tokens": {"default": W.MAX_TOKENS,
                                "ragas_faithfulness": FAITHFULNESS_MAX_TOKENS,
                                "ragas_coverage": "512 + 14/fact, capped 4096",
                                "checklist_answer": CHECKLIST_MAX_TOKENS},
                 "record_store": os.path.relpath(store.path, HERE)}, info)
            W.assert_dataset_stable(info)
            with Run("w2-baselines", params=params, replicate=rep, seed=seed,
                     inputs=[info["pairs_file"]], spec=W.SPEC, allow_dirty=True) as run:
                run.register_prompts(prompts)
                chunk_records = []

                def gen_once(kind, block):
                    """Cached, once-per-consultation generation at temp 0."""
                    key = f"{block['stratum']}|{block['consultation']}"
                    cache, pname, field = ((facts_cache, "ragas_extract", "facts")
                                           if kind == "facts"
                                           else (check_cache, "checklist_gen", "checklist"))
                    hit = cache.get(key)
                    if hit is not None:
                        return hit
                    prompt = W.render(prompts[pname], transcript=block["transcript"])
                    vals, _, metas, _ = W.judge(
                        prompt, role_cfg["role"], 1, GEN_SEED, temperature=0.0,
                        reasoning_effort=role_cfg["reasoning_effort"],
                        parser=lambda t: (lambda j: j.get(field) if isinstance(j, dict)
                                          and isinstance(j.get(field), list) else None)(
                                              W.parse_json_blob(t)))
                    guard.add(W.receipt_totals(W.receipts(metas))["cost_usd"])
                    if vals[0] is None:
                        # A baseline that cannot parse its own once-per-consultation
                        # generation is a property of that baseline, not a reason to
                        # destroy a multi-thousand-judgement run (which is what raising
                        # here did on 2026-08-13, at 882/3642 records). Record it, mark
                        # every note of this consultation a parse_failure for this arm,
                        # and carry on - the failure count is itself reportable.
                        GEN_FAILURES.append({"arm": pname, "key": key})
                        print(f"  GEN FAILURE {pname} {key} - notes marked parse_failure, continuing")
                        cache.put(key, None)
                        return None
                    cache.put(key, vals[0])
                    return vals[0]

                def worker(job):
                    arm, note, block = job
                    base_seed = seed * 100000 + note["item_index"] * 10
                    metas = []

                    def call(pname, prompt, parser, seed_offset=0, max_tokens=None):
                        vals, _, ms, _ = W.judge(prompt, role_cfg["role"], 1,
                                                 base_seed + seed_offset, temperature=1.0,
                                                 reasoning_effort=role_cfg["reasoning_effort"],
                                                 parser=parser, max_tokens=max_tokens)
                        metas.extend(ms)
                        return vals[0]

                    ctx = ArmCtx(prompts, block["transcript"], note["text"], call,
                                 lambda kind, b=block: gen_once(kind, b))
                    agg, flagged, detail = ARM_FNS[arm](ctx)

                    rcs = W.receipts(metas)
                    tot = W.receipt_totals(rcs)
                    rec = {"key": f"{arm}|r{rep}|{note['note_key']}", "cell": arm, "arm": arm,
                           "format": "score", "k": 1, "temperature": 1.0, "replicate": rep,
                           "run_seed": seed, "base_seed": base_seed, "model_key": args.model,
                           "role": role_cfg["role"],
                           # metas is EMPTY when the arm short-circuits on a generation
                           # failure (no judge call was made) - that path is exactly the
                           # one the 2026-08-13 fix introduced, so it must not index.
                           "model": metas[0]["model"] if metas else None,
                           **W.note_fields(note),
                           "samples": [agg], "aggregate": agg, "detail": detail,
                           "length_mismatch": bool(detail.get("length_mismatch")),
                           "n_unknown_verdicts": detail.get("n_unknown"),
                           "verdict": None if flagged is None else ("FAIL" if flagged else "PASS"),
                           "flagged": flagged, "parse_failure": agg is None,
                           "receipts": rcs, "totals": tot, "run_id": run.run_id,
                           "t": round(time.time(), 3), "substrate_frozen": info["frozen"],
                           "dataset_version": info["dataset_version"]}
                    store.put(rec)
                    chunk_records.append(rec)
                    guard.add(tot["cost_usd"])

                for b in chunk:
                    for kind, arm in (("facts", "ragas"), ("checklist", "checklist")):
                        if arm in arms:
                            gen_once(kind, b)  # serial: also warms the transcript prefix
                    jobs = [(a, n, b) for n in b["notes"] for a in arms
                            if not store.has(f"{a}|r{rep}|{n['note_key']}")]
                    jobs.sort(key=lambda j: (j[1]["note_role"] != "clean", j[0]))
                    W.run_block(jobs, worker, args.workers, warmup="ragas" not in arms
                                and "checklist" not in arms)
                    print(f"  r{rep} chunk{ci} {b['stratum']}/{b['consultation']}: {len(jobs)} "
                          f"judgements | credits ${guard.total:.2f} | {time.time() - t0:.0f}s",
                          flush=True)
                run.save("w2_baselines.json", {"tag": tag, "arms": arms, "replicate": rep,
                                               "seed": seed, "params": params,
                                               "transcript_sources": tprov,
                                               "n_records": len(chunk_records),
                                               "records": chunk_records})
    store.close()
    print(f"\ndone: {len(store.records)} records; credits ${guard.spent_here:.2f} this session")
    if GEN_FAILURES:
        print(f"  generation failures: {len(GEN_FAILURES)} consultation-arm pairs unparseable "
              f"(notes recorded as parse_failure): {GEN_FAILURES}")
    print(f"cached generation: {facts_cache.path}, {check_cache.path}")


if __name__ == "__main__":
    main()
