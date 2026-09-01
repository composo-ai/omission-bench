"""The reproducibility regime's own validation run: pinned models, per-call seeds, one
manifest, and the gates a paper-bound run has to pass.

Exercises the whole chain before the judge benchmark burns credits: models.lock.json ->
llm() (OpenRouter) -> Run() -> manifest -> cost ledger -> check_manifests. REAL calls on
judge-primary, tiny scale:

Neither input it samples ships with this release (hard_negatives_balanced.json and
authored_scenarios.json are pre-audit construction files), so this module is the record
of what the regime validated rather than runnable from a clone; point --pairs at any
file of the released pair shape to exercise the same chain.

    2 pairs (first add + first omit from hard_negatives_balanced.json - pre-audit data,
    permitted for smoke only) x 2 cells x 3 replicates (seeds 11/23/37), each replicate a
    separate Run("w1-smoke", ...):
      - det-FCbin-k1-t0   deterministic regime: FC-bin prompt, k=1, temp 0 requested, seeded
      - sto-FCscore-k8-t1 stochastic regime: FC-score prompt, k=8 as actually routed, temp 1,
                          winsorize8 aggregate

Also records the pre-registered routing findings (the data the k=8 routing decision is
made from), in a 7th manifested run (stage=routing-probes):
      (a) does OpenRouter pass n=8 through for openai/gpt-5.4 in one request? usage/cost of
          the n=8 attempt vs 8 independent calls (incl. whether the provider prompt-cache
          discount lands on repeats - route-2 economics);
      (b) are reasoning_effort ("none") and seed accepted/honored via OpenRouter?
Findings: results/w1-smoke/<probe_run_id>/routing_findings.json.

Afterwards asserts: 7 new run dirs, every manifest validates, ledger grew 7 lines, and the
k=1 replicates agree on all items (or prints the disagreement for the
deterministic-agreement metric).

Usage: python3 record/w1_smoke.py [-y]     (-y skips the spend confirmation)
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import argparse, json, os, re, sys

from common import HERE, RESULTS, Run, llm, winsorize8
import check_manifests

SPEC = "the reproducibility regime's written spec"
INPUTS = ["hard_negatives_balanced.json", "authored_scenarios.json"]
SEEDS = [11, 23, 37]  # the smoke-only replicate seed triple

# The ablation-grid prompts, verbatim (inlined for the smoke only - the judge benchmark's
# own runners load them from w2_prompts/ at the pre-registration commit).
FC_BIN = """CONSULTATION TRANSCRIPT:
{transcript}

CLINICAL NOTE (AI-generated from the transcript above):
{note}

---

You are evaluating the clinical note above against the consultation transcript it was generated from.

Evaluation criterion:
Reward clinical notes that a careful clinician would accept as faithful AND complete: no fabricated, inverted/negated, hardened, wrong-value, wrong-subject, or wrong-laterality findings, and no omitted clinically-significant finding, diagnosis, medication, safety-net, or red flag. Penalise any difference from the consultation that would change the clinician's understanding or the patient's care.

Assess the note strictly against this criterion. Give a brief assessment (2-6 sentences), then end your response with exactly one line:
Verdict: PASS
or
Verdict: FAIL
PASS = the note satisfies the criterion. FAIL = the note violates the criterion in at least one way."""

FC_SCORE = """CONSULTATION TRANSCRIPT:
{transcript}

CLINICAL NOTE (AI-generated from the transcript above):
{note}

---

You are evaluating the clinical note above against the consultation transcript it was generated from.

Evaluation criterion:
Reward clinical notes that a careful clinician would accept as faithful AND complete: no fabricated, inverted/negated, hardened, wrong-value, wrong-subject, or wrong-laterality findings, and no omitted clinically-significant finding, diagnosis, medication, safety-net, or red flag. Penalise any difference from the consultation that would change the clinician's understanding or the patient's care.

Assess the note strictly against this criterion. Give a brief assessment (2-6 sentences), then end your response with exactly one line in the format:
Score: <integer between 0 and 10>
where 10 = the note fully satisfies the criterion and 0 = a severe violation. Reserve 9-10 for notes with no violation of the criterion."""


def parse_verdict(text):
    m = re.findall(r"Verdict:\s*(PASS|FAIL)", text or "", re.I)
    return m[-1].upper() if m else None


def parse_score(text):
    m = re.findall(r"Score:\s*(\d+)", text or "")
    return int(m[-1]) if m else None


def load_items():
    pairs = json.load(open(os.path.join(HERE, "hard_negatives_balanced.json")))
    chosen = [next(p for p in pairs if p["type"] == "add"),
              next(p for p in pairs if p["type"] == "omit")]
    tx = {s["id"]: s["transcript"] for s in json.load(open(os.path.join(HERE, "authored_scenarios.json")))}
    return [dict(p, transcript=tx[p["id"]]) for p in chosen]


def routing_probes(items):
    """The pre-registered routing measurements. One manifested run; returns the findings dict."""
    prompt8 = FC_SCORE.format(transcript=items[0]["transcript"], note=items[0]["errored"])
    # allow_dirty: the tree is shared with the corpus build, whose in-flight files are
    # disjoint from this run's; the manifest discloses every uncommitted path as
    # git_dirty_paths, which is the disclosure regime all released manifests record.
    with Run("w1-smoke", params={"stage": "routing-probes", "cell": None}, replicate=0,
             seed=SEEDS[0], inputs=INPUTS, spec=SPEC, allow_dirty=True) as run:
        run.register_prompts({"grid_FC_score": FC_SCORE, "grid_FC_bin": FC_BIN})
        f = {"model": "openai/gpt-5.4", "route": ["openai"], "probes": {}}

        # (a) n=8 passthrough + cost: llm(k=8) tries one n=8 request then tops up with
        # independent calls; requests[0] IS the n=8 attempt. One extra k=1 call makes a
        # clean set of 8 singles for the comparison.
        _, meta8 = llm(prompt8, "judge-primary", temperature=1.0, k=8, seed=SEEDS[0])
        _, meta1 = llm(prompt8, "judge-primary", temperature=1.0, k=1, seed=SEEDS[0] + 99)
        n8 = meta8["requests"][0]
        singles = meta8["requests"][1:] + meta1["requests"]
        f["probes"]["n8_passthrough"] = {
            "n_requested": n8["n_requested"], "n_returned": n8["n_returned"],
            "passthrough": n8["n_returned"] == 8, "k_impl_used": meta8["k_impl"],
            "n8_attempt": {"usage": n8["usage"], "cost_usd": n8["cost"]},
            "eight_independent_calls": {
                "count": len(singles),
                "usage_summed": {k: sum(r["usage"][k] for r in singles) for k in singles[0]["usage"]},
                "cost_usd_summed": round(sum(r["cost"] or 0 for r in singles), 8),
                "cached_tokens_per_call": [r["usage"]["cached_tokens"] for r in singles],
                "prompt_tokens_per_call": [r["usage"]["prompt_tokens"] for r in singles],
                "cost_per_call": [r["cost"] for r in singles]},
        }

        # (b1) reasoning_effort passthrough: none vs medium on a prompt that invites reasoning.
        q = "How many prime numbers are there between 100 and 140? Answer with just the count."
        eff = {}
        for effort in ("none", "medium"):
            _, m = llm(q, "judge-primary", temperature=1.0, k=1, seed=7,
                       reasoning_effort=effort, max_tokens=2000)
            eff[effort] = {"reasoning_tokens": m["usage"]["reasoning_tokens"],
                           "completion_tokens": m["usage"]["completion_tokens"],
                           "cost_usd": m["cost_usd_reported"]}
        f["probes"]["reasoning_effort_passthrough"] = dict(
            eff, accepted_without_error=True,
            honored=eff["none"]["reasoning_tokens"] == 0 < eff["medium"]["reasoning_tokens"])

        # (b2) seed + temperature: identical seeded calls at temp 0 and at temp 1.
        prompt_b = FC_BIN.format(transcript=items[0]["transcript"], note=items[0]["errored"])
        det = {}
        for temp in (0.0, 1.0):
            t1, _ = llm(prompt_b, "judge-primary", temperature=temp, k=1, seed=SEEDS[0])
            t2, _ = llm(prompt_b, "judge-primary", temperature=temp, k=1, seed=SEEDS[0])
            det[f"temp{temp:g}"] = {"identical_output": t1[0] == t2[0],
                                    "same_verdict": parse_verdict(t1[0]) == parse_verdict(t2[0])}
        f["probes"]["seed_and_temperature"] = dict(
            det, seed_accepted_without_error=True, temperature_accepted_without_error=True,
            note="OpenRouter lists no temperature support for openai/gpt-5.4 (supported_parameters, "
                 "2026-07-29); the param is accepted silently - identical_output above is the "
                 "empirical determinism check, and the k=1 replicate agreement is the real one.")

        run.save("routing_findings.json", f)
        return f, run


def smoke_cell(cell, items, replicate, seed):
    """One (cell, replicate) = one Run. Returns the per-item results dict."""
    with Run("w1-smoke", params={"cell": cell}, replicate=replicate, seed=seed,
             inputs=INPUTS, spec=SPEC, allow_dirty=True) as run:  # dirty-tree disclosure, see above
        run.register_prompts({"grid_FC_bin": FC_BIN} if cell.startswith("det")
                             else {"grid_FC_score": FC_SCORE})
        res = {"cell": cell, "replicate": replicate, "seed": seed, "items": [], "pairs": []}
        for p in items:
            per_kind = {}
            for kind in ("clean", "errored"):
                if cell == "det-FCbin-k1-t0":
                    texts, meta = llm(FC_BIN.format(transcript=p["transcript"], note=p[kind]),
                                      "judge-primary", temperature=0.0, k=1, seed=seed)
                    item = {"id": p["id"], "type": p["type"], "note": kind,
                            "verdict": parse_verdict(texts[0]),
                            "parse_failure": parse_verdict(texts[0]) is None}
                    per_kind[kind] = item["verdict"]
                else:  # sto-FCscore-k8-t1
                    texts, meta = llm(FC_SCORE.format(transcript=p["transcript"], note=p[kind]),
                                      "judge-primary", temperature=1.0, k=8, seed=seed)
                    scores = [parse_score(t) for t in texts]
                    ok = [s for s in scores if s is not None]
                    agg = winsorize8(ok) if len(ok) == 8 else None
                    item = {"id": p["id"], "type": p["type"], "note": kind,
                            "samples": scores, "winsorized_mean": agg,
                            "k_impl": meta["k_impl"], "parse_failure": agg is None}
                    per_kind[kind] = agg
                res["items"].append(item)
            if cell == "det-FCbin-k1-t0":
                disc = per_kind["errored"] == "FAIL" and per_kind["clean"] == "PASS"
            else:
                disc = (per_kind["errored"] is not None and per_kind["clean"] is not None
                        and per_kind["errored"] < per_kind["clean"])
            res["pairs"].append({"id": p["id"], "type": p["type"], "discriminates": disc})
        run.save("results.json", res)
        return res, run


def main():
    ap = argparse.ArgumentParser(description="reproducibility smoke run (real OpenRouter calls, ~$1-2 credits)")
    ap.add_argument("-y", "--yes", action="store_true", help="skip the spend confirmation")
    args = ap.parse_args()

    items = load_items()
    n_calls = 17 + 6 + 3 * 2 * 2 * (1 + 8)  # probes + effort/seed probes + 6 cell runs
    print(f"reproducibility smoke: ~{n_calls} OpenRouter calls on judge-primary, est ~$1.5-2.0 of credits")
    if not args.yes:
        if input("proceed? [y/N] ").strip().lower() != "y":
            sys.exit("aborted")

    ledger = os.path.join(RESULTS, "cost_ledger.jsonl")
    n_ledger_before = sum(1 for _ in open(ledger)) if os.path.exists(ledger) else 0
    runs = []

    findings, probe_run = routing_probes(items)
    runs.append(probe_run)
    n8 = findings["probes"]["n8_passthrough"]
    print(f"probe (a): n=8 -> {n8['n_returned']} choice(s) returned; "
          f"k=8 routes as {n8['k_impl_used']}")

    det_results = []
    for cell in ("det-FCbin-k1-t0", "sto-FCscore-k8-t1"):
        for rep, seed in enumerate(SEEDS, 1):
            res, run = smoke_cell(cell, items, rep, seed)
            runs.append(run)
            if cell.startswith("det"):
                det_results.append(res)
            print(f"  {cell} r{rep} (seed {seed}): "
                  + ", ".join(f"{p['id']}/{p['type']}:{'disc' if p['discriminates'] else 'MISS'}"
                              for p in res["pairs"]))

    # ---- assertions
    assert len(runs) == 7 and all(os.path.isdir(r.dir) for r in runs), "expected 7 run dirs"
    problems, warnings = [], []
    for r in runs:
        check_manifests.validate_manifest(os.path.join(r.dir, "manifest.json"), problems, warnings)
    for w in warnings:
        print(f"  WARN {w}")
    assert not problems, "manifest validation failed:\n" + "\n".join(problems)
    n_ledger_after = sum(1 for _ in open(ledger))
    assert n_ledger_after - n_ledger_before == 7, \
        f"ledger grew {n_ledger_after - n_ledger_before} lines, expected 7"

    # k=1 replicate agreement (the deterministic-agreement reading, at smoke scale)
    verdicts = {}
    for res in det_results:
        for it in res["items"]:  # key needs type: the 2 smoke pairs share a consultation id
            verdicts.setdefault((it["id"], it["type"], it["note"]), []).append(it["verdict"])
    agree = {k: len(set(v)) == 1 for k, v in verdicts.items()}
    rate = sum(agree.values()) / len(agree)
    print(f"k=1 agreement across 3 replicates (seeds {SEEDS}, temp 0 requested): "
          f"{sum(agree.values())}/{len(agree)} items ({rate:.0%})")
    for k, ok in agree.items():
        if not ok:
            print(f"  disagreement on {k}: {verdicts[k]}")

    spent = sum(json.loads(l).get("cost_usd") or 0
                for l in open(ledger).readlines()[n_ledger_before:])
    print(f"smoke complete: 7 runs, manifests valid, ledger +7 lines, "
          f"~${spent:.2f} credits spent")
    print(f"routing findings: {os.path.join(probe_run.dir, 'routing_findings.json')}")


if __name__ == "__main__":
    main()
