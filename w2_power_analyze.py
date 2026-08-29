"""w2_power_analyze.py - read-out for the judge-power test. No model calls.

Each powered arm has exactly one effort-none control that differs from it ONLY in
reasoning effort + token cap (prompt, model, temperature, seed, substrate, flag rule all
held). The table this produces is therefore a dose-response: what does test-time compute
buy a monolithic judge, and does it close the gap to the pipeline?

  power-FC-med / power-FC-high    vs FC-score-k1 @ none      (grid-main2, 3 reps)
  power-factlist-high             vs factlist arm @ none     (w2-factlist, 1 rep)
  power-gepa-high                 vs gepa-optimized @ none   (w2-strong, subset-restricted)
  ceiling references              B2 0.801 / B3 0.752 / per-fact rule 20.6% @ 2.1%

Every comparator is recomputed from its own store and checked against FINDINGS before it
is quoted; the script stops if a check fails. Also emits per-arm compute accounting
(prompt/completion/reasoning tokens and receipts cost per note) - the compute-matched
table a colleague's point 4/5 asks for.

    .venv/bin/python w2_power_analyze.py
"""
import argparse, importlib.util, json, os, sys

from common import HERE, RESULTS
import w2_common as W

EXPERIMENT = "w2-power"
STORE = "results/w2-power/_state/power.jsonl"
ENG_STORE = "results/w2-power/_state/power-eng.jsonl"
GRID = "results/w2-ablation/_state/grid-main2.jsonl"
FACTLIST_STORE = "results/w2-factlist/_state/factlist.jsonl"
STRONG_STORE = "results/w2-strong/_state/arms-main.jsonl"
BASELINES_STORE = "results/w2-baselines/_state/arms-main.jsonl"
B2_STORE = "results/w2-pipeline/_state/confirm-B2.jsonl"
B3_STORE = "results/w2-pipeline/_state/confirm-B3.jsonl"
SUBSET = "master/arms_confirm_subset.json"

PUBLISHED = {"B2_paired": 0.801, "B3_paired": 0.752, "rule_det": 0.206, "rule_fa": 0.0213,
             "k8_full_corpus": 0.634, "factlist_paired": 0.634, "gepa_full_corpus": 0.549,
             "eng_subset_control": 0.635}
K8_SUBSET_RANGE = (0.531, 0.649)

# power-engineered-high lives in its own store (ENG_STORE); the 23 Aug P1 review found
# this read-out had never ingested it, which is how "0.975-1.000 on every powered arm"
# shipped while the engineered arm sat at 0.725 (FINDINGS 30.1's dated correction).
# Ingested since 2026-08-24 so that correction reproduces from this one artifact.
POWER_ARMS = ["power-FC-med", "power-FC-high", "power-factlist-high", "power-gepa-high",
              "power-engineered-high"]


def load_extract_data():
    path = os.path.join(HERE, "figures", "extract_data.py")
    if not os.path.exists(path):
        raise SystemExit(f"cannot find {path} - the shared measure conventions live there")
    spec = importlib.util.spec_from_file_location("_pw_extract", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_pw_extract"] = mod
    spec.loader.exec_module(mod)
    return mod


def compute_block(recs):
    """Mean tokens + receipts cost per note-judgement, from stored totals."""
    tot = [r.get("totals") or {} for r in recs]
    have = [t for t in tot if t.get("calls")]
    if not have:
        return None
    n = len(have)
    return {"records_with_receipts": n,
            "prompt_tokens": round(sum(t.get("prompt_tokens") or 0 for t in have) / n),
            "completion_tokens": round(sum(t.get("completion_tokens") or 0 for t in have) / n),
            "reasoning_tokens": round(sum(t.get("reasoning_tokens") or 0 for t in have) / n),
            "calls_per_note": round(sum(t.get("calls") or 0 for t in have) / n, 2),
            "cost_usd_per_note": round(sum(t.get("cost_usd") or 0 for t in have) / n, 5)}


def swept(blk):
    sw = blk.get("swept_fa10")
    return (f"{100 * sw['det']:.1f}% ({sw['det_k']}/{sw['n_err']}) @ "
            f"{100 * sw['fa']:.1f}% ({sw['fa_k']}/{sw['n_clean']})") if sw else "n/a"


def main():
    ap = argparse.ArgumentParser(description="judge-power read-out")
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--out-json", default=f"results/{EXPERIMENT}/analysis.json")
    ap.add_argument("--out-md", default=f"results/{EXPERIMENT}/SUMMARY.md")
    args = ap.parse_args()

    X = load_extract_data()
    subset, sinfo = W.load_dataset(path=SUBSET)
    keep = {p["pair_id"] for p in subset}
    kc = {(p["stratum"], p["id"]) for p in subset}
    on_subset = lambda r: ((r.get("note_role") == "clean"
                            and (r.get("stratum"), r.get("consultation")) in kc)
                           or r.get("pair_id") in keep)

    problems = []

    def chk(name, got, want, tol=0.002):
        if got is None or abs(got - want) > tol:
            problems.append(f"{name}: recomputed {got!r}, published {want}")

    # ---- controls + ceilings, each verified --------------------------------------
    grid = X.load(GRID)
    cells = {}
    for cell in ("FC-score-k1", "FC-score-k8"):
        recs = [r for r in grid if r["cell"] == cell and on_subset(r)]
        blk = X.arm_block(recs)
        blk["n_replicates"] = len({r["replicate"] for r in recs})
        blk["compute"] = compute_block(recs)
        cells[cell] = blk
    chk("17.1 FC-score-k8 full corpus",
        X.arm_block([r for r in grid if r["cell"] == "FC-score-k8"])["paired_omissions"]["paired"],
        PUBLISHED["k8_full_corpus"], 0.0015)

    fl = X.load(FACTLIST_STORE)
    factlist = X.arm_block(fl)
    factlist["n_replicates"] = len({r["replicate"] for r in fl})
    factlist["compute"] = compute_block(fl)
    chk("28 factlist paired", factlist["paired_omissions"]["paired"],
        PUBLISHED["factlist_paired"], 0.0015)

    strong = X.load(STRONG_STORE)
    gepa_all = [r for r in strong if (r.get("cell") or r.get("arm")) == "gepa-optimized"]
    chk("18.2 gepa full corpus", X.arm_block(gepa_all)["paired_omissions"]["paired"],
        PUBLISHED["gepa_full_corpus"], 0.0015)
    gepa_recs = [r for r in gepa_all if on_subset(r)]
    gepa = X.arm_block(gepa_recs)
    gepa["n_replicates"] = len({r["replicate"] for r in gepa_recs})
    gepa["compute"] = compute_block(gepa_recs)

    eng_all = [r for r in strong if (r.get("cell") or r.get("arm")) == "engineered-completeness"]
    eng_recs = [r for r in eng_all if on_subset(r)]
    eng = X.arm_block(eng_recs)
    eng["n_replicates"] = len({r["replicate"] for r in eng_recs})
    eng["compute"] = compute_block(eng_recs)
    chk("30.1 engineered subset control", eng["paired_omissions"]["paired"],
        PUBLISHED["eng_subset_control"], 0.0015)

    ragas = None
    if os.path.exists(os.path.join(HERE, BASELINES_STORE)):
        base = X.load(BASELINES_STORE)
        rr = [r for r in base if (r.get("cell") or r.get("arm")) == "ragas" and on_subset(r)]
        if rr:
            ragas = X.arm_block(rr)
            ragas["n_replicates"] = len({r["replicate"] for r in rr})
            ragas["compute"] = compute_block(rr)

    # confirm-B3.jsonl holds the full 495-pair evaluation set since 2026-08-24
    # (FINDINGS 31); the 30-series ceilings are the held-out confirmation run.
    b2, b3 = X.load(B2_STORE), [r for r in X.load(B3_STORE) if on_subset(r)]
    pipe = {"B2": X.arm_block(b2, flagged=None), "B3": X.arm_block(b3, flagged=None)}
    pipe["B2"]["compute"], pipe["B3"]["compute"] = compute_block(b2), compute_block(b3)
    chk("19.1 B2 paired", pipe["B2"]["paired_omissions"]["paired"], PUBLISHED["B2_paired"], 0.0015)
    chk("19.1 B3 paired", pipe["B3"]["paired_omissions"]["paired"], PUBLISHED["B3_paired"], 0.0015)
    predicate = lambda r: any(
        m.get("severity") == "critical" and m.get("verdict") == "absent"
        for m in ((r.get("detail") or {}).get("missing_facts") or []))
    rule = X.arm_block(b3, flagged=predicate)
    chk("19.3 rule det", rule["absolute"]["det"]["p"], PUBLISHED["rule_det"])
    chk("19.3 rule FA", rule["absolute"]["fa"]["p"], PUBLISHED["rule_fa"])
    if problems:
        raise SystemExit("COMPARATOR CHECK FAILED - refusing to report against unverified "
                         "comparators:\n  " + "\n  ".join(problems))
    print("comparator check OK (k8 full-corpus 0.634; factlist 0.634; gepa full-corpus "
          "0.549; B2 0.801; B3 0.752; rule 20.6% @ 2.1%)")

    # ---- the powered arms --------------------------------------------------------
    path = os.path.join(HERE, args.store)
    if not os.path.exists(path):
        raise SystemExit(f"no store at {args.store} - the run has not produced it.")
    pw = X.load(args.store)
    if os.path.exists(os.path.join(HERE, ENG_STORE)):
        pw += X.load(ENG_STORE)
    arms = {}
    for an in POWER_ARMS:
        # Since 2026-08-24 power.jsonl also holds power-gepa-high over the FULL 495-pair
        # evaluation set (FINDINGS 31); this read-out is the 151-pair held-out
        # dose-response of FINDINGS 30, so every powered arm is restricted to the
        # confirmation subset here. The evaluation-set read-out is
        # w2_evalfull_analyze.py -> results/w2-evalfull/analysis.json.
        recs = [r for r in pw if r.get("arm") == an and on_subset(r)]
        if not recs:
            print(f"  ({an}: no records yet)")
            continue
        blk = X.arm_block(recs)
        blk["n_records"] = len(recs)
        blk["parse_failures"] = sum(1 for r in recs if r.get("parse_failure"))
        blk["effort"] = recs[0].get("reasoning_effort")
        blk["compute"] = compute_block(recs)
        arms[an] = blk

    controls = {"power-FC-med": ("FC-score-k1 @ none (3 reps)", cells["FC-score-k1"]),
                "power-FC-high": ("FC-score-k1 @ none (3 reps)", cells["FC-score-k1"]),
                "power-factlist-high": ("factlist @ none (1 rep)", factlist),
                "power-gepa-high": ("gepa-optimized @ none (subset, "
                                    f"{gepa['n_replicates']} reps)", gepa),
                "power-engineered-high": ("engineered-completeness @ none (subset, "
                                          f"{eng['n_replicates']} reps)", eng)}

    out = {"experiment": EXPERIMENT,
           "substrate": {"file": sinfo["pairs_file"], "sha256": sinfo["sha256"],
                         "n_pairs": sinfo["n_pairs"], "by_class": sinfo["by_class"],
                         "n_consultations": len(kc)},
           "powered_arms": arms,
           "controls": {"FC-score-k1_none": cells["FC-score-k1"],
                        "FC-score-k8_none": cells["FC-score-k8"],
                        "factlist_none": factlist, "gepa_none_subset": gepa,
                        "engineered_none_subset": eng,
                        "ragas_subset": ragas},
           "ceilings": {"pipeline_B2": pipe["B2"], "pipeline_B3": pipe["B3"],
                        "B3_per_fact_rule": {"det": rule["absolute"]["det"],
                                             "fa": rule["absolute"]["fa"]}}}

    dest = os.path.join(HERE, args.out_json)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=1)

    def crow(name, blk):
        c = blk.get("compute") or {}
        p = lambda d: (f"{d['paired']:.3f}" if isinstance(d, dict)
                       and d.get("paired") is not None else "n/a")
        return (f"{name:<42} {p(blk.get('paired_omissions')):>9} "
                f"{p(blk.get('paired_commissions')):>10} {swept(blk):>24} "
                f"{c.get('reasoning_tokens', 0) or 0:>7} "
                f"{c.get('cost_usd_per_note', 0) or 0:>9.4f}")

    print(f"\n{'design':<42} {'paired om':>9} {'paired co':>10} {'det @FA<=10%':>24} "
          f"{'r-tok':>7} {'$/note':>9}")
    print(crow("FC-score-k1 @ none/1024 (3 reps)", cells["FC-score-k1"]))
    for an in ("power-FC-med", "power-FC-high"):
        if an in arms:
            print(crow(f"  {an} ({arms[an]['effort']}/12k)", arms[an]))
    print(crow("factlist @ none/1024 (1 rep)", factlist))
    if "power-factlist-high" in arms:
        print(crow("  power-factlist-high (high/12k)", arms["power-factlist-high"]))
    print(crow(f"gepa-optimized @ none/1024 (subset)", gepa))
    if "power-gepa-high" in arms:
        print(crow("  power-gepa-high (high/12k)", arms["power-gepa-high"]))
    print(crow("engineered @ none/1024 (subset)", eng))
    if "power-engineered-high" in arms:
        print(crow("  power-engineered-high (high/12k)", arms["power-engineered-high"]))
    print(crow("FC-score-k8 @ none (grid best, 3 reps)", cells["FC-score-k8"]))
    if ragas:
        print(crow("RAGAS-style @ none (subset)", ragas))
    print(crow("pipeline B2", pipe["B2"]))
    print(crow("pipeline B3", pipe["B3"]))
    print(f"{'B3 + per-fact rule':<42} {'':>9} {'':>10} "
          f"{100 * rule['absolute']['det']['p']:>10.1f}% @ "
          f"{100 * rule['absolute']['fa']['p']:.1f}%")

    write_md(out, cells, factlist, gepa, eng, ragas, pipe, rule, arms,
             os.path.join(HERE, args.out_md))
    print(f"\nwritten: {os.path.relpath(dest, HERE)}, {args.out_md}")
    return out


def write_md(out, cells, factlist, gepa, eng, ragas, pipe, rule, arms, path):
    L = []
    A = L.append
    sub = out["substrate"]
    A("# The judge-power test: does test-time compute close the monolithic gap?")
    A("")
    A("Every grid cell ran at `reasoning_effort=\"none\"`, `max_tokens=1024`; the arms "
      "that beat them spend more tokens over more calls (RAGAS-style: 4096-token "
      "faithfulness + a fact-scaled coverage budget; the pipeline: one call per fact). "
      "This experiment gives the SAME monolithic prompts a reasoning budget and asks "
      "whether compute, rather than task structure, was the missing ingredient. Each "
      "powered arm differs from its effort-none control in reasoning effort + token cap "
      "only - prompt (sha-verified), model, temperature, seed, substrate and flag rule "
      "all held.")
    A("")
    A(f"**Substrate:** `{sub['file']}` (sha {sub['sha256'][:12]}) - §19.1's confirmation "
      f"set, {sub['n_pairs']} held-out pairs over {sub['n_consultations']} consultations "
      "plus 47 clean twins. **Judge:** `openai/gpt-5.4` (role judge-primary), temp 1.0, "
      "k=1, seed 11, one replicate per powered arm.")
    A("")
    A("| design | effort/cap | paired omissions | paired commissions | swept det @ FA<=10% "
      "| reasoning tok/note | $/note |")
    A("|---|---|---|---|---|---|---|")

    def row(name, blk, effort, bold=False):
        c = blk.get("compute") or {}
        p = lambda d: (f"{d['paired']:.3f}" if isinstance(d, dict)
                       and d.get("paired") is not None else "n/a")
        b = "**" if bold else ""
        A(f"| {b}{name}{b} | {effort} | {b}{p(blk.get('paired_omissions'))}{b} | "
          f"{p(blk.get('paired_commissions'))} | {swept(blk)} | "
          f"{c.get('reasoning_tokens', 'n/a')} | {c.get('cost_usd_per_note', 'n/a')} |")

    row("FC-score-k1 (3 reps)", cells["FC-score-k1"], "none/1024")
    for an, label in (("power-FC-med", "FC-score-k1 POWERED"),
                      ("power-FC-high", "FC-score-k1 POWERED")):
        if an in arms:
            row(f"{label}", arms[an], f"{arms[an]['effort']}/12k", bold=True)
    row("factlist (1 rep)", factlist, "none/1024")
    if "power-factlist-high" in arms:
        row("factlist POWERED", arms["power-factlist-high"], "high/12k", bold=True)
    row(f"gepa-optimized (subset, {gepa['n_replicates']} reps)", gepa, "none/1024")
    if "power-gepa-high" in arms:
        row("gepa-optimized POWERED", arms["power-gepa-high"], "high/12k", bold=True)
    row(f"engineered (subset, {eng['n_replicates']} reps)", eng, "none/1024")
    if "power-engineered-high" in arms:
        row("engineered POWERED", arms["power-engineered-high"], "high/12k", bold=True)
    row("FC-score-k8, grid best (3 reps)", cells["FC-score-k8"], "none/1024, k=8")
    if ragas:
        row(f"RAGAS-style (subset, {ragas['n_replicates']} reps)", ragas, "multi-call")
    row("pipeline B2", pipe["B2"], "none, per-fact")
    row("pipeline B3", pipe["B3"], "none+high audit")
    A(f"| B3 + per-fact critical rule | | | | "
      f"{100 * rule['absolute']['det']['p']:.1f}% @ "
      f"{100 * rule['absolute']['fa']['p']:.1f}% | | |")
    A("")
    k1r = cells["FC-score-k1"]["paired_omissions"]["per_replicate"]
    k8r = cells["FC-score-k8"]["paired_omissions"]["per_replicate"]
    A(f"Control replicate ranges on this subset: FC-score-k1 "
      f"{min(k1r):.3f}-{max(k1r):.3f}; FC-score-k8 {min(k8r):.3f}-{max(k8r):.3f}. "
      "Powered arms are one replicate each - read them against those ranges.")
    A("")
    A("## Residual conditioning (paired, omissions)")
    A("")
    A("| design | complete | partial-weak | partial-strong |")
    A("|---|---|---|---|")
    rows = [("FC-score-k1 @ none", cells["FC-score-k1"]), ("factlist @ none", factlist),
            ("gepa @ none", gepa), ("engineered @ none", eng)]
    rows += [(an, arms[an]) for an in POWER_ARMS if an in arms]
    rows += [("pipeline B2", pipe["B2"]), ("pipeline B3", pipe["B3"])]
    for name, blk in rows:
        r = blk.get("paired_by_residual") or {}
        g = lambda k: (f"{r[k]['paired']:.3f}"
                       if r.get(k) and r[k].get("paired") is not None else "n/a")
        A(f"| {name} | {g('complete')} | {g('partial-weak')} | {g('partial-strong')} |")
    A("")
    A("## Provenance")
    A("")
    A(f"- Powered arms: `{STORE}` + `{ENG_STORE}` (runner `w2_power.py`), restricted to "
      "the 151-pair confirmation subset (the store also carries the FINDINGS 31 "
      f"evaluation-set records); smoke `results/{EXPERIMENT}/_state/power-smoke.jsonl`")
    A(f"- Controls: `{GRID}`, `{FACTLIST_STORE}`, `{STRONG_STORE}` (subset-restricted), "
      f"`{BASELINES_STORE}`")
    A(f"- Ceilings: `{B2_STORE}`, `{B3_STORE}`")
    A("- Compute columns are receipts-based (OpenRouter usage incl. reasoning tokens); "
      "pipeline rows amortise extraction/audit per consultation, so their $/note is the "
      "check-stage share only - the deployment ladder in FINDINGS carries the full "
      "amortised prices.")
    A("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(L))


if __name__ == "__main__":
    main()
