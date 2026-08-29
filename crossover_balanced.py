"""P2.1 - the honest 4-arm crossover on balanced (add/change/omit) verified-clean pairs.

Arms (all on the SAME pairs, same ground truth clean=PASS / errored=FAIL):
  naive       : plain faithfulness judge (claude -p PASS/FAIL)
  steelman    : a colleague's v14 (the best conventional faithfulness judge) - the arm Phase G lacked
  Composo legacy    : align-20251111 (no retrieval) + the completeness-aware criterion
  Composo retrieval : align-20260109 (retrieve-per-output core) + criterion

Metric: per pair, did the arm score the errored note below its clean twin (discriminate)?
Reported OVERALL and PER TYPE. The thesis prediction: steelman catches add/change but is BLIND to
omit; Composo separates across all three. Honest either way - reports raw numbers + score gaps.

Usage: python crossover_balanced.py [--threshold 0.5]
Output: crossover_balanced.json
"""
import json, os, re, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from statistics import mean
from common import claude, make_composo, HERE
from v14_judge import v14_judge


def _arg(flag, d=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else d


THRESH = float(_arg("--threshold", "0.5"))
HN = json.load(open(os.path.join(HERE, "hard_negatives_balanced.json")))
SCEN = {s["id"]: s for s in json.load(open(os.path.join(HERE, "authored_scenarios.json")))}

CRIT = ("Reward clinical notes that a careful clinician would accept as faithful AND complete: no "
        "fabricated, inverted/negated, hardened, wrong-value, wrong-subject, or wrong-laterality "
        "findings, and no omitted clinically-significant finding, diagnosis, medication, safety-net, "
        "or red flag. Penalise any difference from the consultation that would change the clinician's "
        "understanding or the patient's care.")
NAIVE = ("Is this clinical note faithful to the consultation transcript? Reply ONLY one word: PASS or "
         "FAIL.\n\nTRANSCRIPT:\n{t}\n\nNOTE:\n{n}")


def naive(t, n):
    out = claude(NAIVE.format(t=t[:40000], n=n), model="claude-opus-4-8", effort="medium")
    return "FAIL" if re.search(r"\bFAIL\b", out, re.I) else "PASS"


def composo_score(client, t, n):
    try:
        r = client.evaluate(messages=[
            {"role": "user", "content": f"Consultation transcript:\n{t[:40000]}\n\nWrite the clinical note."},
            {"role": "assistant", "content": n}], criteria=CRIT)
        return float(r.score)
    except Exception:
        return None


def main():
    legacy = make_composo("align-20251111")
    retr = make_composo("align-20260109")

    def run(hn):
        t = SCEN[hn["id"]]["transcript"]
        o = {"id": hn["id"], "type": hn["type"], "change": hn.get("change")}
        o["naive_clean"], o["naive_err"] = naive(t, hn["clean"]), naive(t, hn["errored"])
        o["steel_clean"] = "FAIL" if v14_judge(t, hn["clean"]) else "PASS"     # v14 flags=FAIL
        o["steel_err"] = "FAIL" if v14_judge(t, hn["errored"]) else "PASS"
        o["legacy_clean"], o["legacy_err"] = composo_score(legacy, t, hn["clean"]), composo_score(legacy, t, hn["errored"])
        o["retr_clean"], o["retr_err"] = composo_score(retr, t, hn["clean"]), composo_score(retr, t, hn["errored"])
        return o

    with ThreadPoolExecutor(max_workers=5) as ex:
        res = list(ex.map(run, HN))
    json.dump(res, open(os.path.join(HERE, "crossover_balanced.json"), "w"), indent=1)

    def disc_binary(r, p):   # naive / steelman: errored FAIL AND clean PASS
        return r[f"{p}_err"] == "FAIL" and r[f"{p}_clean"] == "PASS"

    def disc_score(r, p):    # composo: errored below clean AND clean above threshold
        ce, cl = r.get(f"{p}_err"), r.get(f"{p}_clean")
        return ce is not None and cl is not None and ce < cl and cl >= THRESH

    arms = [("naive", disc_binary, "naive"), ("steelman(v14)", disc_binary, "steel"),
            ("Composo legacy", disc_score, "legacy"), ("Composo retrieval", disc_score, "retr")]
    types = sorted({h["type"] for h in HN})
    n = len(res)
    type_counts = {t: sum(1 for h in HN if h["type"] == t) for t in types}
    print(f"\n=== Balanced crossover: {n} verified-clean pairs ({type_counts}) ===")

    def rate(fn, key, subset):
        s = [r for r in res if r["type"] in subset]
        return f"{sum(fn(r, key) for r in s)}/{len(s)}"

    print(f"\n{'arm':20} {'overall':>9} " + " ".join(f"{t:>8}" for t in types))
    for name, fn, key in arms:
        row = " ".join(f"{rate(fn,key,{t}):>8}" for t in types)
        print(f"{name:20} {rate(fn,key,set(types)):>9} {row}")

    # threshold-free score gaps for Composo arms, per type
    print("\nComposo mean score gap (clean - err), threshold-free:")
    for key in ("legacy", "retr"):
        for t in types + ["ALL"]:
            sub = [r for r in res if (t == "ALL" or r["type"] == t)]
            gaps = [r[f"{key}_clean"] - r[f"{key}_err"] for r in sub
                    if r.get(f"{key}_clean") is not None and r.get(f"{key}_err") is not None]
            if gaps:
                print(f"  {key:7} {t:7}: {mean(gaps):+.3f}  (n={len(gaps)})")
    print("\nsaved -> crossover_balanced.json")


if __name__ == "__main__":
    main()
