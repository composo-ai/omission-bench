"""Phase D - the faithfulness yardstick. Ports a colleague's two best CONVENTIONAL judges to
run on the lead author's Claude plan (`claude -p`):
  - v14  : the HAND-TUNED prompt (his kappa=0.77 / recall 1.00 / 0-FN reference on ACI-Bench)
  - gepa : the DSPy-GEPA optimised prompt (v23b; his kappa=0.46 - i.e. GEPA LOST to hand-tuning)
Both are binary YES/NO "does the note contain >=1 clinically-meaningful factual error" judges,
and both are FAITHFULNESS-ONLY (v14 explicitly: "Omissions are NOT errors"). That's the point:
they're the steelman for assertion errors, structurally blind to omission.

`python v14_judge.py`  -> re-baseline both on the ACI-Bench test split (the pre-scaling gate:
reproduce ~a colleague's numbers on our model before we trust them on our own notes).
Import `v14_judge(t, n)` / `gepa_judge(t, n)` elsewhere to score our generated notes.
"""
import csv, json, os, re
from concurrent.futures import ThreadPoolExecutor
from common import claude, HERE

# Both prompts this script judges with ship in this repository: the deployed
# faithfulness judge as `w2_prompts/v14_as_shipped.txt`, and the optimiser's
# winning judge as `gepa/judge_prompt.txt`. The study read them from its own
# working tree; the released copies are byte-identical.
_PROMPTS = os.path.dirname(os.path.abspath(__file__))
V14_PROMPT = open(os.path.join(_PROMPTS, "w2_prompts", "v14_as_shipped.txt")).read()   # has {transcript}/{summary}
GEPA_PROMPT = open(os.path.join(_PROMPTS, "gepa", "judge_prompt.txt")).read()          # no placeholders

ERR_CATS = ("Hallucination", "Inference", "Misunderstanding")


def _verdict(out):
    """YES (>=1 error) / NO from a judge reply. Prefer an explicit 'Verdict:' line, else last token."""
    u = (out or "").upper()
    m = re.search(r"VERDICT:\s*(YES|NO)", u)
    if m:
        return m.group(1)
    toks = re.findall(r"\b(YES|NO)\b", u)
    return toks[-1] if toks else None


def v14_judge(transcript, note, timeout=300):
    """True if the v14 hand-tuned judge flags >=1 faithfulness error."""
    p = V14_PROMPT.replace("{transcript}", transcript[:40000]).replace("{summary}", note)
    return _verdict(claude(p, timeout=timeout, model="claude-opus-4-8", effort=os.environ.get("STEELMAN_EFFORT","medium"))) == "YES"


def gepa_judge(transcript, note, timeout=300):
    """True if the GEPA-optimised judge flags >=1 faithfulness error."""
    p = GEPA_PROMPT + f"\n\n### Transcript\n{transcript[:40000]}\n\n### Clinical note\n{note}\n\n### Final answer (exactly YES or NO)\n"
    return _verdict(claude(p, timeout=timeout, model="claude-opus-4-8", effort=os.environ.get("STEELMAN_EFFORT","medium"))) == "YES"


# ---------------------------------------------------------------- ACI-Bench re-baseline
def load_test_docs():
    docs = []
    csv.field_size_limit(10_000_000)
    splits = json.load(open(os.path.join(ACI, "splits.json")))
    test_ids = {k for k, v in splits.items() if v == "test"}
    with open(os.path.join(ACI, "data/acibench/NaturalHallucinationDataset-Outputs-open.csv")) as f:
        for row in csv.DictReader(f):
            did = f"{row['file']}__{row['model_id']}"
            if did in test_ids:
                gold = any(int(row[c]) > 0 for c in ERR_CATS)
                docs.append({"id": did, "transcript": row["transcript"], "summary": row["summary"], "gold": gold})
    return docs


def kappa(gold, pred):
    n = len(gold)
    po = sum(g == p for g, p in zip(gold, pred)) / n
    pg, pp = sum(gold) / n, sum(pred) / n
    pe = pg * pp + (1 - pg) * (1 - pp)
    return (po - pe) / (1 - pe) if (1 - pe) else 0.0


def scores(gold, pred):
    tp = sum(g and p for g, p in zip(gold, pred)); fp = sum((not g) and p for g, p in zip(gold, pred))
    fn = sum(g and (not p) for g, p in zip(gold, pred)); tn = sum((not g) and (not p) for g, p in zip(gold, pred))
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"kappa": round(kappa(gold, pred), 3), "recall": round(rec, 2), "precision": round(prec, 2),
            "f1": round(f1, 2), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    docs = load_test_docs()
    print(f"ACI-Bench test docs: {len(docs)} (positive/has-error: {sum(d['gold'] for d in docs)})")

    def judge(d):
        return {"id": d["id"], "gold": d["gold"],
                "v14": v14_judge(d["transcript"], d["summary"]),
                "gepa": gepa_judge(d["transcript"], d["summary"])}

    with ThreadPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(judge, docs))
    json.dump(res, open(os.path.join(HERE, "v14_acibench_results.json"), "w"), indent=1)

    gold = [r["gold"] for r in res]
    print("\n=== Re-baseline on ACI-Bench (our model: opus-4-8/medium via claude -p) ===")
    print("v14  (hand-tuned):", scores(gold, [r["v14"] for r in res]))
    print("gepa (v23b)      :", scores(gold, [r["gepa"] for r in res]))
    print("\n(a colleague's gpt-5.5 reference: v14 kappa 0.77 / recall 1.00 / F1 0.97 ; gepa all-errors kappa 0.46)")
    print("gate: v14 should land near a colleague's; if so we trust both judges on our own notes.")
    print("saved -> v14_acibench_results.json")


if __name__ == "__main__":
    main()
