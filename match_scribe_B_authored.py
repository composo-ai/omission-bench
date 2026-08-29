"""Match the 10 downloaded scribe_B notes (/tmp/scribe_B_new/*.txt) to their AUTHORED consult by content
(Cohere cosine vs each scenario transcript), 1:1 greedy assignment, save scribe_B_notes/authored__<id>.txt.
"""
import json, os, glob
import numpy as np
import cohere
from dotenv import load_dotenv
from common import HERE

load_dotenv(os.path.join(HERE, "secrets.env"))
CO = cohere.ClientV2(os.environ["COHERE_API_KEY"])

SUBSET = ["ha_sudden_thunderclap", "ha_migraine_aura_pill", "gastro_contact_history", "phone_uti",
          "throat_viral_noabx", "cp_reflux_gord", "diabetes_review", "eczema_sites_otc",
          "penicillin_delabel", "mh_followup_ssri"]
SCEN = {s["id"]: s for s in json.load(open(os.path.join(HERE, "authored_scenarios.json")))}
NOTES_DIR = os.path.join(HERE, "scribe_B_notes")


def embed(texts, kind="search_document"):
    r = CO.embed(texts=texts, model="embed-v4.0", input_type=kind, embedding_types=["float"])
    v = np.array(r.embeddings.float, dtype=np.float64)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


def main():
    scribe_B = [(fp, open(fp).read().strip()) for fp in sorted(glob.glob("/tmp/scribe_B_new/*.txt"))]
    scribe_B = [(fp, t) for fp, t in scribe_B if len(t) > 120]
    print(f"{len(scribe_B)} scribe_B notes vs {len(SUBSET)} consults")

    he = embed([t for _, t in scribe_B])
    ce = embed([SCEN[i]["transcript"][:6000] for i in SUBSET])
    sim = he @ ce.T   # (n_scribe_B, n_consult)

    # greedy 1:1: highest similarity pair first
    pairs = sorted(((sim[k, j], k, j) for k in range(len(scribe_B)) for j in range(len(SUBSET))), reverse=True)
    used_h, used_c, assign = set(), set(), {}
    for sc, k, j in pairs:
        if k in used_h or j in used_c:
            continue
        used_h.add(k); used_c.add(j); assign[k] = (j, sc)

    os.makedirs(NOTES_DIR, exist_ok=True)
    print(f"\n{'scribe_B file':32} -> {'consult':24} sim")
    rows = []
    for k, (fp, t) in enumerate(scribe_B):
        j, sc = assign[k]
        cid = SUBSET[j]
        open(os.path.join(NOTES_DIR, f"authored__{cid}.txt"), "w").write(t)
        rows.append((cid, sc))
        print(f"  {os.path.basename(fp):30} -> {cid:24} {sc:.3f}")
    matched = {SUBSET[assign[k][0]] for k in assign}
    missing = set(SUBSET) - matched
    print(f"\nmatched {len(matched)}/10 | missing: {missing or 'none'}")
    lo = [(c, s) for c, s in rows if s < 0.55]
    if lo:
        print(f"LOW-confidence (<0.55), eyeball these: {lo}")
    print("saved -> scribe_B_notes/authored__*.txt")


if __name__ == "__main__":
    main()
