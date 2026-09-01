"""dedup_estimate.py - how many DISTINCT errors do the 618 verified findings represent?

The census's findings are not merged across discovery passes, so one underlying error surfaced
by two hunts counts twice (the census paper discloses this; the note-level headline is immune).
Without a deduplication estimate, a reader cannot tell what 618 is a count OF - whether the
111-member allergy cluster is 111 distinct failures or nearer 40.

Two estimates, one cheap and offline, one model-assisted, reported together:

1. OFFLINE (free): within each note, union-find over pairs whose description embeddings
   (the clustering stage's cached Cohere embed-v4.0 vectors) sit above a cosine threshold.
   Reported at 0.80 / 0.85 / 0.90 as a sensitivity band, because "same underlying error"
   has no sharp phrasing-similarity boundary.

2. MODEL-ASSISTED (bought, capped): one call per note that carries more than one verified
   finding (132 of the 177 notes with any). The model sees the full note plus every finding
   on it, and partitions them into groups where correcting the note once would resolve the
   whole group. Grouping the findings of one note jointly beats pairwise similarity because
   it is transitive by construction and sees the note. Constructor pin, deterministic seeds,
   manifests - the census's own transport discipline (second_panel.py precedent).

The headline: 618 verified findings correspond to approximately N distinct errors, where
N = 45 single-finding notes + the model's group count over the 132 multi-finding notes.
A sample of the model's merge decisions is hand-checked before the number is used (the
groups file exists for exactly that read).

    python3 census/dedup_estimate.py --smoke     # first 5 notes, read the groups
    python3 census/dedup_estimate.py -y          # the rest
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import argparse, hashlib, json, os, time

import numpy as np

from common import HERE, RESULTS, Run
import taxonomy_common as T
from w2_common import RecordStore, SpendGuard, confirm, run_block

EXPERIMENT = "dedup-estimate"
VERIFIED = "master/findings_verified_master.json"
EMB_CACHE = "master/findings_embeddings.npz"
OUTDIR = os.path.join(RESULTS, EXPERIMENT)
SPEC = "deduplication estimate over the census's 618 verified findings"
THRESHOLDS = (0.80, 0.85, 0.90)

PROMPT = """You are auditing findings from a documentation-error census of AI-written clinical notes.
Below is ONE note written by an AI scribe, followed by {k} verified error findings on that note,
each from a separate discovery pass. Passes overlap, so several findings may describe the SAME
underlying error in different words.

Group the findings: two findings belong to the same group when they describe the same underlying
documentation error - that is, correcting the note in one place would resolve both. Findings that
merely concern the same topic but describe different defects (e.g. one fact omitted and a different
fact fabricated in the same section) are different groups. If unsure, keep them separate.

THE NOTE:
{note}

THE FINDINGS:
{findings}

Return ONLY JSON: {{"groups": [[finding numbers of group 1], [group 2], ...], "reasons": ["one short sentence per group of 2+ explaining why its members are one error", ...]}}
Every finding number from 1 to {k} must appear in exactly one group."""


def load_verified():
    d = json.load(open(os.path.join(HERE, VERIFIED)))
    v = [f for f in d["all_issues"] if (f.get("verdict") or {}).get("is_real")]
    if len(v) != 618:
        raise SystemExit(f"{len(v)} verified, expected 618")
    return v


def offline_estimate(v):
    texts = [(f.get("description") or "").strip() for f in v]
    key = hashlib.sha256("\x00".join(texts).encode()).hexdigest()
    z = np.load(os.path.join(HERE, EMB_CACHE), allow_pickle=False)
    if bytes(z["key"]).decode() != key:
        raise SystemExit("embedding cache key mismatch - rebuild findings_embeddings.npz first")
    emb = z["emb"]
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    by_note = {}
    for i, f in enumerate(v):
        by_note.setdefault(f["note_key"], []).append(i)
    out = {}
    for t in THRESHOLDS:
        parent = list(range(len(v)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        n_pairs = 0
        for idx in by_note.values():
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    i, j = idx[a], idx[b]
                    if float(emb[i] @ emb[j]) >= t:
                        n_pairs += 1
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            parent[ri] = rj
        n_groups = len({find(i) for i in range(len(v))})
        out[str(t)] = {"cosine_threshold": t, "n_merged_pairs": n_pairs,
                       "n_distinct": n_groups,
                       "dedup_factor": round(618 / n_groups, 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="first 5 multi-finding notes only")
    ap.add_argument("--warn-usd", type=float, default=10.0)
    ap.add_argument("--stop-usd", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    v = load_verified()
    os.makedirs(OUTDIR, exist_ok=True)

    offline = offline_estimate(v)
    print("offline (embedding union-find within note):")
    for t, r in offline.items():
        print(f"  cos>={t}: {r['n_distinct']} distinct ({r['n_merged_pairs']} merged pairs, "
              f"factor {r['dedup_factor']})")

    units, _ = T.note_units()
    notes = {u["note_key"]: u for u in units}
    by_note = {}
    for f in v:
        by_note.setdefault(f["note_key"], []).append(f)
    multi = {k: fs for k, fs in by_note.items() if len(fs) > 1}
    singles = len(by_note) - len(multi)
    todo_keys = sorted(multi)
    if args.smoke:
        todo_keys = todo_keys[:5]

    store = RecordStore(EXPERIMENT, "groups-r1")
    have = set(store.records)
    todo_keys = [k for k in todo_keys if f"{k}|dedup" not in have]
    print(f"\n{len(multi)} multi-finding notes ({singles} single-finding); "
          f"{len(todo_keys)} calls to buy this run")
    if not todo_keys:
        analyze(v, by_note, store, offline)
        return
    if not args.yes:
        confirm(f"buy {len(todo_keys)} grouping calls on the constructor pin?", args.yes)

    guard = SpendGuard(warn=args.warn_usd, stop=args.stop_usd, scope="run")
    usage = T.new_usage()
    prompts = {"dedup_group": PROMPT}
    params = {"stage": "dedup-group", "model_role": T.CONSTRUCTOR_ROLE,
              "n_notes": len(todo_keys), "smoke": args.smoke,
              "record_store": os.path.relpath(store.path, HERE)}
    with Run(EXPERIMENT, params=params, inputs=T.run_inputs(extra=["dedup_estimate.py", VERIFIED]),
             spec=SPEC, allow_dirty=args.allow_dirty, spend="openrouter_credits") as run:
        run.register_prompts(prompts)

        def worker(note_key):
            fs = multi[note_key]
            u = notes[note_key]
            listing = "\n".join(
                f"{i}. [{f.get('mode')}] {f.get('description')}\n"
                f"   note quote: {f.get('note_quote') or '-'}\n"
                f"   transcript quote: {f.get('source_quote') or '-'}"
                for i, f in enumerate(fs, 1))
            key = f"{note_key}|dedup"
            prompt = PROMPT.format(k=len(fs), note=u["note"], findings=listing)
            obj, meta = T.route_call(prompt, key, "openrouter", kind="panel",
                                     role=T.CONSTRUCTOR_ROLE)
            groups = obj.get("groups") if isinstance(obj, dict) else None
            ok = (isinstance(groups, list)
                  and sorted(x for g in groups for x in g) == list(range(1, len(fs) + 1)))
            store.put({"key": key, "note_key": note_key, "n_findings": len(fs),
                       "finding_ids": [f["finding_id"] for f in fs],
                       "groups": groups if ok else None,
                       "reasons": (obj or {}).get("reasons") if ok else None,
                       "parse_ok": ok, "raw": None if ok else str(obj)[:2000],
                       "model": meta.get("model"), "seed": meta.get("seed"),
                       "error": meta.get("error"), "cost_usd": meta.get("cost_usd"),
                       "usage": meta.get("usage"), "run_id": run.run_id,
                       "t": round(time.time(), 3)})
            T.usage_add(usage, meta)
            guard.add(meta.get("cost_usd"))

        run_block(todo_keys, worker, args.workers, warmup=False)
        run.save("summary.json", {"n_calls": len(todo_keys), "cost_usd": usage["cost_usd"],
                                  "errors": usage["errors"]})
    print(f"spent ${usage['cost_usd']:.2f} over {len(todo_keys)} calls, {usage['errors']} errors")
    analyze(v, by_note, store, offline)


def analyze(v, by_note, store, offline):
    multi = {k: fs for k, fs in by_note.items() if len(fs) > 1}
    singles = len(by_note) - len(multi)
    recs = {r["note_key"]: r for r in store.records.values()}
    missing = [k for k in multi if k not in recs or not recs[k].get("parse_ok")]
    scored = [k for k in multi if k in recs and recs[k].get("parse_ok")]
    n_groups = sum(len(recs[k]["groups"]) for k in scored)
    n_findings_scored = sum(len(multi[k]) for k in scored)
    merged_groups = []
    for k in scored:
        fs = multi[k]
        for gi, g in enumerate(recs[k]["groups"]):
            if len(g) > 1:
                merged_groups.append({
                    "note_key": k, "n": len(g),
                    "finding_ids": [fs[i - 1]["finding_id"] for i in g],
                    "descriptions": [fs[i - 1]["description"] for i in g],
                    "reason": (recs[k].get("reasons") or [None] * (gi + 1))[gi]
                              if isinstance(recs[k].get("reasons"), list) and gi < len(recs[k]["reasons"]) else None})
    # finding_id -> global group id (singleton notes get their own group each)
    fid2g, gid = {}, 0
    for nk in sorted(by_note):
        fs = by_note[nk]
        if len(fs) == 1 or nk not in recs or not recs[nk].get("parse_ok"):
            for f in fs:
                fid2g[f["finding_id"]] = gid
                gid += 1
        else:
            for g in recs[nk]["groups"]:
                for i in g:
                    fid2g[fs[i - 1]["finding_id"]] = gid
                gid += 1
    # per-cluster distinct-group counts (is the 111-member allergy cluster 111 distinct
    # failures or nearer 40?). A group can span clusters, so
    # these are per-cluster reads, not a partition of the global total.
    clusters_path = os.path.join(HERE, "master", "findings_clusters.json")
    per_cluster = {}
    if os.path.exists(clusters_path):
        assign = json.load(open(clusters_path))["assignments"]
        cl_n, cl_g = {}, {}
        for fid, cid in assign.items():
            cl_n[cid] = cl_n.get(cid, 0) + 1
            cl_g.setdefault(cid, set()).add(fid2g[fid])
        per_cluster = {str(cid): {"n_findings": cl_n[cid], "n_distinct": len(cl_g[cid])}
                       for cid in sorted(cl_n, key=lambda c: -cl_n[c])}

    out = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "offline_embedding_estimate": offline,
           "per_cluster_distinct": per_cluster,
           "model_pass": {
               "notes_multi": len(multi), "notes_single": singles,
               "notes_scored": len(scored), "notes_missing_or_unparsed": missing,
               "n_findings_on_scored_notes": n_findings_scored,
               "n_groups_on_scored_notes": n_groups,
               "n_distinct_estimate": singles + n_groups + sum(len(multi[k]) for k in missing),
               "note": "missing/unparsed notes counted at one error per finding (no merges), "
                       "i.e. conservatively AGAINST deduplication",
           },
           "n_verified": len(v), "n_notes_with_any": len(by_note)}
    if scored:
        out["model_pass"]["dedup_factor"] = round(
            len(v) / out["model_pass"]["n_distinct_estimate"], 3)
    with open(os.path.join(OUTDIR, "dedup_estimate.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(OUTDIR, "merged_groups.json"), "w") as fh:
        json.dump(merged_groups, fh, indent=1)
    print(json.dumps(out["model_pass"], indent=1)[:800])
    print(f"wrote {OUTDIR}/dedup_estimate.json and merged_groups.json "
          f"({len(merged_groups)} merged groups for the hand-check)")


if __name__ == "__main__":
    main()
