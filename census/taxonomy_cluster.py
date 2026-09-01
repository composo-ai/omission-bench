"""The census, stage 3 of 4: cluster the verified findings into subcategories under the
published frame.

The June pilot's clustering recipe, pointed at the master findings:

  1. embed each finding DESCRIPTION (not the note) with Cohere embed-v4.0,
     input_type="clustering", behind the validated bias prefix
     "Failure mode described in this analysis:\\n";
  2. UMAP(-> 15d, n_neighbors, metric=cosine, seed 42) -> HDBSCAN(min_cluster_size,
     min_samples);
  3. one mode per dense pocket, anchor = L2-normalised medoid, members ordered
     nearest-anchor first, HDBSCAN noise (-1) dropped;
  4. label each cluster from its 20 most-central descriptions with one LLM call.

**Clustering does not decide the top-level categories.** The taxonomy is
not purely emergent: `taxonomy_frame.json` fixes the published top-level
categories, and clustering mints SUBCATEGORIES beneath them. Every cluster is
placed under a Tier-2 frame category by member vote when >=60% of its members agree,
otherwise the labelling call is asked to place it; a cluster neither route can place
is reported as **novel_unmapped** in its own section, never folded into a frame
category to make the table look tidy. A novel mode is a finding ABOUT the frame.

The UMAP, HDBSCAN and medoid recipe is the June pilot's, ported faithfully. Three
departures from it are deliberate: the embedding vendor, `n_neighbors` (the pilot
default over-smoothed the ~170-finding June pool into two mega-clusters, so the
`--sweep` decides it here), and one-shot clustering of a closed corpus rather than
incremental assignment against a durable registry, since there is no registry here
for cluster identity to stay stable against.

Cohere embeddings are the study's one direct-cash exception; everything
else honours --route. Embeddings are cached by content hash, so re-labelling or
re-sweeping the same pool never re-pays Cohere.

    python3 census/taxonomy_cluster.py --in master/findings_verified_master.json --sweep
    python3 census/taxonomy_cluster.py --in master/findings_verified_master.json -y
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
from collections import Counter

import numpy as np

from common import HERE, Run
import taxonomy_common as T

EMBED_MODEL = "embed-v4.0"
EMBED_PREFIX = "Failure mode described in this analysis:\n"     # verbatim from the June pilot
UMAP_N_COMPONENTS, UMAP_METRIC, RANDOM_STATE = 15, "cosine", 42
CACHE = os.path.join(T.MASTER, "findings_embeddings.npz")

LABEL_PROMPT = """These are real failure descriptions from one cluster of AI clinical-note errors (most-central first).
Give the cluster a SHORT name (3-7 words) and a 1-2 sentence description of the shared failure mode.
Then place it in the reporting frame below. Choose the ONE frame category the cluster belongs under, or "novel" if
this cluster is genuinely a different kind of failure from all of them - do NOT force a poor fit, a novel mode is a
useful result. Be specific and diagnostic; avoid generic labels.

FRAME CATEGORIES:
{frame}

Return ONLY JSON {{"label":"...","description":"...","frame_category":"<one key above, or novel>","frame_reason":"<short>"}}.

SAMPLES:
{samples}"""


def frame_menu():
    """The Tier-2 keys the labelling call may choose from, with one-line definitions."""
    rows = []
    for c in T.load_frame()["tier2"]:
        if c["key"] == "open":
            continue
        instr = c.get("pass_instruction") or ""
        if not instr:
            from taxonomy_discover import MODE_PASSES
            instr = MODE_PASSES.get(c.get("pass") or "", "")
        if not instr:
            instr = c.get("why_not_hunted", "")
        rows.append(f"- {c['key']}: {instr[:180]}")
    return "\n".join(rows)


def place_cluster(members, llm_choice):
    """(tier2, tier1, method) for one cluster.

    Member vote first - it is deterministic and it is what the frame's own pass
    derivation guarantees. The labelling call only decides where the members do not
    already agree, which is mostly clusters built out of open-pass findings.
    """
    votes = Counter(m.get("frame_tier2") for m in members if m.get("frame_tier2"))
    if votes:
        top, n = votes.most_common(1)[0]
        if n / len(members) >= 0.60:
            return top, T.tier1_of_tier2(top), f"member vote {n}/{len(members)}"
    valid = {c["key"] for c in T.load_frame()["tier2"]} - {"open"}
    if llm_choice in valid:
        return llm_choice, T.tier1_of_tier2(llm_choice), "label call"
    return None, None, "unplaced"


# ---------------------------------------------------------------- embed
def embed(texts, no_cache=False):
    """Cohere embeddings, content-hash cached (the one paid call made outside --route)."""
    key = hashlib.sha256("\x00".join(texts).encode()).hexdigest()
    if not no_cache and _cache_key_matches(CACHE, key):
        emb = np.load(CACHE, allow_pickle=False)["emb"].astype(np.float64)
        print(f"  embeddings: cache hit ({emb.shape[0]} vectors, no Cohere spend)")
        return emb
    if not os.environ.get("COHERE_API_KEY"):
        raise SystemExit("COHERE_API_KEY missing from secrets.env - clustering needs it "
                         "(the one direct-cash exception to the study's routing).")
    import cohere
    co = cohere.ClientV2(os.environ["COHERE_API_KEY"])
    wrapped = [EMBED_PREFIX + t for t in texts]
    out = []
    for i in range(0, len(wrapped), 96):
        r = co.embed(texts=wrapped[i:i + 96], model=EMBED_MODEL,
                     input_type="clustering", embedding_types=["float"])
        out.extend(r.embeddings.float)
        print(f"  embedded {min(i + 96, len(wrapped))}/{len(wrapped)}", flush=True)
    emb = np.array(out, dtype=np.float64)
    np.savez_compressed(CACHE, emb=emb, key=np.frombuffer(key.encode(), dtype=np.uint8))
    return emb


def _cache_key_matches(path, key):
    try:
        z = np.load(path, allow_pickle=False)
        return bytes(z["key"]).decode() == key
    except Exception:
        return False


# ---------------------------------------------------------------- cluster
def cluster_pool(embeddings, n_neighbors, min_cluster_size, min_samples, seed=RANDOM_STATE):
    import hdbscan, umap
    pts = np.asarray(embeddings, dtype=np.float32)
    nn = min(n_neighbors, max(2, pts.shape[0] - 1))
    reduced = umap.UMAP(n_components=UMAP_N_COMPONENTS, n_neighbors=nn, metric=UMAP_METRIC,
                        random_state=seed, n_jobs=1).fit_transform(pts)
    return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                           min_samples=min_samples).fit_predict(reduced)


def l2(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def medoid_idx(P):
    Pn = P / np.clip(np.linalg.norm(P, axis=1, keepdims=True), 1e-12, None)
    return int(np.argmax((Pn @ Pn.T).mean(axis=1)))


def propose_modes(embeddings, labels, min_cluster_size):
    P = np.asarray(embeddings, dtype=np.float64)
    modes = []
    for cid in sorted({int(x) for x in labels}):
        if cid == -1:
            continue
        idx = np.where(labels == cid)[0]
        if len(idx) < min_cluster_size:
            continue
        cp = P[idx]
        anchor = l2(cp[medoid_idx(cp)])
        cpn = cp / np.clip(np.linalg.norm(cp, axis=1, keepdims=True), 1e-12, None)
        order = np.argsort(-(cpn @ anchor), kind="stable")
        modes.append({"cluster_id": cid, "member_indices": [int(idx[i]) for i in order]})
    return modes


# ---------------------------------------------------------------- sweep
def sweep(emb, findings, args):
    """Parameter grid + a seed-stability read at the chosen config."""
    from sklearn.metrics import adjusted_rand_score
    rows = []
    print(f"\nparameter sweep over {len(findings)} findings "
          f"(flagged 'good' = 6-16 clusters and no cluster over 45% of the pool)")
    for nn in (10, 15):
        for mcs in (10, 8, 6, 5):
            for ms in (1, 2, 3):
                lab = cluster_pool(emb, nn, mcs, ms)
                sizes = sorted((v for k, v in Counter(lab).items() if k != -1), reverse=True)
                noise = int(np.sum(lab == -1))
                good = bool(6 <= len(sizes) <= 16 and (sizes[0] if sizes else 0) <= len(findings) * 0.45)
                rows.append({"n_neighbors": nn, "min_cluster_size": mcs, "min_samples": ms,
                             "n_clusters": len(sizes), "noise": noise, "sizes": sizes[:12],
                             "good": good})
                print(f"  nn={nn} mcs={mcs:2} ms={ms}: {len(sizes):2} clusters, noise={noise:4}, "
                      f"sizes={sizes[:12]}{'  <-- good' if good else ''}")
    labs = [cluster_pool(emb, args.n_neighbors, args.min_cluster_size, args.min_samples, seed=s)
            for s in (42, 43, 44)]
    aris = [float(adjusted_rand_score(labs[i], labs[j])) for i, j in ((0, 1), (0, 2), (1, 2))]
    stab = {"config": {"n_neighbors": args.n_neighbors, "min_cluster_size": args.min_cluster_size,
                       "min_samples": args.min_samples},
            "umap_seeds": [42, 43, 44],
            "n_clusters_per_seed": [len({int(x) for x in l} - {-1}) for l in labs],
            "pairwise_ari": [round(a, 4) for a in aris],
            "mean_ari": round(float(np.mean(aris)), 4)}
    print(f"\nseed stability at the chosen config {stab['config']}: "
          f"clusters {stab['n_clusters_per_seed']}, mean pairwise ARI {stab['mean_ari']}")
    path = os.path.join(T.MASTER, "findings_cluster_sweep.json")
    json.dump({"n_findings": len(findings), "grid": rows, "seed_stability": stab,
               "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
              open(path, "w"), indent=1)
    print(f"saved -> {os.path.relpath(path, HERE)}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="census stage 3: cluster verified findings into subcategories")
    ap.add_argument("--in", dest="infile", default="master/findings_verified_master.json")
    ap.add_argument("--route", default="openrouter", choices=["openrouter", "plan"])
    ap.add_argument("--n-neighbors", type=int, default=10, help="UMAP n_neighbors (v2 value: 10)")
    ap.add_argument("--min-cluster-size", type=int, default=10, help="HDBSCAN min_cluster_size")
    ap.add_argument("--min-samples", type=int, default=5, help="HDBSCAN min_samples")
    ap.add_argument("--include-refuted", action="store_true",
                    help="cluster ALL candidates, not just panel survivors")
    ap.add_argument("--sweep", action="store_true", help="parameter sweep + seed stability, no labels")
    ap.add_argument("--no-cache", action="store_true", help="re-pay Cohere rather than reuse the cache")
    ap.add_argument("--out-prefix", default="findings")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    data = json.load(open(os.path.join(HERE, args.infile)))
    findings = data["all_issues"]
    if not args.include_refuted:
        findings = [f for f in findings if (f.get("verdict") or {}).get("is_real", True)]
    texts = [(f.get("description") or "").strip() for f in findings]
    keep = [i for i, t in enumerate(texts) if t]
    if len(keep) < len(findings):
        print(f"  ! {len(findings) - len(keep)} findings have no description and cannot be "
              "embedded - dropped from the taxonomy (they stay in the verified file)")
    findings = [findings[i] for i in keep]
    texts = [texts[i] for i in keep]

    print(f"taxonomy | {len(findings)} findings from {args.infile} "
          f"({'all candidates' if args.include_refuted else 'panel survivors only'})")
    if len(findings) < args.min_cluster_size * 2:
        print(f"  ! pool of {len(findings)} is smaller than 2x min_cluster_size "
              f"({args.min_cluster_size}) - HDBSCAN will likely return everything as noise. "
              "Lower --min-cluster-size or wait for more findings.")
    emb = embed(texts, no_cache=args.no_cache)

    if args.sweep:
        sweep(emb, findings, args)
        return

    labels = cluster_pool(emb, args.n_neighbors, args.min_cluster_size, args.min_samples)
    modes = propose_modes(emb, labels, args.min_cluster_size)
    n_noise = int(np.sum(labels == -1))
    print(f"clusters formed: {len(modes)} | noise (unassigned): {n_noise}/{len(findings)}")
    if not modes:
        print("no clusters formed - nothing to label. Run --sweep and pick a working config.")
        return
    T.confirm(f"buy {len(modes)} cluster-label calls?", args.yes)

    params = T.manifest_params(
        {"stage": "cluster", "in": args.infile, "n_neighbors": args.n_neighbors,
         "min_cluster_size": args.min_cluster_size, "min_samples": args.min_samples,
         "umap_seed": RANDOM_STATE, "embed_model": EMBED_MODEL,
         "only_verified": not args.include_refuted, "n_findings": len(findings),
         "n_clusters": len(modes), "n_noise": n_noise},
        args.route, [])
    scaffold = []
    with Run(T.EXPERIMENT, params=params, inputs=T.run_inputs([args.infile]), spec=T.SPEC,
             allow_dirty=args.allow_dirty, spend=T.spend_label(args.route)) as run:
        run.register_prompts({"cluster_label": LABEL_PROMPT})
        usage = T.new_usage()
        menu = frame_menu()
        for m in modes:
            members = [findings[i] for i in m["member_indices"]]
            samples = [mm.get("description", "") for mm in members][:20]
            obj, meta = T.route_call(
                LABEL_PROMPT.format(frame=menu, samples="\n".join(f"- {s}" for s in samples)),
                f"label|{m['cluster_id']}|{args.n_neighbors}|{args.min_cluster_size}",
                args.route, kind="label")
            T.usage_add(usage, meta)
            name = (obj or {}).get("label", "Unlabelled") if isinstance(obj, dict) else "Unlabelled"
            desc = (obj or {}).get("description", "") if isinstance(obj, dict) else ""
            choice = (obj or {}).get("frame_category") if isinstance(obj, dict) else None
            t2, t1, how = place_cluster(members, choice)
            scaffold.append({
                "label": name, "description": desc, "size": len(members),
                "frame_tier2": t2, "frame_tier1": t1, "frame_placed_by": how,
                "frame_label_choice": choice,
                "frame_reason": (obj or {}).get("frame_reason") if isinstance(obj, dict) else None,
                "novel_unmapped": t2 is None,
                "high_salience": sum(mm.get("salience") == "high" for mm in members),
                "critical_rubric": sum(mm.get("severity_rubric") == "critical" for mm in members),
                "modes_inside": dict(Counter(mm.get("mode") for mm in members).most_common(10)),
                "vendors": dict(Counter(mm.get("scribe") for mm in members)),
                "substrates": dict(Counter(mm.get("source") for mm in members)),
                "consultations": len({mm.get("consultation") for mm in members}),
                "examples": [{"consult": f"{mm.get('consultation')}/{mm.get('scribe')}/{mm.get('template')}",
                              "description": mm.get("description"), "salience": mm.get("salience"),
                              "severity_rubric": mm.get("severity_rubric")} for mm in members[:3]],
                "member_finding_ids": [findings[i].get("finding_id") for i in m["member_indices"]],
            })
        scaffold.sort(key=lambda c: -c["size"])
        run.save("cluster_labels.json", {"params": params, "scaffold": scaffold})

    out = {"stage": "cluster", "in": args.infile,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "params": params, "n_findings": len(findings), "n_clusters": len(modes),
           "n_noise": n_noise, "scaffold": scaffold,
           "assignments": {findings[i].get("finding_id"): int(labels[i]) for i in range(len(findings))}}
    jpath = os.path.join(T.MASTER, f"{args.out_prefix}_clusters.json")
    json.dump(out, open(jpath, "w"), indent=1)

    novel = [c for c in scaffold if c["novel_unmapped"]]
    placed = [c for c in scaffold if not c["novel_unmapped"]]
    fr = T.load_frame()
    L = ["# Failure taxonomy: subcategories under the published frame", "",
         f"Frame `{T.FRAME_FILE}` v{fr['frame_version']} (`{T.frame_sha256()[:12]}`). "
         f"Clustering recipe: Cohere embed-v4.0 -> UMAP -> HDBSCAN "
         f"(n_neighbors={args.n_neighbors}, min_cluster_size={args.min_cluster_size}, "
         f"min_samples={args.min_samples}), run on {len(findings)} "
         f"{'candidate' if args.include_refuted else 'verified'} scribe failures across "
         f"{len({f.get('scribe') for f in findings})} vendors. "
         f"**{len(modes)} subcategories formed** - {len(placed)} placed under a frame "
         f"category, {len(novel)} that no frame category fits - and {n_noise} findings "
         "stayed unassigned (HDBSCAN noise).", "",
         "The top-level categories are fixed by the published frame and are not up for "
         "rediscovery. What emerges from the data is the level below: the specific, "
         "recurring shapes each published category takes in these notes.", ""]

    def block(c):
        out = [f"### {c['label']}  (n={c['size']}, {c['high_salience']} high-salience, "
               f"{c['critical_rubric']} rubric-critical)", "", c["description"], "",
               f"- placed by: {c['frame_placed_by']}"
               + (f" - {c['frame_reason']}" if c.get("frame_reason") else ""),
               f"- vendors: {c['vendors']}", f"- substrates: {c['substrates']}",
               f"- distinct consultations: {c['consultations']}",
               f"- discovery passes inside: {c['modes_inside']}", "- examples:"]
        for e in c["examples"]:
            out.append(f"  - [{e['consult']}] ({e['salience']}/{e['severity_rubric']}) "
                       f"{e['description']}")
        return out + [""]

    for t1 in [t["key"] for t in fr["tier1"]]:
        here = [c for c in placed if c["frame_tier1"] == t1]
        if not here:
            continue
        t1def = next(t for t in fr["tier1"] if t["key"] == t1)
        L += [f"## {t1def['name']} ({t1}) - {sum(c['size'] for c in here)} findings in "
              f"{len(here)} subcategories", "", f"*{t1def['our_operational_definition']}*", ""]
        for t2 in sorted({c["frame_tier2"] for c in here}):
            sub = [c for c in here if c["frame_tier2"] == t2]
            L += [f"**{t2}** - {sum(c['size'] for c in sub)} findings", ""]
            for c in sub:
                L += block(c)
    if novel:
        L += ["## Novel - no frame category fits", "",
              "These clusters could not be placed under any published category, by member "
              "vote or by the labelling call. They are the frame's own blind spots and "
              "belong in the write-up as such, not folded into a neighbouring row.", ""]
        for c in novel:
            L += block(c)
    mpath = os.path.join(T.MASTER, f"{args.out_prefix}_taxonomy_scaffold.md")
    open(mpath, "w").write("\n".join(L))

    print("\n=== subcategories under the frame ===")
    for c in scaffold:
        tag = c["frame_tier2"] or "NOVEL"
        print(f"  [{tag:22}] {c['label'][:42]:44} n={c['size']:4}  "
              f"high-sal={c['high_salience']:3}  vendors={sorted(c['vendors'])}")
    if novel:
        print(f"  ! {len(novel)} clusters are novel-unmapped - report them, do not absorb them")
    print(f"\nsaved -> {os.path.relpath(jpath, HERE)} + {os.path.relpath(mpath, HERE)}")


if __name__ == "__main__":
    main()
