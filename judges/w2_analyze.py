"""The judge benchmark's shared metrics, intervals and handoff. Every read-out uses these.

Reads the record stores under results/w2-{ablation,v14,baselines,strong}/_state/,
validates every contributing run_id against its run manifest, and emits:

  w2_results.json    every reported number with CIs, PLUS the two fields the later
                     stages consume by name:
                       anchor_selection  - per-cell mean omission detection + FA rate,
                                           the exact input to the rule that picks the
                                           anchor judge design
                       youden_thresholds - per-score-cell Youden's J optimal threshold,
                                           the single pre-registered source of the frozen
                                           threshold the external-anchor runs and the
                                           later stages reuse
  w2_surface.json    THE factorial surface: per arm, detection by residual level
                     (complete / partial / the add + change controls) x severity level
                     (critical / supporting / peripheral), plus detection against
                     surviving-site strength and count, plus the clean-note row
  w2_surface.tsv     the same surface in long format - one row per (arm, residual,
                     severity) - so a plotting script needs no reshaping
  w2_table_main.tex  the main results table

THE TWO ANALYSES, KEPT APART. The pre-registered confirmatory reading is computed on the
add/change/omit-COMPLETE subset only - the design the readings were written against - and
reported as `reading`. The factorial surface is computed over everything and reported as
the exploratory main contribution. Mixing them would let the new partial class quietly
move a pre-registered number.

Statistics standard (the same in every stage): Wilson 95% CIs on every proportion,
exact McNemar on discordant pairs, cluster bootstrap resampled at CONSULTATION level,
Holm-Bonferroni within each pre-registered family, every number mean +/- sd over runs,
primary inference on the per-item majority-over-runs outcome.

Nothing here fires an API call - analysis is re-runnable offline from results/ alone.

    python3 judges/w2_analyze.py                 # the real runs
    python3 judges/w2_analyze.py --smoke         # smoke stores only
    python3 judges/w2_analyze.py --smoke --cost-report   # + measured cost model
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import argparse, glob, json, math, os, random
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

from common import HERE, RESULTS
import check_manifests
import w2_common as W

EXPERIMENTS = {"w2-ablation": "grid", "w2-v14": "v14", "w2-baselines": "baselines",
               "w2-strong": "strong"}
GRID_CELLS = W.CELLS
# The surface's axes. The residual axis is ordered by how much of the fact SURVIVES -
# complete removal, then a weak fragment, then a fully recoverable mention - which is the
# order the figure is plotted along and the direction the effect is predicted to run.
# omission_factorial.py builds complete / partial-strong / partial-weak; "partial" is the
# ungraded label older artefacts carry. add/change are the non-omission controls, and the
# clean notes are the intact condition, reported as false alarms rather than detection.
RESIDUAL_LEVELS = W.RESIDUAL_LEVELS          # ordered; extras seen in data are appended
CONTROL_LEVELS = W.CONTROL_LEVELS
SEVERITY_LEVELS = ["critical", "supporting", "peripheral", "ungraded"]
STRENGTHS = ["explicit", "paraphrase", "partial"]
# The pre-registered confirmatory design: everything the frozen 281 contained.
CONFIRMATORY_LEVELS = ["complete"] + CONTROL_LEVELS


def order_levels(levels):
    """Canonical order first (residual axis, then controls), then anything unrecognised."""
    known = RESIDUAL_LEVELS + CONTROL_LEVELS
    seen = [l for l in known if l in levels]
    return seen + sorted(l for l in levels if l not in known)


def is_omission_level(level):
    return level not in CONTROL_LEVELS and level != "unknown"
# Family A/B contrasts C1-C5 (C6 is the severity correlation).
CONTRASTS = [("C1", "F-score-k8", "FC-score-k8", "criterion effect at the recipe config"),
             ("C2", "F-bin-k1", "FC-bin-k1", "criterion effect, simplest config"),
             ("C3", "FC-bin-k1", "FC-score-k1", "format effect"),
             ("C4", "FC-score-k1", "FC-score-k8", "ensemble effect"),
             ("C5", "FC-bin-k1", "FC-score-k8", "one prompt line vs the full recipe")]


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=1.959963984540054):
    if not n:
        return {"k": k, "n": n, "p": None, "lo": None, "hi": None}
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": k, "n": n, "p": round(p, 6), "lo": round(max(0.0, c - h), 6),
            "hi": round(min(1.0, c + h), 6)}


def mcnemar_exact(a_out, b_out):
    """Exact McNemar over items both arms scored. a_out/b_out: {item: bool}."""
    keys = sorted(set(a_out) & set(b_out))
    b = sum(1 for k in keys if a_out[k] and not b_out[k])
    c = sum(1 for k in keys if b_out[k] and not a_out[k])
    n = b + c
    p = 1.0 if n == 0 else float(stats.binomtest(b, n, 0.5).pvalue)
    return {"n_items": len(keys), "discordant_a_only": b, "discordant_b_only": c,
            "p_exact": round(p, 8),
            "a_rate": round(sum(a_out[k] for k in keys) / len(keys), 6) if keys else None,
            "b_rate": round(sum(b_out[k] for k in keys) / len(keys), 6) if keys else None}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, input {name: p} -> {name: p_adj}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[name] = round(running, 8)
    return out


def cluster_bootstrap(clusters, stat, draws, seed=20260810):
    """Percentile CI for `stat` under resampling of CONSULTATIONS with replacement.

    clusters: {consultation: payload}; stat: list-of-payloads -> float or None.
    """
    keys = sorted(clusters)
    if len(keys) < 2:
        return {"lo": None, "hi": None, "draws": 0, "n_clusters": len(keys)}
    rng = random.Random(seed)
    vals = []
    for _ in range(draws):
        sample = [clusters[keys[rng.randrange(len(keys))]] for _ in keys]
        v = stat(sample)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if len(vals) < 20:
        return {"lo": None, "hi": None, "draws": len(vals), "n_clusters": len(keys)}
    a = np.array(vals)
    return {"lo": round(float(np.percentile(a, 2.5)), 6),
            "hi": round(float(np.percentile(a, 97.5)), 6),
            "draws": len(vals), "n_clusters": len(keys)}


def auc_lower_is_positive(pos, neg):
    """P(pos < neg) + 0.5 P(tie): the errored-scores-lower discrimination AUC."""
    if not pos or not neg:
        return None
    p, n = np.asarray(pos, float), np.asarray(neg, float)
    less = float((p[:, None] < n[None, :]).sum())
    ties = float((p[:, None] == n[None, :]).sum())
    return round((less + 0.5 * ties) / (len(p) * len(n)), 6)


def msd(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return {"mean": None, "sd": None, "n_runs": 0, "per_run": vals}
    return {"mean": round(float(np.mean(v)), 6),
            "sd": round(float(np.std(v, ddof=1)), 6) if len(v) > 1 else 0.0,
            "n_runs": len(v), "per_run": [None if x is None else round(float(x), 6) for x in vals]}


# ---------------------------------------------------------------- loading
def load_records(smoke=False, tag_filter=None):
    recs, stores, run_ids = [], [], set()
    for exp in EXPERIMENTS:
        for path in sorted(glob.glob(os.path.join(RESULTS, exp, "_state", "*.jsonl"))):
            tag = os.path.basename(path)[:-6]
            is_smoke = tag.endswith("-smoke")
            if is_smoke != smoke:
                continue
            if tag_filter and tag_filter not in tag:
                continue
            n = 0
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    r["experiment"], r["tag"] = exp, tag
                    recs.append(r)
                    run_ids.add((exp, r.get("run_id")))
                    n += 1
            stores.append({"experiment": exp, "tag": tag, "path": os.path.relpath(path, HERE),
                           "n_records": n})
    # Supersession, across stores as well as within one. The identity of a judgement is
    # (arm, model, replicate, note) - NOT the store tag - so a re-run under a fresh tag
    # replaces the older record instead of being counted beside it. Two stores that hold
    # the same arm at the same model and replicate are the same measurement twice; the
    # newest wins by wall clock. model_key is in the key because the generality rows run
    # the same cells on a different judge and must stay separate.
    dedup, dropped = {}, Counter()
    for r in recs:
        key = (r.get("cell"), r.get("model_key"), r.get("replicate"), r.get("note_key"))
        prev = dedup.get(key)
        if prev is None:
            dedup[key] = r
            continue
        newer = r if (r.get("t") or 0) >= (prev.get("t") or 0) else prev
        dropped[(prev if newer is r else r)["tag"]] += 1
        dedup[key] = newer
    if dropped:
        print(f"  {sum(dropped.values())} superseded record(s) dropped (same "
              "arm+model+replicate+note in more than one store; newest kept): "
              + ", ".join(f"{t} {n}" for t, n in dropped.most_common()))
    for s in stores:
        s["superseded_dropped"] = dropped.get(s["tag"], 0)
    return list(dedup.values()), stores, run_ids


def validate_runs(run_ids):
    """Validate every contributing manifest, with ONE deliberate relaxation.

    check_manifests still enforces the clean-tree rule, but the judge runs dropped it:
    unrelated work writes this tree continuously, so a judge run discloses its commit
    plus the exact dirty paths instead of refusing to start. A dirty manifest
    that DOES disclose its paths is therefore a warning here, not a problem. Everything
    else check_manifests says - missing manifests, input-hash drift, call errors,
    incomplete status - is still fatal.
    """
    problems, warnings, seen = [], [], 0
    for exp, rid in sorted(x for x in run_ids if x[1]):
        path = os.path.join(RESULTS, exp, rid, "manifest.json")
        if not os.path.exists(path):
            problems.append(f"{exp}/{rid}: manifest.json missing")
            continue
        seen += 1
        raw = []
        m = check_manifests.validate_manifest(path, raw, warnings)
        disclosed = bool((m or {}).get("git_dirty_paths"))
        for p in raw:
            if "git_dirty is" in p and disclosed:
                warnings.append(p + " [dirty tree allowed for judge runs, paths disclosed]")
            else:
                problems.append(p)
    return {"runs_checked": seen, "problems": problems, "warnings": warnings,
            "ok": not problems,
            "note": "clean-tree check relaxed for judge runs that disclose git_dirty_paths"}


# ---------------------------------------------------------------- per-cell views
def residual_level_of(rec):
    """The surface's row for one errored record.

    Takes whatever level the runner recorded - the residual classes are the dataset's
    to name, so an unfamiliar one gets its own row rather than being folded into a
    familiar one. Falls back to the error type only for records written before the
    factorial existed, where every omission was complete by construction (that is what
    surviving the ground-truth gate's truly_absent check meant).
    """
    lvl = (rec.get("residual_level") or "").strip().lower()
    if lvl == "partial":
        # The bare "partial" is the ungraded label: records written before the
        # 2026-08-12 classifier fix carry it because the runner read the coarse class
        # ("omit-partial") ahead of the dataset's graded residual_level, flattening the
        # surface's most informative axis. Every such record still carries the surviving
        # mention's strength, and the dataset graded weak-vs-strong from exactly that,
        # so recover it here rather than reporting a two-row surface. Verified against
        # dataset_v2: this reproduces all 148 graded partials.
        return W.STRENGTH_TO_LEVEL.get(
            (rec.get("residual_strength") or "").strip().lower(), "partial")
    if lvl:
        return lvl
    t = rec.get("pair_type")
    return "complete" if t == "omit" else (t if t in CONTROL_LEVELS else "unknown")


def severity_of(rec):
    s = (rec.get("severity") or rec.get("importance") or "").strip().lower()
    return s if s in SEVERITY_LEVELS else "ungraded"


def cell_views(records):
    """{cell: {replicate: {"pairs": [...], "cleans": [...]}}}"""
    by = defaultdict(lambda: defaultdict(lambda: {"clean": {}, "err": []}))
    for r in records:
        slot = by[r["cell"]][r["replicate"]]
        if r["note_role"] == "clean":
            slot["clean"][r["note_key"]] = r
        else:
            slot["err"].append(r)

    out = {}
    for cell, reps in by.items():
        out[cell] = {}
        for rep, slot in reps.items():
            fmt = next(iter(slot["err"] + list(slot["clean"].values())))["format"]
            pairs, cleans = [], []
            for ck, cr in sorted(slot["clean"].items()):
                cleans.append({"clean_key": ck, "consultation": cr["consultation"],
                               "stratum": cr["stratum"], "aggregate": cr["aggregate"],
                               "flagged": bool(cr.get("flagged")),
                               "parse_failure": bool(cr.get("parse_failure"))})
            for er in sorted(slot["err"], key=lambda r: r["pair_id"] or r["note_key"]):
                cr = slot["clean"].get(er["clean_key"])
                if cr is None:
                    continue
                ea, ca = er["aggregate"], cr["aggregate"]
                pf = bool(er.get("parse_failure") or cr.get("parse_failure"))
                if pf:
                    disc, gap = False, None  # parse_failure = non-discrimination (conservative)
                elif fmt == "score":
                    disc, gap = (ea < ca), (ca - ea)
                else:
                    disc, gap = (er["verdict"] == "FAIL" and cr["verdict"] == "PASS"), (ca - ea)
                pairs.append({"pair_id": er["pair_id"], "consultation": er["consultation"],
                              "stratum": er["stratum"], "type": er["pair_type"],
                              "importance": er.get("importance"), "disc": disc, "gap": gap,
                              "err_agg": ea, "clean_agg": ca, "tie": (ea is not None and ca is not None
                                                                      and ea == ca),
                              "parse_failure": pf,
                              # factorial coordinates (w2_common.factorial_of)
                              "residual_level": residual_level_of(er),
                              "severity": severity_of(er),
                              "residual_strength": er.get("residual_strength"),
                              "residual_n": er.get("residual_n_surviving"),
                              "matched_key": er.get("matched_key"),
                              "pair_class": er.get("pair_class"),
                              "split": er.get("split") or "eval"})
            out[cell][rep] = {"format": fmt, "k": next(iter(slot["err"] + list(slot["clean"].values())))["k"],
                              "pairs": pairs, "cleans": cleans}
    return out


def majority(values):
    v = [bool(x) for x in values]
    return sum(v) * 2 > len(v) if v else False


def cell_metrics(view, boot, boot_auc):
    """Every reported metric for one cell, over its replicates."""
    reps = sorted(view)
    fmt = view[reps[0]]["format"]

    # ---- per-replicate rates
    def rate(rep, pred, pool="pairs"):
        rows = view[rep][pool]
        rows = [r for r in rows if pred(r)]
        return rows

    per_run = {"overall": [], "by_type": defaultdict(list), "fa": [],
               "by_importance": defaultdict(list), "tie": [], "parse_failure": [],
               "auc": [], "pairwise_auc": [], "gap_by_type": defaultdict(list),
               "spearman": [], "by_stratum": defaultdict(list)}
    for rep in reps:
        P, C = view[rep]["pairs"], view[rep]["cleans"]
        per_run["overall"].append(sum(p["disc"] for p in P) / len(P) if P else None)
        per_run["tie"].append(sum(p["tie"] for p in P) / len(P) if P else None)
        per_run["parse_failure"].append(sum(p["parse_failure"] for p in P) / len(P) if P else None)
        per_run["fa"].append(sum(c["flagged"] for c in C) / len(C) if C else None)
        for t in ("add", "change", "omit"):
            sub = [p for p in P if p["type"] == t]
            per_run["by_type"][t].append(sum(p["disc"] for p in sub) / len(sub) if sub else None)
            gaps = [p["gap"] for p in sub if p["gap"] is not None]
            per_run["gap_by_type"][t].append(float(np.mean(gaps)) if gaps else None)
        for st in sorted({p["stratum"] for p in P}):
            sub = [p for p in P if p["stratum"] == st]
            per_run["by_stratum"][st].append(sum(p["disc"] for p in sub) / len(sub) if sub else None)
        for imp in ("critical", "supporting", "peripheral"):
            sub = [p for p in P if p["type"] == "omit" and p["importance"] == imp]
            per_run["by_importance"][imp].append(
                sum(p["disc"] for p in sub) / len(sub) if sub else None)
        # AUC (score-like cells only): errored vs clean, lower = more errored
        if fmt == "score":
            pos = [p["err_agg"] for p in P if p["err_agg"] is not None]
            neg = [c["aggregate"] for c in C if c["aggregate"] is not None]
            per_run["auc"].append(auc_lower_is_positive(pos, neg))
            mp = [(p["err_agg"], p["clean_agg"]) for p in P
                  if p["err_agg"] is not None and p["clean_agg"] is not None]
            per_run["pairwise_auc"].append(
                round(sum((1.0 if e < c else 0.5 if e == c else 0.0) for e, c in mp) / len(mp), 6)
                if mp else None)
            om = [(W.IMPORTANCE_ORDINAL.get(p["importance"], 2), p["gap"]) for p in P
                  if p["type"] == "omit" and p["gap"] is not None]
            if len(om) >= 3 and len({x for x, _ in om}) > 1:
                rho = stats.spearmanr([x for x, _ in om], [g for _, g in om])
                per_run["spearman"].append({"rho": float(rho.statistic), "p": float(rho.pvalue),
                                            "n": len(om)})
            else:
                per_run["spearman"].append(None)
        else:
            per_run["auc"].append(None)
            per_run["pairwise_auc"].append(None)
            per_run["spearman"].append(None)

    # ---- majority-over-runs (the primary inference surface)
    disc_by_pair, meta_by_pair = defaultdict(list), {}
    for rep in reps:
        for p in view[rep]["pairs"]:
            disc_by_pair[p["pair_id"]].append(p["disc"])
            meta_by_pair[p["pair_id"]] = p
    maj = {pid: majority(v) for pid, v in disc_by_pair.items()}
    fa_by_clean = defaultdict(list)
    clean_meta = {}
    for rep in reps:
        for c in view[rep]["cleans"]:
            fa_by_clean[c["clean_key"]].append(c["flagged"])
            clean_meta[c["clean_key"]] = c
    maj_fa = {ck: majority(v) for ck, v in fa_by_clean.items()}

    def wilson_over(sel):
        ks = [pid for pid in maj if sel(meta_by_pair[pid])]
        return wilson(sum(maj[p] for p in ks), len(ks))

    clusters = defaultdict(list)
    for pid, m in maj.items():
        clusters[meta_by_pair[pid]["consultation"]].append((meta_by_pair[pid], m))
    fa_clusters = defaultdict(list)
    for ck, m in maj_fa.items():
        fa_clusters[clean_meta[ck]["consultation"]].append(m)

    def boot_rate(sel):
        def stat(sample):
            rows = [m for grp in sample for meta, m in grp if sel(meta)]
            return (sum(rows) / len(rows)) if rows else None
        return cluster_bootstrap(clusters, stat, boot)

    out = {
        "format": fmt, "k": view[reps[0]]["k"], "replicates": reps,
        "n_pairs": len(maj), "n_clean_notes": len(maj_fa),
        "discrimination": {
            "overall": {"mean_sd": msd(per_run["overall"]), "majority": wilson_over(lambda m: True),
                        "cluster_boot": boot_rate(lambda m: True)},
            "by_type": {t: {"mean_sd": msd(per_run["by_type"][t]),
                            "majority": wilson_over(lambda m, t=t: m["type"] == t),
                            "cluster_boot": boot_rate(lambda m, t=t: m["type"] == t)}
                        for t in ("add", "change", "omit")},
            "by_stratum": {st: {"mean_sd": msd(v)} for st, v in per_run["by_stratum"].items()},
        },
        "OD_cell": {"mean_sd": msd(per_run["by_type"]["omit"]),
                    "majority": wilson_over(lambda m: m["type"] == "omit"),
                    "cluster_boot": boot_rate(lambda m: m["type"] == "omit")},
        "OD_by_importance": {
            imp: {"mean_sd": msd(per_run["by_importance"][imp]),
                  "majority": wilson_over(lambda m, i=imp: m["type"] == "omit"
                                          and m["importance"] == i)}
            for imp in ("critical", "supporting", "peripheral")},
        "false_alarm": {
            "mean_sd": msd(per_run["fa"]),
            "majority": wilson(sum(maj_fa.values()), len(maj_fa)),
            "cluster_boot": cluster_bootstrap(
                fa_clusters, lambda s: (lambda r: sum(r) / len(r) if r else None)(
                    [m for g in s for m in g]), boot)},
        "tie_rate": msd(per_run["tie"]),
        "parse_failure_rate": msd(per_run["parse_failure"]),
        "auc_pooled": msd(per_run["auc"]),
        "auc_pairwise": msd(per_run["pairwise_auc"]),
        "score_gap_by_type": {t: msd(per_run["gap_by_type"][t]) for t in ("add", "change", "omit")},
        "_majority": maj, "_majority_fa": maj_fa, "_meta": meta_by_pair,
    }
    if fmt == "score":
        sp = [s for s in per_run["spearman"] if s]
        out["severity_spearman"] = {
            "rho": msd([s["rho"] for s in sp]) if sp else msd([]),
            "p_per_run": [None if s is None else round(s["p"], 8) for s in per_run["spearman"]],
            "n_omit_pairs_used": sp[0]["n"] if sp else 0,
            "hypothesis": "higher importance -> larger score gap (directional, pre-registered)"}
        # cluster-bootstrap CI on the pooled (mean-over-runs) gap/importance correlation
        gap_pool = defaultdict(list)
        for pid, m in meta_by_pair.items():
            if m["type"] == "omit" and m["gap"] is not None:
                gap_pool[m["consultation"]].append(
                    (W.IMPORTANCE_ORDINAL.get(m["importance"], 2), m["gap"]))

        def rho_stat(sample):
            rows = [x for g in sample for x in g]
            if len(rows) < 3 or len({a for a, _ in rows}) < 2:
                return None
            r = stats.spearmanr([a for a, _ in rows], [b for _, b in rows]).statistic
            return None if (r is None or math.isnan(r)) else float(r)
        out["severity_spearman"]["cluster_boot"] = cluster_bootstrap(gap_pool, rho_stat, boot_auc)
    return out


# ------------------------------------------------------- the factorial surface
def majority_view(view):
    """(maj, meta, maj_fa, clean_meta) - the per-item majority-over-runs outcomes the
    surface and the matched contrast both read. Same primary-inference surface as
    cell_metrics uses, factored out so the two cannot drift apart."""
    disc_by_pair, meta_by_pair = defaultdict(list), {}
    fa_by_clean, clean_meta = defaultdict(list), {}
    for rep in sorted(view):
        for p in view[rep]["pairs"]:
            disc_by_pair[p["pair_id"]].append(p["disc"])
            meta_by_pair[p["pair_id"]] = p
        for c in view[rep]["cleans"]:
            fa_by_clean[c["clean_key"]].append(c["flagged"])
            clean_meta[c["clean_key"]] = c
    return ({pid: majority(v) for pid, v in disc_by_pair.items()}, meta_by_pair,
            {ck: majority(v) for ck, v in fa_by_clean.items()}, clean_meta)


def _row(view, maj, meta, sel, boot, clusters, label):
    """One surface cell: n, per-run mean +- sd, majority Wilson, cluster-bootstrap CI."""
    ids = [pid for pid in maj if sel(meta[pid])]
    if not ids:
        return None
    per_run = []
    for rep in sorted(view):
        sub = [p for p in view[rep]["pairs"] if sel(p)]
        per_run.append(sum(p["disc"] for p in sub) / len(sub) if sub else None)

    def stat(sample):
        rows = [m for grp in sample for mt, m in grp if sel(mt)]
        return (sum(rows) / len(rows)) if rows else None
    out = dict(label)
    out.update({"n_pairs": len(ids), "n_consultations": len({meta[p]["consultation"] for p in ids}),
                "detection": msd(per_run),
                "majority": wilson(sum(maj[p] for p in ids), len(ids)),
                "cluster_boot": cluster_bootstrap(clusters, stat, boot)})
    return out


def surface(view, boot):
    """Detection by residual level x severity level, for one arm - the paper's figure.

    Rows: complete / partial omissions and the add + change controls, crossed with
    critical / supporting / peripheral (plus an "all" column and an "any omission"
    row). The intact condition - nothing removed - cannot show detection, so it is
    reported as this arm's clean-note false-alarm rate, which is the same quantity
    read from the other side.
    """
    maj, meta, maj_fa, clean_meta = majority_view(view)
    clusters = defaultdict(list)
    for pid, m in maj.items():
        clusters[meta[pid]["consultation"]].append((meta[pid], m))
    fa_clusters = defaultdict(list)
    for ck, m in maj_fa.items():
        fa_clusters[clean_meta[ck]["consultation"]].append(m)

    present = order_levels({p["residual_level"] for p in meta.values()})
    rows = []
    levels = [(lvl, lambda p, l=lvl: p["residual_level"] == l) for lvl in present]
    levels.append(("omission-any", lambda p: is_omission_level(p["residual_level"])))
    levels.append(("all", lambda p: True))
    for lvl, lsel in levels:
        for sev in SEVERITY_LEVELS + ["all"]:
            def sel(p, ls=lsel, s=sev):
                return ls(p) and (s == "all" or p["severity"] == s)
            r = _row(view, maj, meta, sel, boot, clusters,
                     {"residual_level": lvl, "severity": sev})
            if r:
                rows.append(r)

    def cell(lvl, sev):
        for r in rows:
            if r["residual_level"] == lvl and r["severity"] == sev:
                return r["detection"]["mean"]
        return None

    def delta(a, b):
        return None if (a is None or b is None) else round(a - b, 6)

    intact = {"condition": "intact (nothing removed) = clean notes",
              "n_clean_notes": len(maj_fa),
              "false_alarm_majority": wilson(sum(maj_fa.values()), len(maj_fa)),
              "false_alarm_cluster_boot": cluster_bootstrap(
                  fa_clusters, lambda s: (lambda r: sum(r) / len(r) if r else None)(
                      [m for g in s for m in g]), boot)}
    partial_levels = [l for l in present if l.startswith("partial")]
    slopes = {"severity_slope": {l: delta(cell(l, "critical"), cell(l, "peripheral"))
                                 for l in present if is_omission_level(l)},
              "residual_drop_vs_complete": {
                  l: {"all": delta(cell("complete", "all"), cell(l, "all")),
                      "critical": delta(cell("complete", "critical"), cell(l, "critical"))}
                  for l in partial_levels},
              "note": "descriptive differences of point estimates, not tests - the tests are "
                      "the matched complete-vs-partial McNemar (family D) and the Wilson/"
                      "bootstrap CIs on each cell. Positive severity slope = the arm has "
                      "taste; positive residual drop = a surviving mention hides the omission."}
    # back-compatible scalars for anything already reading these names
    slopes["severity_slope_complete"] = slopes["severity_slope"].get("complete")
    drops = [v["all"] for v in slopes["residual_drop_vs_complete"].values() if v["all"] is not None]
    slopes["residual_drop_all"] = round(float(np.mean(drops)), 6) if drops else None
    return {"rows": rows, "intact": intact, "slopes": slopes,
            "axes": {"residual_level": present + ["omission-any", "all"],
                     "severity": SEVERITY_LEVELS + ["all"]}}


def strength_view(view, boot):
    """Detection against how much of the fact survived: the strongest surviving mention
    (explicit / paraphrase / partial) and how many mentions survived. Partial pairs only -
    a complete omission has no residual by definition."""
    maj, meta, _, _ = majority_view(view)
    clusters = defaultdict(list)
    for pid, m in maj.items():
        clusters[meta[pid]["consultation"]].append((meta[pid], m))

    def partial(p):
        return p["residual_level"].startswith("partial")
    rows = []
    for s in STRENGTHS + ["unrecorded"]:
        def sel(p, s=s):
            return partial(p) and (p.get("residual_strength") or "unrecorded") == s
        r = _row(view, maj, meta, sel, boot, clusters, {"residual_strength": s})
        if r:
            rows.append(r)
    for lo, hi, label in ((1, 1, "1"), (2, 2, "2"), (3, 99, "3+")):
        def sel(p, lo=lo, hi=hi):
            n = p.get("residual_n")
            return partial(p) and n is not None and lo <= n <= hi
        r = _row(view, maj, meta, sel, boot, clusters, {"n_surviving_sites": label})
        if r:
            rows.append(r)
    return rows


def matched_contrast(view):
    """Exact McNemar on the matched pair-of-pairs: the SAME fact in the SAME note,
    removed completely vs left with a residual. Same severity, same consultation, so the
    residual is the only thing that differs - which is what makes this the sharp test.

    One contrast per partial level present (the factorial builds partial-strong and
    partial-weak), plus the pooled complete-vs-any-partial comparison.
    """
    maj, meta, _, _ = majority_view(view)
    by_key = defaultdict(dict)
    for pid, m in maj.items():
        mt = meta[pid]
        if is_omission_level(mt["residual_level"]) and mt.get("matched_key"):
            by_key[mt["matched_key"]][mt["residual_level"]] = m
    partial_levels = order_levels({l for v in by_key.values() for l in v
                                   if l.startswith("partial")})
    out = {}
    for lvl in partial_levels + (["any-partial"] if len(partial_levels) > 1 else []):
        both = {}
        for k, v in by_key.items():
            if "complete" not in v:
                continue
            if lvl == "any-partial":
                got = [v[l] for l in partial_levels if l in v]
                if got:
                    both[k] = (v["complete"], any(got))
            elif lvl in v:
                both[k] = (v["complete"], v[lvl])
        if not both:
            continue
        out[lvl] = dict(mcnemar_exact({k: a for k, (a, _) in both.items()},
                                      {k: b for k, (_, b) in both.items()}),
                        n_matched_facts=len(both), a="complete", b=lvl,
                        metric=f"detection: complete vs {lvl} removal of the same fact")
    return out or None


def youden(view):
    """Per-score-cell Youden's J optimal threshold, on the mean-over-runs aggregate
    per note. Rule: flag iff aggregate <= threshold (matches this benchmark's score-cell
    flag convention, so the external anchors and the later runs can apply it unchanged)."""
    reps = sorted(view)
    if view[reps[0]]["format"] != "score":
        return None
    err, cln = defaultdict(list), defaultdict(list)
    for rep in reps:
        for p in view[rep]["pairs"]:
            if p["err_agg"] is not None:
                err[p["pair_id"]].append(p["err_agg"])
        for c in view[rep]["cleans"]:
            if c["aggregate"] is not None:
                cln[c["clean_key"]].append(c["aggregate"])
    E = [float(np.mean(v)) for v in err.values()]
    C = [float(np.mean(v)) for v in cln.values()]
    if not E or not C:
        return None
    cands = sorted({round(x, 4) for x in E + C})
    grid = [(cands[0] - 0.5)] + [(a + b) / 2 for a, b in zip(cands, cands[1:])] + [cands[-1] + 0.5]
    best = None
    for t in grid:
        tpr = sum(1 for x in E if x <= t) / len(E)
        fpr = sum(1 for x in C if x <= t) / len(C)
        j = tpr - fpr
        if best is None or j > best["J"] + 1e-12:
            best = {"threshold": round(float(t), 4), "J": round(j, 6),
                    "tpr": round(tpr, 6), "fpr": round(fpr, 6)}

    def per_rep(rep):
        e = [p["err_agg"] for p in view[rep]["pairs"] if p["err_agg"] is not None]
        c = [x["aggregate"] for x in view[rep]["cleans"] if x["aggregate"] is not None]
        if not e or not c:
            return None
        cd = sorted({round(x, 4) for x in e + c})
        g = [(cd[0] - 0.5)] + [(a + b) / 2 for a, b in zip(cd, cd[1:])] + [cd[-1] + 0.5]
        return max(g, key=lambda t: sum(1 for x in e if x <= t) / len(e)
                   - sum(1 for x in c if x <= t) / len(c))

    return dict(best, rule="flag iff aggregate <= threshold",
                source="mean-over-runs aggregate per note",
                n_errored=len(E), n_clean=len(C),
                per_replicate=[None if (v := per_rep(r)) is None else round(float(v), 4)
                               for r in reps])


# ---------------------------------------------------------------- cost model
def cost_model(records, pairs_info=None, master_pairs=None, transcripts=None):
    r"""Measure what the smoke was commissioned to measure, then project the grid.

    Three measured numbers: \$/call by cell family, the k=8 effective input
    multiplier, and the prompt-cache hit rate achieved under pair-major ordering.
    Prices are FITTED from the receipts (least squares on uncached-in / cached-in /
    out tokens) rather than assumed, so the cached-token discount is measured too.
    """
    rows = []
    for r in records:
        for rc in r.get("receipts", []):
            if rc.get("cost") is None:
                continue
            rows.append({"cell": r["cell"], "fmt": r.get("format"), "k": r.get("k", 1),
                         "pt": rc["pt"], "ct": rc["ct"], "cached": rc.get("cached", 0),
                         "cost": rc["cost"], "chars": r.get("prompt_chars"),
                         "tx_chars": r.get("transcript_chars")})
    if not rows:
        return {"error": "no receipts in the loaded records"}

    A = np.array([[r["pt"] - r["cached"], r["cached"], r["ct"]] for r in rows], float) / 1e6
    y = np.array([r["cost"] for r in rows], float)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = float(np.abs(A @ coef - y).mean())
    price = {"uncached_in_per_mtok": round(float(coef[0]), 4),
             "cached_in_per_mtok": round(float(coef[1]), 4),
             "out_per_mtok": round(float(coef[2]), 4),
             "mean_abs_residual_usd": round(resid, 8),
             "n_calls_fitted": len(rows)}
    price["cached_discount"] = (round(1 - coef[1] / coef[0], 4) if coef[0] else None)

    tot_pt = sum(r["pt"] for r in rows)
    cache_hit = round(sum(r["cached"] for r in rows) / tot_pt, 6) if tot_pt else None
    tx_frac = None
    tx = [r for r in rows if r.get("tx_chars") and r.get("chars")]
    if tx:
        tx_frac = round(float(np.mean([r["tx_chars"] / r["chars"] for r in tx])), 6)

    by_cell = {}
    for cell in sorted({r["cell"] for r in rows}):
        sub = [r for r in rows if r["cell"] == cell]
        by_cell[cell] = {
            "calls": len(sub),
            "usd_per_call": round(float(np.mean([r["cost"] for r in sub])), 8),
            "prompt_tokens_mean": round(float(np.mean([r["pt"] for r in sub])), 1),
            "completion_tokens_mean": round(float(np.mean([r["ct"] for r in sub])), 1),
            "cache_hit_rate": round(float(np.sum([r["cached"] for r in sub])
                                          / max(1, np.sum([r["pt"] for r in sub]))), 6)}
    per_judgement = {}
    for r in records:
        if not r.get("totals"):
            continue
        per_judgement.setdefault(r["cell"], []).append(r["totals"]["cost_usd"])
    for cell, v in per_judgement.items():
        by_cell.setdefault(cell, {})["usd_per_judgement"] = round(float(np.mean(v)), 8)

    def fam(fmt, k):
        sub = [r for r in rows if r["fmt"] == fmt and r["k"] == k]
        if not sub:
            return None
        return {"calls": len(sub),
                "usd_per_call": round(float(np.mean([r["cost"] for r in sub])), 8),
                "input_usd_per_call": round(float(np.mean(
                    [(r["pt"] - r["cached"]) / 1e6 * coef[0] + r["cached"] / 1e6 * coef[1]
                     for r in sub])), 8),
                "completion_tokens_mean": round(float(np.mean([r["ct"] for r in sub])), 1),
                "cache_hit_rate": round(float(np.sum([r["cached"] for r in sub])
                                              / max(1, np.sum([r["pt"] for r in sub]))), 6)}

    families = {"score_k1": fam("score", 1), "bin_k1": fam("bin", 1),
                "score_k8": fam("score", 8), "bin_k8": fam("bin", 8)}
    k8_mult = {}
    for f in ("score", "bin"):
        a, b = families.get(f"{f}_k1"), families.get(f"{f}_k8")
        if a and b:
            k8_mult[f] = {
                "nominal_input_multiplier": 8.0,
                "effective_input_multiplier": round(8 * b["input_usd_per_call"]
                                                    / a["input_usd_per_call"], 4),
                "cost_multiplier_per_judgement": round(
                    8 * b["usd_per_call"] / a["usd_per_call"], 4)}

    # ---- the cache mechanism, measured rather than assumed
    # prompt_tokens = a*chars + b, fitted from the receipts; the provider caches the
    # longest common PREFIX in 128-token blocks with a ~1024-token floor. Because every
    # grid prompt puts {transcript} first and the notes of a block differ after it, the
    # prefix that actually gets cached is the transcript. Where the transcript alone is
    # under the floor, NOTHING caches - which the smoke confirmed exactly (a 3.7k-char
    # transcript cached 0/216 calls, a 7.3k-char one cached 208/216 at 1792 tokens).
    ch = [r for r in rows if r.get("chars")]
    tok_a = tok_b = None
    if ch:
        X = np.array([[r["chars"], 1] for r in ch], float)
        tok_a, tok_b = np.linalg.lstsq(X, np.array([r["pt"] for r in ch], float), rcond=None)[0]

    def cache_prefix(transcript_chars):
        """Cached prefix tokens for a transcript of this length (0 below the floor)."""
        t = tok_a * transcript_chars + tok_b
        return float((t // 128) * 128) if t >= 1024 else 0.0

    cache_model = {"prompt_tokens_per_char": round(float(tok_a), 6) if tok_a else None,
                   "prompt_tokens_intercept": round(float(tok_b), 2) if tok_b else None,
                   "chars_per_token": round(1 / float(tok_a), 3) if tok_a else None,
                   "cache_floor_tokens": 1024, "cache_block_tokens": 128,
                   "transcript_chars_needed_to_cache":
                       round((1024 - float(tok_b)) / float(tok_a)) if tok_a else None,
                   "verified_against":
                       [{"transcript_chars": tc,
                         "predicted_prefix": cache_prefix(tc),
                         "observed_max_cached": max(r["cached"] for r in rows
                                                    if r.get("tx_chars") == tc)}
                        for tc in sorted({r["tx_chars"] for r in rows if r.get("tx_chars")})],
                   "conclusion": "ordering keeps same-transcript calls adjacent (necessary), but "
                                 "the achieved rate is set by TRANSCRIPT LENGTH, not by ordering - "
                                 "an A/B of note-major vs block-major on the short transcript moved "
                                 "the hit rate by 0.0pp (both 0%)."}

    out = {"fitted_prices": price, "cache_hit_rate_overall": cache_hit,
           "transcript_share_of_prompt_chars": tx_frac,
           "by_cell": by_cell, "by_family": families, "k8_input_multiplier": k8_mult,
           "cache_model": cache_model,
           "note": "cache_hit_rate = cached_tokens / prompt_tokens over every receipt, under "
                   "note-major sub-blocks with a serial warm-up call"}

    # ---- projection to the full grid, priced per note from the master file itself
    if master_pairs and transcripts and tok_a:
        blocks, notes = W.note_units(master_pairs, transcripts)
        scaffold = 620  # chars of fixed grid scaffold around {transcript}/{note}
        n_reps = 3
        out_tok = {f: (families[f] or {}).get("completion_tokens_mean") for f in families}
        proj, calls, warm_in, cold_in = 0.0, 0, 0.0, 0.0
        cacheable = {"consultations": 0, "notes": 0}
        by_stratum = defaultdict(lambda: {"consultations": 0, "usd": 0.0, "caches": 0})
        for b in blocks:
            prefix = cache_prefix(len(b["transcript"]))
            by_stratum[b["stratum"]]["consultations"] += 1
            by_stratum[b["stratum"]]["caches"] += 1 if prefix else 0
            cacheable["consultations"] += 1 if prefix else 0
            for cell in GRID_CELLS:
                crit, fmt, k = W.parse_cell(cell)
                ot = out_tok.get(f"{fmt}_k{k}") or out_tok.get(f"{fmt}_k1") or 200
                for n in b["notes"]:
                    pt = tok_a * (len(b["transcript"]) + len(n["text"]) + scaffold) + tok_b
                    cached = min(prefix, pt)
                    # one cold call per (consultation, replicate); the rest run warm
                    ncalls = k * n_reps
                    ncold = min(n_reps, ncalls) if cached else ncalls
                    c_warm = ((pt - cached) / 1e6 * coef[0] + cached / 1e6 * coef[1]
                              + ot / 1e6 * coef[2])
                    c_cold = pt / 1e6 * coef[0] + ot / 1e6 * coef[2]
                    proj += c_warm * (ncalls - ncold) + c_cold * ncold
                    calls += ncalls
                    warm_in += cached * (ncalls - ncold)
                    cold_in += pt * ncalls
            cacheable["notes"] += len(b["notes"]) if prefix else 0
        out["projection_full_grid"] = {
            "basis": "8 cells x 3 replicates x every unique note in the master pair file, each "
                     "note priced at its own length with the measured cache model applied "
                     "per consultation",
            "n_pairs": len(master_pairs), "n_consultations": len(blocks),
            "n_unique_notes": len(notes), "calls": calls,
            "consultations_that_will_cache": cacheable["consultations"],
            "consultations_total": len(blocks),
            "usd_projected": round(proj, 2),
            "usd_if_nothing_cached": round(
                proj + (warm_in / 1e6) * (coef[0] - coef[1]), 2),
            "by_stratum": {k: {"consultations": v["consultations"],
                               "will_cache": v["caches"]} for k, v in sorted(by_stratum.items())},
            "note": "an earlier costing assumed 8 x 411 pairs x 2 notes x 3 reps = ~89K calls; "
                    "the pre-registered clean-twin dedup (one clean note per consultation, "
                    "reused across its 3 pairs) makes it "
                    f"{calls:,}."}

        # ---- the remaining judge arms, at measured per-call rates where we have them
        n_notes, n_consults, reps = len(notes), len(blocks), 3
        measured = {}
        for exp in ("w2-v14", "w2-baselines"):
            sub = [r for r in records if r.get("experiment") == exp and r.get("receipts")]
            per_arm = defaultdict(list)
            for r in sub:
                for rc in r["receipts"]:
                    if rc.get("cost") is not None:
                        per_arm[r["cell"]].append(rc["cost"])
            for arm, v in per_arm.items():
                measured[arm] = round(float(np.mean(v)), 8)
        arm_calls = {"v14-asis": n_notes * reps, "v14-noexcl": n_notes * reps,
                     "v14-incl": n_notes * reps,
                     "geval": 2 * n_notes * reps,
                     "ragas": 2 * n_notes * reps + n_consults,
                     "checklist": 1 * n_notes * reps + n_consults,
                     "temp0_supplement": 4 * n_notes * reps}
        other = {}
        gpt_call = float(np.mean([r["cost"] for r in rows]))
        for arm, nc in arm_calls.items():
            rate = measured.get(arm, gpt_call)
            other[arm] = {"calls": nc, "usd_per_call": round(rate, 8),
                          "usd": round(nc * rate, 2),
                          "rate_source": "measured at smoke" if arm in measured
                                         else "grid mean $/call (arm not smoked)"}
        try:  # generality rows: same token profile, lock prices
            lock = json.load(open(os.path.join(HERE, "models.lock.json")))["models"]
            base = lock["judge-primary"]["price_usd_per_mtok"]
            mean_pt = float(np.mean([r["pt"] for r in rows]))
            mean_cached = float(np.mean([r["cached"] for r in rows]))
            mean_ct = float(np.mean([r["ct"] for r in rows]))
            for role, key, nc in (("judge-qwen", "qwen_row", (4 + 8) * n_notes * reps),
                                  ("judge-opus", "opus_row", 4 * n_notes * reps + 8 * n_notes)):
                pr = lock[role]["price_usd_per_mtok"]
                # no cached-price line published per role: scale the measured discount
                disc = coef[1] / coef[0] if coef[0] else 0.1
                per = ((mean_pt - mean_cached) / 1e6 * pr["in"]
                       + mean_cached / 1e6 * pr["in"] * disc + mean_ct / 1e6 * pr["out"])
                other[key] = {"calls": nc, "usd_per_call": round(per, 8),
                              "usd": round(nc * per, 2),
                              "rate_source": f"models.lock price for {role} x the measured "
                                             f"token profile (in {base['in']}->{pr['in']}/Mtok)"}
        except Exception:
            pass
        out["projection_other_arms"] = dict(
            other, total_usd=round(sum(v["usd"] for v in other.values()), 2),
            total_calls=sum(v["calls"] for v in other.values()))
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="judge benchmark analysis: metrics, CIs, downstream handoff")
    ap.add_argument("--smoke", action="store_true", help="read the -smoke record stores")
    ap.add_argument("--tag-filter", default=None)
    ap.add_argument("--boot", type=int, default=10000, help="cluster-bootstrap draws (proportions)")
    ap.add_argument("--boot-auc", type=int, default=2000, help="draws for correlation/AUC CIs")
    ap.add_argument("--boot-surface", type=int, default=2000,
                    help="draws per surface cell (many cells x many arms; Wilson CIs are the "
                         "primary interval there, the bootstrap is the cluster-aware check)")
    ap.add_argument("--out", default=None, help="output path (default w2_results.json)")
    ap.add_argument("--cost-report", action="store_true",
                    help="also measure the cost model and project the full grid")
    ap.add_argument("--allow-invalid-manifests", action="store_true")
    args = ap.parse_args()

    records, stores, run_ids = load_records(smoke=args.smoke, tag_filter=args.tag_filter)
    if not records:
        raise SystemExit("no records found - run w2_grid.py first "
                         f"(looked for {'smoke' if args.smoke else 'non-smoke'} stores)")
    val = validate_runs(run_ids)
    print(f"loaded {len(records)} records from {len(stores)} store(s); "
          f"{val['runs_checked']} run manifests checked, {len(val['problems'])} problem(s)")
    for p in val["problems"][:10]:
        print(f"  ! {p}")
    if val["problems"] and not (args.smoke or args.allow_invalid_manifests):
        raise SystemExit("manifest validation failed - "
                         "fix, or pass --allow-invalid-manifests to inspect anyway")

    views = cell_views(records)
    cells = {c: cell_metrics(v, args.boot, args.boot_auc) for c, v in sorted(views.items())}
    thresholds = {c: t for c, v in sorted(views.items()) if (t := youden(v))}

    # ---- the factorial surface (exploratory main contribution) + its matched test
    surfaces = {c: surface(v, args.boot_surface) for c, v in sorted(views.items())}
    strengths = {c: s for c, v in sorted(views.items()) if (s := strength_view(v, args.boot_surface))}
    famD = {c: m for c, v in sorted(views.items()) if (m := matched_contrast(v))}
    if famD:  # Holm across every (arm, partial level) contrast in the family
        flat = {f"{arm}/{lvl}": m["p_exact"] for arm, lv in famD.items() for lvl, m in lv.items()}
        for name, p in holm(flat).items():
            arm, lvl = name.rsplit("/", 1)
            famD[arm][lvl]["p_holm"] = p

    # ---- the pre-registered confirmatory subset: add + change + COMPLETE omissions only
    def confirmatory(r):
        return r["note_role"] == "clean" or residual_level_of(r) in CONFIRMATORY_LEVELS
    conf_records = [r for r in records if confirmatory(r)]
    subset_differs = len(conf_records) != len(records)
    if subset_differs:
        conf_views = cell_views(conf_records)
        conf_cells = {c: cell_metrics(v, args.boot, args.boot_auc)
                      for c, v in sorted(conf_views.items())}
    else:
        conf_views, conf_cells = views, cells

    # ---- Family A/B: exact McNemar on the majority-over-runs outcomes, Holm within family.
    # Computed on the CONFIRMATORY subset (add/change/complete) - these contrasts were
    # pre-registered against that design, and the partial class must not move them.
    famA, famB, pA, pB = {}, {}, {}, {}
    for name, a, b, why in CONTRASTS:
        if a in conf_cells and b in conf_cells:
            A = {k: v for k, v in conf_cells[a]["_majority"].items()
                 if conf_cells[a]["_meta"][k]["type"] == "omit"}
            B = {k: v for k, v in conf_cells[b]["_majority"].items()
                 if conf_cells[b]["_meta"][k]["type"] == "omit"}
            famA[name] = dict(mcnemar_exact(A, B), a=a, b=b, why=why, metric="omission detection")
            pA[name] = famA[name]["p_exact"]
            famB[name] = dict(mcnemar_exact(conf_cells[a]["_majority_fa"],
                                            conf_cells[b]["_majority_fa"]),
                              a=a, b=b, why=why, metric="clean-note false alarm")
            pB[name] = famB[name]["p_exact"]
    # C6: the severity/graded-salience correlation joins Family A's Holm adjustment
    for cell in ("FC-score-k8", "FC-score-k1", "F-score-k8", "F-score-k1"):
        sp = conf_cells.get(cell, {}).get("severity_spearman")
        if sp and sp["p_per_run"] and any(p is not None for p in sp["p_per_run"]):
            ps = [p for p in sp["p_per_run"] if p is not None]
            famA[f"C6/{cell}"] = {"metric": "severity/gap Spearman", "cell": cell,
                                  "rho_mean_sd": sp["rho"], "p_per_run": sp["p_per_run"],
                                  "p_exact": round(float(np.mean(ps)), 8),
                                  "why": "does the score track how much the omission matters"}
            pA[f"C6/{cell}"] = famA[f"C6/{cell}"]["p_exact"]
    for name, p in holm(pA).items():
        famA[name]["p_holm"] = p
    for name, p in holm(pB).items():
        famB[name]["p_holm"] = p

    # ---- Family C: the grid against the field (v14, baselines, strong systems)
    def omit_maj(cell):
        return {k: v for k, v in conf_cells[cell]["_majority"].items()
                if conf_cells[cell]["_meta"][k]["type"] == "omit"}

    famC = {}
    if "v14-asis" in conf_cells and "v14-noexcl" in conf_cells:
        famC["v14-asis_vs_v14-noexcl"] = dict(
            mcnemar_exact(omit_maj("v14-asis"), omit_maj("v14-noexcl")),
            metric="omission detection")
    grid_present = [c for c in GRID_CELLS if c in conf_cells]
    if grid_present:
        best = max(grid_present,
                   key=lambda c: (conf_cells[c]["OD_cell"]["majority"]["p"] if
                                  conf_cells[c]["OD_cell"]["majority"]["p"] is not None else -1.0))
        for bl in ("geval", "ragas", "checklist", "engineered-completeness",
                   "engineered-completeness-k8", "gepa-optimized"):
            if bl in conf_cells:
                famC[f"{best}_vs_{bl}"] = dict(
                    mcnemar_exact(omit_maj(best), omit_maj(bl)), metric="omission detection",
                    note=("strong system reported alongside the ablation, not inside it"
                          if bl not in ("geval", "ragas", "checklist") else "published baseline"))
    if famC:
        for name, p in holm({k: v["p_exact"] for k, v in famC.items()}).items():
            famC[name]["p_holm"] = p

    # ---- anchor_selection (the later stages consume this by name), on the confirmatory subset
    anchor_cells = {}
    for c in grid_present:
        anchor_cells[c] = {
            "od_omit_mean": conf_cells[c]["OD_cell"]["mean_sd"]["mean"],
            "od_omit_sd": conf_cells[c]["OD_cell"]["mean_sd"]["sd"],
            "od_omit_majority": conf_cells[c]["OD_cell"]["majority"]["p"],
            "fa_mean": conf_cells[c]["false_alarm"]["mean_sd"]["mean"],
            "fa_sd": conf_cells[c]["false_alarm"]["mean_sd"]["sd"],
            "n_omit_pairs": conf_cells[c]["OD_cell"]["majority"]["n"],
            "n_clean_notes": conf_cells[c]["n_clean_notes"]}
    ingredients = {c: (W.parse_cell(c)[2] == 8) + (W.parse_cell(c)[1] == "score")
                   for c in anchor_cells}
    eligible = [c for c, v in anchor_cells.items()
                if v["fa_mean"] is not None and v["fa_mean"] <= 0.20]
    anchor, fallback = None, False
    best_od = max([anchor_cells[c]["od_omit_mean"] for c in eligible
                   if anchor_cells[c]["od_omit_mean"] is not None], default=None)
    if best_od is not None and best_od >= 8 / 30:
        anchor = sorted(eligible, key=lambda c: (-(anchor_cells[c]["od_omit_mean"] or 0.0),
                                                 ingredients[c],
                                                 anchor_cells[c]["fa_mean"]))[0]
    elif anchor_cells:
        anchor, fallback = "FC-score-k8", True
    anchor_selection = {
        "rule": "highest mean omission detection among cells with clean-note FA <= 20%; ties -> "
                "fewer ingredients (k=1 over k=8, binary over scored), then lower FA. Fallback: "
                "no cell >= 8/30 omission detection -> FC-score-k8.",
        "cells": anchor_cells, "eligible_cells": sorted(eligible),
        "anchor": anchor, "fallback_applied": fallback}

    # ---- reading assignment (the pre-registered decision rules), on the
    # confirmatory subset only - the design these rules were written against.
    # NB every comparison below tests `is not None` explicitly: a cell with a
    # PERFECT 0.0 false-alarm rate is exactly the case an `or`-default would
    # silently discard, and that cell is the one most likely to win.
    def od(c):
        return (conf_cells.get(c, {}).get("OD_cell", {}).get("mean_sd", {}) or {}).get("mean")

    def fa(c):
        return (conf_cells.get(c, {}).get("false_alarm", {}).get("mean_sd", {}) or {}).get("mean")

    def le(v, bound):
        return v is not None and v <= bound

    def ge(v, bound):
        return v is not None and v >= bound
    usable = [c for c in grid_present if le(fa(c), 0.10)]
    od_best = max([od(c) for c in usable if od(c) is not None], default=None)
    f_cells = [c for c in grid_present if c.startswith("F-")]
    r1 = bool(usable) and od("FC-bin-k1") is not None and all([
        ge(od("FC-bin-k1"), 0.50),
        od_best is not None and od("FC-bin-k1") >= od_best - 0.10,
        le(fa("FC-bin-k1"), 0.10),
        all(le(od(c), 0.20) for c in f_cells)])
    r2_cells = [c for c in grid_present if c.startswith("FC-") and c != "FC-bin-k1"
                and ge(od(c), 0.50) and le(fa(c), 0.10)]
    r3 = not any(ge(od(c), 0.50) and le(fa(c), 0.10) for c in grid_present)
    reading = {"reading_1_criterion_only": r1,
               "reading_2_candidate_cells": r2_cells,
               "reading_2_requires_significant_mcnemar_vs_FC-bin-k1": [
                   {"cell": c, "contrast": n, "p_holm": v.get("p_holm")}
                   for c in r2_cells for n, v in famA.items()
                   if v.get("a") == "FC-bin-k1" and v.get("b") == c],
               "reading_3_no_usable_cell": r3,
               "usable_cells_fa_le_10pct": usable, "OD_best": od_best,
               "guard_partial_criterion_effect": any(od(c) is not None and od(c) > 0.20
                                                    for c in f_cells),
               "guard_all_FC_cells_noisy": bool(grid_present) and all(
                   fa(c) is not None and fa(c) > 0.10
                   for c in grid_present if c.startswith("FC-")),
               "computed_on": ("confirmatory subset: add + change + omit-complete"
                               if subset_differs else "the whole record set (no partial "
                               "omissions present)"),
               "note": "point estimates are means over runs; precedence Reading 1 > 2 > 3. "
                       "Reading 2 is confirmed only if its Holm-adjusted McNemar vs FC-bin-k1 "
                       "is p < 0.05, else reported as suggestive."}

    for c in list(cells.values()) + list(conf_cells.values()):  # drop internal maps
        for k in ("_majority", "_majority_fa", "_meta"):
            c.pop(k, None)

    result = {
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "spec": W.SPEC, "smoke": args.smoke,
        "stores": stores, "n_records": len(records),
        "manifest_validation": val,
        "substrate": {"frozen": all(r.get("substrate_frozen") for r in records),
                      "pairs_files": sorted({r.get("tag") for r in records})},
        "stats_standard": {"ci": "Wilson 95%", "paired_test": "exact McNemar on discordant pairs",
                           "bootstrap": f"cluster bootstrap at consultation level, {args.boot} draws",
                           "multiplicity": "Holm-Bonferroni within each pre-registered family",
                           "primary_inference": "per-item majority over runs"},
        "cells": cells,
        "confirmatory": {
            "definition": "add + change + omit-complete pairs and their clean twins - the "
                          "design the pre-registered readings were written against",
            "n_records": len(conf_records), "differs_from_full_set": subset_differs,
            "cells": conf_cells if subset_differs else "identical to `cells`"},
        "families": {"A_omission_discrimination": famA, "B_clean_false_alarm": famB,
                     "C_secondary": famC, "D_matched_complete_vs_partial": famD},
        "anchor_selection": anchor_selection,
        "youden_thresholds": thresholds,
        "reading": reading,
        "surface": {"per_arm": surfaces, "residual_strength": strengths,
                    "long_table": "w2_surface_smoke.tsv" if args.smoke else "w2_surface.tsv"},
        "arms_present": sorted(cells),
    }
    if args.cost_report:
        mp, tinfo = None, None
        try:
            mp, pinfo = W.load_pairs(allow_unfrozen=True)
            tinfo, _ = W.transcript_index()
        except Exception as e:
            print(f"  (projection skipped: {e})")
        result["cost_model"] = cost_model(records, master_pairs=mp, transcripts=tinfo)

    out = args.out or os.path.join(HERE, "w2_results_smoke.json" if args.smoke else "w2_results.json")
    json.dump(result, open(out, "w"), indent=1)
    print(f"wrote {os.path.relpath(out, HERE)}")

    # ---- the surface, in the two shapes the paper needs
    sfx = "_smoke" if args.smoke else ""
    spath = os.path.join(HERE, f"w2_surface{sfx}.json")
    all_levels = order_levels({r["residual_level"] for s in surfaces.values()
                               for r in s["rows"]
                               if r["residual_level"] not in ("omission-any", "all")})
    json.dump({"generated_utc": result["generated_utc"], "smoke": args.smoke,
               "axes": {"residual_level": all_levels + ["omission-any", "all"],
                        "severity": SEVERITY_LEVELS + ["all"],
                        "ordering": "residual levels run from complete removal (nothing "
                                    "survives) to a fully recoverable surviving mention; the "
                                    "intact condition sits beyond the end of that axis and is "
                                    "reported as the false-alarm rate"},
               "reading_guide": {
                   "cell": "detection rate = the arm scored the errored note worse than its "
                           "clean twin, per-item majority over replicates",
                   "intact_row": "no detection is definable when nothing was removed; the "
                                 "intact condition is reported as the clean-note false-alarm "
                                 "rate for the same arm",
                   "ci": "Wilson 95% on the majority outcome; cluster_boot resamples "
                         f"CONSULTATIONS ({args.boot_surface} draws)"},
               "per_arm": surfaces, "residual_strength": strengths,
               "matched_complete_vs_partial": famD},
              open(spath, "w"), indent=1)
    print(f"wrote {os.path.relpath(spath, HERE)}")

    tsv = ["\t".join(["arm", "arm_family", "residual_level", "severity", "n_pairs",
                      "n_consultations", "det_mean", "det_sd", "det_majority",
                      "wilson_lo", "wilson_hi", "boot_lo", "boot_hi"])]
    fam_of = {r.get("cell"): r.get("arm_family") for r in records if r.get("arm_family")}
    for arm in sorted(surfaces):
        for r in surfaces[arm]["rows"]:
            w, b, d = r["majority"], r["cluster_boot"], r["detection"]
            tsv.append("\t".join(str(x) if x is not None else "" for x in [
                arm, fam_of.get(arm, "grid" if arm in GRID_CELLS else "other"),
                r["residual_level"], r["severity"], r["n_pairs"], r["n_consultations"],
                d["mean"], d["sd"], w["p"], w["lo"], w["hi"], b["lo"], b["hi"]]))
        i = surfaces[arm]["intact"]
        w = i["false_alarm_majority"]
        b = i["false_alarm_cluster_boot"]
        tsv.append("\t".join(str(x) if x is not None else "" for x in [
            arm, fam_of.get(arm, "grid" if arm in GRID_CELLS else "other"),
            "intact", "n/a", i["n_clean_notes"], "", "", "", w["p"], w["lo"], w["hi"],
            b["lo"], b["hi"]]))
    tpath2 = os.path.join(HERE, f"w2_surface{sfx}.tsv")
    open(tpath2, "w").write("\n".join(tsv) + "\n")
    print(f"wrote {os.path.relpath(tpath2, HERE)}")

    # ---- LaTeX main table
    tex = [r"\begin{tabular}{lrrrrr}", r"\toprule",
           r"Cell & Overall & Add & Change & Omit (OD) & FA \\", r"\midrule"]
    for c in grid_present + [x for x in sorted(cells) if x not in grid_present]:
        m = cells[c]

        def f(d):
            w = d.get("majority") or {}
            return (f"{w['p']*100:.1f} [{w['lo']*100:.0f},{w['hi']*100:.0f}]"
                    if w.get("p") is not None else "--")
        tex.append(f"{c} & {f(m['discrimination']['overall'])} & "
                   + " & ".join(f(m["discrimination"]["by_type"][t])
                                for t in ("add", "change", "omit"))
                   + f" & {f(m['false_alarm'])} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    tpath = os.path.join(HERE, "w2_table_main_smoke.tex" if args.smoke else "w2_table_main.tex")
    open(tpath, "w").write("\n".join(tex) + "\n")
    print(f"wrote {os.path.relpath(tpath, HERE)}")

    # ---- console summary
    print("\ncell                 OD(omit)   FA      AUC     n_pairs")
    for c in sorted(cells):
        m = cells[c]
        o = m["OD_cell"]["mean_sd"]["mean"]
        a = m["false_alarm"]["mean_sd"]["mean"]
        u = m["auc_pooled"]["mean"]
        print(f"  {c:<18} {'--' if o is None else f'{o:6.1%}'}  "
              f"{'--' if a is None else f'{a:6.1%}'}  "
              f"{'--' if u is None else f'{u:5.3f}'}   {m['n_pairs']}")
    # ---- the surface, on the console (the shape of the paper's figure)
    sev_cols = [s for s in SEVERITY_LEVELS
                if any(r["severity"] == s for a in surfaces.values() for r in a["rows"])]
    if sev_cols:
        print("\nsurface: detection by residual level x severity (majority over runs)")
        for arm in sorted(surfaces):
            print(f"  {arm}")
            head = "    " + f"{'level':<14}" + "".join(f"{s:>13}" for s in sev_cols + ["all"])
            print(head)
            for lvl in surfaces[arm]["axes"]["residual_level"][:-1]:  # drop the "all" row
                row = {r["severity"]: r for r in surfaces[arm]["rows"]
                       if r["residual_level"] == lvl}
                if not row:
                    continue
                cells_txt = ""
                for s in sev_cols + ["all"]:
                    r = row.get(s)
                    cells_txt += (f"{r['majority']['p']:>8.1%}({r['n_pairs']:>3})"
                                  if r and r["majority"]["p"] is not None else f"{'--':>13}")
                print(f"    {lvl:<14}{cells_txt}")
            i = surfaces[arm]["intact"]
            fa_p = i["false_alarm_majority"]["p"]
            fa_txt = (f"{fa_p:>8.1%}({i['n_clean_notes']:>3})" if fa_p is not None
                      else f"{'--':>13}")
            print(f"    {'intact (FA)':<14}{fa_txt}")
            sl = surfaces[arm]["slopes"]
            print(f"    slopes: severity {sl['severity_slope']}, "
                  f"residual drop vs complete "
                  f"{ {k: v['all'] for k, v in sl['residual_drop_vs_complete'].items()} }")
            for lvl, m in (famD.get(arm) or {}).items():
                print(f"    matched complete-vs-{lvl}: {m['n_matched_facts']} facts, "
                      f"complete {m['a_rate']} vs {lvl} {m['b_rate']}, "
                      f"p={m['p_exact']} (Holm {m.get('p_holm')})")

    print(f"\nanchor_selection.anchor = {anchor_selection['anchor']}"
          f"{' (fallback)' if fallback else ''}")
    print(f"youden_thresholds: " + ", ".join(f"{c}={t['threshold']}"
                                             for c, t in thresholds.items()) or "none")
    if args.cost_report and "cost_model" in result:
        cm = result["cost_model"]
        print(f"\ncost: cache hit {cm.get('cache_hit_rate_overall')}, "
              f"fitted in/cached/out per Mtok = "
              f"{cm['fitted_prices']['uncached_in_per_mtok']}/"
              f"{cm['fitted_prices']['cached_in_per_mtok']}/"
              f"{cm['fitted_prices']['out_per_mtok']}")
        for f, v in (cm.get("by_family") or {}).items():
            if v:
                print(f"  {f:<10} ${v['usd_per_call']:.5f}/call  cache {v['cache_hit_rate']:.1%}")
        for f, v in (cm.get("k8_input_multiplier") or {}).items():
            print(f"  k=8 {f}: effective input multiplier "
                  f"{v['effective_input_multiplier']:.2f}x (nominal 8x)")
        cmod = cm.get("cache_model") or {}
        if cmod.get("transcript_chars_needed_to_cache"):
            print(f"  cache floor: a transcript must exceed "
                  f"{cmod['transcript_chars_needed_to_cache']:,} chars to cache at all "
                  f"({cmod['chars_per_token']} chars/token, 1024-token floor)")
        p = cm.get("projection_full_grid")
        if p:
            print(f"  projected full grid: {p['calls']:,} calls "
                  f"(pre-registered dedup applied), "
                  f"${p['usd_projected']:,.0f}; ${p['usd_if_nothing_cached']:,.0f} if nothing cached")
            print(f"    {p['consultations_that_will_cache']}/{p['consultations_total']} "
                  f"consultations clear the cache floor: "
                  + ", ".join(f"{k} {v['will_cache']}/{v['consultations']}"
                              for k, v in p["by_stratum"].items()))


if __name__ == "__main__":
    main()
