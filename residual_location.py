#!/usr/bin/env python3
"""Where does the surviving mention sit? - the residual-location split on partial omissions.

The paper calls partial-strong omissions "the most clinically deceptive case": a fact's
load-bearing instance is removed while a strong mention survives elsewhere. The clinician
review of the paper plan (2026-08-14) pointed out that whether that is deceptive at all
depends on WHERE the survivor sits. If the penicillin allergy is gone from the allergies
block but still stated two lines up in the same section, a reader scanning that section
still meets the fact and the note is arguably not deficient - so a judge that does not flag
it is right, not blind. If the survivor is in a different section entirely, the fact is only
recoverable by reading the whole note, which is Biro's misplaced-text class (FINDINGS 16.6),
and "deceptive" earns its keep.

This script splits every `omit-partial` pair by residual location and re-reads the same
purchased verdicts under that split. It buys nothing: no model call, no new judgement.

Two derivations, kept apart in the output because they are not the same instrument:

  site_map      the 114 factorial pairs. `master/fact_sites.json` gives the section of every
                mapped site; the pair records `primary_site_id`, `removed_site_ids` and
                `kept_site_ids`, so removed-section and residual-section are both read off
                the map with no inference. This is the primary evidence.

  diff_derived  the 34 relabelled partial-seed pairs, which predate the site map and were
                built by v1 model rewrite, so they carry no site ids. The removed region is
                recovered by diffing the clean note against the errored one and taking the
                largest deleted block; the residual is located by finding the auditor's
                quoted span in the errored note. Sections come from an anchor map built out
                of the mapper's OWN labels for that consultation, so the label vocabulary is
                identical across derivations. Validated against the 2,544 site labels the
                mapper assigned directly: 97.9% agreement, 23 labels unlocatable. Secondary
                evidence, reported separately and never pooled silently.

Conventions follow FINDINGS 15, 17.3 and 19: paired figures are tie-adjusted (half the tie
mass) and relative to the pair's own clean twin; absolute figures are flag rates quoted with
the false-alarm rate at the same operating point; the coarse `partial` label is refined via
`w2_common.STRENGTH_TO_LEVEL`.

Usage:  python3 residual_location.py [--out master/residual_location_analysis.json]
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import difflib
import json
import math
import os
import random
import re
import statistics
import sys

import w2_common as W

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "master", "dataset_v2.json")
SITES = os.path.join(HERE, "master", "fact_sites.json")
GRID_STORE = os.path.join(HERE, "results", "w2-ablation", "_state", "grid-main2.jsonl")
PIPE = {t: os.path.join(HERE, "results", "w2-pipeline", "_state", f"confirm-{t}.jsonl")
        for t in ("B2", "B3")}
BEST_CELL = "FC-score-k8"          # FINDINGS 17.1: the best monolithic cell
STRENGTH_RANK = {"explicit": 3, "paraphrase": 2, "partial": 1}

# --------------------------------------------------------------- section vocabulary
# 107 raw labels collapse to 68 canonical ones. The role map below is a reading of those
# labels, not a property of the data: every note in this corpus is free text with headers,
# so `registry_list` means "a header whose content is conventionally a list" and NOT an EHR
# structured field. That distinction is made explicit in the output and in FINDINGS.
_ROLES = {
    "narrative_history": ("history history of present illness hpc hpi presenting complaint pc "
                          "chief complaint cc subjective hx ice unheaded history"),
    "registry_list": ("review of systems review of symptoms rs se sh social history shx dh dhx "
                      "medications medication current medications pmh pmhx past medical history "
                      "medical history fh fhx fmh family history past surgical history "
                      "surgical history allergies allergies/adverse reactions vitals "
                      "vitals reviewed risk factors gu ls"),
    "exam_objective": ("examination physical exam physical examination exam ex oex o/e "
                       "o/e (remote) examination (face to face) results procedure"),
    "assessment_plan": ("plan pln assessment and plan assessment impression imp ddx instructions "
                        "safety-netting safety net safety-net safety netting treatment discussion"),
}
_MULTIWORD = ["history of present illness", "presenting complaint", "chief complaint",
              "unheaded history", "review of systems", "review of symptoms", "social history",
              "current medications", "past medical history", "medical history", "family history",
              "past surgical history", "surgical history", "allergies/adverse reactions",
              "vitals reviewed", "risk factors", "physical exam", "physical examination",
              "o/e (remote)", "examination (face to face)", "assessment and plan",
              "safety-netting", "safety net", "safety-net", "safety netting"]


def _build_role_map():
    out = {}
    for role, blob in _ROLES.items():
        rest = blob
        # longest first, or "physical exam" eats the front of "physical examination"
        for phrase in sorted(_MULTIWORD, key=len, reverse=True):
            if phrase in rest:
                out[phrase] = role
                rest = rest.replace(phrase, " ")
        for tok in rest.split():
            out.setdefault(tok, role)
    return out


ROLE_MAP = _build_role_map()


def canon_section(label):
    """One canonical section key. Drops the sub-heading the auditor sometimes appends
    ('ASSESSMENT AND PLAN - 3. Hypertension'), trailing punctuation and case, so that
    'Plan', 'Plan:' and 'PLAN' are one section and a same/different comparison means what
    it says."""
    if label in (None, "", "(none)"):
        return "(none)"
    label = re.split(r"\s+[-/]\s+", label)[0]
    return re.sub(r"\s+", " ", label.strip().rstrip(":;.,-#–— ").strip()).lower()


def section_role(key):
    if key in (None, "(none)"):
        return "no_header"
    return ROLE_MAP.get(key, "unclassified")


# --------------------------------------------------------------- note geometry
def section_anchors(note, labels):
    """(char_offset, canonical_label) for every mapper-used label that can be located as a
    header line in this note. Tolerates inline headers ('Plan: start amoxicillin') and any
    trailing punctuation, which is what the corpus's GP-style notes actually contain."""
    lines, pos = [], 0
    for line in note.split("\n"):
        lines.append((pos, line))
        pos += len(line) + 1
    out = []
    for lab in labels:
        key = canon_section(lab)
        if key == "(none)":
            continue
        pat = re.compile(r"^\s*" + re.escape(key) + r"\s*[:;.,\-–—#]*\s*(.*)$", re.I)
        for start, line in lines:
            if pat.match(line.strip()):
                out.append((start, key))
                break
    out.sort()
    return out


def section_at(anchors, idx):
    cur = "(none)"
    for start, key in anchors:
        if start <= idx:
            cur = key
        else:
            break
    return cur


def line_form_at(note, idx):
    """Is the char offset inside list-formatted text or prose? Mechanical, so the
    structured-versus-narrative question gets an answer from the note itself rather than
    from the header taxonomy. A line counts as list-formatted if it is a bullet, a numbered
    item, or a short unpunctuated field-value line."""
    start = note.rfind("\n", 0, idx) + 1
    end = note.find("\n", idx)
    line = note[start:end if end != -1 else len(note)].strip()
    if not line:
        return "blank"
    if re.match(r"^([-*•–]|\d+[.)]|[a-z][.)])\s", line):
        return "list"
    words = len(line.split())
    if words <= 12 and not line.endswith((".", "!", "?")):
        return "list"
    if ":" in line and words <= 14 and not line.endswith("."):
        return "list"
    return "prose"


# --------------------------------------------------------------- statistics
def wilson(k, n, z=1.959963984540054):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"lo": round(max(0.0, c - h), 4), "hi": round(min(1.0, c + h), 4)}


def cluster_boot(values_by_cluster, draws=5000, seed=20260816):
    """95% interval on a mean, resampling CONSULTATIONS not pairs - two pairs from one note
    share a clean twin and a transcript, so pair-level resampling would understate the
    interval. Same convention as w2_analyze."""
    clusters = list(values_by_cluster.values())
    if len(clusters) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        pool = []
        for _ in range(len(clusters)):
            pool.extend(clusters[rng.randrange(len(clusters))])
        if pool:
            means.append(sum(pool) / len(pool))
    if not means:
        return None
    means.sort()
    return {"lo": round(means[int(0.025 * len(means))], 4),
            "hi": round(means[int(0.975 * len(means))], 4),
            "draws": draws, "n_clusters": len(clusters)}


def two_prop_z(k1, n1, k2, n2):
    if not n1 or not n2:
        return None
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if not se:
        return None
    z = (k1 / n1 - k2 / n2) / se
    return {"z": round(z, 3), "p": round(2 * (1 - _ncdf(abs(z))), 5)}


def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def fisher_exact_2x2(a, b, c, d):
    """Two-sided Fisher exact - the honest test when a cell holds five pairs. Table is
    [[a, b], [c, d]] with a/c the event counts."""
    n = a + b + c + d
    if not n:
        return None
    r1, c1 = a + b, a + c

    def logc(m, k):
        if k < 0 or k > m:
            return None
        return math.lgamma(m + 1) - math.lgamma(k + 1) - math.lgamma(m - k + 1)

    def p_of(x):
        t1, t2 = logc(r1, x), logc(n - r1, c1 - x)
        if t1 is None or t2 is None:
            return 0.0
        return math.exp(t1 + t2 - logc(n, c1))

    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    ps = {x: p_of(x) for x in range(lo, hi + 1)}
    obs, tot = ps.get(a, 0.0), sum(ps.values())
    ext = sum(p for p in ps.values() if p <= obs * (1 + 1e-9))
    return round(min(1.0, ext / tot), 5) if tot else None


def paired_tie_adj(rows):
    """1 if the errored note scores strictly below its clean twin, 0.5 on a tie, 0 above."""
    rows = [(e, c) for e, c in rows if e is not None and c is not None]
    if not rows:
        return None
    return {"paired": round(sum(1.0 if e < c else 0.5 if e == c else 0.0
                                for e, c in rows) / len(rows), 4),
            "n": len(rows),
            "strict_wins": sum(1 for e, c in rows if e < c),
            "ties": sum(1 for e, c in rows if e == c),
            "losses": sum(1 for e, c in rows if e > c)}


# --------------------------------------------------------------- split assignment
def assign_splits(dataset, sites):
    cons = {c["key"]: c for c in sites["consultations"]}
    out, warn = [], []
    for p in dataset["pairs"]:
        if p.get("class") != "omit-partial":
            continue
        rec = {"pair_id": p["pair_id"], "source": p["source"], "consultation": p["key"],
               "stratum": p["stratum"], "split": p.get("split"), "eval_set": p.get("eval_set"),
               "residual_level": p["residual_level"], "severity": p.get("severity"),
               "n_surviving_mapped": (p.get("residual") or {}).get("n_surviving"),
               "residual_max_strength": (p.get("residual") or {}).get("max_strength")}
        if p.get("fact_uid") and p.get("primary_site_id") is not None:
            fk = "|".join(p["fact_uid"].split("|")[2:])
            fact = next(f for f in cons[p["key"]]["facts"] if f["fact_key"] == fk)
            smap = {s["site_id"]: s for s in fact["sites"]}
            note = cons[p["key"]]["note"]
            prim = smap[p["primary_site_id"]]
            removed = canon_section(prim.get("section"))
            kept = [smap[i] for i in (p.get("kept_site_ids") or []) if i in smap]
            residual = [canon_section(s.get("section")) for s in kept]
            rec.update(derivation="site_map",
                       removed_primary_section=removed,
                       removed_all_sections=[canon_section(smap[i].get("section"))
                                             for i in (p.get("removed_site_ids") or [])
                                             if i in smap],
                       residual_sections=residual,
                       residual_strengths=[s.get("strength") for s in kept],
                       removed_line_form=(line_form_at(note, prim["char_span"][0])
                                          if prim.get("char_span") else None),
                       residual_line_form=[line_form_at(note, s["char_span"][0])
                                           for s in kept if s.get("char_span")])
        else:
            note, err = p["clean"], p["errored"]
            labels = {s.get("section") for f in cons[p["key"]]["facts"]
                      for s in f.get("sites", [])}
            labels |= {s.get("section") for s in (p.get("residual") or {}).get("sites") or []}
            anch_c, anch_e = section_anchors(note, labels), section_anchors(err, labels)
            ops = difflib.SequenceMatcher(None, note, err, autojunk=False).get_opcodes()
            dels = [(i1, i2) for tag, i1, i2, _, _ in ops
                    if tag in ("delete", "replace") and i2 - i1 >= 8]
            removed = section_at(anch_c, max(dels, key=lambda d: d[1] - d[0])[0]) if dels else None
            sites_r = (p.get("residual") or {}).get("sites") or []
            best = max((STRENGTH_RANK.get(s.get("strength"), 0) for s in sites_r), default=0)
            strongest = [s for s in sites_r if STRENGTH_RANK.get(s.get("strength"), 0) == best]
            residual, forms = [], []
            for s in strongest:
                q = (s.get("quote") or "").strip()
                idx = err.find(q) if q else -1
                if idx < 0 and len(q) > 40:
                    idx = err.find(q[:40])
                if idx >= 0:
                    residual.append(section_at(anch_e, idx))
                    forms.append(line_form_at(err, idx))
                else:
                    residual.append(canon_section(s.get("section")))
                    forms.append(None)
                    warn.append({"pair_id": p["pair_id"],
                                 "issue": "residual quote not found verbatim in errored note; "
                                          "fell back to the auditor's own section label"})
            if removed is None:
                warn.append({"pair_id": p["pair_id"],
                             "issue": "no deleted block >= 8 chars found by diff; split unassigned"})
            rec.update(derivation="diff_derived",
                       removed_primary_section=removed,
                       removed_all_sections=sorted({section_at(anch_c, i1) for i1, _ in dels}),
                       residual_sections=residual,
                       residual_strengths=[s.get("strength") for s in strongest],
                       removed_line_form=(line_form_at(note, max(dels, key=lambda d: d[1] - d[0])[0])
                                          if dels else None),
                       residual_line_form=forms)
        rem = rec["removed_primary_section"]
        res = rec["residual_sections"]
        rec["location"] = (None if rem is None or not res
                           else "same_section" if any(r == rem for r in res)
                           else "different_section")
        rec["removed_section_role"] = section_role(rem)
        rec["residual_section_roles"] = [section_role(r) for r in res]
        rec["residual_in_assessment_plan"] = ("assessment_plan" in rec["residual_section_roles"]
                                              if res else None)
        rec["residual_in_registry_list"] = ("registry_list" in rec["residual_section_roles"]
                                            if res else None)
        rec["residual_form"] = ("list" if "list" in (rec["residual_line_form"] or [])
                                else "prose" if rec["residual_line_form"] else None)
        out.append(rec)
    return out, warn


# --------------------------------------------------------------- outcome joins
def grid_view(cell=BEST_CELL):
    by = collections.defaultdict(lambda: {"clean": {}, "err": []})
    with open(GRID_STORE) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("cell") != cell:
                continue
            slot = by[r["replicate"]]
            if r["note_role"] == "clean":
                slot["clean"][r["note_key"]] = r
            else:
                slot["err"].append(r)
    return by


def grid_cell_stats(by, pair_ids):
    """Per-replicate tie-adjusted paired discrimination and flag rate over a set of pairs,
    plus the consultation-clustered interval on the replicate-averaged per-pair score."""
    ids = set(pair_ids)
    per_rep_paired, per_rep_flag = [], []
    per_pair = collections.defaultdict(list)
    per_pair_flag = collections.defaultdict(list)
    consult = {}
    for rep in sorted(by):
        slot = by[rep]
        vals, flags = [], []
        for e in slot["err"]:
            if e.get("pair_id") not in ids:
                continue
            c = slot["clean"].get(e["clean_key"])
            if c is None:
                continue
            consult[e["pair_id"]] = e["consultation"]
            if e.get("parse_failure") or c.get("parse_failure"):
                v = 0.0                       # conservative, as in w2_analyze
            else:
                ea, ca = e["aggregate"], c["aggregate"]
                v = 1.0 if ea < ca else 0.5 if ea == ca else 0.0
            vals.append(v)
            per_pair[e["pair_id"]].append(v)
            fl = bool(e.get("flagged"))
            flags.append(fl)
            per_pair_flag[e["pair_id"]].append(fl)
        if vals:
            per_rep_paired.append(sum(vals) / len(vals))
            per_rep_flag.append(sum(flags) / len(flags))
    if not per_pair:
        return None
    means = {pid: sum(v) / len(v) for pid, v in per_pair.items()}
    by_cluster = collections.defaultdict(list)
    for pid, m in means.items():
        by_cluster[consult[pid]].append(m)
    maj_k = sum(1 for pid, v in per_pair_flag.items() if sum(v) * 2 > len(v))
    return {"n_pairs": len(per_pair), "n_consultations": len(by_cluster),
            "paired_tie_adjusted": {
                "mean": round(statistics.mean(per_rep_paired), 4),
                "sd": round(statistics.stdev(per_rep_paired), 4) if len(per_rep_paired) > 1 else 0.0,
                "per_run": [round(x, 4) for x in per_rep_paired],
                "cluster_boot": cluster_boot(by_cluster)},
            "flag_rate_spec_threshold": {
                "mean": round(statistics.mean(per_rep_flag), 4),
                "per_run": [round(x, 4) for x in per_rep_flag],
                "majority_k": maj_k, "n": len(per_pair_flag),
                "majority_rate": round(maj_k / len(per_pair_flag), 4),
                "wilson_majority": wilson(maj_k, len(per_pair_flag))}}


def load_pipeline(tier):
    recs = [json.loads(l) for l in open(PIPE[tier]) if l.strip()]
    clean = {r["note_key"]: r for r in recs if r["note_role"] == "clean"}
    err = [r for r in recs if r["note_role"] == "errored"]
    return recs, clean, err


def critical_rule(rec):
    """FINDINGS 19.3: flag the note iff any fact the audit stage graded `critical` is
    verdicted `absent`. Read off the per-fact detail, never off an aggregate."""
    for m in (rec.get("detail") or {}).get("missing_facts") or []:
        if m.get("severity") == "critical" and m.get("verdict") == "absent":
            return True
    return False


def sweep_threshold(err_scores, clean_scores, cap=0.10):
    """The arm-level operating point, chosen ONCE on the full omission set and then held
    fixed for every split. Choosing a threshold inside a split would be tuning on it."""
    cand = sorted({round(v, 9) for v in list(err_scores) + list(clean_scores)})
    best = None
    for t in cand + [(cand[-1] + 1e-6) if cand else 1.0]:
        fa_k = sum(1 for v in clean_scores if v < t)
        fa = fa_k / len(clean_scores) if clean_scores else None
        if fa is None or fa > cap + 1e-12:
            continue
        det_k = sum(1 for v in err_scores if v < t)
        cell = (det_k / len(err_scores), -fa, t)
        if best is None or cell > best[0]:
            best = (cell, {"threshold": round(t, 6), "fa": round(fa, 4), "fa_k": fa_k,
                           "n_clean": len(clean_scores)})
    return best[1] if best else None


def pipeline_cell_stats(clean, err, pair_ids, threshold, tier):
    ids = set(pair_ids)
    rows, scores, flags_rule = [], [], []
    by_cluster = collections.defaultdict(list)
    for e in err:
        if e.get("pair_id") not in ids:
            continue
        c = clean.get(e["clean_key"])
        if c is None:
            continue
        ea, ca = e["aggregate"], c["aggregate"]
        rows.append((ea, ca))
        scores.append(ea)
        by_cluster[e["consultation"]].append(1.0 if ea < ca else 0.5 if ea == ca else 0.0)
        if tier == "B3":
            flags_rule.append(critical_rule(e))
    if not rows:
        return None
    det_k = sum(1 for v in scores if v < threshold) if threshold is not None else None
    out = {"n_pairs": len(rows), "n_consultations": len(by_cluster),
           "paired_tie_adjusted": dict(paired_tie_adj(rows),
                                       cluster_boot=cluster_boot(by_cluster)),
           "swept_detection_at_arm_threshold": (
               None if threshold is None else
               {"threshold": threshold, "det_k": det_k, "n": len(scores),
                "det": round(det_k / len(scores), 4), "wilson": wilson(det_k, len(scores))})}
    if tier == "B3":
        k = sum(flags_rule)
        out["per_fact_critical_rule"] = {"det_k": k, "n": len(flags_rule),
                                         "det": round(k / len(flags_rule), 4),
                                         "wilson": wilson(k, len(flags_rule))}
    return out


# --------------------------------------------------------------- validation
def validate(by, pipeline):
    """Reproduce the published conditioning numbers before splitting anything. If these
    drift, the split figures below are computed on a different substrate and must not be
    quoted."""
    checks = []

    def lvl(r):
        l = (r.get("residual_level") or "").strip().lower()
        if l == "partial":
            return W.STRENGTH_TO_LEVEL.get((r.get("residual_strength") or "").strip().lower(),
                                           "partial")
        return l

    for level, want in (("complete", 0.690), ("partial-weak", 0.607), ("partial-strong", 0.526)):
        per = []
        for rep in sorted(by):
            slot = by[rep]
            vals = []
            for e in slot["err"]:
                if e.get("pair_type") != "omit" or lvl(e) != level:
                    continue
                c = slot["clean"].get(e["clean_key"])
                if c is None:
                    continue
                if e.get("parse_failure") or c.get("parse_failure"):
                    vals.append(0.0)
                    continue
                ea, ca = e["aggregate"], c["aggregate"]
                vals.append(1.0 if ea < ca else 0.5 if ea == ca else 0.0)
            per.append(sum(vals) / len(vals))
        got = round(statistics.mean(per), 3)
        checks.append({"what": f"FINDINGS 17.3 {BEST_CELL} paired, residual={level}",
                       "published": want, "recomputed": got, "ok": abs(got - want) <= 0.001})
    for tier, want in (("B2", 0.561), ("B3", 0.470)):
        clean, err = pipeline[tier][1], pipeline[tier][2]
        rows = [(e["aggregate"], clean[e["clean_key"]]["aggregate"]) for e in err
                if e["pair_type"] == "omit" and e["residual_level"] == "partial-strong"
                and e["clean_key"] in clean]
        got = paired_tie_adj(rows)["paired"]
        checks.append({"what": f"FINDINGS 19.8 {tier} paired, partial-strong",
                       "published": want, "recomputed": round(got, 3),
                       "ok": abs(round(got, 3) - want) <= 0.001})
    clean, err = pipeline["B3"][1], pipeline["B3"][2]
    ps = [e for e in err if e["pair_type"] == "omit" and e["residual_level"] == "partial-strong"]
    checks.append({"what": "FINDINGS 19.3 per-fact critical rule, partial-strong",
                   "published": "0/33", "recomputed": f"{sum(critical_rule(e) for e in ps)}/{len(ps)}",
                   "ok": sum(critical_rule(e) for e in ps) == 0 and len(ps) == 33})
    return checks


def validate_anchor_map(sites):
    """How good is the note-geometry instrument the diff_derived branch leans on? Assign a
    section to every site the mapper labelled directly, using only the anchor map, and score
    it against the mapper's own label. This is the error rate on the secondary derivation."""
    ok = bad = unlocatable = 0
    disagreements = collections.Counter()
    for c in sites["consultations"]:
        note = c["note"]
        labels = {s.get("section") for f in c["facts"] for s in f.get("sites", [])}
        anchors = section_anchors(note, labels)
        located = {k for _, k in anchors}
        for f in c["facts"]:
            for s in f.get("sites", []):
                span, want = s.get("char_span"), canon_section(s.get("section"))
                if not span:
                    continue
                if want != "(none)" and want not in located:
                    unlocatable += 1
                    continue
                got = section_at(anchors, span[0])
                if got == want:
                    ok += 1
                else:
                    bad += 1
                    disagreements[f"{want} -> {got}"] += 1
    return {"what": "anchor-map section assignment scored against the mapper's own site labels",
            "agree": ok, "disagree": bad, "label_not_locatable_as_a_line": unlocatable,
            "agreement": round(ok / (ok + bad), 4) if ok + bad else None,
            "top_disagreements": dict(disagreements.most_common(8))}


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "master",
                                                  "residual_location_analysis.json"))
    ap.add_argument("--cell", default=BEST_CELL)
    args = ap.parse_args()

    dataset = json.load(open(DATASET))
    sites = json.load(open(SITES))
    assign, warnings = assign_splits(dataset, sites)
    by = grid_view(args.cell)
    pipeline = {t: load_pipeline(t) for t in ("B2", "B3")}

    checks = validate(by, pipeline)
    if not all(c["ok"] for c in checks):
        print("VALIDATION FAILED - refusing to write", file=sys.stderr)
        for c in checks:
            print("  ", c, file=sys.stderr)
        sys.exit(2)

    # ---- supply tables
    def tally(rows, keyfn):
        return dict(collections.Counter(keyfn(r) for r in rows))

    supply = {}
    for level in ("partial-strong", "partial-weak"):
        rows = [r for r in assign if r["residual_level"] == level]
        ev = [r for r in rows if r["split"] == "eval"]
        supply[level] = {
            "n_all": len(rows), "n_eval": len(ev),
            "by_derivation": tally(rows, lambda r: r["derivation"]),
            "location_all": tally(rows, lambda r: r["location"]),
            "location_eval": tally(ev, lambda r: r["location"]),
            "location_by_derivation": {d: tally([r for r in rows if r["derivation"] == d],
                                                lambda r: r["location"])
                                       for d in ("site_map", "diff_derived")},
            "removed_section_role": tally(rows, lambda r: r["removed_section_role"]),
            "residual_section_role_first": tally(rows, lambda r: (r["residual_section_roles"] or
                                                                  [None])[0]),
            "residual_in_assessment_plan": tally(rows, lambda r: r["residual_in_assessment_plan"]),
            "residual_in_registry_list": tally(rows, lambda r: r["residual_in_registry_list"]),
            "residual_line_form": tally(rows, lambda r: r["residual_form"]),
            "section_transitions": dict(collections.Counter(
                f'{r["removed_primary_section"]} -> {(r["residual_sections"] or ["?"])[0]}'
                for r in rows).most_common()),
            "severity_by_location": {loc: tally([r for r in rows if r["location"] == loc],
                                                lambda r: r["severity"])
                                     for loc in ("same_section", "different_section")},
            # a note with no headers at all scores "same section" trivially - there is only
            # one section - so the headed-only count is the one the claim rests on
            "same_section_headed_only": sum(1 for r in rows if r["location"] == "same_section"
                                            and r["removed_primary_section"] != "(none)"),
            "same_section_unheaded_note": sum(1 for r in rows if r["location"] == "same_section"
                                              and r["removed_primary_section"] == "(none)"),
            "unclassified_sections": sorted({r["removed_primary_section"] for r in rows
                                             if r["removed_section_role"] == "unclassified"}
                                            | {s for r in rows for s, ro in
                                               zip(r["residual_sections"],
                                                   r["residual_section_roles"])
                                               if ro == "unclassified"}),
            "same_section_pairs": [{k: r[k] for k in
                                    ("pair_id", "derivation", "severity", "stratum", "split",
                                     "removed_primary_section", "residual_sections",
                                     "residual_strengths")}
                                   for r in rows if r["location"] == "same_section"],
        }

    # ---- the split definitions actually evaluated
    def sel(level, pred):
        return [r["pair_id"] for r in assign if r["residual_level"] == level and pred(r)]

    splits = {}
    for level in ("partial-strong", "partial-weak"):
        splits[level] = {
            "same_section": sel(level, lambda r: r["location"] == "same_section"),
            "different_section": sel(level, lambda r: r["location"] == "different_section"),
            "residual_in_assessment_plan": sel(level, lambda r: r["residual_in_assessment_plan"]),
            "residual_outside_assessment_plan": sel(
                level, lambda r: r["residual_in_assessment_plan"] is False),
            "residual_in_registry_list": sel(level, lambda r: r["residual_in_registry_list"]),
            "residual_in_prose_section": sel(level,
                                             lambda r: r["residual_in_registry_list"] is False),
            "residual_list_formatted": sel(level, lambda r: r["residual_form"] == "list"),
            "residual_prose_formatted": sel(level, lambda r: r["residual_form"] == "prose"),
            "all": sel(level, lambda r: True),
            "site_map_only_all": sel(level, lambda r: r["derivation"] == "site_map"),
            "site_map_only_same_section": sel(level, lambda r: r["derivation"] == "site_map"
                                              and r["location"] == "same_section"),
            "site_map_only_different_section": sel(level, lambda r: r["derivation"] == "site_map"
                                                   and r["location"] == "different_section"),
        }

    # ---- arm-level operating points, fixed once
    thresholds = {}
    for tier in ("B2", "B3"):
        _, clean, err = pipeline[tier]
        thresholds[tier] = sweep_threshold(
            [e["aggregate"] for e in err if e["pair_type"] == "omit"],
            [c["aggregate"] for c in clean.values()], cap=0.10)

    # ---- the false-alarm bases. A clean note has no residual, so false alarms belong to the
    # arm, not to a split; every detection figure below is read against these.
    fa = {}
    per_rep_fa = []
    for rep in sorted(by):
        C = list(by[rep]["clean"].values())
        per_rep_fa.append(sum(bool(c.get("flagged")) for c in C) / len(C))
    maj_fa = sum(1 for k in by[min(by)]["clean"]
                 if sum(bool(by[r]["clean"][k].get("flagged")) for r in by) * 2 > len(by))
    fa["grid_" + args.cell] = {"n_clean": len(by[min(by)]["clean"]),
                               "mean": round(statistics.mean(per_rep_fa), 4),
                               "per_run": [round(x, 4) for x in per_rep_fa],
                               "majority_k": maj_fa,
                               "majority_rate": round(maj_fa / len(by[min(by)]["clean"]), 4),
                               "basis": "the grid's own spec thresholds"}
    for tier in ("B2", "B3"):
        _, clean, _ = pipeline[tier]
        entry = {"n_clean": len(clean),
                 "swept": thresholds[tier],
                 "basis": "arm-level FA <= 10% operating point"}
        if tier == "B3":
            k = sum(critical_rule(c) for c in clean.values())
            entry["per_fact_critical_rule"] = {"fa_k": k, "n": len(clean),
                                               "fa": round(k / len(clean), 4),
                                               "wilson": wilson(k, len(clean))}
        fa["pipeline_" + tier] = entry

    results = {"grid_" + args.cell: {}, "pipeline_B2": {}, "pipeline_B3": {}}
    for level, spl in splits.items():
        for name, ids in spl.items():
            results["grid_" + args.cell].setdefault(level, {})[name] = grid_cell_stats(by, ids)
            for tier in ("B2", "B3"):
                _, clean, err = pipeline[tier]
                results["pipeline_" + tier].setdefault(level, {})[name] = pipeline_cell_stats(
                    clean, err, ids, (thresholds[tier] or {}).get("threshold"), tier)

    # ---- the contrast tests the question actually asks
    contrasts = {}
    for level in ("partial-strong", "partial-weak"):
        g = results["grid_" + args.cell][level]
        b3 = results["pipeline_B3"][level]
        pair_tests = {}
        for a, b in (("same_section", "different_section"),
                     ("residual_in_assessment_plan", "residual_outside_assessment_plan")):
            ga, gb = g.get(a), g.get(b)
            if ga and gb:
                pair_tests[f"grid_{a}_vs_{b}"] = {
                    "a": {"n": ga["n_pairs"], "paired": ga["paired_tie_adjusted"]["mean"],
                          "flag_majority": ga["flag_rate_spec_threshold"]["majority_rate"]},
                    "b": {"n": gb["n_pairs"], "paired": gb["paired_tie_adjusted"]["mean"],
                          "flag_majority": gb["flag_rate_spec_threshold"]["majority_rate"]},
                    "delta_paired": round(ga["paired_tie_adjusted"]["mean"]
                                          - gb["paired_tie_adjusted"]["mean"], 4),
                    "fisher_on_flag_majority": fisher_exact_2x2(
                        ga["flag_rate_spec_threshold"]["majority_k"],
                        ga["n_pairs"] - ga["flag_rate_spec_threshold"]["majority_k"],
                        gb["flag_rate_spec_threshold"]["majority_k"],
                        gb["n_pairs"] - gb["flag_rate_spec_threshold"]["majority_k"])}
            ba, bb = b3.get(a), b3.get(b)
            if ba and bb:
                pair_tests[f"B3_rule_{a}_vs_{b}"] = {
                    "a": {"n": ba["n_pairs"], "det": ba["per_fact_critical_rule"]["det"]},
                    "b": {"n": bb["n_pairs"], "det": bb["per_fact_critical_rule"]["det"]},
                    "fisher": fisher_exact_2x2(
                        ba["per_fact_critical_rule"]["det_k"],
                        ba["n_pairs"] - ba["per_fact_critical_rule"]["det_k"],
                        bb["per_fact_critical_rule"]["det_k"],
                        bb["n_pairs"] - bb["per_fact_critical_rule"]["det_k"])}
        contrasts[level] = pair_tests

    out = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "what": ("partial omissions split by where the surviving mention sits relative to the "
                 "removed load-bearing one - the pre-draft gate on the paper's "
                 "'most clinically deceptive case' claim. Re-reads purchased verdicts only; "
                 "no model call."),
        "script": "residual_location.py",
        "sources": {"dataset": "master/dataset_v2.json",
                    "dataset_version": dataset.get("dataset_version"),
                    "fact_sites": "master/fact_sites.json",
                    "fact_sites_instrument": sites.get("instrument_version"),
                    "grid_store": "results/w2-ablation/_state/grid-main2.jsonl",
                    "grid_cell": args.cell,
                    "pipeline_stores": ["results/w2-pipeline/_state/confirm-B2.jsonl",
                                        "results/w2-pipeline/_state/confirm-B3.jsonl"]},
        "conventions": {
            "paired": "tie-adjusted (half the tie mass), errored note against its own clean twin",
            "grid_absolute": "flag rate at the grid's own spec thresholds, majority over 3 "
                             "replicates; the clean-note false-alarm rate is a property of the "
                             "cell, not of a split, and is quoted at cell level only",
            "pipeline_absolute": "detection at the arm-level FA <= 10% operating point, chosen "
                                 "ONCE on all 131 confirmation omissions and then held fixed "
                                 "across splits",
            "per_fact_rule": "FINDINGS 19.3 - flag iff any audit-graded critical fact is "
                             "verdicted absent (B3 only; B2 carries severity: null)",
            "intervals": "Wilson on counts, consultation-clustered bootstrap on paired means",
            "structured_vs_narrative": ("no note in this corpus has an EHR structured field. "
                                        "`registry_list` is a header whose content is "
                                        "conventionally a list (ROS, medications, PMH, social, "
                                        "allergies, vitals); `residual_line_form` measures the "
                                        "note text itself. Neither is a structured field and "
                                        "the paper must not call them one.")},
        "validation": checks,
        "derivation_validation": validate_anchor_map(sites),
        "operating_points": thresholds,
        "false_alarm_bases": fa,
        "supply": supply,
        "split_definitions": {lvl: {k: len(v) for k, v in spl.items()}
                              for lvl, spl in splits.items()},
        "results": results,
        "contrasts": contrasts,
        "assignment_warnings": warnings,
        "per_pair": assign,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, sort_keys=False)
    print(f"wrote {args.out}")
    for level in ("partial-strong", "partial-weak"):
        s = supply[level]
        print(f"\n{level}: n={s['n_all']} (eval {s['n_eval']})  location {s['location_all']}")
        print(f"  by derivation {s['location_by_derivation']}")
        for name in ("same_section", "different_section", "residual_in_assessment_plan",
                     "residual_outside_assessment_plan", "all"):
            g = results["grid_" + args.cell][level].get(name)
            b2 = results["pipeline_B2"][level].get(name)
            b3 = results["pipeline_B3"][level].get(name)
            if not g:
                continue
            print(f"  {name:36s} n={g['n_pairs']:3d} grid={g['paired_tie_adjusted']['mean']:.3f} "
                  f"flag={g['flag_rate_spec_threshold']['majority_rate']:.3f} "
                  f"B2={(b2 or {}).get('paired_tie_adjusted', {}).get('paired')} "
                  f"B3={(b3 or {}).get('paired_tie_adjusted', {}).get('paired')} "
                  f"rule={(b3 or {}).get('per_fact_critical_rule', {}).get('det_k')}/"
                  f"{(b3 or {}).get('per_fact_critical_rule', {}).get('n')}")


if __name__ == "__main__":
    main()
