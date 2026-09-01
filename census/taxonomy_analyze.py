"""The census read-out, stage 4 of 4: rates, their replication, and the published tables.

Offline: reads the verified findings (and the discovery record stores for the
stability probe) and emits every number this stage owes downstream. No API calls.

  master/findings_rates.json   per-vendor and per-substrate verified-finding rates with
                               Wilson CIs, the mode and severity distributions, the
                               trapped-vs-trap-blind comparison with a consultation-level
                               cluster bootstrap, the CREOLA/Biro comparison inputs, and
                               the 3-run stability probe's per-note sd.
  master/cross_vendor_matches.json + master/cross_vendor_matches.md
                               the "not a vendor bug" beat: the same consultation and
                               the same failure family found independently in >=2
                               vendors, using cross_scribe_matches.py's family
                               matcher (imported, not copied, so it cannot drift).

Statistics come from w2_analyze (wilson, cluster_bootstrap) so this stage uses the
study's one implementation of each, not a second one.

    python3 census/taxonomy_analyze.py --in master/findings_verified_master.json
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import argparse, importlib.util, json, os, statistics, sys, time
from collections import Counter, defaultdict

from common import HERE, RESULTS
import taxonomy_common as T
from w2_analyze import cluster_bootstrap, wilson

# The failure-family matcher, imported from the pilot script itself.
_spec = importlib.util.spec_from_file_location(
    "cross_scribe_matches", os.path.join(HERE, "pilot", "scripts", "cross_scribe_matches.py"))
_xs = importlib.util.module_from_spec(_spec)
sys.path.insert(0, HERE)
_spec.loader.exec_module(_xs)
family, FAMILIES = _xs.family, _xs.FAMILIES
T.assert_family_keys_known(FAMILIES)

# Findings are placed in the published frame (taxonomy_frame.json), not in an ad-hoc
# rollup. Targeted passes are frame-derived so their placement is exact; open-pass
# findings are placed by the failure-family matcher, and what it cannot place is reported
# as "unmapped" rather than absorbed. The published comparator figures live in the
# frame file with their citations, denominators and verification status - nothing here
# invents one.
def place(f):
    return T.frame_place(f, family_fn=family)


def placed_tier1(f):
    return place(f)[1] or "unmapped"


def placed_tier2(f):
    return place(f)[0]


# ---------------------------------------------------------------- positional decay
# AWS, arXiv 2606.25656 (2026): LLM extraction recall decays sharply with the position of
# the target span inside a ~15K-token prompt - 26% recall at 30-40% depth, 0% at 70-80%.
# Our discovery pass is exactly that task shape (find every instance of X in a long
# transcript+note), so the same decay would be an instrument limitation, and one we can
# test for free from spans we already store. No API calls: this is arithmetic over quotes.
#
# The honest caveat, stated here because it governs how the numbers may be read: error
# OPPORTUNITY is not uniform across a note. Plans and safety-netting live at the end,
# demographics at the start. A raw early-skew is therefore NOT proof of positional decay.
# The discriminating test is the length interaction - if the skew is attention decay it
# should get WORSE as the transcript gets longer, because there is more depth to decay
# over. If the skew is constant across length quartiles it is more likely to be where the
# errors actually are. Both are computed below.
_WS = None


def _norm(s):
    global _WS
    if _WS is None:
        import re as _re
        _WS = _re.compile(r"\s+")
    return _WS.sub(" ", (s or "").strip().lower())


def locate(quote, text):
    """(normalised start position in [0,1], how) for `quote` inside `text`, else (None, why).

    Graded, because models do not always quote byte-exactly: exact match, then a
    distinctive prefix, then the rarest long word in the quote. The method used is
    returned with every hit so the locate rate can be discounted by how it was reached.
    """
    q, t = _norm(quote), _norm(text)
    if not q or q in ("-", "n/a", "none") or not t:
        return None, "no_quote"
    n = len(t)
    i = t.find(q)
    if i >= 0:
        return i / n, "exact"
    for L in (80, 50, 30):
        if len(q) > L:
            i = t.find(q[:L])
            if i >= 0:
                return i / n, f"prefix{L}"
    words = sorted({w for w in q.split() if len(w) >= 6}, key=len, reverse=True)
    for w in words[:6]:
        c = t.count(w)
        if c == 1:
            return t.find(w) / n, "unique_word"
    for w in words[:6]:
        i = t.find(w)
        if i >= 0:
            return i / n, "word"
    return None, "not_found"


def _dist(vals, label):
    """One position distribution against a uniform baseline."""
    import statistics
    n = len(vals)
    if n < 5:
        return {"n": n, "note": "too few located spans to characterise"}
    from scipy import stats as _st
    ks = _st.kstest(vals, "uniform")
    deciles = [0] * 10
    for v in vals:
        deciles[min(9, int(v * 10))] += 1
    out = {"n": n, "mean": round(statistics.mean(vals), 4),
           "median": round(statistics.median(vals), 4),
           "uniform_baseline_mean": 0.5,
           "decile_counts": deciles,
           "decile_shares": [round(d / n, 4) for d in deciles],
           "share_first_30pct": round(sum(1 for v in vals if v < 0.3) / n, 4),
           "share_last_30pct": round(sum(1 for v in vals if v >= 0.7) / n, 4),
           "ks_stat_vs_uniform": round(float(ks.statistic), 4),
           "ks_p": float(f"{ks.pvalue:.3e}")}
    out["skew"] = ("early" if out["mean"] < 0.45 else
                   "late" if out["mean"] > 0.55 else "roughly uniform")
    out["label"] = label
    return out


def position_bias(candidates, verified, units):
    """Where in the note / transcript / prompt do our findings actually come from?"""
    notes = {u["note_key"]: u for u in units}
    rows, how = [], Counter()
    for f in candidates:
        u = notes.get(f["note_key"])
        if not u:
            continue
        note, tx = u["note"], u["transcript"][:40000]
        npos, nhow = locate(f.get("note_quote"), note)
        tpos, thow = locate(f.get("source_quote"), tx)
        how[f"note:{nhow}"] += 1
        how[f"transcript:{thow}"] += 1
        ln, lt = len(_norm(note)), len(_norm(tx))
        tot = ln + lt
        # Prompt depth: the discovery prompt is [constant preamble][TRANSCRIPT][NOTE], so
        # the note sits deepest. The ~1k-char preamble is excluded - it is constant across
        # every call and would only compress the scale.
        depth = None
        if tpos is not None:
            depth = (tpos * lt) / tot if tot else None
        elif npos is not None:
            depth = (lt + npos * ln) / tot if tot else None
        rows.append({"fid": f.get("finding_id"), "scribe": f.get("scribe"),
                     "pass_type": "open" if f.get("pass") == "open" else "targeted",
                     "tier2": f.get("frame_tier2") or f.get("pass"),
                     "note_pos": npos, "tx_pos": tpos, "depth": depth,
                     "tx_len": len(tx),
                     "verified": bool((f.get("verdict") or {}).get("is_real"))})
    vset = {f.get("finding_id") for f in verified}
    for r in rows:
        r["verified"] = r["verified"] or r["fid"] in vset

    def pull(key, rs):
        return [r[key] for r in rs if r.get(key) is not None]

    out = {"reference": "AWS arXiv 2606.25656 - extraction recall 26% at 30-40% prompt "
                        "depth, 0% at 70-80%, in ~15K-token prompts",
           "method": "normalised start position of each stored quote inside its note "
                     "(note_quote) and inside its transcript (source_quote), matched "
                     "exact -> distinctive prefix -> rarest long word; unlocatable spans "
                     "are excluded and counted, not imputed",
           "why_transcript_too": "omissions carry no note_quote by construction (the "
                                 "content is missing from the note), so a note-only "
                                 "analysis would silently drop the category the study "
                                 "cares most about. The transcript position covers them.",
           "n_findings_considered": len(rows),
           "locate_methods": dict(how.most_common()),
           "locate_rate": {
               "note_quote": round(sum(1 for r in rows if r["note_pos"] is not None) / max(len(rows), 1), 4),
               "source_quote": round(sum(1 for r in rows if r["tx_pos"] is not None) / max(len(rows), 1), 4)},
           }
    for field, name in (("note_pos", "note_position"), ("tx_pos", "transcript_position"),
                        ("depth", "prompt_depth")):
        blk = {"all_candidates": _dist(pull(field, rows), "all candidates"),
               "panel_verified": _dist(pull(field, [r for r in rows if r["verified"]]), "verified"),
               "by_pass_type": {pt: _dist(pull(field, [r for r in rows if r["pass_type"] == pt]), pt)
                                for pt in ("targeted", "open")},
               "by_vendor": {v: _dist(pull(field, [r for r in rows if r["scribe"] == v]), v)
                             for v in sorted({r["scribe"] for r in rows})}}
        if name == "prompt_depth":
            blk["caveat"] = (
                "DESCRIPTIVE ONLY - do not read the KS test against uniform here. Depth is "
                "taken from the transcript span where one is located and from the note span "
                "otherwise, and the transcript occupies a fixed early fraction of every "
                "prompt (~80-90%). So the depth distribution is bounded by construction and "
                "will look early-skewed whatever the model does. The interpretable measures "
                "are transcript_position and note_position, each uniform-comparable within "
                "its own text.")
        out[name] = blk

    # The discriminating test: does the skew deepen as the transcript gets longer?
    lens = sorted({r["tx_len"] for r in rows})
    if len(lens) >= 4 and rows:
        qs = [lens[int(len(lens) * q)] for q in (0.25, 0.5, 0.75)]

        def bucket(r):
            return ("q1_shortest" if r["tx_len"] <= qs[0] else "q2" if r["tx_len"] <= qs[1]
                    else "q3" if r["tx_len"] <= qs[2] else "q4_longest")
        out["length_interaction"] = {
            "why": "if the early-skew is attention decay it should WORSEN with transcript "
                   "length (more depth to decay over). If it is flat across quartiles, the "
                   "skew more likely reflects where errors actually occur in a note.",
            "quartile_cuts_chars": qs,
            "transcript_position_mean_by_quartile": {
                b: _dist([r["tx_pos"] for r in rows if bucket(r) == b and r["tx_pos"] is not None], b)
                for b in ("q1_shortest", "q2", "q3", "q4_longest")}}
    # top frame categories, since omission is the one that matters most
    out["transcript_position_by_tier2"] = {
        k: _dist([r["tx_pos"] for r in rows if r["tier2"] == k and r["tx_pos"] is not None], k)
        for k, _ in Counter(r["tier2"] for r in rows).most_common(8)}
    return out


def rate_block(notes, findings_by_note, predicate=lambda f: True):
    """Per-note finding-rate summary over a set of note records."""
    counts = [sum(1 for f in findings_by_note.get(n["note_key"], []) if predicate(f)) for n in notes]
    n = len(counts)
    if not n:
        return {"n_notes": 0}
    with_any = sum(1 for c in counts if c)
    return {"n_notes": n, "n_findings": sum(counts),
            "findings_per_note_mean": round(sum(counts) / n, 4),
            "findings_per_note_sd": round(statistics.stdev(counts), 4) if n > 1 else 0.0,
            "findings_per_note_median": statistics.median(counts),
            "findings_per_note_max": max(counts),
            "notes_with_any": wilson(with_any, n)}


def main():
    ap = argparse.ArgumentParser(description="census read-out: rates + cross-vendor replication")
    ap.add_argument("--in", dest="infile", default="master/findings_verified_master.json")
    ap.add_argument("--clusters", default="master/findings_clusters.json")
    ap.add_argument("--discovery", default="master/findings_master.json",
                    help="raw discovery output, for the position-bias baseline over ALL "
                         "candidates (the verified file holds only the salience-filtered pool)")
    ap.add_argument("--draws", type=int, default=10000, help="cluster-bootstrap draws")
    ap.add_argument("--out-prefix", default="")
    args = ap.parse_args()

    data = json.load(open(os.path.join(HERE, args.infile)))
    findings = data["all_issues"]
    real = [f for f in findings if (f.get("verdict") or {}).get("is_real")]
    # Every rate is computed over PANEL-COVERED notes only. A note whose consultation the
    # panel never reached has zero verified findings because nothing judged it, and
    # including it would silently deflate every rate in this file. taxonomy_verify stamps
    # `panel_covered` per note; older files without the stamp are treated as fully covered.
    all_notes = data["per_note"]
    notes = [n for n in all_notes if n.get("panel_covered", True)]
    if len(notes) < len(all_notes):
        print(f"panel coverage: {len(notes)} of {len(all_notes)} notes were judged "
              f"({len({n['consultation'] for n in notes})} of "
              f"{len({n['consultation'] for n in all_notes})} consultations). All rates below "
              "are over the judged notes; the rest are excluded, not counted as clean.")
    by_note = defaultdict(list)
    for f in real:
        by_note[f["note_key"]].append(f)
    cand_by_note = defaultdict(list)
    for f in findings:
        cand_by_note[f["note_key"]].append(f)

    print(f"analyze | {len(notes)} notes | {len(findings)} candidates -> {len(real)} verified "
          f"({len(real) / len(findings):.1%} survived)" if findings else "analyze | no findings")

    # ---- rates: overall, per vendor, per substrate, per vendor x substrate
    rates = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "source_file": args.infile, "panel": data.get("panel"), "route": data.get("route"),
             "n_notes": len(notes), "n_consultations": len({n["consultation"] for n in notes}),
             "n_notes_discovered": len(all_notes),
             "n_consultations_discovered": len({n["consultation"] for n in all_notes}),
             "panel_coverage": round(len(notes) / len(all_notes), 4) if all_notes else None,
             "coverage_note": "all rates are over panel-covered notes; discovery covered the "
                              "wider set and its candidate counts are reported separately",
             "corpus_census": T.corpus_census(*T.note_units()),
             "n_candidates": len(findings), "n_verified": len(real),
             "panel_survival": wilson(len(real), len(findings)) if findings else None,
             "overall": rate_block(notes, by_note),
             "overall_candidates": rate_block(notes, cand_by_note),
             "by_vendor": {}, "by_substrate": {}, "by_vendor_substrate": {},
             "by_template_scribe_A": {}}
    for v in sorted({n["scribe"] for n in notes}):
        sub = [n for n in notes if n["scribe"] == v]
        rates["by_vendor"][v] = rate_block(sub, by_note)
        rates["by_vendor"][v]["high_salience"] = rate_block(
            sub, by_note, lambda f: f.get("salience") == "high")
        rates["by_vendor"][v]["rubric_critical"] = rate_block(
            sub, by_note, lambda f: f.get("severity_rubric") == "critical")
    for s in sorted({n["source"] for n in notes}):
        sub = [n for n in notes if n["source"] == s]
        rates["by_substrate"][s] = rate_block(sub, by_note)
    for v in sorted({n["scribe"] for n in notes}):
        for s in sorted({n["source"] for n in notes}):
            sub = [n for n in notes if n["scribe"] == v and n["source"] == s]
            if sub:
                rates["by_vendor_substrate"][f"{v}/{s}"] = rate_block(sub, by_note)
    for tmpl in sorted({n["template"] for n in notes if n["scribe"] == "scribe_A"}):
        sub = [n for n in notes if n["scribe"] == "scribe_A" and n["template"] == tmpl]
        rates["by_template_scribe_A"][tmpl] = rate_block(sub, by_note)

    # ---- distributions
    rates["distributions"] = {
        "frame_tier1": dict(Counter(placed_tier1(f) for f in real).most_common()),
        "frame_tier2": dict(Counter(placed_tier2(f) for f in real).most_common()),
        "frame_placed_by": dict(Counter(place(f)[2].split(":")[0] for f in real).most_common()),
        "mode": dict(Counter(f.get("mode") for f in real).most_common()),
        "pass_that_found_it": dict(Counter(f.get("pass") for f in real).most_common()),
        "check": dict(Counter(f.get("check") for f in real)),
        "salience_panel": dict(Counter(f.get("salience") for f in real)),
        "severity_rubric": dict(Counter(f.get("severity_rubric") for f in real)),
        "severity_differ_guess": dict(Counter(f.get("severity") for f in real)),
        "cut_by_mode": dict(Counter(f.get("mode") for f in findings
                                    if not (f.get("verdict") or {}).get("is_real")).most_common()),
        "discovered_by": dict(Counter(f.get("discovered_by", "discover") for f in real)),
    }

    # ---- trapped (authored) vs trap-blind, per-note verified finding rate
    rates["trapped_vs_trapblind"] = trapped_vs_trapblind(notes, by_note, args.draws)

    # ---- CREOLA / Biro comparison inputs
    rates["literature_comparison_inputs"] = literature_inputs(notes, real, by_note)

    # ---- 3-run stability probe
    rates["stability_probe"] = stability_probe()

    # ---- positional decay (offline arithmetic over stored quotes, no API calls)
    dpath = os.path.join(HERE, args.discovery)
    pool = findings
    if os.path.exists(dpath):
        pool = json.load(open(dpath))["all_issues"]
        print(f"position bias: {len(pool)} discovery candidates "
              f"(the verified file holds the {len(findings)} that cleared the salience filter)")
    units, _ = T.note_units()
    rates["position_bias"] = position_bias(pool, real, units)

    # ---- cluster rollup, if the taxonomy has been built
    cpath = os.path.join(HERE, args.clusters)
    if os.path.exists(cpath):
        c = json.load(open(cpath))
        rates["taxonomy"] = {
            "n_clusters": c["n_clusters"], "n_noise": c["n_noise"],
            "params": {k: c["params"].get(k) for k in
                       ("n_neighbors", "min_cluster_size", "min_samples", "umap_seed")},
            "clusters": [{"label": s["label"], "size": s["size"], "vendors": s["vendors"],
                          "substrates": s["substrates"], "consultations": s["consultations"],
                          "high_salience": s["high_salience"],
                          "cross_vendor": len(s["vendors"]) >= 2} for s in c["scaffold"]]}

    path = os.path.join(T.MASTER, f"{args.out_prefix}findings_rates.json")
    json.dump(rates, open(path, "w"), indent=1)

    # ---- cross-vendor replication
    matches = cross_vendor(real, args.out_prefix)
    rates["cross_vendor"] = matches["summary"]
    json.dump(rates, open(path, "w"), indent=1)

    # ---- the frame frequency table
    tpath = frame_table_md(rates, os.path.join(T.MASTER, f"{args.out_prefix}frame_frequency_table.md"))

    print_summary(rates)
    print(f"\nsaved -> {os.path.relpath(path, HERE)} + {os.path.relpath(tpath, HERE)}")


# ---------------------------------------------------------------- trap inflation
def trapped_vs_trapblind(notes, by_note, draws):
    """Trapped-30 (authored) vs trap-blind-10 per-note verified-finding rate.

    Cluster bootstrap at CONSULTATION level (the study standard) on the difference in
    mean findings per note. This is one half of the trap-inflation reading - the judge
    omission-detection half comes from the judge benchmark's runs, and the Holm
    correction across the two comparisons is applied where they meet.
    """
    strata = {}
    for s in ("authored", "trapblind"):
        strata[s] = [n for n in notes if n["source"] == s]
    if not strata["authored"] or not strata["trapblind"]:
        return {"available": False,
                "why": "needs both the authored and trapblind strata in the audited corpus"}
    counts = {n["note_key"]: len(by_note.get(n["note_key"], [])) for n in notes}
    clusters = defaultdict(list)
    for n in notes:
        if n["source"] in strata:
            clusters[n["consultation"]].append((n["source"], counts[n["note_key"]]))

    def diff(sample):
        a = [c for grp in sample for src, c in grp if src == "authored"]
        b = [c for grp in sample for src, c in grp if src == "trapblind"]
        if not a or not b:
            return None
        return sum(a) / len(a) - sum(b) / len(b)

    obs = diff([v for v in clusters.values()])
    ci = cluster_bootstrap(dict(clusters), diff, draws)
    includes_zero = None
    if ci.get("lo") is not None:
        includes_zero = bool(ci["lo"] <= 0 <= ci["hi"])
    return {"available": True,
            "authored": rate_block(strata["authored"], by_note),
            "trapblind": rate_block(strata["trapblind"], by_note),
            "difference_findings_per_note": round(obs, 4) if obs is not None else None,
            "cluster_bootstrap_ci": ci, "ci_includes_zero": includes_zero,
            "reading": ("no trap inflation detected on the finding-rate comparison"
                        if includes_zero else
                        "trap inflation detected on the finding-rate comparison - report it "
                        "explicitly and move headline generalisation claims to PriMock/ACI")
                       if includes_zero is not None else "CI not estimable",
            "note": "one half of the trap-inflation check; the judge omission-detection half "
                    "comes from the judge benchmark, and Holm-Bonferroni across the two "
                    "comparisons is applied where they are combined"}


# ---------------------------------------------------------------- literature table
def literature_inputs(notes, real, by_note):
    """Our side of the published-frame comparison, plus the frame's own comparators.

    Emits our numbers computed against the frame, and carries each published study's
    figures across from taxonomy_frame.json with its citation, denominator and
    verification status attached. Nothing here fabricates a comparator, and any study
    the frame marks `is_frequency_comparator: false` is carried as naming-only.
    """
    fr = T.load_frame()
    n = len(notes)
    t1_counts = Counter(placed_tier1(f) for f in real)
    t2_counts = Counter(placed_tier2(f) for f in real)

    def block(key, getter):
        with_any = sum(1 for x in notes
                       if any(getter(f) == key for f in by_note.get(x["note_key"], [])))
        c = (t1_counts if getter is placed_tier1 else t2_counts).get(key, 0)
        return {"n_findings": c,
                "share_of_findings": round(c / len(real), 4) if real else None,
                "per_note_rate": round(c / n, 4) if n else None,
                "notes_with_any": wilson(with_any, n)}

    by_t1 = {t["key"]: block(t["key"], placed_tier1) for t in fr["tier1"]}
    by_t1["unmapped"] = block("unmapped", placed_tier1)
    by_t2 = {}
    for c in fr["tier2"]:
        if c["key"] == "open":
            continue
        b = block(c["key"], placed_tier2)
        b["tier1"] = c.get("tier1")
        b["hunted"] = bool(c.get("hunted"))
        if not c.get("hunted"):
            b["measured"] = False
            b["why_not_measured"] = c["why_not_hunted"]
        by_t2[c["key"]] = b
    by_t2["unmapped"] = block("unmapped", placed_tier2)

    hi = sum(1 for x in notes if any(f.get("salience") == "high" for f in by_note.get(x["note_key"], [])))
    crit = sum(1 for x in notes
               if any(f.get("severity_rubric") == "critical" for f in by_note.get(x["note_key"], [])))
    comparators = [{"id": s["id"], "citation": s["citation"],
                    "verify_at": s.get("verify_at"),
                    "classes": s.get("classes"),
                    "headline_numbers": s.get("headline_numbers") or s.get("what_is_known"),
                    "crosswalk_to_our_tier1": s.get("crosswalk_to_our_tier1"),
                    "comparable_measures": s.get("comparable_measures"),
                    "not_comparable": s.get("not_comparable"),
                    "is_frequency_comparator": s.get("is_frequency_comparator"),
                    "why_not": s.get("why_not")}
                   for s in fr["published_taxonomies"]]
    return {"frame_version": fr["frame_version"], "frame_sha256": T.frame_sha256(),
            "comparability_rules": fr["comparability_rules"],
            "verification_status": fr["verification_status"]["summary"],
            "n_notes": n, "n_consultations": len({x["consultation"] for x in notes}),
            "n_verified_findings": len(real),
            "verified_findings_per_note": round(len(real) / n, 4) if n else None,
            "notes_with_any_verified": wilson(sum(1 for x in notes if by_note.get(x["note_key"])), n),
            "notes_with_high_salience": wilson(hi, n),
            "notes_with_rubric_critical": wilson(crit, n),
            "by_tier1": by_t1, "by_tier2": by_t2,
            "published_comparators": comparators}


def frame_table_md(rates, path):
    """The frame-frequency table: our shares beside the published ones, with the
    summing rules the frame declares applied rather than left to the reader."""
    fr = T.load_frame()
    li = rates["literature_comparison_inputs"]
    n, nf = li["n_notes"], li["n_verified_findings"]
    L = ["# Frame frequency table", "",
         f"Frame `{T.FRAME_FILE}` v{li['frame_version']} (`{li['frame_sha256'][:12]}`). "
         f"Our column: {nf} verified findings over {n} notes "
         f"({li['n_consultations']} consultations), two cross-family skeptics with a "
         "tiebreak on disagreement.", "",
         f"**Verification status of every published number below:** {li['verification_status']}", "",
         "## Tier 1 - the published spine (Biro et al. 2025)", "",
         "| category | our findings | our share | our per-note rate | notes with >=1 (95% CI) | Biro A | Biro B | Kernberg | Anderson |",
         "|---|---|---|---|---|---|---|---|---|"]
    biro = {"omission": ("83%", "54%"), "addition": ("4%", "11%"),
            "wrong_output": ("6%", "10%"), "misplaced_or_irrelevant": ("6%", "25%")}
    kern = {"omission": "86.3%", "addition": "10.5%", "wrong_output": "3.2%",
            "misplaced_or_irrelevant": "not a class"}
    ande = {"omission": "76%", "addition": "(in commission)", "wrong_output": "(in commission)",
            "misplaced_or_irrelevant": "not a class"}
    for t in fr["tier1"]:
        k = t["key"]
        b = li["by_tier1"][k]
        w = b["notes_with_any"]
        L.append(f"| **{t['name']}** | {b['n_findings']} | "
                 f"{(b['share_of_findings'] or 0):.1%} | {b['per_note_rate']} | "
                 f"{w['p']:.1%} [{w['lo']:.1%}, {w['hi']:.1%}] | {biro[k][0]} | {biro[k][1]} | "
                 f"{kern[k]} | {ande[k]} |")
    u = li["by_tier1"]["unmapped"]
    L += [f"| _unmapped (open-pass findings the frame could not place)_ | {u['n_findings']} | "
          f"{(u['share_of_findings'] or 0):.1%} | {u['per_note_rate']} | - | - | - | - | - |", "",
          "Biro A/B are the two commercial products in Biro et al. (n=66 and n=61 errors); their "
          "class distribution differed significantly between products (Fisher exact p=0.002), so "
          "the two columns are the honest spread, not a range to average. Kernberg is raw GPT-4 "
          "rather than a deployed scribe. Anderson's 'commission' merges our addition and wrong "
          "output, and their 'partially correct' spans omission and wrong output, so only their "
          "omission share maps cleanly.", "",
          "### Sums the frame requires before comparing", ""]
    add = li["by_tier1"]["addition"]["n_findings"]
    wro = li["by_tier1"]["wrong_output"]["n_findings"]
    L += [f"- **vs CREOLA / Anderson**: their hallucination / commission bucket = our addition + "
          f"wrong output = {add + wro} findings "
          f"({(add + wro) / nf:.1%} of ours) against our omission "
          f"{li['by_tier1']['omission']['share_of_findings']:.1%}.",
          "- CREOLA reports 1.47% hallucinated and 3.45% omitted **sentences**; those are per-"
          "sentence rates and cannot be set beside our per-note rates without re-basing, which "
          "we do not do. Their omission-to-hallucination ratio (2.3:1) is the comparable quantity.",
          "- Taylor et al. is the only study whose denominator matches our 'notes with >=1' column: "
          "accidental omission 18%, hallucination 11.5%, accidental inclusion 9.3%, bias 1.1% of "
          "356 notes.", "",
          "## Tier 2 - mechanism level", "",
          "| category | tier 1 | findings | share | per-note | notes with >=1 |", "|---|---|---|---|---|---|"]
    for k, b in li["by_tier2"].items():
        if b.get("measured") is False:
            L.append(f"| {k} | {b.get('tier1') or '-'} | _not measured_ | _not measured_ | "
                     f"_not measured_ | _not measured_ |")
            continue
        w = b["notes_with_any"]
        L.append(f"| {k} | {b.get('tier1') or '-'} | {b['n_findings']} | "
                 f"{(b['share_of_findings'] or 0):.1%} | {b['per_note_rate']} | "
                 f"{w['p']:.1%} [{w['lo']:.1%}, {w['hi']:.1%}] |")
    L += ["", "Two frame categories are declared not hunted (coding_terminology, "
          "bias_stigmatising). They read as not measured, never as zero - see the frame file "
          "for why each was left out.", "",
          "## Rules carried from the frame", ""]
    L += [f"{i + 1}. {r}" for i, r in enumerate(li["comparability_rules"])]
    L += ["", "## Sources", ""]
    for c in li["published_comparators"]:
        tag = "frequency comparator" if c["is_frequency_comparator"] else "naming only"
        L += [f"- **{c['id']}** ({tag}) - {c['citation']}",
              f"  - verify at: {c['verify_at']}"]
        if c.get("not_comparable"):
            L.append(f"  - not comparable: {c['not_comparable']}")
        if c.get("why_not"):
            L.append(f"  - why naming-only: {c['why_not']}")
    open(path, "w").write("\n".join(L))
    return path


# ---------------------------------------------------------------- stability probe
def stability_probe():
    """Per-note candidate-finding counts across the 3 discovery runs on the seeded
    15-consultation subsample. Reads the discovery record stores
    directly, so it works whether or not runs 2/3 were ever panel-verified."""
    state = os.path.join(RESULTS, T.EXPERIMENT, "_state")
    if not os.path.isdir(state):
        return {"available": False, "why": "no record stores yet"}
    per_run = defaultdict(dict)          # run -> note_key -> n_issues
    for fn in sorted(os.listdir(state)):
        if not fn.endswith(".jsonl") or not fn.startswith("discover"):
            continue
        for line in open(os.path.join(state, fn)):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "pass" not in r or "run" not in r:
                continue
            d = per_run[r["run"]].setdefault(r["note_key"], 0)
            per_run[r["run"]][r["note_key"]] = d + r.get("n_issues", 0)
    runs = sorted(per_run)
    if len(runs) < 2:
        return {"available": False, "runs_present": runs,
                "why": "the stability probe needs >=2 discovery runs over the seeded subsample"}
    shared = set(per_run[runs[0]])
    for r in runs[1:]:
        shared &= set(per_run[r])
    if not shared:
        return {"available": False, "runs_present": runs, "why": "no notes shared across runs"}
    sds, means = [], []
    for nk in sorted(shared):
        vals = [per_run[r][nk] for r in runs]
        means.append(sum(vals) / len(vals))
        sds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
    return {"available": True, "runs": runs, "n_notes": len(shared),
            "per_run_findings_per_note": {str(r): round(
                sum(per_run[r][nk] for nk in shared) / len(shared), 4) for r in runs},
            "mean_per_note_sd": round(sum(sds) / len(sds), 4),
            "max_per_note_sd": round(max(sds), 4),
            "mean_per_note_mean": round(sum(means) / len(means), 4),
            "measure": "candidate (pre-panel) findings per note across discovery runs"}


# ---------------------------------------------------------------- cross-vendor
def cross_vendor(real, prefix=""):
    """Same consultation, same failure family, independently in >=2 vendors.

    cross_scribe_matches.py's `family()` regex matcher, imported rather than copied.
    One change: the pilot script keyed on the bare consultation id; here the key is
    (source, id), because the master corpus mixes four substrates and a bare id is no
    longer unique across them.
    """
    idx = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in real:
        idx[f["consultation"]][family(f)][f.get("scribe", "scribe_A")].append(f)

    matches, cells = [], 0
    for cid, fams in idx.items():
        for fam, by_scribe in fams.items():
            cells += 1
            if len(by_scribe) >= 2:
                matches.append({"consultation": cid, "family": fam,
                                "vendors": sorted(by_scribe),
                                "n_vendors": len(by_scribe),
                                "high_salience": max((sum(x.get("salience") == "high" for x in v)
                                                      for v in by_scribe.values()), default=0),
                                "rubric_critical": max((sum(x.get("severity_rubric") == "critical"
                                                            for x in v)
                                                        for v in by_scribe.values()), default=0),
                                "examples": {s: {"description": v[0].get("description"),
                                                 "note_quote": (v[0].get("note_quote") or "")[:200],
                                                 "salience": v[0].get("salience")}
                                             for s, v in sorted(by_scribe.items())}})
    matches.sort(key=lambda m: (-m["n_vendors"], -m["high_salience"], m["consultation"]))
    summary = {"n_consultation_family_cells": cells, "n_matches": len(matches),
               "match_rate_of_cells": round(len(matches) / cells, 4) if cells else None,
               "n_consultations_with_a_match": len({m["consultation"] for m in matches}),
               "by_n_vendors": dict(Counter(m["n_vendors"] for m in matches)),
               "by_family": dict(Counter(m["family"] for m in matches).most_common()),
               "n_high_salience_matches": sum(1 for m in matches if m["high_salience"])}

    out = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "summary": summary, "matches": matches}
    jpath = os.path.join(T.MASTER, f"{prefix}cross_vendor_matches.json")
    json.dump(out, open(jpath, "w"), indent=1)

    L = ["# Cross-vendor matches - 'not a vendor bug'", "",
         "Same consultation, the same failure family appearing **independently in >=2 "
         "commercial scribes**. Verified findings only. The vendors' notes look nothing alike "
         "(terse SOAP vs long-form narrative) but the dangerous failure *types* recur.", "",
         f"**{len(matches)} cross-vendor matches** across "
         f"{summary['n_consultations_with_a_match']} consultations "
         f"({summary['by_n_vendors'].get(3, 0)} found in all three).", ""]
    for m in matches:
        L.append(f"## {m['consultation']} - {m['family']}  ({', '.join(m['vendors'])})"
                 f"{'  [HIGH-salience]' if m['high_salience'] else ''}")
        for s, ex in m["examples"].items():
            L.append(f"- **{s}**: {ex['description']}")
            if ex["note_quote"] and ex["note_quote"] != "-":
                L.append(f'  - note: "{ex["note_quote"]}"')
        L.append("")
    mpath = os.path.join(T.MASTER, f"{prefix}cross_vendor_matches.md")
    open(mpath, "w").write("\n".join(L))
    print(f"cross-vendor: {len(matches)} matches over {summary['n_consultations_with_a_match']} "
          f"consultations -> {os.path.relpath(jpath, HERE)}")
    return out


# ---------------------------------------------------------------- print
def print_summary(r):
    print("\n=== verified-finding rates ===")
    o = r["overall"]
    print(f"  overall: {o['n_findings']} verified over {o['n_notes']} notes = "
          f"{o['findings_per_note_mean']}/note (sd {o['findings_per_note_sd']}); "
          f"{o['notes_with_any']['p']:.1%} of notes carry >=1")
    for v, b in r["by_vendor"].items():
        print(f"  {v:6}: {b['findings_per_note_mean']:5.2f}/note over {b['n_notes']:4} notes | "
              f">=1 in {b['notes_with_any']['p']:.1%} "
              f"[{b['notes_with_any']['lo']:.2f},{b['notes_with_any']['hi']:.2f}] | "
              f"high-salience {b['high_salience']['findings_per_note_mean']:.2f}/note")
    for s, b in r["by_substrate"].items():
        print(f"  {s:10}: {b['findings_per_note_mean']:5.2f}/note over {b['n_notes']:4} notes")
    w = r["trapped_vs_trapblind"]
    if w.get("available"):
        ci = w["cluster_bootstrap_ci"]
        print(f"\n=== trapped vs trap-blind (finding-rate half) ===\n  authored {w['authored']['findings_per_note_mean']}"
              f"/note vs trap-blind {w['trapblind']['findings_per_note_mean']}/note; "
              f"diff {w['difference_findings_per_note']} "
              f"CI [{ci.get('lo')}, {ci.get('hi')}] -> {w['reading']}")
    sp = r["stability_probe"]
    print(f"\nstability probe: " + (f"runs {sp['runs']}, mean per-note sd {sp['mean_per_note_sd']} "
                                    f"over {sp['n_notes']} notes" if sp.get("available")
                                    else f"not available ({sp.get('why')})"))
    print("\n=== frame frequencies ===")
    li = r["literature_comparison_inputs"]
    for k, b in li["by_tier1"].items():
        if b["n_findings"] or k != "unmapped":
            print(f"  {k:24} {b['n_findings']:5} findings  {(b['share_of_findings'] or 0):6.1%} "
                  f"of all  {b['notes_with_any']['p']:6.1%} of notes carry >=1")
    print("  (Biro's own shares: omission 54-83%, addition 4-11%, wrong output 6-10%, "
          "misplaced 6-25% - two products, unverified against the primary source)")
    print("\ntop verified tier2:", dict(list(r["distributions"]["frame_tier2"].items())[:8]))

    pb = r.get("position_bias") or {}
    if pb.get("transcript_position"):
        print("\n=== positional decay check (AWS arXiv 2606.25656) ===")
        print(f"  quotes located: note {pb['locate_rate']['note_quote']:.1%}, "
              f"transcript {pb['locate_rate']['source_quote']:.1%} "
              f"of {pb['n_findings_considered']} candidates")
        for name in ("transcript_position", "note_position", "prompt_depth"):
            a = pb[name]["all_candidates"]
            v = pb[name]["panel_verified"]
            if a.get("n", 0) < 5:
                continue
            tag = "  [descriptive only - bounded by construction]" if name == "prompt_depth" else ""
            print(f"  {name:20} candidates mean {a['mean']:.3f} ({a['skew']}, "
                  f"first30% {a['share_first_30pct']:.1%} vs last30% {a['share_last_30pct']:.1%}, "
                  f"KS {a['ks_stat_vs_uniform']:.3f} p={a['ks_p']:.1e}){tag}")
            if v.get("n", 0) >= 5:
                print(f"  {'':20} verified   mean {v['mean']:.3f} ({v['skew']})")
        li = pb.get("length_interaction")
        if li:
            print("  transcript-position mean by transcript-length quartile "
                  "(decay would push these DOWN as length rises):")
            for b, d in li["transcript_position_mean_by_quartile"].items():
                if d.get("n", 0) >= 5:
                    print(f"    {b:12} n={d['n']:5} mean {d['mean']:.3f}")


if __name__ == "__main__":
    main()
