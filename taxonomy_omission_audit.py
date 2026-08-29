"""taxonomy_omission_audit.py - why did the panel refute the omissions?

Built on the coordinator's 2026-08-12 instruction. The smoke saw ZERO of 15 filtered
`omission` candidates survive the two-skeptic panel, and omission is the category the
paper's whole argument rests on: the claim is that faithfulness judges are structurally
blind to it. A near-zero omission rate is therefore either the study's most important
result or its most serious instrument failure, and the two look identical in a table.

This script reads the refusals and classifies them. A stratified sample of REFUTED
omission candidates goes to an impartial model (the auditor role - not the model that
generated the candidate) together with the FULL note, and each refusal is placed in one
of four buckets:

  a. present_elsewhere - the allegedly-missing fact IS in the note, somewhere other than
     where the differ looked. This is a partial-capture / redundancy phenomenon, NOT a
     non-finding: the note is disorganised rather than incomplete. At high rate this
     means the refute standard needs a partial-omission-aware revision before any
     omission rate is publishable, because "documented in the wrong place" is being
     scored as "documented".
  b. not_required - the fact is genuinely absent but a competent clinician would not
     have been expected to document it (scope, not omission).
  c. not_material - documented-absent and legitimately omissible: real but clinically
     immaterial. This is a salience judgement and the panel is entitled to make it.
  d. spurious - the candidate misread the transcript; the fact was never there.

Only (d) - and arguably (b) - is a clean refutation. A pool dominated by (a) means the
category rates are measuring the wrong thing.

The classifier is told to quote the note text that grounds an (a) verdict, so every
`present_elsewhere` call is checkable rather than asserted.

    .venv/bin/python taxonomy_omission_audit.py                     # default: omission, n=20
    .venv/bin/python taxonomy_omission_audit.py --category omission --n 30
"""
import argparse, json, os, random, time
from collections import Counter, defaultdict

from common import HERE, Run
import taxonomy_common as T

CLASSIFY = """FULL CLINICAL NOTE:
{note}

---
A discovery tool flagged the following as an OMISSION - clinically significant content present in the consultation transcript but missing from the note above. An adversarial reviewer then REFUTED it.

claimed missing content: {description}
transcript says: {source_quote}
reviewer's stated reason for refuting: {reasons}

Your job is to work out WHY this refutation happened. Read the full note carefully before answering.

Classify into exactly one bucket:
- "present_elsewhere": the allegedly-missing content IS in the note somewhere - a different section, different wording, folded into another sentence. The note recorded it; the discovery pass just did not find it where it looked. If you choose this you MUST quote the note text that contains it.
- "not_required": the content is genuinely absent from the note, but a competent clinician writing this note would not be expected to document it.
- "not_material": the content is genuinely absent and a clinician might have documented it, but its absence would not change anyone's understanding of the patient or their care.
- "spurious": the claim is wrong - the transcript does not actually contain the content, or the discovery tool mischaracterised it.
- "wrongly_cut": the refutation was WRONG. The content is genuinely absent from the note, a competent clinician WOULD be expected to document it, and its absence could change a reader's understanding of the patient or their care. Choose this whenever the flagged omission looks like a real, reportable documentation failure that the reviewer talked down.

Do not favour the reviewer. The reviewer was instructed to refute if at all defensible and may have over-applied that instruction; "wrongly_cut" exists precisely to catch that, and choosing it when it fits is the point of this exercise.

Return ONLY JSON {{"bucket":"present_elsewhere"|"not_required"|"not_material"|"spurious"|"wrongly_cut","note_quote":"<the note text proving present_elsewhere, or '-'>","reason":"<one line>"}}."""


def stratified(pool, n, seed=20260812):
    """Sample across vendor x substrate so the answer is not one vendor's habit."""
    buckets = defaultdict(list)
    for f in pool:
        buckets[(f.get("scribe"), f.get("source"))].append(f)
    for k in buckets:
        random.Random(seed).shuffle(buckets[k])
    out, i = [], 0
    while len(out) < n and any(len(v) > i for v in buckets.values()):
        for k in sorted(buckets):
            if len(buckets[k]) > i and len(out) < n:
                out.append(buckets[k][i])
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser(description="classify why the panel refuted a category")
    ap.add_argument("--in", dest="infile", default="master/findings_verified_master.json")
    ap.add_argument("--category", default="omission", help="frame tier-2 category to audit")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--tag", default=None,
                    help="record-store tag. Change it when the CLASSIFY prompt changes: "
                         "finding_ids are (note_key, pass, index) and the smoke notes are a "
                         "subset of the full corpus, so ids CAN collide and a stale record "
                         "would silently answer with a retired instrument.")
    ap.add_argument("--route", default="openrouter", choices=["openrouter", "plan"])
    ap.add_argument("--role", default=T.AUDITOR_ROLE,
                    help="classifier role - must NOT be the model that generated the candidates")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget-usd", type=float, default=15.0)
    ap.add_argument("--stage-cap-usd", type=float, default=T.STAGE_CAP_DEFAULT)
    ap.add_argument("--authorize-spend", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    data = json.load(open(os.path.join(HERE, args.infile)))
    cat = args.category
    refuted = [f for f in data["all_issues"]
               if (f.get("frame_tier2") or f.get("pass")) == cat
               and not (f.get("verdict") or {}).get("is_real")]
    verified = [f for f in data["all_issues"]
                if (f.get("frame_tier2") or f.get("pass")) == cat
                and (f.get("verdict") or {}).get("is_real")]
    total = len(refuted) + len(verified)
    if not refuted:
        raise SystemExit(f"no refuted {cat!r} candidates in {args.infile}")
    print(f"{cat}: {len(verified)}/{total} survived the panel "
          f"({len(verified) / total:.1%}); auditing {min(args.n, len(refuted))} of "
          f"{len(refuted)} refusals")

    sample = stratified(refuted, args.n)
    print("  sample spans:", dict(Counter(f"{f.get('scribe')}/{f.get('source')}" for f in sample)))

    units, _ = T.note_units()
    notes = {u["note_key"]: u for u in units}
    store = T.RecordStore(T.EXPERIMENT, args.tag or f"omission-audit-{cat}")
    todo = [f for f in sample if not store.has(f"audit|{f['finding_id']}")]
    if todo:
        T.confirm(f"buy {len(todo)} classification calls on {args.role}?", args.yes)
    guard = T.BudgetGuard(budget=args.budget_usd, authorized=args.authorize_spend,
                          route=args.route, stage_cap=args.stage_cap_usd)
    usage = T.new_usage()

    if todo:
        params = T.manifest_params(
            {"stage": "omission-audit", "category": cat, "n_sampled": len(sample),
             "classifier_role": args.role, "n_refuted_total": len(refuted),
             "survival_rate": round(len(verified) / total, 4)}, args.route, units)
        with Run(T.EXPERIMENT, params=params, inputs=T.run_inputs([args.infile]), spec=T.SPEC,
                 allow_dirty=args.allow_dirty, spend=T.spend_label(args.route)) as run:
            run.register_prompts({"classify_refusal": CLASSIFY})

            def worker(f):
                u = notes.get(f["note_key"])
                if not u:
                    return
                reasons = f["verdict"].get("reasons") or {}
                if isinstance(reasons, dict):
                    reasons = "; ".join(f"{k}: {v}" for k, v in reasons.items() if v)
                obj, meta = T.route_call(
                    CLASSIFY.format(note=u["note"], description=f.get("description"),
                                    source_quote=f.get("source_quote", "-"), reasons=reasons or "-"),
                    f"audit|{f['finding_id']}", args.route, kind="panel",
                    role=args.role, effort="medium")
                b = (obj or {}).get("bucket") if isinstance(obj, dict) else None
                if b not in ("present_elsewhere", "not_required", "not_material", "spurious",
                             "wrongly_cut"):
                    b = None
                store.put({"key": f"audit|{f['finding_id']}", "finding_id": f["finding_id"],
                           "note_key": f["note_key"], "scribe": f.get("scribe"),
                           "source": f.get("source"), "bucket": b,
                           "note_quote": (obj or {}).get("note_quote") if isinstance(obj, dict) else None,
                           "reason": (obj or {}).get("reason") if isinstance(obj, dict) else None,
                           "description": f.get("description"),
                           "salience": f.get("salience"),
                           "severity_differ": f.get("severity"),
                           "source_quote": f.get("source_quote"),
                           "panel_reasons": reasons, "error": meta.get("error"),
                           "cost_usd": meta.get("cost_usd"), "run_id": run.run_id})
                T.usage_add(usage, meta)
                guard.add(meta.get("cost_usd"))

            T.run_block(todo, worker, args.workers, warmup=False)
        store.close()

    got = [store.records[f"audit|{f['finding_id']}"] for f in sample
           if store.has(f"audit|{f['finding_id']}")]
    counts = Counter(r.get("bucket") for r in got)
    n = len(got)
    out = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "category": cat, "source_file": args.infile, "classifier_role": args.role,
           "n_candidates_total": total, "n_verified": len(verified),
           "survival_rate": round(len(verified) / total, 4),
           "n_refuted": len(refuted), "n_audited": n,
           "buckets": dict(counts),
           "bucket_shares": {k: round(v / n, 4) for k, v in counts.items()} if n else {},
           # Survival by salience is the second half of the question. If HIGH-salience
           # omissions survive at a reasonable rate and only medium ones are cut as
           # immaterial, a low overall rate is the panel doing its job. If high-salience
           # omissions are also near zero, the bar is suppressing the study's headline.
           "survival_by_salience": {
               s: {"n": sum(1 for f in refuted + verified if f.get("salience") == s),
                   "verified": sum(1 for f in verified if f.get("salience") == s)}
               for s in ("high", "medium", "low")},
           "refused_by_salience_x_bucket": dict(Counter(
               f"{r.get('salience')}|{r.get('bucket')}" for r in got)),
           "cost_usd": round(usage["cost_usd"], 6),
           "records": got}
    path = os.path.join(T.MASTER, f"omission_audit_{args.tag or cat}.json")
    json.dump(out, open(path, "w"), indent=1)

    print(f"\n=== why the panel refuted {cat} (n={n}) ===")
    for b in ("wrongly_cut", "present_elsewhere", "not_required", "not_material",
              "spurious", None):
        if counts.get(b):
            print(f"  {str(b):18} {counts[b]:3}  ({counts[b] / n:.0%})")
    print("\nsurvival by the candidate's salience:")
    for s, b in out["survival_by_salience"].items():
        if b["n"]:
            print(f"  {s:7} {b['verified']:4}/{b['n']:<4} = {b['verified'] / b['n']:6.1%}")
    hi_cut = [r for r in got if r.get("salience") == "high"]
    if hi_cut:
        print(f"  ! {len(hi_cut)} of the audited refusals were HIGH salience - "
              f"buckets {dict(Counter(r.get('bucket') for r in hi_cut))}")
    pe = [r for r in got if r.get("bucket") == "present_elsewhere"]
    if pe:
        print(f"\n-- present_elsewhere examples (the note DID record it) --")
        for r in pe[:5]:
            print(f"  [{r['scribe']}/{r['source']}] claimed missing: {(r['description'] or '')[:95]}")
            print(f"     note actually says: {(r.get('note_quote') or '-')[:110]}")
    wc = [r for r in got if r.get("bucket") == "wrongly_cut"]
    if wc:
        print(f"\n-- wrongly_cut examples (the panel talked down a real omission) --")
        for r in wc[:6]:
            print(f"  [{r['scribe']}/{r['source']}] ({r.get('salience')}) "
                  f"{(r['description'] or '')[:100]}")
            print(f"     why: {(r.get('reason') or '')[:130]}")
    wshare = counts.get("wrongly_cut", 0) / n if n else 0
    print(f"\nWRONGLY CUT: {counts.get('wrongly_cut', 0)}/{n} = {wshare:.0%} of sampled refusals")
    if wshare >= 0.30:
        print("!! The panel is over-cutting this category. The published rate is an UNDERCOUNT "
              "and must be disclosed as one - and if this is omission, it is the same blind spot "
              "the paper documents in judges, appearing in our own ground-truth instrument.")
    share = counts.get("present_elsewhere", 0) / n if n else 0
    print()
    if share >= 0.30:
        print(f"!! {share:.0%} of refusals are 'present_elsewhere'. The panel is scoring "
              "'documented in the wrong place' as 'documented', so the omission rate is NOT a "
              "clean measure of missing content. STOP: the refute standard needs a "
              "partial-omission-aware revision before these category rates are published.")
    elif share >= 0.15:
        print(f"! {share:.0%} of refusals are 'present_elsewhere' - material but not dominant. "
              "Report the omission rate WITH this share stated beside it.")
    else:
        print(f"'present_elsewhere' is {share:.0%} of refusals - the refute standard is not "
              "systematically mistaking misplacement for completeness. The low survival rate "
              "reads as a real property of the panel's bar, not an artefact.")
    print(f"saved -> {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    main()
