"""taxonomy_discover.py - W-D step 9b, stage 1: wide-cast failure discovery over every
commercial scribe note (scribe_A + scribe_B + scribe_C).

**The pass set is derived from the published frame** (`taxonomy_frame.json`, the lead author's
2026-08-12 decision 5), not from a list hardcoded here. Every Tier-2 frame category
marked `hunted: true` contributes one targeted pass, plus the open emergent pass. Nine
of the eleven targeted passes are June's `discover.py` MODE_PASSES entries, pulled from
that module at import so their text cannot drift; the two the frame adds carry their
instruction in the frame file itself. There is exactly one copy of every instruction
string in the repo, and `--print-passes` shows which is which.

The prompt TEMPLATES (`DIFFER`, `OPEN_DIFFER`) are imported verbatim from `discover.py`
- the June instrument that produced the 174 workshop findings. Discovery's block order
is unchanged too: measured on 2026-08-12, an Anthropic route via OpenRouter returns
zero cached prompt tokens across sequential identical-prefix calls (caching there needs
`cache_control` breakpoints, which live in W1-owned `common.llm()`), so reordering the
prompt transcript-first would buy no cache and would change the instrument for nothing.
The panel prompt IS reordered, because its OpenAI legs do cache - see taxonomy_verify.

Deliberately NOT ported from discover.py: its fact-sheet pass (`--source authored`
only). Running an extra check on one substrate would inflate that substrate's finding
rate, and the trapped-vs-trap-blind rate comparison (WD-R2) is exactly what this stage
feeds. So it is the same pass set for every note in the corpus.

    # what the frame says we hunt, and where each instruction comes from
    .venv/bin/python taxonomy_discover.py --print-passes

    # smoke: notes spanning all three vendors
    .venv/bin/python taxonomy_discover.py --smoke --allow-dirty -y

    # the full pass
    .venv/bin/python taxonomy_discover.py --route openrouter -y
"""
import argparse, json, os, time
from collections import Counter

from common import HERE, Run
import taxonomy_common as T

# The prompt templates, verbatim (see T.import_verbatim for why argv is blanked).
JUNE_PASSES, DIFFER, OPEN_DIFFER = T.import_verbatim("discover", ["MODE_PASSES", "DIFFER", "OPEN_DIFFER"])
TRANSCRIPT_CHARS = 40000                       # discover.py's own truncation, kept


def build_passes():
    """{pass_name: (instruction, provenance)} for every targeted pass the frame declares.

    A frame pass whose provenance says 'verbatim from discover.py' takes its text FROM
    discover.py, so the frame can never quietly restate the June instrument in different
    words. A frame pass discover.py does not have must carry its own `pass_instruction`.
    """
    out = {}
    for c in T.load_frame()["tier2"]:
        p = c.get("pass")
        if not c.get("hunted") or not p or p == "open":
            continue
        prov = c.get("pass_provenance", "")
        if p in JUNE_PASSES:
            if c.get("pass_instruction"):
                raise SystemExit(
                    f"frame pass {p!r} both restates an instruction and exists in discover.py - "
                    "one source only. Drop `pass_instruction` from the frame entry.")
            out[p] = (JUNE_PASSES[p], prov or "verbatim from discover.py MODE_PASSES")
        else:
            if not c.get("pass_instruction"):
                raise SystemExit(f"frame pass {p!r} is not in discover.py and carries no "
                                 "`pass_instruction` - nothing to send.")
            out[p] = (c["pass_instruction"], prov or "new in the frame")
    return out


MODE_PASSES = {k: v[0] for k, v in build_passes().items()}
PASS_PROVENANCE = {k: v[1] for k, v in build_passes().items()}
MODES = T.frame_passes()                       # targeted passes in frame order, then "open"


def print_passes():
    fr = T.load_frame()
    rec = fr["june_pass_reconciliation"]
    print(f"frame {fr['frame_version']} ({T.frame_sha256()[:12]}) - {len(MODES)} passes per note\n")
    for m in MODES:
        if m == "open":
            print(f"  open              [verbatim from discover.py OPEN_DIFFER] - emergent, "
                  "names its own mode")
            continue
        t2 = T.tier2_for_pass(m)
        print(f"  {m:18} -> tier2 {t2} -> tier1 {T.tier1_of_tier2(t2)}\n"
              f"    [{PASS_PROVENANCE[m]}] {MODE_PASSES[m][:110]}...")
    print(f"\nvs June's instrument: kept {len(rec['kept_verbatim'])}, "
          f"added {[a['pass'] for a in rec['added']]}, "
          f"renamed {rec['renamed'] or 'none'}, dropped {rec['dropped'] or 'none'}")
    for c in fr["tier2"]:
        if not c.get("hunted"):
            print(f"  NOT hunted: {c['key']} - {c['why_not_hunted'][:150]}...")


def pick_smoke_notes(units, n=6):
    """~n notes spanning all three vendors, both scribe_A templates and several substrates.

    Deterministic: walk the all-vendor consultations, taking one substrate at a time in
    turn so the sample is never all ACI or all authored, and alternate which scribe_A
    template each consultation contributes - so measured cost/note is not dominated by
    one template or one transcript length. Consultations are taken whole (all vendors),
    because a smoke that splits a consultation cannot exercise cross-vendor behaviour.
    """
    from collections import defaultdict
    have, by_src = defaultdict(set), defaultdict(list)
    for u in units:
        have[(u["source"], u["id"])].add(u["scribe"])
    for c, v in sorted(have.items()):
        if {"scribe_A", "scribe_B", "scribe_C"} <= v:
            by_src[c[0]].append(c)
    order, i = [], 0
    while any(len(v) > i for v in by_src.values()):
        for s in sorted(by_src):
            if len(by_src[s]) > i:
                order.append(by_src[s][i])
        i += 1
    out = []
    for i, c in enumerate(order):
        tmpl = "short" if i % 2 == 0 else "detailed"
        got = [u for u in units if (u["source"], u["id"]) == c
               and not (u["scribe"] == "scribe_A" and u["template"] != tmpl)]
        if len(out) + len(got) > n and out:
            break
        out += got
    return out


def main():
    ap = argparse.ArgumentParser(description="W-D step 9b stage 1: findings discovery")
    ap.add_argument("--route", default="openrouter", choices=["openrouter", "plan"])
    ap.add_argument("--discovery-role", default=T.CONSTRUCTOR_ROLE,
                    help="models.lock.json role that generates candidates (default constructor = "
                         "claude-opus-5, the June-comparable instrument). Changing this changes "
                         "the instrument, so use a DIFFERENT --tag: discovery records are keyed "
                         "on (note, pass, run) and would otherwise be reused across models.")
    ap.add_argument("--effort", default=T.REASONING_EFFORT,
                    help="reasoning_effort for the discovery role ('none' to send none)")
    ap.add_argument("--vendors", default=",".join(T.VENDORS))
    ap.add_argument("--sources", default=None, help="substrate filter, e.g. authored,primock")
    ap.add_argument("--modes", default=None, help="pass subset (default: every frame pass)")
    ap.add_argument("--limit", type=int, default=None, help="first N consultations")
    ap.add_argument("--smoke", action="store_true", help="notes spanning all three vendors")
    ap.add_argument("--smoke-notes", type=int, default=6, help="notes in the smoke sample")
    ap.add_argument("--print-passes", action="store_true",
                    help="show the frame-derived pass set and its provenance, then exit")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-buy passes whose stored record carries a transport/parse error. "
                         "A failed pass is recorded as 'no issues from that pass', which is the "
                         "right pre-registered fallback but is indistinguishable from a genuine "
                         "clean pass on resume - so the retry has to be asked for explicitly.")
    ap.add_argument("--stability", action="store_true",
                    help="restrict to the seeded 15-consultation stability subsample")
    ap.add_argument("--run", type=int, default=1, help="run index (stability probe uses 1,2,3)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=25, help="notes per Run() manifest")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--budget-usd", type=float, default=250.0)
    ap.add_argument("--stage-cap-usd", type=float, default=T.STAGE_CAP_DEFAULT,
                    help="cap on the WHOLE taxonomy stage across every script and resume")
    ap.add_argument("--warn-usd", type=float, default=1500.0)
    ap.add_argument("--stop-usd", type=float, default=2500.0)
    ap.add_argument("--authorize-spend", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true", help="smoke only: run on a dirty tree")
    ap.add_argument("--no-assemble", action="store_true", help="skip writing findings_master.json")
    ap.add_argument("-y", "--yes", action="store_true")
    args = ap.parse_args()

    if args.print_passes:
        print_passes()
        return

    modes = [m.strip() for m in args.modes.split(",")] if args.modes else MODES
    for m in modes:
        if m not in MODES:
            raise SystemExit(f"unknown pass {m!r}; known: {MODES}")

    units, skipped = T.note_units(vendors=[v.strip() for v in args.vendors.split(",")],
                                  sources=[s.strip() for s in args.sources.split(",")] if args.sources else None,
                                  limit=args.limit)
    all_units = units
    if args.stability:
        units = [u for u in units if (u["source"], u["id"]) in T.stability_consults(all_units)]
    if args.smoke:
        units = pick_smoke_notes(units, args.smoke_notes)

    tag = args.tag or ("discover-smoke" if args.smoke else
                       f"discover{'-stability' if args.stability else ''}-r{args.run}")
    store = T.RecordStore(T.EXPERIMENT, tag)

    jobs_all = [(u, m) for u in units for m in modes]

    def done(u, m):
        k = f"{u['note_key']}|{m}|r{args.run}"
        if not store.has(k):
            return False
        if args.retry_errors and (store.records.get(k) or {}).get("error"):
            return False
        return True

    todo = [(u, m) for u, m in jobs_all if not done(u, m)]
    n_err = sum(1 for r in store.records.values() if r.get("error"))
    print(T.corpus_report(all_units, skipped))
    print(f"discover | route={args.route} | run={args.run} | tag={tag}")
    print(f"  {len(units)} notes x {len(modes)} passes = {len(jobs_all)} calls; "
          f"{len(jobs_all) - len(todo)} already done, {len(todo)} to buy")
    if n_err:
        print(f"  {n_err} stored records carry an error"
              + (" - being retried" if args.retry_errors else
                 " - re-run with --retry-errors to re-buy them (they currently count as "
                 "'no issues from that pass')"))
    if args.smoke:
        for u in units:
            print(f"    smoke note: {u['note_key']}  ({len(u['transcript'])} transcript chars)")
    if not todo:
        print("nothing to do")
    else:
        est = "unbilled (Claude plan)" if args.route == "plan" else "billed to OpenRouter credits"
        T.confirm(f"buy {len(todo)} discovery calls ({est})?", args.yes)

        guard = T.BudgetGuard(budget=args.budget_usd, warn=args.warn_usd, stop=args.stop_usd,
                              authorized=args.authorize_spend, route=args.route,
                              stage_cap=args.stage_cap_usd)
        prompts = {"differ": DIFFER, "open_differ": OPEN_DIFFER,
                   "mode_passes": json.dumps(MODE_PASSES, indent=1),
                   "pass_provenance": json.dumps(PASS_PROVENANCE, indent=1)}
        by_note = {}
        for u, m in todo:
            by_note.setdefault(u["note_key"], (u, []))[1].append(m)
        pending = list(by_note.values())
        t0, made, usage = time.time(), 0, T.new_usage()

        for ci, chunk in enumerate(T.chunked(pending, args.chunk), 1):
            params = T.manifest_params(
                {"stage": "discover", "modes": modes, "run": args.run, "chunk": ci,
                 "discovery_role": args.discovery_role, "discovery_effort": args.effort,
                 "workers": args.workers, "tag": tag,
                 "record_store": os.path.relpath(store.path, HERE),
                 "smoke": args.smoke, "stability_subsample": args.stability},
                args.route, units)
            with Run(T.EXPERIMENT, params=params, replicate=args.run, seed=T.STABILITY_SEED,
                     inputs=T.run_inputs(), spec=T.SPEC, allow_dirty=args.allow_dirty,
                     spend=T.spend_label(args.route)) as run:
                run.register_prompts(prompts)

                def worker(job):
                    u, mode = job
                    key = f"{u['note_key']}|{mode}|r{args.run}"
                    if mode == "open":
                        prompt = OPEN_DIFFER.format(transcript=u["transcript"][:TRANSCRIPT_CHARS],
                                                    note=u["note"])
                    else:
                        prompt = DIFFER.format(mode=mode, instr=MODE_PASSES[mode],
                                               transcript=u["transcript"][:TRANSCRIPT_CHARS],
                                               note=u["note"])
                    obj, meta = T.route_call(prompt, key, args.route, kind="discover",
                                             role=args.discovery_role,
                                             effort=None if args.effort == "none" else args.effort)
                    issues = (obj or {}).get("issues", []) if isinstance(obj, dict) else []
                    if not isinstance(issues, list):
                        issues = []
                    clean = []
                    for i, it in enumerate(issues):
                        if not isinstance(it, dict):
                            continue
                        it = dict(it)
                        if mode == "open":
                            it["check"] = "open"           # keep the model's emergent label
                            it.setdefault("mode", "open")
                        else:
                            it["mode"], it["check"] = mode, "transcript"
                        it["finding_id"] = f"{u['note_key']}#{mode}#{i}"
                        it["pass"] = mode
                        # frame placement, stamped at birth for targeted passes; open-pass
                        # findings are placed later by taxonomy_analyze, which has the
                        # June family matcher loaded.
                        t2 = T.tier2_for_pass(mode)
                        if t2:
                            it["frame_tier2"], it["frame_tier1"] = t2, T.tier1_of_tier2(t2)
                            it["frame_placed_by"] = "pass"
                        clean.append(it)
                    rec = {"key": key, "note_key": u["note_key"], "scribe": u["scribe"],
                           "source": u["source"], "id": u["id"], "template": u["template"],
                           "consultation": u["consultation"], "pass": mode, "run": args.run,
                           "issues": clean, "n_issues": len(clean),
                           "route": meta.get("route"), "model": meta.get("model"),
                           "role": meta.get("role"),
                           "seed": meta.get("seed"), "error": meta.get("error"),
                           "usage": meta.get("usage"), "cost_usd": meta.get("cost_usd"),
                           "wall_s": meta.get("wall_s"),
                           "transcript_chars": len(u["transcript"][:TRANSCRIPT_CHARS]),
                           "note_chars": len(u["note"]),
                           "transcript_source": u["transcript_source"],
                           "run_id": run.run_id, "t": round(time.time(), 3)}
                    store.put(rec)
                    T.usage_add(usage, meta)
                    guard.add(meta.get("cost_usd"))

                jobs = [(u, m) for u, ms in chunk for m in ms]
                # No warm-up call: the June prompts put the mode instruction BEFORE the
                # transcript, so two passes over one note share almost no prefix and there
                # is no provider cache to prime. Reordering would change the instrument.
                made += T.run_block(jobs, worker, args.workers, warmup=False)
                print(f"  chunk {ci}: {len(jobs)} calls | {usage['errors']} errors "
                      f"| ${usage['cost_usd']:.2f} | {time.time() - t0:.0f}s", flush=True)
                run.save("discover_chunk.json",
                         {"tag": tag, "chunk": ci, "run": args.run, "params": params,
                          "note_keys": [u["note_key"] for u, _ in chunk]})

        store.close()
        print(f"\ndiscovery: {made} calls this session | {usage['errors']} transport/parse errors "
              f"| ${usage['cost_usd']:.4f}")
        if usage["calls"]:
            pt, ct = usage["prompt_tokens"], usage["completion_tokens"]
            print(f"  tokens: {pt:,} in + {ct:,} out ({usage['reasoning_tokens']:,} reasoning); "
                  f"${usage['cost_usd'] / max(usage['calls'], 1):.4f}/call")
            print(f"  {T.cache_line(usage)}")
            print(f"  stage spend: ${guard.stage_total:.2f} of ${guard.stage_cap:.0f}")

    if not args.no_assemble:
        assemble(store, tag, args.run)


def assemble(store, tag, run):
    """Roll the record store into master/findings_master.json (raw candidate findings)."""
    recs = [r for r in store.records.values() if r.get("run") == run]
    findings, per_note = [], {}
    for r in recs:
        pn = per_note.setdefault(r["note_key"], {
            "note_key": r["note_key"], "scribe": r["scribe"], "source": r["source"],
            "id": r["id"], "template": r["template"], "consultation": r["consultation"],
            "n_passes": 0, "n_failed_passes": 0, "n_issues": 0})
        pn["n_passes"] += 1
        pn["n_failed_passes"] += 1 if r.get("error") else 0
        pn["n_issues"] += r["n_issues"]
        for it in r["issues"]:
            findings.append({"note_key": r["note_key"], "scribe": r["scribe"], "source": r["source"],
                             "id": r["id"], "template": r["template"],
                             "consultation": r["consultation"], "run": run, **it})
    out = {"stage": "discover", "tag": tag, "run": run,
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "n_notes": len(per_note), "n_findings": len(findings),
           "n_calls": len(recs), "n_failed_calls": sum(1 for r in recs if r.get("error")),
           "cost_usd": round(sum(r.get("cost_usd") or 0 for r in recs), 6),
           "routes": dict(Counter(r.get("route") for r in recs)),
           "per_note": sorted(per_note.values(), key=lambda p: p["note_key"]),
           "all_issues": findings}
    # Only the canonical full pass owns the spec-named filename; every other tag
    # (smoke, stability runs 2/3, ad-hoc subsets) writes beside it.
    name = "findings_master.json" if tag == "discover-r1" else f"findings_master_{tag}.json"
    path = os.path.join(T.MASTER, name)
    json.dump(out, open(path, "w"), indent=1)
    print(f"\n=== {len(findings)} candidate findings across {len(per_note)} notes ===")
    print("by pass:     ", dict(Counter(f["pass"] for f in findings)))
    print("frame tier1: ", dict(Counter(f.get("frame_tier1") or "(open, placed later)"
                                        for f in findings).most_common()))
    print("by mode:     ", dict(Counter(f.get("mode") for f in findings).most_common(12)))
    print("by severity: ", dict(Counter(f.get("severity") for f in findings)))
    print("by salience: ", dict(Counter(f.get("salience") for f in findings)))
    print("by vendor:   ", dict(Counter(f["scribe"] for f in findings)))
    print("by substrate:", dict(Counter(f["source"] for f in findings)))
    with_any = sum(1 for p in out["per_note"] if p["n_issues"])
    print(f"notes with >=1 candidate: {with_any}/{len(per_note)}")
    print(f"saved -> {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    main()
