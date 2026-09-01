"""The adversarial critic panel over newly authored scenarios.

Three independent critics per scenario, then auto-revise any flagged scenario and re-audit it:
  - realism      : a senior UK GP - is the consult clinically realistic and error-free?
  - factsheet    : internal-consistency audit - is every must_contain in the transcript? is every
                   salience_trap a real moment? is every must_not_contain a genuine error?
  - contamination: does it read like a natural recorded transcript, or leak test-artifact/meta text?

Usage: python corpus/critique_scenarios.py [--in authored_scenarios_new.json] [--workers 6] [--no-revise]
                                    [--critics realism,contamination] [--out F.json] [--report F.md]
Output: critiqued_scenarios.json (final, possibly revised) + critique_report.md (audit trail).

--critics restricts the panel: the trap-blind batch runs realism+contamination only - its
scenarios carry no fact_sheet, which is also why _scn_text and revise() tolerate a missing
fact_sheet key. Default = all three critics, unchanged behaviour.
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
from common import claude_json, HERE

MODEL = "claude-opus-4-8"
EFFORT = "medium"
TIMEOUT = 420


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


INFILE = _arg("--in", "authored_scenarios_new.json")
WORKERS = int(_arg("--workers", "6"))
REVISE = "--no-revise" not in sys.argv
OUTFILE = _arg("--out", "critiqued_scenarios.json")
REPORT = _arg("--report", "critique_report.md")


def _scn_text(s):
    txt = (f"id: {s['id']}\npresenting_complaint: {s.get('presenting_complaint','')}\n\n"
           f"TRANSCRIPT:\n{s['transcript']}")
    if "fact_sheet" in s:
        txt += f"\n\nFACT_SHEET:\n{json.dumps(s['fact_sheet'], ensure_ascii=False, indent=1)}"
    return txt


CRITICS = {
 "realism": """You are a senior UK GP. Review ONLY whether this mock primary-care consultation
TRANSCRIPT is clinically realistic and free of errors. Flag: clinically wrong or implausible content,
unsafe/incorrect management, wrong drug/dose for the situation, dialogue a real GP or patient would
never say, or anything that breaks realism. Ignore the fact_sheet. Be strict but fair - minor stylistic
nits are not issues.""",
 "factsheet": """You are auditing INTERNAL CONSISTENCY between the transcript and its fact_sheet.
Check three things and flag any failure: (a) every `must_contain` item is genuinely supported by the
transcript (flag any that isn't actually said); (b) every `salience_trap` corresponds to a REAL moment
in the dialogue where a scribe could plausibly get it wrong (flag any trap with no such moment); (c)
every `must_not_contain` is a genuine ERROR for this consult (flag any that the transcript actually
supports, i.e. would be wrongly penalised). Also flag any internal contradiction.""",
 "contamination": """Does this TRANSCRIPT read like a natural, real recorded GP consultation? Flag any
test-artifact / contamination: narrator or meta text, the dialogue telegraphing or labelling the
'trap'/answer, unnatural phrasing that signals it's a constructed test, anything an AI-author leaked,
or any real-world patient identifier. A clean transcript reads exactly like a real consult.""",
}

ACTIVE = [c.strip() for c in (_arg("--critics") or ",".join(CRITICS)).split(",") if c.strip()]
assert all(c in CRITICS for c in ACTIVE), f"unknown critic in --critics: {ACTIVE}"
RECHECK = "factsheet" if "factsheet" in ACTIVE else ACTIVE[0]


def run_critic(scn, name):
    prompt = (f"{CRITICS[name]}\n\n{_scn_text(scn)}\n\n"
              'Output ONLY JSON: {"verdict": "ok" | "issues", "issues": [ {"what": <concise issue>, '
              '"severity": "minor"|"material"} ]}. Empty issues list if verdict is ok.')
    r = claude_json(prompt, model=MODEL, effort=EFFORT, timeout=TIMEOUT, retries=1)
    if not r:
        return {"verdict": "error", "issues": []}
    return r


def revise(scn, all_issues):
    issues_txt = "\n".join(f"- [{i.get('severity','?')}] ({src}) {i.get('what','')}"
                           for src, items in all_issues.items() for i in items)
    has_fs = "fact_sheet" in scn
    keep = "(style, length, the deliberate salience traps)" if has_fs else "(style, length)"
    consistency = ("Maintain strict internal consistency: the fact_sheet must derive only from the "
                   "transcript." if has_fs else "Do NOT add a fact_sheet or any other key.")
    keys = "id, presenting_complaint, transcript" + (", fact_sheet" if has_fs else "")
    prompt = (f"Here is a mock GP consultation scenario and the issues a review panel found. FIX ONLY "
              f"these issues; keep everything else {keep} intact. "
              f"{consistency}\n\n"
              f"{_scn_text(scn)}\n\nISSUES TO FIX:\n{issues_txt}\n\n"
              f"Output ONLY the full corrected JSON object with keys: {keys}. "
              f"Escape newlines inside strings as \\n.")
    obj = claude_json(prompt, model=MODEL, effort="high", timeout=540, retries=2)
    if obj and "transcript" in obj and (not has_fs or "fact_sheet" in obj):
        if not has_fs:
            obj.pop("fact_sheet", None)
        obj["id"] = scn["id"]
        obj.setdefault("presenting_complaint", scn.get("presenting_complaint"))
        if "_cluster" in scn:
            obj["_cluster"] = scn["_cluster"]
        return obj
    return None


def main():
    scns = [s for s in json.load(open(os.path.join(HERE, INFILE))) if "_error" not in s]
    print(f"Critiquing {len(scns)} scenarios ({WORKERS} workers, revise={REVISE}, critics={ACTIVE}).")

    jobs = [(s, c) for s in scns for c in ACTIVE]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(lambda j: (j[0]["id"], j[1], run_critic(j[0], j[1])), jobs))

    by_id = {s["id"]: {"scn": s, "critics": {}} for s in scns}
    for sid, cname, res in results:
        by_id[sid]["critics"][cname] = res

    # decide revisions (any material issue)
    to_revise = []
    for sid, d in by_id.items():
        material = {c: [i for i in r.get("issues", []) if i.get("severity") == "material"]
                    for c, r in d["critics"].items()}
        d["material"] = {c: v for c, v in material.items() if v}
        if REVISE and d["material"]:
            to_revise.append(sid)

    print(f"Flagged for revision (material issues): {to_revise or 'none'}")
    if to_revise:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            revised = list(ex.map(lambda sid: (sid, revise(by_id[sid]["scn"], by_id[sid]["material"])), to_revise))
        for sid, robj in revised:
            if robj:
                by_id[sid]["scn"] = robj
                by_id[sid]["revised"] = True
                # re-audit on revised version (factsheet critic, or the first active one)
                by_id[sid]["recheck"] = run_critic(robj, RECHECK)
            else:
                by_id[sid]["revise_failed"] = True

    # write final scenarios (stripped of _cluster kept as metadata) + report
    final = [d["scn"] for d in by_id.values()]
    json.dump(final, open(os.path.join(HERE, OUTFILE), "w"), ensure_ascii=False, indent=1)

    lines = ["# Critic-panel report\n"]
    n_clean = 0
    for sid, d in by_id.items():
        flags = []
        for c, r in d["critics"].items():
            for i in r.get("issues", []):
                flags.append(f"  - [{c}/{i.get('severity','?')}] {i.get('what','')}")
        if not flags:
            n_clean += 1
        status = "REVISED" if d.get("revised") else ("REVISE-FAILED" if d.get("revise_failed") else "clean" if not flags else "minor-only")
        lines.append(f"## {sid} - {status}")
        lines += flags or ["  (no issues)"]
        if d.get("recheck"):
            rc = d["recheck"]
            lines.append(f"  re-audit after revision: {rc.get('verdict')} "
                         f"({len(rc.get('issues',[]))} residual)")
        lines.append("")
    open(os.path.join(HERE, REPORT), "w").write("\n".join(lines))
    print(f"clean {n_clean}/{len(scns)} | revised {len(to_revise)} -> {OUTFILE} + {REPORT}")


if __name__ == "__main__":
    main()
