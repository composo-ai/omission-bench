"""precision_sitting.py - builds the blind adjudication pack for the census precision sitting.

The cold reads of P2 and its GenAI4Health cut converge on one central soundness gap: no human
has assessed a random sample of the 618 verified findings, so the census headline has a
measured floor and an unmeasured ceiling (a panel that upholds a candidate the transcript
does not support inflates the count, and nothing bounds that). This pack is the fix the
reviewer named: a physician blind-adjudication of randomly sampled verified findings,
reported as a precision estimate.

Design (BRIEF-census-strong-accept item 1):
  - 30 verified findings drawn uniformly at random from the 618 (seed below).
  - 15 foils: panel-REFUSED candidates drawn uniformly from the high-salience refused
    stratum (panel's own salience re-rating = high), so they are not trivially
    distinguishable from survivors. The foils are what makes "blind" true: without them
    the adjudicator knows every item was verified.
  - All 45 shuffled with the same seeded RNG; no status label anywhere in the pack.
  - Per item: the finding's mode + description, its evidence quotes, the FULL note and
    the full transcript (truncated at the same 40,000 characters the panel saw). The A.4
    sitting printed only the relevant fragment, which is exactly why it could not
    adjudicate the present-elsewhere bucket - not repeated here.
  - Two questions per item: the verdict (genuine failure a reviewing clinician should
    accept / not a genuine failure / cannot judge), and - only where the verdict is
    genuine - a severity grade against the written rubric (printed once at the top), so
    the same sitting carries a severity regrade on the sampled items without
    contaminating the precision read (the severity question is asked identically on
    every genuine item, foil or not).

Outputs (results/precision-sitting/):
  adjudication-pack.md   the pack the lead author reads - contains NO status information
  key.json               item -> finding_id, status, panel verdict detail (for unblinding)
  manifest.json          seed, draw sizes, sha256 of both id lists

    .venv/bin/python precision_sitting.py
"""
import hashlib, json, os, random

from common import HERE, RESULTS
import taxonomy_common as T

SEED = 20260825
N_VERIFIED = 30
N_FOILS = 15
VERIFIED = "master/findings_verified_master.json"
RUBRIC = "specs/severity-rubric.md"
OUTDIR = os.path.join(RESULTS, "precision-sitting")
TRANSCRIPT_CHARS = 40000   # what the panel saw (taxonomy_verify.TRANSCRIPT_CHARS)


def sha(ids):
    return hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()


def main():
    d = json.load(open(os.path.join(HERE, VERIFIED)))
    ai = d["all_issues"]
    if len(ai) != 5898:
        raise SystemExit(f"master holds {len(ai)} candidates, expected 5,898")
    verified = [f for f in ai if (f.get("verdict") or {}).get("is_real")]
    if len(verified) != 618:
        raise SystemExit(f"{len(verified)} verified, expected 618")
    refused_high = [f for f in ai if not (f.get("verdict") or {}).get("is_real")
                    and (f.get("verdict") or {}).get("salience") == "high"]

    rng = random.Random(SEED)
    sample_v = rng.sample(sorted(verified, key=lambda f: f["finding_id"]), N_VERIFIED)
    sample_f = rng.sample(sorted(refused_high, key=lambda f: f["finding_id"]), N_FOILS)
    items = [(f, "verified") for f in sample_v] + [(f, "refused_foil") for f in sample_f]
    rng.shuffle(items)

    all_units, _skipped = T.note_units()
    units = {u["note_key"]: u for u in all_units}
    rubric = open(os.path.join(HERE, RUBRIC)).read().strip()

    L = ["# Census precision sitting - blind adjudication pack",
         "",
         "45 candidate findings from the census's audit of the three scribe products, in random",
         "order. Some survived the verification panel and some the panel refused; the mix and the",
         "order are concealed, and nothing in an item says which it is. For each item you get the",
         "candidate finding (its claimed failure mode, description and evidence quotes), then the",
         "FULL note the product wrote, then the full consultation transcript.",
         "",
         "**Question 1, every item - the verdict.** Reading the note against the transcript, is this",
         "a genuine documentation failure a reviewing clinician should accept as real?",
         "",
         "- `genuine` - a real failure; the note asserts, omits or distorts something the",
         "  transcript does not support, in a way a reviewing clinician would accept as a defect",
         "- `not-genuine` - not a real failure (the claim misreads the note or transcript, the",
         "  content is actually captured, the inference is clinically justified, etc.)",
         "- `cannot-judge` - the materials are insufficient to decide",
         "",
         "**Question 2, only where your verdict is `genuine` - the severity grade**, against the",
         "rubric below: `critical` / `supporting` / `peripheral`.",
         "",
         "Answer as one line per item, e.g. `7: genuine critical` or `12: not-genuine`",
         "or `31: cannot-judge`. An answer sheet to copy is at the end of the pack.",
         "",
         "---",
         "",
         "## The severity rubric (for question 2)",
         "",
         rubric,
         "",
         "---", ""]

    key = []
    for i, (f, status) in enumerate(items, 1):
        u = units.get(f["note_key"])
        if not u:
            raise SystemExit(f"no note unit for {f['note_key']}")
        L += [f"## Item {i}",
              "",
              f"**Claimed failure mode**: {f.get('mode')}",
              "",
              f"**The candidate finding**: {f.get('description')}",
              "",
              f"**Note-side evidence quote**: {f.get('note_quote') or '-'}",
              "",
              f"**Transcript-side evidence quote**: {f.get('source_quote') or '-'}",
              "",
              f"### The full note (Scribe {'A' if f['scribe']=='scribe_A' else 'B' if f['scribe']=='scribe_B' else 'C'})",
              "", "```", u["note"].strip(), "```", "",
              "### The full transcript",
              "", "```", u["transcript"][:TRANSCRIPT_CHARS].strip(), "```", "",
              f"**Item {i} verdict**: `genuine` / `not-genuine` / `cannot-judge`; if genuine,",
              "severity `critical` / `supporting` / `peripheral`.",
              "", "---", ""]
        key.append({"item": i, "finding_id": f["finding_id"], "status": status,
                    "note_key": f["note_key"], "consultation": f["consultation"],
                    "scribe": f["scribe"], "pass": f.get("pass"),
                    "frame_tier2": f.get("frame_tier2"),
                    "panel_salience": (f.get("verdict") or {}).get("salience"),
                    "severity_rubric": f.get("severity_rubric"),
                    "panel_decided_by": "tiebreak" if (f.get("verdict") or {}).get("split")
                                        else "unanimous"})

    L += ["## Answer sheet", "", "```"]
    L += [f"{i}: " for i in range(1, len(items) + 1)]
    L += ["```", ""]

    os.makedirs(OUTDIR, exist_ok=True)
    pack = os.path.join(OUTDIR, "adjudication-pack.md")
    with open(pack, "w") as fh:
        fh.write("\n".join(L))
    with open(os.path.join(OUTDIR, "key.json"), "w") as fh:
        json.dump(key, fh, indent=1)
    man = {"seed": SEED, "n_verified_pool": len(verified), "n_refused_high_pool": len(refused_high),
           "n_verified_drawn": N_VERIFIED, "n_foils_drawn": N_FOILS,
           "verified_ids_sha256": sha([f["finding_id"] for f in sample_v]),
           "foil_ids_sha256": sha([f["finding_id"] for f in sample_f]),
           "transcript_chars": TRANSCRIPT_CHARS,
           "design": "uniform draw from the 618 verified; foils uniform from the "
                     "high-salience refused stratum (panel re-rating); shuffled; "
                     "verdict + conditional severity grade per item"}
    with open(os.path.join(OUTDIR, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    n_cons = len({k['consultation'] for k in key})
    print(f"wrote {pack} ({os.path.getsize(pack)//1024} KB), {len(items)} items over "
          f"{len({k['note_key'] for k in key})} notes / {n_cons} consultations")


if __name__ == "__main__":
    main()
