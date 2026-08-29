# browser-harness heredoc body: batch-drive scribe_C over consults (real-time via BlackHole).
# Flow (validated on consult08): New encounter -> Start encounter -> scribe_C_play.sh <wav> (real-time
# ~7-10min) -> Generate note -> poll -> Copy note -> pbpaste -> save scribe_C_notes/primock__<id>.txt.
# Reads /tmp/scribe_C_files.json [{id, wav}]. Writes /tmp/scribe_C_batch_results.json. Defensive per-consult.
import json, time, subprocess, os

HERE = os.path.dirname(os.path.abspath(__file__))
NOTES_DIR = os.path.join(HERE, "scribe_C_notes")
os.makedirs(NOTES_DIR, exist_ok=True)
files = json.load(open("/tmp/scribe_C_files.json"))
results = []


def click_btn(rx):
    return js("""(() => {const b=[...document.querySelectorAll('button,[role=button]')].find(x=>new RegExp(%s,'i').test((x.innerText||'').trim()) && (x.innerText||'').trim().length<30); if(b){b.click(); return true;} return false;})()""" % json.dumps(rx))


def process(item):
    rec = {"id": item["id"], "source": "primock", "scribe": "scribe_C", "status": "?"}
    # 1. new encounter
    click_btn("^new encounter$"); time.sleep(4)
    # 2. start encounter (record)
    if not click_btn("^start encounter$"):
        rec["status"] = "no_start"; return rec
    time.sleep(4)
    rec_state = js("""(() => /generate note|pause encounter/i.test(document.body.innerText))()""")
    if not rec_state:
        rec["status"] = "not_recording"; screenshot(f"/tmp/scribe_C_fail_{item['id']}.png"); return rec
    # 3. play audio real-time into BlackHole (blocks ~7-10 min)
    pr = subprocess.run(["bash", os.path.join(HERE, "scribe_C_play.sh"), item["wav"]], capture_output=True, text=True)
    time.sleep(3)
    # 4. generate note
    if not click_btn("generate note"):
        rec["status"] = "no_generate"; screenshot(f"/tmp/scribe_C_fail_{item['id']}.png"); return rec
    # 5. poll for the note (clinical headings), up to ~3min
    note = ""
    for _ in range(18):
        time.sleep(10)
        if js("""(() => /Chief Complaint|Presenting|History of Present|Assessment|Plan|Impression/i.test(document.body.innerText))()"""):
            click_btn("copy note"); time.sleep(1.5)
            note = subprocess.run(["pbpaste"], capture_output=True, text=True).stdout.strip()
            if len(note) > 150:
                break
    rec["note"] = note
    rec["note_len"] = len(note)
    rec["status"] = "ok" if len(note) > 150 else "empty_note"
    if len(note) > 150:
        open(os.path.join(NOTES_DIR, f"primock__{item['id']}.txt"), "w").write(note)
    else:
        screenshot(f"/tmp/scribe_C_fail_{item['id']}.png")
    return rec


for n, item in enumerate(files, 1):
    print(f"[{n}/{len(files)}] scribe_C/{item['id']} ...", flush=True)
    try:
        r = process(item)
    except Exception as e:
        r = {"id": item["id"], "status": "exc:" + str(e)[:80]}
    print(f"    -> {r['status']} (len={r.get('note_len','-')})", flush=True)
    results.append(r)
    json.dump(results, open("/tmp/scribe_C_batch_results.json", "w"), indent=1)

ok = sum(1 for r in results if r["status"] == "ok")
print(f"\nDONE: {ok}/{len(files)} scribe_C notes -> scribe_C_notes/")
