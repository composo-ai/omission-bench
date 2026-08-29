# browser-harness heredoc body: batch-drive scribe_B over an audio manifest.
# Pure UI automation (NO claude -p / Claude-plan usage). Reads /tmp/scribe_B_files.json
# [{id, mp3, source}], processes each (new session -> upload -> wait note -> scrape via
# Copy->pbpaste), saves scribe_B_notes/<source>__<id>.txt, and writes /tmp/scribe_B_batch_results.json.
# Defensive: per-file try/except, bounded polling, screenshot on failure, continue.
import json, time, subprocess, os

HERE = os.path.dirname(os.path.abspath(__file__))
NOTES_DIR = os.path.join(HERE, "scribe_B_notes")
os.makedirs(NOTES_DIR, exist_ok=True)
files = json.load(open("/tmp/scribe_B_files.json"))
results = []


def rect_of(text_rx):
    return js("""(() => {
      const rx=new RegExp(%s,'i');
      for(const el of document.querySelectorAll('button,a,[role=button],[role=menuitem],div,span,li,label')){
        const t=(el.innerText||el.textContent||'').trim();
        if(t && t.length<40 && rx.test(t)){const r=el.getBoundingClientRect(); if(r.width>0&&r.height>0) return {x:r.x,y:r.y,w:r.width,h:r.height,t};}
      }
      return null;})()""" % json.dumps(text_rx))


def click_text(text_rx):
    r = rect_of(text_rx)
    if not r:
        return False
    click(int(r["x"] + r["w"] / 2), int(r["y"] + r["h"] / 2))
    return True


def state():
    return js("""(() => ({
      uploadDialog:/Upload a recording/i.test(document.body.innerText),
      transcribing:/transcrib|processing|generating note|in progress/i.test(document.body.innerText),
      hasNote:/Presenting|History of Presenting|Assessment|Impression|Plan:|Past Medical/i.test(document.body.innerText)
    }))()""")


def process(item):
    rec = {"id": item["id"], "source": item["source"], "scribe": "scribe_B", "status": "?"}
    # 0. close any stuck dialog/modal
    for _ in range(2):
        try:
            cdp("Input.dispatchKeyEvent", type="keyDown", key="Escape", code="Escape", windowsVirtualKeyCode=27)
            cdp("Input.dispatchKeyEvent", type="keyUp", key="Escape", code="Escape", windowsVirtualKeyCode=27)
        except Exception:
            pass
        time.sleep(0.4)
    # 1. new session
    if not click_text("^New session$"):
        click(96, 72)
    # 2. wait for the Transcribe chevron to actually render (page can skeleton-load), then click it
    ch = None
    for _ in range(20):  # up to ~40s
        time.sleep(2)
        ch = js("""(() => {const el=document.querySelector('[aria-label="Open transcription mode menu"]');
                  if(!el)return null; const r=el.getBoundingClientRect();
                  return r.width>0?{x:Math.round(r.x+r.width/2),y:Math.round(r.y+r.height/2)}:null;})()""")
        if ch:
            break
    if not ch:
        rec["status"] = "no_chevron"
        screenshot(f"/tmp/scribe_B_fail_{item['id']}.png")
        return rec
    click(int(ch["x"]), int(ch["y"]))
    time.sleep(1.5)
    # 3. Upload session audio
    if not click_text("Upload session audio"):
        rec["status"] = "no_upload_menu"
        screenshot(f"/tmp/scribe_B_fail_{item['id']}.png")
        return rec
    time.sleep(1.5)
    # 4. upload the file
    try:
        cdp("Page.setInterceptFileChooserDialog", enabled=True)
    except Exception:
        pass
    try:
        upload_file("input[type=file]", item["mp3"])
    except Exception as e:
        rec["status"] = "upload_err:" + str(e)[:60]
        return rec
    # 5. wait for the note editor to populate with the clinical note (bounded ~9 min)
    SCRAPE = """(() => {
      for (const ce of document.querySelectorAll('[contenteditable=true]')) {
        const t=(ce.innerText||'').trim();
        if (t.length>150 && /Presenting|History|Assessment|Impression|Plan|Complaint|Examination/i.test(t)) return t;
      }
      let best=''; document.querySelectorAll('div,section,article').forEach(el=>{
        const t=(el.innerText||'').trim();
        if(/Presenting|Assessment|Impression|Plan/i.test(t) && t.length>150 && t.length<7000 && !/New session|My Templates|Notifications/i.test(t)){
          if(!best||t.length<best.length) best=t;}});
      return best;})()"""
    deadline = time.time() + 540
    note = ""
    while time.time() < deadline:
        time.sleep(10)
        note = js(SCRAPE) or ""
        if len(note) > 150:
            time.sleep(4)  # let it finish rendering
            note = js(SCRAPE) or note
            break
    rec["note"] = note
    rec["note_len"] = len(note)
    rec["status"] = "ok" if len(note) > 150 else "empty_note:" + json.dumps(state())
    if len(note) > 120:
        open(os.path.join(NOTES_DIR, f"{item['source']}__{item['id']}.txt"), "w").write(note)
    else:
        screenshot(f"/tmp/scribe_B_fail_{item['id']}.png")
    return rec


for n, item in enumerate(files, 1):
    print(f"[{n}/{len(files)}] {item['source']}/{item['id']} ...", flush=True)
    r = {"id": item["id"], "source": item["source"], "status": "?"}
    for attempt in range(2):  # retry once on failure (e.g. transient backend error)
        try:
            r = process(item)
        except Exception as e:
            r = {"id": item["id"], "source": item["source"], "status": "exc:" + str(e)[:80]}
        if r.get("status") == "ok":
            break
        time.sleep(8)  # backend breather before retry
    print(f"    -> {r['status']} (len={r.get('note_len','-')})", flush=True)
    results.append(r)
    json.dump(results, open("/tmp/scribe_B_batch_results.json", "w"), indent=1)
    time.sleep(5)  # pace between consults to avoid backend throttling

ok = sum(1 for r in results if r["status"] == "ok")
print(f"\nDONE: {ok}/{len(files)} notes scraped -> scribe_B_notes/ ; results -> /tmp/scribe_B_batch_results.json")
