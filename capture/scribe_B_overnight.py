#!/usr/bin/env python3
"""scribe_B batch upload runner - works through master/scribe_B_queue.json (build_scribe_B_queue.py).

Per consult (validated end-to-end on authored/asthma_review, 2026-08-10):
  1. bh-chrome (port 9335) via browser-harness subprocess: sidebar [aria-label="New session"]
     -> verify the URL flips to a NEW /sessions/<id> -> transcription-mode chevron
     [aria-label="the transcription-mode menu"] -> "the upload-audio item" -> attach the wav to
     the DIALOG-scoped input ("[role=dialog] input[type=file]" - the page also holds a schedule
     CSV uploader as its FIRST input[type=file]; a bare selector hits that one and nothing
     transcribes) -> upload auto-starts,
     dialog closes itself when done.
  2. Poll THAT session URL (never the sidebar's idea of "current") until the rich-text
     note editor holds a stable clinical note (scribe_B auto-transcribes and auto-generates on its
     default note mode; feature-tour dialogs are dismissed as they appear).
  3. Content-verify before saving (attribute by CONTENT, never by upload order):
     scribe_B_match.Matcher.verify_containment - IDF-weighted containment of the note's terms in
     the intended transcript >= 0.28 AND intended is argmax over all 145 consults (calibrated
     on the 116 known-good pairs from the earlier capture round: correct min 0.289, wrong max
     0.257, rank-1 116/116).
     Pass -> scribe_B_notes/<source>__<id>.txt; fail -> scribe_B_notes/_unmatched/ + logged.
  4. JSONL log master/scribe_B_capture_log.jsonl (scribe_C log shape + scribe_B extras).

The product's own UI labels are redacted throughout this record; restore them from
the product interface alongside the addresses.
Cap rule: any upload refusal / paywall / limit dialog -> screenshot, master/scribe_B_cap_report.json,
clean STOP (exit 4). Never work around a cap - whether to continue is a human decision.
Politeness: STOPS cleanly between consults if the Scribe C real-time runner is detected
(pgrep -f scribe_C_overnight; that runner is not part of this release - the two captures
shared one browser).
Resume-safe: entries whose note file already exists (>150 bytes) are skipped, so re-running
continues the queue. 3 consecutive failures abort the run.

Usage:
  python3 scribe_B_overnight.py --preflight
  python3 scribe_B_overnight.py [--limit N] [--max-minutes M]   # run (serial, one upload at a time)
  python3 scribe_B_overnight.py --sweep       # safety net: scrape today's unclaimed sessions,
                                           # content-match them to unclaimed queue ids
Exit codes: 0 ok/nothing-pending, 2 aborted (3 consecutive failures), 3 scribe_C runner active,
            4 cap detected, 5 logged out, 6 preflight fail.
"""
# This module sits one directory below the repository root. It imports modules from the
# root and from the other topic directories by bare name, so the root goes on the import
# path first and `_modulepath` adds the rest. Every path it builds from `HERE` is relative
# to the root rather than to this directory. The optimiser modules in `gepa/` have done
# the same since before the release.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _modulepath  # noqa: E402,F401 - puts the topic directories on sys.path
import argparse, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repository root
QUEUE = os.path.join(HERE, "master", "scribe_B_queue.json")
LOG = os.path.join(HERE, "master", "scribe_B_capture_log.jsonl")
CAP_REPORT = os.path.join(HERE, "master", "scribe_B_cap_report.json")
NOTES_DIR = os.path.join(HERE, "scribe_B_notes")
UNMATCHED_DIR = os.path.join(NOTES_DIR, "_unmatched")
STEP_JSON = "/tmp/scribe_B_step.json"
STEP_NOTE = "/tmp/scribe_B_note_step.txt"
scribe_B_URL = "https://scribe.scribe_B.com/"

sys.path.insert(0, HERE)

# ---------------------------------------------------------------- browser phases
# Shared prologue: attach the scribe_B tab, dismiss overlay dialogs, tiny utils.
PROLOGUE = r'''
import json, os, re, time
OUT = "/tmp/scribe_B_step.json"
def emit(**kw):
    json.dump(kw, open(OUT, "w")); print("RESULT", kw.get("status"))
def attach_scribe_B():
    try:
        tabs = list_tabs(include_chrome=False)
    except Exception:
        tabs = []
    tgt = None
    for t in tabs:
        if "scribe.scribe_B.com" in (t.get("url") or ""):
            tgt = t; break
    if tgt:
        switch_tab(tgt)
    else:
        new_tab("https://scribe.scribe_B.com/")
    wait_for_load(); time.sleep(2)
    return page_info()["url"]
def logged_out(url=None):
    u = url or page_info()["url"]
    if re.search(r"accounts\.google|/login|/signin|auth", u, re.I):
        return True
    return bool(js("""(() => /Sign in|Log in to scribe_B|Continue with Google/i.test(document.body.innerText)
        && !/My sessions/i.test(document.body.innerText))()"""))
def dismiss(buttons_rx, rounds=5):
    for _ in range(rounds):
        hit = js("""(() => {
          for (const d of document.querySelectorAll('[role=dialog]')) {
            const dt=(d.innerText||'');
            if (/the upload dialog/i.test(dt)) continue;   // never dismiss the upload dialog
            for (const b of d.querySelectorAll('button')) {
              const t=(b.innerText||'').trim();
              if (new RegExp(%s,'i').test(t)) { b.click(); return t; }
            }
          }
          return null;})()""" % json.dumps(buttons_rx))
        if not hit:
            break
        time.sleep(1.0)
LIMIT_RX = r"(session limit|limit reached|out of sessions|no sessions left|ran out|quota|upgrade to (continue|keep)|trial (has )?(ended|expired)|maximum number|exceeded)"
def limit_text():
    return js("""(() => {
      const rx=new RegExp(%s,'i');
      for (const d of document.querySelectorAll('[role=dialog],[role=alertdialog],[class*=toast],[class*=Toast],[class*=banner]')) {
        const t=(d.innerText||'').trim();
        if (t && rx.test(t)) return t.slice(0,600);
      }
      return null;})()""" % json.dumps(LIMIT_RX))
def session_id(url):
    m = re.search(r"/sessions/(\d+)", url or "")
    return m.group(1) if m else None
'''

PHASE_A = PROLOGUE + r'''
WAV = os.environ["scribe_B_WAV"]; BASE = os.path.basename(WAV)
t0 = time.time()
url = attach_scribe_B()
if logged_out(url):
    emit(status="logged_out", url=url); raise SystemExit
dismiss(r"^(Next|Got it|Done|Not now|Skip|Finish|Dismiss|Close)$")
lt = limit_text()
if lt:
    capture_screenshot("/tmp/scribe_B_cap_dialog.png")
    emit(status="limit", limit_text=lt, screenshot="/tmp/scribe_B_cap_dialog.png"); raise SystemExit
prev_id = session_id(page_info()["url"])
# --- new session ---
new_url = None
for attempt in range(2):
    js("""(() => {const b=document.querySelector('[aria-label="New session"]'); if(b) b.click(); return !!b;})()""")
    for _ in range(15):
        time.sleep(2)
        u = page_info()["url"]; sid = session_id(u)
        if sid and sid != prev_id:
            new_url = u; break
    if new_url:
        break
if not new_url:
    capture_screenshot("/tmp/scribe_B_fail_newsession.png")
    emit(status="no_new_session", screenshot="/tmp/scribe_B_fail_newsession.png"); raise SystemExit
sid = session_id(new_url)
# --- open transcription-mode menu -> the upload-audio item ---
ok = False
for attempt in range(2):
    ch = None
    for _ in range(20):                       # the chevron can skeleton-load
        ch = js("""(() => {const el=document.querySelector('[aria-label="the transcription-mode menu"]');
                 if(!el)return null; const r=el.getBoundingClientRect();
                 return r.width>0?{x:r.x+r.width/2,y:r.y+r.height/2}:null;})()""")
        if ch: break
        time.sleep(2)
    if not ch: continue
    click_at_xy(int(ch["x"]), int(ch["y"])); time.sleep(1.5)
    hit = js("""(() => {
      for (const el of document.querySelectorAll('[role=menu] *')) {
        if (/^the upload-audio item$/.test((el.innerText||'').trim())) {
          const q=el.getBoundingClientRect(); if(q.width>0){ return {x:q.x+q.width/2,y:q.y+q.height/2}; } }
      } return null;})()""")
    if not hit:
        dismiss(r"^(Next|Got it|Done|Not now|Skip|Finish|Dismiss|Close)$"); time.sleep(1); continue
    click_at_xy(int(hit["x"]), int(hit["y"])); time.sleep(1.5)
    for _ in range(8):
        if js("""(() => {const d=[...document.querySelectorAll('[role=dialog]')].find(x=>/the upload dialog/i.test(x.innerText||'')); return !!d;})()"""):
            ok = True; break
        time.sleep(1)
    if ok: break
if not ok:
    capture_screenshot("/tmp/scribe_B_fail_menu.png")
    emit(status="no_upload_dialog", session_url=new_url, screenshot="/tmp/scribe_B_fail_menu.png"); raise SystemExit
# --- attach to the DIALOG-scoped file input (never the page's first input[type=file]) ---
try:
    upload_file("[role=dialog] input[type=file]", WAV)
except Exception as e:
    emit(status="attach_fail", session_url=new_url, err=str(e)[:200]); raise SystemExit
listed = False
for _ in range(10):
    time.sleep(1)
    if js("""(() => {const d=[...document.querySelectorAll('[role=dialog]')].find(x=>/the upload dialog/i.test(x.innerText||'')); return d?d.innerText.includes(%s):false;})()""" % json.dumps(BASE)):
        listed = True; break
if not listed:
    capture_screenshot("/tmp/scribe_B_fail_attach.png")
    emit(status="attach_not_listed", session_url=new_url, screenshot="/tmp/scribe_B_fail_attach.png"); raise SystemExit
# --- upload runs by itself; dialog closes when done ---
t_up = time.time(); closed = False
while time.time() - t_up < 300:
    time.sleep(3)
    d = js("""(() => {const d=[...document.querySelectorAll('[role=dialog]')].find(x=>/the upload dialog/i.test(x.innerText||'')); return d?(d.innerText||'').slice(0,400):null;})()""")
    if d is None:
        closed = True; break
    lt = limit_text()
    if lt:
        capture_screenshot("/tmp/scribe_B_cap_dialog.png")
        emit(status="limit", session_url=new_url, limit_text=lt, screenshot="/tmp/scribe_B_cap_dialog.png"); raise SystemExit
    if re.search(r"(failed|error|too large|unsupported)", d, re.I):
        capture_screenshot("/tmp/scribe_B_fail_upload.png")
        emit(status="upload_error", session_url=new_url, dialog=d[:300], screenshot="/tmp/scribe_B_fail_upload.png"); raise SystemExit
if not closed:
    capture_screenshot("/tmp/scribe_B_fail_upload.png")
    emit(status="upload_stuck", session_url=new_url, screenshot="/tmp/scribe_B_fail_upload.png"); raise SystemExit
emit(status="ok", session_url=new_url, session_id=sid, secs_upload=round(time.time()-t0, 1))
'''

PHASE_B = PROLOGUE + r'''
SESSION_URL = os.environ["scribe_B_SESSION_URL"]
MAX_WAIT = float(os.environ.get("scribe_B_MAX_WAIT", "480"))
want = session_id(SESSION_URL)
attach_scribe_B()
if session_id(page_info()["url"]) != want:
    goto_url(SESSION_URL); wait_for_load(); time.sleep(2)
if session_id(page_info()["url"]) != want:
    emit(status="wrong_session", url=page_info()["url"]); raise SystemExit
SCRAPE = """(() => {
  let best='';
  for (const ce of document.querySelectorAll('[contenteditable=true]')) {
    const t=(ce.innerText||'').trim();
    if (t.length>best.length) best=t;
  }
  return best;})()"""
t0 = time.time(); note = ""; prev_len = -1
while time.time() - t0 < MAX_WAIT:
    time.sleep(12)
    try:
        dismiss(r"^(Next|Got it|Done)$", rounds=2)
        lt = limit_text()
        if lt:
            capture_screenshot("/tmp/scribe_B_cap_dialog.png")
            emit(status="limit", limit_text=lt, screenshot="/tmp/scribe_B_cap_dialog.png"); raise SystemExit
        if session_id(page_info()["url"]) != want:  # view drifted - go back, never scrape elsewhere
            goto_url(SESSION_URL); wait_for_load(); time.sleep(2); continue
        cur = js(SCRAPE) or ""
    except SystemExit:
        raise
    except Exception:
        # transient CDP/evaluate hiccup (e.g. context destroyed mid-render) - retry next poll
        prev_len = -1
        continue
    ready = len(cur) > 150 and re.search(
        r"(Subjective|Objective|Assessment|Impression|Plan|Presenting Complaint|History of Presenting|Past Medical|Examination|Chief Complaint)", cur, re.I)
    if ready and len(cur) == prev_len:              # stable across two polls -> done rendering
        note = cur; break
    prev_len = len(cur) if ready else -1
if not note:
    capture_screenshot(os.environ.get("scribe_B_FAIL_PNG", "/tmp/scribe_B_fail_note.png"))
    emit(status="note_timeout", waited=round(time.time()-t0, 1)); raise SystemExit
# NB: no backslash escapes inside JS passed to js() - the transport decodes them
# (a /\n+/ regex literal arrives with a REAL newline inside and is a SyntaxError).
raw_title = js("""(() => {const h=document.querySelector('h1');
  if(!h) return null;
  const w=h.parentElement;
  return ((w?w.innerText:h.innerText)||'').slice(0,160);})()""")
title = " / ".join(s.strip() for s in (raw_title or "").splitlines() if s.strip())[:120] or None
open("/tmp/scribe_B_note_step.txt", "w").write(note)
emit(status="ok", note_len=len(note), title=title, secs_note=round(time.time()-t0, 1))
'''

SWEEP_LIST = PROLOGUE + r'''
TODAY_RX = os.environ["scribe_B_TODAY_RX"]        # e.g. "10 Aug"
url = attach_scribe_B()
if logged_out(url):
    emit(status="logged_out"); raise SystemExit
dismiss(r"^(Next|Got it|Done|Not now|Skip|Finish|Dismiss|Close)$")
# session rows are TRs in the sessions-panel table (one per session); the panel
# scrolls, so page through it collecting row text keys (clicks happen later by key,
# never by stale coordinates)
COLLECT = """(() => {
  const out=[];
  for (const el of document.querySelectorAll('tr')) {
    const t=(el.innerText||'').trim();
    if (!t || t.length>200) continue;
    if (!t.includes('the add-patient cell')) continue;
    if (!new RegExp(%s).test(t)) continue;           // today's date badge
    const q=el.getBoundingClientRect();
    if (q.width<100 || q.height<25 || q.height>150) continue;
    out.push(t.slice(0,150));
  }
  return out;})()""" % json.dumps(TODAY_RX)
SCROLLER = """(() => {
  let el=document.querySelector('tbody');
  while (el && !(el.scrollHeight > el.clientHeight + 20)) el = el.parentElement;
  if (!el) return null;
  el.__bh_scroller = true;
  return {h: el.clientHeight, sh: el.scrollHeight, top: el.scrollTop};
})()"""
seen = []
sc = js(SCROLLER)
for hop in range(12):
    for t in (js(COLLECT) or []):
        if t not in seen:
            seen.append(t)
    if not sc:
        break
    at_end = js("""(() => {
      let el=document.querySelector('tbody');
      while (el && !el.__bh_scroller) el = el.parentElement;
      if (!el) return true;
      const before = el.scrollTop;
      el.scrollTop = before + el.clientHeight;
      return el.scrollTop === before;   // no movement -> bottom
    })()""")
    time.sleep(0.8)
    if at_end:
        for t in (js(COLLECT) or []):
            if t not in seen:
                seen.append(t)
        break
# scroll back to top so later phases see a sane panel
js("""(() => {let el=document.querySelector('tbody');
  while (el && !el.__bh_scroller) el = el.parentElement;
  if (el) el.scrollTop = 0; return true;})()""")
emit(status="ok", rows=[{"t": t} for t in seen])
'''

SWEEP_OPEN = PROLOGUE + r'''
KEY = os.environ["scribe_B_ROW_KEY"]        # the row's innerText collected by SWEEP_LIST
r = js("""(() => {
  const key=%s;
  for (const el of document.querySelectorAll('tr')) {
    const t=(el.innerText||'').trim().slice(0,150);
    if (t === key) {
      el.scrollIntoView({block:'center'});
      const q=el.getBoundingClientRect();
      return {x:Math.round(q.x+q.width/2), y:Math.round(q.y+q.height/2)};
    }
  }
  return null;})()""" % json.dumps(KEY))
if not r:
    emit(status="row_not_found"); raise SystemExit
time.sleep(0.5)
click_at_xy(r["x"], r["y"]); time.sleep(3)
u = page_info()["url"]; sid = session_id(u)
note = js("""(() => {
  let best='';
  for (const ce of document.querySelectorAll('[contenteditable=true]')) {
    const t=(ce.innerText||'').trim(); if (t.length>best.length) best=t;
  } return best;})()""") or ""
open("/tmp/scribe_B_note_step.txt", "w").write(note)
emit(status="ok", session_url=u, session_id=sid, note_len=len(note))
'''


# ---------------------------------------------------------------- driver helpers
def harness(program, env_extra=None, timeout=420):
    """Run one bounded browser-harness phase; result comes back via STEP_JSON."""
    if os.path.exists(STEP_JSON):
        os.remove(STEP_JSON)
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    try:
        p = subprocess.run(["browser-harness"], input=program, text=True,
                           capture_output=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "harness_timeout"}
    if os.path.exists(STEP_JSON):
        try:
            return json.load(open(STEP_JSON))
        except Exception:
            pass
    return {"status": "harness_crash",
            "stderr": (p.stderr or "")[-400:], "stdout": (p.stdout or "")[-200:]}


def scribe_C_active():
    return subprocess.run(["pgrep", "-f", "scribe_C_overnight"],
                          capture_output=True).returncode == 0


def log_jsonl(**kw):
    kw["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(LOG, "a") as f:
        f.write(json.dumps(kw) + "\n")


def pending_entries():
    q = json.load(open(QUEUE))
    out = []
    for e in q["entries"]:
        p = os.path.join(HERE, e["note"])
        if not (os.path.exists(p) and os.path.getsize(p) > 150):
            out.append(e)
    return q, out


def write_cap_report(entry, res, done_this_run):
    n_saved = len([f for f in os.listdir(NOTES_DIR)
                   if f.endswith(".txt") and os.path.getsize(os.path.join(NOTES_DIR, f)) > 150])
    _, pend = pending_entries()
    rep = {"detected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "while_uploading": {"id": entry["id"], "source": entry["source"], "seq": entry["seq"]},
           "captured_this_run_before_cap": done_this_run,
           "total_scribe_B_notes_saved": n_saved,
           "queue_remaining": len(pend),
           "limit_text": res.get("limit_text"),
           "screenshot": res.get("screenshot"),
           "session_url": res.get("session_url")}
    json.dump(rep, open(CAP_REPORT, "w"), indent=1)
    return rep


def save_verified(entry, note, matcher):
    """Content gate + save. Returns (status, verdict)."""
    v = matcher.verify_containment(note, entry["id"])
    if v["ok"]:
        open(os.path.join(HERE, entry["note"]), "w").write(note)
        return "ok", v
    os.makedirs(UNMATCHED_DIR, exist_ok=True)
    fn = os.path.join(UNMATCHED_DIR,
                      time.strftime("%Y%m%dT%H%M%S") + f"__intended_{entry['source']}__{entry['id']}.txt")
    open(fn, "w").write(note)
    return "unmatched", v


# ---------------------------------------------------------------- modes
def preflight():
    fail = 0
    def chk(ok, msg):
        nonlocal fail
        print(("OK   " if ok else "FAIL ") + msg)
        if not ok:
            fail = 1
    import shutil
    chk(shutil.which("browser-harness") is not None, "tool: browser-harness")
    chk(shutil.which("pgrep") is not None, "tool: pgrep")
    try:
        import urllib.request
        urllib.request.urlopen("http://127.0.0.1:9335/json/version", timeout=5)
        chk(True, "bh-chrome CDP on 9335")
    except Exception:
        chk(False, "bh-chrome not reachable on 9335 (run bh-chrome)")
    chk(os.path.exists(QUEUE), f"queue exists: {QUEUE}")
    q, pend = pending_entries()
    missing = [e["id"] for e in pend if not os.path.exists(os.path.join(HERE, e["wav"]))]
    chk(not missing, f"audio present for all {len(pend)} pending" if not missing
        else f"audio missing for {len(missing)}: {missing[:5]}")
    from scribe_B_match import Matcher
    m = Matcher()
    chk(len(m.transcripts) >= 140, f"matcher transcripts loaded: {len(m.transcripts)}")
    unknown = [e["id"] for e in pend if e["id"] not in m.transcripts]
    chk(not unknown, "every queued id has a transcript" if not unknown else f"no transcript for {unknown[:5]}")
    chk(not scribe_C_active(), "scribe_C runner not active")
    free_gb = shutil.disk_usage("/").free // 2**30
    chk(free_gb >= 5, f"disk: {free_gb}G free")
    print(f"\npending: {len(pend)} of {len(q['entries'])} queue entries "
          f"({sum((e['dur_s'] or 0) for e in pend)/3600:.1f} h audio)")
    print("PREFLIGHT " + ("PASS" if fail == 0 else "FAIL"))
    return 6 if fail else 0


def run(limit=0, max_minutes=0.0):
    from scribe_B_match import Matcher
    matcher = Matcher()
    q, pend = pending_entries()
    if not pend:
        print("queue empty - nothing pending")
        return 0
    if limit:
        pend = pend[:limit]
    print(f"RUN START {time.strftime('%F %H:%M:%S')}: {len(pend)} consults this chunk "
          f"(of {len(pending_entries()[1])} pending)")
    t_run = time.time()
    consec_fail = 0
    done_n = 0
    for e in pend:
        tag = f"{e['source']}/{e['id']}"
        if max_minutes and (time.time() - t_run) > max_minutes * 60:
            print(f"CHUNK BUDGET reached ({max_minutes} min) - stopping cleanly")
            break
        if scribe_C_active():
            print("STOP: scribe_C_overnight runner detected - yielding the browser")
            log_jsonl(id=e["id"], source=e["source"], variant=None, status="skipped_scribe_C_active",
                      note_len=0, dur_s=e["dur_s"])
            return 3
        wav = os.path.join(HERE, e["wav"])
        if not os.path.exists(wav):
            print(f"--- SKIP {tag}: wav missing")
            log_jsonl(id=e["id"], source=e["source"], variant=None, status="wav_missing",
                      note_len=0, dur_s=e["dur_s"])
            continue
        notep = os.path.join(HERE, e["note"])
        if os.path.exists(notep) and os.path.getsize(notep) > 150:
            print(f"--- skip {tag} (note exists)")
            continue
        print(f"===== [{tag}] upload  {time.strftime('%H:%M:%S')} =====", flush=True)
        a = harness(PHASE_A, {"scribe_B_WAV": wav}, timeout=420)
        if a.get("status") == "limit":
            print(f"  CAP HIT: {a.get('limit_text', '')[:200]}")
            rep = write_cap_report(e, a, done_n)
            log_jsonl(id=e["id"], source=e["source"], variant=None, status="limit",
                      note_len=0, dur_s=e["dur_s"], limit_text=a.get("limit_text"))
            print(json.dumps(rep, indent=1))
            return 4
        if a.get("status") == "logged_out":
            print("  STOP: scribe_B is logged out")
            return 5
        if a.get("status") != "ok":
            print(f"  upload phase FAILED: {a.get('status')} {a.get('err','')}")
            log_jsonl(id=e["id"], source=e["source"], variant=None,
                      status="start_fail:" + str(a.get("status")), note_len=0, dur_s=e["dur_s"])
            consec_fail += 1
            if consec_fail >= 3:
                print("ABORT: 3 consecutive failures")
                return 2
            time.sleep(8)
            continue
        print(f"  session {a['session_id']}  (upload {a['secs_upload']}s)")
        fail_png = f"/tmp/scribe_B_fail_{e['id']}.png"
        b = harness(PHASE_B, {"scribe_B_SESSION_URL": a["session_url"],
                              "scribe_B_MAX_WAIT": "480", "scribe_B_FAIL_PNG": fail_png},
                    timeout=600)
        if b.get("status") == "limit":
            print(f"  CAP HIT during processing: {b.get('limit_text', '')[:200]}")
            rep = write_cap_report(e, {**b, "session_url": a["session_url"]}, done_n)
            log_jsonl(id=e["id"], source=e["source"], variant=None, status="limit",
                      note_len=0, dur_s=e["dur_s"], session_id=a["session_id"],
                      limit_text=b.get("limit_text"))
            print(json.dumps(rep, indent=1))
            return 4
        if b.get("status") != "ok":
            print(f"  note phase FAILED: {b.get('status')} (screenshot: {fail_png})")
            log_jsonl(id=e["id"], source=e["source"], variant=None,
                      status="note_fail:" + str(b.get("status")), note_len=0,
                      dur_s=e["dur_s"], session_id=a.get("session_id"),
                      stderr=b.get("stderr"))
            consec_fail += 1
            if consec_fail >= 3:
                print("ABORT: 3 consecutive failures")
                return 2
            time.sleep(8)
            continue
        note = open(STEP_NOTE).read()
        status, v = save_verified(e, note, matcher)
        log_jsonl(id=e["id"], source=e["source"], variant=None, status=status,
                  note_len=len(note), dur_s=e["dur_s"], session_id=a["session_id"],
                  title=b.get("title"), cont=v["cont"], best_id=v["best_id"],
                  best_cont=v["best_cont"], cosine_sim=v["cosine_sim"],
                  secs_upload=a["secs_upload"], secs_note=b["secs_note"])
        if status == "ok":
            done_n += 1
            consec_fail = 0
            print(f"  saved {e['note']} ({len(note)} chars, cont={v['cont']}, "
                  f"note wait {b['secs_note']}s)")
        else:
            consec_fail += 1
            print(f"  UNMATCHED: intended {e['id']} but best={v['best_id']} "
                  f"(cont {v['cont']} vs {v['best_cont']}) -> _unmatched/")
            if consec_fail >= 3:
                print("ABORT: 3 consecutive failures (unmatched counts - "
                      "session targeting may be broken)")
                return 2
        time.sleep(6)
    _, pend_after = pending_entries()
    print(f"CHUNK DONE {time.strftime('%F %H:%M:%S')}: {done_n} captured this run; "
          f"{len(pend_after)} still pending; log: {LOG}")
    return 0


def sweep():
    """List today's sessions, scrape any whose note isn't claimed, content-match to
    unclaimed queue ids. Safety net for interleaved/mixed-up sessions."""
    from scribe_B_match import Matcher
    matcher = Matcher()
    q, pend = pending_entries()
    claimed_sessions = set()
    if os.path.exists(LOG):
        for line in open(LOG):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "ok" and r.get("session_id"):
                claimed_sessions.add(str(r["session_id"]))
    today_rx = time.strftime("%-d %b")     # e.g. "10 Aug"
    lst = harness(SWEEP_LIST, {"scribe_B_TODAY_RX": today_rx}, timeout=180)
    if lst.get("status") != "ok":
        print(f"sweep: could not list sessions ({lst.get('status')})")
        return 2
    rows = lst.get("rows", [])
    print(f"sweep: {len(rows)} sessions dated '{today_rx}' visible; "
          f"{len(claimed_sessions)} already claimed; {len(pend)} queue ids unclaimed")
    rescued = 0
    for row in rows:
        o = harness(SWEEP_OPEN, {"scribe_B_ROW_KEY": row["t"]}, timeout=120)
        if o.get("status") != "ok" or not o.get("session_id"):
            continue
        if str(o["session_id"]) in claimed_sessions:
            continue
        if o.get("note_len", 0) < 150:
            print(f"  {o['session_id']} ({row['t'][:50]}): no note ({o.get('note_len')} chars)")
            continue
        note = open(STEP_NOTE).read()
        scores = {c: matcher.containment(note, c) for c in matcher.tr_tokens}
        best_id, best_cont = max(scores.items(), key=lambda kv: kv[1])
        target = next((e for e in pend if e["id"] == best_id), None)
        if target and best_cont >= matcher.CONT_MIN:
            notep = os.path.join(HERE, target["note"])
            if not (os.path.exists(notep) and os.path.getsize(notep) > 150):
                open(notep, "w").write(note)
                rescued += 1
                claimed_sessions.add(str(o["session_id"]))
                log_jsonl(id=target["id"], source=target["source"], variant=None,
                          status="ok", note_len=len(note), dur_s=target["dur_s"],
                          session_id=o["session_id"], cont=round(best_cont, 4),
                          best_id=best_id, best_cont=round(best_cont, 4),
                          cosine_sim=None, via="sweep")
                print(f"  RESCUED {target['source']}/{target['id']} from session "
                      f"{o['session_id']} (cont={best_cont:.3f})")
                continue
        if best_cont >= matcher.CONT_MIN and target is None:
            # best match is a consult whose note is already captured - a known/duplicate
            # session, nothing to rescue
            print(f"  session {o['session_id']}: matches already-captured {best_id} "
                  f"(cont={best_cont:.3f}) - skipping")
            continue
        os.makedirs(UNMATCHED_DIR, exist_ok=True)
        fn = os.path.join(UNMATCHED_DIR, time.strftime("%Y%m%dT%H%M%S")
                          + f"__sweep_session_{o['session_id']}.txt")
        open(fn, "w").write(note)
        log_jsonl(id=None, source=None, variant=None, status="sweep_unmatched",
                  note_len=len(note), session_id=o["session_id"],
                  best_id=best_id, best_cont=round(best_cont, 4))
        print(f"  session {o['session_id']}: best={best_id} cont={best_cont:.3f} "
              f"- not claimable -> _unmatched/")
    print(f"sweep done: {rescued} rescued")
    return 0


def rescue():
    """Re-scrape sessions the LOG knows about (note_fail with a recorded session_id)
    whose queue id is still pending - cheaper and more precise than a sweep."""
    from scribe_B_match import Matcher
    matcher = Matcher()
    q, pend = pending_entries()
    pend_by_id = {e["id"]: e for e in pend}
    cand = {}
    if os.path.exists(LOG):
        for line in open(LOG):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if (str(r.get("status", "")).startswith("note_fail") and r.get("session_id")
                    and r.get("id") in pend_by_id):
                cand[r["id"]] = str(r["session_id"])
    if not cand:
        print("rescue: nothing to rescue (no failed-with-session-id pending entries)")
        return 0
    print(f"rescue: {len(cand)} candidates: {sorted(cand)}")
    n = 0
    for cid, sid in cand.items():
        e = pend_by_id[cid]
        url = f"https://scribe.scribe_B.com/sessions/{sid}"
        b = harness(PHASE_B, {"scribe_B_SESSION_URL": url, "scribe_B_MAX_WAIT": "90",
                              "scribe_B_FAIL_PNG": f"/tmp/scribe_B_fail_rescue_{cid}.png"},
                    timeout=240)
        if b.get("status") != "ok":
            print(f"  {cid}: rescue scrape failed ({b.get('status')})")
            continue
        note = open(STEP_NOTE).read()
        status, v = save_verified(e, note, matcher)
        log_jsonl(id=cid, source=e["source"], variant=None, status=status,
                  note_len=len(note), dur_s=e["dur_s"], session_id=sid,
                  title=b.get("title"), cont=v["cont"], best_id=v["best_id"],
                  best_cont=v["best_cont"], cosine_sim=v["cosine_sim"], via="rescue")
        if status == "ok":
            n += 1
            print(f"  RESCUED {e['source']}/{cid} (cont={v['cont']})")
        else:
            print(f"  {cid}: scraped but UNMATCHED (best={v['best_id']} {v['best_cont']})")
    print(f"rescue done: {n} saved")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--rescue", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="max consults this invocation")
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="stop starting new consults after this many minutes")
    a = ap.parse_args()
    os.makedirs(NOTES_DIR, exist_ok=True)
    if a.preflight:
        sys.exit(preflight())
    if a.sweep:
        sys.exit(sweep())
    if a.rescue:
        sys.exit(rescue())
    if scribe_C_active():
        print("STOP: scribe_C_overnight runner active - not touching the browser")
        sys.exit(3)
    pidfile = "/tmp/scribe_B_overnight.pid"
    if os.path.exists(pidfile):
        old = open(pidfile).read().strip()
        if old and subprocess.run(["kill", "-0", old], capture_output=True).returncode == 0:
            print(f"STOP: another scribe_B_overnight.py is running (pid {old})")
            sys.exit(2)
    open(pidfile, "w").write(str(os.getpid()))
    try:
        sys.exit(run(limit=a.limit, max_minutes=a.max_minutes))
    finally:
        os.remove(pidfile)


if __name__ == "__main__":
    main()
