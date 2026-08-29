"""Shared primitives for the scribe-study experiments (E1-E5).

Import from here so experiment scripts stay thin and consistent:
    from common import claude, scribe_AClient, make_composo, load_primock, DANGEROUS_MODES, HERE

Two model transports live here and they are not interchangeable. `llm()` is the
paper-bound one: every benchmark and judge call goes through it, on OpenRouter, against a
role pinned in `models.lock.json`, and it asserts that provider rather than accepting
another. `claude()` and `claude_json()` are the construction-and-audit transport: they
shell out to the local `claude` command-line binary, which runs against a Claude
subscription rather than an API key, and they carry fact-sheet extraction, scenario
authoring, the critique panels and the census instruments. `claude()` has no OpenRouter
route of its own - a stage that needed one calls `llm()` instead, which is what
`taxonomy_common.route_call` does by default - but setting BATCH_LLM=openai sends every
`claude()`/`claude_json()` call to an OpenAI model instead, and the study used that
whenever the subscription's session limit was reached.

Composo via the cp- key in secrets.env. scribe_A via OAuth client-creds in secrets.env.
"""
import json, os, re, subprocess, threading, time
import httpx
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, "secrets.env"))
PRIMOCK = os.path.join(HERE, "../../experiments/ai_scribe/dataset/primock57_parsed.json")


# ---------------------------------------------------------------- LLM
# Provider switch: set BATCH_LLM=openai to route ALL claude()/claude_json() batch calls to a strong
# OpenAI model (default gpt-5.5, reasoning=none) - OFF the lead author's Claude plan (used when the plan limit is
# hit). Default (unset) = the lead author's Claude plan via `claude -p`. Embeddings/scribe_A/Composo are unaffected.
_OAI = None
_OAI_USAGE = {"in": 0, "out": 0, "calls": 0}


def _openai_client():
    global _OAI
    if _OAI is None:
        from openai import OpenAI
        _OAI = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=600, max_retries=4)
    return _OAI


def _openai_call(prompt, timeout):
    m = os.environ.get("BATCH_OPENAI_MODEL", "gpt-5.5")
    reff = os.environ.get("BATCH_OPENAI_REASONING", "none")   # gpt-5.5 supports 'none'
    kw = {"model": m, "messages": [{"role": "user", "content": prompt}]}
    if reff:
        kw["reasoning_effort"] = reff
    try:
        r = _openai_client().chat.completions.create(timeout=timeout, **kw)
        u = getattr(r, "usage", None)
        if u:
            _OAI_USAGE["in"] += getattr(u, "prompt_tokens", 0) or 0
            _OAI_USAGE["out"] += getattr(u, "completion_tokens", 0) or 0
            _OAI_USAGE["calls"] += 1
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return ""


def claude(prompt, timeout=240, model="claude-sonnet-4-6", retries=1, effort="medium"):
    """Run a one-shot prompt. Default = the lead author's Claude plan via `claude -p` (cwd=/tmp + strict-mcp
    avoids the MCP-load timeout). If BATCH_LLM=openai, routes off-plan to a strong OpenAI model.
    Resilient: degrades to "" rather than raising (so one slow call can't crash a ThreadPool batch).

    effort: overrides the session's effortLevel for the claude-plan path. None = inherit session."""
    if os.environ.get("BATCH_LLM") == "openai":
        return _openai_call(prompt, timeout=max(timeout, 600))
    cmd = ["claude", "-p", prompt, "--output-format", "text", "--strict-mcp-config", "--model", model]
    if effort:
        cmd += ["--effort", effort]
    for attempt in range(retries + 1):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd="/tmp")
            return r.stdout.strip()
        except subprocess.TimeoutExpired:
            if attempt >= retries:
                return ""
        except Exception:
            if attempt >= retries:
                return ""
    return ""


def claude_json(prompt, timeout=240, model="claude-sonnet-4-6", effort="medium", retries=1):
    """claude() but parse the first {...} or [...] JSON blob out of the reply."""
    out = claude(prompt, timeout=timeout, model=model, effort=effort, retries=retries)
    m = re.search(r"(\{.*\}|\[.*\])", out, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


# ---------------------------------------------------------------- scribe_A (real scribe, API)
scribe_A_TEMPLATES = {
    "short": os.environ.get("SCRIBE_A_TEMPLATE_SHORT", "TEMPLATE_UUID_REDACTED_1"),      # Short report (terse 3-section SOAP)
    "detailed": os.environ.get("SCRIBE_A_TEMPLATE_DETAILED", "TEMPLATE_UUID_REDACTED_2"),   # Detailed patient consultation
}

class scribe_AClient:
    """Token auto-refresh (300s) + text-transcript -> note. Thread-safe."""
    def __init__(self):
        self._tok, self._exp, self._lock = None, 0.0, threading.Lock()
        self.env = os.environ["scribe_A_ENV"]
        self.tenant = os.environ["scribe_A_TENANT"]
        self.base = f"https://api.{self.env}.scribe_A.app"
        self.cx = httpx.Client(follow_redirects=True, timeout=120)

    def token(self):
        with self._lock:
            if time.time() > self._exp - 30:
                r = self.cx.post(
                    f"https://auth.{self.env}.scribe_A.app/realms/{self.tenant}/protocol/openid-connect/token",
                    data={"client_id": os.environ["scribe_A_CLIENT_ID"],
                          "client_secret": os.environ["scribe_A_CLIENT_SECRET"],
                          "grant_type": "client_credentials", "scope": "openid"})
                r.raise_for_status()
                j = r.json()
                self._tok, self._exp = j["access_token"], time.time() + j.get("expires_in", 300)
            return self._tok

    def generate(self, transcript, template="short"):
        """Return (note_text, stringDocument). `template` = key in scribe_A_TEMPLATES or a raw UUID."""
        tid = scribe_A_TEMPLATES.get(template, template)
        H = {"Authorization": f"Bearer {self.token()}", "Tenant-Name": self.tenant,
             "Content-Type": "application/json", "X-scribe_A-Retention-Policy": "none"}
        body = {"outputLanguage": "en", "templateRef": {"templateId": tid},
                "context": [{"type": "text", "text": transcript[:40000]}]}
        r = self.cx.post(f"{self.base}/v2/documents/", headers=H, json=body)
        r.raise_for_status()
        doc = r.json()["document"]
        sd = doc.get("stringDocument", {}) or {}
        note = "\n\n".join(v.strip() for v in sd.values() if isinstance(v, str) and v.strip())
        return note, sd


# ---------------------------------------------------------------- Composo (eval, cp- key)
def make_composo(model_core="align-20260109"):
    """Composo client on the retrieval core (default align-20251111 is legacy/no-retrieval)."""
    from composo import Composo
    return Composo(api_key=os.environ["COMPOSO_API_KEY"],
                   base_url="https://platform.composo.ai", model_core=model_core)


# ---------------------------------------------------------------- data + taxonomy
def load_primock(n=None):
    data = json.load(open(PRIMOCK))
    return data[:n] if n else data


# Failure taxonomy: (key, injection-instruction, default criticality). Drives E2 graded injection,
# the E1 differ vocabulary, and the E3 scaffold sanity-check.
DANGEROUS_MODES = [
    ("negation",     "flip a negation/polarity (e.g. 'no chest pain' -> 'chest pain', 'denies' -> 'reports')", "critical"),
    ("dose_value",   "change a medication dose, frequency or a numeric value", "critical"),
    ("omission",     "drop a stated red-flag finding or a stated drug allergy", "critical"),
    ("laterality",   "swap left<->right", "critical"),
    ("attribution",  "swap who a finding belongs to (patient symptom <-> family history)", "moderate"),
    ("modality",     "harden assertion ('suggested'/'considered' -> 'diagnosed'/'started')", "moderate"),
    ("temporal",     "change onset/duration/tense ('for a couple of days' -> 'a couple of days ago')", "moderate"),
    ("fabrication",  "add a fabricated finding or diagnosis not supported by the transcript", "moderate"),
]


# ---------------------------------------------------------------- W-D fact-sheet schema
# The extracted-fact-sheet schema (W-D spec 3.1 + Amendment 2026-07-29b part H). Lives here so the
# extraction path and the critic/revise path validate against ONE definition and cannot drift:
# extraction rejects an out-of-schema sheet, so a revision must be held to the same contract.
FACT_SHEET_MODES = {"omission", "negation", "dose_value", "laterality", "attribution",
                    "modality_hardening", "temporal", "fabrication", "decision_status", "anchoring"}
FACT_SHEET_IMPORTANCE = {"critical", "supporting", "peripheral"}
FACT_SHEET_LOAD_BEARING = {"high", "medium"}


def validate_fact_sheet(fs):
    """Shape-check a fact_sheet; return a list of problems (empty list = valid)."""
    probs = []
    if not isinstance(fs, dict):
        return ["fact_sheet is not an object"]
    for key in ("must_contain", "must_not_contain", "salience_traps"):
        if not isinstance(fs.get(key), list):
            probs.append(f"{key} missing or not a list")
    if probs:
        return probs
    if not fs["must_contain"]:
        probs.append("must_contain is empty")
    for i, it in enumerate(fs["must_contain"]):
        if not (isinstance(it, dict) and it.get("fact") and it.get("evidence")):
            probs.append(f"must_contain[{i}] lacks fact/evidence")
        elif it.get("load_bearing") not in FACT_SHEET_LOAD_BEARING:
            probs.append(f"must_contain[{i}] load_bearing={it.get('load_bearing')!r}")
    for i, it in enumerate(fs["must_not_contain"]):
        if not (isinstance(it, dict) and it.get("assertion") and it.get("why_wrong")):
            probs.append(f"must_not_contain[{i}] lacks assertion/why_wrong")
    for i, it in enumerate(fs["salience_traps"]):
        if not (isinstance(it, dict) and it.get("trap") and it.get("correct_handling")):
            probs.append(f"salience_traps[{i}] lacks trap/correct_handling")
            continue
        if it.get("mode") not in FACT_SHEET_MODES:
            probs.append(f"salience_traps[{i}] mode={it.get('mode')!r}")
        if it.get("importance") not in FACT_SHEET_IMPORTANCE:
            probs.append(f"salience_traps[{i}] importance={it.get('importance')!r}")
    return probs


# ---------------------------------------------------------------- W1: pinning + provenance
# specs/w1-reproducibility.md 3.4, as amended 2026-07-29 (A3: OpenRouter routing, k-sampling
# fallback, manifest_version 2, spend-typed cost ledger). Paper-bound calls use llm() + Run()
# ONLY - the exploratory paths above (claude/_openai_call) are unchanged and stay off-paper.
import gzip, hashlib, platform, sys
from datetime import datetime, timezone

MODELS_LOCK = os.path.join(HERE, "models.lock.json")
RESULTS = os.path.join(HERE, "results")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
CURRENT_RUN = None  # set by Run.__enter__; llm() logs into it
_RETRY_LOCK = threading.Lock()  # guards llm()'s retry counter across parallel k-samples


def resolve_model(role):
    """Look a model ROLE (e.g. 'judge-primary') up in models.lock.json. Raises if unpinned."""
    lock = json.load(open(MODELS_LOCK))
    e = lock["models"].get(role)
    if e is None or not e.get("resolved"):
        raise KeyError(f"model role {role!r} not pinned in models.lock.json - pin before a paper-bound run")
    return e


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_state():
    """(commit, dirty_paths) for the harness subtree. Run outputs (results/) and bytecode
    caches do not count as dirty - the manifest's git_commit identifies the CODE + DATA
    state, and a multi-run session necessarily accumulates results between commits."""
    def _run(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE).stdout.strip()
    def _counts(path):  # porcelain paths are repo-root-relative
        return not (path.startswith("results/") or "/results/" in path or "__pycache__" in path)
    commit = _run("rev-parse", "HEAD")
    dirty = [l.strip() for l in _run("status", "--porcelain", ".").splitlines()
             if l[3:] and _counts(l[3:])]
    return commit, dirty


def _package_versions():
    import importlib.metadata as md
    out = {"python": platform.python_version()}
    for p in ("httpx", "openai", "requests"):
        try:
            out[p] = md.version(p)
        except Exception:
            out[p] = None
    return out


def _openrouter_call(body, timeout):
    """One OpenRouter chat request. Returns the parsed response; raises on any HTTP/API error
    or an all-empty completion (R1: an empty string must never look like a verdict).

    Timeouts are explicit per phase rather than one scalar. A scalar `timeout=` does bound the
    read, but it bounds the gap BETWEEN bytes, so a socket the far end keeps warm without
    finishing the body never trips it - which is how a worker pool ends up wedged at 0% CPU
    with no error to catch (seen on the 2026-08-12 site-map run and, separately, by W-F).
    `read` is the caller's budget; connect/write/pool are short, because a request that cannot
    even establish a connection in 20s is not going to be rescued by waiting seven minutes.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY missing from secrets.env")
    tmo = httpx.Timeout(connect=20.0, read=float(timeout), write=60.0, pool=20.0)
    r = httpx.post(OPENROUTER_URL, headers={"Authorization": f"Bearer {key}"},
                   json=body, timeout=tmo)
    r.raise_for_status()
    j = r.json()
    if "error" in j:
        raise RuntimeError(f"OpenRouter error: {json.dumps(j['error'])[:500]}")
    if not j.get("choices"):
        raise RuntimeError(f"OpenRouter returned no choices: {json.dumps(j)[:500]}")
    if all(not (c.get("message", {}).get("content") or "").strip() for c in j["choices"]):
        raise RuntimeError("OpenRouter returned only empty completions (R1) - "
                           f"finish_reason={j['choices'][0].get('finish_reason')!r}; raise max_tokens?")
    return j


def llm(prompt, role, temperature=1.0, k=1, seed=None, reasoning_effort="none",
        max_tokens=1024, timeout=600, max_retries=4, cache_prefix=None):
    """Paper-bound LLM call via OpenRouter (W1 3.4 as amended 2026-07-29 A3). Pinned model
    only, explicit params, NEVER returns '' silently - retries then RAISES (R1).

    cache_prefix (optional, added 2026-08-12): the leading substring of `prompt` that is
    stable across a batch - typically the transcript + reference note, which every arm and
    every replicate re-sends unchanged. When given AND the pinned model is an Anthropic one,
    the message is split into two content parts and the prefix carries Anthropic's explicit
    `cache_control: ephemeral` marker, cutting its input price by 90% on the second and later
    calls that share it. OpenAI routes cache automatically and are left untouched, so passing
    this on a gpt-* role is a silent no-op rather than an error. Default None reproduces the
    previous single-string body byte for byte, which is why in-flight jobs are unaffected.
    Anthropic will not cache a prefix below its ~1,024-token floor - short prefixes are
    accepted and simply do not cache, so callers should only mark genuinely large prefixes.
    meta["cache_prefix_chars"] records what was marked, for the manifest.

    k>1 ensemble contract (A3 + Amendment 2026-07-29b): one n=k request is tried first; if
    the route returns fewer than k choices (OpenRouter drops n on many routes - verified for
    openai/gpt-5.4, 2026-07-29), tops up transparently with independent calls at the same
    params, seed offset by sample index so honored seeds cannot collapse the ensemble.
    Statistically equivalent either way; meta["k_impl"] records which implementation ran.

    Returns (texts, meta): texts = list of k completion strings; meta = model slug, served
    provider(s), generation id(s), returned model string(s), summed usage (incl. reasoning +
    cached tokens), OpenRouter-reported credit cost, k_impl, per-request detail. Logs into
    CURRENT_RUN when inside a Run context; on final failure the run's error count is bumped
    before the raise, so a failed call can never vanish from the manifest.
    """
    e = resolve_model(role)
    assert e["provider"] == "openrouter", f"llm() routes via OpenRouter only (A3), got {e['provider']!r}"
    route = e.get("route") or {}

    _anthropic = e["resolved"].startswith("anthropic/")
    _cache_on = bool(cache_prefix) and _anthropic and prompt.startswith(cache_prefix)
    if cache_prefix and _anthropic and not prompt.startswith(cache_prefix):
        raise ValueError("cache_prefix must be a leading substring of prompt "
                         f"(prefix {len(cache_prefix)} chars did not match)")

    def _content():
        if not _cache_on:
            return prompt
        return [{"type": "text", "text": cache_prefix,
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": prompt[len(cache_prefix):]}]

    def body(n, s):
        b = {"model": e["resolved"], "messages": [{"role": "user", "content": _content()}],
             "usage": {"include": True}}
        if route.get("order"):
            b["provider"] = {"order": route["order"],
                             "allow_fallbacks": bool(route.get("allow_fallbacks", False))}
        if temperature is not None:
            b["temperature"] = temperature
        if n > 1:
            b["n"] = n
        if s is not None:
            b["seed"] = s
        if reasoning_effort:
            b["reasoning_effort"] = reasoning_effort
        if max_tokens:
            b["max_tokens"] = max_tokens
        return b

    texts, requests, retried = [], [], 0

    def record(j, s, n):
        u = j.get("usage") or {}
        requests.append({
            "generation_id": j.get("id"), "provider": j.get("provider"),
            "returned_model": j.get("model"), "system_fingerprint": j.get("system_fingerprint"),
            "n_requested": n, "n_returned": len(j["choices"]), "seed": s,
            "usage": {"prompt_tokens": u.get("prompt_tokens", 0),
                      "completion_tokens": u.get("completion_tokens", 0),
                      "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
                      "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)},
            "cost": u.get("cost")})
        texts.extend((c["message"]["content"] or "").strip() for c in j["choices"][: k - len(texts)])

    if k > 1:  # try provider-side n-sampling once; any failure/short return -> independent calls
        try:
            record(_openrouter_call(body(k, seed), timeout), seed, k)
        except Exception:
            retried += 1
        n_sampled = len(texts) >= k
    else:
        n_sampled = True

    def _one(i):
        """One independent completion (sample index i), with its own retry loop. Seed is
        offset by the sample index so an honored seed cannot collapse the ensemble."""
        nonlocal retried
        s = None if seed is None else seed + i
        last = None
        for attempt in range(max_retries + 1):
            try:
                return _openrouter_call(body(1, s), timeout), s
            except Exception as ex:
                last = ex
                with _RETRY_LOCK:
                    retried += 1
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"llm() failed after {max_retries + 1} attempts "
                           f"(role={role}, model={e['resolved']}): {last}")

    if len(texts) < k:
        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(k - len(texts), 8)) as ex:
                for j, s in ex.map(_one, range(len(texts), k)):  # in sample order
                    record(j, s, 1)
        except Exception:
            if CURRENT_RUN is not None:
                CURRENT_RUN._calls["errors"] += 1
                CURRENT_RUN._calls["retried"] += retried
            raise

    meta = {
        "role": role, "model": e["resolved"], "route_requested": route.get("order"),
        "k_impl": "n_sampling" if n_sampled else "independent_calls",
        "cache_prefix_chars": len(cache_prefix) if _cache_on else 0,
        "providers": sorted({r["provider"] for r in requests if r["provider"]}),
        "generation_ids": [r["generation_id"] for r in requests],
        "returned_models": sorted({r["returned_model"] for r in requests if r["returned_model"]}),
        "system_fingerprints": sorted({r["system_fingerprint"] for r in requests if r["system_fingerprint"]}),
        "usage": {f: sum(r["usage"][f] for r in requests)
                  for f in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens")},
        "cost_usd_reported": round(sum(r["cost"] or 0 for r in requests), 8),
        "requests": requests, "retried": retried,
    }
    if CURRENT_RUN is not None:
        CURRENT_RUN.log_call(role, e["resolved"],
                             {"temperature": temperature, "k": k, "seed": seed,
                              "reasoning_effort": reasoning_effort, "max_tokens": max_tokens},
                             prompt, texts, meta)
    return texts, meta


def winsorize8(values):
    """THE shared k=8 aggregator (W2 3.2, implemented once here so it cannot fork - W3/W-B/W5
    import this). Sort the 8 values s1 <= ... <= s8, replace the min with the 2nd-lowest and
    the max with the 2nd-highest, return the arithmetic mean of the resulting 8 values."""
    if len(values) != 8:
        raise ValueError(f"winsorize8 takes exactly 8 values, got {len(values)}")
    s = sorted(values)
    s[0], s[-1] = s[1], s[-2]
    return sum(s) / 8.0


class Run:
    """Provenance-manifested run context (W1 3.3/3.5; manifest_version 2 per Amendment A3).
    EVERY paper-bound experiment wraps itself in one:

        with Run("w2-ablation", params={...}, replicate=1, seed=11,
                 inputs=["pairs_master_frozen.json"], spec="specs/w2-ablation-grid.md") as run:
            ...
            run.save("results.json", res)

    Creates results/<experiment>/<run_id>/ (run_id = <YYYYMMDD-HHMMSS>-<git7>-r<replicate>),
    writes manifest.json (status=running) on enter, finalises it (status, models+routes,
    usage, cost, outputs) on exit, and appends a spend-typed line to results/cost_ledger.jsonl.
    Refuses to start if the harness subtree is git-dirty (gate G3), unless allow_dirty=True
    (smoke/exploratory only, per W1 3.4's carve-out) - in which case the manifest discloses
    the exact uncommitted paths as git_dirty_paths. A dirty manifest can never back a paper
    number: check_manifests accepts disclosed-dirty for w1-smoke only.
    """
    def __init__(self, experiment, params=None, replicate=1, seed=None, inputs=(),
                 spec=None, allow_dirty=False, spend="openrouter_credits"):
        commit, dirty = _git_state()
        if dirty and not allow_dirty:
            raise RuntimeError("harness tree is git-dirty - commit before a paper-bound run (G3): "
                               + "; ".join(dirty))
        self.run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{commit[:7]}-r{replicate}"
        self.dir = os.path.join(RESULTS, experiment, self.run_id)
        os.makedirs(self.dir)
        self.spend = spend
        self._calls = {"total": 0, "errors": 0, "retried": 0}
        self._usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0}
        self._cost_reported, self._models, self._t0 = 0.0, {}, time.time()
        self._percall = gzip.open(os.path.join(self.dir, "per_call.jsonl.gz"), "at")
        self.manifest = {
            "manifest_version": 2, "experiment": experiment, "run_id": self.run_id,
            "replicate": replicate, "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": commit, "git_dirty": bool(dirty), "spec": spec,
            "script": os.path.basename(sys.argv[0]) if sys.argv[0] else None,
            "params": params or {}, "seed": seed,
            "inputs": [{"file": f, "sha256": _sha256(os.path.join(HERE, f))} for f in inputs],
            "prompts_sha256": {}, "package_versions": _package_versions(),
        }
        if dirty:  # allow_dirty runs disclose exactly what was uncommitted (never paper-bound)
            self.manifest["git_dirty_paths"] = dirty
        self._flush()

    def log_call(self, role, resolved, params, prompt, texts, meta):
        self._calls["total"] += meta.get("requests") and len(meta["requests"]) or 1
        self._calls["retried"] += meta.get("retried", 0)
        for f in self._usage:
            self._usage[f] += meta["usage"].get(f, 0)
        self._cost_reported += meta.get("cost_usd_reported") or 0
        m = self._models.setdefault(role, {
            "role": role, "resolved": resolved, "provider": "openrouter",
            "route": {"requested": meta.get("route_requested"), "served": []},
            "temperature": params.get("temperature"), "k": params.get("k", 1),
            "seed": params.get("seed"), "reasoning_effort": params.get("reasoning_effort"),
            "k_impl": meta.get("k_impl"), "system_fingerprints": []})
        for p in meta.get("providers", []):
            if p not in m["route"]["served"]:
                m["route"]["served"].append(p)
        for fp in meta.get("system_fingerprints", []):
            if fp not in m["system_fingerprints"]:
                m["system_fingerprints"].append(fp)
        if meta.get("k_impl") == "independent_calls":  # a single n-drop marks the whole role
            m["k_impl"] = "independent_calls"
        self._percall.write(json.dumps({
            "t": time.time(), "role": role, "model": resolved, "params": params,
            "prompt": prompt, "outputs": texts,
            "generation_ids": meta.get("generation_ids"), "providers": meta.get("providers"),
            "returned_models": meta.get("returned_models"), "k_impl": meta.get("k_impl"),
            "usage": meta.get("usage"), "cost": meta.get("cost_usd_reported"),
            "requests": meta.get("requests")}) + "\n")

    def register_prompts(self, prompts):
        """prompts = {name: template_text}. Saved verbatim + hashed into the manifest."""
        json.dump(prompts, open(os.path.join(self.dir, "prompts.json"), "w"), indent=1)
        self.manifest["prompts_sha256"] = {
            k: hashlib.sha256(v.encode()).hexdigest() for k, v in prompts.items()}
        self._flush()

    def save(self, name, obj):
        json.dump(obj, open(os.path.join(self.dir, name), "w"), indent=1)

    def _flush(self):
        json.dump(self.manifest, open(os.path.join(self.dir, "manifest.json"), "w"), indent=1)

    def __enter__(self):
        global CURRENT_RUN
        CURRENT_RUN = self
        return self

    def __exit__(self, exc_type, exc, tb):
        global CURRENT_RUN
        CURRENT_RUN = None
        self._percall.close()
        if self._cost_reported:  # measured credits from OpenRouter usage accounting - preferred
            cost, source = round(self._cost_reported, 6), "openrouter_usage_reported"
        elif len(self._models) == 1:  # lock-price fallback; multi-model stays None (conservative)
            lock = json.load(open(MODELS_LOCK))
            price = {v["resolved"]: v.get("price_usd_per_mtok") or {} for v in lock["models"].values()}
            p = price.get(next(iter(self._models.values()))["resolved"], {})
            cost = round((self._usage["prompt_tokens"] / 1e6) * (p.get("in") or 0)
                         + (self._usage["completion_tokens"] / 1e6) * (p.get("out") or 0), 6)
            source = "lock_prices"
        else:
            cost, source = None, None
        self.manifest.update({
            "status": "failed" if exc_type else "complete",
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "models": list(self._models.values()), "calls": self._calls,
            "usage": self._usage, "cost_usd": cost, "cost_source": source,
            "spend": self.spend,
            "outputs": sorted(f for f in os.listdir(self.dir) if f != "manifest.json")})
        self._flush()
        with open(os.path.join(RESULTS, "cost_ledger.jsonl"), "a") as f:
            f.write(json.dumps({
                "run_id": self.run_id, "experiment": self.manifest["experiment"],
                "date": self.manifest["finished_utc"], "status": self.manifest["status"],
                "calls": self._calls["total"], **self._usage,
                "cost_usd": cost, "spend": self.spend,
                "wall_s": round(time.time() - self._t0)}) + "\n")
        return False
