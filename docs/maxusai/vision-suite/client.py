#!/usr/bin/env python3
"""THE ollama request path for this suite. One function, used by everything.

SPEC vision-harness-reuse.md (H1: one runner, one set of helpers) and
[ADR 0028]. Before this module there were two hand-built request paths --
vision_suite.gen() and finetext_probe.run() -- and they had already drifted:

  * vision_suite defaulted to /api/generate, finetext_probe defaulted to
    /api/generate, while run_engine_compare.sh passed ENDPOINT=chat to both, so
    the default was dead code that only fired when someone invoked a probe by
    hand -- silently measuring a different endpoint than the campaign did.
  * vision_suite normalised `thinking` out of the chat envelope; finetext_probe
    did not, and persisted neither the response body nor the reasoning text. A
    low recall number from that probe was therefore unexaminable: the evidence
    was gone the moment the request returned.
  * the context-overflow 400 was translated into an actionable error in
    vision_suite only. The same misconfiguration against finetext surfaced as a
    bare `HTTP Error 400`.

Every one of those is the same class of bug: a second copy of the hardest part.
Add features HERE. A probe that needs something this does not do should grow
this function, not fork it.
"""
import base64
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

from sampling import sampling_for

DIR = os.path.dirname(os.path.abspath(__file__))

# Transport-failure backoff, in seconds. Overridable for tests; a host that is
# genuinely down should fail in under a minute rather than hang a campaign.
RETRY_BACKOFF = [int(x) for x in
                 os.environ.get("RETRY_BACKOFF", "5,15,30").split(",") if x]


def default_num_ctx():
    return int(os.environ.get("NUM_CTX", "16384"))


def default_num_predict():
    return int(os.environ.get("NUM_PREDICT", "2200"))


def think_on():
    return os.environ.get("THINK", "false") == "on"


def endpoint():
    """chat by default. /api/chat is the endpoint upstream keeps current;
    /api/generate has lagged on newer features. It is also what every campaign
    measured, since run_engine_compare.sh has always passed ENDPOINT=chat.

    MEASURED 2026-08-19, gemma4:31b-it-q4_K_M on 0.32.14-rc0-dynres (CUDA):
    the two endpoints are token-IDENTICAL -- prompt_eval_count 1511 vs 1511
    think-off, 1514 vs 1514 think-on, byte-identical answers think-off. ollama's
    /api/generate applies the model's template too (it is not raw unless
    raw:true), so there is no chat-template overhead to differ over.

    The real difference is THINKING, and it is MODEL-SPECIFIC, not a blanket
    property of the endpoint:

      gemma4:31b-it-q4_K_M   /api/generate thinking 0 chars, /api/chat 2021,
                             eval_count 1281 vs 1277 -- generated either way,
                             returned only by chat
      nemotron3:33b-q4_K_M   /api/generate returns 3909 chars (variants.py)
      qwen3.8:27b-q4_K_M     /api/generate returns reasoning normally

    So do not read this as "/api/generate is broken". For gemma4 on this build it
    silently loses the reasoning half while still paying for it, and since the
    suite must work across models regardless, chat is the safe default.

    preflight/probes.py stays pinned to /api/generate because its expectations
    were calibrated there and equivalence is measured, not guaranteed, across
    models and builds -- endpoint_compare.py exists to re-check it."""
    return os.environ.get("ENDPOINT", "chat")


def fingerprint(text):
    """Short stable hash of an exact string. No normalisation — whitespace is
    part of the prompt."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def b64_file(path):
    return base64.b64encode(open(path, "rb").read()).decode()


def context_error(e, num_predict, num_ctx):
    """Turn the server's context-overflow 400 into an actionable message.

    The invariant is prompt + num_predict <= num_ctx, and the server enforces it
    before generating. Raising NUM_PREDICT without raising NUM_CTX therefore does
    not relieve truncation, it converts it into a bare `HTTP Error 400: Bad
    Request` that reads as a model or payload fault. It is neither — it is a
    harness misconfiguration, and this says so with the numbers needed to fix it."""
    try:
        body = e.read().decode()
    except Exception:
        body = ""
    m = re.search(r"n_prompt_tokens\\?[\"']?:\s*(\d+).*?n_ctx\\?[\"']?:\s*(\d+)", body, re.S)
    if e.code == 400 and ("exceed_context_size" in body or m):
        if m:
            need, ctx = int(m.group(1)), int(m.group(2))
            return RuntimeError(
                f"num_ctx too small: request needs {need} tokens but num_ctx is {ctx}. "
                f"prompt + num_predict must fit num_ctx (this call used "
                f"num_predict={num_predict}, num_ctx={num_ctx}). "
                f"Raise NUM_CTX to at least {need + 2048} (leaving headroom) — "
                f"raising NUM_PREDICT alone cannot fix this.")
        return RuntimeError(
            f"context overflow with num_predict={num_predict}, num_ctx={num_ctx}: {body[:200]}")
    return RuntimeError(f"HTTP {e.code}: {body[:200]}")


def generate(host, model, prompt, images, num_predict=None, num_ctx=None,
             fmt="json", extra_opts=None, endpoint_override=None,
             think=None, apply_sampling=True, timeout=None,
             send_think="auto", use_env_opts=True, raw=False):
    """One vision request. Returns the server dict, normalised.

    `images` are base64 strings. The returned dict always carries `response`
    (answer text) and `thinking` (reasoning text) at the top level whichever
    endpoint ran, plus `_num_predict` / `_num_ctx` stamped with what ACTUALLY
    ran -- gen_opts, env and defaults all feed in, and recording the request's
    own values is what stops a score drifting from the window it was measured at.
    """
    # num_ctx=False OMITS the option entirely, leaving the server's own default.
    # measure.py's published token-budget numbers were taken without it, and
    # injecting one can reload the runner with different settings -- the counts
    # would survive but the payload would no longer be the one that was
    # calibrated. None means "use the harness default"; a number pins it.
    omit_ctx = num_ctx is False
    num_ctx = default_num_ctx() if num_ctx is None else num_ctx
    num_predict = default_num_predict() if num_predict is None else num_predict
    # think: None reads THINK from the environment (the campaign path); a bool
    # PINS it. variants.py isolates think-on as an experimental arm and must not
    # be steered by an env var meant for the runner.
    think = think_on() if think is None else bool(think)

    # Sampling is per-model and per-think-mode, NOT a hardcoded temperature 0.
    # Think-off is greedy (every published baseline depends on that); think-on
    # uses the model card's values, because greedy decoding is what made
    # reasoning fail to terminate. See sampling.py and
    # ../runaway-reasoning-under-think.md.
    opts = {"num_predict": num_predict}
    if not omit_ctx:
        opts["num_ctx"] = num_ctx
    # apply_sampling=False leaves sampling entirely to the caller. measure.py's
    # token-budget protocol counts prompt_eval_count with num_predict=1 and must
    # send exactly the payload its published numbers were measured with -- adding
    # per-model sampling there would be a silent change to a calibrated probe.
    if apply_sampling:
        opts.update(sampling_for(model, think))
    # use_env_opts=False blocks the ambient Runner options below. measure.py's
    # protocol EXISTS to measure the image token budget, so inheriting
    # IMAGE_MIN_TOKENS/IMAGE_MAX_TOKENS from the environment would let a stray
    # export silently rewrite the very quantity being measured -- and HEAD's
    # measure.py read no environment at all.
    if use_env_opts and os.environ.get("KV_CACHE_TYPE"):
        opts["kv_cache_type"] = os.environ["KV_CACHE_TYPE"]
    # Fork-only per-request vision budget (visionServerArgs in llm/llama_server.go,
    # arch-gated to gemma4 and nemotron_h_omni). Pinning these to upstream's
    # effective defaults turns a fork build into a BUDGET-MATCHED CONTROL, which is
    # the only way to separate "our larger token budget changed the result" from
    # "the llama.cpp payload differs" when comparing against a stock server on a
    # different LLAMA_CPP_VERSION. These are Runner options — changing them
    # reloads the model.
    for env, opt in (("IMAGE_MIN_TOKENS", "image_min_tokens"),
                     ("IMAGE_MAX_TOKENS", "image_max_tokens")):
        if use_env_opts and os.environ.get(env):
            opts[opt] = int(os.environ[env])
    # MTP / speculative draft depth (--spec-draft-n-max on the llama-server
    # side). Sent whenever DRAFT_NUM_PREDICT is set, INCLUDING "0": server-side
    # routes.go zeroes the option unless it was explicitly set, so "0" is the
    # only way to state "MTP off" rather than "unspecified", and an arm that
    # omitted it could not be distinguished from one that disabled it. Also a
    # Runner option -- changing it reloads the model. Inert on the MLX runner,
    # which never reads draft_num_predict and picks depth adaptively instead.
    if use_env_opts and os.environ.get("DRAFT_NUM_PREDICT") not in (None, ""):
        opts["draft_num_predict"] = int(os.environ["DRAFT_NUM_PREDICT"])
    if extra_opts:
        opts.update(extra_opts)

    payload = {"model": model, "stream": False, "options": opts}
    # raw skips the chat template on /api/generate. token_split.py's --server
    # mode counts prompt_eval_count of bare text, and a template wrapper would
    # add tokens that belong to the template, not the text. Inert unless set
    # (H4); meaningless on /api/chat, so callers pair it with
    # endpoint_override="generate".
    if raw:
        payload["raw"] = True
    if fmt:
        payload["format"] = fmt
    # send_think is TRI-STATE, because the callers genuinely differ and collapsing
    # them silently changed an experiment:
    #
    #   "auto"  send "think": false only when thinking is off; OMIT it when on.
    #           This is what vision_suite/finetext/extbench have always done.
    #   True    always send "think": <bool>, in BOTH directions. variants.py's
    #           thinkon arm needs this -- its whole purpose is to PIN thinking on,
    #           and omitting the field leaves it to a server default that
    #           measure.py:101 shows is model- and build-specific. An earlier
    #           version of this function could never emit "think": true, which
    #           silently turned that arm into "whatever the model defaults to".
    #   False   never send it. measure.py's calibrated count probes were measured
    #           with no such field, and ollama's template can render differently
    #           depending on it -- worth a token or two, invisible in a quality
    #           score, decisive when prompt_eval_count IS the measurement.
    if send_think is True:
        payload["think"] = bool(think)
    elif send_think == "auto" and not think:
        payload["think"] = False

    # endpoint_override PINS the endpoint regardless of env, for callers whose
    # published numbers were calibrated on one of them. Measured equivalent on
    # token counts (see endpoint()), but equivalence is a measurement per model
    # and build, not a guarantee -- and /api/generate drops reasoning text.
    ep = endpoint_override or endpoint()
    if ep == "chat":
        msg_ = {"role": "user", "content": prompt}
        if images:
            msg_["images"] = images
        payload["messages"] = [msg_]
        url = host + "/api/chat"
    else:
        payload["prompt"] = prompt
        # An EMPTY image list is a text-only request and must omit the key
        # entirely: measure.py's text-prefix calibration compares an
        # image-bearing request against a genuinely image-free one, and sending
        # "images": [] is neither.
        if images:
            payload["images"] = images
        url = host + "/api/generate"

    body = json.dumps(payload).encode()
    tmo = timeout or int(os.environ.get("HTTP_TIMEOUT", "1800"))
    # RETRY TRANSPORT FAILURES, NEVER A DETERMINISTIC REJECTION.
    #
    # A server restart, a dropped connection or a 502/503/504 is a fact about the
    # moment, not about the request -- retrying is free of meaning-change and
    # saves a campaign. A 4xx is the server telling us the request itself is
    # wrong: a context-overflow 400 will fail identically forever, and retrying
    # it three times just delays an actionable error by 50 seconds.
    #
    # Measured 2026-08-19: a container restart on the remote host killed rep 3 of
    # a 3-repeat sweep with "Remote end closed connection without response",
    # after ~23 minutes of generation. Reps 1 and 2 were byte-identical, so the
    # lost cell cost an hour and told us nothing new.
    #
    # The backoff is 5s / 15s / 30s: long enough for a container to come back,
    # short enough not to mask a host that is genuinely down. Each attempt is
    # announced, because a silent retry hides exactly the event a reader needs to
    # know about when a cell's timing looks strange.
    attempt, retries, last = 0, [], None
    while True:
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=tmo))
            break
        except urllib.error.HTTPError as e:
            if e.code in (502, 503, 504) and attempt < len(RETRY_BACKOFF):
                last = f"HTTP {e.code}"
            else:
                raise context_error(e, num_predict, num_ctx) from None
        except Exception as e:                      # URLError, RemoteDisconnected, socket
            if attempt >= len(RETRY_BACKOFF):
                raise RuntimeError(
                    f"{type(e).__name__}: {e} (after {attempt} retries at "
                    f"{RETRY_BACKOFF[:attempt]}s)") from None
            last = f"{type(e).__name__}: {e}"
        wait = RETRY_BACKOFF[attempt]
        attempt += 1
        print(f"##### TRANSPORT RETRY {attempt}/{len(RETRY_BACKOFF)} in {wait}s "
              f"— {last}", flush=True)
        time.sleep(wait)
        retries.append({"attempt": attempt, "waited_s": wait, "error": last})

    # Normalise both envelopes to the same shape. /api/generate returns
    # `response` and `thinking` at the top level; /api/chat nests them under
    # `message` as `content` and `thinking`.
    msg = r.get("message") or {}
    if "response" not in r:
        r["response"] = msg.get("content", "")
    r["thinking"] = r.get("thinking") or msg.get("thinking", "") or ""
    # A cell that needed retries had the server disappear underneath it. Its
    # timings are not comparable with a clean cell's, and that must be visible in
    # the scores rather than inferred from a log nobody kept.
    if retries:
        r["_retries"] = retries
    # WORKLOAD FINGERPRINT. A score is only comparable with another score of the
    # SAME workload, and until now nothing recorded what the workload was: a
    # prompt edit silently made every prior number incomparable, with no way to
    # detect it afterwards. This is the version-locking a benchmark suite needs
    # (3DMark cites a scene version with every score); it lives here because
    # every arm's request passes through this function, so no arm can forget it.
    #
    # Exact bytes, no normalisation — whitespace IS the prompt. The image is
    # fingerprinted too: fixtures are gitignored and regenerated, so a
    # re-rendered visimgs/ would otherwise change what was measured invisibly.
    r["_prompt_sha"] = fingerprint(prompt)
    r["_images_sha"] = fingerprint("".join(images)) if images else None
    r["_host"] = host
    r["_server_version"] = server_version(host)
    r["_num_predict"] = num_predict
    r["_num_ctx"] = None if omit_ctx else num_ctx
    # Stamped for the same reason as the two above (ADR 0012 rule 1): a score
    # measured with MTP on and one measured with it off are otherwise
    # indistinguishable in the file, and the arm would live only in the tag.
    if "draft_num_predict" in opts:
        r["_draft_num_predict"] = opts["draft_num_predict"]
    return r


def persist(tag, name, r):
    """Write the answer and the reasoning to disk, and return their char counts.

    The reasoning text is the reason this exists. gen() always captured it and
    nothing ever wrote it down, so for every think-on cell ever run "how many
    tokens did it spend thinking" is unrecoverable -- eval_count is the total and
    the two halves were discarded at the end of the request. That is the headline
    number for req/hour planning. token_split.py turns these files into exact
    thinking/answer/control token counts afterwards.

    Reasoning goes to its own file so every existing consumer of resp_*.json is
    untouched, and is only written when non-empty so think-off cells add nothing.
    """
    text = r.get("response", "") or ""
    think = r.get("thinking", "") or ""
    open(f"{DIR}/resp_{tag}_{name}.json", "w").write(text)
    if think:
        open(f"{DIR}/think_{tag}_{name}.txt", "w").write(think)
    return {"thinking_chars": len(think), "answer_chars": len(text)}


_VERSION_CACHE = {}


def server_version(host, timeout=10):
    """The serving build's version string, cached per host for the process.

    Recorded on every cell because a score whose HOST and BUILD are unknown is
    not comparable with anything -- the same failure ADR 0012 rule 6 fixed for
    num_ctx, and worse here: host changes throughput ~4x, and build changes
    BEHAVIOUR (gemma4 returns no reasoning on /api/generate for one build and
    fine on another). Reconstructing this from tag-name conventions afterwards is
    not evidence, and nothing in the data would stop an Apple cell being pooled
    with a CUDA one."""
    if host not in _VERSION_CACHE:
        try:
            r = json.load(urllib.request.urlopen(host + "/api/version", timeout=timeout))
            _VERSION_CACHE[host] = r.get("version")
        except Exception:
            _VERSION_CACHE[host] = None
    return _VERSION_CACHE[host]


def unload(host, model, timeout=120):
    """Force the model out of memory so the next request loads it cold.

    `keep_alive: 0` is the remote equivalent of RESTART_CMD. The runner's
    cold-start hook restarts the serving PROCESS, which is impossible against a
    host you do not control -- so every remote campaign silently ran warm while
    local ones ran cold, and their load_duration and first-token latency were
    never comparable. preflight/probes.py has used this call since it was
    written; it simply lived on the wrong side of the harness.

    NOT a full process restart: it evicts the model but leaves the server's own
    caches and any other loaded model alone. That is the right granularity for
    per-model cold start and is what preflight relies on."""
    req = urllib.request.Request(
        host + "/api/generate",
        data=json.dumps({"model": model, "keep_alive": 0}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=timeout).read()
        return True
    except Exception:
        return False


def loaded(host, timeout=15):
    """Models currently resident, as [(name, bytes)]. [] on any error."""
    try:
        r = json.load(urllib.request.urlopen(host + "/api/ps", timeout=timeout))
        return [(m["name"], m.get("size", 0)) for m in r.get("models", [])]
    except Exception:
        return []


def evict_others(host, keep=None, wait_s=90, poll=3):
    """Unload every resident model except `keep`, and WAIT for it to take effect.

    The wait is the whole point. `keep_alive: 0` returns as soon as the request
    is accepted, not when the weights are actually out of memory, so a sweep that
    unloads and immediately loads the next model can hold BOTH at once. On a
    sweep of large models that is how a host runs out of memory partway through a
    campaign -- and the failure lands on whichever model happened to be next,
    which reads as that model being too big rather than as the harness never
    having freed the previous one.

    `OLLAMA_MAX_LOADED_MODELS=1` is the server-side answer and is mandatory for
    a locally served sweep (SPEC apple-silicon-build). It is unavailable against
    a host you do not control, which is exactly when this matters.

    Returns (evicted, still_resident)."""
    evicted = []
    for name, _ in loaded(host):
        if keep and name == keep:
            continue
        if unload(host, name):
            evicted.append(name)
    if not evicted:
        # `keep` staying resident is the desired state, not a failure — reporting
        # it as "still resident" made a clean no-op look like a stuck eviction.
        return [], [n for n, _ in loaded(host) if n != keep]
    waited = 0
    while waited < wait_s:
        resident = [n for n, _ in loaded(host) if n != keep]
        if not resident:
            return evicted, []
        time.sleep(poll)
        waited += poll
    return evicted, [n for n, _ in loaded(host) if n != keep]


def evict_all(host, wait_s=90, poll=3):
    """Unload EVERY resident model and wait. Returns (evicted, still_resident).

    Distinct from evict_others(keep=X) in intent, not just in arguments: that one
    makes room for a model about to load, this one hands the host back. Use it
    before yielding a shared machine to someone else, after a campaign, or when a
    run died and left weights resident -- the harness has already had to free a
    host that way once, and doing it by hand is how the wrong model gets killed.

    Anything it could not evict is RETURNED, not raised: a model held by another
    client is not this harness's to kill, and failing over it would be worse than
    proceeding with the fact recorded."""
    return evict_others(host, keep=None, wait_s=wait_s, poll=poll)


if __name__ == "__main__":
    # Tiny CLI so run_engine_compare.sh can cold-start a remote model without
    # duplicating the request (SPEC H9: one request path, shell included).
    import sys
    if len(sys.argv) == 4 and sys.argv[1] == "unload":
        ok = unload(sys.argv[2], sys.argv[3])
        print(f"unload {sys.argv[3]}: {'ok' if ok else 'failed (continuing)'}")
    elif len(sys.argv) == 3 and sys.argv[1] == "evict-all":
        gone, stuck = evict_all(sys.argv[2])
        print(f"evicted {gone or 'nothing'}"
              + (f"; STILL RESIDENT after wait: {stuck}" if stuck else ""))
    elif len(sys.argv) in (3, 4) and sys.argv[1] == "evict":
        keep = sys.argv[3] if len(sys.argv) == 4 else None
        gone, stuck = evict_others(sys.argv[2], keep=keep)
        print(f"evicted {gone or 'nothing'}"
              + (f"; STILL RESIDENT after wait: {stuck}" if stuck else ""))
    else:
        sys.exit("usage: client.py unload    <host> <model>\n"
                 "       client.py evict     <host> [keep-model]\n"
                 "       client.py evict-all <host>")
