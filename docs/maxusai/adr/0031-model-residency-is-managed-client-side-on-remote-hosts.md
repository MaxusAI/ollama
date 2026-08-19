# ADR 0031: model residency and cold start are managed client-side on hosts the harness does not control

- **Status:** accepted 2026-08-19. Implemented as `client.unload()`,
  `client.loaded()` and `client.evict_others()`, wired into
  `run_engine_compare.sh` and specified as
  [SPEC H10](../spec/vision-harness-reuse.md).
- **Date:** 2026-08-19

## Context

Two harness invariants were only ever true on a machine we own.

**Cold start.** `RESTART_CMD` restarts the serving *process* between models, and
every published latency and `load_duration` figure assumes it ran. It cannot run
against a host we do not control, and the runner did not say so — it simply
skipped the step. Remote campaigns therefore measured warm loads while local ones
measured cold, and nothing in the scores distinguished them.

**Residency.** `OLLAMA_MAX_LOADED_MODELS=1` is
[mandatory for a served sweep](../spec/apple-silicon-build.md), and it is a
*server-side* environment variable fixed when the server starts. On a remote
endpoint it is whatever that host's operator chose. A sweep that loads model
after model can therefore leave every previous model resident, and on a host
serving several ports (10.8.0.6 runs five) the harness is not even the only
consumer of that memory.

The failure mode is the expensive kind: it lands on whichever model happens to be
next in the list, so it reads as *that model being too big* rather than as the
harness never having freed the previous one. This host has already been driven
to 106 GB used and 53.89 GB of swap exactly that way.

`keep_alive: 0` solves both, and `preflight/probes.py` has used it since it was
written — to force a fresh `load_hparams` block. It simply lived on the wrong
side of the harness, available to preflight and to nothing else.

## Decision

**1. Cold start falls back to `keep_alive: 0` when `RESTART_CMD` is absent.**
Per-model eviction, not a process restart. `COLD_START=0` opts out. When
`RESTART_CMD` is set, behaviour is unchanged — the fallback is inert by default
in the sense [H4](../spec/vision-harness-reuse.md) requires.

**2. Before loading a model, every OTHER resident model is evicted, and the
harness WAITS for the memory to come back.** Unloading the incoming model is
close to a no-op; the memory risk is the outgoing one. `keep_alive: 0` returns
when the request is *accepted*, not when the weights are gone, so an unload
immediately followed by a load can hold both models at once — which is the
precise shape of the OOM this is meant to prevent. `evict_others()` polls
`/api/ps` until the eviction is observable, with a bounded wait.

**3. A process restart and a `keep_alive: 0` eviction are NOT equivalent, and
results from the two are not interchangeable.** Eviction frees the model; the
server's own caches and any other loaded model survive. That is the right
granularity for per-model cold start, and it is what preflight relies on, but a
`load_duration` measured after eviction must not be quoted against one measured
after a full restart.

**4. Two eviction verbs, because they are different intents.**
`evict_others(host, keep=X)` makes room for a model about to load;
`evict_all(host)` hands the host back — after a campaign, before yielding a
shared machine, or when a run died and left weights resident. That last case is
not hypothetical: this host has already had to be freed by hand mid-session, and
doing it by hand is how the wrong model gets killed.

**5. This lives in `client.py`, including the shell entry point.** `client.py
unload <host> <model>`, `client.py evict <host> [keep]` and `client.py evict-all
<host>` exist so the runner does not grow its own `curl` —
[H9](../spec/vision-harness-reuse.md) is one request path, and a shell script
issuing its own API call is a second one.

**6. Transport failures are retried with backoff; deterministic rejections are
not.** 5s / 15s / 30s, then fail with the attempt history in the message. A
dropped connection, a socket error or a 502/503/504 is a fact about the moment
rather than about the request, and a restarting container answers 503 before it
answers properly. A 4xx is the opposite: a context-overflow 400 will fail
identically forever, so retrying it three times only delays an actionable error
by 50 seconds. `urlopen` is called exactly once for a 400, and there is a test
asserting it.

Measured 2026-08-19: a container restart on 10.8.0.6 killed rep 3 of a 3-repeat
sweep with `Remote end closed connection without response`, after ~23 minutes of
generation. Reps 1 and 2 were already byte-identical, so an hour was spent
learning nothing. Retries are announced on stdout and recorded as `_retries` on
the cell — a cell whose server vanished mid-generation has timings that are not
comparable with a clean cell's, and that must be visible rather than inferred
from a log nobody kept.

## Consequences

- Remote campaigns now cold-start per model. Their `load_duration` becomes
  meaningful for the first time, and is comparable to other eviction-based runs
  but not to `RESTART_CMD` runs (decision 3).
- A sweep across large models on a shared host no longer depends on that host's
  `OLLAMA_MAX_LOADED_MODELS`. It still cannot stop *another* tenant loading a
  model, which is why endpoint contention remains a separate check
  (`endpoint_exclusive` in preflight).
- The wait costs wall-clock on every model switch. That is deliberate: the
  alternative is a campaign that dies partway through and blames the wrong model.
- A campaign now survives a host restart instead of losing the cell it was on.
  It does NOT survive a host that stays down: the backoff totals 50 seconds by
  design, so a dead endpoint fails fast rather than hanging a sweep overnight.
- `evict_others()` returns what it could not evict rather than raising. A model
  held by another client is not this harness's to kill, and failing the run over
  it would be worse than proceeding with the fact recorded.

## Alternatives considered

**Require `OLLAMA_MAX_LOADED_MODELS=1` on every benchmarked host.** Rejected: it
is not ours to set on a host we do not control, and a requirement that cannot be
enforced is a requirement that is silently unmet.

**Unload only the model about to be loaded.** Rejected — this was the first
implementation and it is close to useless. The incoming model is usually not
resident; the outgoing one is.

**Fire `keep_alive: 0` and proceed immediately.** Rejected. The call returns on
acceptance, so without the poll the harness can still hold two models
simultaneously, which is the exact condition being prevented.
