# ADR 0024 — Locate faults by observation before proposing a fix

Status: accepted, 2026-08-13

## Context

Chasing the qwen35moe `MUL_MAT_ID` crash
([[qwen35moe-mmq-investigation]]) cost roughly a day, and three of the hypotheses
along the way were wrong in the same way: a value that mattered was **reasoned about rather
than printed**. One of them was built and tested, and the patch turned out to modify a branch
that cannot execute, in a file the crashing op never enters.

The specific failures, all real:

- "`ids != nullptr` on the crashing call" — the pointer was never logged. It was inferred from
  a shape signature.
- "MMQ was exercised and exonerated" — the instrumented build only ever recorded one branch
  value. MMQ was never instrumented, and MMQ turned out to be the answer.
- "fusion-off does not help" — that probe ran text-only at `-c 8192` and never touched the
  image path it was supposed to exonerate.
- "`GGML_CUDA_FORCE_CUBLAS=ON` still crashes" — only a *build* log survived. No run log, no
  container. This claim partitioned the search space.
- A whole size bisection ran ascending in one warm process, where a prior large allocation
  masks the fault. Every "pass" in it was uninterpretable.

The underlying condition was economic: a full image build took ~90 minutes. At that price,
reasoning to a conclusion and building once feels rational, and instrumenting to find out
feels wasteful. It is the wrong trade whenever the reasoning can be wrong.

## Decision

**For any fault whose location is not directly observed, the first build is an instrumented
build, not a candidate fix.**

Concretely, for a crash in a compute backend:

1. **Make the attribution real.** Asynchronous error checks report where the error was
   *noticed*. Synchronise at the point of detection (`cudaStreamSynchronize` before
   `cudaGetLastError`) and disable graph capture, or the reported op is not evidence.
2. **Print the geometry.** The faulting op, tensor names, and every `src`/`dst` dim — not a
   summary, the actual numbers.
3. **Log every branch, not the one you suspect.** Instrumentation that can only confirm your
   hypothesis will confirm it.
4. **Capture a passing baseline under the same instrumentation.** A fault line alone is a
   data point; a fault line beside a clean run of the same code path is a controlled
   comparison, and it is usually what identifies the variable.
5. **A hypothesis earns a build only when it explains every known fact.** Explaining one fact
   while leaving the others unaddressed is the signature of a wrong hypothesis.

And for verification of any fix:

6. **Cold, target request first, fresh container** — for any failure mode that could be
   state-dependent.
7. **Run an unfixed control from the same tree, in the same session.** Otherwise a machine
   that has drifted into a forgiving state reads as a successful fix.
8. **Confirm the fix executes the same path** it was meant to correct, rather than routing
   around it.
9. **Verify the instrument itself**: that the env var reached the process, that the intended
   payload loaded, and that the patch is in the binary. Note `llama-server --version` reports
   a SHA from the CPU build stage that is invariant to `llama/compat/*.patch` and to the arch
   list, so it cannot establish any of this — and the preflight harness pins that same SHA
   (ADR 0011).

## Consequences

Instrumented builds become routine rather than a last resort, which is only affordable because
the dev loop was reduced from ~90 minutes to ~6–8 ([[spec/fast-platform-dev-loops]]).
The two decisions are a pair: the fast loop is what makes "measure first" the cheap option
rather than the disciplined one.

Some instrumentation is expensive at runtime — the per-node synchronise serialises the graph —
so instrumented patches are numbered `9xx` in `llama/compat/` and must never ship in a release
build.

The cost is one extra build cycle before any fix attempt. Against three failed hypotheses,
one of which consumed a 90-minute build and an hour of analysis, this is not a close call.

## Not covered

This ADR is about faults whose *location* is unknown. Where a fault is already localised — a
failing test with a stack trace, an assertion with a line number — go straight to the fix.
