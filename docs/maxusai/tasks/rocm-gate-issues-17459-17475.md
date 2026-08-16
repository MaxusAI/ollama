# ROCm/gfx1151: reproduce the two issues that hold the AMD upgrade gate

**For an agent on 10.8.0.4.** Read [`amd-upgrade-gate.md`](../amd-upgrade-gate.md) first.

Two upstream bug reports keep the AMD/gfx1151 deployment pinned at 0.32.1. Neither is being
worked on: one is in triage with its reproduction questioned, the other was closed for
silence. **We have the hardware both were reported on. Nobody else has volunteered.**

The goal is not to fix ollama. It is to produce evidence good enough that the issues move —
and, either way, to tell us whether the gate can ever lift.

> [!IMPORTANT]
> **Both were reported against 0.32.5 (payload b10091). Our host runs 0.32.1 (b9888).**
> Reproducing them therefore requires building and running the payload the gate blocks.
> That is allowed **for testing on a scratch port**, and only that. Do not `docker restart`
> the production container onto it, do not touch `ollama-rocm`, and do not change what
> `:11434` serves. Roll back to the running image the moment a test finishes.
>
> **Synthetic documents only.** #17475 is a PII leak; its reporter used real insurance
> records. Generate stand-ins (the vision-suite `visimgs/` invoice generator is fine) and
> put a unique marker string in the donor. Never use customer data for this.

## Host facts

| | |
|---|---|
| Host | `10.8.0.4`, `glenn-NucBox-EVO-X2`, Ryzen AI Max+ 395 / Radeon 8060S, **gfx1151** (RDNA 3.5) |
| SSH | `ssh -i ~/.ssh/id_ed25519_NucBox-EVO-X2-glenn glenn@10.8.0.4` |
| ROCm | 7.2.1 · HIP 7.2.53211 |
| Production container | `ollama-rocm`, image `maxusai-ollama:0.32.1-rocm-dynres-296eb020`, port 11434 — **leave alone** |
| Deployment defs | `~/deployments/ollama/docker/ollama-rocm/` (compose, Makefile, `.env`) |
| Repo checkout | `/opt/github/MaxusAI/ollama` — on an old branch with only compat 001/002; `git fetch` before using |

---

## 1. [#17459](https://github.com/ollama/ollama/issues/17459) — gemma4 emits repeated `<unused49>` when `think=false`

**Status: open.** A contributor offered to take it; nothing landed. A maintainer
(`rick-github`) then questioned the reproduction itself:

> The gemma output files show no sign of `<unused49>`. Was the model reloaded?

**Why we are well placed:** the reporter's hardware is a *Framework Desktop Max+ 395 with
Radeon 8060S* — the same gfx1151 silicon as this host, different chassis. If it is
hardware- or ROCm-specific, we can show that. If it is not, we can show that too, which is
just as useful to a maintainer stuck on "can you even reproduce this".

**Reported shape:** `gemma4:31b` via `/api/chat` with `"think": false` emits runs of
`<unused49>`; the same request is fine with `"think": true`. Also breaks the VS Code
extension, which disables thinking.

**What to produce:**

1. A minimal `/api/chat` request with `"think": false` that shows the tokens, and the same
   request with `"think": true` that does not. Same model, same prompt, same session.
2. **Answer the maintainer's question directly** — was the model reloaded between the two?
   Force a cold reload for each arm and say so, with `load_duration` from the response as
   evidence. A warm/cold difference would be the most valuable single finding here, and it
   is exactly the failure mode this fork keeps hitting.
3. Whether it reproduces on 0.32.1/b9888 (our production payload) as well as on 0.32.5.
   If b9888 is clean, that narrows it to the same b9888→b10091 window that produced the
   MMQ regression — see [`mmq-padding-regression-window.md`](../mmq-padding-regression-window.md).
4. Runner log covering both arms, `--verbose` output, and exact model digest.

**Note on our own gemma4 patches.** This fork carries 004 (`gemma4-budget-fill`). Run the
comparison on **stock `ollama/ollama:0.32.5`** as well as our build, or a maintainer will
rightly discount it. If stock is clean and ours is not, that is our bug, not theirs — and
we need to know that.

---

## 2. [#17475](https://github.com/ollama/ollama/issues/17475) — cross-request content leakage on a shared slot

**Status: closed as `not_planned`** after a *single* maintainer comment asking for server
logs, image type and dimensions, and the prompt. There is no sign the reporter replied. It
was closed for silence, not because anyone judged it harmless.

**What it claims.** Under concurrent vision load with client aborts, one request's image
content appears in another's output. Found in production: one customer's VIN written into
five other customers' extraction results, verified absent from those source images.

Their controlled A/B, which is unusually good and worth replicating exactly:

| protocol | result |
|---|---|
| V1 — strict alternation donor→victim, single-threaded, ×6 | 0/6 contaminated |
| V2 — abort donor mid-generation (8 s client timeout), then victim, ×6 | 0/6 contaminated |
| V3 — 3 concurrent threads: ① donor extracts, half aborted at 8 s ② noise loop alternating a text model and another VLM ③ victim extracts ×8 | **8/8 contaminated** |

So neither slot reuse nor abort residue alone triggers it; the combination does.

**Their runner config:** `-np 1`, context-shift, `-c 8192`, `--no-jinja`,
`--chat-template chatml`, model `qwen3-vl:30b-a3b-thinking`, `think: true`,
`temperature: 0`, `num_predict: 6144`.

**Why we are well placed:** this fork already cold-restarts the server between every
benchmark cell *because of this bug* — `run_grid.sh` cites it by number. We have the
harness, two VLMs, and a text model for the noise loop.

**What to produce:**

1. V1/V2/V3 run on gfx1151, with a synthetic donor carrying a unique marker string and a
   victim verified not to contain it. Report the contamination counts per protocol, not a
   narrative.
2. **Exactly what the maintainer asked for**: full server logs across the run, image type
   and pixel dimensions, and the verbatim prompt. That request is why the issue died.
3. Whether it reproduces on b9888 as well as b10091. If our production payload leaks too,
   that is a live production concern on this host, not just a gate question — say so
   immediately rather than finishing the matrix.
4. Whether `-np 1` is what our deployment actually runs. Check the runner command line in
   the container logs; if we run higher parallelism the exposure differs.

---

## Reporting back

Write `rocm-gate-issues-result.md` next to this file, and:

- **Say which payload each result came from.** b9888 and b10091 are different code; a result
  without its payload is not a result.
- **Distinguish observed from inferred**, per [ADR 0024](../adr/0024-locate-faults-before-fixing-them.md).
  Three hypotheses died in the MMQ investigation because a value was reasoned about rather
  than printed.
- **A clean run is not proof of absence** for #17475 — it is a concurrency race, and the
  reporter needed three protocols to surface it. Report protocol-by-protocol counts.
- If either reproduces, the write-up is the deliverable. Whether to post it upstream is
  Glenn's call: ggml-org and ollama both restrict AI-written posts, and #17475 touches PII.
  See [`upstream-mmq-submission-material.md`](../upstream-mmq-submission-material.md) for how
  that was handled last time.

## What this unblocks

The gate lifts only when both are **fixed, with the fix in the target tag** — not merely
closed. #17475 being closed as `not_planned` already satisfies a naive reading, which is why
the gate wording was corrected. Right now neither has a route to being fixed, so the AMD host
stays on b9888 indefinitely by default. Evidence from this task is what turns that from a
default into a decision.
