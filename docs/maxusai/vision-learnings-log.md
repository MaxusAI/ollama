# Vision harness — learnings log

Append-only. Each entry is a **claim sharp enough to be wrong**, the number that
supports it, and the artefact that now enforces it. An entry with no enforcing
artefact is a note, not a learning — it will be forgotten and re-learned.

Format: `### YYYY-MM-DD — claim` / **Evidence** / **Enforced by** / **Cost**.
"Cost" is what not knowing it already cost, in cells, hours, or wrong published
numbers. It is the field that decides whether the enforcement was worth building.

---

## 2026-08-19 — model behaviour

### 2026-08-19 — Pinning norm-1000 removes image geometry as a variable
A model that never has to *name* the frame it works in cannot leak its internal
resize into the coordinates.

- **Evidence** — norm-1000 pin: **55 of 56** cells convert 6/6 across 14
  geometries × 2 models × 2 think modes. Same models, same geometries, `real`
  pin: qwen3.8 **14/14**, qwen3.6 **1/14**.
- **Enforced by** — SPEC C13–C18 §4, [ADR 0030](adr/0030-bbox-conformance-is-scoped-to-image-geometry.md).
- **Cost** — the contract shipped on a declaration-honesty argument (21/21 vs
  5/21) when a correctness argument was available and stronger.

### 2026-08-19 — The reported frame is not a function of input size
It snaps to canonical sizes and moves in **both** directions.

- **Evidence** — qwen3.8 returns `2560×1440` for **4 different inputs** and
  `2337×1754` for **2 more**. Frame/input ratio spans **0.67×–1.44×** (qwen3.8)
  and **0.46×–2.50×** (qwen3.6). 8 of 13 qwen3.6 cells and 6 of 13 qwen3.8 cells
  are below 1.0×.
- **Enforced by** — SPEC C17 (amended; its original text asserted the opposite),
  `bbox_contract_real_1img`.
- **Cost** — C17 shipped claiming every usable anchor reports a frame *larger*
  than the input. That was an artefact of only ever sending 1920×1080.

### 2026-08-19 — qwen3.6 think-on does not terminate; the anchor bounds it
Not an efficiency gap — a qualitative difference.

- **Evidence** — `multi_3img`: **122,880 tokens** (= `num_predict`), **313,054
  characters** of reasoning, **0 characters** of answer, byte-identical across 2
  reps. `multi_3img_anchored`: **10,910 tokens**, complete and correct.
  Reproduces on Apple Metal *and* CUDA Blackwell, two different builds.
- **Enforced by** — [campaign doc](vision-campaign-2026-08-19-qwen36-anchored.md),
  `CTX_LADDER` 131072 rung.
- **Cost** — the unanchored configuration has no req/hour rate at all. It was
  previously recorded as a `q4_bbox_hit` ❌, which reads as a grounding failure.

### 2026-08-19 — C7's silent-failure rate is a property of the image, not the validator
- **Evidence** — **4 silent failures in 14** qwen3.6 cells under a `real` pin
  (`self_check` passes, anchor converts 2–3 of 6), against the **1 in 107**
  measured at 16:9 HD. 9 of 14 correctly rejected.
- **Enforced by** — SPEC C14 (response-only conditions), C7 amendment.
- **Cost** — a single-number failure rate was quoted for a validator whose rate
  varies by two orders of magnitude with the input.

### 2026-08-19 — The anchor's value is per-model and inverts
- **Evidence** — under a `real` pin, qwen3.8's anchor converts **6/6 at all 14**
  geometries while best-fit reaches only 1/6 in five of them. qwen3.6's anchor
  converts 6/6 **once**; best-fit reaches 6/6 at **all fourteen**.
- **Enforced by** — ADR 0030 decision 5.
- **Cost** — "the anchor rescues the response" was a qwen3.8 finding stated as
  general.

### 2026-08-19 — `/api/generate` drops reasoning for gemma4 — and only gemma4
- **Evidence** — gemma4:31b-it-q4_K_M think-on: `/api/generate` returns **0**
  chars of reasoning, `/api/chat` returns **2021**, with `eval_count` **1281 vs
  1277** — generated either way, returned by one. nemotron3 returns **3909**
  chars on `generate`; qwen3.8 returns reasoning there normally.
  Token counts are **identical** across endpoints (1511/1511 think-off,
  1514/1514 think-on): ollama templates on `generate` too.
- **Enforced by** — `endpoint_compare.py`, suite default flipped to chat.
- **Cost** — I first wrote the rationale as "chat adds template tokens", which is
  false, and then as "`/api/generate` is behind", which over-generalised from one
  model. Both were reasoning, not measurement.

---

## 2026-08-19 — method

### 2026-08-19 — A capped cell is a floor, not a cost
`eval_count == num_predict` establishes "needs more than X", never "costs X".

- **Evidence** — at `num_ctx` 65536 the cell reported 57,344; at 131072 it
  reported 122,880. Doubling the ceiling changed the number and nothing else.
- **Enforced by** — [ADR 0012](adr/0012-benchmark-report-templates.md) rule 8;
  `CTX_LADDER` now ends at 131072.
- **Cost** — a diagnostic was one restatement away from being quoted as a cost.

### 2026-08-19 — Valid JSON is not usable JSON
- **Evidence** — the qwen3.6 q8_0 cell **parses cleanly** under `json.loads`;
  the model had serialised `answers` into a *string inside an unrelated array*.
  Scored 0/3 with chart 5/5. Recovered values (`q1: 2`, `q2: Q4/128`) match the
  same model's passing arm **exactly**. Rescoring 18 cells flips **exactly 3**;
  every other cell is byte-stable.
- **Enforced by** — `salvage.py`, SPEC C12 amendment, `salvage_method` per cell.
- **Cost** — a model failure was published that never happened, and the
  conclusion drawn from it ("the anchor makes q8_0 worse") was backwards.

### 2026-08-19 — 1920×1080 was never a clean control
- **Evidence** — `1080/32 = 33.75`, `1080/48 = 22.5`. Misaligned for **every**
  arch in the corpus; the fixture carried a 0.7%–2.2% pad in every measurement
  ever taken.
- **Enforced by** — SPEC C15 (alignment twins derived from `patch_stride`).

### 2026-08-19 — Contention corrupts measurements in both directions
- **Evidence** — loadavg **148** while a peer session ran timing-calibrated
  tests. Their measurements were invalidated; so were my `gen_tps`/`prefill_tps`
  fields, which I was about to publish as comparable.
- **Enforced by** — `endpoint_exclusive` in preflight; throughput omitted from
  the geometry campaign with the reason stated.

---

## 2026-08-19 — harness engineering

### 2026-08-19 — Duplication concentrates in the hardest part
- **Evidence** — **5** files had grown their own request code and had already
  diverged on 4 axes: endpoint default, `thinking` normalisation, response and
  reasoning persistence, context-overflow 400 translation. `finetext_probe` had
  **all four**, and its scores looked ordinary throughout.
- **Enforced by** — SPEC H9, `client.py`.

### 2026-08-19 — Consolidation must not normalise a calibrated payload
- **Evidence** — collapsing `send_think` into a boolean made
  `client.generate()` unable to emit `"think": true`, silently turning
  `variants.py`'s think-on arm into "whatever the server defaults to".
  `measure.py`'s payload had to stay byte-identical; its **19/19** text prefix is
  the proof it did.
- **Enforced by** — tri-state `send_think`, `num_ctx=False`, `use_env_opts`,
  `apply_sampling`; `test_client.py`.

### 2026-08-19 — Score-level tests give zero signal on the wire
- **Evidence** — **2 blockers** shipped green through **57** existing tests. One
  (`finetext_probe` `NameError`) fired *after* the inference completed, so the
  model was paid for and the result discarded; under `set -eu` it aborts the
  campaign.
- **Enforced by** — `test_client.py`, **25** tests asserting payload shape.
  Verified by reverting the fix and watching the test fail.
- **Cost** — an import-only check reported "imports OK" on a function that could
  not complete a single call.

### 2026-08-19 — Server-side knobs do not exist on hosts you do not own
- **Evidence** — `OLLAMA_MAX_LOADED_MODELS=1` is mandated by SPEC
  apple-silicon-build and is fixed when the *server* starts. 10.8.0.6 serves
  **5** ports. This host was previously driven to **106 GB** used and **53.89 GB**
  swap by a model left resident.
- **Enforced by** — [ADR 0031](adr/0031-model-residency-is-managed-client-side-on-remote-hosts.md),
  SPEC H10, `client.evict_others()` / `evict_all()`.
- **Also** — `keep_alive: 0` returns on *acceptance*, not completion, so the
  wait is load-bearing: without it a sweep holds two models at once.

### 2026-08-19 — Retry transport, never a deterministic rejection
- **Evidence** — a container restart killed rep 3 after **~23 minutes** of
  generation, when reps 1–2 were already byte-identical. A context-overflow 400
  fails identically forever; retrying it costs **50 s** and changes nothing.
- **Enforced by** — ADR 0031 decision 6; backoff 5/15/30 s; test asserts
  `urlopen` is called **exactly once** on a 400.

---

## 2026-08-19 — recurring failure patterns (mine)

Grouped because they share one cause: **acting on a mental model instead of
checking actual state.**

| pattern | occurrences | evidence |
|---|---|---|
| Watched a marker file instead of the real completion condition | **2** | two watchers left polling sentinels with no writer, 2h51m and 3h20m |
| Published a number before the run that produced it landed | **2** | "53 of 54" (true: 55 of 56); frame ranges quoted as the sub-1.0 subset |
| Stated a rationale I had not measured | **2** | "chat adds template tokens" (identical); "`/api/generate` is behind" (gemma4 only) |
| Generalised from one model | **2** | anchor rescues; endpoint drops reasoning |
| Edited a module while a sweep ran against it | **1** | `NameError` mid-import, 1 cell lost |
| Over-broad regex deletion | **1** | 4 prompt constants removed with the function being replaced |

**What actually caught these:** the adversarial verification pass, not the test
suite and not review-by-reading. It found 2 blockers and 3 wrong published
numbers that 57 tests and my own re-reading had missed.

**The cheapest guard found so far** is re-deriving a number from disk immediately
before writing it down. It caught the frame-range error in this very document.
