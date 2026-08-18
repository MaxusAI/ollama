# Grammar-aware speculation is correct and inert: the depth controller cannot leave 0

MaxusAI-fork reference. Measured 2026-08-18 on CUDA (RTX PRO 6000 Blackwell) from
the merge of [#191](https://github.com/MaxusAI/ollama/pull/191). Companion to
[`format:"json"` disables speculative decoding](mlx-constrained-decode-disables-speculation.md),
which is the note that motivated the feature and whose penalty this was built to close.

> **The one thing to take away:** with `OLLAMA_MLX_GRAMMAR_SPECULATION=1` the runner
> emits correct constrained output and is **not faster**, because it never proposes a
> single draft. The depth controller starts at 0, a parked round teaches it nothing,
> so it stays at 0 for the whole request. The feature cannot start.

## The measurement

`gemma4:31b-nvfp4`, `visimgs/scene_hd.png` (1920×1080), same prompt in both arms,
`think:false`, `num_ctx:8192`, warmed, n=3. Gate off vs gate on, same binary.

| binary | `format:"json"` | unconstrained |
|---|---|---|
| published baseline (`eb0ad43`) | 21.13 | 36.53 |
| `main` @ `640a349c` (n=1) | 22.98 | — |
| stage 3 + fixes, **gate OFF** | **23.63** (23.81 / 23.09 / 23.98) | **37.47** (36.92 / 38.22 / 37.27) |
| stage 3 + fixes, **gate ON** | **22.25** (21.96 / 22.48 / 22.32) | 34.00 (34.64 / 33.08 / 34.29) |

Gate off reproduces the published −42% as −37% here, which is what validates the
harness. Gate on is not an improvement on it — it is fractionally worse.

## What *is* fixed

Stage 3 shipped output that ignored the grammar entirely (`eval_count` matched the
no-format arm exactly). That is closed. Every constrained response on every arm in
the table came back as **bare, parseable JSON** — no markdown fence, `parses=True`.
The correctness claim rests on device behaviour, not only on the host tests.

## Why it does not speculate

The stats, from the same requests as the table:

```
iterations=126 drafted=0 accepted=0 acceptance=0.00 avg_draft=0.00 max_draft=0 \
  grammar_truncated=0 grammar_no_legal_draft=0 depth_over_time="0.0/0 0.0/0 ... 0.0/0"
iterations=121 drafted=1 accepted=0 acceptance=0.00 avg_draft=0.01 max_draft=1 ...
```

`drafted=0` on nearly every round — and **not** because the grammar rejected the
proposals. `grammar_no_legal_draft=0` and `grammar_truncated=0`, so the
`errNoLegalDraft` fallback never fired either. Nothing was ever proposed.

The loop that produces this, all in `x/mlxrunner/speculate.go`:

- a round only proposes when `s.limit > 0` (`speculate.go:232`);
- `s.limit` is refreshed **only** in `endRound`, from `depth.next()` (`speculate.go:170`);
- the acceptance model only learns when `observed > 0` (`speculate.go:167`), and a
  **parked round reports `observed=0`**.

Depth starts at zero, so the first round of every request parks. Parking produces no
observation, so the controller learns nothing, so depth stays zero, so the next round
parks. It is a cold-start deadlock, not a slow ramp. The one round that did draft
(`max_draft=1`) was a cadence probe whose single token was rejected — evidence
*against* raising depth, so it settles straight back to 0.

The same binary unconstrained climbs normally, which is what rules out a broken
controller in general: `depth_over_time="3.6/5 5.1/6 5.0/5 ... 5.9/7"`, acceptance
0.72–0.93.

## The remaining work

Some way for a grammar-constrained request to reach depth ≥ 1 and hold it long enough
to gather acceptance data. Three candidates, none obviously correct, and the choice
wants measuring rather than arguing:

- enter a constrained request at depth ≥ 1 instead of at `scheduled`, which is 0 on a
  cold controller;
- let a parked round contribute an observation rather than `observed=0`;
- give the probe cadence a floor under a grammar, so one rejected probe cannot pin
  depth at 0 for the rest of the request.

Until one of these lands, the penalty in
[the companion note](mlx-constrained-decode-disables-speculation.md) is untouched and
the gate should stay off. It ships off; nothing is enabled by default.

## Provenance and limits

- n=3 per arm, one model, one image, one prompt. Enough to act on; not a campaign.
- Measured with a **Go-only binary swap** over `maxusai/ollama:0.32.14-rc0-dynres-mlxfix`.
  The branch changes only `x/mlxrunner/*.go`; `git diff eb0ad43...HEAD` over
  `x/mlxrunner/mlx`, `MLX_VERSION`, `MLX_C_VERSION`, `CMakeLists.txt`, `cmake/`,
  `llama/` and `ml/` is empty apart from two Go files and a README, so the native
  payload is identical and the swap is sound. Verified by `sha256sum` inside the image,
  and by `main` reproducing the published baseline through the same path.
- The gate-on unconstrained column reads 34.00 against 37.47 off. The gate should not
  touch that path at all. This host is shared and its load moved a great deal across
  the session, so that is **not** claimed as a regression on n=3 — it wants both arms
  re-run back to back on a quiet box before anyone acts on it.

## Two traps that cost hours, recorded so they do not again

- **Pin `num_ctx`.** Unset, ollama derives the default from total VRAM, and this host's
  105.5 GiB yields `default_num_ctx=262144`, which collapses decode to **1.48 tok/s**
  (versus 37.97 at 8192 — a 25× swing with nothing else changed). A benchmark run at
  the default looks like a catastrophic regression in whatever it is testing.
- **Set `think:false`.** `gemma4`'s renderer advertises thinking, and `/api/generate`
  with `format` and thinking enabled is the documented case in
  [generate + think + format returned an empty response](generate-think-format-empty-response.md).
  Left unset it returned a markdown ```` ```json ```` fence rather than constrained
  output, which reads exactly like the grammar being ignored — and is not.
