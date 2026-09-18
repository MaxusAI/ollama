# Open regression: gemma4:31b-nvfp4 think-on loses a 9px OCR tier on 0.34.0

**Status: reproducible, cause unidentified.** Four hypotheses are falsified
below; a fifth and better lead — greedy nondeterminism — is recorded at the end
and is where the next person should start.

Measured 2026-09-18 on 10.8.0.3, powermode 2, benchmark server on `:11436` from
scratch installs (never the deployed binary — `serve-apple-mlx.sh` restarts with
`pkill -f "$BIN serve"` and production's cmdline is `/opt/github/MaxusAI/ollama/ollama serve`).

## The finding

`gemma4:31b-nvfp4`, think **on**, `finetext` arm, `num_ctx=16384`
`num_predict=8192`. The 9px exact-match recall tier drops from 4 to 3 between
`0.33.2-maxusai-2b95b4a5` and `0.34.0-maxusai-8a7ba949`.

Twelve runs, three per build per grammar condition:

```
0.34.0-maxusai-8a7ba949  constrained    9px = [3, 3, 3]
0.34.0-maxusai-8a7ba949  unconstrained  9px = [3, 3, 3]
0.33.2-maxusai-2b95b4a5  constrained    9px = [4, 4, 4]
0.33.2-maxusai-2b95b4a5  unconstrained  9px = [4, 4, 4]
```

Six more under forced greedy decoding (`temperature 0`, `top_k 1`):

```
0.34.0-maxusai-8a7ba949  GREEDY think-on  9px = [3, 3, 3]
0.33.2-maxusai-2b95b4a5  GREEDY think-on  9px = [4, 4, 4]
```

18 runs, no overlap. This is not sampling luck.

## What is NOT affected

**Think-off.** The full campaign scored all five nvfp4 models think-off on both
builds: every OCR tier, every box/label/colour count, every invoice field and
every multi-image verdict identical, answer-token counts identical. Only scene
IoU moved, in the third decimal, and upward on four of five. The 31b think-off
row held `4,4,4,4,3` on both builds — the same arm that moves under think-on.

Think-off also runs under `format:"json"` (the default in `client.generate`), so
constrained sampling is exercised there too and found inert.

## Four falsified hypotheses

Recorded because they are the four explanations anyone reaches for first, and
each cost a run to kill.

**1. #301's whitespace bound.** `max_whitespace_cnt` is present on 0.34.0 and
absent on 0.33.2, and acts only on constrained output — so it looked like the
delta. Falsified: removing the grammar entirely changes nothing on either build
(rows 2 and 4 above). #301 is exonerated.

**2. Reasoning length.** 0.34.0 terminated its reasoning earlier on the first
samples seen (1821 vs 2046 thinking chars), which suggested truncated reasoning
costing fine-tier recall. Falsified by the full set: 0.33.2 produced the
shortest run of all at **1545** thinking chars and still scored 4, while 0.34.0
thought **2423** and still scored 3. Thinking length does not predict the
outcome across a 1.6x range.

**3. The resize kernel.** A `b9888..b10864` diff of the gemma4 branch shows
`image_resize_algo` moving to `RESIZE_ALGO_BICUBIC` alongside the token-limit
change, which reads as a simultaneous b10864 change. It is not: b10630 — the
payload behind 0.33.2 — already sets BICUBIC, so the kernel is identical on both
sides. The MLX path has always used CatmullRom (Keys a=-0.5, the bicubic family)
regardless. See `llama/compat/README.md`.

**4. The sampling path.** Think-off is greedy and unaffected; think-on is
non-greedy and moved, so sampling was the sharpest remaining asymmetry.
Falsified: the gap persists unchanged under forced greedy.

## Corrected: greedy spread exists on BOTH builds; drafting amplifies it

An earlier revision of this document claimed 0.34.0 is nondeterministic under
greedy while 0.33.2 is deterministic, and offered that as the better bisect
target. **That claim was a three-sample artefact and is withdrawn.** It is kept
here rather than deleted, because it is the same small-sample error this
repository has made before and the correction is the useful part.

Pooling every greedy run (`temperature 0`, `top_k 1`, think-on, same prompt):

| build / config | runs | identical |
|---|---|---|
| `0.33.2-maxusai-2b95b4a5` | 6 | 5x `eval=1588 think=2046`, 1x `1584/2038` |
| `0.34.0-maxusai-8a7ba949`, drafting **off** | 3 | 2x `1598/2423`, 1x `1591/2046` |
| `0.34.0-maxusai-8a7ba949`, drafting **on** | 3 | 0 — three distinct |

What this supports:

- **MLX greedy decoding has baseline run-to-run spread on both builds.** The
  fork's own comment at `x/mlxrunner/speculate.go` already said so — "within the
  run-to-run spread MLX already has".
- **Drafting under a grammar amplifies it substantially**: zero identical runs
  out of three with it on, two of three with it off. `draftUnderGrammar` does
  not exist on 0.33.2 at all; it arrived with `068a98cd0`, which adopted
  upstream's default and put it behind `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR`. That
  knob is read once at process init, so it is a server-start setting, not a
  per-request one.

What this does NOT support: that 0.34.0 introduced nondeterminism. The first
three 0.33.2 runs came up identical and a headline was built on them; a later
run in the identical configuration produced an outlier.

**The tier drop is unaffected by any of this.** 9px is 3 on 0.34.0 and 4 on
0.33.2 in twelve runs each — constrained, unconstrained, greedy, and greedy with
drafting disabled. 24 runs, perfect separation, no configuration moves it. That
is the finding to bisect; the determinism question is a separate and much softer
observation.

## Next step not taken

Splitting the fold at `907deffd` (v0.34.0, MLX `ce916dbb`) against `8a7ba949`
(v0.34.1, MLX `d9add9d1`) halves the search. It needs a rebuild: only the two
endpoints are archived, and the `907deffd` install was discarded. Archiving it
would have been the right call, since it was a distinct Go/MLX pairing.

## Reproducing

Both pairings are archived, binary AND payload, so neither needs a rebuild:

```
~/.ollama/binaries/ollama-0.33.2-maxusai-2b95b4a5  + payload-0.33.2-maxusai-2b95b4a5/
~/.ollama/binaries/ollama-0.34.0-maxusai-8a7ba949  + payload-0.34.0-maxusai-8a7ba949/
```

Assemble each as a self-contained install (`<dir>/ollama` plus `<dir>/lib/ollama`);
`libOllamaRoots()` checks `<exeDir>/lib/ollama` first on darwin, so it resolves
its own payload and never consults the fork checkout.

**Assert the served version before measuring.** `serve-apple-mlx.sh` kills only
its own `$BIN`, so switching builds leaves the previous one holding `:11436`;
the new server fails to bind, the readiness curl is answered by the process that
was supposed to be replaced, and the run measures the wrong build under the right
label. Free the port by PORT, then check `/api/version` matches before running.
