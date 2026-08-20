# TASK: nemotron think-on + format json underreports eval_count

**Opened:** 2026-08-20. **Status:** open — server-side accounting bug,
live-reproduced on `0.32.14-rc0-dynres-0-ga5d6590` (CUDA host). Until fixed,
nemotron3 think-on cells cannot carry a `Think tok` split, and their token
counts (`Gen tok`, gen tok/s) are understated; `s/req` is unaffected (it
reduces to wall-clock).

## Evidence

`token_split.py --server` counts persisted text with the server's OWN
tokenizer (`prompt_eval_count` of a raw-mode request — no vocab to mismatch),
and its acceptance gate refused nemotron at **54/54 cells, residues −25 to
−15102**: the thinking + answer text tokenizes to MORE tokens than the
`eval_count` that supposedly produced it, with the deficit scaling with
thinking size. An independent BPE rebuilt from the GGUF's own token/merge
arrays agreed (−14092 min). Every other campaign model reconciles exactly
(control residues in [0, 29] over 53–54 cells), so the harness pipeline is
not the cause.

Live single-request reproduction (2026-08-20, chat endpoint, `think: true`,
`format: json` — the exact campaign path):

    eval_count 353 | think chars 570 | answer chars 18 | done stop
    server-counted: think tok 362 + answer tok 10 = 372 → control −19

The server reports fewer generated tokens than the visible text it returned.
The persisted text is not duplicated (checked: 41,696-char multi thinking is
unique text; 2.4 chars per counted token where this vocab runs ~4 for prose).

## Where to look

`nemotron_h_omni` has no native think tags, so it takes the marker-deferred
thinking path in `server/routes.go` (`deferring && deferViaMarker`, ~782 and
~3189), plus the format-constrained continuation machinery. The two-pass
metrics fix already sums `pass1.EvalCount` (`res.Metrics.EvalCount +=
pass1.EvalCount`, routes.go:868 generate / :3178 chat — the R6 / ADR 0010
fix), so some branch of the marker/continuation path is generating visible
text whose tokens land in neither pass's count. The deficit being a small
constant-ish −19…−25 on think-light cells and ~−15k on think-heavy ones
suggests the uncounted portion IS (most of) the deferred thinking.

## Consequences while open

- T1 `Think tok` renders `—` for nemotron3 think-on — correctly: writing a
  split would assert `eval = think + answer + control`, which the server
  itself currently violates.
- Published nemotron think-on `Gen tok` (e.g. 5308 for the scene cell) and
  gen tok/s (230) are **understated** by the uncounted thinking tokens.
  `s/req` stays valid: `eval/gen_tps` cancels to measured wall-clock.
- The CONTEXT-ladder capped-check compares an undercounted `eval_count`
  against `num_predict`; a nemotron think-on cell can therefore generate past
  the nominal budget before the harness sees a cap. `done_reason` (PR #207)
  is the reliable cap signal for this model.

## Acceptance

- Root cause identified in the marker-deferral/continuation path; fix makes
  `eval_count` cover every generated token including deferred thinking.
- `token_split.py --server` gate passes for nemotron3 think-on tags
  (control residues in [0, 64]) — that IS the regression test.
- Re-render T1 for cudafull1: nemotron `Think tok` fills; `Gen tok` and
  gen tok/s restated with a note that prior values were undercounted.
