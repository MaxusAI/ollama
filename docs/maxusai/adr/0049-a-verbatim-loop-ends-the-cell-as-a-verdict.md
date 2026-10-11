# ADR 0049: A verbatim loop ends the cell as a verdict

Date: 2026-10-11 · Status: proposed. Opt-in (`LOOP_DETECT=1`); first used by the muse-glimmer campaign's think-on
and OCRBench arms (#463).

## Context

Under greedy decoding (the suite's think-off arms, and think-on at `THINK_TEMPERATURE=0`), a model that starts
repeating itself verbatim does not recover. The same context produces the same continuation, so the cell spends the
rest of `num_predict` on the loop. Under think-on the context ladder then climbs from 16384 to 131072 and repeats
the loop at every rung, with `num_predict = num_ctx - 8192`.

The muse-glimmer campaign's think-off pass (2026-10-11) showed every form of it:
- **MLX `30b-mxfp8`, `bboxm_pin_noanc_named`:** one box object's keys, repeated to 4350 chars.
- **MLX `30b-nvfp4`, `multi_3img`:** a 72-char fragment, repeated until it hit the length cap at every rung up to
  131072.
- **GGUF `30b`, two cases:** a run of one token. ollama's own guard aborts at 100 repeats with an HTTP 500 and no
  output (`llm/llama_server.go:2483`).

"Tokens to finish is the result; never cap to save tokens" stands. A loop is the model failing to finish, not a model
that would have finished given more budget, so ending it early loses nothing and labels it correctly.

## Decision

1. **`loop_detect.py` judges a stream as it arrives.** It is stdlib only, and the same file is shared with
   ai-invoice-to-xero's `ollama_toolkit` and the structured-docs backend. It reports one of three failure modes:
   - `loop detected:thinking`: the last 400 chars of the thinking trace occur 3 times in it.
   - `loop detected:content`: the same test, on the answer.
   - `loop detected:token-run`: one stream emitted the same whitespace-stripped chunk 64 times in a row. That is
     below the server's 100, so the client ends it first and keeps the output.

   Three copies, not two: one earlier copy is how a model restates a table. The check runs every 64 new chars over
   the last 16384 chars, so its cost is linear in the stream.
2. **`LOOP_DETECT=1` makes `client.generate` stream.** On a loop it closes the connection, which cancels the
   generation server-side, and returns the partial output as a **finished** response:
   - `done_reason` is `"loop"`, and `failure_mode` is `"loop detected:<kind>"`.
   - The loop's fields: kind, stream, period, repeats, chars at detection, snippet, parameters.
   - `eval_count` is the number of streamed chunks, and the durations are None (the server sent no final metrics).
3. **A loop is a verdict, not an error and not a cap.**
   - The suite scores the partial output as it scores a capped answer.
   - `arm_done` treats the block as finished, so it is not re-run on resume.
   - `capped_arms` does not climb for it.
4. **Every streamed block says so.** `client.loop_fields` stamps `stream: true` and the `loop_detect` parameters,
   plus the loop's fields when it fires, on vision_suite and finetext_probe blocks alike.
5. **Off by default.** A campaign that turns it on mixes nothing: every block carries the stamp.

## Validation

- **Replay:** every saved response in the glimmer and v0.40.2 campaign trees (1817 files) was fed to the detector.
  - It fired on 4. All four were cells that ran to the length cap: the two `multi_3img` and the two
    `bboxm_pin_noanc_named` above.
  - It fired at 800-1312 chars, against the 2182-4350 the cells generated.
  - It fired on no cell that finished.
- **Coverage gap:** saved responses hold answers only, so the thinking detector's real-data check is the think-on arm
  itself.
- **Tests:** `test_loop_detect.py` (16 tests, including those two real outputs as fixtures) and `test_client.py`
  (`TestLoopDetect`, `TestLoopFields`).

## Consequences

- **Think-on cells that loop end at the rung where they loop,** as `failure_mode: loop detected:thinking`. Before,
  they climbed to the ceiling and ended NOT CONVERGED. A NOT CONVERGED cell under detection is therefore a
  non-looping one that genuinely needs more.
- **A looped cell has no throughput.** Its tok/s and s/req are underivable and are excluded like a capped cell's.
- **Streaming changes no generation.** It is the same request with `stream: true`. The drafted MLX tags' text already
  depends on wall-clock draft depth (`mlx-greedy-has-baseline-spread`), and this neither fixes nor worsens that.
