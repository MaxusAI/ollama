# ADR 0049: A verbatim loop ends the cell as a verdict

Date: 2026-10-11 · Status: proposed. Opt-in (`LOOP_DETECT=1`); first used by the muse-glimmer campaign's think-on
and OCRBench arms (#463).

## Context

Under greedy decoding (the suite's think-off arms, and think-on at `THINK_TEMPERATURE=0`), a model that starts
repeating a block verbatim, back to back, usually keeps repeating it. The same context tends to produce the same
continuation, so the cell spends the rest of `num_predict` on the loop. Under think-on the context ladder then climbs from 16384 to 131072 and repeats
the loop at every rung, with `num_predict = num_ctx - 8192`.

The muse-glimmer campaign's think-off pass (2026-10-11) showed every form of it:
- **MLX `30b-mxfp8`, `bboxm_pin_noanc_named`:** one box object's keys, repeated to 4350 chars.
- **MLX `30b-nvfp4`, `multi_3img`:** a 72-char fragment, repeated until it hit the length cap at every rung up to
  131072.
- **GGUF `30b`, two cases:** a run of one token. ollama's own guard aborts at 100 repeats with an HTTP 500 and no
  output (`llm/llama_server.go:2483`).

"Tokens to finish is the result; never cap to save tokens" stands, and this decision is a stated exception to it.
What a loop verdict guarantees is narrow: the stream's last 400 chars, plus two periods, are one block repeated three
times back to back. It does not guarantee the model would never have finished. The CUDA host's replay (#469 review)
found such loops on 57 length-capped cells, and on 3 of 1,505 think-on traces that went on to finish. Ending a cell
at the loop gives up those rare recoveries: the cell is labelled, not measured to its end. In exchange, a loop that
runs to the cap (57 of the 60 on that replay) costs one verdict at the rung where it starts, instead of the rest of
`num_predict` at every rung of the ladder.

## Decision

1. **`loop_detect.py` judges a stream as it arrives.** It is stdlib only, and the same file is shared with
   ai-invoice-to-xero's `ollama_toolkit` and the structured-docs backend. It reports one of three failure modes:
   - `loop detected:thinking`: the last 400 chars of the thinking trace occur 3 times in it, back to back. The last
     400 chars plus two periods must be one block repeated.
   - `loop detected:content`: the same test, on the answer.
   - `loop detected:token-run`: one stream emitted the same whitespace-stripped chunk 64 times in a row. That is
     below the server's 100, so the client ends it first and keeps the output.

   Three copies, not two: one earlier copy is how a model restates a table. Back to back, not anywhere: three
   copies with other text between them are how a thinking trace restates its draft after each check. The check runs every 64 new chars over
   the last 16384 chars, so its cost is linear in the stream.
2. **`LOOP_DETECT=1` makes `client.generate` stream.** On a loop it closes the connection, which cancels the
   generation server-side, and returns the partial output as a **finished** response:
   - `done_reason` is `"loop"`, and `failure_mode` is `"loop detected:<kind>"`.
   - The loop's fields: kind, stream, period, repeats, chars at detection, snippet, parameters.
   - `eval_count` is the number of streamed chunks, and the durations are None (the server sent no final metrics).
3. **A loop the server held back is judged at the end.** A server can send a reply in its final chunk: glimmer's
   parser holds back a reply that never writes its message header (the #464 path), so on the release build its
   format + think-off loops stream nothing until the generation ends. The final chunk is fed to the detector too.
   A loop in a reply the server stopped at `num_predict` becomes the same verdict, with the server's own metrics
   kept and `server_done_reason: "length"` recorded. A reply that finished is never a loop, however repetitive.
   Such a loop is labelled, not cut early: that needs the server to stream.
4. **A loop is a verdict, not an error and not a cap.**
   - The suite scores the partial output as it scores a capped answer.
   - `was_capped` is False for `done_reason: "loop"` at any `eval_count`. A held-back loop stopped at
     `num_predict` keeps `eval_count == num_predict`, and the arithmetic fallback would otherwise call it capped.
   - `arm_done` treats the block as finished, so it is not re-run on resume.
   - `capped_arms` does not climb for it.
5. **Every streamed block says so.** `client.loop_fields` stamps `stream: true` and the `loop_detect` parameters,
   plus the loop's fields when it fires, on vision_suite and finetext_probe blocks alike.
6. **Off by default.** A campaign that turns it on mixes nothing: every block carries the stamp.

## Validation

- **Replay:** every saved response in the glimmer and v0.40.2 campaign trees (1817 files) was fed to the detector.
  - It fired on 4. All four were cells that ran to the length cap: the two `multi_3img` and the two
    `bboxm_pin_noanc_named` above.
  - It fired at 800-1312 chars, against the 2182-4350 the cells generated.
  - It fired on no cell that finished.
- **Thinking traces, replayed on the CUDA host (#469 review):** 1,641 traces (`client.save` writes them).
  - Counting copies alone, the detector fired on 17-18 of the 1,505 think-on traces that finished
    (`done_reason: "stop"`). 13 of them were `multi_3img` cells with q1, q2 and q4 all right: a draft restated
    between checks, not a loop.
  - Requiring the copies back to back left 3 fires on finished traces, all contiguous greedy loops that then
    recovered. It kept all 57 fires on length-capped cells.
- **Thinking traces, replayed on the Metal host:** 2,176 traces saved in this checkout's worktrees.
  - Finished (`stop`): 11 fires by copy count, 3 back to back. Length-capped (`length`): 57 and 57.
  - Two verdicts a live think-on run had already recorded under the copy count were restated drafts:
    `gemma4:31b-nvfp4` `multi_3img_anchored` (earlier copies 4118 and 7580 chars back) and `bboxm_pin_anc_pos`
    (963 and 3307 chars back). The back-to-back rule calls neither.
- **Tests:** `test_loop_detect.py` (18 tests, including those two real outputs as fixtures, a restated draft that is
  not a loop, and a loop that starts after one) and `test_client.py` (`TestLoopDetect`, `TestLoopFields`).

## Consequences

- **Think-on cells that loop end at the rung where they loop,** as `failure_mode: loop detected:thinking`. Before,
  they climbed to the ceiling and ended NOT CONVERGED. A NOT CONVERGED cell under detection is therefore a
  non-looping one that genuinely needs more.
- **A looped cell has no decode rate.** `metrics_block` sets `gen_tps` to None for every loop, including a held-back
  one whose server metrics would allow it: that rate is a loop's, not a finishing model's. Prefill ended before the
  loop, so `prefill_tps` is kept when the server reported it.
- **Streaming changes no generation.** It is the same request with `stream: true`. The drafted MLX tags' text already
  depends on wall-clock draft depth (`mlx-greedy-has-baseline-spread`), and this neither fixes nor worsens that.
