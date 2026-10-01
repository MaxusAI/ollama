---
name: vision-campaign
description: Run a vision-suite campaign on one host and publish it as data. Covers the host profile, run_engine_compare.sh with a cold server per cell, reruns as knobs, checkpoint digests, the score bundle, generator tables placed by script, and the public-repo checks before the PR. Use when asked to benchmark or evaluate vision models on a host (a new GPU, a cloud VM, a new build), to "run the vision suite", to re-run one cell, or to write up and publish a campaign. NOT for validating an image before deploy (ollama-preflight), for diagnosing a think-on loop (kv-loop-check), or for external benchmarks such as OCRBench (extbench.py; docs/maxusai/ocrbench.md).
---

# Vision campaigns: run one host, publish it as data

This skill owns the **procedure**. The rules are SPEC `vision-harness-reuse` (H1–H26) and the report
templates of ADR 0012. The worked example is
`docs/maxusai/vision-campaign-2026-09-18-mlx8a7ba949-nvfp4.md` with its bundle
`docs/maxusai/vision-suite/bench-runs/vision-campaign-2026-09-18-mlx8a7ba949.json`. The 2026-09-30
H100 campaign (#431) follows this procedure end to end.

## Already settled: do not re-measure

- **There is one runner** (SPEC H1, ADR 0028). `vision-suite/run_engine_compare.sh` is the only loop
  over models, and an arm is a set of environment knobs, never a new script (H2).
- **Think-off is the campaign; think-on is a narrowing.** Run think-on only for the tags a decision
  needs. Keep the previous campaign's narrowing so the two stay comparable: 2026-09-18 and 2026-09-30
  both ran think-on for `gemma4:31b-nvfp4` and `qwen3.8:27b-nvfp4` only.
- **A tag is not a checkpoint** (H17, ADR 0038). The 2026-09-30 H100 campaign pulled the 2026-09-18
  campaign's five nvfp4 tags, and four of them came back as different artifacts. Two of those four
  kept their config digest. `gemma4:26b-nvfp4` has had three manifests since 2026-08-21. Cite manifest
  digests every time, and never assume two hosts hold the same weights under one tag.
- **The MLX row and the GGUF row of one model are different artifacts.** Comparing the two engines
  also compares two quantisation formats; say so in the write-up.
- **A difference between hosts in the same cell is not a finding by itself.** Hopper (sm_90) and
  sm_120 select different cuBLAS kernels, and MLX on CUDA is not MLX on Metal. The 2026-09-30
  campaign records the H100's numbers and leaves the comparison out of scope.

## Procedure

1. **Preflight the build first** with the `ollama-preflight` skill. A campaign on a build that fails
   preflight measures the failure.
2. **Serve one model at a time.** Set `OLLAMA_MAX_LOADED_MODELS=1` (AGENTS.md). On a GPU shared with
   another server, also set `OLLAMA_MLX_MEMORY_LIMIT` (see the `ollama-preflight` skill).
3. **Profile the machine on the serving host** (H26, ADR 0046):
   `python3 docs/maxusai/tools/host_profile.py --out DIR --label <collab label> --fs models=<models dir> --ollama http://127.0.0.1:11434`.
   Run it there, not on the client that drives the suite.
4. **Record the checkpoints** with `python3 docs/maxusai/vision-suite/store_audit.py --digests` on the
   serving host's store (H17).
5. **Run the campaign** from `docs/maxusai/vision-suite/`, think-off first, then the think-on
   narrowing as a second run:

   ```sh
   MODELS="gemma4:12b-nvfp4 gemma4:31b-nvfp4 qwen3.8:27b-nvfp4" THINK_MODES=false \
     RESTART_CMD='systemctl restart ollama' \
     ./run_engine_compare.sh http://127.0.0.1:11434
   ```

   - **`RESTART_CMD` gives every cell a cold server.** The runner then waits for `/api/version`
     itself. A systemd service needs the right to restart it; a container is restarted instead
     (README "Method").
   - **Let the context ladder run.** The rung it settles on is a result (AGENTS.md, H4a).
6. **Re-run a cell as an arm, never as a loop.**

   ```sh
   ONLY_TESTS=scene_single REPEATS=3 TAG_PREFIX=rerun_ MODELS=gemma4:12b-nvfp4 THINK_MODES=false \
     RESTART_CMD='systemctl restart ollama' ./run_engine_compare.sh http://127.0.0.1:11434
   ```

   This writes the tags `rerun_1_…` to `rerun_3_…`. Render each one with
   `summarize_engine_compare.py --think false --prefix rerun_1_ --expect scene_single gemma4:12b-nvfp4`.
7. **Bundle the score files without changing them** into
   `vision-suite/bench-runs/vision-campaign-<date>-<label>.json`.
   - **Its shape is `{meta, cells: [{tag, scores, finetext_probe}]}`.** Each cell is that tag's
     `scores_<tag>.json` and `ft_<tag>.json`, copied as they are.
   - **`meta` names the run:** the build, the MLX and llama.cpp pins, the host, `host_profile`, the
     power mode, the wall clock, the scope and `rendered_tables` (the doc).
   - **No bundling tool exists yet,** so build it with a script and never edit a cell.
   - **Commit the host profile beside the bundle** as `<bundle stem>.host-profile.json` (H26).
8. **Render the tables; never type them** (H7, ADR 0012 rule 8). Use
   `summarize_engine_compare.py --dir <dir> --think false <models…>`, then `--think on`.
   - **Put the output into the doc with a script:** write the prose around placeholders, and replace
     each placeholder with the renderer's stdout.
   - **Keep the provenance footer the renderer prints** (H13).
9. **Write the doc** as `docs/maxusai/vision-campaign-<date>-<label>.md`. It holds:
   - the provenance header (ADR 0012 rule 1): date, host, power mode, build and payload, endpoint and
     sampling;
   - the checkpoint table by manifest digest (H17);
   - the tables, what they show, and the limits;
   - a Reproducing section that re-renders from the bundle alone.
10. **Prove the doc re-renders from the bundle.** Unpack the bundle into a temporary directory and
    run the renderer on it. The doc must contain that output byte for byte.
11. **Check before pushing, with checks that stop the push.** A grep whose match scrolls past is not a
    check.
    - `python3 docs/maxusai/tools/check_no_names.py --denylist-file ~/.config/collab/denylist`
      (AGENTS.md). In CI it also checks the PR's title, body and commit messages.
    - `python3 docs/maxusai/tools/check_source_paths.py --changed-since origin/main`
    - For a cloud host, add a check that fails on its project, instance and host names:
      `if git diff origin/main | grep -nE '<project>|<instance>|<hostname>'; then exit 1; fi`.
      Keep that pattern out of the tree.
12. **Open one PR per result set,** with the doc, the bundle and the profile together.

## Traps

- **Don't trust MLX on CUDA's default kernel cache under a service user.** MLX caches compiled kernels
  in `/tmp/mlx/<version>/ptx`. If another user created that directory first, for example by running
  the MLX tests, the service gets permission denied there. If `/tmp` is a tmpfs, the cache is also
  lost at every reboot. Point `MLX_PTX_CACHE_DIR`, and `CUDA_CACHE_PATH` for the CUDA driver's
  cache, at durable disk that the service owns.
- **Don't build rerun tags with a shell loop.** On 2026-09-30 a loop over `${n}` wrote tags that the
  summarizer could not find. `REPEATS` and `TAG_PREFIX` write the tags the summarizers read (H2, H6).
- **Don't build a tag by hand.** Tags turn `:` and `.` into `_`, so `qwen3.8:27b-nvfp4` becomes
  `qwen3_8_27b-nvfp4`. Use `tag_for` (H6).
- **Don't quote a rate from a capped cell.** A rate is real only when the cell terminated; a capped
  cell renders as `capped` (AGENTS.md).
- **Don't keep run output in `/tmp`.** It does not survive a reboot, and on many cloud images it is a
  tmpfs. Keep score files, logs and the bundle on durable disk until the PR has merged.
