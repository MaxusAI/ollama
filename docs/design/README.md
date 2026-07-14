# Design notes

- [Gemma 4 vision token budgets](gemma4-vision-token-budgets.md) — `image_min_tokens` / `image_max_tokens`, ollamarunner, scheduler reload, and related behavior (preserved from the implementation plan).
- [Gemma 4 vision token budgets — upstream rebase & forward-port notes](gemma4-vision-token-budgets-upstream-rebase.md) — why the feature was rebased onto `f63eea3d` (the last upstream commit with the Go runner still wired), verification results, and what a forward-port to current `main` (llama-server / `mtmd`) would require.
