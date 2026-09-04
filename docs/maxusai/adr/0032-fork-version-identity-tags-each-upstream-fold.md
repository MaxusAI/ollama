# ADR 0032: the fork's version identity is a fork tag cut at each upstream fold

- **Status:** accepted 2026-08-27. Implemented as tag `v0.33.0-dynres` (at
  `51718870`), the widened `cuda-dynres-903` `version_pattern`
  ([PR #218](https://github.com/MaxusAI/ollama/pull/218)), and the H11
  equivalence note in [SPEC](../spec/vision-harness-reuse.md).
- **Date:** 2026-08-27

## Context

A served build stamps itself at build time from
`git describe --tags --first-parent` over the **fork's** tags
(`scripts/env.sh`), not from the upstream release it contains. After #217
folded upstream `v0.33.0`, the newest fork tag on main's first-parent line
was still `v0.32.14-dynres`, so the first `sync-0.33.0` image truthfully
reported `0.32.14-dynres-112-g5171887` — an identity naming a
five-releases-old lineage for a current-release codebase. The operator read
it as the wrong code having shipped. An identity that needs git archaeology
to interpret fails the one job H11 gives it.

Two constraints shape the fix. First, preflight profiles pattern-match the
version string **before** trusting any measurement, and the patterns are
deliberately narrow (`^0\.32\.`) — an unplanned identity change bricks the
gate. Second, expectations are keyed to the **payload** (`payload_pin`), not
the version family: #217 moved the Go tree to v0.33.0 while llama.cpp stayed
`b10488`, so the measured numbers did not move and must not be re-recorded
under a new profile.

## Decision

Cut a fork tag at every upstream fold, named `v<upstream-release>-dynres`,
on the merge commit that lands the fold (`v0.33.0-dynres` at `51718870`).
The release build for a fold is stamped from that tag
(`0.33.0-dynres-0-g…`); interim builds between folds describe from it.
Cutting the tag is part of the fold task, alongside the
`LLAMA_CPP_VERSION` review.

Version families sharing an unchanged payload share a preflight profile:
the pattern widens (`0.32` → `0.3[23]`, PR #218) and `payload_pin` remains
the provenance anchor, per ADR 0011. A new profile is cut only when the
payload moves, exactly as before.

Where a re-tag re-stamps an already-built source (as here), the two version
strings are **one build** for H11 comparability:
`0.32.14-dynres-112-g5171887` ≡ `0.33.0-dynres-0-g5171887` — both are main
@ `51718870`, payload `9d77fa172`. SPEC H13's MIXED rule treats
equivalences recorded in this ADR as a single build; anything not recorded
here stays two builds.

## Consequences

- `/api/version` on the deployed lineage answers with the upstream release
  the code actually contains, carrying the fork's own suffix.
- The version gate keeps rejecting genuinely unknown lineages: `0.34.x`
  fails until its fold cuts a tag and widens the pattern — deliberately a
  conscious step, not an accident to be discovered at preflight time.
- Run artifacts recorded 2026-08-26/27 under the `0.32.14-dynres-112` string
  (the first full preflight of `sync-0.33.0`, the first hours of vsuite on
  it) remain valid and equivalent to the re-stamped build.
- The habit of leaving version identity implicit after a fold is retired;
  #217's task doc had flagged the tag as optional, and this ADR closes that
  option.

## Amendment 2026-09-04 — point tags between folds

The deployed 0.33.2 build was `0.33.2-dynres-5-g2b95b4a` (the #238 merge),
five first-parent commits past `v0.33.2-dynres` — so the README's "the tag is
the fixed point to roll back to" named a commit that was not what ran. Rule:

- A deploy that is not the fold commit gets a **point tag** on the deployed
  commit, `v<release>-dynres.N` (precedent: `v0.32.1-dynres.2/.3`). Cut
  `v0.33.2-dynres.1` at `2b95b4a5`. The payload is unchanged, so the profile is
  unchanged; `payload_pin` stays the provenance anchor.
- `scripts/env.sh` stamps from `git describe --tags --first-parent`, so after a
  point tag every later build on `main` describes as
  `<release>-dynres.N-<n>-g<sha>`. The two lineage patterns
  (`cuda-dynres-903`, `mlx-cuda`) therefore admit `-dynres(\.\d+)?`; pinned
  profiles do not (test `TestLineageProfilesTrackOneVersionFamily`).
- Recorded H11/H13 equivalence: `0.33.2-dynres-5-g2b95b4a` ≡
  `0.33.2-dynres.1-0-g2b95b4a` — one build (main @ `2b95b4a5`, payload
  `d222767c7`, MLX `c793734e`).
- The README carries a **Deployed** line next to **Current fold** whenever the
  two differ; updating it is part of a deploy, as the fold pointer is part of
  a fold.

