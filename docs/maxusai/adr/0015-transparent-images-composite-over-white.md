# ADR 0015: Transparent images composite over white, matching the mlx-vlm reference

- **Status:** accepted, implemented 2026-08-10 on fork `main` lineage. Leaves
  an open cross-engine question — see Consequences.
- **Date:** 2026-08-10
- **Deciders:** MaxusAI fork maintainers

## Context

`Model.NewVisionInput` (`x/models/gemma4/vision.go`) scaled the decoded image
into an **alpha-premultiplied** `*image.RGBA` (`xdraw.CatmullRom.Scale` with
`xdraw.Src`) and then read the R/G/B bytes straight out of `dst.Pix` without
ever dividing by alpha. Every pixel with `A < 255` therefore reached the
vision tower scaled by `A/255` — composited over **black** — and fully
transparent regions became pure black regardless of their stored RGB.

Nothing intended this; it is a consequence of Go's premultiplied colour
model. No reference pipeline behaves this way: HF/mlx-vlm's `convert_to_rgb`
composites over **white**, while plain PIL `convert("RGB")` and the GGUF
runner's `stb_image` 4→3 conversion simply **drop** alpha and keep the stored
RGB. The user-visible symptom is a logo or chart exported with a transparent
background being described as sitting on a black background — and the same
image answering differently on the MLX and GGUF paths.

## Decision

Flatten non-opaque images onto a white canvas *before* the resize, matching
the reference's `convert_to_rgb` → `resize` order.

The reference is this file's own declared one: the header states it is
"ported from mlx_vlm … models/gemma4", and
`docs/maxusai/mlx-vision-ecosystem-and-sizing-parity.md` names mlx-vlm's
`Gemma4ImageProcessor` as the upstream reference. That processor's
`do_convert_rgb` step is transformers' `convert_to_rgb`, which alpha-composites
onto white. Compositing over white is also the existing in-repo precedent
(`x/imagegen/image.go` `flattenAlpha`).

Opaque images — nearly all of them, and every existing golden and e2e
fixture — skip the branch entirely and reach the scaler byte-for-byte
unchanged. Enforced by `TestNewVisionInputCompositesAlphaOverWhite` over both
lineages; the pre-existing exact-value layout tests still pass unmodified.

## Options considered

- **Composite over white** (chosen) — matches the declared reference, so the
  MLX path reproduces the numbers mlx-vlm would produce for the same image,
  which is the property the golden-vector harness exists to check.
- **Drop alpha, keeping stored RGB** — what the GGUF runner does via
  `stb_image`, and what plain PIL `convert("RGB")` does. It would give
  byte-parity with the fork's *other* engine, but diverge from the reference
  this file is ported from and therefore from the goldens. A real contender;
  see the open question below.
- **Status quo (composite over black)** — matches no reference, and is the
  behaviour being fixed.

## Consequences

- Transparent images are no longer silently darkened, and fully transparent
  regions no longer read as black.
- **Open question, deliberately not decided here:** MLX and GGUF still differ
  for images carrying alpha — MLX now composites over white, GGUF drops
  alpha. They agree in effect only when the RGB stored under the
  transparency is white-ish. Making them agree is a product decision that
  should be taken once, engine-wide; whichever way it goes, the change is
  a one-line swap in this function plus the matching expected values in
  `TestNewVisionInputCompositesAlphaOverWhite`.
- Cost: one full-size RGBA buffer, allocated only for images that actually
  carry transparency.
