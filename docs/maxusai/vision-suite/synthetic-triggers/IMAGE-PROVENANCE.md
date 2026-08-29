# Provenance and rights of the images in this directory

## Generated — no third-party rights, reproducible from source

These are produced by the scripts here from a few lines of numpy. They contain
no photographic content and carry no third-party rights. Regenerate them with
`gen_typhoon_trigger.py` / `gen.py`; the shipped files are byte-identical to
the artifacts that were engine-tested.

- `trigger_typhoon_c70_dx37_dy35_1350x1800.png`  (preferred typhoon trigger)
- `trigger_typhoon_c70_halfphase_1350x1800.png`
- `control_typhoon_c70_phase0_1350x1800.png`     (healthy paired control)
- `trigger_checker56_1350x1800.png`
- `trigger_stripes56_1350x1800.png`

**Prefer these when sharing externally** — they need no attribution and no
rights discussion.

## Photographic — NASA, public domain

- `base_nasa_20040421_exp9_02.jpg` — "Russian Flight Control Room", NASA
  `nasa_id=20040421_exp9_02`, 2004-04-20, NASA HQ.
  **Photo credit: NASA/Bill Ingalls.**
- `trigger_071_nasa_contrast15.png` — the above at contrast x1.5, fitted under
  0.7.1's pixel cap by `gen_071_trigger.py`.

Taken by a NASA staff photographer in the course of his duties, so it is a work
of the US federal government and not subject to copyright in the US. The NASA
images API returns no rights restriction for this asset. NASA asks that its
imagery not be used to imply endorsement; using it as a numerical-fault test
input does not.

The base is kept in-tree deliberately rather than fetched at run time: the
fault depends on exact pixel values, and a re-encoded download would not be
guaranteed to reproduce it.

## Never in this repository

No customer or client imagery, and nothing derived from it — including a
model's *description* of such an image. See the redaction note in `README.md`.
