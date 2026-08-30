#!/usr/bin/env python3
"""The checks. Every expected value comes from expectations.toml; nothing is
hardcoded here except the shapes of the assertions themselves."""
import json
import os
import re
import subprocess
import sys
import time

from probes import (ProbeError, container_logs, grep_binary_marker,
                    ladder_image_b64, llama_cpp_build, mlx_build,
                    mlx_describe_commit, parse_pixel_lines, poison_image_b64)

PASS, FAIL, SKIP, NEEDS_BASELINE, ERROR, CONTENTION = (
    "PASS", "FAIL", "SKIP", "NEEDS_BASELINE", "ERROR", "CONTENTION")

SUITE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def result(name, status, summary, arch=None, expected=None, actual=None,
           diagnosis=None, **extra):
    r = {"check": name, "arch": arch, "status": status, "summary": summary}
    if expected is not None:
        r["expected"] = expected
    if actual is not None:
        r["actual"] = actual
    if diagnosis:
        r["diagnosis"] = diagnosis
    r.update(extra)
    return r


# --------------------------------------------------------------------------
# 1. Version string — the gate. Nothing downstream is trustworthy without it.
# --------------------------------------------------------------------------

def check_version(client, profile, profile_id):
    """11434/11435/11436 are all occupied on 10.8.0.6; a canary on 11436 once
    answered from the wrong server and only a mismatched version string caught
    it. This check gates the whole run: on FAIL, nothing else executes."""
    try:
        actual = client.version()
    except ProbeError as exc:
        return result("version", ERROR, f"cannot reach {client.host}: {exc}")
    pattern = profile["version_pattern"]
    if not re.match(pattern, actual):
        return result(
            "version", FAIL,
            f"server at {client.host} is not the build this profile describes",
            expected=f"match {pattern}  (profile {profile_id})", actual=actual,
            diagnosis="Wrong server on this port, or the image was built from a "
                      "different commit. Every measurement below would be "
                      "attributed to the wrong build — run aborted.")
    return result("version", PASS, f"{actual} matches profile {profile_id}",
                  expected=pattern, actual=actual)


def check_image_tag(client, profile, image_tag, container):
    """The operator names an image tag; confirm the running container is it."""
    expected = profile.get("reference_image")
    if not container:
        return result("image_tag", SKIP, "no container resolved (remote host?)",
                      expected=image_tag)
    try:
        actual = subprocess.run(
            ["docker", "inspect", container, "--format", "{{.Config.Image}}"],
            capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as exc:
        return result("image_tag", SKIP, f"docker inspect unavailable: {exc}")
    if not actual:
        return result("image_tag", SKIP, f"could not inspect {container}")
    if image_tag and actual != image_tag:
        return result("image_tag", FAIL,
                      f"container {container} is not running the named image",
                      expected=image_tag, actual=actual,
                      diagnosis="The tag under test and the tag actually serving "
                                "this port differ.")
    return result("image_tag", PASS, f"{container} runs {actual}",
                  expected=image_tag or expected, actual=actual)


# --------------------------------------------------------------------------
# 1b. Payload identity (llama.cpp build)
# --------------------------------------------------------------------------

def sha_prefix_match(expected, actual, min_len=7):
    """A short sha matches a pin when one prefixes the other.

    Written twice before (payload_pin and the MLX pin), which mattered because
    the naive symmetric form accepts a pin SHORTER than the reported sha: a
    3-character mlx_build passed against a commit it does not name. Comparing
    only min(len) characters, and refusing fewer than min_len of them, closes
    that in both callers at once.
    """
    n = min(len(expected or ""), len(actual or ""))
    if n < min_len:
        return False
    return expected[:n] == actual[:n]


def check_payload_pin(profile, container, exec_cmd=None):
    """Assert the running payload's llama.cpp SHA matches what this profile was
    measured against.

    Without this, a llama.cpp bump inherits the previous payload's expectations
    silently: b10091 and b10353 both matched `^0\\.32\\.5-dynres-` and resolved to
    cuda-dynres-005, so the second was validated against numbers measured on the
    first. It passed — but only by luck, and a bump that *did* move sizing would
    have surfaced as an unexplained ladder failure instead of a payload change."""
    expected = profile.get("llama_cpp_build")
    if not expected:
        return result("payload_pin", SKIP,
                      "profile records no llama_cpp_build to assert against",
                      diagnosis="Add llama_cpp_build to this profile so a "
                                "llama.cpp bump fails loudly instead of "
                                "inheriting stale expectations. See README.md.")
    if not container:
        return result("payload_pin", SKIP,
                      "no container resolved; cannot read llama-server --version",
                      expected=expected)
    try:
        actual = llama_cpp_build(container, exec_cmd=exec_cmd)
    except Exception as exc:
        return result("payload_pin", ERROR, f"could not read build sha: {exc}",
                      expected=expected)
    if not sha_prefix_match(expected, actual):
        return result(
            "payload_pin", FAIL, "llama.cpp payload is not the one measured",
            expected=expected, actual=actual,
            diagnosis="The compiled payload changed. Every ladder, payload proof "
                      "and pinned budget below was measured on a different "
                      "llama.cpp. Re-measure deliberately and update this "
                      "profile with provenance — do NOT edit values to go green.")
    return result("payload_pin", PASS,
                  f"llama.cpp build {actual} matches the measured payload",
                  expected=expected, actual=actual)


# --------------------------------------------------------------------------
# 2. Go-side patch marker
# --------------------------------------------------------------------------

def check_patch_marker(profile, container, exec_cmd=None):
    expected = profile.get("expect_patch_marker", 1)
    if not container:
        return result("go_patch_marker", SKIP,
                      "no container resolved; cannot grep the binary",
                      expected=expected)
    try:
        actual = grep_binary_marker(container, exec_cmd=exec_cmd)
    except Exception as exc:
        return result("go_patch_marker", ERROR, f"grep failed: {exc}",
                      expected=expected)
    if actual != expected:
        return result(
            "go_patch_marker", FAIL, "--image-max-tokens marker count is wrong",
            expected=expected, actual=actual,
            diagnosis="0 means a stock ollama/ollama binary — visionServerArgs "
                      "has no gemma4/nemotron_h_omni branch, so the budget flags "
                      "are never passed.")
    return result("go_patch_marker", PASS,
                  f"--image-max-tokens present in binary ({actual})",
                  expected=expected, actual=actual)


# --------------------------------------------------------------------------
# 2b. Poison probe — the qwen2.5vl fp16-accumulate overflow canary (#214)
# --------------------------------------------------------------------------

def is_degenerate_decode(response, done_reason):
    """The #214 poison fingerprint: done_reason null, or one glyph repeated
    ('?'x31 on the clip path, '!'x31 on the 0.7.1 Go engine). A short or
    empty response is NOT this signature — the num_predict trap and
    think-budget exhaustion produce those and mean something else."""
    if done_reason is None:
        return True
    text = (response or "").strip()
    return len(text) > 5 and len(set(text)) == 1


def check_poison_probe(client, expect, profile_id,
                       container=None, log_cmd=None):
    """Send the synthetic 1.06x-fp16-ceiling checkerboard to the recorded
    model on a fresh slot and require a healthy decode, then a text-only
    follow-up on the same slot to require no poisoning residue.

    This is a defect-class canary, not an arch baseline: qwen2.5vl's vision
    tower carries final-block massive activations that overflow fp16 GEMM
    accumulation on CUDA/HIP (#214, measured). The launcher closes the class
    by injecting GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 into qwen25vl runners
    (docs/maxusai/qwen25vl-cublas-f32-env.md); a build without that gate — or
    a container overriding it to f16 — fails this check deterministically."""
    if not expect:
        return result("poison_probe", SKIP,
                      "no poison-probe expectation recorded for this profile")
    model = expect["model"]
    if model not in client.tags():
        return result("poison_probe", FAIL, f"{model} is not on this server",
                      expected=model)
    client.unload(model)
    since = time.time() - 5
    try:
        trig = client.generate(model, "Describe this image in one sentence.",
                               images=[poison_image_b64()], num_predict=48,
                               num_ctx=8192, label="poison_probe")
        after = client.generate(model, "Reply with the single word OK.",
                                num_predict=8, num_ctx=8192,
                                label="poison_probe_after")
    except ProbeError as exc:
        return result("poison_probe", ERROR, str(exc))
    finally:
        client.unload(model)
    bad_trig = is_degenerate_decode(trig.get("response"), trig.get("done_reason"))
    bad_after = is_degenerate_decode(after.get("response"), after.get("done_reason"))
    if bad_trig or bad_after:
        where = "trigger request" if bad_trig else "follow-up (slot residue)"
        r = trig if bad_trig else after
        return result(
            "poison_probe", FAIL, f"degenerate decode on the {where}",
            expected="healthy decode of the synthetic checkerboard trigger",
            actual=f"done_reason={r.get('done_reason')!r} "
                   f"head={(r.get('response') or '')[:40]!r}",
            diagnosis="The fp16-accumulate cuBLAS path is live for this model: "
                      "the qwen25vl runner gate is missing from this binary or "
                      "overridden to f16 in the container environment. See "
                      "docs/maxusai/qwen25vl-cublas-f32-env.md and "
                      "docs/maxusai/qwen25vl-3b-poison-image-garbage-decode.md.")
    # Node-level corroboration, when the operator has enabled the meter
    # (llama/compat/801) and the harness can read this container's logs. The
    # decode verdict is downstream of the fault by the whole language model;
    # n_inf at ffn_down is the fault itself. Overflow that happens not to
    # produce a degenerate decode is invisible to the text check and caught
    # here, so this can only ever turn a PASS into a FAIL.
    node = _poison_node_evidence(container, since, log_cmd)
    if node.get("bad"):
        return result(
            "poison_probe", FAIL,
            f"decode looked healthy but {node['bad']} non-finite element(s) "
            f"reached {node['name']}",
            expected="no non-finite values at the vision tower's f16 matmuls",
            actual=node["line"],
            diagnosis="fp16 accumulation overflowed even though the text came "
                      "back readable — the gate is absent or partial. The "
                      "decode check alone would have passed this build. See "
                      "docs/maxusai/qwen25vl-cublas-f32-env.md.")
    return result(
        "poison_probe", PASS,
        f"1.06x-ceiling trigger decodes healthily "
        f"({len((trig.get('response') or '').strip())} chars, "
        f"done_reason={trig.get('done_reason')!r}); slot clean after"
        + node.get("note", ""),
        actual=(trig.get("response") or "")[:60])


def _poison_node_evidence(container, since, log_cmd=None):
    """Read CLIP_NODE_STATS lines this probe emitted, if the meter is on.

    Returns {} shaped as {"bad": int, "name": str, "line": str} on overflow,
    or {"note": str} describing why there is no node-level evidence. Absence
    of the meter is NOT a failure: it is off by default and most builds under
    test will not have OLLAMA_CLIP_NODE_STATS set.
    """
    if not container:
        return {"note": "; no node-level check (no container resolved)"}
    try:
        lines = [l for l in container_logs(container, since, log_cmd).splitlines()
                 if "CLIP_NODE_STATS" in l]
    except Exception as exc:                                  # noqa: BLE001
        return {"note": f"; no node-level check (log read failed: {exc})"}
    if not lines:
        return {"note": "; no node-level check (meter off — set "
                        "OLLAMA_CLIP_NODE_STATS=ffn_down to enable)"}
    worst = None
    for line in lines:
        kv = dict(p.split("=", 1) for p in line.split() if "=" in p)
        try:
            bad = int(kv.get("n_inf", 0)) + int(kv.get("n_nan", 0))
        except ValueError:
            continue
        if bad and (worst is None or bad > worst["bad"]):
            worst = {"bad": bad, "name": kv.get("name", "?"), "line": line.strip()[:160]}
    if worst:
        return worst
    return {"note": f"; node meter clean over {len(lines)} node(s)"}


# --------------------------------------------------------------------------

def check_mlx_payload_pin(profile, container, since, log_cmd=None):
    """Assert the running MLX payload is the one this profile was measured on.

    payload_pin does this for llama.cpp, but on mlx-metal that is the WRONG
    payload: those profiles set `patchset = []` because the compat patches do
    not apply, and each one names an MLX pin. Without this, a second MLX bump
    under an unchanged base version would match `version_pattern`, resolve the
    profile, and silently inherit ladders and budgets measured on different MLX
    — the b10091/b10353 accident payload_pin was written for.

    It also catches BINARY/PAYLOAD SKEW, which no version string can express:
    the MLX library is dlopen'd from libOllamaRoots(), which falls back to the
    repo's build/lib/ollama, so an ollama binary does not carry its MLX with it.
    Measured 2026-08-30: the archived 0.33.0 binary, run after the repo was
    rebuilt at MLX c793734, reported 0.32.1-37-gc793734 and this check failed it
    against 27fec909 — old Go, new MLX, a pairing nothing was measured on.

    Reading the value needs log access: a container, or a --log-cmd, which is
    how the native macOS path supplies it. probes.mlx_build enforces the time
    window PER LINE rather than trusting the fetch command to have honoured
    {since} — a `cat` template drops the window entirely; see the note there.
    """
    name = "mlx_payload_pin"
    expected = profile.get("mlx_build")
    if not expected:
        return result(name, SKIP,
                      "profile records no mlx_build to assert against",
                      diagnosis="Add mlx_build (the MLX_VERSION commit this "
                                "profile was measured on) so an MLX bump fails "
                                "loudly instead of inheriting stale "
                                "expectations. Platforms with no MLX payload "
                                "leave it unset and keep this skip.")
    if not container and not log_cmd:
        return result(name, SKIP,
                      "no container and no --log-cmd; cannot read the engine "
                      "init line",
                      expected=expected,
                      diagnosis="Pass --log-cmd 'cat <serve log>' on a native "
                                "host so the MLX pin is asserted rather than "
                                "assumed.")
    try:
        actual, seen = mlx_build(container, since, log_cmd)
    except Exception as exc:
        return result(name, ERROR, f"could not read logs: {exc}",
                      expected=expected)

    # FAIL, not SKIP, for the reason check_payload_proof already FAILs here:
    # this check only runs where a pin is declared, i.e. an MLX platform where a
    # load must have happened during the run. SKIP is not counted against the
    # exit code, so a deploy gated on it goes green with the pin unasserted.
    if actual is None:
        if seen:
            return result(
                name, FAIL,
                f"{seen} MLX engine-init line(s) found, none inside this run's "
                f"window", expected=expected,
                diagnosis="Those lines predate this run, so they describe a "
                          "PREVIOUS server process and not the payload under "
                          "test — the stale-log read that a --log-cmd catting "
                          "an accumulating file produces. Truncate or rotate "
                          "the serve log before the run.")
        return result(
            name, FAIL, "no MLX engine-init line in the log window",
            expected=expected,
            diagnosis='The runner logs "MLX engine initialized" when the MLX '
                      "runner starts. None appeared: either nothing loaded on "
                      "the MLX path during this run, or --log-cmd points at the "
                      "wrong file. Nothing was verified — this is not a pass.")

    short, dirty = mlx_describe_commit(actual)
    if dirty:
        return result(
            name, FAIL, f"MLX was built from a dirty tree ({actual})",
            expected=expected, actual=actual,
            diagnosis="`git describe --dirty` means the MLX source carried "
                      "uncommitted changes at build time, so the payload is not "
                      "the pinned commit whatever sha it reports — exactly the "
                      "modified-payload case this check exists to catch.")
    if not short:
        return result(
            name, FAIL,
            f"MLX reports {actual}, which carries no commit to check against "
            f"the pin", expected=expected, actual=actual,
            diagnosis="The build passes --long, which guarantees a -g<sha> "
                      "suffix even at an exact tag, so a missing suffix means "
                      "--always fired with no reachable tag. The payload cannot "
                      "be identified.")
    if not sha_prefix_match(expected, short):
        return result(
            name, FAIL, "MLX payload is not the one measured",
            expected=expected, actual=actual,
            diagnosis="The MLX payload changed. Every ladder and budget in this "
                      "profile was measured on a different MLX. Re-measure "
                      "deliberately and add a new profile with provenance (ADR "
                      "0011) — do NOT edit values to go green.")
    return result(name, PASS,
                  f"MLX build {short} matches the measured payload ({actual})",
                  expected=expected, actual=actual)


# --------------------------------------------------------------------------
# 3. Payload patch proof — from the MODEL-LOAD LOG, never the binary
# --------------------------------------------------------------------------

def check_payload_proof(expect, arch, container, since, log_cmd=None):
    """N == max_tokens * S^2 where S = patch_size * n_merge.

    Static inspection of libmtmd.so is explicitly NOT used: `strings` is absent
    from the ollama images so in-container greps return misleading zeros, and
    <img>/</img> literals appear in stock too (InternVL uses them). An RTTI
    occurrence-count delta was suggestive but never proof.
    """
    name = "payload_proof"
    if not container:
        # Two different skips wore the same message. "No container resolved" is
        # a fact about THIS run — bring one and the check runs. On a platform
        # whose runner emits no load_hparams line at all, no run will ever read
        # these budgets, and reporting that as a missing container invites the
        # next operator to go looking for one. The block says which it is.
        if expect.get("budgets_observed") is False:
            return result(
                name, SKIP,
                "budget/pixel values are not observable on this platform",
                arch=arch,
                diagnosis="This block's budgets were established without a load "
                          "log (the native MLX path emits none) and are recorded "
                          "as budgets_observed = false. Nothing here is waiting "
                          "on a container; see the profile notes for how the "
                          "values were established.")
        return result(name, SKIP, "no container resolved; cannot read the load log",
                      arch=arch)

    expect_absent = expect.get("expect_no_pixel_log_line", False)
    try:
        logs = container_logs(container, since, log_cmd)
    except Exception as exc:
        return result(name, ERROR, f"could not read logs: {exc}", arch=arch)

    lines = parse_pixel_lines(logs)

    if expect_absent:
        # An unpatched projector never calls set_limit_image_tokens(), leaving the
        # values at the -1 sentinel, and the log lines are gated on value > 0.
        # Absence is the proof.
        if lines:
            return result(name, FAIL,
                          "pixel budget logged, but this payload should have none",
                          arch=arch, expected="no image_*_pixels line",
                          actual=[f"{d['kind']}={d['value']}" for d in lines],
                          diagnosis="This profile describes a pre-002 payload where "
                                    "the flags are inert. A budget line means the "
                                    "wrong payload is deployed.")
        return result(name, PASS, "no pixel budget logged, as expected for this payload",
                      arch=arch, expected="no image_*_pixels line", actual="absent")

    want = {"min": expect["image_min_pixels"], "max": expect["image_max_pixels"]}
    stride, bmin, bmax = (expect["patch_stride"], expect["budget_min_tokens"],
                          expect["budget_max_tokens"])
    derivation = (f"min {bmin}*{stride}^2={want['min']}, "
                  f"max {bmax}*{stride}^2={want['max']}")

    if not lines:
        return result(name, FAIL, "no load_hparams pixel budget in the fresh log",
                      arch=arch, expected=derivation, actual="no matching log line",
                      diagnosis="Either the model never loaded during this run (so "
                                "no fresh load_hparams block was emitted), or the "
                                "payload lacks the budget patch entirely.")

    # Pair the lines into (min, max) blocks in log order — one per model load —
    # and read the last DEFAULT-budget block. A pinned probe legitimately logs
    # min == max, which is not what this check is about. payload_proof already
    # runs before any pinned probe, so this is belt-and-braces against a future
    # reordering silently turning a pinned block into the "proof".
    blocks, cur = [], {}
    for d in lines:
        if d["kind"] in cur:
            blocks.append(cur)
            cur = {}
        cur[d["kind"]] = d
    if cur:
        blocks.append(cur)

    def is_pinned(b):
        return ("min" in b and "max" in b
                and b["min"]["value"] == b["max"]["value"])

    usable = blocks
    if want["min"] != want["max"]:
        usable = [b for b in blocks if not is_pinned(b)] or blocks
    got = usable[-1] if usable else {}

    # Which bounds this arch's flags actually set. Both, for the arches whose
    # visionServerArgs case passes --image-min-tokens AND --image-max-tokens
    # (gemma4, nemotron_h_omni). The qwen VL family passes only the min: its max
    # is llama.cpp's structural set_limit_image_tokens(8, 4096) ceiling and is
    # "not tunable through --image-max-tokens" (llm/llama_server.go, on
    # qwenVLImageMaxTokens). Demanding "(custom value)" on a bound no flag can
    # set makes a correct build fail, so the arch declares what to expect.
    #
    # Checked in BOTH directions. A bound declared uncustomised that starts
    # logging as custom means a flag began being passed, which is as much a
    # change as one going missing — and silently tolerating it would hide the
    # arch gate being edited.
    custom_bounds = set(expect.get("custom_bounds", ["min", "max"]))

    bad = []
    for kind, value in want.items():
        if kind not in got:
            bad.append(f"{kind}: missing")
        elif got[kind]["value"] != value:
            bad.append(f"{kind}: expected {value}, got {got[kind]['value']}")
        elif kind in custom_bounds and not got[kind]["custom"]:
            bad.append(f"{kind}: value right but not marked '(custom value)' — "
                       f"the flags were not applied")
        elif kind not in custom_bounds and got[kind]["custom"]:
            bad.append(f"{kind}: marked '(custom value)' but this arch passes no "
                       f"flag for it — the arch gate changed")
    actual = {k: f"{v['value']}{' (custom value)' if v['custom'] else ''}"
              for k, v in got.items()}
    if bad:
        return result(name, FAIL, "; ".join(bad), arch=arch,
                      expected=derivation, actual=actual,
                      diagnosis="The Go binary passes the flags but the llama.cpp "
                                "payload is not honouring them, or the budget "
                                "defaults changed without this file being updated.")
    marked = "/".join(sorted(custom_bounds)) if custom_bounds else "neither bound"
    return result(name, PASS,
                  f"budget logged as custom on {marked} ({derivation})",
                  arch=arch, expected=derivation, actual=actual)


# --------------------------------------------------------------------------
# 4. Token ladder — PER-ARCH verdict logic
# --------------------------------------------------------------------------

def check_ladder(client, expect, arch, sizes, baseline, tol_default=2):
    """Same image at five 16:9 geometries, delta against the text-only baseline.

    The verdict is per-arch and MUST NOT be shared. A flat ladder means an
    unpatched payload for nemotron (dynamic resolution never engaged) but is the
    CORRECT result for gemma4 under 004, which budget-fills every image to the
    ceiling. A shared heuristic gets this exactly backwards.
    """
    model = expect["model"]
    want = expect["ladder"]
    tol = expect.get("ladder_tolerance", tol_default)
    scaling = expect["scaling"]

    got, rows = [], []
    for size, exp in zip(sizes, want):
        try:
            delta, resp = client.visual_tokens(model, size, baseline)
        except ProbeError as exc:
            return result("token_ladder", ERROR, f"{size}: {exc}", arch=arch)
        got.append(delta)
        rows.append({"size": size, "expected": exp, "actual": delta,
                     "delta": delta - exp, "ok": abs(delta - exp) <= tol,
                     "queue_wait_s": resp["_queue_wait_s"]})

    is_flat = len(set(got)) == 1
    mismatches = [r for r in rows if not r["ok"]]

    diagnosis = None
    if mismatches:
        if scaling == "dynamic" and is_flat:
            diagnosis = (f"Ladder is FLAT at {got[0]} on an arch whose payload "
                         f"should scale with resolution — the dynamic-resolution "
                         f"patch is not in this payload (the Go flags are parsed "
                         f"but inert). This is the unpatched-payload signature.")
        elif scaling == "flat" and not is_flat:
            diagnosis = ("Ladder VARIES on an arch that should budget-fill to a "
                         "constant — the 004 budget-fill behaviour is missing, or "
                         "the ceiling changed.")
        else:
            diagnosis = ("Values moved but the shape is right: most likely a "
                         "budget or preprocessing change. Re-measure and update "
                         "expectations.toml if the new numbers are intended.")

    status = PASS if not mismatches else FAIL
    summary = (f"{len(rows) - len(mismatches)}/{len(rows)} geometries within +/-{tol}"
               + (f" (shape: {'flat' if is_flat else 'scaling'}, expected {scaling})"
                  if mismatches else ""))
    return result("token_ladder", status, summary, arch=arch,
                  expected=dict(zip(sizes, want)), actual=dict(zip(sizes, got)),
                  diagnosis=diagnosis, rows=rows, shape="flat" if is_flat else "scaling",
                  expected_shape=scaling)


# --------------------------------------------------------------------------
# 5. Pinned-budget probe — the 005 defect class
# --------------------------------------------------------------------------

def check_pinned_image_token_budget(client, expect, arch, baseline, marker_allowance=2):
    """image_min_tokens == image_max_tokens. Pre-005, nemotron pinned to 3328
    delivered 3390 — 60 grid tokens over its own ceiling.

    Two independent assertions:
      1. the exact measured regression value, and
      2. the class invariant `delivered - markers <= ceiling`, which catches a
         NEW overshoot at a number nobody has measured yet.
    """
    pin = expect.get("pinned")
    name = "pinned_image_token_budget"
    # An arch can be structurally unable to pin, which is not the same as nobody
    # having measured it yet. visionServerArgs is arch-gated: gemma4 and
    # nemotron_h_omni build their flags from the request options, while the qwen
    # VL family returns a fixed --image-min-tokens and never reads opts, so
    # image_min_tokens/image_max_tokens are inert there. Saying "not yet
    # measured" would send someone to measure something that cannot move.
    na = expect.get("pinned_not_applicable")
    if na and not pin:
        return result(name, SKIP, f"pinned budget does not apply to this arch: {na}",
                      arch=arch)
    if not pin:
        return result(name, SKIP,
                      "no pinned-budget expectation recorded for this arch",
                      arch=arch,
                      diagnosis="Omitted deliberately: the 005 defect was found on "
                                "nemotron and this arch's probe has not been "
                                "measured. See README.md to add one.")

    model, size, pinv = expect["model"], pin["size"], pin["pin_tokens"]
    tol = pin.get("tolerance", 4)
    ceiling = expect["budget_max_tokens"]
    sub = []

    # --- pinned arm (forces a runner reload; grouped by the caller) ---
    try:
        delta, _ = client.visual_tokens(model, size, baseline,
                                        image_min_tokens=pinv, image_max_tokens=pinv)
    except ProbeError as exc:
        return result(name, ERROR, f"pinned probe failed: {exc}", arch=arch)

    want = pin["expect_tokens"]
    exact_ok = abs(delta - want) <= tol
    sub.append({"arm": "pinned", "pin": pinv, "expected": want, "actual": delta,
                "ok": exact_ok})

    grid = delta - marker_allowance
    ceiling_ok = True
    if pin.get("enforce_ceiling_invariant", True):
        ceiling_ok = grid <= ceiling
        sub.append({"arm": "ceiling_invariant",
                    "expected": f"grid <= {ceiling}", "actual": grid,
                    "ok": ceiling_ok})

    # --- unpinned control, same image, same run ---
    control_ok = True
    if "control_expect_tokens" in pin:
        ctol = pin.get("control_tolerance", 4)
        try:
            cdelta, _ = client.visual_tokens(model, size, baseline)
        except ProbeError as exc:
            return result(name, ERROR, f"control probe failed: {exc}", arch=arch)
        cwant = pin["control_expect_tokens"]
        control_ok = abs(cdelta - cwant) <= ctol
        sub.append({"arm": "unpinned_control", "expected": cwant, "actual": cdelta,
                    "ok": control_ok})

    diagnosis = None
    if not ceiling_ok:
        diagnosis = (f"OVERSHOOT: pinned to {pinv} but delivered {grid} grid tokens "
                     f"({delta} incl. markers) — {grid - ceiling} over the ceiling. "
                     f"This is the 005 defect class; the pinned dyn_size path is "
                     f"not clamping. Check llama/compat/005-*.patch is applied.")
    elif not exact_ok:
        diagnosis = (f"Pinned cost moved ({want} -> {delta}) but stays under the "
                     f"ceiling, so this is a behaviour change, not the 005 defect. "
                     f"Re-measure and update expectations.toml if intended.")
    elif not control_ok:
        diagnosis = ("The unpinned control moved while the pinned arm held — a "
                     "pinned probe leaked into the default budget, or the default "
                     "changed.")

    status = PASS if (exact_ok and ceiling_ok and control_ok) else FAIL
    return result(name, status,
                  f"pinned {pinv} -> {delta} (ceiling {ceiling}, +{marker_allowance} markers)",
                  arch=arch, expected=want, actual=delta, diagnosis=diagnosis, arms=sub)


# --------------------------------------------------------------------------
# 6. think + format non-empty
# --------------------------------------------------------------------------

def schema_violations(obj, schema, path="$"):
    """Violations of a DELIBERATE SUBSET of JSON Schema: type (object/array/
    string/integer/number/boolean), properties, required, items.

    Hand-written because the harness takes no third-party dependency, and
    because the subset is the point: these probes assert that constrained
    decoding produced the requested SHAPE, not that a general validator agrees.
    Anything outside the subset is ignored rather than guessed at — an unknown
    keyword must not manufacture a failure.

    bool is checked before int on purpose: in Python True is an int, so a
    boolean would otherwise satisfy {"type": "integer"}.
    """
    out = []
    t = schema.get("type")
    if t == "object" and not isinstance(obj, dict):
        return [f"{path}: expected object, got {type(obj).__name__}"]
    if t == "array" and not isinstance(obj, list):
        return [f"{path}: expected array, got {type(obj).__name__}"]
    if t == "string" and not isinstance(obj, str):
        return [f"{path}: expected string, got {type(obj).__name__}"]
    if t == "boolean" and not isinstance(obj, bool):
        return [f"{path}: expected boolean, got {type(obj).__name__}"]
    if t == "integer" and (isinstance(obj, bool) or not isinstance(obj, int)):
        return [f"{path}: expected integer, got {type(obj).__name__}"]
    if t == "number" and (isinstance(obj, bool)
                          or not isinstance(obj, (int, float))):
        return [f"{path}: expected number, got {type(obj).__name__}"]

    if isinstance(obj, dict):
        for key in schema.get("required", []):
            if key not in obj:
                out.append(f"{path}: missing required key {key!r}")
        for key, sub in (schema.get("properties") or {}).items():
            if key in obj:
                out.extend(schema_violations(obj[key], sub, f"{path}.{key}"))
    if isinstance(obj, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(obj):
            out.extend(schema_violations(item, schema["items"], f"{path}[{i}]"))
    return out


def check_think_format(client, expect, arch, min_num_predict):
    """Stock returns an empty `response` for nemotron3/qwen3.6 when think:true is
    combined with format:"json". The fork emits valid JSON after thinking.

    num_predict is floored: at 120 three probes returned response:"" with
    eval_count exactly 120 — the whole allowance spent inside an unclosed
    thinking block. That reads as a vision failure and is not one, so `thinking`
    is judged alongside `response`.
    """
    cfg = expect.get("think_format")
    name = "think_format"
    if not cfg:
        return result(name, SKIP, "no think+format expectation recorded", arch=arch)

    np_ = cfg.get("num_predict", min_num_predict)
    if np_ < min_num_predict:
        return result(name, ERROR,
                      f"num_predict {np_} is below the enforced floor {min_num_predict}",
                      arch=arch,
                      diagnosis="Refusing to run: a low num_predict manufactures a "
                                "false vision failure. Raise it in expectations.toml.")
    # The schema describes the shape this prompt ALREADY asks for, deliberately.
    # A schema is only meaningful if the prompt requests the same object — but
    # the prompt must also stay the one the recorded token counts were measured
    # on. Asking for a different shape (measured 2026-08-30: "count the coloured
    # shapes and name each colour") sent gemma4:12b into runaway thinking, 4000
    # tokens and an empty response, so the probe failed a healthy build. Keep
    # the task fixed; constrain its shape.
    try:
        resp = client.generate(
            expect["model"],
            "List three visual facts about this image as JSON: "
            '{"facts": ["...", "...", "..."]}',
            images=[ladder_image_b64("1024x576")], num_predict=np_,
            think=True, fmt=(cfg.get("schema") or "json"), label="think_format")
    except ProbeError as exc:
        return result(name, ERROR, str(exc), arch=arch)

    body = (resp.get("response") or "").strip()
    thinking = (resp.get("thinking") or "").strip()
    eval_count = resp.get("eval_count", 0)

    failures, diagnosis = [], None
    if cfg.get("require_nonempty_response", True) and not body:
        failures.append("response is empty")
        if eval_count >= np_:
            diagnosis = (f"eval_count ({eval_count}) hit num_predict ({np_}) with an "
                         f"empty response and {len(thinking)} chars of thinking — "
                         f"NOT a vision failure. This gate samples at temperature 0, "
                         f"which is off-policy for thinking mode and can stop "
                         f"reasoning from ever terminating; raising num_predict does "
                         f"NOT fix that (measured: five num_ctx rungs to 128K, never "
                         f"converged). Re-run the probe with the model card's "
                         f"sampling before touching the budget — see "
                         f"docs/maxusai/runaway-reasoning-under-think.md. Only if it "
                         f"still caps on-policy is the budget genuinely too low.")
        else:
            diagnosis = (f"Generated {eval_count} tokens then emitted no JSON body, "
                         f"well under the {np_} budget. That is the stock "
                         f"think+format signature — the fork's fix is missing.")
    if cfg.get("require_nonempty_thinking", False) and not thinking:
        failures.append("thinking is empty")
    if cfg.get("require_valid_json", True) and body:
        try:
            parsed = json.loads(body)
        except Exception as exc:
            failures.append(f"response is not valid JSON: {exc}")
        else:
            # A schema turns this from "did it parse" into "is it the shape we
            # asked for". Without it, ADR 0033's grammar engine was gated by
            # json.loads() alone, and a regression emitting well-formed JSON of
            # the wrong shape — the exact failure constrained decoding prevents
            # — passed clean.
            if cfg.get("schema"):
                failures.extend(schema_violations(parsed, cfg["schema"]))

    status = PASS if not failures else FAIL
    return result(name, status,
                  "; ".join(failures) if failures
                  else f"valid JSON after thinking ({eval_count} tokens)",
                  arch=arch,
                  expected="non-empty JSON response with think:true + format:json",
                  actual={"response_chars": len(body), "thinking_chars": len(thinking),
                          "eval_count": eval_count},
                  diagnosis=diagnosis)


# --------------------------------------------------------------------------
# 7. Extraction quality — delegates scoring to the existing vision_suite.py
# --------------------------------------------------------------------------

def quality_eligible(scores, tests=None):
    """Split suite scores into (eligible-for-quality, capped-arm-names).

    A capped arm scores json_valid: False as a side effect of truncation, so
    counting it in the quality denominator misattributes a harness setting to
    the model (ADR 0012 conv 9). `tests` scopes the split to THIS run's
    requested arms: the scores file is shared per (platform, arch) tag and
    vision_suite resumes into it, so without the scope a capped arm left by
    another profile's test list fails a run that never asked for it.
    Delegates to the suite's own was_capped — the ONE definition (SPEC H5);
    this file was the third consumer reading raw score fields around it. The
    import lives here rather than at module top because release lineages
    carry preflight/ without the suite — and on those lineages check_quality
    has already SKIPped before scoring.
    """
    if SUITE_DIR not in sys.path:
        sys.path.insert(0, SUITE_DIR)
    from summarize_engine_compare import was_capped
    eligible, capped = {}, []
    for name, blk in scores.items():
        if tests is not None and name not in tests:
            continue
        if not isinstance(blk, dict) or "error" in blk:
            continue
        if was_capped(blk):
            capped.append(name)
        else:
            eligible[name] = blk
    return eligible, capped


def check_quality(host, quality, expect, arch, tag, timeout=5400):
    """Runs vision_suite.py and applies thresholds to the scores it already
    computes. The suite reports; this turns the report into a verdict."""
    name = "extraction_quality"
    if not quality or quality.get("status") != "measured":
        return result(name, SKIP, "no quality thresholds recorded for this arch",
                      arch=arch)

    tests = quality.get("tests", ["scene_single", "document_single"])
    model = expect["model"]

    # The harness is cherry-picked to lineages that do not carry the whole vision
    # suite (release/0.32.1-dynres has preflight/ but no vision_suite.py). Skip
    # with a reason rather than erroring — a missing scorer is not a build defect,
    # and reporting it as one is the false-failure class this harness exists to
    # remove. Token and payload checks are unaffected.
    if not os.path.exists(os.path.join(SUITE_DIR, "vision_suite.py")):
        return result(name, SKIP, "vision_suite.py is not present in this tree",
                      arch=arch,
                      diagnosis="Extraction scoring is delegated to the vision "
                                "suite, which is not on every lineage. Run the "
                                "quality arm from a tree that carries it.")

    # visimgs/ is gitignored, so on a fresh clone vision_suite.py fails at import
    # (it loads ground_truth.json at module level). Generate the scenes first.
    ground_truth = os.path.join(SUITE_DIR, "visimgs", "ground_truth.json")
    if not os.path.exists(ground_truth):
        gen = subprocess.run(["python3", "gen_scenes.py"], cwd=SUITE_DIR,
                             capture_output=True, text=True, timeout=600)
        if not os.path.exists(ground_truth):
            return result(name, ERROR, "could not generate scenes for the quality run",
                          arch=arch,
                          actual=(gen.stdout + gen.stderr)[-400:],
                          diagnosis="gen_scenes.py needs Pillow and the DejaVu fonts "
                                    "(/usr/share/fonts/truetype/dejavu/). Install them "
                                    "or run without --quality.")

    env = dict(os.environ, THINK="false", NUM_PREDICT="2200",
               ONLY_TESTS=",".join(tests), HTTP_TIMEOUT=str(timeout))
    scores_path = os.path.join(SUITE_DIR, f"scores_{tag}.json")
    try:
        proc = subprocess.run(
            ["python3", "vision_suite.py", host, tag, model],
            cwd=SUITE_DIR, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return result(name, ERROR, f"vision_suite.py exceeded {timeout}s", arch=arch,
                      diagnosis="A whole-suite timeout with a healthy server is the "
                                "queue-starvation signature — check for another "
                                "client on this endpoint.")
    if not os.path.exists(scores_path):
        return result(name, ERROR, "vision_suite.py wrote no scores file", arch=arch,
                      actual=(proc.stdout or proc.stderr)[-600:])

    with open(scores_path) as fh:
        scores = json.load(fh)

    metrics, failures = {}, []
    eligible, capped_arms_ = quality_eligible(scores, tests=tests)
    valid = [bool(s.get("json_valid")) for s in eligible.values()]
    errored = [t for t, s in scores.items()
               if (t in tests) and isinstance(s, dict) and "error" in s]
    if errored:
        failures.append(f"tests errored: {', '.join(errored)}")
    if capped_arms_:
        # Capped is not a quality verdict — it is an unfinished measurement.
        # Surfaced so a run with capped arms cannot read as fully assessed.
        failures.append(f"arms capped, not scored: {', '.join(capped_arms_)} "
                        f"(raise NUM_PREDICT / the ladder)")
    if valid:
        metrics["json_valid"] = sum(valid) / len(valid)
        if metrics["json_valid"] < quality.get("min_json_valid", 1.0):
            failures.append(f"json_valid {metrics['json_valid']:.2f} "
                            f"< {quality['min_json_valid']}")
    scene = eligible.get("scene_single", {})
    if scene.get("labels_total"):
        metrics["label_recall"] = scene["labels_found"] / scene["labels_total"]
        floor = quality.get("min_label_recall")
        if floor is not None and metrics["label_recall"] < floor:
            failures.append(f"label_recall {metrics['label_recall']:.2f} < {floor}")
    doc = eligible.get("document_single", {})
    if doc.get("items_total"):
        metrics["qty_price_exact"] = doc["qty_price_right"] / doc["items_total"]
        floor = quality.get("min_qty_price_exact")
        if floor is not None and metrics["qty_price_exact"] < floor:
            failures.append(f"qty_price_exact {metrics['qty_price_exact']:.2f} < {floor}")

    status = PASS if not failures else FAIL
    return result(name, status,
                  "; ".join(failures) if failures else
                  " ".join(f"{k}={v:.2f}" for k, v in metrics.items()),
                  arch=arch,
                  expected={k: v for k, v in quality.items()
                            if k.startswith("min_")},
                  actual={k: round(v, 3) for k, v in metrics.items()},
                  scores_file=os.path.relpath(scores_path, SUITE_DIR))


# --------------------------------------------------------------------------
# Endpoint exclusivity — queue starvation is invisible without this
# --------------------------------------------------------------------------

def check_exclusivity(client, threshold_s=10.0):
    """A vision_suite.py run once failed all three tests with "timed out" at
    exactly 3 x 1800s while the server was perfectly healthy — another client was
    saturating the single slot. A saturated endpoint reports a FALSE FAILURE, so
    contention is detected and named rather than measured through.

    Signal: wall-clock minus the server's own total_duration is time spent
    queued behind someone else's request. Every probe in the run already recorded
    one, so this is a verdict over the whole run rather than an extra request.
    """
    waits = client.queue_waits
    worst_label, worst = max(waits, key=lambda kv: kv[1]) if waits else ("", 0.0)
    loaded = []
    try:
        loaded = [m.get("name") for m in client.ps()]
    except ProbeError:
        pass

    if worst > threshold_s:
        return result(
            "endpoint_exclusive", CONTENTION,
            f"requests queued up to {worst:.1f}s behind another client "
            f"(worst: {worst_label})",
            expected=f"queue wait <= {threshold_s}s", actual=f"{worst:.1f}s",
            diagnosis="Another client is using this endpoint. Results from a "
                      "contended run are not trustworthy — a saturated single slot "
                      "produces timeouts that look like model failures. Stop the "
                      "other client (e.g. pilot_teacher_v3_exam.py) and re-run.",
            models_loaded=loaded)
    return result("endpoint_exclusive", PASS,
                  f"no contention detected (worst queue wait {worst:.1f}s)",
                  expected=f"queue wait <= {threshold_s}s", actual=f"{worst:.1f}s",
                  models_loaded=loaded)
