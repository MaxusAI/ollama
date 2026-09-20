#!/usr/bin/env python3
"""Probe layer for the pre-deploy regression harness.

Stdlib only, deliberately: this runs on build hosts that have an ollama image and
nothing else. Pillow is NOT required — ladder images are geometry, and token cost
is a function of geometry, not content.

Nothing in here asserts. Assertions live in checks.py, expected values live in
expectations.toml.
"""
import base64
import datetime
import json
import os
import re
import shlex
import shutil
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zlib

DIR = os.path.dirname(os.path.abspath(__file__))
IMGDIR = os.path.join(DIR, "ladderimgs")

# One prompt for both the text-only baseline and the image probes, so the text
# tokens cancel in the subtraction. See Ollama.text_baseline().
PROBE_PROMPT = "Describe briefly."


# --------------------------------------------------------------------------
# Deterministic PNG generation (no Pillow)
# --------------------------------------------------------------------------

def _chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def _render_png(w, h):
    """Gradient + 64px gridlines. Deterministic, compresses small, and has real
    edge structure so no degenerate all-one-colour decode path is exercised."""
    row0 = bytearray(w * 3)
    for x in range(w):
        row0[3 * x] = (x * 255) // max(1, w - 1)
        row0[3 * x + 2] = 200 if x % 64 == 0 else 60
    raw = bytearray()
    for y in range(h):
        row = bytearray(row0)
        row[1::3] = bytes([(y * 255) // max(1, h - 1)]) * w
        if y % 64 == 0:
            row[2::3] = bytes([220]) * w
        raw += b"\x00" + row
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + _chunk(b"IEND", b""))


def ladder_image_b64(size):
    """Base64 of the WxH ladder image, generating and caching it on first use."""
    path = os.path.join(IMGDIR, f"{size}.png")
    if not os.path.exists(path):
        w, h = (int(v) for v in size.lower().split("x"))
        os.makedirs(IMGDIR, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(_render_png(w, h))
        os.replace(tmp, path)
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


POISON_W, POISON_H, POISON_PITCH = 1350, 1800, 56


def poison_image_b64():
    """Base64 of the qwen2.5vl fp16-overflow trigger: a 56 px black/white
    checkerboard at 1350x1800. Pixel-identical to
    docs/maxusai/make_poison_repro_image.py (#214) but rendered with the
    stdlib PNG writer above so the harness stays dependency-free. The pattern
    measures 69,120 max |activation| at the 3B tower's final block -- 1.06x
    fp16's 65,504 ceiling -- and deterministically garbles fp16-accumulate
    CUDA serving, while f32/bf16 accumulation decodes it correctly."""
    path = os.path.join(IMGDIR, f"poison_checker{POISON_PITCH}_{POISON_W}x{POISON_H}.png")
    if not os.path.exists(path):
        w, h, p = POISON_W, POISON_H, POISON_PITCH
        rows = []
        for offset in (0, 1):
            row = bytearray(w * 3)
            for x in range(w):
                if ((x // p) + offset) % 2:
                    row[3 * x] = row[3 * x + 1] = row[3 * x + 2] = 255
            rows.append(bytes(row))
        raw = bytearray()
        for y in range(h):
            raw += b"\x00" + rows[(y // p) % 2]
        ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        png = (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
               + _chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + _chunk(b"IEND", b""))
        os.makedirs(IMGDIR, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(png)
        os.replace(tmp, path)
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


# --------------------------------------------------------------------------
# Ollama client
# --------------------------------------------------------------------------

class ProbeError(RuntimeError):
    pass


class Ollama:
    def __init__(self, host, timeout=1800):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.queue_waits = []          # every observed (label, seconds) queue delay

    def _post(self, path, payload, timeout=None):
        req = urllib.request.Request(
            self.host + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as fh:
                return json.load(fh)
        except urllib.error.HTTPError as exc:
            raise ProbeError(f"{path} HTTP {exc.code}: {exc.read()[:400]!r}") from exc
        except Exception as exc:
            raise ProbeError(f"{path}: {exc}") from exc

    def _get(self, path, timeout=30):
        try:
            with urllib.request.urlopen(self.host + path, timeout=timeout) as fh:
                return json.load(fh)
        except Exception as exc:
            raise ProbeError(f"{path}: {exc}") from exc

    def version(self):
        return self._get("/api/version").get("version", "")

    def tags(self):
        return [m["name"] for m in self._get("/api/tags").get("models", [])]

    def ps(self):
        return self._get("/api/ps").get("models", [])

    def unload(self, model):
        """Force the runner to drop the model so the next request emits a fresh
        load_hparams block. Without this, a payload proof can be read off a log
        line written by a PREVIOUS build."""
        try:
            self._post("/api/generate", {"model": model, "keep_alive": 0}, timeout=120)
        except ProbeError:
            pass
        time.sleep(2)

    def generate(self, model, prompt, images=None, num_predict=1, num_ctx=16384,
                 image_min_tokens=None, image_max_tokens=None, think=None,
                 fmt=None, label=""):
        """One /api/generate call. Returns the server response plus timing.

        DELIBERATELY /api/generate, NOT /api/chat -- do not "align" this with
        vision_suite.py, which defaults to chat. Every expectation in
        expectations.toml (text_baseline, the token ladder, pinned budgets) was
        calibrated against generate. Measured 2026-08-19 the two endpoints are
        token-identical on gemma4:31b-it-q4_K_M (1511/1511 think-off,
        1514/1514 think-on) -- ollama templates on /api/generate too, so there is
        no chat-template overhead. The pin is kept anyway because that
        equivalence is a per-model, per-build measurement rather than a
        guarantee, and ADR 0011 treats these expectations as versioned code.
        Re-check with endpoint_compare.py before assuming it holds elsewhere.

        NOTE for think-mode checks: /api/generate returned NO reasoning text for
        GEMMA4 on 0.32.14-rc0-dynres while /api/chat returned 2021 chars for the
        same request -- but nemotron3 and qwen3.8 return it fine on generate, so
        this is per-model, not endpoint-wide. Anything here that needs the
        reasoning ITSELF rather than its token count must not assume this
        endpoint provides it for every arch.

        `queue_wait` is wall-clock minus the server's own reported total_duration:
        the time this request spent waiting for a slot. It is the only reliable
        signal that another client is saturating the endpoint — a saturated server
        looks completely healthy and simply times requests out.
        """
        opts = {"num_predict": num_predict, "num_ctx": num_ctx, "temperature": 0}
        if image_min_tokens is not None:
            opts["image_min_tokens"] = image_min_tokens
        if image_max_tokens is not None:
            opts["image_max_tokens"] = image_max_tokens
        payload = {"model": model, "prompt": prompt, "stream": False, "options": opts}
        if images:
            payload["images"] = images
        if fmt:
            payload["format"] = fmt
        if think is not None:
            payload["think"] = think

        t0 = time.monotonic()
        resp = self._post("/api/generate", payload)
        wall = time.monotonic() - t0

        total = resp.get("total_duration", 0) / 1e9
        queue_wait = max(0.0, wall - total) if total else 0.0
        resp["_wall_s"] = round(wall, 2)
        resp["_server_total_s"] = round(total, 2)
        resp["_queue_wait_s"] = round(queue_wait, 2)
        self.queue_waits.append((label or "probe", round(queue_wait, 2)))
        return resp

    def visual_tokens(self, model, size, baseline, **kw):
        """prompt_eval_count for one image minus the calibrated text prefix."""
        resp = self.generate(model, PROBE_PROMPT,
                             images=[ladder_image_b64(size)], label=f"{size}", **kw)
        return resp["prompt_eval_count"] - baseline, resp

    def text_baseline(self, model):
        """Text-only prompt_eval_count for PROBE_PROMPT.

        Reported as a diagnostic and compared against image_prefix(); it is NOT
        the right subtrahend for visual_tokens(). See image_prefix().
        """
        resp = self.generate(model, PROBE_PROMPT, num_predict=1, label="baseline")
        return resp["prompt_eval_count"], resp

    def image_prefix(self, model, size_a="256x144", size_b="512x288"):
        """The text prefix as tokenised INSIDE an image-bearing request.

        This is load-bearing, and one prompt everywhere is only half of it.

        Trap 1 — mismatched prompts. measure.py used to baseline with "Hi" and
        probe with "Describe briefly.", so the text-length difference landed in
        every row (18 vs 21 tokens on nemotron3:33b-q8).

        Trap 2 — and this is the one a matched prompt does NOT fix. Attaching an
        image can change how the template renders the surrounding text, so the
        text-only count is still the wrong subtrahend. Measured 2026-08-08 on
        the :11437 canary, same prompt, same model: text-only 21, but the prefix
        inside an image request 20. Subtracting the text-only count therefore
        reads every nemotron image exactly 1 token LOW. gemma4:31b measures 19
        both ways, so the offset is arch-specific and cannot be hardcoded.

        Recovered without trusting a text-only probe and without assuming a grid:
        for one fixed prompt and two images A and B, each of the three counts
        carries the prefix P exactly once, so

            count(A) + count(B) - count(A, B)
              = (P + cA) + (P + cB) - (P + cA + cB)
              = P

        Returns (prefix, text_only, detail) so callers can report the delta.
        """
        a, _ = self.visual_tokens(model, size_a, 0)
        b, _ = self.visual_tokens(model, size_b, 0)
        resp = self.generate(model, PROBE_PROMPT,
                             images=[ladder_image_b64(size_a),
                                     ladder_image_b64(size_b)],
                             label=f"{size_a}+{size_b}")
        both = resp["prompt_eval_count"]
        prefix = a + b - both
        text_only, _ = self.text_baseline(model)
        return prefix, text_only, {"one_a": a, "one_b": b, "both": both,
                                   "size_a": size_a, "size_b": size_b}


# --------------------------------------------------------------------------
# Container log access — for the payload patch proof
# --------------------------------------------------------------------------

PIXEL_RE = re.compile(
    r"load_hparams:\s+image_(min|max)_pixels:\s+(\d+)(\s+\(custom value\))?")


def find_container(port, explicit=None):
    """Resolve the container serving `port`. Explicit name always wins."""
    if explicit:
        return explicit
    if not shutil.which("docker"):
        return None
    try:
        out = subprocess.run(
            ["docker", "ps", "--filter", f"publish={port}", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=30).stdout.split()
        return out[0] if out else None
    except Exception:
        return None


def container_logs(container, since_epoch, log_cmd=None):
    """Logs written since `since_epoch`. `--since` is load-bearing: it is what
    guarantees the load_hparams line we read was emitted by THIS build during
    THIS run, not left over from a previous one.

    THE TRAILING `Z` IS PART OF THAT GUARANTEE, not formatting. Docker assumes
    the CLIENT'S LOCAL TIMEZONE for a timestamp that carries no zone, while the
    value here is UTC (`time.gmtime`). On any host that is not at UTC the two
    disagree by the offset, and the window silently becomes the wrong window:
    east of UTC it opens hours EARLY and admits load_hparams lines from previous
    loads of other models; west of UTC it opens hours LATE and admits nothing.

    Measured on a +1000 host against a live container, asking for 60 seconds:

        since='2026-08-17T06:05:31'   -> 51 load_hparams lines
        since='2026-08-17T06:05:31Z'  ->  0 load_hparams lines

    `errors="replace"` is load-bearing too, for a different reason: a serve log
    carries the runner's raw stdout, and one undecodable byte in 25 MB made the
    default strict decode raise UnicodeDecodeError — turning every log-derived
    check on that host into an ERROR about a server that was perfectly healthy.
    Every consumer of this text is a regex over ASCII log lines, so a U+FFFD
    where a stray byte was costs nothing and keeps the window readable.

    Both failure modes are silent. The early window is the dangerous one,
    because a caller that would otherwise see no lines and report `TODO` or
    SKIP instead receives a complete, plausible, wrong answer belonging to a
    different model — which is how a wrong row reaches expectations.toml.
    """
    since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(since_epoch))
    if log_cmd:
        cmd = log_cmd.format(container=container, since=since)
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              errors="replace", timeout=120)
    else:
        proc = subprocess.run(
            ["docker", "logs", "--since", since, container],
            capture_output=True, text=True, errors="replace", timeout=120)
    return (proc.stdout or "") + (proc.stderr or "")


def parse_pixel_lines(text):
    """[{'kind': 'max', 'value': 3407872, 'custom': True}, ...] in log order."""
    return [{"kind": m.group(1), "value": int(m.group(2)), "custom": bool(m.group(3))}
            for m in PIXEL_RE.finditer(text)]


# Runner-launch boundaries, for attributing a load_hparams block to the model
# that emitted it. The Go server logs one "starting llama-server" line per
# runner spawn, and its cmd string carries the --image-*-tokens flags — the
# very values payload_proof verifies the payload honoured. This lives HERE and
# not in checks.py for the same reason MLX_VERSION_RE does: it is a runtime log
# format that will drift, and the person who fixes the drift reads this file.
LAUNCH_RE = re.compile(r'msg="starting llama-server" cmd="([^"]*)"')
IMG_MIN_FLAG_RE = re.compile(r"--image-min-tokens (\d+)")
IMG_MAX_FLAG_RE = re.compile(r"--image-max-tokens (\d+)")
PATCH_SIZE_RE = re.compile(r"load_hparams: patch_size:\s+(\d+)")
N_MERGE_RE = re.compile(r"load_hparams: n_merge:\s+(\d+)")


def parse_load_segments(text):
    """Split a log window at runner-launch lines; parse each load separately.

    Returns [{'launch': bool, 'min_tokens': int|None, 'max_tokens': int|None,
              'patch_size': int|None, 'n_merge': int|None,
              'pixels': parse_pixel_lines(segment)}], in log order.

    Text before the first launch line becomes a launch=False segment: those
    lines belong to a load started before the window and can be attributed to
    nothing, which is exactly how they must be treated. Two models resident on
    one server interleave their loads in one log; grading "the last block in
    the window" against one arch's expectations is how a healthy gemma4 deploy
    failed its smoke against qwen3.8's (correct) numbers on 2026-09-02 — the
    same failure family as the warm-up Reserve() line that nearly sank a
    working CUDA fix. Unattributed log parsing.
    """
    bounds = list(LAUNCH_RE.finditer(text))
    edges = [0] + [m.start() for m in bounds] + [len(text)]
    out = []
    for i in range(len(edges) - 1):
        seg = text[edges[i]:edges[i + 1]]
        if not seg.strip():
            continue
        cmd_m = LAUNCH_RE.search(seg)
        cmd = cmd_m.group(1) if cmd_m else ""
        mn = IMG_MIN_FLAG_RE.search(cmd)
        mx = IMG_MAX_FLAG_RE.search(cmd)
        ps = PATCH_SIZE_RE.search(seg)
        nm = N_MERGE_RE.search(seg)
        out.append({
            "launch": bool(cmd_m),
            "min_tokens": int(mn.group(1)) if mn else None,
            "max_tokens": int(mx.group(1)) if mx else None,
            "patch_size": int(ps.group(1)) if ps else None,
            "n_merge": int(nm.group(1)) if nm else None,
            "pixels": parse_pixel_lines(seg),
        })
    return out


# MLX payload identity, from the runner's engine-init line. This lives HERE and
# not in checks.py for the same reason llama_cpp_build does: checks.py hardcodes
# "nothing except the shapes of the assertions themselves", and a `git describe`
# string is a runtime format that will drift — llama.cpp already moved its
# --version output once (see llama_cpp_build below), and the person who fixes
# that drift reads this file.
MLX_VERSION_RE = re.compile(r'"MLX version"=(\S+)')
SLOG_TIME_RE = re.compile(r"\btime=(\S+)")
# mlx/CMakeLists.txt runs `git describe --tags --first-parent
# --abbrev=7 --long --dirty --always`. --long guarantees the -g<sha> suffix even
# at an exact tag, so a MISSING suffix means --always fired with no reachable
# tag; --dirty appends after it, which is why the sha group is not $-anchored.
MLX_DESCRIBE_RE = re.compile(r"-g([0-9a-f]{7,40})(-dirty)?$")


def mlx_describe_commit(version):
    """('c793734', dirty_bool) from a describe string, or (None, dirty_bool)."""
    m = MLX_DESCRIBE_RE.search(version or "")
    if not m:
        return None, (version or "").endswith("-dirty")
    return m.group(1), bool(m.group(2))


def mlx_build(container, since_epoch, log_cmd=None):
    """The MLX version the runner reported, WINDOWED to since_epoch.

    Returns (version_or_None, lines_seen_total).

    THE WINDOW IS ENFORCED HERE, per line, and that is the whole point of this
    function. container_logs() renders the window by substituting {since} into
    the caller's --log-cmd template, so a template that omits the placeholder —
    `cat <serve log>`, which is the only form that works on the native macOS
    path, where the log is a file and not `docker logs` — silently returns the
    WHOLE file. Measured 2026-08-30: a month-old engine-init line, in a run that
    loaded no model at all, satisfied a five-second window and the pin check
    returned PASS. Filtering on each line's own slog timestamp makes the window
    real regardless of how the lines were fetched.

    A line whose timestamp cannot be parsed is treated as OUT of the window: it
    cannot be shown to belong to this run, and cannot-confirm must never read as
    confirmed. lines_seen lets the caller say "saw N, none in window", which is
    the stale-log fingerprint.
    """
    text = container_logs(container, since_epoch, log_cmd)
    newest, seen = None, 0
    for line in text.splitlines():
        m = MLX_VERSION_RE.search(line)
        if not m:
            continue
        seen += 1
        t = SLOG_TIME_RE.search(line)
        if not t:
            continue
        try:
            when = datetime.datetime.fromisoformat(t.group(1)).timestamp()
        except ValueError:
            continue
        if when >= since_epoch:
            newest = m.group(1)
    return newest, seen


MLX_PAYLOAD_SO = "/usr/lib/ollama/mlx*/libmlx.so"
MLX_PAYLOAD_VERSION_RE = re.compile(r"\b(\d[\d.]*-\d+-g[0-9a-f]{7,40})\b")


def mlx_build_payload(container, path=MLX_PAYLOAD_SO, exec_cmd=None):
    """The MLX version string the SHIPPED payload carries, read from its own libmlx.so.

    The engine-init line stays the first source and this is only the fallback, because just the line names the MLX
    the running binary actually dlopen'd — the binary/payload skew case check_mlx_payload_pin was written for. But
    that line exists only if something loaded on the MLX path during the run, and a CUDA profile's preflight loads
    llama.cpp models. Measured 2026-09-12: pinning mlx_build on cuda-dynres-903 turned the check's SKIP into "no MLX
    engine-init line in the log window" and failed the whole run on a payload that was in fact correct.

    Reading the library is the move llama_cpp_build already makes for llama-server: the payload's own identity,
    rather than the version string of whatever is checked out. `grep -a` is the only tool these images carry.

    Returns the version string, or None when there is no MLX payload to read — a CPU or ROCm image legitimately has
    none, and that must read as "nothing to assert here", never as a pass.
    """
    pattern = "[0-9][0-9.]*-[0-9][0-9]*-g[0-9a-f][0-9a-f]*"
    cmd = ([exec_cmd.format(container=container)] if exec_cmd else
           ["docker", "exec", container, "sh", "-c",
            f'for f in {path}; do [ -f "$f" ] && grep -a -o -m1 "{pattern}" "$f" && break; done || true'])
    try:
        proc = subprocess.run(cmd, shell=bool(exec_cmd), capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    m = MLX_PAYLOAD_VERSION_RE.search((proc.stdout or "") + (proc.stderr or ""))
    return m.group(1) if m else None


def gpu_toolchain(container, exec_cmd=None):
    """The GPU toolchain the running payload was built against, read from the
    SONAMEs shipped beside it. Returns e.g. "rocm-7.2.4", "cuda-12.8", or None.

    This is the third identity a profile needs and the one nothing recorded.
    `llama_cpp_build` pins the payload source and `mlx_build` pins the MLX
    library, but a rebuild of an UNCHANGED payload against a different ROCm or
    CUDA reshuffles the math libraries underneath it -- rocBLAS and hipBLASLt
    select Tensile kernels per matrix shape, so a point release changes which
    kernel serves which GEMM. The ollama version string does not move, the
    llama.cpp SHA does not move, and every expectation in the profile is
    inherited on merit it has not earned. That is the b10091/b10353 accident
    with the compiler swapped for the payload.

    ROCm: rocBLAS carries it. `librocblas.so.5.2.70204` -> 70204 -> 7.2.4; the
    last four digits are minor and patch, whatever precedes them is the major,
    so 100000 reads as 10.0.0. Verified on hardware 2026-09-20 against two
    images known to differ only in ROCMVERSION (70201 vs 70204). libamd_comgr
    is NOT usable -- it reported 3.0.0 on both.

    CUDA: libcudart's SONAME is already the toolkit version, so it is read
    directly. NOT verified on a CUDA host -- no CUDA device on this estate. If
    it misreads there, fix the parser; do not widen a profile to accommodate it.
    """
    # A stamp beside the payload wins when present, because from ROCm 10 the
    # SONAMEs no longer carry the release version AT ALL. Verified on
    # rocm/dev-ubuntu-24.04:10.0.0-full 2026-09-20: librocblas.so.5.6 is the
    # rocBLAS library version, and no sibling encodes 10.0.0 either
    # (libhipblas.so.3.6, libhipblaslt.so.1.4, libhsa-runtime64.so.1.21.0).
    # TheRock decoupled component SONAMEs from the release on purpose; the
    # version survives only in the install prefix (/opt/rocm/core-10.0), which
    # is not copied into the payload. A number that is not in the artifact
    # cannot be parsed out of it, so the build stamps it instead.
    cmd = ([exec_cmd.format(container=container)] if exec_cmd else
           ["docker", "exec", container, "sh", "-c",
            "cat /usr/lib/ollama/rocm*/ROCM_VERSION /usr/lib/ollama/cuda*/CUDA_VERSION 2>/dev/null; "
            "ls /usr/lib/ollama/rocm*/librocblas.so.* "
            "/usr/lib/ollama/cuda*/libcudart.so.* 2>/dev/null || true"])
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=60, shell=bool(exec_cmd)).stdout
    except Exception:
        return None

    best = None
    for line in out.splitlines():
        line = line.strip()
        # A bare version on its own line is a stamp file's contents.
        m = re.fullmatch(r"(\d+\.\d+(?:\.\d+)?)", line)
        if m:
            return "rocm-%s" % m.group(1) if "rocm" in out else "cuda-%s" % m.group(1)
        m = re.search(r"librocblas\.so\.\d+\.\d+\.(\d{5,})$", line)
        if m:
            raw = m.group(1)
            major, minor, patch = raw[:-4], raw[-4:-2], raw[-2:]
            cand = "rocm-%d.%d.%d" % (int(major), int(minor), int(patch))
            best = cand if best is None else best
            continue
        m = re.search(r"libcudart\.so\.(\d+)\.(\d+)", line)
        if m:
            cand = "cuda-%s.%s" % (m.group(1), m.group(2))
            best = cand if best is None else best
    return best


def llama_cpp_build(container, path="/usr/lib/ollama/llama-server", exec_cmd=None):
    """The llama.cpp source SHA the *running payload* was compiled from, read
    from `llama-server --version` (e.g. "version: 1 (f8def7fe1)" -> "f8def7fe1").

    This is the payload's real identity. The ollama version string is not: two
    builds three releases apart both reported `0.32.5-dynres-*` while their
    payloads moved b10091 -> b10353, so an expectation set measured on one was
    silently applied to the other. Read from the container, never from the
    checkout's LLAMA_CPP_VERSION, which describes whatever is checked out now
    rather than what the server under test is running."""
    # Three routes, and the third exists because the first two both assume a
    # container: with neither an exec_cmd nor a container, ["docker", "exec",
    # None, ...] used to die on the None in argv rather than say what was
    # missing. Native hosts run the payload directly -- no shell, so a path
    # with spaces or braces needs no quoting and cannot be re-interpreted.
    if exec_cmd:
        cmd, shell = [exec_cmd.format(container=container)], True
    elif container:
        cmd, shell = ["docker", "exec", container, "sh", "-c",
                      f"{path} --version 2>&1 | head -2 || true"], False
    else:
        cmd, shell = [path, "--version"], False
    proc = subprocess.run(cmd, shell=shell, capture_output=True,
                          text=True, timeout=120)
    out = (proc.stdout or "") + (proc.stderr or "")
    # Two formats in the wild, because llama.cpp changed it between b10353 and
    # b10434 and the fork spans both:
    #   b10353: "version: 1 (f8def7fe1)"
    #   b10434: "version: 0.1.0-dev (build 1, commit 7e4c0a968)"
    # Matching only the first turns a payload bump into "could not read build
    # sha", which reads as a broken probe rather than a new format.
    m = (re.search(r"commit\s+([0-9a-f]{7,40})", out)
         or re.search(r"version:\s*\S+\s*\(([0-9a-f]{7,40})\)", out))
    if not m:
        raise ProbeError(f"no build sha from llama-server --version: {out[:300]!r}")
    return m.group(1)


def grep_binary_marker(container, path="/usr/bin/ollama", exec_cmd=None):
    """grep -c -- --image-max-tokens on the Go binary: 1 on a fork build, 0 on
    stock ollama/ollama. This is a GO-side marker only — it says nothing about
    whether the llama.cpp payload carries the compat patches. The payload proof
    is the model-load log, never a binary inspection."""
    cmd = ([exec_cmd.format(container=container)] if exec_cmd else
           ["docker", "exec", container, "sh", "-c",
            f"grep -c -- --image-max-tokens {path} || true"])
    proc = subprocess.run(cmd, shell=bool(exec_cmd), capture_output=True,
                          text=True, timeout=120)
    digits = re.findall(r"\d+", proc.stdout or "")
    if not digits:
        raise ProbeError(f"no count from grep: {(proc.stdout + proc.stderr)[:300]!r}")
    return int(digits[0])


# --------------------------------------------------------------------------
# The Metal tensor API — the M5 Neural Accelerators
# --------------------------------------------------------------------------
# Worth 2.14x prefill on the GGUF path (docs/maxusai/m5-neural-accelerators.md)
# and silent in every direction when it is lost. The three routes below are the
# three places the answer lives: the HOST (nax_probe), the PAYLOAD (the marker
# in the built llama-server) and the RUNNING SERVER (the discovery fallback
# this fork logs once at startup).

NAX_PROBE_SRC = os.path.join(DIR, "nax_probe.m")
TENSOR_MARKER = "GGML_METAL_HAS_TENSOR"
_NAX_BUILD = []          # [TemporaryDirectory, exe] — built once per process


def nax_probe(env=None, src=None, timeout=180):
    """Build nax_probe.m, run it, return its JSON verdict.

    The distinction this function exists to keep: a probe that CANNOT BE BUILT
    raises ProbeError — that is a gap in the harness (no clang, no Metal
    framework) and must never read as a verdict about the host. A probe that
    builds and answers no returns a dict with has_tensor False, which is a
    verdict and belongs in a FAIL.

    `env` overlays the process environment; a None value deletes a variable, so
    a caller can ask the counterfactual (GGML_METAL_TENSOR_DISABLE=1) without
    disturbing its own environment.
    """
    src = src or NAX_PROBE_SRC
    if not os.path.exists(src):
        raise ProbeError(f"nax_probe source is missing: {src}")
    if not _NAX_BUILD:
        tmp = tempfile.TemporaryDirectory(prefix="nax_probe.")
        exe = os.path.join(tmp.name, "nax_probe")
        cmd = ["xcrun", "clang", "-fobjc-arc", "-framework", "Foundation",
               "-framework", "Metal", src, "-o", exe]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ProbeError(f"cannot build nax_probe ({cmd[0]} {cmd[1]}): {exc}")
        if proc.returncode != 0 or not os.path.exists(exe):
            raise ProbeError("cannot build nax_probe: "
                             + ((proc.stderr or proc.stdout).strip()[:400] or "no output"))
        _NAX_BUILD.extend([tmp, exe])   # the TemporaryDirectory must outlive the call
    run_env = dict(os.environ)
    for k, v in (env or {}).items():
        if v is None:
            run_env.pop(k, None)
        else:
            run_env[k] = v
    try:
        proc = subprocess.run([_NAX_BUILD[1]], capture_output=True, text=True,
                              timeout=timeout, env=run_env)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProbeError(f"cannot run nax_probe: {exc}")
    try:
        return json.loads(proc.stdout)
    except ValueError:
        raise ProbeError("nax_probe emitted no JSON: "
                         + ((proc.stdout + proc.stderr).strip()[:300] or "no output"))


# routes.go logs this once per server start, and discovery runs AFTER it (the
# order is Listening -> "discovering available GPUs..."), which is what makes it
# usable as the boundary between this process's discovery and the last one's.
LISTEN_RE = re.compile(r'msg="Listening on (\S+) \(version[^"]*"')
# discover/runner.go, reached from llm.ShouldRetryWithMetalTensorDisabled. The
# retry sets GGML_METAL_TENSOR_DISABLE=1 for discovery AND, via
# recordPersistentRunnerEnv, for every runner this server process later spawns.
TENSOR_FALLBACK_RE = re.compile(
    r'^.*msg="retrying llama-server GPU discovery with Metal tensor API disabled".*$',
    re.M)


def parse_metal_tensor_discovery(text, port):
    """{'anchored': bool, 'window': str, 'fallback': str|None, 'ports': [...]}

    Only the CURRENT server process's discovery counts, and only the server
    under test's. serve.err.log on the Mac host is 25 MB spanning 69 restarts of
    a server that shares the file with nothing — but four ollama servers share
    the host, an operator names the log by hand, and the anchor taken as "the
    last Listening on line" would then attribute another server's discovery to
    the build under test. Worse, being the LAST one, it would put a fallback
    logged by the server under test OUTSIDE the window and report PASS. The
    anchor line carries the port; this matches on it.

    Without an anchor for `port` there is nothing to attribute to, so `anchored`
    is False and the caller must skip rather than guess. `ports` is what the log
    did contain, so the skip can say what it found instead.
    """
    anchors, ports = [], []
    for m in LISTEN_RE.finditer(text or ""):
        seen = m.group(1).rpartition(":")[2]
        if seen not in ports:
            ports.append(seen)
        if seen == str(port):
            anchors.append(m)
    if not anchors:
        return {"anchored": False, "window": "", "fallback": None, "ports": ports}
    window = text[anchors[-1].end():]
    m = TENSOR_FALLBACK_RE.search(window)
    return {"anchored": True, "window": window, "ports": ports,
            "fallback": m.group(0).strip() if m else None}


def metal_tensor_discovery(container, port, log_cmd=None):
    """parse_metal_tensor_discovery over the WHOLE log — deliberately unwindowed.

    Discovery happens at server start, which is before any preflight window
    opens; a `since` here would reliably return nothing. The anchor for this
    server's own port is what bounds it instead.
    """
    return parse_metal_tensor_discovery(container_logs(container, 0, log_cmd), port)


def launched_runner_paths(container, since_epoch, log_cmd=None):
    """The llama-server executables THIS window's runner launches ran, oldest
    first.

    THE WINDOW IS ENFORCED HERE, per line, for the reason mlx_build's docstring
    already records: `cat <serve log>` — the only --log-cmd form that works on
    the native macOS path — cannot substitute {since}, so container_logs returns
    the whole file, 69 restarts of it. Trusting it unfiltered means inspecting a
    binary some earlier run launched, since archived by the deploy flow, and
    labelling it "launched by this run".

    A launch whose timestamp will not parse is treated as OUTSIDE the window:
    cannot-confirm must never read as confirmed. A path containing a space would
    split wrongly — the Go side logs an unquoted command line, so there is
    nothing better to parse, and checks.py falls back to the listening
    executable when this finds nothing.
    """
    out = []
    for line in container_logs(container, since_epoch, log_cmd).splitlines():
        m = LAUNCH_RE.search(line)
        if not m:
            continue
        cmd = m.group(1).split()
        if not cmd:
            continue
        t = SLOG_TIME_RE.search(line)
        if not t:
            continue
        try:
            when = datetime.datetime.fromisoformat(t.group(1)).timestamp()
        except ValueError:
            continue
        if when >= since_epoch:
            out.append(cmd[0])
    return out


def local_listener_exe(port):
    """Absolute path of the executable listening on <port> on THIS machine.

    darwin-only by construction: `ps -o comm=` prints the full path there, while
    Linux truncates comm to 15 characters. Returns None rather than raising —
    every caller treats "cannot tell" as a skip.
    """
    try:
        pids = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    pid = (pids.stdout or "").split()
    # Two processes on one port is the kickstart window, when the outgoing and
    # incoming servers overlap. Taking the first would let a dying process
    # supply the binary for a verdict about the new one. Ambiguous is not an
    # answer; every caller treats None as a skip.
    if len(pid) != 1:
        return None
    try:
        ps = subprocess.run(["ps", "-p", pid[0], "-o", "comm="],
                            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    exe = (ps.stdout or "").strip()
    return exe if exe.startswith("/") else None


# The two variables that decide the gate from outside ggml. A server started by
# launchd with DISABLE=1 runs without the accelerators and says nothing about
# it anywhere — no log line, no API field — so the only way to see it is to read
# the environment of the process that is actually serving.
TENSOR_ENV_VARS = ("GGML_METAL_TENSOR_DISABLE", "GGML_METAL_TENSOR_ENABLE")


def server_env(port):
    """The environment of the process listening on <port>, or None when macOS
    will not show it.

    `ps -wwE` prints argv and the environment run together with no delimiter, so
    the environment is recovered as the SUFFIX after the same process's plain
    `command=`. Harvesting every =-bearing token instead reads argv as
    environment — `sh -c 'OLLAMA_HOST=... ollama serve'` is enough — and that is
    worse than failing: it makes an unreadable environment look readable, so the
    caller confidently DELETES the variables it was supposed to honour.

    None and {} are NOT the same answer and the caller must not collapse them.
    The kernel hides the environment of PLATFORM binaries (`codesign -dv` prints
    "Platform identifier"), which is why `ps -wwE` on /bin/sleep shows nothing
    while an ollama server — an ordinary ad-hoc signed binary — shows all of it.
    Measured on macOS 26.6.2, both cases.
    """
    if not local_listener_exe(port):
        return None
    try:
        pids = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                              capture_output=True, text=True, timeout=30)
        pid = (pids.stdout or "").split()
        if len(pid) != 1:          # same reason as local_listener_exe
            return None
        argv = subprocess.run(["ps", "-p", pid[0], "-ww", "-o", "command="],
                              capture_output=True, text=True, errors="replace",
                              timeout=30).stdout.strip()
        both = subprocess.run(["ps", "-p", pid[0], "-wwE", "-o", "command="],
                              capture_output=True, text=True, errors="replace",
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not argv or not both.startswith(argv):
        return None                     # raced, or truncated differently
    env = {}
    for token in both[len(argv):].split():
        if "=" in token:
            k, v = token.split("=", 1)
            env.setdefault(k, v)
    return env or None


def lib_ollama_llama_server(exe):
    """The llama-server `exe` would spawn, or None.

    Mirrors ml/path.go libOllamaPathCandidates() for darwin, INCLUDING where it
    stops: libOllamaPathExists() is os.Stat().IsDir(), so ollama takes the first
    candidate DIRECTORY that exists and looks no further. Walking past an empty
    lib/ollama to find some other llama-server would report on a binary ollama
    would never load. The returned path may therefore not exist, and the caller
    is expected to surface that rather than read it as "no tensor kernels".

    EvalSymlinks first, as path.go does: the deploy flow archives binaries and
    swaps them by name, so searching beside the symlink is searching the wrong
    directory. The two candidates path.go derives from the server's working
    directory are omitted, because the harness cannot know it.
    """
    if not exe:
        return None
    d = os.path.dirname(os.path.realpath(exe))
    for cand in (os.path.join(d, "lib", "ollama"),
                 os.path.join(d, "..", "lib", "ollama"),
                 os.path.join(d, "build", "lib", "ollama"),
                 os.path.join(d, "dist", "darwin-arm64", "lib", "ollama"),
                 os.path.join(d, "dist", "darwin"),
                 d):
        cand = os.path.normpath(cand)
        if os.path.isdir(cand):
            return os.path.join(cand, "llama-server")
    return None


def binary_marker_count(path, needle):
    """`strings -a <path> | grep -c <needle>` — how many strings in a compiled
    artefact contain <needle>.

    Raises ProbeError when the artefact cannot be read at all. `grep -c` prints
    0 on empty input and the `|| true` swallows the exit status, so a missing
    file, a missing `strings`, or a path the deploy flow has since archived
    would otherwise all count as zero — reported as "this payload has no tensor
    kernels", a FAIL against a healthy build, and the opposite of what the host
    check does with a missing toolchain.

    Same caveat as grep_binary_marker: this is a BUILD-side fact. It says the
    payload carries the kernels, never that the host will run them.
    """
    if not os.path.exists(path):
        raise ProbeError(f"no llama-server at {path}")
    cmd = ("strings -a " + shlex.quote(path) + " | grep -c -- "
           + shlex.quote(needle) + " || true")
    proc = subprocess.run(["sh", "-c", cmd], capture_output=True, text=True,
                          errors="replace", timeout=300)
    if proc.stderr.strip():
        raise ProbeError(f"strings failed on {path}: {proc.stderr.strip()[:200]}")
    digits = re.findall(r"\d+", proc.stdout or "")
    if not digits:
        raise ProbeError(f"no count from strings|grep: {(proc.stdout)[:200]!r}")
    return int(digits[0])
