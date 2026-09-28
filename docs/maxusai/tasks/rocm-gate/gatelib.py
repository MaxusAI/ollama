"""Shared plumbing for the ROCm gate reproductions (docs/maxusai/tasks/rocm-gate-issues-17459-17475.md).

Scratch container on :11435 only. Production (ollama-rocm, :11434) is never touched,
only checked for resident models before each run so contention is a recorded fact.
"""
import json
import os
import subprocess
import time
import urllib.error
import urllib.request

def _need(name):
    raise SystemExit(f"{name} must be set (see the comment above)")


PORT = 11435
HOST = f"http://127.0.0.1:{PORT}"
NAME = "ollama-gate"
# Host-specific, so it is supplied by the environment rather than committed:
#   GATE_STORE       the model store to bind-mount (production's, read-write)
#   GATE_GPU_GROUPS  comma-separated GIDs for /dev/kfd and /dev/dri on this host
STORE = os.environ.get("GATE_STORE") or _need("GATE_STORE")
GPU_GROUPS = " ".join(f"--group-add {g}" for g in
                      filter(None, os.environ.get("GATE_GPU_GROUPS", "").split(",")))
OUT = os.path.dirname(os.path.abspath(__file__))

# Production's own environment. The KV cache is f16 since 2026-09-26 (amd-upgrade-gate.md, the 2026-09-26
# decision); the 2026-09-18 results in rocm-gate-issues-result.md ran with q8_0, which production had then.
PROD_ENV = {
    "OLLAMA_FLASH_ATTENTION": "1",
    "OLLAMA_KV_CACHE_TYPE": "f16",
    "OLLAMA_NUM_PARALLEL": "2",
}

# #17475's reporter ran stock defaults with a single slot.
REPORTER_ENV = {
    "OLLAMA_NUM_PARALLEL": "1",
}

# Always present. OLLAMA_NOPRUNE matters most: a server that cannot parse a newer
# manifest could treat that model's blobs as unreferenced and delete them from the
# production store this container shares.
BASE_ENV = {
    "OLLAMA_HOST": "0.0.0.0:11434",
    "OLLAMA_MODELS": "/root/.ollama/models",
    "ROCR_VISIBLE_DEVICES": "0",
    "OLLAMA_DEBUG": "1",
    "OLLAMA_NOPRUNE": "1",
}


def sh(cmd, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"{cmd}\n{r.stderr}")
    return r.stdout.strip()


def api(path, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(HOST + path, data,
                                 {"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def production_resident():
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=5) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception as e:
        return [f"unreachable: {e}"]


def start(image, env, label):
    sh(f"docker rm -f {NAME}", check=False)
    full = dict(BASE_ENV, **env)
    envs = " ".join(f"-e {k}={v}" for k, v in full.items())
    sh(f"docker run -d --name {NAME} --restart no "
       f"--device /dev/kfd --device /dev/dri {GPU_GROUPS} "
       f"--security-opt seccomp=unconfined --cap-add SYS_PTRACE --ipc host --shm-size 16g "
       f"--log-opt max-size=500m --log-opt max-file=2 "
       f"-v {STORE}:/root/.ollama -p {PORT}:11434 {envs} {image}")
    for _ in range(90):
        try:
            v = api("/api/version", timeout=3)["version"]
            break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError(f"{image} did not come up on :{PORT}")
    meta = {"label": label, "image": image, "server_version": v, "env": full,
            "image_id": sh(f"docker image inspect -f '{{{{.Id}}}}' {image}")[:19],
            "production_resident_at_start": production_resident(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(f"== {label}: {image} -> {v}", flush=True)
    return meta


def stop(label):
    path = os.path.join(OUT, f"serverlog_{label}.txt")
    sh(f"docker logs {NAME} > {path} 2>&1", check=False)
    sh(f"docker rm -f {NAME}", check=False)
    return path


def offload_evidence(logpath):
    """ADR 0024: observed, not inferred. Quote the lines that prove GPU use."""
    keys = ("offloaded", "using device ROCm", "ROCm0 model buffer", "load_backend",
            "--load-mode", "--direct-io", "starting llama-server", "starting llama server")
    out = []
    with open(logpath, errors="replace") as fh:
        for line in fh:
            if any(k in line for k in keys):
                out.append(line.rstrip()[:400])
    # keep it readable: first occurrence of each kind, plus every offload line
    seen, keep = set(), []
    for l in out:
        k = next(k for k in keys if k in l)
        if k == "offloaded" or k not in seen:
            keep.append(l)
            seen.add(k)
    return keep[:40]


def unload(model):
    try:
        api("/api/generate", {"model": model, "keep_alive": 0}, timeout=120)
    except Exception:
        pass
    for _ in range(30):
        if model not in [m["name"] for m in api("/api/ps").get("models", [])]:
            return True
        time.sleep(1)
    return False


def save(name, obj):
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(obj, fh, indent=1)
