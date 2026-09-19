#!/usr/bin/env python3
"""Does the local ollama store still hold what the library serves under that tag?

    python3 store_audit.py                      # every tag in the default store
    python3 store_audit.py gemma4 qwen3.5       # only tags matching these substrings
    python3 store_audit.py --store /var/lib/... # a store somewhere else
    python3 store_audit.py --digests            # tag -> local manifest digest, for citing in a record

A published tag is not a fixed artifact. `gemma4:31b-nvfp4` was re-published with a
bf16 vision tower while our copy keeps the nvfp4 one: same name, same config blob,
1248 layers of which 194 are different weights. `ollama show` reports neither the
layer set nor a content digest, so nothing surfaces the swap — the first symptom is a
measurement that will not reproduce, on a host that pulled later.

This walks the store's manifests, asks the registry for the same tags, and prints the
ones that moved, with the tensor groups affected and the local->registry size ratio
(~3.55x is nvfp4 -> bf16, ~2x is fp8 -> bf16). Tags the registry does not serve under
that name are listed separately: those are local builds, and nothing upstream can
change them.

Cite the local manifest digest (`--digests`) in any record that names a model, so the
measurement identifies its artifact and not just a mutable label.
"""
import argparse
import collections
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_REGISTRY = "https://registry.ollama.ai"
MANIFEST_ACCEPT = "application/vnd.docker.distribution.manifest.v2+json"

# Tensor-name fragments that put a layer on the vision (or audio) path rather than the
# language model. MLX-format models carry one layer per tensor, so the split is exact;
# GGUF models are a single `model` layer and only ever compare whole.
VISION_HINTS = ("vision_tower", "vision_embedder", "embed_vision", "multi_modal",
                "vision_model", "mm.", "v.")
AUDIO_HINTS = ("embed_audio", "audio_tower", "a.")


def default_store():
    return os.environ.get("OLLAMA_MODELS") or os.path.expanduser("~/.ollama/models")


def classify(name):
    """Which part of the model a tensor layer belongs to."""
    if not name:
        return "blob"
    if any(h in name for h in VISION_HINTS):
        return "vision"
    if any(h in name for h in AUDIO_HINTS):
        return "audio"
    if "language_model" in name or name.startswith("model.") or name.startswith("blk."):
        return "language"
    return "other"


def manifest_digest(raw):
    """The digest ollama would address this manifest by: sha256 over its bytes."""
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def read_store(store, namespace="registry.ollama.ai/library"):
    """Every tag under the store's manifest namespace: {"model:tag": (manifest, digest)}."""
    root = os.path.join(store, "manifests", *namespace.split("/"))
    out = {}
    if not os.path.isdir(root):
        return out
    for model in sorted(os.listdir(root)):
        d = os.path.join(root, model)
        if not os.path.isdir(d):
            continue
        for tag in sorted(os.listdir(d)):
            path = os.path.join(d, tag)
            if not os.path.isfile(path):
                continue
            with open(path, "rb") as f:
                raw = f.read()
            try:
                out[f"{model}:{tag}"] = (json.loads(raw), manifest_digest(raw))
            except ValueError:
                continue
    return out


def fetch(model, tag, registry=DEFAULT_REGISTRY, timeout=30):
    """The manifest the registry serves for this tag right now, plus its content digest."""
    req = urllib.request.Request(f"{registry}/v2/library/{model}/manifests/{tag}",
                                 headers={"Accept": MANIFEST_ACCEPT,
                                          "User-Agent": "maxusai-store-audit/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw), r.headers.get("Docker-Content-Digest") or manifest_digest(raw)


def compare(local, remote):
    """What moved between two manifests of the same tag.

    Returns moved-layer count, the tensor groups they fall in, whether the config blob
    is unchanged (which is what makes a swap invisible to `ollama show`), and the mean
    remote/local size ratio over layers matched by tensor name.
    """
    rem = {l["digest"] for l in remote["layers"]}
    rem_by_name = {l.get("name"): l["size"] for l in remote["layers"] if l.get("name")}
    moved = [l for l in local["layers"] if l["digest"] not in rem]
    groups = collections.Counter(classify(l.get("name")) for l in moved)
    ratios = [rem_by_name[l["name"]] / l["size"] for l in moved
              if l.get("name") in rem_by_name and l.get("size")]
    return {
        "moved": len(moved),
        "total": len(local["layers"]),
        "groups": dict(groups),
        "config_same": local.get("config", {}).get("digest") == remote.get("config", {}).get("digest"),
        "ratio": round(sum(ratios) / len(ratios), 2) if ratios else None,
        "names": [l.get("name") for l in moved if l.get("name")][:5],
    }


def dtype_reading(ratio):
    """What a size ratio implies, for the common quantisation moves. None when unclear."""
    if ratio is None:
        return ""
    for lo, hi, text in ((3.3, 3.8, "nvfp4 → bf16"), (1.8, 2.2, "fp8 → bf16"),
                         (0.26, 0.31, "bf16 → nvfp4"), (0.45, 0.55, "bf16 → fp8")):
        if lo <= ratio <= hi:
            return text
    return ""


def render(rows):
    """Markdown for the tags that moved. `rows` is [(tag, local_digest, remote_digest, compare())]."""
    out = ["| tag | local manifest | registry manifest | layers changed | where | size ratio | reading |",
           "|---|---|---|---|---|---|---|"]
    for tag, ldig, rdig, c in rows:
        where = ", ".join(f"{k} {v}" for k, v in sorted(c["groups"].items(), key=lambda kv: -kv[1]))
        reading = dtype_reading(c["ratio"])
        out.append(f"| `{tag}` | `{ldig[7:19]}` | `{rdig[7:19]}` | {c['moved']} / {c['total']} | "
                   f"{where} | {c['ratio'] or '—'}× | {reading or '—'} |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("match", nargs="*", help="only tags containing one of these substrings")
    ap.add_argument("--store", default=default_store())
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    ap.add_argument("--digests", action="store_true", help="print tag -> local manifest digest and stop")
    a = ap.parse_args(argv)

    store = read_store(a.store)
    if not store:
        sys.exit(f"no manifests under {a.store}")
    picked = {t: v for t, v in store.items() if not a.match or any(m in t for m in a.match)}

    if a.digests:
        for tag, (_, dig) in sorted(picked.items()):
            print(f"{tag:<50} {dig}")
        return 0

    moved, identical, local_only, failed = [], [], [], []
    for tag, (loc, ldig) in sorted(picked.items()):
        model, _, name = tag.partition(":")
        try:
            rem, rdig = fetch(model, name, a.registry)
        except urllib.error.HTTPError as e:
            (local_only if e.code == 404 else failed).append((tag, e.code))
            continue
        except Exception as e:                                        # noqa: BLE001
            failed.append((tag, str(e)))
            continue
        c = compare(loc, rem)
        (moved if c["moved"] else identical).append((tag, ldig, rdig, c))

    if moved:
        print(f"## Tags whose weights changed under the same name ({len(moved)})\n")
        print(render(moved))
        print()
        for tag, _, _, c in moved:
            if c["names"]:
                print(f"- `{tag}` first moved tensors: " + ", ".join(f"`{n}`" for n in c["names"]))
        print()
    print(f"{len(identical)} tag(s) identical to the registry, {len(moved)} changed, "
          f"{len(local_only)} not served under that name (local builds), {len(failed)} unreadable.")
    if local_only:
        print("\nLocal only: " + ", ".join(f"`{t}`" for t, _ in local_only))
    if failed:
        print("\nUnreadable: " + ", ".join(f"`{t}` ({e})" for t, e in failed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
