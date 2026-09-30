#!/usr/bin/env python3
"""Which machine did this run on?

    python3 docs/maxusai/tools/host_profile.py              # Markdown table on stdout
    python3 docs/maxusai/tools/host_profile.py --json       # the profile as JSON
    python3 docs/maxusai/tools/host_profile.py --flat       # {"dotted.key": value}, one per fact
    python3 docs/maxusai/tools/host_profile.py --out DIR    # DIR/host-profile.json and .md
        [--label HOST/SURFACE] [--fs ROLE=PATH ...] [--ollama URL]

Records what a benchmark result needs beside it to be compared with one from another host: the
host's collab label, the cloud machine when there is one (provider, zone, region, machine type,
CPU platform), the OS, CPU, memory, each GPU with its PCIe link, the filesystems named with --fs,
and the driver, toolkit and Ollama versions. Commit it beside the run it describes, e.g.
vision-suite/bench-runs/<run>/host-profile.json. Two profiles compare field by field:

    diff <(jq -S . A/host-profile.json) <(jq -S . B/host-profile.json)

WHY. A throughput or load-time number means little without the machine under it, and the machine
is rarely what its name suggests. Measured 2026-09-30 on a GCP a3-highgpu-1g VM: its H100 SXM5
supports PCIe Gen5 x16, but the port above it, as the VM sees it, tops out at Gen4, so the link
runs at Gen4 x16 -- half the host bandwidth, and nothing in the machine type or the GPU's product
name says so. An APU's GPU memory is whatever the firmware carved out, plus GTT; a Mac's is its
RAM. Runs under bench-runs/ name their host in prose, when at all, so they cannot be compared on
these facts, let alone filtered by them.

ATOMIC FIELDS. Every value is one fact of one type, so profiles can be filtered and joined:
numbers carry their unit in the key (memory_total_mib, l3_cache_mib, power_limit_w,
max_core_clock_mhz), states are booleans (ecc_enabled, cpu.flags.amx_bf16), categories are
lower-case words (vendor, provider, family), versions are strings without distribution
suffixes. Filter a set of profiles with jq:

    jq -s 'map(select(any(.gpus[]; .pcie_link_gen < 5)) | .label)' */host-profile.json

--flat prints the same profile as one "dotted.key": value per fact -- a row for a table.

EVERY FIELD IS OPTIONAL. A probe with nothing to report -- no GPU, no cloud, a tool that is not
installed -- leaves its field null, and the Markdown leaves the row out. Nothing needs root: PCIe
link speeds come from sysfs, not `lspci -vv`.

WHAT IT READS. Linux: lscpu, /proc, DMI, the PCI display devices in sysfs (vendor, link speed
and width of the device and of the port above it, NUMA node), nvidia-smi for NVIDIA GPUs,
amdgpu's sysfs memory files and the KFD topology for AMD GPUs, `lspci -mm` for device names.
macOS: sysctl and system_profiler. Clouds, recognised from DMI: GCP (then its metadata server)
and AWS. Ollama: /api/version.

NO IDENTIFIERS. This fork is public (AGENTS.md, "Name roles, never people") and a profile is
made to be committed. It never records a hostname, a user or account name, a cloud project,
instance name or id, a physical-host id, a serial number, a GPU UUID, a filesystem path or a
device name. The host is named by its collab label (~/.config/collab/identity, or --label),
which exists to be written down; --fs records the filesystem under the ROLE you give it, never
the PATH.
"""
import argparse
import datetime
import json
import os
import platform
import re
import subprocess
import sys
import urllib.request

SCHEMA = "host-profile/1"
IDENTITY_FILE = os.path.expanduser("~/.config/collab/identity")
GT_S_TO_GEN = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}
PCI_GPU_VENDORS = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}
CLOUD_VENDORS = {"Google": "gcp", "Amazon EC2": "aws"}   # DMI sys_vendor
# Values firmware leaves in DMI fields it was never given.
DMI_PLACEHOLDERS = {"", "default string", "to be filled by o.e.m.", "system product name",
                    "system manufacturer", "not specified", "not applicable", "none", "o.e.m."}
X86_FLAGS = ("avx2", "avx512f", "avx512_vnni", "avx512_bf16", "avx512_fp16",
             "amx_tile", "amx_bf16", "amx_int8")
ARM_FLAGS = ("asimd", "sve", "sve2", "bf16", "i8mm", "sme", "sme2")
DARWIN_ARM_FLAGS = {"bf16": "FEAT_BF16", "i8mm": "FEAT_I8MM", "sme": "FEAT_SME", "sme2": "FEAT_SME2"}
NVIDIA_FIELDS = ("pci.bus_id", "name", "memory.total", "compute_cap", "vbios_version",
                 "power.limit", "power.max_limit", "clocks.max.sm", "clocks.max.memory",
                 "ecc.mode.current", "mig.mode.current", "persistence_mode")


# ---------------------------------------------------------------- small readers

def run(*cmd, timeout=30):
    """A command's stdout, or None when it is missing, fails or hangs."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout.strip() if p.returncode == 0 else None


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def http(url, headers=None, timeout=2):
    try:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode().strip()
    except (OSError, ValueError):
        return None


def match(pattern, text):
    m = re.search(pattern, text or "")
    return m.group(1) if m else None


# ---------------------------------------------------------------- value parsers

def num(text):
    """'81559' -> 81559, '700.00' -> 700.0; anything else ('[N/A]', None) -> None."""
    if text is None:
        return None
    text = str(text).strip()
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return None


def text_value(text):
    """A string with nvidia-smi's and DMI's 'nothing here' spellings mapped to None."""
    if text is None:
        return None
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):     # [N/A], [Not Supported]
        return None
    return None if text.lower() in DMI_PLACEHOLDERS | {"n/a"} else text


def on_off(text):
    """nvidia-smi's Enabled/Disabled as a boolean, None when it has no answer."""
    t = (text or "").strip().lower()
    if t in ("enabled", "on", "yes", "true", "1"):
        return True
    if t in ("disabled", "off", "no", "false", "0"):
        return False
    return None


def size_mib(text):
    """'105 MiB (1 instance)' -> 105.0, '1.5 GiB' -> 1536.0, '512 KiB' -> 0.5."""
    m = re.match(r"\s*([\d.]+)\s*([KMGT])i?B", text or "")
    if not m:
        return None
    return float(m.group(1)) * {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 ** 2}[m.group(2)]


def bytes_to_mib(text):
    b = num(text)
    return b // 2 ** 20 if isinstance(b, int) and b > 0 else None


def link_gen(speed):
    """sysfs link speed '16.0 GT/s PCIe' -> PCIe generation 4; 'Unknown' -> None."""
    m = re.match(r"\s*([\d.]+)\s*GT/s", speed or "")
    return GT_S_TO_GEN.get(float(m.group(1))) if m else None


def gfx_target(version):
    """KFD gfx_target_version 110501 -> 'gfx1151', 90010 -> 'gfx90a'; 0 (a CPU node) -> None."""
    v = num(version)
    if not isinstance(v, int) or v <= 0:
        return None
    return f"gfx{v // 10000}{(v // 100) % 100:x}{v % 100:x}"


def upstream_version(version):
    """A Debian version without its revision: '2.32.3-1+cuda13.4' -> '2.32.3'."""
    return version.rsplit("-", 1)[0] if version and "-" in version else version


def pci_address(bus_id):
    """nvidia-smi's '00000000:04:00.0' -> sysfs's '0000:04:00.0'."""
    domain, rest = bus_id.split(":", 1)
    return f"{int(domain, 16):04x}:{rest}".lower()


def lesser(*values):
    known = [v for v in values if v is not None]
    return min(known) if known else None


# ---------------------------------------------------------------- the host

def collab_label(explicit=None, path=IDENTITY_FILE):
    """--label, else the first line of the collab identity file."""
    if explicit:
        return explicit.strip()
    for line in (read(path) or "").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return None


def dmi(root="/sys/class/dmi/id"):
    return {k: text_value(read(os.path.join(root, k)))
            for k in ("sys_vendor", "product_name", "board_vendor", "board_name")}


def cloud_from(dmi_info, metadata=None):
    """The cloud machine, recognised from DMI; None on anything else.

    Asks GCP's metadata server only for machine facts. Project, instance name and id, and the
    physical host are there too, and are deliberately never requested (NO IDENTIFIERS)."""
    provider = CLOUD_VENDORS.get(dmi_info.get("sys_vendor"))
    if not provider:
        return None
    cloud = {"provider": provider, "zone": None, "region": None, "machine_type": None,
             "cpu_platform": None, "image": None, "on_host_maintenance": None,
             "automatic_restart": None, "preemptible": None}
    if provider == "aws":           # Nitro instances put the instance type in DMI
        cloud["machine_type"] = dmi_info.get("product_name")
        return cloud
    meta = metadata or (lambda path: http(
        "http://metadata.google.internal/computeMetadata/v1/" + path, {"Metadata-Flavor": "Google"}))
    base = lambda v: v.rsplit("/", 1)[-1] if v else None
    zone = base(meta("instance/zone"))
    try:
        sched = json.loads(meta("instance/scheduling/?recursive=true") or "{}")
    except ValueError:
        sched = {}
    cloud.update(zone=zone, region=zone.rsplit("-", 1)[0] if zone else None,
                 machine_type=base(meta("instance/machine-type")),
                 cpu_platform=meta("instance/cpu-platform"), image=base(meta("instance/image")),
                 on_host_maintenance=sched.get("onHostMaintenance"),
                 automatic_restart=on_off(sched.get("automaticRestart")),
                 preemptible=on_off(sched.get("preemptible")))
    return cloud


def os_release(text):
    fields = dict(re.findall(r'^(\w+)="?([^"\n]*)"?$', text or "", re.M))
    return fields.get("NAME"), fields.get("VERSION_ID"), fields.get("PRETTY_NAME")


def os_info():
    family = platform.system().lower()
    if family == "darwin":
        version = run("sw_vers", "-productVersion")
        name, pretty = "macOS", f"macOS {version}" if version else "macOS"
    else:
        name, version, pretty = os_release(read("/etc/os-release"))
    arch = platform.machine()
    return {"family": family, "name": name, "version": version, "pretty_name": pretty,
            "kernel": platform.release(), "arch": {"aarch64": "arm64"}.get(arch, arch)}


def lscpu_fields(text):
    """lscpu -J, flat (util-linux >= 2.38 by default) or hierarchical, as {field: data}."""
    fields = {}

    def walk(entries):
        for e in entries:
            fields[e["field"].rstrip(":").strip()] = e.get("data")
            walk(e.get("children") or [])
    walk(json.loads(text)["lscpu"])
    return fields


def cpu_features(cpuinfo_text):
    """The flags this profile tracks, each True or False; None when /proc/cpuinfo has none."""
    for line in (cpuinfo_text or "").splitlines():
        key, _, value = line.partition(":")
        if key.strip() in ("flags", "Features"):
            have = set(value.split())
            wanted = X86_FLAGS if key.strip() == "flags" else ARM_FLAGS
            return {f: f in have for f in wanted}
    return None


def cpu_linux(lscpu_text, cpuinfo_text):
    f = lscpu_fields(lscpu_text) if lscpu_text else {}
    sockets = num(f.get("Socket(s)"))
    per_socket = num(f.get("Core(s) per socket")) or num(f.get("Core(s) per cluster"))
    vendor = f.get("Vendor ID")
    return {
        "vendor": {"GenuineIntel": "intel", "AuthenticAMD": "amd"}.get(vendor, (vendor or "").lower() or None),
        "model": f.get("Model name"),
        "logical_cpus": num(f.get("CPU(s)")),
        "sockets": sockets,
        "cores": sockets * per_socket if sockets and per_socket else None,
        "threads_per_core": num(f.get("Thread(s) per core")),
        "performance_cores": None,
        "efficiency_cores": None,
        "numa_nodes": num(f.get("NUMA node(s)")),
        "l3_cache_mib": size_mib(f.get("L3") or f.get("L3 cache")),
        "max_mhz": num(f.get("CPU max MHz")),
        "flags": cpu_features(cpuinfo_text),
    }, f.get("Hypervisor vendor")


def cpu_darwin(sysctl):
    brand = sysctl("machdep.cpu.brand_string")
    logical, physical = num(sysctl("hw.logicalcpu")), num(sysctl("hw.physicalcpu"))
    apple = bool(brand and brand.startswith("Apple"))
    return {
        "vendor": "apple" if apple else ((brand or "").split("(")[0].strip().lower() or None),
        "model": brand,
        "logical_cpus": logical,
        "sockets": 1 if brand else None,
        "cores": physical,
        "threads_per_core": logical // physical if logical and physical else None,
        "performance_cores": num(sysctl("hw.perflevel0.physicalcpu")) if apple else None,
        "efficiency_cores": num(sysctl("hw.perflevel1.physicalcpu")) if apple else None,
        "numa_nodes": None,
        "l3_cache_mib": None,
        "max_mhz": None,
        "flags": {k: sysctl(f"hw.optional.arm.{v}") == "1" for k, v in DARWIN_ARM_FLAGS.items()} if apple else None,
    }


# ---------------------------------------------------------------- GPUs

def empty_gpu(vendor):
    return {"vendor": vendor, "name": None, "pci_name": None, "pci_bus_id": None, "pci_vendor_id": None,
            "pci_device_id": None, "driver": None, "memory_total_mib": None, "memory_gtt_mib": None,
            "unified_memory": None, "compute_capability": None, "gfx_target": None, "gpu_cores": None,
            "metal_family": None, "vbios": None, "power_limit_w": None, "power_max_limit_w": None,
            "max_core_clock_mhz": None, "max_memory_clock_mhz": None, "ecc_enabled": None,
            "mig_enabled": None, "persistence_enabled": None, "pcie_link_gen": None, "pcie_link_width": None,
            "pcie_device_max_gen": None, "pcie_device_max_width": None, "pcie_upstream_max_gen": None,
            "pcie_upstream_max_width": None, "pcie_gen_now": None, "pcie_width_now": None,
            "numa_node": None, "local_cpus": None}


def pci_gpus(sysfs="/sys"):
    """NVIDIA, AMD and Intel display devices from sysfs, with their PCIe links.

    pcie_link_gen/width is what the link can train to: the lesser of the device's own maximum and
    that of the port above it. pcie_gen_now drops while a GPU idles, so compare the maximum."""
    base = os.path.join(sysfs, "bus/pci/devices")
    gpus = []
    for addr in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        dev = os.path.join(base, addr)
        vendor = PCI_GPU_VENDORS.get(read(os.path.join(dev, "vendor")))
        if not vendor or not (read(os.path.join(dev, "class")) or "").startswith("0x03"):
            continue
        up = os.path.dirname(os.path.realpath(dev))      # the bridge or root port above it
        g = empty_gpu(vendor)
        d_gen, d_w = link_gen(read(dev + "/max_link_speed")), num(read(dev + "/max_link_width"))
        u_gen, u_w = link_gen(read(up + "/max_link_speed")), num(read(up + "/max_link_width"))
        numa = num(read(dev + "/numa_node"))
        g.update(pci_bus_id=addr, pci_vendor_id=read(dev + "/vendor"), pci_device_id=read(dev + "/device"),
                 driver=os.path.basename(os.path.realpath(dev + "/driver")) if os.path.exists(dev + "/driver") else None,
                 memory_total_mib=bytes_to_mib(read(dev + "/mem_info_vram_total")),
                 memory_gtt_mib=bytes_to_mib(read(dev + "/mem_info_gtt_total")),
                 name=text_value(read(dev + "/product_name")),
                 pcie_link_gen=lesser(d_gen, u_gen), pcie_link_width=lesser(d_w, u_w),
                 pcie_device_max_gen=d_gen, pcie_device_max_width=d_w,
                 pcie_upstream_max_gen=u_gen, pcie_upstream_max_width=u_w,
                 pcie_gen_now=link_gen(read(dev + "/current_link_speed")),
                 pcie_width_now=num(read(dev + "/current_link_width")),
                 numa_node=numa if numa is not None and numa >= 0 else None,
                 local_cpus=read(dev + "/local_cpulist"))
        gpus.append(g)
    return gpus


def kfd_gfx_targets(sysfs="/sys"):
    """PCI address -> gfx target from amdgpu's KFD topology; needs no ROCm userspace."""
    base = os.path.join(sysfs, "class/kfd/kfd/topology/nodes")
    targets = {}
    for node in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        props = {}
        for line in (read(os.path.join(base, node, "properties")) or "").splitlines():
            parts = line.split()
            if len(parts) == 2:
                props[parts[0]] = parts[1]
        gfx, loc = gfx_target(props.get("gfx_target_version")), num(props.get("location_id"))
        if gfx and isinstance(loc, int):
            domain = num(props.get("domain")) or 0
            targets[f"{domain:04x}:{loc >> 8:02x}:{(loc & 0xff) >> 3:02x}.{loc & 7}"] = gfx
    return targets


def nvidia_smi_rows(text):
    """nvidia-smi --query-gpu=<NVIDIA_FIELDS> --format=csv,noheader,nounits -> {pci address: fields}."""
    rows = {}
    for line in (text or "").splitlines():
        v = [x.strip() for x in line.split(",")]
        if len(v) != len(NVIDIA_FIELDS):
            continue
        r = dict(zip(NVIDIA_FIELDS, v))
        rows[pci_address(r["pci.bus_id"])] = {
            "name": text_value(r["name"]),
            "memory_total_mib": num(r["memory.total"]),
            "compute_capability": text_value(r["compute_cap"]),
            "vbios": text_value(r["vbios_version"]),
            "power_limit_w": num(r["power.limit"]),
            "power_max_limit_w": num(r["power.max_limit"]),
            "max_core_clock_mhz": num(r["clocks.max.sm"]),
            "max_memory_clock_mhz": num(r["clocks.max.memory"]),
            "ecc_enabled": on_off(r["ecc.mode.current"]),
            "mig_enabled": on_off(r["mig.mode.current"]),
            "persistence_enabled": on_off(r["persistence_mode"]),
        }
    return rows


def lspci_names(text):
    """lspci -mm -D -> {pci address: device name}."""
    names = {}
    for line in (text or "").splitlines():
        quoted = re.findall(r'"([^"]*)"', line)
        if line.split() and len(quoted) >= 3:
            names[line.split()[0].lower()] = quoted[2]
    return names


def gpus_linux(sysfs="/sys"):
    gpus = pci_gpus(sysfs)
    if not gpus:
        return gpus
    names = lspci_names(run("lspci", "-mm", "-D"))
    nvidia = nvidia_smi_rows(run("nvidia-smi", "--query-gpu=" + ",".join(NVIDIA_FIELDS),
                                 "--format=csv,noheader,nounits"))
    gfx = kfd_gfx_targets(sysfs)
    for g in gpus:
        g["pci_name"] = names.get(g["pci_bus_id"])
        g.update({k: v for k, v in nvidia.get(g["pci_bus_id"], {}).items() if v is not None})
        g["gfx_target"] = gfx.get(g["pci_bus_id"])
        g["name"] = g["name"] or g["pci_name"]
    return gpus


def darwin_hardware(system_profiler_json):
    """system_profiler SPHardwareDataType SPDisplaysDataType -json -> (machine, gpus).

    Serial number, hardware UUID and provisioning UDID are in the same record; none is read."""
    d = json.loads(system_profiler_json)
    hw = (d.get("SPHardwareDataType") or [{}])[0]
    machine = {"vendor": "Apple", "model": hw.get("machine_name"), "model_id": hw.get("machine_model")}
    gpus = []
    for dsp in d.get("SPDisplaysDataType") or []:
        vendor_text = dsp.get("spdisplays_vendor") or ""
        vendor = next((v.lower() for v in ("Apple", "AMD", "Intel", "NVIDIA") if v in vendor_text), None)
        g = empty_gpu(vendor)
        g.update(name=dsp.get("sppci_model"), gpu_cores=num(dsp.get("sppci_cores")),
                 metal_family=(dsp.get("spdisplays_mtlgpufamilysupport") or "").replace("spdisplays_", "") or None,
                 unified_memory=True if vendor == "apple" else None,
                 memory_total_mib=size_mib((dsp.get("spdisplays_vram") or "").replace("GB", "GiB").replace("MB", "MiB")))
        gpus.append(g)
    return machine, gpus


# ---------------------------------------------------------------- the rest

def filesystem(role, path):
    """The filesystem holding PATH, recorded under ROLE. Neither the path nor the mount source is
    kept: a home directory names its user, and an LVM volume group often names its host."""
    fs = {"role": role, "fstype": None, "size_gib": None, "raid_level": None, "raid_devices": None}
    try:
        st = os.statvfs(os.path.expanduser(path))
    except OSError:
        return fs
    fs["size_gib"] = round(st.f_blocks * st.f_frsize / 2 ** 30, 1)
    found = run("findmnt", "-n", "-o", "SOURCE,FSTYPE", "-T", os.path.expanduser(path))
    if found and len(found.split()) == 2:
        source, fs["fstype"] = found.split()
        md = os.path.basename(os.path.realpath(source))
        if md.startswith("md"):
            fs["raid_level"] = read(f"/sys/block/{md}/md/level")
            fs["raid_devices"] = num(read(f"/sys/block/{md}/md/raid_disks"))
    return fs


def ollama_url(explicit=None):
    url = explicit or os.environ.get("OLLAMA_HOST") or "127.0.0.1:11434"
    if "://" not in url:
        url = "http://" + url
    url = url.replace("://0.0.0.0", "://127.0.0.1")
    return url if re.search(r":\d+$", url) else url + ":11434"


def dpkg_version(*packages):
    for p in packages:
        v = run("dpkg-query", "-W", "-f=${Version}", p)
        if v:
            return v
    return None


def software(ollama):
    smi = run("nvidia-smi")
    driver = run("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
    module = run("modinfo", "-F", "license", "nvidia")
    nvcc = run("nvcc", "--version") or run("/usr/local/cuda/bin/nvcc", "--version")
    version = http(ollama.rstrip("/") + "/api/version")
    try:
        ollama_version = json.loads(version).get("version") if version else None
    except ValueError:
        ollama_version = None
    return {
        "nvidia_driver_version": driver.splitlines()[0] if driver else None,
        "nvidia_kernel_module": {"NVIDIA": "proprietary", "Dual MIT/GPL": "open"}.get(module) if module else None,
        "cuda_driver_version": match(r"CUDA (?:UMD )?Version:\s*([\d.]+)", smi),
        "cuda_toolkit_version": match(r"release [\d.]+, V([\d.]+)", nvcc),
        "cudnn_version": upstream_version(dpkg_version("libcudnn9-cuda-13", "libcudnn9-cuda-12")),
        "nccl_version": upstream_version(dpkg_version("libnccl2")),
        "rocm_version": upstream_version(read("/opt/rocm/.info/version")),
        "amdgpu_driver_version": read("/sys/module/amdgpu/version"),
        "docker_version": run("docker", "version", "--format", "{{.Server.Version}}", timeout=10),
        "nvidia_container_toolkit_version": match(r"version ([\d.]+)", run("nvidia-ctk", "--version")),
        "dcgm_version": match(r"[Vv]ersion\s*:\s*([\d.]+)", run("dcgmi", "--version")),
        "ollama_version": ollama_version,
        "go_version": match(r"go version go([\d.]+)", run("go", "version") or run("/usr/local/go/bin/go", "version")),
        "python_version": platform.python_version(),
    }


def collect(label=None, fs=(), ollama=None, identity_file=IDENTITY_FILE):
    label = collab_label(label, identity_file)
    label_host, _, surface = (label or "").partition("/")
    osi = os_info()
    if osi["family"] == "darwin":
        sysctl = lambda name: run("sysctl", "-n", name)
        sp = run("system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json", timeout=60)
        machine, gpus = darwin_hardware(sp) if sp else ({"vendor": "Apple", "model": None, "model_id": None}, [])
        cpu = cpu_darwin(sysctl)
        machine["virtualized"] = sysctl("kern.hv_vmm_present") == "1"
        machine["hypervisor"] = None
        memory_gib = (num(sysctl("hw.memsize")) or 0) / 2 ** 30 or None
        cloud = None
    else:
        d = dmi()
        cpu, hypervisor = cpu_linux(run("lscpu", "-J"), read("/proc/cpuinfo"))
        machine = {"vendor": d["sys_vendor"] or d["board_vendor"],
                   "model": d["product_name"] or d["board_name"], "model_id": None,
                   "virtualized": bool(hypervisor), "hypervisor": hypervisor}
        memory_gib = (num(match(r"MemTotal:\s+(\d+)", read("/proc/meminfo"))) or 0) / 2 ** 20 or None
        cloud = cloud_from(d)
        gpus = gpus_linux()
    for i, g in enumerate(gpus):
        g["index"] = i
    return {
        "schema": SCHEMA,
        "collected_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label": label,
        "label_host": label_host or None,
        "label_surface": surface or None,
        "machine": machine,
        "cloud": cloud,
        "os": osi,
        "cpu": cpu,
        "memory_total_gib": round(memory_gib, 1) if memory_gib else None,
        "gpu_count": len(gpus),
        "gpus": gpus,
        "filesystems": [filesystem(role, path) for role, path in fs],
        "software": software(ollama_url(ollama)),
    }


def flatten(value, prefix=""):
    """{"gpus": [{"name": "x"}]} -> {"gpus.0.name": "x"}: one key per fact, for tables."""
    if isinstance(value, dict):
        return {k: v for key, item in value.items() for k, v in flatten(item, f"{prefix}{key}.").items()}
    if isinstance(value, list):
        return {k: v for i, item in enumerate(value) for k, v in flatten(item, f"{prefix}{i}.").items()}
    return {prefix[:-1]: value}


# ---------------------------------------------------------------- Markdown

SOFTWARE_LABELS = {"cuda_toolkit_version": "CUDA toolkit", "cudnn_version": "cuDNN", "nccl_version": "NCCL",
                   "rocm_version": "ROCm", "amdgpu_driver_version": "amdgpu", "docker_version": "Docker",
                   "nvidia_container_toolkit_version": "NVIDIA Container Toolkit", "dcgm_version": "DCGM",
                   "ollama_version": "Ollama", "go_version": "Go", "python_version": "Python"}


def fmt(v):
    return f"{v:g}" if isinstance(v, float) else str(v)


def opt(template, *values):
    """template filled in when every value is known; '' otherwise."""
    return template.format(*map(fmt, values)) if all(v is not None and v != "" for v in values) else ""


def on_off_text(v):
    return {True: "on", False: "off"}.get(v)


def count(n, noun):
    return f"{fmt(n)} {noun}{'' if n == 1 else 's'}" if n is not None else ""


def machine_name(m):
    """'Google' + 'Google Compute Engine' reads as the latter, not 'Google Google Compute Engine'."""
    vendor, model = m.get("vendor"), m.get("model")
    if vendor and model and model.lower().startswith(vendor.lower()):
        return model
    return " ".join(x for x in (vendor, model) if x)


def render_markdown(p):
    """The profile as a two-column table; a row with nothing known is left out."""
    rows = []

    def row(key, text):
        text = (text or "").strip().lstrip(",").strip()
        if text:
            rows.append(f"| {key} | {text} |")

    m, c, osi, cloud, s = p["machine"], p["cpu"], p["os"], p["cloud"], p["software"]
    row("Label", p["label"])
    row("Machine", machine_name(m) + opt(" ({})", m.get("model_id"))
        + opt(", virtual machine on {}", m.get("hypervisor")))
    if cloud:
        row("Cloud", opt("{}", cloud["provider"]) + opt(" {}", cloud["machine_type"])
            + opt(", CPU platform {}", cloud["cpu_platform"])
            + opt(", zone {} (region {})", cloud["zone"], cloud["region"]))
    row("OS", opt("{}", osi.get("pretty_name")) + opt(", kernel {}", osi.get("kernel")) + opt(", {}", osi.get("arch")))
    per_socket = c["cores"] // c["sockets"] if c["cores"] and c["sockets"] else None
    topology = ""
    if all(v is not None for v in (c["logical_cpus"], c["sockets"], per_socket, c["threads_per_core"])):
        topology = (f": {count(c['logical_cpus'], 'logical CPU')} = {count(c['sockets'], 'socket')} × "
                    f"{count(per_socket, 'core')} × {count(c['threads_per_core'], 'thread')}")
    row("CPU", opt("{}", c["model"])
        + (opt(": {} performance + {} efficiency cores", c["performance_cores"], c["efficiency_cores"]) or topology)
        + opt(", {}", count(c["numa_nodes"], "NUMA node")) + opt(", L3 {} MiB", c["l3_cache_mib"]))
    if c["flags"] is not None:
        row("CPU features", ", ".join(f for f, on in c["flags"].items() if on) or "none of those tracked")
    row("Memory", opt("{} GiB", p["memory_total_gib"]))
    for f in p["filesystems"]:
        row(f"Filesystem ({f['role']})", opt("{}", f["fstype"]) + opt(", {} GiB", f["size_gib"])
            + opt(", {} of {} devices", f["raid_level"], f["raid_devices"]))
    for g in p["gpus"]:
        i = g["index"]
        memory = "unified memory" if g["unified_memory"] else (
            opt("{} MiB", g["memory_total_mib"]) + opt(" + {} MiB GTT", g["memory_gtt_mib"]))
        kind = (opt("compute capability {}", g["compute_capability"]) or opt("{}", g["gfx_target"])
                or opt("{} GPU cores", g["gpu_cores"]) + opt(", {}", g["metal_family"]))
        row(f"GPU {i}", opt("{}", g["name"]) + (opt(" ({})", g["pci_name"]) if g["pci_name"] != g["name"] else "")
            + opt(", {}", memory) + opt(", {}", kind) + opt(", VBIOS {}", g["vbios"]))
        row(f"GPU {i} limits", opt("{} W", g["power_limit_w"]) + opt(", core up to {} MHz", g["max_core_clock_mhz"])
            + opt(", memory up to {} MHz", g["max_memory_clock_mhz"])
            + opt(", ECC {}", on_off_text(g["ecc_enabled"])) + opt(", MIG {}", on_off_text(g["mig_enabled"])))
        notes = []
        if g["pcie_device_max_gen"] and (g["pcie_device_max_gen"], g["pcie_device_max_width"]) != (
                g["pcie_link_gen"], g["pcie_link_width"]):
            notes.append(opt("the GPU supports Gen{} x{}", g["pcie_device_max_gen"], g["pcie_device_max_width"]))
        if g["pcie_gen_now"] and g["pcie_gen_now"] != g["pcie_link_gen"]:
            notes.append(opt("Gen{} x{} when profiled: the link slows while the GPU idles",
                             g["pcie_gen_now"], g["pcie_width_now"]))
        row(f"GPU {i} PCIe", opt("Gen{} x{}", g["pcie_link_gen"], g["pcie_link_width"])
            + (f" ({'; '.join(notes)})" if notes else "")
            + opt(", NUMA node {}", g["numa_node"]) + opt(", CPUs {}", g["local_cpus"]))
    row("NVIDIA driver", opt("{}", s["nvidia_driver_version"]) + opt(" ({} kernel module)", s["nvidia_kernel_module"])
        + opt(", CUDA {}", s["cuda_driver_version"]))
    row("Software", ", ".join(f"{name} {s[k]}" for k, name in SOFTWARE_LABELS.items() if s.get(k)))
    title = p["label"] or machine_name(m) or "unknown host"
    return "\n".join([f"# Host profile: {title}", "",
                      f"Collected {p['collected_at']} ({p['schema']}).", "",
                      "| | |", "|---|---|", *rows]) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", help="collab label HOST/SURFACE (default: the first line of "
                                    "~/.config/collab/identity)")
    ap.add_argument("--fs", action="append", default=[], metavar="ROLE=PATH",
                    help="also record the filesystem holding PATH, under ROLE (e.g. models=~/.ollama/models); "
                         "the path is not recorded")
    ap.add_argument("--ollama", help="Ollama server to ask for its version (default: $OLLAMA_HOST, "
                                     "else 127.0.0.1:11434)")
    out = ap.add_mutually_exclusive_group()
    out.add_argument("--json", action="store_true", help="print the profile as JSON")
    out.add_argument("--flat", action="store_true", help='print {"dotted.key": value}, one per fact')
    ap.add_argument("--out", metavar="DIR", help="write DIR/host-profile.json and DIR/host-profile.md")
    args = ap.parse_args(argv)
    fs = []
    for item in args.fs:
        role, sep, path = item.partition("=")
        if not sep or not role or not path:
            ap.error(f"--fs takes ROLE=PATH, got {item!r}")
        fs.append((role, path))

    profile = collect(args.label, fs, args.ollama)
    markdown = render_markdown(profile)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "host-profile.json"), "w") as f:
            json.dump(profile, f, indent=2)
            f.write("\n")
        with open(os.path.join(args.out, "host-profile.md"), "w") as f:
            f.write(markdown)
    if args.json:
        print(json.dumps(profile, indent=2))
    elif args.flat:
        print(json.dumps(flatten(profile), indent=2))
    else:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
