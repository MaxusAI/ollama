#!/usr/bin/env python3
"""Tests for host_profile.py, on output captured from real hosts with identifiers removed."""
import json
import os
import socket
import tempfile
import unittest

import host_profile as hp

# Captured 2026-09-30 from `lscpu -J` on a GCP a3-highgpu-1g VM (util-linux 2.41, which prints
# a flat list), trimmed to the fields the profile reads.
LSCPU_FLAT = json.dumps({"lscpu": [
    {"field": "Architecture:", "data": "x86_64"},
    {"field": "CPU(s):", "data": "26"},
    {"field": "Vendor ID:", "data": "GenuineIntel"},
    {"field": "Model name:", "data": "Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz"},
    {"field": "Thread(s) per core:", "data": "2"},
    {"field": "Core(s) per socket:", "data": "13"},
    {"field": "Socket(s):", "data": "1"},
    {"field": "Hypervisor vendor:", "data": "KVM"},
    {"field": "L3 cache:", "data": "105 MiB (1 instance)"},
    {"field": "NUMA node(s):", "data": "1"},
]})

# The same facts in the hierarchical form, which names the cache "L3" under a parent entry.
LSCPU_NESTED = json.dumps({"lscpu": [
    {"field": "CPU(s):", "data": "26"},
    {"field": "Vendor ID:", "data": "GenuineIntel", "children": [
        {"field": "Model name:", "data": "Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz", "children": [
            {"field": "Thread(s) per core:", "data": "2"},
            {"field": "Core(s) per socket:", "data": "13"},
            {"field": "Socket(s):", "data": "1"}]}]},
    {"field": "Caches (sum of all):", "data": None, "children": [
        {"field": "L3:", "data": "105 MiB (1 instance)"}]},
]})

CPUINFO_X86 = "processor\t: 0\nflags\t\t: fpu sse avx2 avx512f avx512_bf16 amx_tile amx_bf16\n"
CPUINFO_ARM = "processor\t: 0\nFeatures\t: fp asimd sve bf16\n"

# Same VM: `nvidia-smi --query-gpu=<NVIDIA_FIELDS> --format=csv,noheader,nounits`.
NVIDIA_SMI = ("00000000:04:00.0, NVIDIA H100 80GB HBM3, 81559, 9.0, 96.00.DB.00.02, 700.00, 700.00, "
              "1980, 2619, Enabled, Disabled, Enabled\n")

# Same VM: `lspci -mm -D`, the GPU's line.
LSPCI = ('0000:04:00.0 "3D controller" "NVIDIA Corporation" "GH100 [H100 SXM5 80GB]" -ra1 -p00 '
         '"NVIDIA Corporation" "Device 16c1"\n')

# Captured 2026-09-30 from `system_profiler SPHardwareDataType SPDisplaysDataType -json` on an
# Apple M5 Max. The identifying fields stay in, with made-up values, to show they are not read.
SYSTEM_PROFILER = json.dumps({
    "SPHardwareDataType": [{
        "_name": "hardware_overview", "chip_type": "Apple M5 Max", "machine_model": "Mac17,7",
        "machine_name": "MacBook Pro", "number_processors": "proc 18:6:0:12", "physical_memory": "128 GB",
        "serial_number": "SERIAL-FIXTURE", "platform_UUID": "UUID-FIXTURE", "provisioning_UDID": "UDID-FIXTURE"}],
    "SPDisplaysDataType": [{
        "_name": "Apple M5 Max", "spdisplays_mtlgpufamilysupport": "spdisplays_metal4",
        "spdisplays_vendor": "sppci_vendor_Apple", "sppci_bus": "spdisplays_builtin", "sppci_cores": "40",
        "sppci_device_type": "spdisplays_gpu", "sppci_model": "Apple M5 Max"}],
})


def write(root, rel, text):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text + "\n")


def fake_sysfs(root):
    """What the VM's sysfs shows for its H100 -- a Gen5 device under a port that tops out at Gen4
    -- plus an AMD APU on the root complex (gfx1151, carve-out and GTT, no PCIe link), a virtual
    display adapter and an NVIDIA network device, neither of which is a GPU."""
    port = "devices/pci0000:01/0000:01:00.0"
    write(root, port + "/max_link_speed", "16.0 GT/s PCIe")
    write(root, port + "/max_link_width", "16")
    h100 = port + "/0000:04:00.0"
    for name, value in {"class": "0x030200", "vendor": "0x10de", "device": "0x2330",
                        "max_link_speed": "32.0 GT/s PCIe", "max_link_width": "16",
                        "current_link_speed": "16.0 GT/s PCIe", "current_link_width": "16",
                        "numa_node": "0", "local_cpulist": "0-25"}.items():
        write(root, f"{h100}/{name}", value)
    apu = "devices/pci0000:00/0000:c5:00.0"
    for name, value in {"class": "0x030000", "vendor": "0x1002", "device": "0x1586",
                        "mem_info_vram_total": str(96 * 2 ** 30), "mem_info_gtt_total": str(16 * 2 ** 30),
                        "current_link_speed": "Unknown", "max_link_speed": "Unknown",
                        "numa_node": "-1"}.items():
        write(root, f"{apu}/{name}", value)
    virtual = "devices/pci0000:00/0000:00:02.0"
    write(root, virtual + "/class", "0x030000")
    write(root, virtual + "/vendor", "0x1af4")
    nic = "devices/pci0000:00/0000:00:05.0"
    write(root, nic + "/class", "0x020000")
    write(root, nic + "/vendor", "0x10de")
    os.makedirs(os.path.join(root, "bus/pci/devices"))
    for rel in (h100, apu, virtual, nic):
        os.symlink(os.path.join(root, rel), os.path.join(root, "bus/pci/devices", os.path.basename(rel)))
    driver = os.path.join(root, "bus/pci/drivers/nvidia")
    os.makedirs(driver)
    os.symlink(driver, os.path.join(root, h100, "driver"))
    write(root, "class/kfd/kfd/topology/nodes/0/properties",
          "cpu_cores_count 16\ngfx_target_version 0\nlocation_id 0\ndomain 0")
    write(root, "class/kfd/kfd/topology/nodes/1/properties",
          f"simd_count 80\ngfx_target_version 110501\nlocation_id {0xc5 << 8}\ndomain 0")


class TestValues(unittest.TestCase):
    def test_a_link_speed_is_a_generation(self):
        self.assertEqual([hp.link_gen(s) for s in ("2.5 GT/s PCIe", "8.0 GT/s PCIe", "16.0 GT/s PCIe",
                                                   "32.0 GT/s PCIe", "Unknown", None)], [1, 3, 4, 5, None, None])

    def test_a_gfx_target_version_is_a_target_name(self):
        self.assertEqual([hp.gfx_target(v) for v in ("110501", "90010", "90402", "110000", "0")],
                         ["gfx1151", "gfx90a", "gfx942", "gfx1100", None])

    def test_sizes_are_mib(self):
        self.assertEqual([hp.size_mib(s) for s in ("105 MiB (1 instance)", "1.5 GiB", "512 KiB", "n/a")],
                         [105, 1536, 0.5, None])

    def test_versions_lose_their_debian_revision(self):
        self.assertEqual([hp.upstream_version(v) for v in ("2.32.3-1+cuda13.4", "9.27.0.42-1", "7.0.2")],
                         ["2.32.3", "9.27.0.42", "7.0.2"])

    def test_placeholders_are_unknown_not_values(self):
        self.assertEqual([hp.text_value(v) for v in ("[N/A]", "[Not Supported]", "To be filled by O.E.M.",
                                                     "Default string", "Mini PC")],
                         [None, None, None, None, "Mini PC"])

    def test_states_are_booleans(self):
        self.assertEqual([hp.on_off(v) for v in ("Enabled", "Disabled", "TRUE", "FALSE", "[N/A]")],
                         [True, False, True, False, None])


class TestCpu(unittest.TestCase):
    def test_flat_and_hierarchical_lscpu_read_the_same(self):
        flat, hypervisor = hp.cpu_linux(LSCPU_FLAT, CPUINFO_X86)
        nested, _ = hp.cpu_linux(LSCPU_NESTED, CPUINFO_X86)
        for key in ("model", "logical_cpus", "sockets", "cores", "threads_per_core", "l3_cache_mib"):
            self.assertEqual(flat[key], nested[key], key)
        self.assertEqual((flat["vendor"], flat["cores"], flat["l3_cache_mib"], hypervisor),
                         ("intel", 13, 105, "KVM"))

    def test_every_tracked_flag_is_true_or_false(self):
        x86 = hp.cpu_features(CPUINFO_X86)
        self.assertEqual(set(x86), set(hp.X86_FLAGS))
        self.assertEqual((x86["amx_bf16"], x86["amx_int8"]), (True, False))
        arm = hp.cpu_features(CPUINFO_ARM)
        self.assertEqual((arm["sve"], arm["sme"]), (True, False))
        self.assertIsNone(hp.cpu_features(""))

    def test_apple_silicon_counts_performance_and_efficiency_cores(self):
        sysctl = {"machdep.cpu.brand_string": "Apple M5 Max", "hw.logicalcpu": "18", "hw.physicalcpu": "18",
                  "hw.perflevel0.physicalcpu": "6", "hw.perflevel1.physicalcpu": "12",
                  "hw.optional.arm.FEAT_BF16": "1", "hw.optional.arm.FEAT_SME": "1"}.get
        cpu = hp.cpu_darwin(sysctl)
        self.assertEqual((cpu["vendor"], cpu["performance_cores"], cpu["efficiency_cores"],
                          cpu["threads_per_core"]), ("apple", 6, 12, 1))
        self.assertEqual((cpu["flags"]["sme"], cpu["flags"]["i8mm"]), (True, False))


class TestLinuxGpus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        fake_sysfs(self.tmp.name)
        self.gpus = {g["vendor"]: g for g in hp.pci_gpus(self.tmp.name)}

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_gpu_vendors_display_devices_count(self):
        self.assertEqual(sorted(self.gpus), ["amd", "nvidia"])

    def test_the_port_above_the_gpu_limits_the_link(self):
        g = self.gpus["nvidia"]
        self.assertEqual((g["pcie_link_gen"], g["pcie_link_width"]), (4, 16))
        self.assertEqual((g["pcie_device_max_gen"], g["pcie_upstream_max_gen"], g["pcie_gen_now"]), (5, 4, 4))
        self.assertEqual((g["numa_node"], g["local_cpus"], g["driver"]), (0, "0-25", "nvidia"))

    def test_an_apu_reports_its_carve_out_and_gtt_and_no_link(self):
        g = self.gpus["amd"]
        self.assertEqual((g["memory_total_mib"], g["memory_gtt_mib"]), (96 * 1024, 16 * 1024))
        self.assertIsNone(g["pcie_link_gen"])
        self.assertIsNone(g["numa_node"])

    def test_the_kfd_topology_names_the_gpu_at_its_pci_address(self):
        self.assertEqual(hp.kfd_gfx_targets(self.tmp.name), {"0000:c5:00.0": "gfx1151"})

    def test_nvidia_smi_values_are_typed(self):
        row = hp.nvidia_smi_rows(NVIDIA_SMI)["0000:04:00.0"]
        self.assertEqual((row["memory_total_mib"], row["power_limit_w"], row["compute_capability"]),
                         (81559, 700.0, "9.0"))
        self.assertEqual((row["ecc_enabled"], row["mig_enabled"], row["persistence_enabled"]), (True, False, True))

    def test_lspci_names_a_device_by_its_address(self):
        self.assertEqual(hp.lspci_names(LSPCI), {"0000:04:00.0": "GH100 [H100 SXM5 80GB]"})


class TestCloud(unittest.TestCase):
    def test_gcp_is_asked_for_machine_facts_and_nothing_that_names_the_project(self):
        answers = {
            "instance/zone": "projects/000000000000/zones/australia-southeast2-b",
            "instance/machine-type": "projects/000000000000/machineTypes/a3-highgpu-1g",
            "instance/cpu-platform": "Intel Sapphire Rapids",
            "instance/image": "projects/debian-cloud/global/images/debian-13-trixie-v20260921",
            "instance/scheduling/?recursive=true":
                '{"automaticRestart":"TRUE","onHostMaintenance":"TERMINATE","preemptible":"FALSE"}',
        }
        asked = []

        def metadata(path):
            asked.append(path)
            return answers.get(path)

        cloud = hp.cloud_from({"sys_vendor": "Google", "product_name": "Google Compute Engine"}, metadata)
        self.assertEqual((cloud["provider"], cloud["zone"], cloud["region"], cloud["machine_type"], cloud["image"]),
                         ("gcp", "australia-southeast2-b", "australia-southeast2", "a3-highgpu-1g",
                          "debian-13-trixie-v20260921"))
        self.assertEqual((cloud["on_host_maintenance"], cloud["automatic_restart"], cloud["preemptible"]),
                         ("TERMINATE", True, False))
        for path in asked:
            self.assertNotRegex(path, r"^project|instance/(name|id|hostname|attributes)|service-accounts")

    def test_aws_takes_the_instance_type_from_dmi(self):
        cloud = hp.cloud_from({"sys_vendor": "Amazon EC2", "product_name": "p5.48xlarge"})
        self.assertEqual((cloud["provider"], cloud["machine_type"], cloud["zone"]), ("aws", "p5.48xlarge", None))

    def test_bare_metal_has_no_cloud(self):
        self.assertIsNone(hp.cloud_from({"sys_vendor": "A Mini PC Maker", "product_name": "Mini PC"}))


class TestDarwin(unittest.TestCase):
    def test_apple_silicon_gpu(self):
        machine, gpus = hp.darwin_hardware(SYSTEM_PROFILER)
        self.assertEqual(machine, {"vendor": "Apple", "model": "MacBook Pro", "model_id": "Mac17,7"})
        g = gpus[0]
        self.assertEqual((g["vendor"], g["name"], g["gpu_cores"], g["metal_family"], g["unified_memory"]),
                         ("apple", "Apple M5 Max", 40, "metal4", True))

    def test_serial_and_hardware_ids_are_not_read(self):
        text = json.dumps(hp.darwin_hardware(SYSTEM_PROFILER))
        for value in ("SERIAL-FIXTURE", "UUID-FIXTURE", "UDID-FIXTURE"):
            self.assertNotIn(value, text)


class TestProfile(unittest.TestCase):
    def profile(self):
        gpu = hp.empty_gpu("nvidia")
        gpu.update(index=0, name="NVIDIA H100 80GB HBM3", pci_name="GH100 [H100 SXM5 80GB]", memory_total_mib=81559,
                   compute_capability="9.0", pcie_link_gen=4, pcie_link_width=16, pcie_device_max_gen=5,
                   pcie_device_max_width=16, pcie_gen_now=4, pcie_width_now=16)
        cpu, hypervisor = hp.cpu_linux(LSCPU_FLAT, CPUINFO_X86)
        return {"schema": hp.SCHEMA, "collected_at": "2026-09-30T00:00:00Z", "label": "gcp-h100/cuda",
                "label_host": "gcp-h100", "label_surface": "cuda",
                "machine": {"vendor": "Google", "model": "Google Compute Engine", "model_id": None,
                            "virtualized": True, "hypervisor": hypervisor},
                "cloud": None, "os": {"family": "linux", "name": None, "version": None, "pretty_name": None,
                                      "kernel": None, "arch": "x86_64"},
                "cpu": cpu, "memory_total_gib": None, "gpu_count": 1, "gpus": [gpu], "filesystems": [],
                "software": {k: None for k in ("nvidia_driver_version", "nvidia_kernel_module",
                                               "cuda_driver_version", *hp.SOFTWARE_LABELS)}}

    def test_markdown_says_what_limits_the_link_and_leaves_unknowns_out(self):
        md = hp.render_markdown(self.profile())
        self.assertIn("| GPU 0 PCIe | Gen4 x16 (the GPU supports Gen5 x16) |", md)
        self.assertIn("| Machine | Google Compute Engine, virtual machine on KVM |", md)
        self.assertIn("26 logical CPUs = 1 socket × 13 cores × 2 threads, 1 NUMA node, L3 105 MiB", md)
        self.assertNotIn("None", md)
        self.assertNotIn("| Memory |", md)

    def test_flat_keys_are_dotted_one_per_fact(self):
        self.assertEqual(hp.flatten({"a": {"b": 1, "c": [{"d": True}]}, "e": None}),
                         {"a.b": 1, "a.c.0.d": True, "e": None})
        flat = hp.flatten(self.profile())
        self.assertEqual((flat["gpus.0.pcie_link_gen"], flat["cpu.flags.amx_bf16"]), (4, True))
        for key, value in flat.items():
            self.assertNotIsInstance(value, (dict, list), key)

    def test_this_host_profiles_without_naming_itself(self):
        profile = hp.collect(label="ci/test", fs=[("root", "/")], ollama="127.0.0.1:9")
        text = json.dumps(profile)
        self.assertEqual(profile["schema"], hp.SCHEMA)
        self.assertEqual(profile["filesystems"][0]["role"], "root")
        parts = {part for key in hp.flatten(profile) for part in key.split(".")}
        for word in ("hostname", "user", "serial", "uuid", "project", "instance", "instance_id",
                     "physical_host", "path", "source"):
            self.assertNotIn(word, parts)
        self.assertNotIn(os.path.expanduser("~"), text)
        self.assertNotIn(socket.gethostname(), hp.flatten(profile).values())

    def test_fs_needs_a_role(self):
        with self.assertRaises(SystemExit):
            hp.main(["--fs", "/"])


if __name__ == "__main__":
    unittest.main()
