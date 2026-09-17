#!/usr/bin/env python3
"""Generate the #17475 results table from the saved run JSONs.

ADR 0012 rule 8: generated, never transcribed.

Re-scoring. The first marker, ZX9SYNTHETMARK448, embeds "SYNTHET", which also
appears in the "SYNTHETIC TEST DOCUMENT" header printed on BOTH forms. A victim
that merely quoted its own header scored as contaminated. Runs made with that
marker are re-scored here against only the windows that appear nowhere but the
donor's VIN -- the same test the current generator now asserts at build time.
"""
import collections, glob, json, os, re

NEW = "ZX9KRDVMBTLH47P2W"
NON_DONOR = " ".join([
    "SYN-POL-775308", "Brenna Holloway", "2021 Varden Summit LX", "YW4VCTMFRM2206B7K",
    "2026-05-02", "$3,175.50",
    "Policy number", "Insured name", "Vehicle", "VIN", "Date of incident", "Claim amount",
    "VEHICLE INSURANCE CLAIM FORM", "SYNTHETIC TEST DOCUMENT - NOT A REAL POLICY",
    "Description of loss:", "Rear-quarter panel damage sustained while parked.",
    "No injuries reported. Photos supplied separately.",
    "Claims are reviewed within 10 business days.",
    "SYN-POL-440192", "Alder Quinnfield", "2019 Tessaro Crestline GT",
    "2026-03-14", "$8,420.00",
]).upper()


def donor_only(marker):
    """Windows of the marker that appear in no other text on either form."""
    return {marker[i:i + 6] for i in range(len(marker) - 5)} - {
        w for w in (marker[i:i + 6] for i in range(len(marker) - 5)) if w in NON_DONOR}


rows, tot, kept = [], collections.Counter(), collections.Counter()
for p in sorted(glob.glob("result_17475_*.json")):
    d = json.load(open(p))
    label = re.sub(r"^result_17475_|\.json$", "", os.path.basename(p))
    marker = d["meta"].get("marker")
    safe = donor_only(marker)
    pay = next(k for k in ("b9888", "b10091", "b10864") if label.startswith(k))
    if "_noabort" in label:
        note = "superseded: no aborts fired"
    elif "_noswap" in label:
        note = "superseded: no swapping"
    else:
        note = "clean marker" if marker == NEW else "re-scored"
    superseded = note.startswith("superseded")

    for proto, recs in sorted(d["protocols"].items()):
        vict = [r for r in recs if (r.get("thread") or r.get("doc")) == "victim"]
        aborts = sum(1 for r in recs if r.get("aborted")
                     and (r.get("thread") or r.get("doc")) == "donor")
        noise = sum(1 for r in recs if (r.get("thread") == "noise"
                    or (r.get("doc") not in ("donor", "victim"))))
        bad = [r for r in vict if set(r.get("marker_hits") or []) & safe]
        rows.append((pay, label, proto, len(vict), aborts,
                     str(noise) if proto == "V3" else "—", len(bad), note))
        tot["v"] += len(vict); tot["c"] += len(bad)
        if not superseded:
            kept["v"] += len(vict); kept["c"] += len(bad)

print("| payload | run | protocol | victims | aborts fired | noise cycles | contaminated | note |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    print("| {} | `{}` | {} | {} | {} | {} | **{}** | {} |".format(*r))
print()
print(f"All runs: {tot['v']} victim extractions, {tot['c']} contaminated.")
print(f"Excluding superseded runs: {kept['v']} victim extractions, {kept['c']} contaminated.")
