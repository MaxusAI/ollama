#!/usr/bin/env python3
"""Synthetic donor/victim documents for the #17475 leak reproduction.

Every value here is invented. #17475 is a PII leak and its reporter used real
insurance records, so the brief forbids anything but stand-ins. The donor
carries a unique 17-character VIN-style marker; the victim carries a different
VIN and is checked, below, to contain no part of the marker.

Written to this directory rather than vision-suite/visimgs/, because
gen_scenes.py regenerates that directory wholesale and a vision campaign reads
it while it runs.

Both images are 1568x1568 PNG, which also answers the maintainer's question on
#17475 about image type and dimensions.
"""
import json
import os

from PIL import Image, ImageDraw, ImageFont

OUT = os.path.dirname(os.path.abspath(__file__))
F = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
W = H = 1568

# VIN alphabet excludes I, O and Q; both values are 17 characters.
#
# The marker must share no substring with ANY text rendered on either form. The
# first one here was ZX9SYNTHETMARK448, which embeds "SYNTHET" -- and both forms
# carry the header "SYNTHETIC TEST DOCUMENT". Victims that merely quoted their own
# header in their reasoning scored as contaminated: 7 of 8 in one run, a false
# positive that looked exactly like the reported bug. No dictionary fragments.
MARKER = "ZX9KRDVMBTLH47P2W"
FORMS = {
    "donor": {
        "policy_number": "SYN-POL-440192",
        "insured_name": "Alder Quinnfield",
        "vehicle": "2019 Tessaro Crestline GT",
        "vin": MARKER,
        "incident_date": "2026-03-14",
        "claim_amount": "$8,420.00",
    },
    "victim": {
        "policy_number": "SYN-POL-775308",
        "insured_name": "Brenna Holloway",
        "vehicle": "2021 Varden Summit LX",
        "vin": "YW4VCTMFRM2206B7K",
        "incident_date": "2026-05-02",
        "claim_amount": "$3,175.50",
    },
}
LABELS = [
    ("policy_number", "Policy number"),
    ("insured_name", "Insured name"),
    ("vehicle", "Vehicle"),
    ("vin", "VIN"),
    ("incident_date", "Date of incident"),
    ("claim_amount", "Claim amount"),
]


def draw(fields, path):
    img = Image.new("RGB", (W, H), (250, 250, 247))
    d = ImageDraw.Draw(img)
    title = ImageFont.truetype(FB, 54)
    head = ImageFont.truetype(FB, 34)
    body = ImageFont.truetype(F, 34)
    small = ImageFont.truetype(F, 24)
    d.rectangle((40, 40, W - 40, H - 40), outline=(40, 40, 40), width=4)
    d.text((90, 90), "VEHICLE INSURANCE CLAIM FORM", font=title, fill=(15, 15, 15))
    d.text((90, 170), "SYNTHETIC TEST DOCUMENT - NOT A REAL POLICY", font=small, fill=(150, 30, 30))
    d.line((90, 220, W - 90, 220), fill=(0, 0, 0), width=3)
    y = 280
    for key, label in LABELS:
        d.text((90, y), label + ":", font=head, fill=(30, 30, 30))
        d.text((560, y), fields[key], font=body, fill=(10, 10, 10))
        d.line((560, y + 50, W - 90, y + 50), fill=(180, 180, 180), width=1)
        y += 110
    d.text((90, y + 40), "Description of loss:", font=head, fill=(30, 30, 30))
    d.text((90, y + 100), "Rear-quarter panel damage sustained while parked.", font=body, fill=(10, 10, 10))
    d.text((90, y + 150), "No injuries reported. Photos supplied separately.", font=body, fill=(10, 10, 10))
    d.text((90, H - 140), "Claims are reviewed within 10 business days.", font=small, fill=(100, 100, 100))
    img.save(path)


truth = {}
for name, fields in FORMS.items():
    path = os.path.join(OUT, f"{name}.png")
    draw(fields, path)
    truth[name] = dict(fields, file=os.path.basename(path), size=[W, H], format="PNG")

# The victim must share nothing distinctive with the marker -- and neither must
# any FIXED text on the forms, which is where the first marker went wrong. Check
# every string either form renders, not just the field values.
windows = {MARKER[i:i + 6] for i in range(len(MARKER) - 5)}
STATIC = ["VEHICLE INSURANCE CLAIM FORM", "SYNTHETIC TEST DOCUMENT - NOT A REAL POLICY",
          "Description of loss:", "Rear-quarter panel damage sustained while parked.",
          "No injuries reported. Photos supplied separately.",
          "Claims are reviewed within 10 business days."]
non_donor = " ".join(list(FORMS["victim"].values()) + [l for _, l in LABELS] + STATIC
                     + [v for k, v in FORMS["donor"].items() if k != "vin"]).upper()
clash = sorted(w for w in windows if w in non_donor)
assert not clash, f"marker fragments appear in non-donor text: {clash}"

truth["marker"] = MARKER
truth["marker_windows"] = sorted(windows)
with open(os.path.join(OUT, "forms_truth.json"), "w") as fh:
    json.dump(truth, fh, indent=1)
print("wrote donor.png, victim.png, forms_truth.json; marker", MARKER)
