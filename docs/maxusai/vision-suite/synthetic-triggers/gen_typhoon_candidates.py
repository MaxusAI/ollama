import numpy as np, os, random
from PIL import Image, ImageDraw, ImageFont
D = os.path.dirname(__file__)
try:
    import matplotlib
    FONT = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf/DejaVuSans.ttf")
except Exception:
    FONT = None
def f(sz):
    try: return ImageFont.truetype(FONT, sz)
    except Exception: return ImageFont.load_default()
def save(n, im): im.save(os.path.join(D, n))

WORDS = ("CERTIFICATE OF COMPLETION AWARDED TO THE INSTITUTE OF PROFESSIONAL "
         "ENGINEERS FOR OUTSTANDING ACHIEVEMENT IN APPLIED SCIENCE AND RESEARCH ").split()

# --- A. certificate-on-wall mimics: bright wall + saturated mat + gold frame + gloss ---
for gi, (W, H) in enumerate([(1176, 644), (1350, 900), (1800, 1000)]):
    for mat in [(198, 30, 70), (225, 20, 60), (150, 25, 90)]:      # saturated reds/magentas
        im = Image.new("RGB", (W, H), (238, 232, 218))              # cream wall
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, W, int(H * 0.30)], fill=(250, 249, 247)) # white ceiling band
        for k, (fx, fy, fw, fh) in enumerate([(0.22, 0.34, 0.24, 0.55), (0.52, 0.40, 0.18, 0.45), (0.75, 0.46, 0.14, 0.38)]):
            x0, y0 = int(W*fx), int(H*fy); x1, y1 = x0+int(W*fw), y0+int(H*fh)
            d.rectangle([x0-6, y0-6, x1+6, y1+6], fill=(212, 175, 55))   # gold frame
            d.rectangle([x0, y0, x1, y1], fill=mat)                       # saturated mat
            px0, py0, px1, py1 = x0+int((x1-x0)*.12), y0+int((y1-y0)*.10), x1-int((x1-x0)*.12), y1-int((y1-y0)*.10)
            d.rectangle([px0, py0, px1, py1], fill=(252, 252, 250))       # document
            fnt = f(max(7, (py1-py0)//22))
            for li in range(14):
                yy = py0 + 8 + li*max(8, (py1-py0)//16)
                if yy > py1-10: break
                d.text((px0+8, yy), " ".join(random.Random(li).sample(WORDS, 3)), font=fnt, fill=(35, 35, 45))
            for t in range(0, (y1-y0)//2, 3):                             # glossy diagonal highlight
                d.line([(x0, y0+t*2), (x0+(y1-y0)//2, y0)], fill=(255, 255, 255), width=1)
        save(f"cert_{W}x{H}_m{mat[0]}.png", im)

# --- B. dense document pages (OCR-style, the #17687 pipeline shape) ---
for (W, H) in [(1275, 1650), (1800, 2330)]:
    for pt, bg in ((9, 255), (13, 255), (9, 246)):
        im = Image.new("RGB", (W, H), (bg, bg, bg)); d = ImageDraw.Draw(im)
        fnt = f(pt * max(1, W // 700))
        y = int(H*0.06)
        rnd = random.Random(pt)
        while y < H - int(H*0.06):
            d.text((int(W*0.08), y), " ".join(rnd.sample(WORDS, min(11, len(WORDS)))), font=fnt, fill=(0, 0, 0))
            y += int(pt * max(1, W // 700) * 1.55)
        save(f"doc_{W}x{H}_pt{pt}_bg{bg}.png", im)

# --- C. document + saturated colour block (the certificate's colour clash, no wall) ---
for (W, H) in [(1176, 644), (1800, 1000)]:
    im = Image.new("RGB", (W, H), (252, 252, 250)); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, int(W*0.38), H], fill=(214, 22, 66))
    fnt = f(max(10, H//34))
    for li in range(26):
        yy = int(H*0.06) + li*int(H/30)
        if yy > H-20: break
        d.text((int(W*0.42), yy), " ".join(random.Random(li).sample(WORDS, 4)), font=fnt, fill=(10, 10, 20))
    save(f"docblock_{W}x{H}.png", im)
print(len([n for n in os.listdir(D) if n.endswith('.png')]), "candidates")
