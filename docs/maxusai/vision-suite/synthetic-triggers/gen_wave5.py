from PIL import Image
import numpy as np, os
D = os.path.dirname(__file__)
def save(n,a): Image.fromarray(np.clip(a,0,255).astype(np.uint8)).save(os.path.join(D,n)); print(n, end=" ")
def rgb(m): return np.where(m[...,None],255,0).astype(np.uint8).repeat(3,axis=2)

# Geometries at ~1.0 MP, every side divisible by 28 => served NATIVE by 0.7.1 (cap ~1265 tok)
GEOMS = {"1148x868": (868,1148), "1120x896": (896,1120), "1400x700": (700,1400), "868x1148": (1148,868)}
for g,(H,W) in GEOMS.items():
    ys,xs = np.mgrid[0:H,0:W]
    for p in (7,14,28,56):
        save(f"c{p}_{g}.png", rgb(((xs//p)+(ys//p))%2==1))
        save(f"sv{p}_{g}.png", rgb((xs//p)%2==1))
    # patch-grid-aligned worst case: alternate whole 28px tokens
    save(f"tok28_{g}.png", rgb(((xs//28)+(ys//28))%2==1))
    # anti-aligned: 14px offset so every token straddles an edge
    save(f"tok28off_{g}.png", rgb((((xs+14)//28)+((ys+14)//28))%2==1))
