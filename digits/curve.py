"""More is good, but not without end: the six-image stack against how many images each digit sees, and against how
many turning directions each digit keeps.
1. Images per digit n = 3, 6, 12, 24, 48, 96 and all (about 126): n images of each digit drawn from the training
   images (five draws), the memory written from them, read on the same 540 test images. The turning directions are
   chosen inside the drawn images (80/20) when each digit has at least 10; below that each digit keeps all the turns
   its few images have (at most 6).
2. Turning directions k = 0..30 with all training images: the test accuracy at each k (three seeds). This curve is
   shown only to see its shape; the result in stack6.py chooses k inside the training images.
Run: python3 curve.py   (numpy only)
"""
import json, os
import numpy as np
from stack6 import load, split, write, read, choose_k, HERE

X, y = load()
Xtr, Xte, ytr, yte = split(X, y, 0.3, 42)
out = {"images_per_digit": {}, "turns": {}}
for n in (3, 6, 12, 24, 48, 96, 10_000):
    accs = []
    for draw in range(5):
        rng = np.random.default_rng(100 + draw)
        idx = np.concatenate([rng.permutation(np.flatnonzero(ytr == c))[:n] for c in range(10)])
        Xs, ys = Xtr[idx], ytr[idx]
        per = int(np.bincount(ys).min())
        k = choose_k(Xs, ys, draw)[0] if per >= 10 else min(6, per // 3)
        accs.append(float((read(write(Xs, ys, k, rng), Xte) == yte).mean() * 100))
    key = "all" if n == 10_000 else str(n)
    out["images_per_digit"][key] = dict(images_per_digit=int(np.bincount(ys).min()) if n < 10_000 else "all (%d to %d)" % (np.bincount(ytr).min(), np.bincount(ytr).max()),
                                        mean=round(float(np.mean(accs)), 2), sd=round(float(np.std(accs, ddof=1)), 2))
    print(key, out["images_per_digit"][key], flush=True)
for k in range(0, 31):
    accs = [float((read(write(Xtr, ytr, k, np.random.default_rng(s)), Xte) == yte).mean() * 100) for s in (0, 1, 2)]
    out["turns"][k] = round(float(np.mean(accs)), 2)
print("turns", out["turns"], flush=True)
json.dump(out, open(os.path.join(HERE, "results", "curve.json"), "w"), indent=1)
