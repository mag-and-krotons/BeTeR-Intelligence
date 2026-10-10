"""The triad tower against how much text it counts. A letter is read as the middle of its triads (x-k, x, x+k),
k = 1..6, the ends of the triads counted together (flowtriads.py), on the same 200,000 held-out letters, with the
number of training letters counted from 240 thousand to all of them. beta for each level is chosen on 50,000
held-aside training letters that are never counted. SMOKE=1 runs a small sample first.
"""
import json, os, time
import numpy as np
import textcells as X
from flowmodel import flow
import flowpm3 as F3
import flowtriads as TR

SMOKE = os.environ.get("SMOKE") == "1"
HERE = os.path.dirname(os.path.abspath(__file__))
BUDGETS = [240_000, 960_000] if SMOKE else [240_000, 960_000, 3_840_000, 15_360_000, None]

s = flow(X.S)
tr, te = X.sides(64); Tm = X.T; K = TR.K
okr = lambda p: (p + K < len(s)) & np.all([Tm[np.minimum(p + d, len(s) - 1)] == Tm[p] for d in range(1, K + 1)], 0)
tr = tr[(s[tr] < 26) & okr(tr)]; te = te[(s[te] < 26) & okr(te)]
val = tr[np.linspace(7, len(tr) - 8, 2_000 if SMOKE else 50_000).astype(np.int64)]
pool = np.setdiff1d(tr, val)
jd = te[np.linspace(0, len(te) - 1, 5_000 if SMOKE else 200_000).astype(np.int64)]
out = []
for b in BUDGETS:
    t0 = time.time()
    cp = pool if b is None else pool[np.linspace(0, len(pool) - 1, b).astype(np.int64)]
    P0 = np.bincount(s[cp], minlength=27).astype(np.float64); P0l = P0[:26] / P0[:26].sum()
    T = {k: TR.table(TR.ends(s, cp, k), s[cp]) for k in range(1, K + 1)}
    betas, aside = TR.choose(T, np.tile(P0l, (len(val), 1)), s, val, s[val])
    levels = TR.tower(T, np.tile(P0l, (len(jd), 1)), s, jd, betas)
    r = dict(letters_counted=int(len(cp)), betas=betas,
             held_out_by_triads={k: dict(bits=round(F3.bits(P, s[jd]), 4), right=round(100 * float((P.argmax(1) == s[jd]).mean()), 2)) for k, P in enumerate(levels) if k},
             seconds=round(time.time() - t0, 1))
    print(json.dumps(r), flush=True); out.append(r); del T
json.dump(out, open(os.path.join(HERE, "results", "curve_tower_smoke.json" if SMOKE else "curve_tower.json"), "w"), indent=1)
