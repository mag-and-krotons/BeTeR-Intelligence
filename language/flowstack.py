"""His digit stack on top of the +3/-3 flow: the middle and the turns.
Each position becomes an image of its readings: the flow out of the 3 before (forward) and the flow back from the 3
after (reverse), in letters and in Panini's grid, each a share over the 26 letters, read as logarithms: 4 x 26 cells.
Each letter keeps the still pattern and the turns of its images (three read forward (a - b) - c, the same three
reversed (c - b) - a: their sum gives the middle, their difference a - c the turn) and its count; an image is read by
count * exp(-r / T). The flow is counted over 3.84 million training positions; the stack is written from 1 million
other training positions (so no image reads its own count); k and T are chosen on 50,000 held-aside positions; judged
once on the same 200,000 held-out letters as flowpm3.py. Control: scrambled text.
"""
import json, os, sys, time
import numpy as np
import textcells as X
import textstack as TS
from flowmodel import flow, N_COUNT
import flowgrid as G
import flowpm3 as F3

N_STACK = 1_000_000


def images(T, P0, s, grid, pos, beta, parts):
    R = {"Fl": F3.read(T[("l", "before")], P0, s, pos, "before", beta), "Rl": F3.read(T[("l", "after")], P0, s, pos, "after", beta),
         "Fg": F3.read(T[("g", "before")], P0, grid, pos, "before", beta), "Rg": F3.read(T[("g", "after")], P0, grid, pos, "after", beta)}
    return np.concatenate([np.log(R[p]) for p in parts], 1).astype(np.float32), R


def run(scrambled, beta=8):
    t0 = time.time()
    s = flow(X.S)
    if scrambled:
        s = s.copy(); np.random.default_rng(99).shuffle(s)
    grid = G.mapper(G.PLACE, 7)[s] * 8 + G.mapper(G.HOW, 7)[s]
    tr, te = X.sides(64); Tm = X.T
    okr = lambda p: (p + 3 < len(s)) & (Tm[np.minimum(p + 3, len(s) - 1)] == Tm[p]) & (Tm[np.minimum(p + 1, len(s) - 1)] == Tm[p])
    tr = tr[(s[tr] < 26) & okr(tr)]; te = te[(s[te] < 26) & okr(te)]
    cp = tr[np.linspace(0, len(tr) - 1, N_COUNT).astype(np.int64)]
    rest = np.setdiff1d(tr, cp)
    sp = rest[np.linspace(0, len(rest) - 1, N_STACK).astype(np.int64)]
    rest2 = np.setdiff1d(rest, sp); val = rest2[np.linspace(0, len(rest2) - 1, 50_000).astype(np.int64)]
    jd = te[np.linspace(0, len(te) - 1, 200_000).astype(np.int64)]
    P0 = np.bincount(s[cp], minlength=27).astype(np.float64); P0 /= P0.sum(); P0l = P0[:26] / P0[:26].sum()
    T = {(rep, side): {m: F3.count(h, s, cp, m, side) for m in range(1, 4)}
         for rep, h in (("l", s), ("g", grid)) for side in ("before", "after")}
    out = dict(text="scrambled" if scrambled else "Webster")
    # the +3/-3 middle alone (product of the two letter readings), for comparison on the same positions
    _, Rj = images(T, P0, s, grid, jd, beta, ["Fl"])
    Pm = F3.combine(Rj, P0l, (1, 1, 0, 0)); y = s[jd]
    out["+3/-3 middle (product)"] = dict(held_out_bits_per_letter=round(F3.bits(Pm, y), 4), right=round(100 * float((Pm.argmax(1) == y).mean()), 2))
    TS.NCLASS = 26; TS.TS = np.geomspace(1e-3, 50, 60)
    for name, parts in (("stack on letters F,R", ["Fl", "Rl"]), ("stack on letters and grid F,R", ["Fl", "Rl", "Fg", "Rg"])):
        Xs, _ = images(T, P0, s, grid, sp, beta, parts); Xv, _ = images(T, P0, s, grid, val, beta, parts); Xj, _ = images(T, P0, s, grid, jd, beta, parts)
        D = Xs.shape[1]; TS.KS = [k for k in [0, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64] if k < D]
        lookup = {int(p): i for i, p in enumerate(sp)}
        TS.cells = lambda arr, pos, way=None, Xs=Xs, lookup=lookup: Xs[[lookup[int(p)] for p in pos]]
        mem = TS.write(s, sp, "x", max(TS.KS))
        b, k, Tt = TS.choose(mem, Xv, s[val], "flat")
        bpl, acc = TS.judge(mem, Xj, s[jd], "flat", k, Tt)
        out[name] = dict(k_turns=k, T=round(Tt, 4), aside=round(b, 4), held_out_bits_per_letter=round(bpl, 4), right=round(100 * acc, 2))
        print(name, out[name], flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    return out


if __name__ == "__main__":
    res = [run(False), run(True)]
    for r in res: print(json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "flowstack.json"), "w"), indent=1)
