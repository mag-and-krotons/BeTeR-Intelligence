"""The flow read +3/-3, as the digits were read three forward and three reversed.
A letter is the middle of what flows into it and what flows out of it. The flow out of the three nodes before it reads
it forward; the flow back from the three nodes after it reads it in reverse; the letter is what stays between the two.
The grid and the flow are two representations of one sound analysis. Each reading is taken in both: the path of
letters, and the same path through Panini's grid (place and how the breath is shaped). The model weighs the four
readings, forward and reverse, letters and grid, and their difference enters through the weights:
    log P(x) = a*log F_letters + b*log R_letters + c*log F_grid + d*log R_grid - (a + b + c + d - 1) * log P0 + const
F = flow out of the 3 before, R = flow back from the 3 after, each leaning on its shorter paths in proportion beta.
Chosen on 50,000 held-aside training positions; counted over 3.84 million positions; judged once on 200,000 held-out
letters whose 3 neighbours on each side lie on the same side of the split. Control: scrambled text.
"""
import json, os, sys, time, itertools
import numpy as np
import textcells as X
from flowmodel import flow, N_COUNT
import flowgrid as G

K = 3
BETAS = [4, 8, 16, 32, 64]


def keys(h, pos, n, side, base=64):
    k = np.zeros(len(pos), np.int64)
    rng = range(n, 0, -1) if side == "before" else range(1, n + 1)
    for off in rng:
        k = k * base + h[pos - off if side == "before" else pos + off]
    return k


def count(h, s, pos, n, side):
    k = keys(h, pos, n, side); x = s[pos]
    u, inv = np.unique(k, return_inverse=True)
    C = np.zeros((len(u), 27), np.float32); np.add.at(C, (inv, x), 1.0)
    return u, C


def read(T, P0, h, pos, side, beta):
    P = np.tile(P0, (len(pos), 1))
    for m in range(1, K + 1):
        u, C = T[m]; k = keys(h, pos, m, side)
        j = np.clip(np.searchsorted(u, k), 0, len(u) - 1); seen = u[j] == k
        c = np.where(seen[:, None], C[j], 0.0).astype(np.float64)
        P = (c + beta * P) / (c.sum(1, keepdims=True) + beta)
    L = P[:, :26]
    return L / L.sum(1, keepdims=True)


def combine(R, P0l, w):
    a, b, c, d = w
    lp = a * np.log(R["Fl"]) + b * np.log(R["Rl"]) + c * np.log(R["Fg"]) + d * np.log(R["Rg"]) - (a + b + c + d - 1) * np.log(P0l)
    lp -= lp.max(1, keepdims=True); p = np.exp(lp)
    return p / p.sum(1, keepdims=True)


def bits(P, y):
    return float(-np.log2(P[np.arange(len(y)), y]).mean())


def run(scrambled):
    t0 = time.time()
    s = flow(X.S)
    if scrambled:
        s = s.copy(); np.random.default_rng(99).shuffle(s)
    grid = G.mapper(G.PLACE, 7)[s] * 8 + G.mapper(G.HOW, 7)[s]          # one grid cell per node: place and how
    tr, te = X.sides(64)
    Tm = X.T
    okr = lambda p: (p + K < len(s)) & (Tm[np.minimum(p + K, len(s) - 1)] == Tm[p]) & (Tm[np.minimum(p + 1, len(s) - 1)] == Tm[p])
    tr = tr[(s[tr] < 26) & okr(tr)]; te = te[(s[te] < 26) & okr(te)]
    cp = tr[np.linspace(0, len(tr) - 1, N_COUNT).astype(np.int64)]
    rest = np.setdiff1d(tr, cp); val = rest[np.linspace(0, len(rest) - 1, 50_000).astype(np.int64)]
    jd = te[np.linspace(0, len(te) - 1, 200_000).astype(np.int64)]
    P0 = np.bincount(s[cp], minlength=27).astype(np.float64); P0 /= P0.sum(); P0l = P0[:26] / P0[:26].sum()
    T = {(rep, side): {m: count(h, s, cp, m, side) for m in range(1, K + 1)}
         for rep, h in (("l", s), ("g", grid)) for side in ("before", "after")}
    weights = [(1, 0, 0, 0)] + [w for w in itertools.product([0, 0.5, 1], repeat=4) if w[0] or w[1]]
    best = {}
    for beta in BETAS:
        Rv = {"Fl": read(T[("l", "before")], P0, s, val, "before", beta), "Rl": read(T[("l", "after")], P0, s, val, "after", beta),
              "Fg": read(T[("g", "before")], P0, grid, val, "before", beta), "Rg": read(T[("g", "after")], P0, grid, val, "after", beta)}
        for w in weights:
            v = bits(combine(Rv, P0l, w), s[val])
            name = "forward letters only" if w == (1, 0, 0, 0) else "+3/-3 letters" if w[2] == w[3] == 0 else "+3/-3 letters and grid"
            if name not in best or v < best[name][0]: best[name] = (v, beta, w)
    out = dict(text="scrambled" if scrambled else "Webster")
    for name, (v, beta, w) in best.items():
        Rj = {"Fl": read(T[("l", "before")], P0, s, jd, "before", beta), "Rl": read(T[("l", "after")], P0, s, jd, "after", beta),
              "Fg": read(T[("g", "before")], P0, grid, jd, "before", beta), "Rg": read(T[("g", "after")], P0, grid, jd, "after", beta)}
        P = combine(Rj, P0l, w); y = s[jd]
        out[name] = dict(beta=beta, weights_F_R_Fgrid_Rgrid=w, aside=round(v, 4), held_out_bits_per_letter=round(bits(P, y), 4),
                         right=round(100 * float((P.argmax(1) == y).mean()), 2))
    out["seconds"] = round(time.time() - t0, 1)
    return out


if __name__ == "__main__":
    res = [run(False), run(True)]
    for r in res: print(json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "flowpm3.json"), "w"), indent=1)
