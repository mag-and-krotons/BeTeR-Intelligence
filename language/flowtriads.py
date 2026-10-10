"""The digits approach carried whole: the middle and the turn.
In the digit stack three images a, b, c are read forward, (a - b) - c, and reversed, (c - b) - a. Their sum is the
middle b (what stays); their difference is a - c, the turn: how the two ends move together around the middle.
The product of the forward and reverse flows (flowreach.py) is only the middle: it treats what comes before a letter and
what comes after it as independent once the letter is fixed. The turn is the two ends counted together. A letter x0 is
the middle of nested triads (x-1, x0, x+1), (x-2, x0, x+2), (x-3, x0, x+3), ...; each triad sits inside the next, as
each node holds the structure within. The tower counts the ends of the triads jointly, innermost first:
    P_k(x) = (count(ends of triads 1..k, x) + beta_k * P_{k-1}(x)) / (count(ends of triads 1..k) + beta_k)
Two bases: the letter shares (P_0 = how often each letter flows), or the middle of the two one-sided flows read 6
nodes each side (P_0 = F * R / shares). beta_k is chosen level by level on 50,000 held-aside positions. Counted over the
same 3.84 million training positions; judged once on the same 200,000 held-out letters as flowreach.py (whose 6
neighbours on each side lie on the same side of the split). Control: scrambled text.
"""
import json, os, sys, time
import numpy as np
import textcells as X
from flowmodel import flow, N_COUNT
import flowpm3 as F3

K = 6
BETAS = [0.5, 1, 2, 4, 8, 16, 32, 64, 128]


def ends(s, pos, k, code=None):
    """The ends of the triads 1..k around each position as one number, innermost first."""
    c = s if code is None else code
    key = np.zeros(len(pos), np.int64)
    for j in range(1, k + 1):
        key = key * 729 + c[pos - j] * 27 + c[pos + j]
    return key


def table(key, x):
    u, c = np.unique(key * 27 + x, return_counts=True)
    uk, ck = np.unique(key, return_counts=True)
    return u, c.astype(np.float64), uk, ck.astype(np.float64)


def look(tab, key):
    u, c, uk, ck = tab
    j = np.clip(np.searchsorted(uk, key), 0, len(uk) - 1); tot = np.where(uk[j] == key, ck[j], 0.0)
    C = np.zeros((len(key), 26)); seen = np.flatnonzero(tot > 0)
    for l in range(26):
        q = key[seen] * 27 + l; i = np.clip(np.searchsorted(u, q), 0, len(u) - 1)
        C[seen, l] = np.where(u[i] == q, c[i], 0.0)
    return C, tot[:, None]


def tower(T, base, s, pos, betas, code=None):
    P = base.copy(); out = [P]
    for k, b in enumerate(betas, 1):
        C, t = look(T[k], ends(s, pos, k, code)); P = (C + b * P) / (t + b); out.append(P)
    return out


def choose(T, base, s, pos, y, code=None):
    """beta level by level on the held-aside positions."""
    P = base.copy(); betas = []; aside = []
    for k in range(1, K + 1):
        C, t = look(T[k], ends(s, pos, k, code)); best = None
        for b in BETAS:
            Q = (C + b * P) / (t + b); v = F3.bits(Q, y)
            if best is None or v < best[0]: best = (v, b, Q)
        betas.append(best[1]); aside.append(round(best[0], 4)); P = best[2]
    return betas, aside


def setup(scrambled):
    s = flow(X.S)
    if scrambled:
        s = s.copy(); np.random.default_rng(99).shuffle(s)
    tr, te = X.sides(64); Tm = X.T
    okr = lambda p: (p + K < len(s)) & np.all([Tm[np.minimum(p + d, len(s) - 1)] == Tm[p] for d in range(1, K + 1)], 0)
    tr = tr[(s[tr] < 26) & okr(tr)]; te = te[(s[te] < 26) & okr(te)]
    cp = tr[np.linspace(0, len(tr) - 1, N_COUNT).astype(np.int64)]
    rest = np.setdiff1d(tr, cp); val = rest[np.linspace(0, len(rest) - 1, 50_000).astype(np.int64)]
    jd = te[np.linspace(0, len(te) - 1, 200_000).astype(np.int64)]
    P0 = np.bincount(s[cp], minlength=27).astype(np.float64); P0 /= P0.sum(); P0l = P0[:26] / P0[:26].sum()
    return s, cp, val, jd, P0, P0l


def sides_middle(s, cp, P0, P0l, pos, beta=32, n=6):
    F3.K = n
    T = {side: {m: F3.count(s, s, cp, m, side) for m in range(1, n + 1)} for side in ("before", "after")}
    M = np.log(F3.read(T["before"], P0, s, pos, "before", beta)) + np.log(F3.read(T["after"], P0, s, pos, "after", beta)) - np.log(P0l)
    F3.K = 3; M -= M.max(1, keepdims=True); M = np.exp(M)
    return M / M.sum(1, keepdims=True)


def run(scrambled):
    t0 = time.time()
    s, cp, val, jd, P0, P0l = setup(scrambled)
    T = {k: table(ends(s, cp, k), s[cp]) for k in range(1, K + 1)}
    out = dict(text="scrambled" if scrambled else "Webster", distinct_ends={k: int(len(T[k][2])) for k in T})
    yv, yj = s[val], s[jd]
    bases = {"shares": (np.tile(P0l, (len(val), 1)), np.tile(P0l, (len(jd), 1))),
             "middle of the two flows (6 each side)": (sides_middle(s, cp, P0, P0l, val), sides_middle(s, cp, P0, P0l, jd))}
    for name, (bv, bj) in bases.items():
        betas, aside = choose(T, bv, s, val, yv)
        levels = tower(T, bj, s, jd, betas)
        out[f"tower on {name}"] = dict(betas=betas, aside_by_level=aside,
                                       held_out_by_level=[dict(k=k, bits=round(F3.bits(P, yj), 4), right=round(100 * float((P.argmax(1) == yj).mean()), 2))
                                                          for k, P in enumerate(levels)])
        print(name, json.dumps(out[f"tower on {name}"]), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    return out


if __name__ == "__main__":
    res = [run(False), run(True)]
    for r in res: print(json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "flowtriads.json"), "w"), indent=1)
