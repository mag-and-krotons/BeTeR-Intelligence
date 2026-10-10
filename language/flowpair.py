"""Next-letter prediction read both ways: the pair for prediction, no future letter used.
Forward: the flow out of the path before the letter, P_F(x | h_-1 .. h_-n) (flowmodel.py).
Reverse: the same question read from the other end. Start at the candidate x and walk the flow back over the path:
    P_R(x | h) ~ P0(x) * prod_{j=1..n} P_back(h_-j | h_-j+1 .. h_-1, x)
each factor the flow back into a node from the nodes after it (the path ending in x), leaning on its shorter paths in
proportion beta. With exact counts both readings are the same number; read through a finite flow they differ, and
their difference is the turn. The middle: log P = a log P_F + b log P_R - (a + b - 1) log P0, a and b chosen on 50,000
held-aside positions, with the paths' lengths and betas. Counted over 3.84 million training positions (the reverse
flow over 3.84 million training nodes, letters and silences); judged once on the same 200,000 held-out letters as
flowmodel.py. Control: scrambled text.
"""
import json, os, sys, time, itertools
import numpy as np
import textcells as X
from flowmodel import flow, N_COUNT

NF = [6, 9, 12]
NR = [3, 4, 6, 9]
BETAS = [4, 8, 16, 32, 64, 128]


def pkey(s, start, m, step):
    """The path s[start], s[start + step], ... (m nodes, nearest first) as one number."""
    k = np.zeros(len(start), np.int64)
    for i in range(m):
        k = k * 27 + s[start + i * step]
    return k


def table(key, x):
    u, c = np.unique(key * 27 + x, return_counts=True)
    uk, ck = np.unique(key, return_counts=True)
    return u, c.astype(np.float64), uk, ck.astype(np.float64)


def total(tab, key):
    _, _, uk, ck = tab
    j = np.clip(np.searchsorted(uk, key), 0, len(uk) - 1)
    return np.where(uk[j] == key, ck[j], 0.0)


def pair_count(tab, key, x):
    u, c, _, _ = tab; q = key * 27 + x
    i = np.clip(np.searchsorted(u, q), 0, len(u) - 1)
    return np.where(u[i] == q, c[i], 0.0)


def forward(TF, P0l, s, pos, n, beta):
    P = np.tile(P0l, (len(pos), 1))
    for m in range(1, n + 1):
        key = pkey(s, pos - 1, m, -1); t = total(TF[m], key)[:, None]
        C = np.stack([pair_count(TF[m], key, l) for l in range(26)], 1)
        P = (C + beta * P) / (t + beta)
    return P / P.sum(1, keepdims=True)


def reverse(TR, Q0, P0l, s, pos, n, beta):
    """log P_R over the 26 candidates (unnormalised)."""
    L = np.tile(np.log(P0l), (len(pos), 1))
    for j in range(1, n + 1):
        y = s[pos - j]; p = Q0[y]                                   # the flow back into h_-j, from its shares upward
        for m in range(1, j):
            key = pkey(s, pos - j + 1, m, 1)
            p = (pair_count(TR[m], key, y) + beta * p) / (total(TR[m], key) + beta)
        base = pkey(s, pos - j + 1, j - 1, 1) if j > 1 else np.zeros(len(pos), np.int64)
        for x in range(26):
            key = base * 27 + x
            L[:, x] += np.log((pair_count(TR[j], key, y) + beta * p) / (total(TR[j], key) + beta))
    return L


def norm(L):
    L = L - L.max(1, keepdims=True); P = np.exp(L)
    return P / P.sum(1, keepdims=True)


def bits(P, y):
    return float(-np.log2(P[np.arange(len(y)), y]).mean())


def run(scrambled):
    t0 = time.time()
    s = flow(X.S)
    if scrambled:
        s = s.copy(); np.random.default_rng(99).shuffle(s)
    tr, te = X.sides(64)
    allr, _ = X.sides(64, letters_only=False)                  # the reverse flow runs into letters and silences alike
    tr = tr[s[tr] < 26]; te = te[s[te] < 26]
    cp = tr[np.linspace(0, len(tr) - 1, N_COUNT).astype(np.int64)]
    rest = np.setdiff1d(tr, cp); val = rest[np.linspace(0, len(rest) - 1, 50_000).astype(np.int64)]
    jd = te[np.linspace(0, len(te) - 1, 200_000).astype(np.int64)]
    # the reverse flow: training nodes whose next max(NR) nodes lie on the training side too (sides() looks back 64)
    Tm = X.T; okn = np.all([Tm[np.minimum(allr + d, len(s) - 1)] == Tm[allr] for d in range(1, max(NR) + 1)], 0) & (allr + max(NR) < len(s))
    rp = allr[okn]; rp = rp[np.linspace(0, len(rp) - 1, N_COUNT).astype(np.int64)]
    P0 = np.bincount(s[cp], minlength=27).astype(np.float64); P0l = P0[:26] / P0[:26].sum()
    Q0 = np.bincount(s[rp], minlength=27).astype(np.float64); Q0 /= Q0.sum()
    TF = {m: table(pkey(s, cp - 1, m, -1), s[cp]) for m in range(1, max(NF) + 1)}
    TR = {m: table(pkey(s, rp + 1, m, 1), s[rp]) for m in range(1, max(NR) + 1)}
    yv, yj = s[val], s[jd]
    bestF = min(((bits(forward(TF, P0l, s, val, n, b), yv), n, b) for n in NF for b in BETAS))
    bestR = min(((bits(norm(reverse(TR, Q0, P0l, s, val, n, b)), yv), n, b) for n in NR for b in BETAS))
    Fv = forward(TF, P0l, s, val, bestF[1], bestF[2]); Rv = reverse(TR, Q0, P0l, s, val, bestR[1], bestR[2])
    lp0 = np.log(P0l)
    grid = [0, 0.25, 0.5, 0.75, 1, 1.25]
    bestM = min(((bits(norm(a * np.log(Fv) + b * Rv - (a + b - 1) * lp0), yv), a, b) for a, b in itertools.product(grid, grid) if a + b > 0))
    Fj = forward(TF, P0l, s, jd, bestF[1], bestF[2]); Rj = reverse(TR, Q0, P0l, s, jd, bestR[1], bestR[2])
    out = dict(text="scrambled" if scrambled else "Webster")
    for name, P, choice in (("forward", Fj, dict(n=bestF[1], beta=bestF[2], aside=round(bestF[0], 4))),
                            ("reverse (read back from the candidate)", norm(Rj), dict(n=bestR[1], beta=bestR[2], aside=round(bestR[0], 4))),
                            ("middle of the pair", norm(bestM[1] * np.log(Fj) + bestM[2] * Rj - (bestM[1] + bestM[2] - 1) * lp0),
                             dict(a=bestM[1], b=bestM[2], aside=round(bestM[0], 4)))):
        out[name] = dict(**choice, held_out_bits_per_letter=round(bits(P, yj), 4), next_letter_right=round(100 * float((P.argmax(1) == yj).mean()), 2))
        print(name, out[name], flush=True)
    Pf, Pr = Fj, norm(Rj)
    dis = Pf.argmax(1) != Pr.argmax(1)
    out["turn"] = dict(disagree=round(100 * float(dis.mean()), 2), forward_right_when_disagree=round(100 * float((Pf.argmax(1) == yj)[dis].mean()), 2),
                       reverse_right_when_disagree=round(100 * float((Pr.argmax(1) == yj)[dis].mean()), 2))
    out["seconds"] = round(time.time() - t0, 1)
    return out


if __name__ == "__main__":
    res = [run(False), run(True)]
    for r in res: print(json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "flowpair.json"), "w"), indent=1)
