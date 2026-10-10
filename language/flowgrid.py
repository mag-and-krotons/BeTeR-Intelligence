"""The flow model with Panini's grid amalgamated.
Panini mapped the sounds by where the breath is shaped and how. Each English letter takes its place in that grid:
  place  throat (k c q g h x a), palate (j y i), the r-row (r: English has no true retroflex), teeth (t d n l s z),
         lips (p b m f v w u), and the junction places of the two junction vowels, e (throat + palate: a + i) and o
         (throat + lips: a + u); silence its own
  how    unvoiced (k c q t p), voiced (g j d b), nose (n m), semivowel (y r l v w), hiss or breath (s z x h f),
         simple vowel (a i u), junction vowel (e o, made by sandhi: a + i, a + u); silence its own
Where a path of letters has been walked rarely, its flow leans on the flow of the same path read through the grid, by
place and by how, as well as on the shorter path: sounds of one row or column share what they have learned, and the
place path carries sandhi, a sound taking the place of the sound it meets.
    P_n(x | path) = (count(path, x) + beta * Q_n(x)) / (count(path) + beta)
    Q_n = w_place * G_place,n + w_how * G_how,n + (1 - w_place - w_how) * P_{n-1}(x | shorter path)
    G_kind,n(x) = (count(grid path, x) + beta * P_{n-1}(x | shorter path)) / (count(grid path) + beta)
Paths up to 9 nodes (the plain flow gained under 0.003 bits a node beyond 9). Weights, beta and length chosen on 50,000
held-aside training positions; counted over 3.84 million positions; judged once on the same 200,000 held-out letters,
beside the plain flow at the same lengths. Control: scrambled text.
"""
import json, os, sys, time
import numpy as np
import textcells as X
from flowmodel import flow, keys, N_COUNT

ORDERS = list(range(1, 10))
BETAS = [8, 16, 32, 64, 128]
WS = [(0, 0), (0.2, 0), (0, 0.2), (0.2, 0.2), (0.4, 0), (0, 0.4), (0.3, 0.3), (0.5, 0.25), (0.25, 0.5)]
PLACE = {**{c: 0 for c in "kcqghxa"}, **{c: 1 for c in "jyi"}, "r": 2, **{c: 3 for c in "tdnlsz"}, **{c: 4 for c in "pbmfvwu"}, "e": 5, "o": 6}
HOW = {**{c: 0 for c in "kcqtp"}, **{c: 1 for c in "gjdb"}, **{c: 2 for c in "nm"}, **{c: 3 for c in "yrlvw"}, **{c: 4 for c in "szxhf"},
       **{c: 5 for c in "aiu"}, **{c: 6 for c in "eo"}}


def mapper(table, silence_code):
    m = np.array([table[chr(97 + i)] for i in range(26)] + [silence_code], np.int64)
    return m


def count(s_hist, s_next, pos, n):
    k = keys(s_hist, pos, n); x = s_next[pos]
    u, inv = np.unique(k, return_inverse=True)
    C = np.zeros((len(u), 27), np.float32); np.add.at(C, (inv, x), 1.0)
    return u, C


def look(table, s_hist, pos, n):
    u, C = table; k = keys(s_hist, pos, n)
    j = np.clip(np.searchsorted(u, k), 0, len(u) - 1); seen = u[j] == k
    c = np.where(seen[:, None], C[j], 0.0).astype(np.float64)
    return c, c.sum(1, keepdims=True)


def predict(T, P0, hist, pos, n, beta, wp, wh):
    P = np.tile(P0, (len(pos), 1))
    for m in range(1, n + 1):
        Q = P
        if wp or wh:
            gp = [];
            for kind, w in (("place", wp), ("how", wh)):
                if w:
                    c, t = look(T[kind][m], hist[kind], pos, m); gp.append((w, (c + beta * P) / (t + beta)))
            Q = (1 - wp - wh) * P + sum(w * G for w, G in gp)
        c, t = look(T["letter"][m], hist["letter"], pos, m)
        P = (c + beta * Q) / (t + beta)
    L = P[:, :26]
    return L / L.sum(1, keepdims=True)


def bits(L, y):
    return float(-np.log2(L[np.arange(len(y)), y]).mean())


def run(scrambled):
    t0 = time.time()
    s = flow(X.S)
    if scrambled:
        s = s.copy(); np.random.default_rng(99).shuffle(s)
    hist = {"letter": s, "place": mapper(PLACE, 7)[s], "how": mapper(HOW, 7)[s]}
    tr, te = X.sides(64); tr = tr[s[tr] < 26]; te = te[s[te] < 26]
    cp = tr[np.linspace(0, len(tr) - 1, N_COUNT).astype(np.int64)]
    rest = np.setdiff1d(tr, cp); val = rest[np.linspace(0, len(rest) - 1, 50_000).astype(np.int64)]
    jd = te[np.linspace(0, len(te) - 1, 200_000).astype(np.int64)]
    P0 = np.bincount(s[cp], minlength=27).astype(np.float64); P0 /= P0.sum()
    T = {kind: {n: count(hist[kind], s, cp, n) for n in ORDERS} for kind in hist}
    best_plain, best_grid = None, None
    for n in ORDERS:
        for b in BETAS:
            for wp, wh in WS:
                v = bits(predict(T, P0, hist, val, n, b, wp, wh), s[val])
                if (wp, wh) == (0, 0) and (best_plain is None or v < best_plain[0]): best_plain = (v, n, b, 0, 0)
                if best_grid is None or v < best_grid[0]: best_grid = (v, n, b, wp, wh)
    out = dict(text="scrambled" if scrambled else "Webster", seconds=None)
    for name, (v, n, b, wp, wh) in (("plain flow", best_plain), ("flow + Panini's grid", best_grid)):
        L = predict(T, P0, hist, jd, n, b, wp, wh); y = s[jd]
        out[name] = dict(order=n, beta=b, w_place=wp, w_how=wh, aside=round(v, 4), held_out_bits_per_letter=round(bits(L, y), 4),
                         next_letter_right=round(100 * float((L.argmax(1) == y).mean()), 2))
    out["seconds"] = round(time.time() - t0, 1)
    return out


if __name__ == "__main__":
    res = [run(False), run(True)]
    for r in res: print(json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "flowgrid.json"), "w"), indent=1)
