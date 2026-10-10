"""The model updated to the flow. No context window: what comes next is where the current goes.
The text is a flow through 27 nodes, the 26 letters (vowels in their prime cells a 5, e 7, i 11, o 13, u 17) and silence
(sacred cell 1, one node; a run of non-letters is one silence). The current arriving at the next step has come along a
path: the node it is at (order 1), the line it came along (order 2), the two lines before that (order 3), and so on.
The flow out of that path is counted; where a path has been walked rarely, its flow leans on the flow of the shorter
path inside it, in proportion beta (the current's own flow plus the broader flow it is part of):
    P_n(x | path) = (count(path, x) + beta * P_{n-1}(x | shorter path)) / (count(path) + beta),   P_0 = the flow's shares.
The order and beta are chosen on 50,000 held-aside training positions. Counted over 3.84 million positions, as many as
the stack wrote; judged once on the same 200,000 held-out letters as the stack (next letter, among the 26 letters).
Control: the flow of the scrambled text.
Run: python3 flowmodel.py
"""
import json, os, sys, time
import numpy as np
import textcells as X

N_COUNT = 3_840_000
ORDERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
BETAS = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def flow(stream):
    return np.where(stream >= 26, 26, stream).astype(np.int64)     # silence one node


def keys(s, pos, n):
    """The path of n nodes ending just before each position, as one number."""
    k = np.zeros(len(pos), np.int64)
    for back in range(n, 0, -1):
        k = k * 27 + s[pos - back]
    return k


def count(s, pos, n):
    """How the current flows out of each path of n nodes: a table of counts per path."""
    k = keys(s, pos, n); x = s[pos]
    u, inv = np.unique(k, return_inverse=True)
    C = np.zeros((len(u), 27), np.float64); np.add.at(C, (inv, x), 1.0)
    return u, C


def predict(tables, P0, s, pos, n, beta):
    """The letter shares (26) at each position, from the paths of order n down to 0."""
    P = np.tile(P0, (len(pos), 1))
    for m in range(1, n + 1):
        u, C = tables[m]; k = keys(s, pos, m)
        j = np.searchsorted(u, k); j = np.clip(j, 0, len(u) - 1); seen = u[j] == k
        c = np.where(seen[:, None], C[j], 0.0); tot = c.sum(1, keepdims=True)
        P = (c + beta * P) / (tot + beta)
    L = P[:, :26]
    return L / L.sum(1, keepdims=True)


def bits(L, y):
    return float(-np.log2(L[np.arange(len(y)), y]).mean())


def run(scrambled):
    t0 = time.time()
    s = flow(X.S)
    if scrambled:
        s = s.copy(); np.random.default_rng(99).shuffle(s)
    tr, te = X.sides(64)
    tr = tr[s[tr] < 26]; te = te[s[te] < 26]
    cp = tr[np.linspace(0, len(tr) - 1, N_COUNT).astype(np.int64)]
    rest = np.setdiff1d(tr, cp); val = rest[np.linspace(0, len(rest) - 1, 50_000).astype(np.int64)]
    jd = te[np.linspace(0, len(te) - 1, 200_000).astype(np.int64)]
    # the flow is counted over every training step that ends at the counted positions (letters and the silences before them)
    P0 = np.bincount(s[cp], minlength=27).astype(np.float64); P0 /= P0.sum()
    tables = {n: count(s, cp, n) for n in ORDERS}
    best = None; aside = {}
    for n in ORDERS:
        for b in BETAS:
            v = bits(predict(tables, P0, s, val, n, b), s[val]); aside[f"{n},{b}"] = round(v, 4)
            if best is None or v < best[0]: best = (v, n, b)
    v, n, b = best
    L = predict(tables, P0, s, jd, n, b)
    y = s[jd]
    acc = float((L.argmax(1) == y).mean())
    unig = P0[:26] / P0[:26].sum()
    return dict(text="scrambled" if scrambled else "Webster", order=n, beta=b, aside=round(v, 4),
                held_out_bits_per_letter=round(bits(L, y), 4), next_letter_right=round(100 * acc, 2),
                unigram_bits=round(float(-(unig * np.log2(unig)).sum()), 4),
                aside_by_order={o: min(val for kk, val in aside.items() if kk.startswith(f"{o},")) for o in ORDERS},
                seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    out = [run(False), run(True)]
    for r in out: print(json.dumps(r))
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "flowmodel.json"), "w"), indent=1)
