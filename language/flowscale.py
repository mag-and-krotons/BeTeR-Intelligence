"""The scale above: the word as a node, read by the same triad. A letter's word, with each candidate letter put in its
place, sits between the word before it and the word after it: the word triad (w-1, w0(x), w+1).
    Q1(x)  = (count(w0(x)) + b1 * shares) / (...)                              the word itself
    QL, QR = the word with the word before it, the word with the word after  (each leaning on Q1): the two sides
    Q2     = QL * QR / Q1                                                     their middle (still)
    Q3(x)  = (count(w-1, w0(x), w+1) + b3 * Q2) / (...)                       the triad counted together (turn)
Words are counted at every fifth word (about the letters' 3.84 million budget), all in training text. The letter mix
(S, TWb, BZ and the return reading, flowcopy.py) and Q3 are then mixed with weights chosen on the held-aside positions
for each depth the word triad was seen at. Control: scrambled text. SMOKE=1 runs a small sample first.
"""
import json, os, sys, time, itertools
import numpy as np
import textcells as X
import flowtriads as TR
import flowword as FW
import flowcopy as FC

SMOKE = os.environ.get("SMOKE") == "1"
COPY = os.environ.get("COPY", "flowcopy")                 # which return reading to stand on
HERE = FW.HERE
B = np.uint64(1000003); K1 = np.uint64(0x9E3779B97F4A7C15); K2 = np.uint64(0xC2B2AE3D27D4EB4F); K3 = np.uint64(0x165667B19E3779F9)
BETAS = [0.25, 1, 4, 16, 64]


def words(s):
    isl = s < 26
    st = np.flatnonzero(isl & ~np.r_[False, isl[:-1]]); en = np.flatnonzero(isl & ~np.r_[isl[1:], False])
    ln = en - st + 1; h = np.zeros(len(st), np.uint64)
    with np.errstate(over="ignore"):
        for o in range(int(ln.max())):
            a = ln > o; h[a] = h[a] * B + (s[st[a] + o].astype(np.uint64) + np.uint64(1))
    wid = np.full(len(s), -1, np.int64); mark = np.zeros(len(s) + 1, np.int64); mark[st] += 1
    idx = np.cumsum(mark[:-1]) - 1; wid[isl] = idx[isl]
    return st, en, ln, h, wid


def table(key):
    u, c = np.unique(key, return_counts=True)
    return u, c.astype(np.float64)


def look(tab, key):
    u, c = tab; j = np.clip(np.searchsorted(u, key), 0, len(u) - 1)
    return np.where(u[j] == key, c[j], 0.0)


def keys(hl, h0, hr):
    with np.errstate(over="ignore"):
        return dict(uni=h0, left=hl * K1 + h0, right=h0 * K2 + hr, tri=(hl * K1 + h0) * K3 + hr)


def counts_for(T, s, words_, pos):
    st, en, ln, h, wid = words_
    wi = wid[pos]; o = pos - st[wi]; L = ln[wi]
    with np.errstate(over="ignore"):
        pw = np.ones(len(pos), np.uint64)
        for e in range(int(L.max())):
            pw = np.where(L - 1 - o > e, pw * B, pw)
        base = h[wi] - (s[pos].astype(np.uint64) + np.uint64(1)) * pw
    hl = h[np.maximum(wi - 1, 0)]; hr = h[np.minimum(wi + 1, len(h) - 1)]
    side = X.T
    ok = (wi > 0) & (wi + 1 < len(h)) & (side[st[np.maximum(wi - 1, 0)]] == side[pos]) & (side[en[np.minimum(wi + 1, len(h) - 1)]] == side[pos])
    C = {k: np.zeros((len(pos), 26)) for k in ("uni", "left", "right", "tri")}
    for x in range(26):
        with np.errstate(over="ignore"):
            hx = base + (np.uint64(x) + np.uint64(1)) * pw
        kk = keys(hl, hx, hr)
        for k in C:
            C[k][:, x] = look(T[k], kk[k]) * (ok if k != "uni" else 1)
    return C


def tower(C, P0l, b):
    b1, bl, br, b3 = b
    n = len(C["uni"]); base = np.tile(P0l, (n, 1))
    lean = lambda c, q, beta: (c + beta * q) / (c.sum(1, keepdims=True) + beta)
    Q1 = lean(C["uni"], base, b1); QL = lean(C["left"], Q1, bl); QR = lean(C["right"], Q1, br)
    Q2 = QL * QR / Q1; Q2 /= Q2.sum(1, keepdims=True)
    return lean(C["tri"], Q2, b3)


def depth(C):
    d = np.zeros(len(C["uni"]), np.int64)
    d = np.where(C["uni"].sum(1) > 0, 1, d); d = np.where((C["left"].sum(1) > 0) | (C["right"].sum(1) > 0), 2, d)
    return np.where(C["tri"].sum(1) > 0, 3, d)


def run(scrambled):
    t0 = time.time()
    s, cp, val, jd, P0, P0l = TR.setup(scrambled)
    tag = "scrambled" if scrambled else "webster"
    W_ = words(s); st, en, ln, h, wid = W_
    side = X.T
    c = np.arange(1, len(h) - 1); c = c[(c % 5 == 0) & (side[st[c - 1]] == 0) & (side[en[c + 1]] == 0) & (side[st[c]] == 0) & (side[en[c]] == 0)]
    kk = keys(h[c - 1], h[c], h[c + 1]); T = {k: table(v) for k, v in kk.items()}
    z = np.load(os.path.join(HERE, f"flowword2_{tag}.npz")); zc = np.load(os.path.join(HERE, f"{COPY}_{tag}.npz"))
    gate = {int(k): v for k, v in json.load(open(os.path.join(HERE, f"{COPY}.json")))[0 if not scrambled else 1]["gate (S, TWb, BZ, C)"].items()}
    names = ["S", "TWb", "BZ", "C"]
    R, pos, y, dep = {}, {}, {}, {}
    for p in ("aside", "judged"):
        R[p] = {n: z[f"{p}_{n}"].astype(np.float64) for n in names[:3]}; R[p]["C"] = zc[f"{p}_C"].astype(np.float64)
        pos[p] = z[f"{p}_pos"]; y[p] = z[f"{p}_y"]; dep[p] = zc[f"{p}_depth"]
        if SMOKE:
            m = slice(0, 4000); R[p] = {n: v[m] for n, v in R[p].items()}; pos[p] = pos[p][m]; y[p] = y[p][m]; dep[p] = dep[p][m]
    Lm = {}
    for p in R:                                                    # the letter mix, gated by the return's depth (flowcopy.py)
        P = np.zeros_like(R[p]["S"])
        for m in np.unique(dep[p]):
            sel = dep[p] == m; P[sel] = FC.mix({n: v[sel] for n, v in R[p].items()}, P0l, names, gate.get(int(m), (0.5, 0.5, 0.5, 0)))
        Lm[p] = P
    Cw = {p: counts_for(T, s, W_, pos[p]) for p in R}
    bw = min(((FW.bits(tower(Cw["aside"], P0l, b), y["aside"]), b) for b in itertools.product(BETAS, repeat=4)))[1]
    Wr = {p: tower(Cw[p], P0l, bw) for p in R}; wd = {p: depth(Cw[p]) for p in R}
    grid = [0, 0.25, 0.5, 0.75, 1, 1.25]
    def mx(a, b, w):
        lp = w[0] * np.log(np.maximum(a, 1e-300)) + w[1] * np.log(np.maximum(b, 1e-300)) - (w[0] + w[1] - 1) * np.log(P0l)
        lp -= lp.max(1, keepdims=True); q = np.exp(lp); return q / q.sum(1, keepdims=True)
    wg = {}
    for d in np.unique(wd["aside"]):
        sel = wd["aside"] == d
        wg[int(d)] = min(((FW.bits(mx(Lm["aside"][sel], Wr["aside"][sel], w), y["aside"][sel]), w) for w in itertools.product(grid, grid) if w[0] > 0))[1] if sel.sum() >= 30 else (1, 0)
    Pj = np.zeros_like(Lm["judged"])
    for d in np.unique(wd["judged"]):
        sel = wd["judged"] == d; Pj[sel] = mx(Lm["judged"][sel], Wr["judged"][sel], wg.get(int(d), (1, 0)))
    yj = y["judged"]
    sc = lambda P: dict(bits=round(FW.bits(P, yj), 4), right=round(100 * float((P.argmax(1) == yj).mean()), 2))
    out = dict(text=tag, word_triads_counted=int(len(c)), word_betas=bw, letters_mix=sc(Lm["judged"]), word_triad_alone=sc(Wr["judged"]), with_the_scale_above=sc(Pj),
               share_by_word_depth={int(d): round(100 * float((wd["judged"] == d).mean()), 2) for d in np.unique(wd["judged"])},
               by_word_depth_right={int(d): [round(100 * float((Lm["judged"].argmax(1) == yj)[wd["judged"] == d].mean()), 1),
                                             round(100 * float((Pj.argmax(1) == yj)[wd["judged"] == d].mean()), 1)] for d in np.unique(wd["judged"])},
               word_gate=wg, seconds=round(time.time() - t0, 1))
    print(json.dumps(out), flush=True)
    np.savez_compressed(os.path.join(HERE, f"flowscale_{COPY}_{'smoke_' if SMOKE else ''}{tag}.npz"), judged_P=Pj.astype(np.float32), judged_y=yj, judged_pos=pos["judged"])
    return out


if __name__ == "__main__":
    res = [run(False)] + ([] if SMOKE else [run(True)])
    json.dump(res, open(os.path.join(HERE, f"flowscale_{COPY}_smoke.json" if SMOKE else f"flowscale_{COPY}.json"), "w"), indent=1)
