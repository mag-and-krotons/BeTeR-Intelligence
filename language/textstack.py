"""His six-image stack carried to what language models approximate: which character comes next.

Text is a line of cells, each holding one of 83 characters. The class is the next character; its "images" are the cells
before it. For each next character, three of its contexts a, b, c are read forward, (a - b) - c, and reversed,
(c - b) - a: their sum gives the middle (what stays), their difference a - c (how its contexts turn). Each character
keeps its still pattern (the mean of its middles), its own turning directions (the main directions of its a - c) and
how many times it came (its count). A new context is read against every character: what is left of it after the
character's still pattern and turns, r_c; its weight is count_c * exp(-r_c / T), the Boltzmann weight. No training: the
memory is written in one pass, each character's on its own, so it can always be added to.

Two ways of reading a context, both as cells filled or not (counts):
  cells8   the last 8 characters, each its own cell (8 x 83)
  halving  the last 32 characters counted in blocks that double going back, 1, 1, 2, 4, 8, 16 (six scales, 6 x 83),
           as the digits' bitmaps were counted in blocks
A second scale was tried, each character's cells against their own spread with its log-spread: it failed (88 bits a
character on 60,000 written positions: a cell a character has never seen has almost no spread and pushes every context
that touches it far away), so the run uses his flat reading. The way of reading, the scale, the number of turning directions
k and the temperature T are chosen inside the training text only
(a held-aside 50,000 positions), then the memory is asked once about the 200,000 held-out positions the trained
networks were judged on (triad-pattern/pattern.py: same text, same split). Control: the same with the text scrambled.
Run: python3 textstack.py
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
V = 83                              # values per cell for the one-hot reading
NCLASS = 83                         # the characters that can come next
WAYS = {"cells8": [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)],
        "halving": [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32)]}
N_WRITE = 3_840_000                 # as many positions as the trained networks saw (30,000 steps of 128)
KS = [0, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
TS = np.geomspace(0.05, 5.0, 40)


def load(scrambled):
    arr = np.load(os.path.join(DATA, "text.npy"))
    if scrambled:
        arr = arr.copy(); np.random.default_rng(99).shuffle(arr)
    z = np.load(os.path.join(DATA, "positions.npz"))
    return arr, z["train"], z["test"]


def cells(arr, pos, way):
    """Each context as counts: block j holds how many of each character lie in it (positions p-a .. p-b)."""
    B = WAYS[way]; X = np.zeros((len(pos), len(B) * V), np.float32); rows = np.arange(len(pos))
    for j, (a, b) in enumerate(B):
        for back in range(a, b + 1):
            X[rows, j * V + arr[pos - back].astype(np.int64)] += 1.0          # one cell per row per step back: no repeats within a step
    return X


def write(arr, pos, way, kmax, chunk=50_000):
    """The memory: per character, its count, the mean of its middles and their spread per cell, and the turning
    directions of its a - c, measured flat and measured against the spread."""
    y = arr[pos]; D = cells(arr, pos[:1], way).shape[1]
    order = np.argsort(y, kind="stable"); pos, y = pos[order], y[order]
    mem = {}
    for c in range(NCLASS):
        pc = pos[y == c]; n = len(pc)
        if n < 3:
            continue
        t = n // 3
        S1 = np.zeros(D); S2 = np.zeros(D)
        for i in range(0, t, chunk):                            # the middles: what stays, and how much it spreads
            b = cells(arr, pc[t + i:t + min(t, i + chunk)], way)
            S1 += b.sum(0); S2 += (b * b).sum(0)
        m = S1 / t; var = np.maximum(S2 / t - m * m, 0) + 1.0 / t
        w = 1 / np.sqrt(var)
        Cf = np.zeros((D, D)); Cw = np.zeros((D, D))
        for i in range(0, t, chunk):
            j = min(t, i + chunk)
            a = cells(arr, pc[i:j], way); b = cells(arr, pc[t + i:t + j], way); cc = cells(arr, pc[2 * t + i:2 * t + j], way)
            fwd = (a - b) - cc; rev = (cc - b) - a
            turn = ((fwd - rev) / 2).astype(np.float64)
            Cf += turn.T @ turn; tw = turn * w; Cw += tw.T @ tw
        _, Uf = np.linalg.eigh(Cf / t); _, Uw = np.linalg.eigh(Cw / t)
        mem[c] = dict(n=n, m=m.astype(np.float32), w=w.astype(np.float32), logdet=float(np.log(var).sum()),
                      U_flat=Uf[:, ::-1][:, :kmax].astype(np.float32), U_spread=Uw[:, ::-1][:, :kmax].astype(np.float32))
    return mem


def residuals_all(mem, X, scale, ks):
    """What is left of each context after each character's still pattern and its first k turns, for every k at once."""
    cs = sorted(mem); R = {k: np.zeros((len(X), len(cs)), np.float32) for k in ks}
    for i, c in enumerate(cs):
        M = mem[c]; Dd = X - M["m"]
        if scale == "spread":
            Dd = Dd * M["w"]
        U = M["U_" + scale]
        full = (Dd * Dd).sum(1) + (M["logdet"] if scale == "spread" else 0.0)
        cum = np.concatenate([np.zeros((len(X), 1), np.float32), np.cumsum((Dd @ U[:, :max(ks)]) ** 2, 1)], 1)
        for k in ks:
            R[k][:, i] = full - cum[:, k]
    return cs, R


def logp_of(mem, cs, R, T):
    logn = np.log(np.array([mem[c]["n"] for c in cs], np.float64))
    L = logn[None] - R.astype(np.float64) / T
    L -= L.max(1, keepdims=True)
    return L - np.log(np.exp(L).sum(1, keepdims=True))


def judge(mem, X, y, scale, k, T):
    cs, R = residuals_all(mem, X, scale, [k]); logp = logp_of(mem, cs, R[k], T)
    idx = {c: i for i, c in enumerate(cs)}; col = np.array([idx.get(t, -1) for t in y])
    bpc = float(-logp[np.arange(len(y)), col].mean() / np.log(2))
    acc = float((np.array(cs)[logp.argmax(1)] == y).mean())
    return bpc, acc


def choose(mem, X, y, scale):
    best = (np.inf, None, None)
    cs, RR = residuals_all(mem, X, scale, KS)
    idx = {c: i for i, c in enumerate(cs)}; col = np.array([idx.get(t, -1) for t in y])
    for k in KS:
        for T in TS:
            b = float(-logp_of(mem, cs, RR[k], T)[np.arange(len(y)), col].mean() / np.log(2))
            if b < best[0]:
                best = (b, k, float(T))
    return best


def run(scrambled):
    t0 = time.time()
    arr, tr, te = load(scrambled)
    write_pos = tr[np.linspace(0, len(tr) - 1, N_WRITE).astype(np.int64)]
    rest = np.setdiff1d(tr, write_pos)
    val = rest[np.linspace(0, len(rest) - 1, 50_000).astype(np.int64)]
    judged = te[np.linspace(0, len(te) - 1, 200_000).astype(np.int64)]
    out = dict(text="scrambled" if scrambled else "Webster 1913", written_positions=N_WRITE)
    best = None; mems = {}
    for way in WAYS:
        tw = time.time(); mem = write(arr, write_pos, way, max(KS)); tw = time.time() - tw
        Xv = cells(arr, val, way); mems[way] = mem; out[f"write_seconds_{way}"] = round(tw, 1)
        for scale in ("flat",):                                 # spread failed at 60,000 written: 88 bpc (see notes)
            b, k, T = choose(mem, Xv, arr[val], scale)
            out[f"aside_bpc_{way}_{scale}"] = [round(b, 4), k, round(T, 3)]
            if best is None or b < best[0]:
                best = (b, way, scale, k, T)
    _, way, scale, k, T = best
    tj = time.time(); bpc, acc = judge(mems[way], cells(arr, judged, way), arr[judged], scale, k, T); tj = time.time() - tj
    out.update(way=way, scale=scale, k_turns=k, T=round(T, 4), held_out_bpc=round(bpc, 4), next_char_accuracy=round(100 * acc, 2),
               judge_seconds=round(tj, 1), total_seconds=round(time.time() - t0, 1))
    y = arr[write_pos]; p = np.bincount(y, minlength=NCLASS) / len(y); p = p[p > 0]
    out["unigram_bits"] = round(float(-(p * np.log2(p)).sum()), 4)
    return out


if __name__ == "__main__":
    for scr in (False, True):
        r = run(scr); print(json.dumps(r), flush=True)
        with open(os.path.join(HERE, "textstack.jsonl"), "a") as f: f.write(json.dumps(r) + "\n")
