"""The flow, counted with no training, on the trained network's own task, and against how much text it counts.
Task: the next character of Webster's dictionary among its 83 characters, judged on the same 200,000 held-out
positions as the trained 0.83-million-weight network (pattern.py of the research record: float weights, 30,000 steps of
128 windows = 3.84 million positions seen, about 6,400 s on one CPU core, 2.410 bits per character).
The flow: what comes next is where the current goes. The path of n characters before a position is counted with what
followed it; a rarely walked path leans on the shorter path inside it in proportion beta:
    P_n(x | path) = (count(path, x) + beta * P_{n-1}(x | shorter path)) / (count(path) + beta),  P_0 = the shares.
The longest path (up to 12) and beta are chosen on 50,000 held-aside training positions that are never counted.
Budgets: how many training positions are counted, from 60 thousand to all 26.5 million.
Control: the text scrambled, at 3.84 million. SMOKE=1 runs a small sample first.
"""
import json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
SMOKE = os.environ.get("SMOKE") == "1"
V, NMAX = 83, 12
BETAS = [0.5, 1, 2, 4, 8, 16, 32, 64, 128]
BUDGETS = [60_000, 240_000] if SMOKE else [60_000, 240_000, 960_000, 3_840_000, 15_360_000, None]
B = np.uint64(1000003); K = np.uint64(0x9E3779B97F4A7C15)


def hashes(arr, pos):
    """The paths of 1..NMAX characters before each position, as 64-bit numbers."""
    H = np.zeros((NMAX, len(pos)), np.uint64); h = np.zeros(len(pos), np.uint64)
    with np.errstate(over="ignore"):
        for n in range(1, NMAX + 1):
            h = h * B + (arr[pos - n].astype(np.uint64) + np.uint64(1)); H[n - 1] = h * K
    return H


def table(h, x):
    o = np.lexsort((x, h)); h, x = h[o], x[o]
    new = np.r_[True, (h[1:] != h[:-1]) | (x[1:] != x[:-1])]
    start = np.flatnonzero(new); cnt = np.diff(np.r_[start, len(h)])
    return h[start], x[start], cnt.astype(np.float32)


def look(tab, hq):
    """Counts of each next character after each queried path (dense), and their total."""
    H, X, Cn = tab
    lo = np.searchsorted(H, hq, "left"); hi = np.searchsorted(H, hq, "right"); ln = hi - lo
    rows = np.repeat(np.arange(len(hq)), ln); idx = np.repeat(lo - np.cumsum(np.r_[0, ln[:-1]]), ln) + np.arange(ln.sum())
    C = np.zeros((len(hq), V), np.float32); C[rows, X[idx]] = Cn[idx]
    return C


def bits(P, y):
    return float(-np.log2(P[np.arange(len(y)), y]).mean())


def run(arr, tr, te, budget):
    t0 = time.time()
    val = tr[np.linspace(7, len(tr) - 8, 2_000 if SMOKE else 50_000).astype(np.int64)]
    pool = np.setdiff1d(tr, val)
    cp = pool if budget is None or budget >= len(pool) else pool[np.linspace(0, len(pool) - 1, budget).astype(np.int64)]
    ev = te[np.linspace(0, len(te) - 1, 5_000 if SMOKE else 200_000).astype(np.int64)]
    P0 = np.bincount(arr[cp], minlength=V).astype(np.float64) + 1e-9; P0 /= P0.sum()
    Hc, Hv, He = hashes(arr, cp), hashes(arr, val), hashes(arr, ev); xc = arr[cp].astype(np.uint8)
    Cv, Ce = [], []
    for n in range(NMAX):
        tab = table(Hc[n], xc); Cv.append(look(tab, Hv[n])); Ce.append(look(tab, He[n])); del tab
    count_seconds = time.time() - t0
    yv, ye = arr[val], arr[ev]
    best = None; aside_by_order = {}
    for beta in BETAS:
        P = np.tile(P0, (len(val), 1))
        for n in range(NMAX):
            C = Cv[n].astype(np.float64); P = (C + beta * P) / (C.sum(1, keepdims=True) + beta)
            b = bits(P, yv); aside_by_order[n + 1] = min(aside_by_order.get(n + 1, 9e9), b)
            if best is None or b < best[0]: best = (b, n + 1, beta)
    _, order, beta = best
    P = np.tile(P0, (len(ev), 1))
    for n in range(order):
        C = Ce[n].astype(np.float64); P = (C + beta * P) / (C.sum(1, keepdims=True) + beta)
    return dict(positions_counted=int(len(cp)), order=order, beta=beta, aside_bits=round(best[0], 4),
                held_out_bits_per_char=round(bits(P, ye), 4), next_char_right=round(100 * float((P.argmax(1) == ye).mean()), 2),
                aside_bits_by_longest_path={k: round(v, 4) for k, v in aside_by_order.items()},
                seconds_to_count=round(count_seconds, 1), seconds_total=round(time.time() - t0, 1))


if __name__ == "__main__":
    arr = np.load(os.path.join(DATA, "text.npy")).astype(np.int64)
    z = np.load(os.path.join(DATA, "positions.npz")); tr, te = z["train"], z["test"]
    out = {"budgets": []}
    for b in BUDGETS:
        r = run(arr, tr, te, b); out["budgets"].append(r); print(json.dumps({k: v for k, v in r.items() if k != "aside_bits_by_longest_path"}), flush=True)
    s = arr.copy(); np.random.default_rng(99).shuffle(s)                 # control: the same characters, order destroyed
    out["scrambled"] = run(s, tr, te, 240_000 if SMOKE else 3_840_000); print("scrambled", json.dumps(out["scrambled"]), flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "flow83_smoke.json" if SMOKE else "flow83.json"), "w"), indent=1)
