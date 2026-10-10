"""The return reading widened (from flowscale.py's errors: repeats changed at one end, or farther away). A recurrence
is now scored by how many letters match on its left (up to 5) plus on its right (up to 5), one side may be short or
empty (score at least 3), within REACH letters on either side, silences dropped as before. A recurrence whose window
would hold the letter itself is cut back so it does not. The reading takes the letters found at the highest score:
    C(x) = (letters found at the best score + alpha * shares) / (number found + alpha)
then mixes with S, TWb, BZ by weights chosen on the held-aside positions per best score. Saved in flowcopy.py's form
(flowcopy2_<reach>_*.npz, flowcopy2_<reach>.json) so flowscale.py can stand on it (COPY=flowcopy2_<reach>).
SMOKE=1 runs a small sample first.
"""
import json, os, sys, time, itertools
import numpy as np
import textcells as X
import flowtriads as TR
import flowword as FW
import flowcopy as FC

SMOKE = os.environ.get("SMOKE") == "1"
REACH = int(os.environ.get("REACH", "160"))
HERE = FW.HERE
M, SMIN, SMAX = 5, 3, 10


def returns(s, pos):
    Li = np.flatnonzero(s < 26); L = s[Li]; side = X.T[Li]; n = len(L)
    i = np.searchsorted(Li, pos); ii = np.clip(i, M, n - M - 1)
    cnt = np.zeros((SMAX + 1, len(pos), 26), np.float32)
    for sign in (1, -1):
        for d in range(2, REACH + 1):
            j = i + sign * d; ok = (j - M >= 0) & (j + M < n) & (i - M >= 0) & (i + M < n)
            jj = np.clip(j, M, n - M - 1)
            ok &= (side[jj - M] == side[ii]) & (side[jj + M] == side[ii]) & (side[ii - M] == side[ii]) & (side[ii + M] == side[ii])
            lm = np.zeros(len(pos), np.int64); rm = np.zeros(len(pos), np.int64); goL = ok.copy(); goR = ok.copy()
            for m in range(1, M + 1):
                goL &= L[ii - m] == L[jj - m]; lm += goL
                goR &= L[ii + m] == L[jj + m]; rm += goR
            if sign == 1: lm = np.minimum(lm, d - 1)                   # j's left window must not reach the letter
            else: rm = np.minimum(rm, d - 1)                            # j's right window must not reach the letter
            sc = lm + rm; use = ok & (sc >= SMIN)
            np.add.at(cnt, (sc[use], np.flatnonzero(use), L[jj[use]]), 1.0)
    found = cnt.sum(2) > 0; depth = np.zeros(len(pos), np.int64)
    for m in range(SMIN, SMAX + 1):
        depth = np.where(found[m], m, depth)
    return cnt, depth


def reading(cnt, depth, P0l, alpha):
    C = np.tile(P0l, (len(depth), 1))
    for m in range(SMIN, SMAX + 1):
        sel = depth == m; c = cnt[m][sel].astype(np.float64)
        C[sel] = (c + alpha * P0l) / (c.sum(1, keepdims=True) + alpha)
    return C


def run(scrambled):
    t0 = time.time()
    s, cp, val, jd, P0, P0l = TR.setup(scrambled)
    tag = "scrambled" if scrambled else "webster"
    z = np.load(os.path.join(HERE, f"flowword2_{tag}.npz")); names = ["S", "TWb", "BZ", "C"]
    R = {p: {n: z[f"{p}_{n}"].astype(np.float64) for n in names[:3]} for p in ("aside", "judged")}
    pos = {p: z[f"{p}_pos"] for p in R}; y = {p: z[f"{p}_y"] for p in R}
    if SMOKE:
        for p in R:
            m = slice(0, 4000); R[p] = {n: v[m] for n, v in R[p].items()}; pos[p] = pos[p][m]; y[p] = y[p][m]
    cnts, dep = {}, {}
    for p in R: cnts[p], dep[p] = returns(s, pos[p])
    alpha = min(((FW.bits(reading(cnts["aside"], dep["aside"], P0l, a), y["aside"]), a) for a in FC.ALPHAS))[1]
    for p in R: R[p]["C"] = reading(cnts[p], dep[p], P0l, alpha)
    grid = [0, 0.25, 0.5, 0.75, 1]; combos = [w for w in itertools.product(grid, repeat=4) if sum(w) > 0]
    gate = {}
    for m in np.unique(dep["aside"]):
        sel = dep["aside"] == m
        gate[int(m)] = min(((FW.bits(FC.mix({n: v[sel] for n, v in R["aside"].items()}, P0l, names, w), y["aside"][sel]), w) for w in combos))[1] if sel.sum() >= 30 else (0.5, 0.5, 0.5, 0)
    P = np.zeros_like(R["judged"]["S"])
    for m in np.unique(dep["judged"]):
        sel = dep["judged"] == m; P[sel] = FC.mix({n: v[sel] for n, v in R["judged"].items()}, P0l, names, gate.get(int(m), (0.5, 0.5, 0.5, 0)))
    yj = y["judged"]
    out = {"text": tag, "reach": REACH, "alpha": alpha,
           "share with a return found (%)": round(100 * float((dep["judged"] > 0).mean()), 2),
           "return alone right where found (%)": round(100 * float((R["judged"]["C"].argmax(1) == yj)[dep["judged"] > 0].mean()), 1),
           "mix with the return": dict(bits=round(FW.bits(P, yj), 4), right=round(100 * float((P.argmax(1) == yj).mean()), 2)),
           "gate (S, TWb, BZ, C)": {str(k): v for k, v in gate.items()}, "seconds": round(time.time() - t0, 1)}
    print(json.dumps(out), flush=True)
    np.savez_compressed(os.path.join(HERE, f"flowcopy2_{REACH}_{'smoke_' if SMOKE else ''}{tag}.npz"), aside_C=R["aside"]["C"].astype(np.float32), judged_C=R["judged"]["C"].astype(np.float32),
                        aside_depth=dep["aside"], judged_depth=dep["judged"])
    return out


if __name__ == "__main__":
    res = [run(False)] + ([] if SMOKE else [run(True)])
    json.dump(res, open(os.path.join(HERE, f"flowcopy2_{REACH}_smoke.json" if SMOKE else f"flowcopy2_{REACH}.json"), "w"), indent=1)
