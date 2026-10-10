"""What flowed returns. The mix's errors (flowword2.py) sit where the dictionary says a word again in another form:
"suffocating suf fo ca ting", "icterus ic te rus", "embody em bod y". The letter is in the text a few letters away,
with silences between the syllables. The return reading: drop the silences (the sound runs on across them), take the m
letters on each side of the letter (m = 2..5), find where the same 2m letters recur within 48 letters on either side,
and read the letter in their middle there. A recurrence whose window would hold the letter itself is not used.
    C(x) = (letters found at the deepest m that recurs + alpha * shares) / (number found + alpha)
Mixed with the saved readings (S, TWb, BZ) by log-linear weights chosen on the held-aside positions for each depth of
recurrence (none, 2, 3, 4, 5). Control: scrambled text. SMOKE=1 runs a small sample first.
"""
import json, os, sys, time, itertools
import numpy as np
import textcells as X
import flowtriads as TR
import flowword as FW

SMOKE = os.environ.get("SMOKE") == "1"
HERE = FW.HERE
D, M = 48, 5
ALPHAS = [0.25, 0.5, 1, 2, 4]


def returns(s, pos):
    """counts[m] (n, 26): letters found in the middle of a recurrence of the m letters on each side; deepest m found."""
    Li = np.flatnonzero(s < 26); L = s[Li]; side = X.T[Li]; n = len(L)
    i = np.searchsorted(Li, pos)
    cnt = np.zeros((M + 1, len(pos), 26), np.float32)
    for sign in (1, -1):
        for d in range(1, D + 1):
            j = i + sign * d
            ok = (j - M >= 0) & (j + M < n) & (i - M >= 0) & (i + M < n)
            jj = np.clip(j, M, n - M - 1); ii = np.clip(i, M, n - M - 1)
            ok &= (side[jj - M] == side[ii]) & (side[jj + M] == side[ii]) & (side[ii - M] == side[ii]) & (side[ii + M] == side[ii])
            eq = ok.copy()
            for m in range(1, M + 1):
                eq &= (L[ii - m] == L[jj - m]) & (L[ii + m] == L[jj + m])
                if m >= 2:
                    use = eq & (d > m)                                  # the window found must not hold the letter itself
                    np.add.at(cnt[m], (np.flatnonzero(use), L[jj[use]]), 1.0)
    found = cnt.sum(2) > 0
    depth = np.zeros(len(pos), np.int64)
    for m in range(2, M + 1):
        depth = np.where(found[m], m, depth)
    return cnt, depth


def reading(cnt, depth, P0l, alpha):
    C = np.tile(P0l, (len(depth), 1))
    for m in range(2, M + 1):
        sel = depth == m
        c = cnt[m][sel].astype(np.float64)
        C[sel] = (c + alpha * P0l) / (c.sum(1, keepdims=True) + alpha)
    return C


def mix(R, P0l, names, w):
    lp = sum(wi * np.log(np.maximum(R[n], 1e-300)) for n, wi in zip(names, w) if wi) - (sum(w) - 1) * np.log(P0l)
    lp -= lp.max(1, keepdims=True); p = np.exp(lp)
    return p / p.sum(1, keepdims=True)


def run(scrambled):
    t0 = time.time()
    s, cp, val, jd, P0, P0l = TR.setup(scrambled)
    tag = "scrambled" if scrambled else "webster"
    z = np.load(os.path.join(HERE, f"flowword2_{tag}.npz"))
    names = ["S", "TWb", "BZ", "C"]
    R = {p: {n: z[f"{p}_{n}"].astype(np.float64) for n in names[:3]} for p in ("aside", "judged")}
    pos = {p: z[f"{p}_pos"] for p in R}; y = {p: z[f"{p}_y"] for p in R}
    if SMOKE:
        for p in R:
            m = slice(0, 4000); R[p] = {n: v[m] for n, v in R[p].items()}; pos[p] = pos[p][m]; y[p] = y[p][m]
    dep = {}; cnts = {}
    for p in R:
        cnts[p], dep[p] = returns(s, pos[p])
    alpha = min(((FW.bits(reading(cnts["aside"], dep["aside"], P0l, a), y["aside"]), a) for a in ALPHAS))[1]
    for p in R:
        R[p]["C"] = reading(cnts[p], dep[p], P0l, alpha)
    out = dict(text=tag, alpha=alpha)
    yj = y["judged"]
    out["share by depth of recurrence (judged, %)"] = {int(m): round(100 * float((dep["judged"] == m).mean()), 2) for m in range(0, M + 1) if (dep["judged"] == m).any()}
    out["return reading alone where found, by depth"] = {int(m): round(100 * float((R["judged"]["C"].argmax(1) == yj)[dep["judged"] == m].mean()), 1)
                                                         for m in range(2, M + 1) if (dep["judged"] == m).any()}
    grid = [0, 0.25, 0.5, 0.75, 1]
    combos = [w for w in itertools.product(grid, repeat=4) if sum(w) > 0]
    before = (0.5, 0.5, 0.5, 0)
    gate = {}
    for m in np.unique(dep["aside"]):
        sel = dep["aside"] == m
        Rm = {n: v[sel] for n, v in R["aside"].items()}
        gate[int(m)] = min(((FW.bits(mix(Rm, P0l, names, w), y["aside"][sel]), w) for w in combos))[1] if sel.sum() >= 30 else before
    P = np.zeros_like(R["judged"]["S"])
    for m in np.unique(dep["judged"]):
        sel = dep["judged"] == m
        P[sel] = mix({n: v[sel] for n, v in R["judged"].items()}, P0l, names, gate.get(int(m), before))
    Pb = mix(R["judged"], P0l, names, before)
    out["mix without the return (S, TWb, BZ at 0.5)"] = dict(bits=round(FW.bits(Pb, yj), 4), right=round(100 * float((Pb.argmax(1) == yj).mean()), 2))
    out["mix with the return, by depth found"] = dict(bits=round(FW.bits(P, yj), 4), right=round(100 * float((P.argmax(1) == yj).mean()), 2))
    out["by depth (right %, without -> with)"] = {int(m): [round(100 * float((Pb.argmax(1) == yj)[dep["judged"] == m].mean()), 1),
                                                            round(100 * float((P.argmax(1) == yj)[dep["judged"] == m].mean()), 1)] for m in np.unique(dep["judged"])}
    out["gate (S, TWb, BZ, C)"] = {str(k): v for k, v in gate.items()}
    out["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(out), flush=True)
    np.savez_compressed(os.path.join(HERE, f"flowcopy_{'smoke_' if SMOKE else ''}{tag}.npz"), aside_C=R["aside"]["C"].astype(np.float32),
                        judged_C=R["judged"]["C"].astype(np.float32), aside_depth=dep["aside"], judged_depth=dep["judged"], judged_P=P.astype(np.float32))
    return out


if __name__ == "__main__":
    res = [run(False)] + ([] if SMOKE else [run(True)])
    json.dump(res, open(os.path.join(HERE, "flowcopy_smoke.json" if SMOKE else "flowcopy.json"), "w"), indent=1)
