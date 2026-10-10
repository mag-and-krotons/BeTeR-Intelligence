"""The diff between the two representations, then the amalgam, read +3/-3.
Place x how as one code per node puts the 26 letters in 21 cells (only c/k/q, h/x, s/z, v/w share one), so a path of
grid cells is nearly the path of letters: that reading adds nothing. The grid's two coordinates are read here as two
flows of their own: the flow of places (where the breath is shaped, 8 values) and the flow of hows (how it is shaped,
8 values). Each is read forward (3 before) and in reverse (3 after), as the letters are; paths of 3 and of 6 nodes.
First the diff: on the held-aside positions, where the letter middle and each grid middle disagree, which is right.
Then the amalgam: log P = sum_i w_i log R_i - (sum_i w_i - 1) log P0, letters fixed at (1, 1), the place and how
weights chosen on 50,000 held-aside positions; judged once on the same 200,000 held-out letters. Control: scrambled text.
"""
import json, os, sys, time, itertools
import numpy as np
import textcells as X
from flowmodel import flow, N_COUNT
import flowgrid as G
import flowpm3 as F3

WS = [0, 0.25, 0.5, 1]
KG = int(os.environ.get("KG", "3"))         # path length of the place and how flows
CTRL = os.environ.get("CTRL", "")             # controls: "letters" puts the letters in the place and how slots; "shuffle<seed>" deals the
                                             # letters to the place and how codes at random (same code sizes)


def read(T, P0, h, pos, side, beta, k):
    F3.K = k; r = F3.read(T, P0, h, pos, side, beta); F3.K = 3
    return r


def readings(T, P0, H, pos, beta):
    return {("F" if side == "before" else "R") + rep: read(T[(rep, side)], P0, H[rep], pos, side, beta, 3 if rep == "l" else KG)
            for rep in H for side in ("before", "after")}


def product(R, P0l, w):
    lp = sum(w[k] * np.log(R[k]) for k in w) - (sum(w.values()) - 1) * np.log(P0l)
    lp -= lp.max(1, keepdims=True); p = np.exp(lp)
    return p / p.sum(1, keepdims=True)


def run(scrambled, beta=8):
    t0 = time.time()
    s = flow(X.S)
    if scrambled:
        s = s.copy(); np.random.default_rng(99).shuffle(s)
    mp, mh = G.mapper(G.PLACE, 7), G.mapper(G.HOW, 7)
    if CTRL.startswith("shuffle"):
        r = np.random.default_rng(int(CTRL[7:])); mp = np.r_[r.permutation(mp[:26]), 7]; mh = np.r_[r.permutation(mh[:26]), 7]
    H = {"l": s, "p": s if CTRL == "letters" else mp[s], "h": s if CTRL == "letters" else mh[s]}
    tr, te = X.sides(64); Tm = X.T; K = max(3, KG)
    okr = lambda p: (p + K < len(s)) & np.all([Tm[np.minimum(p + d, len(s) - 1)] == Tm[p] for d in range(1, K + 1)], 0)
    tr = tr[(s[tr] < 26) & okr(tr)]; te = te[(s[te] < 26) & okr(te)]
    cp = tr[np.linspace(0, len(tr) - 1, N_COUNT).astype(np.int64)]
    rest = np.setdiff1d(tr, cp); val = rest[np.linspace(0, len(rest) - 1, 50_000).astype(np.int64)]
    jd = te[np.linspace(0, len(te) - 1, 200_000).astype(np.int64)]
    P0 = np.bincount(s[cp], minlength=27).astype(np.float64); P0 /= P0.sum(); P0l = P0[:26] / P0[:26].sum()
    T = {(rep, side): {m: F3.count(H[rep], s, cp, m, side) for m in range(1, (3 if rep == "l" else KG) + 1)} for rep in H for side in ("before", "after")}
    out = dict(text="scrambled" if scrambled else "Webster", grid_path=KG, control=CTRL or None)
    Rv = readings(T, P0, H, val, beta); yv = s[val]
    mid = {rep: product(Rv, P0l, {f"F{rep}": 1, f"R{rep}": 1}) for rep in H}
    top = {rep: mid[rep].argmax(1) for rep in H}
    for rep in ("p", "h"):
        dis = top["l"] != top[rep]
        out[f"diff letters vs {rep}"] = dict(disagree=round(100 * float(dis.mean()), 2),
                                            letters_right_when_disagree=round(100 * float((top["l"][dis] == yv[dis]).mean()), 2),
                                            grid_right_when_disagree=round(100 * float((top[rep][dis] == yv[dis]).mean()), 2),
                                            middle_bits=round(F3.bits(mid[rep], yv), 4), middle_right=round(100 * float((top[rep] == yv).mean()), 2))
    out["letters middle (aside)"] = dict(bits=round(F3.bits(mid["l"], yv), 4), right=round(100 * float((top["l"] == yv).mean()), 2))
    best = None
    for wp, wh in itertools.product(itertools.product(WS, WS), itertools.product(WS, WS)):
        w = {"Fl": 1, "Rl": 1, "Fp": wp[0], "Rp": wp[1], "Fh": wh[0], "Rh": wh[1]}
        v = F3.bits(product(Rv, P0l, w), yv)
        if best is None or v < best[0]: best = (v, w)
    Rj = readings(T, P0, H, jd, beta); yj = s[jd]
    for name, w in (("+3/-3 letters", {"Fl": 1, "Rl": 1}), ("+3/-3 letters, place and how", best[1])):
        P = product(Rj, P0l, w)
        out[name] = dict(weights=w, held_out_bits_per_letter=round(F3.bits(P, yj), 4), right=round(100 * float((P.argmax(1) == yj).mean()), 2))
    out["aside_best"] = round(best[0], 4); out["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    res = [run(False)] + ([] if CTRL else [run(True)])
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"flowdiff_k{KG}{CTRL}.json"), "w"), indent=1)
