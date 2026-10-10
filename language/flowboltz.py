"""The Boltzmann reading of the triads, and the grid at the triads' ends.
1. Boltzmann. In the digit stack an image is read by count * exp(-r / T), r what is left of it after the class. Here
   every stored window (the ends of the triads 1..k around a letter, k = 4) speaks for its middle letter, weighted by
   which of its triads match the window being read: each triad j that does not match costs exp(-w_j / T) = e_j, each
   that matches keeps 1 - e_j. Summed over all stored windows this is
       A(x) = sum over subsets S of the triads: count(windows matching on S, middle x) * prod_{j in S} (1 - e_j) * prod_{j not in S} e_j
   and P(x) = A(x) / sum_x A(x). The tower (flowtriads.py) lets only the outermost triads go; here any triad may.
   Two forms: e_j shared by all letters (4 numbers); e_jx per letter, each class keeping which of its triads turn
   freely (its turns), as each digit kept its own (104 numbers). Fitted on 50,000 held-aside positions.
2. The grid at the turn. In the tower, before letting triad k go, its two ends are read through Panini's grid (21
   cells hold the 26 letters, silence its own): the window's outer triad by its sounds' cells. Controls: the letters
   dealt to the same cells at random (three deals).
Counted over the same 3.84 million training positions; judged once on the same 200,000 held-out letters as
flowtriads.py. Control: scrambled text.
"""
import json, os, sys, time, itertools
import numpy as np
import flowgrid as G
import flowpm3 as F3
import flowtriads as TR

KB = 4
SUBSETS = [S for r in range(KB + 1) for S in itertools.combinations(range(1, KB + 1), r)]


def skey(s, pos, S):
    key = np.zeros(len(pos), np.int64)
    for j in S:
        key = key * 729 + s[pos - j] * 27 + s[pos + j]
    return key


def subset_counts(T, s, pos):
    out = np.zeros((len(SUBSETS), len(pos), 26), np.float32)
    for i, S in enumerate(SUBSETS):
        C, _ = TR.look(T[S], skey(s, pos, S)); out[i] = C
    return out


def weights(E):
    """E: (KB, 26) mismatch weights e_jx -> (n_subsets, 26) subset weights."""
    W = np.ones((len(SUBSETS), 26))
    for i, S in enumerate(SUBSETS):
        for j in range(1, KB + 1):
            W[i] *= (1 - E[j - 1]) if j in S else E[j - 1]
    return W


def probs(C, E):
    A = np.einsum("snx,sx->nx", C, weights(E)) + 1e-300
    return A / A.sum(1, keepdims=True)


def fit(C, y, per_class):
    n = len(y); idx = np.arange(n)
    def unpack(th):
        th = th.reshape(KB, 26) if per_class else np.repeat(th[:, None], 26, 1)
        return 1 / (1 + np.exp(-th))
    def f(th):
        E = unpack(th); W = weights(E)
        A = np.einsum("snx,sx->nx", C, W) + 1e-300; Z = A.sum(1)
        ll = np.log(A[idx, y]).sum() - np.log(Z).sum()
        g = np.zeros((KB, 26))
        for j in range(1, KB + 1):
            sgn = np.array([(-1 / (1 - E[j - 1])) if j in S else (1 / E[j - 1]) for S in SUBSETS])   # (s, 26)
            D = np.einsum("snx,sx->nx", C, W * sgn)
            gx = np.zeros(26); np.add.at(gx, y, D[idx, y] / A[idx, y]); gx -= (D / Z[:, None]).sum(0)
            g[j - 1] = gx * E[j - 1] * (1 - E[j - 1])
        g = g if per_class else g.sum(1, keepdims=True)
        return -ll / n, -g.ravel() / n
    th0 = np.full(KB * 26 if per_class else KB, -3.0)
    th, m, v = th0.copy(), np.zeros_like(th0), np.zeros_like(th0)       # plain Adam on the exact gradient
    for i in range(1, 401):
        _, g = f(th); m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        th -= 0.05 * (m / (1 - 0.9 ** i)) / (np.sqrt(v / (1 - 0.999 ** i)) + 1e-8)
    return unpack(th)


def compact(code):
    u = np.unique(code[:26]); m = {c: i for i, c in enumerate(u)}
    return np.array([m[c] for c in code[:26]] + [len(u)], np.int64)


def grid_tower(T, TG, base, s, g, pos, y=None, betas=None):
    """Level k: letters at triads 1..k-1, grid cells at triad k, then letters at triad k."""
    P = base.copy(); chosen = []; aside = []
    for k in range(1, TR.K + 1):
        kg = TR.ends(s, pos, k - 1) * 729 + g[pos - k] * 27 + g[pos + k]
        Cg, tg = TR.look(TG[k], kg); C, t = TR.look(T[k], TR.ends(s, pos, k))
        if betas is None:
            best = None
            for bg, b in itertools.product(TR.BETAS, TR.BETAS):
                Q = (C + b * (Cg + bg * P) / (tg + bg)) / (t + b); v = F3.bits(Q, y)
                if best is None or v < best[0]: best = (v, (bg, b), Q)
            chosen.append(best[1]); aside.append(round(best[0], 4)); P = best[2]
        else:
            bg, b = betas[k - 1]; P = (C + b * (Cg + bg * P) / (tg + bg)) / (t + b)
    return (chosen, aside) if betas is None else P


def run(scrambled):
    t0 = time.time()
    s, cp, val, jd, P0, P0l = TR.setup(scrambled)
    yv, yj = s[val], s[jd]
    out = dict(text="scrambled" if scrambled else "Webster")
    # the tower, for reference on the same positions
    T = {k: TR.table(TR.ends(s, cp, k), s[cp]) for k in range(1, TR.K + 1)}
    bv, bj = np.tile(P0l, (len(val), 1)), np.tile(P0l, (len(jd), 1))
    betas, _ = TR.choose(T, bv, s, val, yv)
    Ptw = TR.tower(T, bj, s, jd, betas)
    out["tower (1..4)"] = dict(bits=round(F3.bits(Ptw[4], yj), 4), right=round(100 * float((Ptw[4].argmax(1) == yj).mean()), 2))
    out["tower (1..6)"] = dict(bits=round(F3.bits(Ptw[6], yj), 4), right=round(100 * float((Ptw[6].argmax(1) == yj).mean()), 2))
    print(out, flush=True)
    # 1. Boltzmann over the triads
    TS = {S: TR.table(skey(s, cp, S), s[cp]) for S in SUBSETS}
    Cv, Cj = subset_counts(TS, s, val), subset_counts(TS, s, jd)
    del TS
    for name, pc in (("Boltzmann, one weight per triad", False), ("Boltzmann, each letter its own turns", True)):
        E = fit(Cv, yv, pc); P = probs(Cj, E)
        out[name] = dict(e_mean_by_triad=[round(float(e), 4) for e in E.mean(1)], bits=round(F3.bits(P, yj), 4), right=round(100 * float((P.argmax(1) == yj).mean()), 2))
        if pc:
            out[name]["e_by_letter"] = {chr(97 + x): [round(float(E[j, x]), 3) for j in range(KB)] for x in range(26)}
            Pb = P
        print(name, {k: v for k, v in out[name].items() if k != "e_by_letter"}, flush=True)
    del Cv, Cj
    # the Boltzmann reading and the tower together (log-linear, weight chosen aside)
    # (aside versions recomputed cheaply from the stored tables would need them again: use a held-aside split of val instead)
    # 2. the grid at the turn
    codes = {"Panini": compact(G.mapper(G.PLACE, 7) * 8 + G.mapper(G.HOW, 7))}
    for d in (1, 2, 3):
        r = np.random.default_rng(d); c = codes["Panini"].copy(); c[:26] = r.permutation(c[:26]); codes[f"random deal {d}"] = c
    for name, code in codes.items():
        g = code[s]
        TG = {k: TR.table(TR.ends(s, cp, k - 1) * 729 + g[cp - k] * 27 + g[cp + k], s[cp]) for k in range(1, TR.K + 1)}
        chosen, aside = grid_tower(T, TG, bv, s, g, val, yv)
        P = grid_tower(T, TG, bj, s, g, jd, betas=chosen)
        out[f"tower with the grid at the turn: {name}"] = dict(betas=chosen, aside=aside[-1], bits=round(F3.bits(P, yj), 4), right=round(100 * float((P.argmax(1) == yj).mean()), 2))
        print(name, out[f"tower with the grid at the turn: {name}"], flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    return out


if __name__ == "__main__":
    res = [run(False), run(True)]
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "flowboltz.json"), "w"), indent=1)
