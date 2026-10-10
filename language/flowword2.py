"""The next round from flowword.py. The cut at silence lifted the tower (84.8%) but left one-letter words blind (43%):
their window is silence on both sides. Here the window may also cross one silence and stop at the second, so a word
reads its whole neighbour (the scale above: the word as a node). Readings added: TWb2 (cut at the second silence) and
BZb (the Boltzmann reading of triads 1..4 on windows cut at the first silence). All readings are mixed, once for all
and per (deepest seen level, place in word), weights chosen on the held-aside positions. SMOKE=1 runs a small sample.
"""
import json, os, sys, time, itertools
import numpy as np
import flowtriads as TR
import flowboltz as BZ
import flowword as FW

SMOKE = os.environ.get("SMOKE") == "1"
HERE = FW.HERE
NAMES = ["S", "TW", "BZ", "TWb", "TWb2", "BZb"]


def cut(s, pos, K, n):
    """Each side of the window runs to its n-th silence; beyond it, silence."""
    B = np.zeros((len(pos), K), np.int64); A = np.zeros((len(pos), K), np.int64)
    cb = np.zeros(len(pos), np.int64); ca = np.zeros(len(pos), np.int64)
    for j in range(1, K + 1):
        b = s[pos - j]; a = s[pos + j]
        B[:, j - 1] = np.where(cb >= n, 26, b); A[:, j - 1] = np.where(ca >= n, 26, a)
        cb += b == 26; ca += a == 26
    return B, A


def bskey(B, A, S):
    key = np.zeros(len(B), np.int64)
    for j in S:
        key = key * 729 + B[:, j - 1] * 27 + A[:, j - 1]
    return key


def mix(R, P0l, w):
    lp = sum(wi * np.log(np.maximum(R[n], 1e-300)) for n, wi in zip(NAMES, w) if wi) - (sum(w) - 1) * np.log(P0l)
    lp -= lp.max(1, keepdims=True); p = np.exp(lp)
    return p / p.sum(1, keepdims=True)


def run(scrambled):
    t0 = time.time()
    s, cp, val, jd, P0, P0l = TR.setup(scrambled)
    tag = "scrambled" if scrambled else "webster"
    z = np.load(os.path.join(HERE, f"flowlook_{tag}.npz"))
    R = {part: {n: z[f"{part}_{n}"].astype(np.float64) for n in ("S", "TW", "BZ")} for part in ("aside", "judged")}
    pos = {part: z[f"{part}_pos"] for part in R}; y = {part: z[f"{part}_y"] for part in R}
    if SMOKE:
        cp = cp[::20]
        for part in R:
            m = slice(0, 3000); R[part] = {n: v[m] for n, v in R[part].items()}; pos[part] = pos[part][m]; y[part] = y[part][m]
    seen = {}
    for n_cut, name in ((1, "TWb"), (2, "TWb2")):
        Bc, Ac = cut(s, cp, TR.K, n_cut)
        T = {k: TR.table(FW.bends(Bc, Ac, k), s[cp]) for k in range(1, TR.K + 1)}
        Ba, Aa = cut(s, pos["aside"], TR.K, n_cut)
        R["aside"][name], sa, betas = FW.btower(T, np.tile(P0l, (len(pos["aside"]), 1)), Ba, Aa, y["aside"])
        Bj, Aj = cut(s, pos["judged"], TR.K, n_cut)
        R["judged"][name], sj, _ = FW.btower(T, np.tile(P0l, (len(pos["judged"]), 1)), Bj, Aj, betas=betas)
        if n_cut == 1:
            seen = {"aside": sa, "judged": sj}
            TS = {S: TR.table(bskey(Bc, Ac, S), s[cp]) for S in BZ.SUBSETS}
            def counts(B, A):
                C = np.zeros((len(BZ.SUBSETS), len(B), 26), np.float32)
                for i, S in enumerate(BZ.SUBSETS):
                    C[i], _ = TR.look(TS[S], bskey(B, A, S))
                return C
            E = BZ.fit(counts(Ba, Aa), y["aside"], False)
            R["aside"]["BZb"] = BZ.probs(counts(Ba, Aa), E); R["judged"]["BZb"] = BZ.probs(counts(Bj, Aj), E)
            del TS
        del T
    place = lambda p: np.where((s[p - 1] == 26) & (s[p + 1] == 26), 3, np.where(s[p - 1] == 26, 0, np.where(s[p + 1] == 26, 1, 2)))
    strat = {part: seen[part] * 4 + place(pos[part]) for part in R}
    combos = [w for w in itertools.product(FW.W, repeat=len(NAMES)) if sum(w) > 0]
    glob = min(((FW.bits(mix(R["aside"], P0l, w), y["aside"]), w) for w in combos))[1]
    gate = {}
    for st in np.unique(strat["aside"]):
        m = strat["aside"] == st
        if m.sum() >= 30:
            Rm = {n: v[m] for n, v in R["aside"].items()}
            gate[int(st)] = min(((FW.bits(mix(Rm, P0l, w), y["aside"][m]), w) for w in combos))[1]
    Pg = np.zeros_like(R["judged"]["S"])
    for st in np.unique(strat["judged"]):
        m = strat["judged"] == st
        Pg[m] = mix({n: v[m] for n, v in R["judged"].items()}, P0l, gate.get(int(st), glob))
    yj = y["judged"]; res = dict(R["judged"]); res["one mix for all"] = mix(R["judged"], P0l, glob); res["mix by what the reader sees"] = Pg
    out = dict(text=tag, global_weights=dict(zip(NAMES, glob)), boltzmann_e_bounded=[round(float(e), 5) for e in E[:, 0]],
               alone_and_mixed={n: dict(bits=round(FW.bits(P, yj), 4), right=round(100 * float((P.argmax(1) == yj).mean()), 2)) for n, P in res.items()})
    pl = place(pos["judged"]); right = {n: P.argmax(1) == yj for n, P in res.items()}
    out["by place in word (right %)"] = {lab: {n: round(100 * float(right[n][pl == i].mean()), 1) for n in res} for i, lab in enumerate(("word start", "word end", "inside", "one-letter word"))}
    anyr = np.zeros(len(yj), bool)
    for n in NAMES: anyr |= right[n]
    out["any reading right"] = round(100 * float(anyr.mean()), 2)
    out["gate"] = {str(k): dict(zip(NAMES, v)) for k, v in gate.items()}
    out["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(out), flush=True)
    np.savez_compressed(os.path.join(HERE, f"flowword2_{'smoke_' if SMOKE else ''}{tag}.npz"), **{f"{part}_{n}": R[part][n].astype(np.float32) for part in R for n in NAMES},
                        **{f"{part}_pos": pos[part] for part in R}, **{f"{part}_y": y[part] for part in R}, **{f"{part}_seen": seen[part] for part in R})
    return out


if __name__ == "__main__":
    res = [run(False)] + ([] if SMOKE else [run(True)])
    json.dump(res, open(os.path.join(HERE, "flowword2_smoke.json" if SMOKE else "flowword2.json"), "w"), indent=1)
