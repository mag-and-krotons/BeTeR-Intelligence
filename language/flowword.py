"""From flowlook.py: what each reading lacks, put back.
1. The triads stop at silence. Inside a word the turn reads 88% of letters; at a word's start or end the tower drops
   to the still's level, because its triads reach across the silence into the next word and split the counts. Here
   each side of a triad window runs to the first silence and no further: that silence is kept, what lies beyond it
   reads as silence too (TWb).
2. The observer weighs. Where the window was seen only shallowly the long one-sided flows (S) do better; where it was
   seen deep the turn does. Each letter is read by
       log P = a log S + b log TW + c log TWb + d log BZ - (a + b + c + d - 1) log shares
   with (a, b, c, d) chosen on the held-aside positions separately for each (deepest seen level of TWb, place in the
   word): what the reader can see before reading the letter.
Readings S, TW, BZ are those saved by flowlook.py on the same positions. SMOKE=1 runs a small sample first.
"""
import json, os, sys, time, itertools
import numpy as np
import flowpm3 as F3
import flowtriads as TR

SMOKE = os.environ.get("SMOKE") == "1"
HERE = os.path.dirname(os.path.abspath(__file__))
W = [0, 0.5, 1]


def bounded(s, pos, K):
    """Each side of the window, cut at the first silence (the silence kept, beyond it silence)."""
    B = np.zeros((len(pos), K), np.int64); A = np.zeros((len(pos), K), np.int64)
    sb = np.zeros(len(pos), bool); sa = np.zeros(len(pos), bool)
    for j in range(1, K + 1):
        b = s[pos - j]; a = s[pos + j]
        B[:, j - 1] = np.where(sb, 26, b); A[:, j - 1] = np.where(sa, 26, a)
        sb |= b == 26; sa |= a == 26
    return B, A


def bends(B, A, k):
    key = np.zeros(len(B), np.int64)
    for j in range(k):
        key = key * 729 + B[:, j] * 27 + A[:, j]
    return key


def btower(T, base, B, A, y=None, betas=None):
    P = base.copy(); chosen = []; seen = np.zeros(len(B), np.int64)
    for k in range(1, TR.K + 1):
        C, t = TR.look(T[k], bends(B, A, k)); seen = np.where(t[:, 0] > 0, k, seen)
        if betas is None:
            best = min(((F3.bits((C + b * P) / (t + b), y), b) for b in TR.BETAS)); chosen.append(best[1]); b = best[1]
        else:
            b = betas[k - 1]
        P = (C + b * P) / (t + b)
    return P, seen, chosen


def bits(P, y):
    return float(-np.log2(np.maximum(P[np.arange(len(y)), y], 1e-300)).mean())


def mix(R, P0l, w):
    lp = sum(wi * np.log(np.maximum(R[n], 1e-300)) for n, wi in zip(("S", "TW", "TWb", "BZ"), w) if wi) - (sum(w) - 1) * np.log(P0l)
    lp -= lp.max(1, keepdims=True); p = np.exp(lp)
    return p / p.sum(1, keepdims=True)


def run(scrambled):
    t0 = time.time()
    s, cp, val, jd, P0, P0l = TR.setup(scrambled)
    tag = "scrambled" if scrambled else "webster"
    z = np.load(os.path.join(HERE, f"flowlook_{tag}.npz"))
    if SMOKE:
        cp = cp[::20]
    R = {part: {n: z[f"{part}_{n}"].astype(np.float64) for n in ("S", "TW", "BZ")} for part in ("aside", "judged")}
    pos = {part: z[f"{part}_pos"] for part in R}; y = {part: z[f"{part}_y"] for part in R}
    if SMOKE:
        for part in R:
            m = slice(0, 3000); R[part] = {n: v[m] for n, v in R[part].items()}; pos[part] = pos[part][m]; y[part] = y[part][m]
    Bc, Ac = bounded(s, cp, TR.K)
    T = {k: TR.table(bends(Bc, Ac, k), s[cp]) for k in range(1, TR.K + 1)}
    out = dict(text=tag, distinct_bounded_windows={k: int(len(T[k][2])) for k in T})
    Ba, Aa = bounded(s, pos["aside"], TR.K)
    R["aside"]["TWb"], seen_a, betas = btower(T, np.tile(P0l, (len(pos["aside"]), 1)), Ba, Aa, y["aside"])
    Bj, Aj = bounded(s, pos["judged"], TR.K)
    R["judged"]["TWb"], seen_j, _ = btower(T, np.tile(P0l, (len(pos["judged"]), 1)), Bj, Aj, betas=betas)
    out["bounded tower betas"] = betas

    def place(p):
        ws, we = s[p - 1] == 26, s[p + 1] == 26
        return np.where(ws & we, 3, np.where(ws, 0, np.where(we, 1, 2)))
    strat_a = seen_a * 4 + place(pos["aside"]); strat_j = seen_j * 4 + place(pos["judged"])
    combos = [w for w in itertools.product(W, repeat=4) if sum(w) > 0]
    glob = min(((bits(mix(R["aside"], P0l, w), y["aside"]), w) for w in combos))
    gate = {}
    for st in np.unique(strat_a):
        m = strat_a == st
        Rm = {n: v[m] for n, v in R["aside"].items()}
        gate[int(st)] = min(((bits(mix(Rm, P0l, w), y["aside"][m]), w) for w in combos))[1] if m.sum() >= 30 else glob[1]
    Pj = np.zeros_like(R["judged"]["S"])
    for st in np.unique(strat_j):
        m = strat_j == st
        Pj[m] = mix({n: v[m] for n, v in R["judged"].items()}, P0l, gate.get(int(st), glob[1]))
    yj = y["judged"]
    names = ["S", "TW", "BZ", "TWb"]
    res = {n: R["judged"][n] for n in names}
    res["one mix for all"] = mix(R["judged"], P0l, glob[1]); res["mix chosen by what the reader sees"] = Pj
    out["global mix weights (S, TW, TWb, BZ)"] = glob[1]
    out["alone and mixed"] = {n: dict(bits=round(bits(P, yj), 4), right=round(100 * float((P.argmax(1) == yj).mean()), 2)) for n, P in res.items()}
    pl = place(pos["judged"])
    out["by place in word (right %)"] = {lab: {n: round(100 * float((res[n].argmax(1) == yj)[pl == i].mean()), 1) for n in res}
                                         for i, lab in enumerate(("word start", "word end", "inside", "one-letter word")) if (pl == i).sum()}
    out["gate (stratum = seen*4 + place)"] = {str(k): v for k, v in gate.items()}
    out["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    res = [run(False)] + ([] if SMOKE else [run(True)])
    json.dump(res, open(os.path.join(HERE, "flowword_smoke.json" if SMOKE else "flowword.json"), "w"), indent=1)
