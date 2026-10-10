"""The control that came out of flowdiff.py, judged held out: the letter read from the n letters before it and the n
after it (n = 3 and 6), the middle of the two flows (product), beta chosen on 50,000 held-aside positions, judged on
200,000 held-out letters whose 6 neighbours on each side lie on the same side of the split. Control: scrambled text.
"""
import json, os, sys, time
import numpy as np
import textcells as X
from flowmodel import flow, N_COUNT
import flowpm3 as F3


def run(scrambled):
    t0 = time.time(); s = flow(X.S)
    if scrambled:
        s = s.copy(); np.random.default_rng(99).shuffle(s)
    tr, te = X.sides(64); Tm = X.T; K = 6
    okr = lambda p: (p + K < len(s)) & np.all([Tm[np.minimum(p + d, len(s) - 1)] == Tm[p] for d in range(1, K + 1)], 0)
    tr = tr[(s[tr] < 26) & okr(tr)]; te = te[(s[te] < 26) & okr(te)]
    cp = tr[np.linspace(0, len(tr) - 1, N_COUNT).astype(np.int64)]
    rest = np.setdiff1d(tr, cp); val = rest[np.linspace(0, len(rest) - 1, 50_000).astype(np.int64)]
    jd = te[np.linspace(0, len(te) - 1, 200_000).astype(np.int64)]
    P0 = np.bincount(s[cp], minlength=27).astype(np.float64); P0 /= P0.sum(); P0l = P0[:26] / P0[:26].sum()
    T = {side: {m: F3.count(s, s, cp, m, side) for m in range(1, K + 1)} for side in ("before", "after")}
    out = dict(text="scrambled" if scrambled else "Webster")
    for n in (3, 6):
        F3.K = n; best = None
        for beta in [2, 4, 8, 16, 32, 64]:
            R = {"Fl": F3.read(T["before"], P0, s, val, "before", beta), "Rl": F3.read(T["after"], P0, s, val, "after", beta)}
            for name, w in (("forward", (1, 0, 0, 0)), ("+/-", (1, 1, 0, 0))):
                R2 = dict(R, Fg=R["Fl"], Rg=R["Rl"]); v = F3.bits(F3.combine(R2, P0l, w), s[val])
                if best is None or (name, v) < (name, best.get(name, (9,))[0]): pass
                best = best or {}
                if name not in best or v < best[name][0]: best[name] = (v, beta)
        for name, (v, beta) in best.items():
            R = {"Fl": F3.read(T["before"], P0, s, jd, "before", beta), "Rl": F3.read(T["after"], P0, s, jd, "after", beta)}
            R2 = dict(R, Fg=R["Fl"], Rg=R["Rl"]); P = F3.combine(R2, P0l, (1, 0, 0, 0) if name == "forward" else (1, 1, 0, 0)); y = s[jd]
            out[f"{name} {n}"] = dict(beta=beta, aside=round(v, 4), held_out_bits_per_letter=round(F3.bits(P, y), 4), right=round(100 * float((P.argmax(1) == y).mean()), 2))
    F3.K = 3; out["seconds"] = round(time.time() - t0, 1); print(json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    res = [run(False), run(True)]
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "flowreach.json"), "w"), indent=1)
