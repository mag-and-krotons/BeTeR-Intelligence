"""What each reading holds that another lacks. All readings on the same positions (the held-out letters of
flowtriads.py, and the 50,000 held-aside ones), saved, then compared:
  F12  the flow out of the path before (up to 12 nodes)          (one side, long reach)
  R6   the flow back from the path after (up to 6 nodes)          (other side)
  S    their middle, F12 * R6 / shares                            (still)
  TW   the triad tower, ends of triads 1..6 counted together      (still and turn, nested: outer triads let go first)
  BZ   the Boltzmann reading over triads 1..4                     (any triad may mismatch, at a cost)
Compared: each alone; where one is right and another wrong; the deepest triad level whose window was seen; where the
letter sits in its word; vowel or consonant. SMOKE=1 runs a small sample first.
"""
import json, os, sys, time
import numpy as np
import flowpm3 as F3
import flowtriads as TR
import flowboltz as BZ
import flowpair as FP

SMOKE = os.environ.get("SMOKE") == "1"
HERE = os.path.dirname(os.path.abspath(__file__))


def readings(s, cp, val, jd, P0, P0l):
    TF = {m: FP.table(FP.pkey(s, cp - 1, m, -1), s[cp]) for m in range(1, 13)}
    F3.K = 6; TA = {m: F3.count(s, s, cp, m, "after") for m in range(1, 7)}
    T = {k: TR.table(TR.ends(s, cp, k), s[cp]) for k in range(1, TR.K + 1)}
    TS = {S: TR.table(BZ.skey(s, cp, S), s[cp]) for S in BZ.SUBSETS}
    yv = s[val]
    betas, _ = TR.choose(T, np.tile(P0l, (len(val), 1)), s, val, yv)
    E = BZ.fit(BZ.subset_counts(TS, s, val), yv, False)
    out = {}
    for name, pos in (("aside", val), ("judged", jd)):
        F = FP.forward(TF, P0l, s, pos, 12, 64)
        R = F3.read(TA, P0, s, pos, "after", 32)
        S = np.log(F) + np.log(R) - np.log(P0l); S = FP.norm(S)
        TWl = TR.tower(T, np.tile(P0l, (len(pos), 1)), s, pos, betas)
        seen = np.zeros(len(pos), np.int64)                       # deepest triad level whose window was seen
        for k in range(1, TR.K + 1):
            _, t = TR.look(T[k], TR.ends(s, pos, k)); seen = np.where(t[:, 0] > 0, k, seen)
        B = BZ.probs(BZ.subset_counts(TS, s, pos), E)
        out[name] = dict(F12=F.astype(np.float32), R6=R.astype(np.float32), S=S.astype(np.float32), TW=TWl[-1].astype(np.float32),
                         BZ=B.astype(np.float32), seen=seen, y=s[pos], pos=pos)
    F3.K = 3
    return out, betas, E


def bits(P, y):
    return float(-np.log2(np.maximum(P[np.arange(len(y)), y], 1e-300)).mean())


def analyse(d, s):
    y = d["y"]; names = ["F12", "R6", "S", "TW", "BZ"]
    right = {n: d[n].argmax(1) == y for n in names}
    rep = {"alone": {n: dict(bits=round(bits(d[n], y), 4), right=round(100 * float(right[n].mean()), 2)) for n in names}}
    rep["only_A_right_vs_B (percent of letters)"] = {f"{a} not {b}": round(100 * float((right[a] & ~right[b]).mean()), 2)
                                                     for a in names for b in names if a != b}
    anyr = np.zeros(len(y), bool)
    for n in names: anyr |= right[n]
    rep["any reading right"] = round(100 * float(anyr.mean()), 2)
    strata = {}
    for k in range(TR.K + 1):
        m = d["seen"] == k
        if m.sum() < 50: continue
        strata[f"deepest seen triad {k}"] = dict(share=round(100 * float(m.mean()), 2),
                                                 **{n: dict(bits=round(bits(d[n][m], y[m]), 3), right=round(100 * float(right[n][m].mean()), 1)) for n in names})
    rep["by deepest seen triad"] = strata
    p = d["pos"]; ws = s[p - 1] == 26; we = s[p + 1] == 26
    for lab, m in (("word start", ws & ~we), ("word end", we & ~ws), ("inside a word", ~ws & ~we), ("one-letter word", ws & we)):
        rep.setdefault("by place in word", {})[lab] = dict(share=round(100 * float(m.mean()), 2),
                                                          **{n: dict(bits=round(bits(d[n][m], y[m]), 3), right=round(100 * float(right[n][m].mean()), 1)) for n in names})
    vow = np.isin(y, [0, 4, 8, 14, 20])
    for lab, m in (("vowels a e i o u", vow), ("consonants", ~vow)):
        rep.setdefault("by letter kind", {})[lab] = dict(share=round(100 * float(m.mean()), 2),
                                                        **{n: dict(bits=round(bits(d[n][m], y[m]), 3), right=round(100 * float(right[n][m].mean()), 1)) for n in names})
    rep["by letter (right %)"] = {chr(97 + x): {n: round(100 * float(right[n][y == x].mean()), 1) for n in ("S", "TW", "BZ")} for x in range(26) if (y == x).sum() > 20}
    return rep


def run(scrambled):
    t0 = time.time()
    s, cp, val, jd, P0, P0l = TR.setup(scrambled)
    if SMOKE:
        cp, val, jd = cp[::20], val[:2000], jd[:5000]
    d, betas, E = readings(s, cp, val, jd, P0, P0l)
    tag = ("smoke_" if SMOKE else "") + ("scrambled" if scrambled else "webster")
    np.savez_compressed(os.path.join(HERE, f"flowlook_{tag}.npz"), **{f"{part}_{k}": v for part in d for k, v in d[part].items()})
    rep = dict(text=tag, tower_betas=betas, boltzmann_e=[round(float(e), 5) for e in E[:, 0]], judged=analyse(d["judged"], s),
               seconds=round(time.time() - t0, 1))
    return rep


if __name__ == "__main__":
    res = [run(False)] + ([] if SMOKE else [run(True)])
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(HERE, "flowlook_smoke.json" if SMOKE else "flowlook.json"), "w"), indent=1)
