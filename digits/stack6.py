"""The six-image stack: handwritten digits read with no training.

For a digit x, three of its images a, b, c are read forward, (a - b) - c, and the same three reversed, (c - b) - a.
  sum of the two readings        = -2b        the middle: what stays (the still pattern)
  difference of the two readings =  2(a - c)  the turn between the two ends
Each digit keeps its still pattern (the mean of its middles) and the main directions of its turns. A new image is
read as the digit whose still pattern, allowed to turn in that digit's own directions, leaves the least behind.

There is no gradient training: the memory is written in one pass (a mean and one singular value decomposition per
digit). The one choice, how many turning directions each digit keeps, is made inside the training images (80/20).

Data: the 1797 UCI optical digits (8 x 8, values 0..16; data/optdigits.tes, CC BY 4.0), stratified 70/30 split with
random_state 42, the split used for the Grassmannian Neural Engine it is compared with. Control: training labels
scrambled. Run: python3 stack6.py   (numpy only)
"""
import json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    a = np.loadtxt(os.path.join(HERE, "..", "data", "optdigits.tes"), delimiter=",", dtype=np.int64)
    return a[:, :64] / 16.0, a[:, 64]


def split(X, y, test_size, seed):
    """Stratified split: per class, a fixed permutation (RandomState(seed)), the first share held out."""
    rng = np.random.RandomState(seed); tr, te = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c); idx = idx[rng.permutation(len(idx))]
        k = int(round(len(idx) * test_size)); te += list(idx[:k]); tr += list(idx[k:])
    tr, te = np.array(sorted(tr)), np.array(sorted(te))
    return X[tr], X[te], y[tr], y[te]


def write(Xs, ys, k, rng):
    mem = {}
    for c in np.unique(ys):
        I = Xs[ys == c][rng.permutation((ys == c).sum())]
        t = len(I) // 3; a, b, cc = I[:t], I[t:2 * t], I[2 * t:3 * t]
        fwd = (a - b) - cc; rev = (cc - b) - a
        still = -((fwd + rev) / 2).mean(0)                   # the middles
        turn = (fwd - rev) / 2                               # a - c: how the writings turn
        _, _, Vt = np.linalg.svd(turn - turn.mean(0), full_matrices=False)
        mem[c] = (still, Vt[:k].T)
    return mem


def read(mem, Xs):
    cs = sorted(mem); R = []
    for c in cs:
        m, U = mem[c]; D = Xs - m
        R.append(((D - (D @ U) @ U.T) ** 2).sum(1))          # what is left after the digit's own turns
    return np.array(cs)[np.argmin(np.array(R), 0)]


def choose_k(Xs, ys, seed, ks=range(0, 31)):
    rng = np.random.default_rng(seed)
    a, b, ya, yb = split(Xs, ys, 0.2, seed)
    sc = {k: float((read(write(a, ya, k, rng), b) == yb).mean()) for k in ks}
    return max(sc, key=sc.get), sc


if __name__ == "__main__":
    X, y = load()
    Xtr, Xte, ytr, yte = split(X, y, 0.3, 42)
    out = []
    for seed in (0, 1, 2):
        t0 = time.time()
        k, _ = choose_k(Xtr, ytr, seed)
        rng = np.random.default_rng(seed)
        mem = write(Xtr, ytr, k, rng)
        acc = (read(mem, Xte) == yte).mean() * 100
        t1 = time.time() - t0
        still_only = (read(write(Xtr, ytr, 0, rng), Xte) == yte).mean() * 100
        ctrl = (read(write(Xtr, rng.permutation(ytr), k, rng), Xte) == yte).mean() * 100     # control: labels scrambled
        memi = {}                                            # written two digits at a time, never shown again
        for pair in ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9)):
            m = np.isin(ytr, pair); memi.update(write(Xtr[m], ytr[m], k, rng))
        inc = (read(memi, Xte) == yte).mean() * 100
        r = dict(seed=seed, k_turns=k, test_all_ten=round(acc, 2), still_only=round(still_only, 2), scrambled_control=round(ctrl, 2),
                 two_at_a_time=round(inc, 2), seconds_incl_choosing_k=round(t1, 2))
        print(json.dumps(r)); out.append(r)
    t0 = time.time(); mem = write(Xtr, ytr, out[0]["k_turns"], np.random.default_rng(0)); tw = time.time() - t0
    t0 = time.time(); _ = read(mem, Xte); tr = time.time() - t0
    timing = dict(train_images=int(len(ytr)), test_images=int(len(yte)), write_seconds=round(tw, 4), read_test_seconds=round(tr, 4))
    print(json.dumps(timing))
    acc = [r["test_all_ten"] for r in out]
    summary = dict(mean=round(float(np.mean(acc)), 2), sd=round(float(np.std(acc, ddof=1)), 2))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(dict(seeds=out, timing=timing, all_ten=summary), open(os.path.join(HERE, "results", "stack6.json"), "w"), indent=1)
