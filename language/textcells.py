"""The text as a row of 30 cells holding numbers, as a digit is a grid of cells holding fills.
His layout: cells 0, 1, 2, 3 are sacred; the vowels sit at the primes, a 5, e 7, i 11, o 13, u 17 (a lands in cell 5
by itself); the 21 consonants fill the other cells 4..29 in order (b 4, c 6, d 8, f 9, g 10, h 12, j 14, k 15, l 16,
m 18, n 19, p 20, q 21, r 22, s 23, t 24, v 25, w 26, x 27, y 28, z 29). Every cell holds the same kind of thing, a
number, read as its share of the whole 30 (a 5/30, z 29/30), so silence (0 to 1) is on the same scale. Silence is not absence: a run of non-letters is one silent position whose number is a zero, a one (what is left
when the rest is taken from the whole), or a ratio of both; tried: 0, 1, 1/2, and its length (space 1/3, comma-like
2/3, full stop or new line 1).
A context is the last n positions (n = 8, 16, 32, 64; 64 is the digit's 8 x 8), each holding its number. The task: the
next letter. His stack as for the digits: each next letter keeps the middles and turns of its contexts (three read
forward, the same three reversed) and its count, and a context is read by count * exp(-r / T).
Everything is chosen on 50,000 held-aside training positions; the best is judged once on 200,000 held-out letters.
Controls: the same layout with the letters numbered in plain alphabetical order (a 4 ... z 29), and the text scrambled.
Run: python3 textcells.py
"""
import json, os, time
import numpy as np
import textstack as TS

HERE = os.path.dirname(os.path.abspath(__file__))
CHARS = json.load(open(os.path.join(TS.DATA, "vocab.json")))["chars"]
LONG, MIDDLE = set(".\n!?"), set(",;:()[]\"`'-")
VOWEL_CELL = {"a": 5, "e": 7, "i": 11, "o": 13, "u": 17}
CONS = [c for c in "abcdefghijklmnopqrstuvwxyz" if c not in VOWEL_CELL]
FREE = [n for n in range(4, 30) if n not in VOWEL_CELL.values()]
HIS = {**VOWEL_CELL, **dict(zip(CONS, FREE))}                           # his cells
ALPHA = {c: 4 + i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}  # control: plain order
SILENCE = {"zero": (0, 0, 0), "one": (1, 1, 1), "half": (0.5, 0.5, 0.5), "length": (1 / 3, 2 / 3, 1)}


def stream():
    """The text as letters (0..25) and silences (26 short, 27 middle, 28 long), with each position's side."""
    f = os.path.join(TS.DATA, "cells_stream.npz")
    if os.path.exists(f):
        z = np.load(f); return z["S"], z["T"]
    arr = np.load(os.path.join(TS.DATA, "text.npy")); test = np.load(os.path.join(TS.DATA, "test_mask.npy"))
    code = np.full(len(CHARS), -1, np.int16); lev = np.zeros(len(CHARS), np.int8)
    for i, ch in enumerate(CHARS):
        c = {"é": "e", "ö": "o", "æ": "a"}.get(ch.lower(), ch.lower())
        if len(c) == 1 and "a" <= c <= "z":
            code[i] = ord(c) - 97
        else:
            lev[i] = 3 if ch in LONG else 2 if ch in MIDDLE else 1
    c = code[arr]; isl = c >= 0
    start = np.flatnonzero(isl | np.concatenate([[True], isl[:-1]]))     # each letter, and the first of each run
    S = np.where(isl[start], c[start], 0).astype(np.uint8)
    # a silent run takes the longest silence in it
    mark = np.zeros(len(arr), bool); mark[start] = True
    runid = np.cumsum(mark) - 1
    L = np.zeros(len(start), np.int8); np.maximum.at(L, runid[~isl], lev[arr[~isl]])
    S = np.where(isl[start], S, 25 + L).astype(np.uint8)
    T = test[start]
    np.savez(f, S=S, T=T); return S, T


S, T = stream()
MODE = {"cells": HIS, "silence": "length", "n": 8}


def value_table():
    v = np.zeros(29, np.float32)
    for ch, cell in MODE["cells"].items():
        v[ord(ch) - 97] = cell / 30.0                                   # each cell's number as its share of the whole 30
    v[26:29] = SILENCE[MODE["silence"]]
    return v


def cells(arr, pos, way=None):
    v = value_table()
    return np.stack([v[arr[pos - back].astype(np.int64)] for back in range(1, MODE["n"] + 1)], 1)


def sides(maxlag, letters_only=True):
    t = T.astype(np.int64); c = np.concatenate([[0], np.cumsum(t)])
    p = np.arange(maxlag, len(S)); s = c[p + 1] - c[p - maxlag]
    ok = S[p] < 26 if letters_only else np.ones(len(p), bool)
    return p[(s == 0) & ok], p[(s == maxlag + 1) & ok]


def run(scrambled, fixed=None):
    arr = S.copy()
    if scrambled:
        arr = arr.copy(); np.random.default_rng(99).shuffle(arr)
    tr, te = sides(64)
    tr = tr[arr[tr] < 26]; te = te[arr[te] < 26]
    write_pos = tr[np.linspace(0, len(tr) - 1, TS.N_WRITE).astype(np.int64)]
    rest = np.setdiff1d(tr, write_pos); val = rest[np.linspace(0, len(rest) - 1, 50_000).astype(np.int64)]
    judged = te[np.linspace(0, len(te) - 1, 200_000).astype(np.int64)]
    y = arr[write_pos]; p = np.bincount(y, minlength=26) / len(y)
    out = dict(text="scrambled" if scrambled else "Webster as 30 cells", unigram_bits=round(float(-(p[p > 0] * np.log2(p[p > 0])).sum()), 4), aside={})
    best = None
    grid = [(fixed["n"],)] if fixed else [(8,), (16,), (32,), (64,)]
    for (n,) in grid:
        MODE["n"] = n; TS.KS = [k for k in [0, 2, 4, 6, 8, 12, 16, 24, 32, 48] if k < n]
        for layout in (("his",) if fixed else ("his", "alphabetical")):
            MODE["cells"] = HIS if layout == "his" else ALPHA
            for sil in ((fixed["silence"],) if fixed else SILENCE):
                MODE["silence"] = sil
                mem = TS.write(arr, write_pos, "x", max(TS.KS))
                b, k, Tt = TS.choose(mem, cells(arr, val), arr[val], "flat")
                out["aside"][f"n{n} {layout} silence={sil}"] = [round(b, 4), k, round(Tt, 3)]
                print(n, layout, sil, round(b, 4), k, round(Tt, 3), flush=True)
                if layout == "his" and (best is None or b < best[0]):
                    best = (b, n, sil, k, Tt)
    _, n, sil, k, Tt = best
    MODE.update(n=n, silence=sil)
    for layout in (("his",) if fixed else ("his", "alphabetical")):  # judged once each, at his best settings
        MODE["cells"] = HIS if layout == "his" else ALPHA
        TS.KS = [kk for kk in [0, 2, 4, 6, 8, 12, 16, 24, 32, 48] if kk < n]
        mem = TS.write(arr, write_pos, "x", max(TS.KS))
        kk, TT = (k, Tt) if layout == "his" else TS.choose(mem, cells(arr, val), arr[val], "flat")[1:]
        bpl, acc = TS.judge(mem, cells(arr, judged), arr[judged], "flat", kk, TT)
        out[f"judged_{layout}"] = dict(n=n, silence=sil, k_turns=kk, T=round(TT, 3), held_out_bits_per_letter=round(bpl, 4), next_letter_right=round(100 * acc, 2))
    return out


if __name__ == "__main__":
    TS.cells = cells; TS.NCLASS = 26; TS.TS = np.geomspace(1e-4, 50.0, 70)      # temperatures wide enough for numbers
    real = run(False); print(json.dumps(real), flush=True)
    ctrl = run(True, fixed=real["judged_his"]); print(json.dumps(ctrl), flush=True)
    with open(os.path.join(HERE, "textcells.jsonl"), "a") as f:
        f.write(json.dumps(real) + "\n"); f.write(json.dumps(ctrl) + "\n")
