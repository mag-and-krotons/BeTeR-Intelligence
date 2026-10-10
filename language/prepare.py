"""Prepare the text: Webster's Unabridged Dictionary (Project Gutenberg EBook #29765, the 1913 Webster).
  python3 prepare.py path/to/29765.txt
writes into ../data:
  text.npy        the text after the Gutenberg header and footer as 83 characters (every character seen at least 1,000
                  times; the rarer ones read as '~')
  test_mask.npy   every 20th block of 50,000 characters held out (1.35 million characters)
  vocab.json      the 83 characters
  positions.npz   training and held-out positions whose 64 characters before lie on the same side (the positions the
                  trained network was trained and judged on)
  cells_stream.npz  the letter stream: letters a..z (accents folded) and one silence per run of non-letters (built by
                  textcells.py on first import)
The results in this repository were computed on a file whose SHA-256 is given in the README.
"""
import hashlib, json, os, sys
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
BLOCK, TEST_EVERY, CTX = 50_000, 20, 64


def positions(test, held):
    ok = test if held else ~test
    idx = np.arange(len(ok))
    last_bad = np.maximum.accumulate(np.where(~ok, idx, -1))
    return np.flatnonzero(idx - last_bad > CTX)                  # a run of more than 64 same-side characters ends here


def main(src):
    raw = open(src, "rb").read()
    print("sha256 of the source:", hashlib.sha256(raw).hexdigest())
    t = raw.decode("utf-8", errors="replace")
    a = t.find("*** START OF"); b = t.find("*** END OF")
    t = t[t.find("\n", a) + 1:b if b > 0 else None]
    c = Counter(t)
    chars = sorted(k for k, v in c.items() if v >= 1000 and (k.isprintable() or k == "\n"))
    if "~" not in chars:
        chars.append("~")
    idx = {ch: i for i, ch in enumerate(chars)}
    arr = np.frombuffer(bytes(idx.get(ch, idx["~"]) for ch in t), dtype=np.uint8)
    test = np.zeros(len(arr), dtype=bool)
    for k in range(len(arr) // BLOCK):
        if k % TEST_EVERY == TEST_EVERY - 1:
            test[k * BLOCK:(k + 1) * BLOCK] = True
    os.makedirs(DATA, exist_ok=True)
    np.save(os.path.join(DATA, "text.npy"), arr); np.save(os.path.join(DATA, "test_mask.npy"), test)
    json.dump({"chars": chars, "length": int(len(arr)), "test_chars": int(test.sum()),
               "rare_as_tilde": int(sum(v for k, v in c.items() if k not in idx))},
              open(os.path.join(DATA, "vocab.json"), "w"), ensure_ascii=False, indent=1)
    np.savez(os.path.join(DATA, "positions.npz"), train=positions(test, False), test=positions(test, True))
    print(len(chars), "characters;", len(arr), "positions;", int(test.sum()), "held out")
    print("sha256 of text.npy:", hashlib.sha256(arr.tobytes()).hexdigest())
    import textcells                                              # builds cells_stream.npz
    print("letter stream:", len(textcells.S), "nodes")


if __name__ == "__main__":
    main(sys.argv[1])
