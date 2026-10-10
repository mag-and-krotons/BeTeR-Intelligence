# Triad Pattern: reading images and language with no training

**Author:** Abhijit Singh. October 2026. Licence: CC BY 4.0.

Two architectures from one principle, the triad of the Triad Principle: a pair read forward and in reverse, the middle that stays, and the turn between the two ends. Neither architecture trains: nothing is fitted by gradient descent, and no network weights exist. Each writes its memory in one pass over the data. Each judges on held-out data it has never seen, beside a scrambled control in which nothing should be found.

| | this architecture, no training | the trained network it is set beside |
|---|---|---|
| **Handwritten digits** (UCI, 1,258 training images, 539 test) | **98.08 ± 0.38%**; memory written in 0.004 s | Grassmannian Neural Engine: 98.08 ± 0.47%; 337,298 trained numbers; about 170 s of training |
| **Next character of Webster's dictionary** (83 characters, the same 200,000 held-out positions, the same 3.84 million training positions seen) | **2.098 bits per character, 58.2% right**; counted in 15 s | 0.83-million-weight network: 2.410 bits per character; about 6,400 s of training on one CPU core |
| **A letter read from both sides** (26 letters, 200,000 held-out letters, 3.84 million counted) | **0.363 bits per letter, 93.0% right** | |

## What the measurements show

**1. No training is needed for this accuracy.**
- The digit memory matches the trained engine's accuracy. It is written in 0.004 s, against minutes of training, and it holds no trained numbers.
- On text, the flow beats the trained network on that network's own task, its own data and its own judged positions: 2.098 against 2.410 bits per character. Counting took 15 s; training took about 6,400 s.
- With a quarter of the network's data (960 thousand positions) the flow already beats it: 2.277 bits.

**2. More is good, but not without end.** Every curve flattens; some turn back.

Digits, images per digit (five draws each):

| images per digit | 3 | 6 | 12 | 24 | 48 | 96 | all (~126) |
|---|---|---|---|---|---|---|---|
| accuracy (%) | 62.1 | 76.9 | 87.2 | 93.6 | 96.0 | 97.7 | 98.0 |

Digits, turning directions each digit keeps. More turns help up to about eleven, then hurt:

| turns | 0 | 2 | 4 | 6 | 8 | 11 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| accuracy (%) | 90.4 | 96.0 | 97.4 | 98.0 | 98.3 | **98.6** | 97.9 | 97.5 | 96.9 | 95.8 |

Text, positions counted (next character, 83 characters):

| positions counted | 60 thousand | 240 thousand | 960 thousand | 3.84 million | 15.4 million | 26.5 million (all) |
|---|---|---|---|---|---|---|
| bits per character | 2.850 | 2.521 | 2.277 | 2.098 | 1.946 | 1.899 |
| next character right | 45.9% | 51.0% | 55.0% | 58.2% | 60.2% | 61.2% |
| seconds to count | 4 | 4 | 7 | 15 | 53 | 91 |

Control: the same characters scrambled, 3.84 million counted: 4.938 bits per character (the characters' own entropy is 4.936).

Each fourfold step buys less: 0.33, 0.24, 0.18, then 0.15 bits. All the text, 6.9 times the network's budget, buys 0.20 bits and 3 points over 3.84 million.

Text, how far back the path reaches (letters and silence; held-aside bits per letter; 3.84 million positions):

| characters back | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bits | 3.573 | 3.079 | 2.642 | 2.349 | 2.220 | 2.175 | 2.156 | 2.148 | 2.146 | 2.144 | 2.142 | 2.141 |

Past nine characters, each further character adds less than 0.003 bits.

**3. Structure moves what more context and more data barely move.** A letter read from both sides, with all counts fixed at 3.84 million:

| reading | what it adds | bits | right |
|---|---|---|---|
| the 6 letters before it, as a context | | 2.242 | 51.8% |
| 3 letters each side, the two sides multiplied | the pair: forward and reverse | 1.618 | 63.2% |
| 3 letters each side, as three triads (x₋ₖ, x, x₊ₖ) counted together | the turn: how the two ends move together | 0.970 | 81.0% |
| 6 triads | | 0.911 | 82.9% |
| the triads stop at silence | the word bounds the window | 0.829 | 84.8% |
| still, turn and Boltzmann readings together | the long reach where the window runs out | 0.676 | 87.5% |
| the return | what flowed comes back: the dictionary repeats a word in syllables, silences dropped | 0.412 | 91.8% |
| the scale above: the word triad (word before, word, word after) | the word as a node | **0.363** | **93.0%** |
| scrambled text, all of the above | | 4.20 | 11.7% |

The same 6 letters read as a context before the letter get 51.8%. Read as three triads around it they get 81.0%.

Counting five times the letters (all 19.4 million) lifts the 6-triad reading from 82.9% to 87.0%:

| letters counted | 240 thousand | 960 thousand | 3.84 million | 15.4 million | 19.4 million (all) |
|---|---|---|---|---|---|
| 6 triads, right | 71.8% | 78.1% | 82.9% | 86.5% | 87.0% |

The structure above lifts it, with no more data, to 93.0%.

## The two architectures

### The six-image stack (`digits/`)

For a digit, three of its images a, b, c are read forward, (a − b) − c, and the same three in reverse, (c − b) − a.
- The sum of the two readings is −2b: the middle, what stays.
- The difference is 2(a − c): the turn between the two ends.

Each digit keeps its still pattern (the mean of its middles) and the main directions of its turns (one singular value decomposition). A new image is read as the digit whose still pattern, allowed to turn in that digit's own directions, leaves the least behind. The number of turns kept is chosen inside the training images.

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| all ten digits (turns chosen: 6, 6, 7) | 98.14 | 97.59 | 98.52 | **98.08 ± 0.38** |
| still pattern only, no turns | 89.98 | 89.61 | 89.80 | 89.80 |
| written two digits at a time, never shown again | 97.77 | 97.96 | 97.40 | 97.71 ± 0.23 |
| control: training labels scrambled | 7.61 | 9.46 | 15.40 | 10.8 (chance 10) |

Each digit's memory is its own: written two digits at a time, nothing written earlier is forgotten.

### The triad flow (`language/`)

The text is a flow through 27 nodes: the 26 letters and silence (a run of non-letters is one silence). What follows a path is counted. A path walked rarely leans on the shorter path inside it, in proportion β:

P_n(x | path) = (count(path, x) + β · P_{n−1}(x | shorter path)) / (count(path) + β)

A letter is read as the middle of its triads, in steps:

| step | what is read | script |
|---|---|---|
| **The pair** | the flow out of the letters before it and the flow back from the letters after it | `flowpm3.py`, `flowreach.py` |
| **The turn** | the ends of the nested triads (x₋ₖ, x, x₊ₖ) counted together, innermost first | `flowtriads.py` |
| **Silence** | each side of a triad stops at the first silence | `flowword.py` |
| **The Boltzmann reading** | every stored window speaks with weight count × e^(−r/T), r the triads that do not match | `flowboltz.py` |
| **The return** | the letters around it recur within 160 letters, silences dropped; the letter in their middle is read | `flowcopy.py`, `flowcopy2.py` |
| **The scale above** | the word triad, with each candidate letter put in its word | `flowscale.py` |

The readings are joined log-linearly. Under a hundred numbers, the joining weights and smoothing constants, are chosen on 50,000 held-aside training positions, one group for each depth at which a return or a word triad was found. That is the only choice made from data beyond counting. `flowlook.py` lays all readings side by side and shows what each holds that another lacks; each later step puts one of those back.

`flow83.py` is the flow on the trained network's own task, at six counting budgets. `curve_tower.py` is the triad tower at five budgets.

**Tried and recorded as failed:**
- the digit stack laid over the flow's readings: 2.42 bits against 1.61 (`flowstack.py`);
- Pāṇini's grid of the sounds as a coarser path (`flowgrid.py`, `flowdiff.py`) and at the triads' ends: no better than random deals of the letters into cells of the same sizes;
- each letter keeping its own turns in the Boltzmann reading: the same as one weight per triad;
- next-letter prediction read both ways (`flowpair.py`): 53.5% against 53.0%. For the next letter, the letters after it are not there to be read.

**What is not shown here.**
- Reading a letter from both sides is not writing text. For the next character the flow reaches 58.2% at 3.84 million positions.
- The trained network beside it is one network of 0.83 million weights.
- Choices that need the meaning, such as "stat[u]s / states" or "cou[n]t / court", remain among the 7% still wrong.

## Running

numpy only.

```bash
cd digits && python3 stack6.py && python3 curve.py
```

The text is Webster's Unabridged Dictionary (Project Gutenberg EBook #29765). The results were computed on a file with SHA-256 `cce41c60d56db8964c2b304ce525187e68c3cb40c07e4cf6de71ef5403944c4d`, which gives `text.npy` with SHA-256 `0b3413f6aca9a422e9a266f8b9c437647eee1f944a5bee056adfe00494027f24`.

```bash
cd language
python3 prepare.py path/to/29765.txt
python3 flow83.py
python3 flowtriads.py
python3 curve_tower.py
python3 flowlook.py && python3 flowword.py && python3 flowword2.py && python3 flowcopy.py
REACH=160 python3 flowcopy2.py && COPY=flowcopy2_160 python3 flowscale.py
```

Every run writes its JSON into `results/`; the files there are the runs these tables come from. Each script with a `SMOKE=1` path runs a small sample first.

## Data

- Handwritten digits: E. Alpaydin and C. Kaynak, *Optical Recognition of Handwritten Digits*, UCI Machine Learning Repository (CC BY 4.0); `data/optdigits.tes`, the 1,797 digits of 8 × 8 cells.
- Text: *Webster's Unabridged Dictionary*, Project Gutenberg EBook #29765 (public domain in the USA).

Research record: [10.5281/zenodo.22129334](https://doi.org/10.5281/zenodo.22129334). Site: <https://mag-and-krotons.github.io>.
