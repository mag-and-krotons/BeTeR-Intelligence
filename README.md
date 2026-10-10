# BeTeR Intelligence

**Reading images and language with no training.** Abhijit Singh, October 2026. Licence: CC BY 4.0.

| | no training | |
|---|---|---|
| **Handwritten digits** (UCI; 1,258 training images, 539 test) | **98.08 ± 0.38%** | memory written in 0.004 s |
| **Language: a letter read from the flow** (Webster's dictionary; 200,000 held-out letters, 3.84 million counted) | **93.0% right, 0.363 bits per letter** | each step runs in minutes on one CPU |

There are no trained parameters. Nothing is fitted by gradient descent and no network weights exist. Each architecture writes its memory in one pass. Each is judged on held-out data it has never seen, beside a scrambled control in which nothing should be found (digits 10.8%, language 11.7%).

Both come from one principle, the triad: a pair read forward and in reverse, the middle that stays, and the turn between the two ends. Every gain in accuracy below came from a step of that principle, not from parameters or data. Each step was tested against a control.

- **The digit memory:** each digit's still pattern and its own turning directions. It matches a trained engine of 337,298 numbers, and it forgets nothing: written two digits at a time, it still reads 97.7%.
- **The language memory:** the flow of the spoken language that the letters write down. Letters stand for sounds; punctuation is not sound and becomes one silence; silence is a node of the flow, not an absence. Language flows before it is written, and the model reads that flow.

## The flaw in "context"

In today's language models, context means the window of symbols before a position. Intelligence is taken to be predicting the next symbol from that window. When it falls short, the window, the parameters and the data are made larger. Each of those three meanings fails in the measurements here. All counts are fixed at the same 3.84 million letters.

**1. Context is not one-sided.** A letter is the middle of two flows: what flows into it and what flows out of it. The same six letters read three ways:

| the same 6 letters | right |
|---|---|
| as a window before the letter (what "context" means now) | 51.8% |
| as the flow into it and the flow out of it, 3 each side (the pair) | 63.2% |
| as three triads (x₋ₖ, x, x₊ₖ), each triad's two ends counted together (the pair and its turn) | **81.0%** |

The information was in the six letters all along. The one-sided window reads half of it (51.8% here; 53% with paths of up to 12 letters).

**2. A longer window adds nothing.** Past nine letters back, each further letter adds less than 0.003 bits. Triads beyond the third add 1.9 points (81.0% to 82.9% at six).

| letters back | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bits per letter (held aside) | 3.573 | 3.079 | 2.642 | 2.349 | 2.220 | 2.175 | 2.156 | 2.148 | 2.146 | 2.144 | 2.142 | 2.141 |

**3. Crossing the silence makes it worse.** When the triads reach across a silence into the next word, they read worse. Stopped at the silence, they read better: 82.9% to 84.8%, and at the start of a word 75.0% to 79.0%. More window, less accuracy: the word bounds the reading.

**4. What helps from far away is the return.** It is not a wider window but the same pattern coming back in another form. The dictionary says a word, then says it again in syllables, "suffocating, suf·fo·ca·ting". Read with the silences dropped, the letters around a letter recur, and the letter in their middle is read there. Where such a return is found, the letter goes from 76–80% to 96–99% right.

**5. The scale above is the same triad.** The word enters as a node: the word before, the word, the word after. It does not enter as more letters. This takes the reading from 91.8% to 93.0%.

| step, counts fixed at 3.84 million letters | bits | right |
|---|---|---|
| six letters as a window before the letter | 2.242 | 51.8% |
| the pair: flows into and out of the letter | 1.618 | 63.2% |
| the turn: three triads, ends counted together | 0.970 | 81.0% |
| six triads | 0.911 | 82.9% |
| the triads stop at silence | 0.829 | 84.8% |
| still, turn and Boltzmann readings together | 0.676 | 87.5% |
| the return | 0.412 | 91.8% |
| the scale above: the word triad | **0.363** | **93.0%** |
| scrambled text, all of the above | 4.20 | 11.7% |

Context is not a window that grows. It is the triad, bounded by silence, returning, and nested by scale. Read that way, the same 3.84 million letters give 93.0%. Five times as many letters, read as triads without the rest of the structure, give 87.0%.

## More is good, but not without end

Every curve here flattens, and one turns back.

**Digits, images per digit** (five draws each):

| images per digit | 3 | 6 | 12 | 24 | 48 | 96 | all (~126) |
|---|---|---|---|---|---|---|---|
| right | 62.1% | 76.9% | 87.2% | 93.6% | 96.0% | 97.7% | 98.0% |

**Digits, turning directions each digit keeps.** More turns help up to eleven, then hurt:

| turns | 0 | 2 | 4 | 6 | 8 | 11 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| right | 90.4% | 96.0% | 97.4% | 98.0% | 98.3% | **98.6%** | 97.9% | 97.5% | 96.9% | 95.8% |

**Language, letters counted** (six triads):

| letters counted | 240 thousand | 960 thousand | 3.84 million | 15.4 million | 19.4 million (all) |
|---|---|---|---|---|---|
| right | 71.8% | 78.1% | 82.9% | 86.5% | 87.0% |

Each fourfold step buys less: 6.3, 4.8, then 3.6 points, and the last 1.26 times buys 0.5.

**At the scale of today's language models.** The laws their builders fitted show the same shape. Loss falls as a power law in parameters and in data, toward an irreducible floor (Kaplan et al. 2020, arXiv:2001.08361; Hoffmann et al. 2022, arXiv:2203.15556, L(N, D) = E + A/N^α + B/D^β). Every tenfold step buys less than the one before. By their own fit, no number of parameters takes the loss below E.

## No training, on the trained network's own terms

Today's models are judged on the next symbol from the window before it, the one-sided reading above. On that task, a network of 0.83 million weights was trained on Webster's dictionary for 30,000 steps of 128 windows (3.84 million positions; about 6,400 s on one CPU core). The flow was counted on the same 83 characters and the same 3.84 million positions, and both were judged on the same 200,000 held-out positions:

| | bits per character | trained parameters | time |
|---|---|---|---|
| trained network, float weights | 2.410 | 831,363 | about 6,400 s |
| the flow, counted | **2.098** (58.2% right) | **0** | **15 s** |
| the flow, a quarter of the data (960 thousand) | 2.277 | 0 | 7 s |
| the flow, all 26.5 million positions | 1.899 (61.2% right) | 0 | 91 s |
| control: scrambled text | 4.938 (the characters' own entropy 4.936) | | |

The flow is at its weakest here: it reads one direction only, the half that "context" keeps. Even so, counting beats training at equal data, and still beats it with a quarter of the data. All 26.5 million positions, 6.9 times the network's budget, buy 0.20 bits.

On digits, the trained Grassmannian Neural Engine reaches 98.08 ± 0.47% with 337,298 trained numbers and about 170 s of training. The six-image stack reaches 98.08 ± 0.38% with none, written in 0.004 s.

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

### The triad flow (`language/`)

The language is a flow through 27 nodes: the 26 letters, which write down sounds, and silence. Punctuation is not sound, so a run of non-letters is one silence. What follows a path is counted. A path walked rarely leans on the shorter path inside it, in proportion β:

P_n(x | path) = (count(path, x) + β · P_{n−1}(x | shorter path)) / (count(path) + β)

| step | what is read | script |
|---|---|---|
| **The pair** | the flow into the letter and the flow out of it | `flowpm3.py`, `flowreach.py` |
| **The turn** | the ends of the nested triads (x₋ₖ, x, x₊ₖ) counted together, innermost first | `flowtriads.py` |
| **Silence** | each side of a triad stops at the first silence | `flowword.py` |
| **The Boltzmann reading** | every stored window speaks with weight count × e^(−r/T), r the triads that do not match | `flowboltz.py` |
| **The return** | the letters around it recur within 160 letters, silences dropped; the letter in their middle is read | `flowcopy.py`, `flowcopy2.py` |
| **The scale above** | the word triad, with each candidate letter put in its word | `flowscale.py` |

The readings are joined log-linearly. Under a hundred numbers, the joining weights and smoothing constants, are chosen on 50,000 held-aside training positions. That is the only choice made from data beyond counting. `flowlook.py` lays all readings side by side and shows what each holds that another lacks; each later step puts one of those back.

**Tried and recorded as failed:**
- the digit stack laid over the flow's readings: 2.42 bits against 1.61 (`flowstack.py`);
- Pāṇini's grid of the sounds as a coarser path (`flowgrid.py`, `flowdiff.py`) and at the triads' ends: no better than random deals of the letters into cells of the same sizes;
- each letter keeping its own turns in the Boltzmann reading: the same as one weight per triad;
- the next letter read both ways (`flowpair.py`): 53.5% against 53.0%; for the next letter the flow out of it is not there to be read.

**Not shown here.** Reading a letter from the flow is not yet writing text. Choices that need the meaning, such as "stat[u]s / states" or "cou[n]t / court", are among the 7% still wrong.

## Running

numpy only.

```bash
cd digits && python3 stack6.py && python3 curve.py
```

The text is Webster's Unabridged Dictionary (Project Gutenberg EBook #29765). The results were computed on a file with SHA-256 `cce41c60d56db8964c2b304ce525187e68c3cb40c07e4cf6de71ef5403944c4d`, which gives `text.npy` with SHA-256 `0b3413f6aca9a422e9a266f8b9c437647eee1f944a5bee056adfe00494027f24`.

```bash
cd language
python3 prepare.py path/to/29765.txt
python3 flowtriads.py && python3 curve_tower.py && python3 flow83.py
python3 flowlook.py && python3 flowword.py && python3 flowword2.py && python3 flowcopy.py
REACH=160 python3 flowcopy2.py && COPY=flowcopy2_160 python3 flowscale.py
```

Every run writes its JSON into `results/`; the files there are the runs these tables come from.

## Data

- Handwritten digits: E. Alpaydin and C. Kaynak, *Optical Recognition of Handwritten Digits*, UCI Machine Learning Repository (CC BY 4.0); `data/optdigits.tes`, the 1,797 digits of 8 × 8 cells.
- Text: *Webster's Unabridged Dictionary*, Project Gutenberg EBook #29765 (public domain in the USA).

Research record: [10.5281/zenodo.22129334](https://doi.org/10.5281/zenodo.22129334). Site: <https://mag-and-krotons.github.io>.
