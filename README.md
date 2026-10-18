# Teaching a Local LLM to Play Chess

**A three-part study in fine-tuning, evaluation, and knowing when your own numbers are lying to you.**
Note: Report is AI-generated, this study was AI-assisted.

---

## Reproducing

Requires an NVIDIA GPU with 12GB+ VRAM, and Stockfish on PATH.

    pip install -r requirements.txt
    python extract_games.py          # streams 100k games from Lichess
    python convert_pgn_dataset.py    # converts to training format
    python finetune.py               # ~11 hours on an RTX 4070
    python compare_both.py           # paired evaluation

Game data, model weights, and evaluation dumps are regenerated
by these scripts rather than committed.

## Abstract

Can a general-purpose language model be fine-tuned to actually play chess, rather than to produce text that looks like chess? This project answers that question across three parts on a single consumer GPU.

**Part 1** trained an 8B model on board-state→move pairs (FEN format) across five configurations, testing dataset size, adapter capacity, epoch count, and targeted data augmentation. All four levers failed. The model produced illegal moves roughly two-thirds of the time on unfamiliar positions, and its legal moves scored no better than random against an engine.

**Part 2** reversed the project's foundational decision — abandoning FEN for PGN move sequences — on the hypothesis that the input format, not model capacity, was the bottleneck. Legality rose from ~32% to 85.4%. Move quality went from indistinguishable-from-random to roughly two pawns better than random. Benchmarked against the 1500-rated humans it was trained to imitate, the model matched them in the opening and blundered three times as often overall.

**Part 3** diagnosed the residual gap as tactical — a single forward pass contains no search — and built explicit lookahead: a linear probe confirmed the board is readable from the model's activations, a value head was trained on game outcomes, and one- to two-ply search was implemented over policy-generated candidates. Search eliminated illegal moves almost entirely (7.2% → 0.2%) and degraded move selection in two-thirds of positions.

**The through-line is methodological.** On four separate occasions a metric produced a result that later proved to be an artifact — including one prediction registered in advance that appeared confirmed and was not. Each was caught by asking whether the measurement could be trusted rather than whether the model could be improved.

---

## Hardware and Environment

| | |
|---|---|
| GPU | RTX 4070 (12 GB VRAM) |
| CPU | Ryzen 7 7700 |
| RAM | 32 GB |
| Base model | `unsloth/Meta-Llama-3.1-8B`, 4-bit quantized |
| Method | LoRA via Unsloth |
| Stack | PyTorch 2.13.0 (CUDA 13.0), `python-chess`, Stockfish 18 |

**Setup issues resolved:** PowerShell execution policy blocking venv activation; project directory accidentally created inside `Windows/System32`; Unsloth's install silently replacing the CUDA PyTorch build with a CPU-only one; the resulting torch/torchvision version mismatch; Hugging Face Xet-backend download stalls.

### Data source

Lichess public game database (CC0). Rather than downloading a ~28 GB monthly archive, games were streamed directly from the server with `requests` and decompressed on the fly with `zstandard`, terminating the stream once enough qualifying games were collected. Both players filtered to 1500+ Elo.

---

# Part 1 — Board State as Input (FEN)

## Design decision

Each training example paired a board position, encoded as a FEN string, with the move played from it. The alternative — training on raw move sequences — was rejected early on the grounds that it would force the model to track board state implicitly, making both learning and failure diagnosis harder. FEN supplies the position explicitly.

*This decision was the central error of Part 1, and is revisited in Part 2.*

## Runs and results

Evaluation metric: **legal-move streak** — how many consecutive legal moves the model produces before an illegal one, averaged over 20 trials.

| Run | Examples | Rank | Epochs | Data | Avg streak | Max |
|-----|----------|------|--------|------|-----------|-----|
| 1 | 3,000 | 16 | 2 | clean | 3.35 | 7 |
| 2 | 20,000 | 16 | 2 | clean | 3.85 | 7 |
| 3 | 20,000 | 32 | 2 | clean | 3.65 | 10 |
| 4 | 20,000 | 32 | 4 | clean | **1.15** | 5 |
| 5 | 30,000 | 16 | 2 | clean + recovery | 3.25 | 8 |

Every run produced a minimum streak of 0 — at least one trial where the model failed on its very first move.

**Run 4 is the clearest single result.** Training loss fell from ~0.45 to ~0.29 while held-out performance collapsed by two-thirds: textbook overfitting, and direct evidence that falling training loss cannot serve as a proxy for quality on this task.

**Runs 1, 2, 3, and 5 span 3.25–3.85**, a range narrow enough that a 20-trial evaluation cannot distinguish them. Differences within that band were treated as tentative rather than as measured effects.

## Recovery-data augmentation

Diagnosing the failure as distribution shift — the model trains only on positions from well-played games, so its own mistakes push it into territory it has never seen — motivated a targeted fix. Real games were perturbed by substituting a random legal move at random points, and Stockfish 18 supplied the correct response for the resulting off-book position (necessary because after a perturbation the real game's continuation is frequently illegal). 20,000 clean examples were merged with 10,000 recovery examples and shuffled.

It did not work. The paired test is covered below.

## Three metrics that lied

### 1. Single-game fallback rate ranked a broken model first

A full-game harness (model proposes, `python-chess` validates, random legal move as fallback) initially reported the badly overfit Run 4 model at a ~46% fallback rate versus ~58% for the healthy Run 2 model. A model known to be broken cannot genuinely be best, so the metric was suspect rather than the models.

Running 10 games per model instead of one:

| Model | Fallback rate |
|---|---|
| Run 5 (recovery) | 41.6% ± 6.4% |
| Run 2 (clean) | 38.3% ± 7.0% |

The standard deviation is the useful output. Single-game readings swing by ±6–7 points, which fully accounts for the contradictory figures (58%, 49%, 46%) drawn from that spread.

### 2. An unpaired quality comparison invented a 182 cp effect

Introducing centipawn loss — how much worse a move is than Stockfish's best — required a baseline, since the number is meaningless alone. A uniformly random legal move was scored on every position.

First result, one run per model: the clean model scored 297 cp against a 304 cp random baseline (no better than random), while the recovery model scored 518 against 745 (roughly 30% better). The wildly different random baselines revealed the flaw: the two models had faced different randomly generated position sets.

Seeding the generator so both faced identical positions stabilized the baseline at 423–428 but produced a 182 cp gap on one seed and 23 cp on the other. A remaining flaw explained the instability: each model's average covered only the positions *it* answered legally — roughly 100 of 300, and not the same 100.

**Paired comparison**, intersecting by FEN so both models were scored only where both played legally:

| Seed | Clean | Recovery | Difference | Std error | Recovery better on |
|------|-------|----------|-----------|-----------|--------------------|
| 42 | 297 cp | 270 cp | +27 cp | 33 cp | 22/43 |
| 1337 | 203 cp | 217 cp | −14 cp | 33 cp | 24/51 |

Both differences fall below one standard error, they point in opposite directions across seeds, and the per-position win rates (51%, 47%) are coin flips. The 182 cp effect was entirely an artifact of self-selected position subsets.

### 3. Short-horizon evaluation overstated capability

Streak evaluation reported ~3.5 legal moves from near-opening positions. Full-game play required the safety net for ~40% of moves. Both were valid; they measured different things, and only the latter reflected real play.

## Part 1 conclusions

> These conclusions were substantially revised by Part 2. The claim that an 8B model "does not learn to play chess" turned out to be a claim about the *input format*. Left unedited because the revision is the point.

**On the model.** An 8B model LoRA fine-tuned on FEN→move pairs does not learn to play chess. It learns the surface form of chess notation — valid SAN strings, plausibly placed check and capture markers, including `#` on non-mating moves — without acquiring board-state understanding underneath. On off-distribution positions it produced illegal moves roughly two-thirds of the time, and its legal moves scored no better than random against an engine.

**On what was tried.** Five configurations across four levers: data volume (small gain), adapter capacity (inconclusive), epochs (actively harmful), targeted augmentation (no effect under a controlled test).

**On evaluation.** Three times, a metric manufactured a result. In each case the correction came from interrogating the measurement rather than the model. The most substantive finding of Part 1 is not about chess: on a task where the underlying capability is weak, a poorly controlled metric will reliably produce the result you were hoping for.

---

# Part 2 — Move Sequences as Input (PGN)

## Hypothesis

Part 1 assumed the format was fine and the model was weak. The opposite had never been tested, because FEN was chosen in the project's first twenty minutes and never revisited.

Three reasons to suspect it:

- **Tokenization.** A FEN string (`rnbqkb1r/pp1p1ppp/4pn2/...`) is split into chunks that do not align with squares, ranks, or pieces. There is no cheap way to learn that a substring means "knight on f6."
- **Prior work.** Published results getting LLMs to play real chess — Karvonen's Chess-GPT and follow-ups demonstrating an extractable internal board representation — train on PGN move sequences, the exact format ruled out at the start.
- **Supervision density.** One FEN example teaches one prediction. One PGN game teaches a prediction at every move, since next-token loss applies across the whole sequence.

## What changed

| | Part 1 (FEN) | Part 2 (PGN) |
|---|---|---|
| Input | Board state string, one position per example | Full move sequence, one game per example |
| Games pulled | 5,000 | 100,000 (99,186 after filtering) |
| Effective examples | 20,000–30,000 move predictions | 50,000 games ≈ 2.9M move predictions |
| Max sequence length | 4,096 | 1,024 |
| Config | rank 16, 2 epochs | rank 16, 1 epoch, batch 2 |

Sequence length was verified rather than assumed: across 2,000 sampled games, token lengths ran to a median of 279, p99 of 648, and a maximum of 824 — **0 of 2,000 exceeded 1,024**, so nothing was silently truncated. One epoch was deliberate: Run 4's overfitting came from repeated passes over a small example set, not a regime that 50,000 unique games represents.

### An expensive infrastructure failure

The first attempt ran at 57 s/it with a 174-hour ETA. The cause was not compute but **VRAM overflow spilling into system RAM over PCIe.** On Windows the NVIDIA driver does not raise an out-of-memory error; it silently falls back to shared memory and runs roughly 40x slower. `nvidia-smi` showed 11,796 / 12,282 MiB.

Two lessons, at a cost of about 12 hours:

1. The reasoning behind batch size 4 was wrong. Shortening sequences 4x does not free 4x the memory — weights and optimizer state are fixed overhead (~6 of 12 GB) regardless of sequence length.
2. The failure mode watched for was wrong. The check that would have caught this in 60 seconds was comparing s/it against a known-good baseline, not waiting for an OOM crash Windows never produces.

Resolved by dropping to batch size 2 *and* restarting from a clean process — the killed run had not released its VRAM, so the first retry inherited a partially occupied GPU. Final run: 6,250 steps, 5h42m, 3.29 s/it.

## Legality results

| Phase | Legal rate |
|-------|-----------|
| Opening (1–10) | **99.5%** |
| Middlegame (11–30) | 79.8% |
| Endgame (31+) | 83.2% |
| **Overall** | **85.4%** |

Against Part 1's best model at roughly 32%, this is a 2.5x improvement on the metric that had defeated every previous configuration.

### A fourth metric failure — this one predicted in advance

Before training, a specific prediction was registered: opening moves appear in nearly every game while move 45 is unique to each match, so the model should master openings and degrade as data grows sparse. The first evaluation appeared to confirm it as a clean monotonic gradient: 98.0% → 79.8% → 74.7%.

Playing a full game by hand revealed the problem. When the model failed, the harness substituted a **uniformly random legal move** — which both wrecks the position and writes into the game history a move no 1500+ player would make. Since the model's prompt *is* the move history, every subsequent prediction was then conditioned on a sequence unlike anything in training. Fallbacks compounded, and late-game figures were partly measuring accumulated contamination.

Replacing the fallback with Stockfish at depth 1 — deliberately weak, but sane:

| Phase | Random fallback | Weak-engine fallback | Δ |
|-------|----------------|---------------------|---|
| Opening (1–10) | 98.0% | 99.5% | +1.5 |
| Middlegame (11–30) | 79.8% | 79.8% | 0.0 |
| Endgame (31+) | 74.7% | **83.2%** | **+8.5** |
| Overall | 82.6% | 85.4% | +2.8 |

Standard deviation across games fell from 7.6% to 4.5%, consistent with removing a source of compounding divergence.

**The gradient is no longer monotonic** — 99.5% → 79.8% → 83.2%, with the middlegame now weakest. That kills the sparsity explanation, since endgame positions are sparser still and score higher. A better account is that difficulty tracks **position complexity, not move number**: middlegames have the most pieces interacting and the largest legal-move sets.

A prediction registered in advance appeared confirmed by the data and was then shown to have been confirmed by an artifact of the instrument — the condition under which a spurious confirmation is hardest to catch.

## Move quality

Paired against a random legal move on identical positions:

| Seed | Model | Random | Difference | Std error | Model better on |
|------|-------|--------|-----------|-----------|-----------------|
| 42 | 350 cp | 570 cp | **220 cp** | 37 cp | 797/1024 (77.8%) |
| 1337 | 378 cp | 548 cp | **170 cp** | 46 cp | 774/1044 (74.1%) |

Four to six standard errors, replicating across seeds in direction and magnitude, with per-position win rates agreeing within four points.

The instrument is identical to the one that found nothing in Part 1:

| | Difference | Std error | Win rate |
|---|---|---|---|
| Part 1 — recovery vs. clean | 27 cp | 33 cp | 51% |
| Part 2 — PGN model vs. random | 220 cp | 37 cp | 78% |

Comparable sample sizes, opposite verdicts. This retroactively validates Part 1's null: the test could always detect a real effect, so the failure to find one there was a fact about the model rather than the measurement.

## Benchmarking against the training distribution

Centipawn loss against Stockfish measures distance from perfect play, which is the wrong yardstick for a model trained to imitate 1500-rated humans. An imitation model cannot exceed its teachers.

**Human baseline**, 1,061 moves from real 1500+ games:

| | Mean | Median | Blunders (300+ cp) |
|---|------|--------|--------------------|
| Human 1500+ | 170 cp | **16 cp** | **5.8%** |

The mean-median gap is the important part. A 1500 player is nearly optimal most of the time — half of all moves fall within 16 cp of best — and occasionally detonates. The mean is driven almost entirely by that ~6% tail.

### Decoding settings had been wrong throughout

A temperature sweep revealed that every quality figure reported to that point had used the worst available setting. Greedy decoding beat sampling on legality by 4–6 points (88.8% vs 83–85%), with a clear mechanism: the top-1 token is the model's most confident continuation and therefore most likely to be legal.

An earlier version of this sweep appeared to show greedy also halving centipawn loss (165 vs 344). Artifact: greedy decoding is deterministic, so all ten "games" were one game replayed ten times, and n=563 was really ~56 positions counted tenfold. After randomizing opening plies, quality differences across temperatures became non-monotonic — the signature of noise. **Greedy wins on legality; temperature does not affect quality.**

### Paired against humans on identical positions

Step through real games; at each position ask the model for a move, score it, score what the human actually played, then push the *human's* move so both stay on one trajectory. Every position shared.

**952 positions, greedy decoding:**

| | Mean | Median | Blunders |
|---|------|--------|----------|
| Model | 325 cp | 22 cp | **17.1%** |
| Human 1500+ | 158 cp | 14 cp | **5.6%** |

**The medians are close. The blunder rates are not.** The model's typical move is roughly as good as a human's; it throws the game away three times as often — matching the experience of playing it, which is sound for a stretch and then several pieces given away in succession.

| Phase | Model | Human | Ratio |
|-------|-------|-------|-------|
| Opening (1–10) | 40 cp | 34 cp | 1.2x |
| Middlegame (11–30) | 518 cp | 242 cp | **2.1x** |

In the opening the model is statistically indistinguishable from a 1500 player. In the middlegame it is more than twice as bad — on positions it did not create, so this is move selection rather than compounding drift.

### A second registered prediction, also falsified

Training on all 99,186 games (2x the data, ~11 hours) was predicted to leave openings unchanged, meaningfully reduce middlegame loss, and cut the blunder rate to 10–12%.

| | 50k games | 99k games | Predicted |
|---|-----------|-----------|-----------|
| Middlegame | 518 cp | **501 cp** | drop meaningfully |
| Blunder rate | 17.1% | **16.4%** | fall to 10–12% |
| Opening | 40 cp | 37 cp | unchanged ✓ |
| Illegal moves | 10.3% | **7.7%** | — |

Middlegame moved 3%; the blunder rate moved 0.7 points. Both sit well inside a standard error of 38 cp. Only the trivially safe prediction held.

**A trap worth flagging:** the headline model−human difference appeared to improve from 167 to 124 cp. It did not. The *human* baseline rose from 158 to 194, because the human comparison set contains only positions where the model played legally — and that set changed when legality improved. The model's own mean went 325 → 318.

### The pattern

**Legality responds to more data; move quality does not.** Legality is syntactic — with enough examples the model learns which continuations are well-formed. Quality is not.

Blunders are tactical. Not hanging a bishop requires checking whether the destination square is defended, which is one ply of lookahead. **A single forward pass through a transformer contains no search.** Imitation yields pattern recognition over positions resembling training positions — precisely where the model already matches humans. Middlegames require calculation, and nothing in the architecture calculates.

---

# Part 3 — Adding Search

## The question

If the residual gap is tactical rather than statistical, more data cannot close it and explicit lookahead might. Search requires two components: a **policy** (position → plausible move), which Part 2 produced, and a **value function** (position → how good is this for me), which did not exist.

A value function reading from the model's activations first requires those activations to encode the position at all.

## Does the model represent the board?

A linear probe was trained on final-token hidden states to predict the occupant of each of the 64 squares (13 classes: empty, six white pieces, six black). The baseline to beat is majority-class prediction, since most squares are empty most of the time.

| Layer | Square accuracy | Above baseline |
|-------|----------------|----------------|
| 8 | 76.9% | +9.6 |
| 16 | 79.7% | +12.4 |
| 24 | 84.5% | +17.2 |
| **31 (final)** | **84.9%** | **+17.6** |

Baseline: 67.3%. At layer 31 the probe recovers roughly 54% of available headroom.

**Accuracy rises monotonically and peaks at the final layer.** The prediction beforehand was the opposite — that late layers would specialize toward next-token prediction and shed general state. Instead the board becomes most readable exactly where the model commits to a move, which is what one would expect if the representation is functionally used rather than incidental.

One limit: it never plateaus, so 84.9% may be a ceiling or simply where the network ran out.

## Value head

39,402 positions were collected from real games, each paired with the eventual game result from the perspective of the side to move (target mean 0.515 — balanced, so the head cannot cheat by favoring one side). An MLP was trained on layer-31 hidden states.

The first attempt overfit hard: test MSE bottomed at 0.2301 around epoch 5, then climbed to 0.2981 by epoch 30 while training loss fell to 0.093. Winner accuracy peaked at 58.4% and decayed toward chance — and the script had been saving the final epoch rather than the best.

Regularized (LayerNorm on input, smaller head, dropout 0.4, weight decay 0.1, early stopping, best-checkpoint saving):

| | Value |
|---|---|
| Constant-prediction baseline MSE | 0.2343 |
| Best test MSE (epoch 7) | **0.2278** (+2.8%) |
| Winner accuracy | **55.9%** |

By the end, train and test loss had nearly converged (0.2213 vs 0.2300) — **not** overfitting, meaning more data would not help. The limit is signal, not sample size.

**The cause is label noise.** The head is asked to predict who wins a game with 30 moves of 1500-level blunders still to come. A position can be objectively winning and still be thrown away — exactly what the blunder analysis showed these players doing. The label is only loosely coupled to the position.

An engine-labeled value head (distilling Stockfish evaluations) would have removed this noise entirely. It was rejected deliberately: the project's premise is a model learning chess from human games, and distilling an engine's judgment would answer a different question.

## Search implementation

- **Candidates:** the greedy policy move plus 12 sampled completions at temperature 0.8, deduplicated, illegal moves discarded.
- **Evaluation:** each resulting position scored by the value head.
- **Depth:** two ply — the opponent's best policy reply is played before evaluating, so the score reflects the position after the likely response.
- **Terminal handling:** checkmate and stalemate detected directly rather than passed to the head.

### What a self-game showed before any metric ran

At move 12 the head rated `Qd8+` highest of four candidates at 0.568 — a queen sacrifice for nothing, immediately answered by `Kxd8`. At move 22 it rated `Rxb7` at 0.852, the highest value of the entire game; the rook was captured two moves later.

A head that ranks hanging a queen and hanging a rook as its best available moves is not evaluating material. In the opening, candidate scores separated by roughly 0.005 — selection by noise, from a head with 55.9% winner accuracy.

## Results

Both methods were scored **in the same loop on identical positions**, avoiding the self-selected-subset error that produced the phantom 182 cp effect in Part 1.

**Legality, over all 516 positions:**

| Method | Illegal |
|---|---|
| Greedy (no search) | 7.2% |
| Search | **0.2%** |

**Move quality, paired on the 479 positions where both played legally:**

| | Mean | Median | Blunders |
|---|------|--------|----------|
| Greedy | 337 cp | **20 cp** | **16.3%** |
| Search | 367 cp | **40 cp** | **21.5%** |
| Human 1500+ | 171 cp | 13 cp | 4.8% |

| | Value |
|---|---|
| Search − Greedy (mean) | +31 cp, SE 49 |
| Search better on | **161/479 (34%)** |

| Phase | Greedy | Search | Human |
|-------|--------|--------|-------|
| Opening (1–10) | 42 cp | 54 cp | 35 cp |
| Middlegame (11–30) | 526 cp | 568 cp | 257 cp |

### Reading this correctly

The mean difference is +31 cp against a standard error of 49 — **less than one standard error, so by the standard applied throughout this project, not an effect.** The unpaired figures (317 vs 415) had suggested a large degradation; that gap was the confound, since search answered 36 positions greedy could not, and those are plausibly the hardest on the board.

But the mean is the wrong statistic here. **Search played better on only 161 of 479 positions — it loses roughly two of every three.** A 34% win rate over 479 paired samples is not noise, and it is the same check that dissolved the recovery-data result in Part 1, pointing the other way. The median doubling (20 → 40) agrees, and a median is robust to tail effects.

Both readings are correct because they measure different things. Search **systematically selects slightly worse moves** while leaving the mean unchanged: the distribution shifts without the tail worsening, because greedy's blunders were already the ceiling.

## Part 3 conclusion

**A value function learned from human game outcomes was strong enough to eliminate illegal moves and too weak to rank legal ones.**

Candidate generation solved legality outright — 7.2% to 0.2% — because discarding illegal candidates requires no judgment, only a rules engine. Ranking the survivors by the value head degraded move selection in two-thirds of positions, because the policy's own ordering carried better information than a 55.9%-accurate evaluator's opinion, and search discards that ordering.

The bottleneck was never search architecture. It was evaluation quality, which was capped by label noise from 1500-level games — upstream of everything Part 3 could build.

---

# Overall Conclusions

**The input format mattered more than every training hyperparameter combined.** Part 1 tested four scaling levers against the wrong variable. Reversing one decision made in the project's first twenty minutes produced a larger improvement than all of them together: legality from ~32% to 85.4%, move quality from random-equivalent to two pawns better than random.

**Different capabilities have different bottlenecks.** Legality responded to data volume, to input format, and finally to candidate filtering. Move quality responded to none of them. Treating "plays chess" as one capability obscured that; separating legality from quality made both tractable to reason about.

**The model has a partial board representation.** Linearly decodable at 84.9% square accuracy, rising monotonically through the network and peaking at the final layer. It is real, and it is not clean enough to support a reliable evaluation function.

**Search without a good evaluator is worse than no search.** It solved the problem requiring no judgment and degraded the one requiring it.

**On method.** Four times, a metric produced a result that proved to be an artifact: a single-game fallback rate that ranked a broken model first; an unpaired comparison that invented a 182 cp effect; a monotonic phase gradient produced by a contaminating safety net; and a decoding sweep in which determinism disguised one game as ten. The fourth is the one worth remembering — it confirmed a prediction registered in advance, which is the hardest case to catch, because a confirmed prediction feels like evidence rather than something to audit.

The corrections shared a shape. In each case the fix was pairing — ensuring two things being compared were measured on the same positions, in the same conditions, with an error bar attached. Every effect that survived pairing held up. Every effect that did not, vanished.

---

# Possible Future Work

- **Stronger teachers.** Refiltering to 2200+ Elo would improve both the imitation target and the value head's label noise, which are the two binding constraints identified. The single highest-value untested change.
- **Engine-labeled value head.** Would remove label noise entirely, at the cost of distilling Stockfish rather than learning from humans — a different project, deliberately not pursued here.
- **Wider candidate sets.** Sampling at temperature 0.8 from a confident policy typically yielded 2–5 distinct candidates out of 30+ legal moves. Better tactical moves may simply never enter the search.
- **Nonlinear probes.** A linear probe recovering 54% of headroom is a lower bound on what the representation contains.
- **Commentary layer.** Optional, and independent of chess strength.
