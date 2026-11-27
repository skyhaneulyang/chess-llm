from transformers import logging as hf_logging
import warnings
import chess
import chess.pgn
import chess.engine
import statistics
import csv

csv_file = open("validation_results.csv", "w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)
candidate_file = open(
    "candidate_results.csv", "w",
    newline="", encoding="utf-8"
)

candidate_writer = csv.writer(candidate_file)

candidate_writer.writerow([
    "game_id",
    "ply",
    "fen",
    "candidate",
    "judge_score",
    "stockfish_cp",
    "selected"
])

writer.writerow([
    "game_id",
    "ply",
    "fen",
    "greedy_move",
    "search_move",
    "greedy_legal",
    "search_legal",
    "greedy_cp_loss",
    "search_cp_loss",
    "human_cp_loss"
])

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")


import torch
import random

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

from search_play import search_move, policy_move, build_prompt  # loads model + head

PGN_FILE = "games_test_oct2026.pgn"
NUM_GAMES = 10
MAX_PLIES = 60
DEPTH = 10
BLUNDER = 300

engine = chess.engine.SimpleEngine.popen_uci("stockfish.exe")
engine.configure({"Threads": 1})


score_cache = {}

def score_after(b, move):
    key = (b.fen(), move.uci())

    if key in score_cache:
        return score_cache[key]

    b.push(move)
    try:
        info = engine.analyse(b, chess.engine.Limit(depth=DEPTH))
        cp = info["score"].pov(not b.turn).score(mate_score=10000)
    finally:
        b.pop()

    score_cache[key] = cp
    return cp


def phase(n):
    if n <= 10:
        return "opening (1-10)"
    if n <= 30:
        return "middle (11-30)"
    return "endgame (31+)"


# paired samples: only positions where BOTH methods produced a legal move
greedy_losses, search_losses, human_losses, phases = [], [], [], []

greedy_illegal = search_illegal = total = 0

with open(PGN_FILE, encoding="utf-8") as f:
    for g in range(NUM_GAMES):
        game = chess.pgn.read_game(f)
        if game is None:
            break

        b = game.board()
        sans = []

        for ply, human_move in enumerate(game.mainline_moves()):
            if ply >= MAX_PLIES or b.is_game_over():
                break

            best = engine.play(b, chess.engine.Limit(depth=DEPTH)).move
            best_score = score_after(b, best)
            h_loss = max(best_score - score_after(b, human_move), 0)

            # method 1: single greedy policy move
            san = policy_move(sans)
            g_mv = None
            if san:
                try:
                    g_mv = b.parse_san(san)
                except Exception:
                    g_mv = None

            # method 2: candidate generation + value-head search
            s_mv, _, candidate_details = search_move(
                b, sans, return_details=True
            )
            for uci, judge_score in candidate_details:
                mv = chess.Move.from_uci(uci)
                sf_score = score_after(b, mv)
                candidate_writer.writerow([
                g + 1,
                ply + 1,
                b.fen(),
                uci,
                judge_score,
                sf_score,
                mv == s_mv
            ])
            candidate_file.flush()

            print(
                f"Candidate: {uci}, "
                f"Judge: {judge_score:.6f}, "
                f"Stockfish: {sf_score}"
            )
            if g_mv is not None and s_mv is not None:
                print(f"Greedy: {g_mv.uci()}, SF: {score_after(b, g_mv)}")
                print(f"Search: {s_mv.uci()}, SF: {score_after(b, s_mv)}")
                print("-" * 40)

            total += 1
            if g_mv is None:
                greedy_illegal += 1
            if s_mv is None:
                search_illegal += 1

            if g_mv is not None and s_mv is not None:
                greedy_losses.append(max(best_score - score_after(b, g_mv), 0))
                search_losses.append(max(best_score - score_after(b, s_mv), 0))
                human_losses.append(h_loss)
                phases.append(phase(ply // 2 + 1))
            writer.writerow([
                g + 1,
                ply + 1,
                b.fen(),
                g_mv.uci() if g_mv is not None else "",
                s_mv.uci() if s_mv is not None else "",
                g_mv is not None,
                s_mv is not None,
                max(best_score - score_after(b, g_mv), 0)
                    if g_mv is not None else "",
                max(best_score - score_after(b, s_mv), 0)
                    if s_mv is not None else "",
                h_loss
            ])
            csv_file.flush()
            sans.append(b.san(human_move))
            b.push(human_move)

        print(f"Game {g + 1}/{NUM_GAMES} done ({len(greedy_losses)} paired positions)")

engine.quit()

n = len(greedy_losses)


def summary(name, losses):
    blunders = sum(1 for x in losses if x >= BLUNDER)
    print(f"  {name:<10}{statistics.mean(losses):>8.0f}"
          f"{statistics.median(losses):>9.0f}{blunders / n:>10.1%}")


print(f"\nLegality over all {total} positions")
print(f"  Greedy illegal: {greedy_illegal}/{total} = {greedy_illegal / total:.1%}")
print(f"  Search illegal: {search_illegal}/{total} = {search_illegal / total:.1%}")

print(f"\nPaired on {n} positions where BOTH played legally")
print(f"  {'':<10}{'mean':>8}{'median':>9}{'blunders':>11}")
summary("Greedy", greedy_losses)
summary("Search", search_losses)
summary("Human", human_losses)

diffs = [s - g for s, g in zip(search_losses, greedy_losses)]
sd = statistics.stdev(diffs) if n > 1 else 0
print()
print(f"  Search - Greedy: {statistics.mean(diffs):+.0f} cp   SE {sd / n ** 0.5:.0f} cp")
print(f"  Search better on: {sum(1 for d in diffs if d < 0)}/{n}")

print("\nBy phase (mean cp loss):")
for k in ["opening (1-10)", "middle (11-30)", "endgame (31+)"]:
    idx = [i for i, p in enumerate(phases) if p == k]
    if idx:
        print(f"  {k}: greedy {statistics.mean([greedy_losses[i] for i in idx]):.0f}, "
              f"search {statistics.mean([search_losses[i] for i in idx]):.0f}, "
              f"human {statistics.mean([human_losses[i] for i in idx]):.0f} (n={len(idx)})")