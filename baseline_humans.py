from transformers import logging as hf_logging
import warnings
import chess
import chess.pgn
import chess.engine
import statistics

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

PGN_FILE = "games_large.pgn"
NUM_GAMES = 20
MAX_PLIES = 60
DEPTH = 10

engine = chess.engine.SimpleEngine.popen_uci("stockfish.exe")


def score_after(board, move):
    board.push(move)
    info = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
    cp = info["score"].pov(not board.turn).score(mate_score=10000)
    board.pop()
    return cp


def phase(move_number):
    if move_number <= 10:
        return "opening (1-10)"
    if move_number <= 30:
        return "middle (11-30)"
    return "endgame (31+)"


buckets = {"opening (1-10)": [], "middle (11-30)": [], "endgame (31+)": []}
all_losses = []
blunders = 0          # moves losing 300+ cp

with open(PGN_FILE, encoding="utf-8") as f:
    for g in range(NUM_GAMES):
        game = chess.pgn.read_game(f)
        if game is None:
            break

        board = game.board()
        for ply, move in enumerate(game.mainline_moves()):
            if ply >= MAX_PLIES or board.is_game_over():
                break

            best = engine.play(board, chess.engine.Limit(depth=DEPTH)).move
            loss = score_after(board, best) - score_after(board, move)
            loss = max(loss, 0)

            buckets[phase(ply // 2 + 1)].append(loss)
            all_losses.append(loss)
            if loss >= 300:
                blunders += 1

            board.push(move)

        print(f"Game {g + 1}/{NUM_GAMES} done")

engine.quit()

print(f"\nHuman baseline (1500+ Elo), {len(all_losses)} moves")
print(f"  Mean cp loss:   {statistics.mean(all_losses):.0f}")
print(f"  Median cp loss: {statistics.median(all_losses):.0f}")
print(f"  Blunders (300+ cp): {blunders}/{len(all_losses)} = {blunders/len(all_losses):.1%}")
print("\nBy phase:")
for k, v in buckets.items():
    if v:
        print(f"  {k}: mean {statistics.mean(v):.0f} cp, median {statistics.median(v):.0f} cp (n={len(v)})")