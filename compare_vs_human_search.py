from transformers import logging as hf_logging
import warnings
import chess
import chess.pgn
import chess.engine
import statistics

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

from search_play import search_move

PGN_FILE = "games_large.pgn"
NUM_GAMES = 10
MAX_PLIES = 60
DEPTH = 10
BLUNDER = 300

engine = chess.engine.SimpleEngine.popen_uci("stockfish.exe")


def score_after(b, move):
    b.push(move)
    info = engine.analyse(b, chess.engine.Limit(depth=DEPTH))
    cp = info["score"].pov(not b.turn).score(mate_score=10000)
    b.pop()
    return cp


def phase(n):
    if n <= 10:
        return "opening (1-10)"
    if n <= 30:
        return "middle (11-30)"
    return "endgame (31+)"


model_losses, human_losses = [], []
buckets = {"opening (1-10)": ([], []), "middle (11-30)": ([], []), "endgame (31+)": ([], [])}
model_blunders = human_blunders = illegal = total = 0

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

            mv, _ = search_move(b, sans)

            total += 1
            if mv is None:
                illegal += 1
            else:
                m_loss = max(best_score - score_after(b, mv), 0)
                model_losses.append(m_loss)
                human_losses.append(h_loss)
                mb, hb = buckets[phase(ply // 2 + 1)]
                mb.append(m_loss)
                hb.append(h_loss)
                if m_loss >= BLUNDER:
                    model_blunders += 1
                if h_loss >= BLUNDER:
                    human_blunders += 1

            sans.append(b.san(human_move))
            b.push(human_move)

        print(f"Game {g + 1}/{NUM_GAMES} done")

engine.quit()

n = len(model_losses)
diffs = [m - h for m, h in zip(model_losses, human_losses)]
sd = statistics.stdev(diffs) if n > 1 else 0

print(f"\nPaired on {n} identical positions (search, {NUM_GAMES} games)")
print(f"  No candidate found: {illegal}/{total} = {illegal / total:.1%}")
print()
print(f"  {'':<10}{'mean':>8}{'median':>9}{'blunders':>11}")
print(f"  {'Model':<10}{statistics.mean(model_losses):>8.0f}"
      f"{statistics.median(model_losses):>9.0f}{model_blunders / n:>10.1%}")
print(f"  {'Human':<10}{statistics.mean(human_losses):>8.0f}"
      f"{statistics.median(human_losses):>9.0f}{human_blunders / n:>10.1%}")
print()
print(f"  Mean difference: {statistics.mean(diffs):.0f} cp   SE {sd / n ** 0.5:.0f} cp")
print("\nBy phase (mean cp loss):")
for k, (mb, hb) in buckets.items():
    if mb:
        print(f"  {k}: model {statistics.mean(mb):.0f}, human {statistics.mean(hb):.0f} (n={len(mb)})")