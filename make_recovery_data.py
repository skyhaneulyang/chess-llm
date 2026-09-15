import chess
import chess.pgn
import chess.engine
import json
import random

STOCKFISH_PATH = "stockfish.exe"   # adjust if it lives elsewhere
INPUT_FILE = "games_sample.pgn"
OUTPUT_FILE = "recovery_dataset.jsonl"

TARGET_EXAMPLES = 10000
PERTURBATIONS_PER_GAME = 3
DEPTH = 8          # engine search depth — higher = better labels, slower

def perturbed_positions(game):
    """Replay a real game, but at a few random points substitute a random
    legal move instead of the one actually played. Yields the resulting
    off-distribution positions."""
    moves = list(game.mainline_moves())
    if len(moves) < 10:
        return

    for _ in range(PERTURBATIONS_PER_GAME):
        ply = random.randint(2, len(moves) - 2)

        board = game.board()
        for move in moves[:ply]:
            board.push(move)

        # pick a legal move that is NOT what the human actually played
        alternatives = [m for m in board.legal_moves if m != moves[ply]]
        if not alternatives:
            continue
        board.push(random.choice(alternatives))

        if board.is_game_over():
            continue
        yield board

def main():
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as pgn, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        while count < TARGET_EXAMPLES:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            for board in perturbed_positions(game):
                result = engine.play(board, chess.engine.Limit(depth=DEPTH))
                best_san = board.san(result.move)   # SAN must be taken BEFORE pushing

                example = {
                    "instruction": f"Board (FEN): {board.fen()}\nWhat's your move?",
                    "output": best_san
                }
                out.write(json.dumps(example) + "\n")
                count += 1

                if count % 500 == 0:
                    print(f"{count} recovery examples generated...")
                if count >= TARGET_EXAMPLES:
                    break

    engine.quit()
    print(f"Done. Wrote {count} examples to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()