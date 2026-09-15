import chess.pgn
import json

INPUT_FILE = "games_large.pgn"
OUTPUT_FILE = "pgn_dataset.jsonl"
MIN_MOVES = 10

def game_to_text(game):
    board = game.board()
    parts = []
    n = 0
    for i, move in enumerate(game.mainline_moves()):
        if i % 2 == 0:
            parts.append(f"{i // 2 + 1}.")
        parts.append(board.san(move))
        board.push(move)
        n = i + 1
    return " ".join(parts), n

def main():
    count = 0
    with open(INPUT_FILE, encoding="utf-8") as pgn, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            try:
                text, n = game_to_text(game)
            except Exception:
                continue
            if n < MIN_MOVES:
                continue
            out.write(json.dumps({"text": text}) + "\n")
            count += 1
            if count % 5000 == 0:
                print(f"{count} games converted...")
    print(f"Done. Wrote {count} games to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()