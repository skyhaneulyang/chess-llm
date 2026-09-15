from unsloth import FastLanguageModel
import chess
import random
import re
import sys
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci("stockfish.exe")

def weak_move(board):
    return engine.play(board, chess.engine.Limit(depth=1)).move

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "run_pgn_50000_r16_e1_b2_model"
MAX_RETRIES = 5

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH, max_seq_length = 1024, load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

MOVE_NUM = re.compile(r"^\d+\.+$")



def show(board, last_move=None):
    print()
    for rank in range(7, -1, -1):
        row = [f"{rank + 1} "]
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            ch = piece.symbol() if piece else "."
            if last_move and sq in (last_move.from_square, last_move.to_square):
                row.append(f"[{ch}]")
            else:
                row.append(f" {ch} ")
        print("".join(row))
    print("   a  b  c  d  e  f  g  h")
    if board.is_check():
        print("  CHECK")
    print()

def build_prompt(sans):
    parts = []
    for i, san in enumerate(sans):
        if i % 2 == 0:
            parts.append(f"{i // 2 + 1}.")
        parts.append(san)
    if len(sans) % 2 == 0:
        parts.append(f"{len(sans) // 2 + 1}.")
    return " ".join(parts)


def model_move(board, sans):
    """Greedy first — highest legality. Sample on retries so a failed
    attempt can produce something different."""
    prompt = build_prompt(sans)
    for attempt in range(MAX_RETRIES):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        if attempt == 0:
            kw = {"do_sample": False}
        else:
            kw = {"do_sample": True, "temperature": 0.7}
        out = model.generate(**inputs, max_new_tokens=8, **kw)
        completion = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):]
        for tok in completion.split():
            if MOVE_NUM.match(tok):
                continue
            try:
                return board.parse_san(tok), attempt + 1
            except Exception:
                pass
            break
    return weak_move(board), None


def human_move(board):
    while True:
        raw = input("Your move (SAN, e.g. e4 / Nf3 / O-O, or 'quit'): ").strip()
        if raw.lower() in ("quit", "exit", "resign"):
            return None
        if raw.lower() == "moves":
            print(" ".join(sorted(board.san(m) for m in board.legal_moves)))
            continue
        try:
            return board.parse_san(raw)
        except Exception:
            print("  Not a legal move here. Type 'moves' to list them.")


def main():
    side = ""
    while side not in ("w", "b"):
        side = input("Play as white or black? (w/b): ").strip().lower()
    human_is_white = side == "w"

    board = chess.Board()
    sans = []
    fallbacks = 0
    last = None

    show(board)

    while not board.is_game_over():
        if board.turn == chess.WHITE and human_is_white or \
           board.turn == chess.BLACK and not human_is_white:
            mv = human_move(board)
            if mv is None:
                print("Resigned.")
                return
        else:
            mv, attempts = model_move(board, sans)
            if attempts is None:
                fallbacks += 1
                print(f"Model: {board.san(mv)}   [fallback — no legal move in {MAX_RETRIES} tries]")
            else:
                print(f"Model: {board.san(mv)}   (attempt {attempts})")

        sans.append(board.san(mv))
        board.push(mv)
        last = mv
        show(board, last)
        
    engine.quit()
    print(f"Game over: {board.result()}  ({board.outcome().termination.name})")
    print(f"Model fallbacks used: {fallbacks}/{len(sans) // 2}")


if __name__ == "__main__":
    main()