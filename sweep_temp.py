from unsloth import FastLanguageModel
from transformers import logging as hf_logging
import warnings
import chess
import chess.engine
import random
import re
import statistics
import sys

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "run_pgn_50000_r16_e1_b2_model"
SETTINGS = [("greedy", None), ("t0.3", 0.3), ("t0.5", 0.5), ("t0.7", 0.7)]
NUM_GAMES = 10
MAX_PLIES = 60
DEPTH = 10

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH, max_seq_length=1024, load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
engine = chess.engine.SimpleEngine.popen_uci("stockfish.exe")

MOVE_NUM = re.compile(r"^\d+\.+$")


def build_prompt(sans):
    parts = []
    for i, san in enumerate(sans):
        if i % 2 == 0:
            parts.append(f"{i // 2 + 1}.")
        parts.append(san)
    if len(sans) % 2 == 0:
        parts.append(f"{len(sans) // 2 + 1}.")
    return " ".join(parts)


def next_move(sans, temp):
    prompt = build_prompt(sans)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    kw = {"do_sample": False} if temp is None else {"do_sample": True, "temperature": temp}
    out = model.generate(**inputs, max_new_tokens=8, **kw)
    comp = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):]
    for tok in comp.split():
        if MOVE_NUM.match(tok):
            continue
        return tok
    return None


def score_after(board, move):
    board.push(move)
    info = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
    cp = info["score"].pov(not board.turn).score(mate_score=10000)
    board.pop()
    return cp


def weak(board):
    return engine.play(board, chess.engine.Limit(depth=1)).move


results = []

for label, temp in SETTINGS:
    random.seed(42)
    losses, legal, total = [], 0, 0

    for g in range(NUM_GAMES):
        board, sans = chess.Board(), []
        for _ in range(random.randint(2, 6)):        # random book-ish opening plies
            mv = random.choice(list(board.legal_moves))
            sans.append(board.san(mv))
            board.push(mv)
        for ply in range(MAX_PLIES):
            if board.is_game_over():
                break

            best = engine.play(board, chess.engine.Limit(depth=DEPTH)).move
            best_score = score_after(board, best)

            san = next_move(sans, temp)
            mv = None
            if san:
                try:
                    mv = board.parse_san(san)
                except Exception:
                    mv = None

            total += 1
            if mv is not None:
                legal += 1
                losses.append(best_score - score_after(board, mv))
            else:
                mv = weak(board)

            sans.append(board.san(mv))
            board.push(mv)

        print(f"  {label} game {g + 1}/{NUM_GAMES} done")

    line = (f"{label:>7}:  {statistics.mean(losses):>4.0f} cp loss   "
            f"{legal / total:>5.1%} legal   (n={len(losses)})")
    print(line)
    results.append(line)

engine.quit()

with open("temp_sweep_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results) + "\n")

print("\n" + "\n".join(results))