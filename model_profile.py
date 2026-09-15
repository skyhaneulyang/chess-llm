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
NUM_GAMES = 20
MAX_PLIES = 60
DEPTH = 10

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH, max_seq_length=1024, load_in_4bit=True
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


def next_move(sans):
    prompt = build_prompt(sans)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
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


def phase(n):
    if n <= 10:
        return "opening (1-10)"
    if n <= 30:
        return "middle (11-30)"
    return "endgame (31+)"


buckets = {
    "opening (1-10)": [],
    "middle (11-30)": [],
    "endgame (31+)": [],
}
all_losses = []
blunders = 0
illegal = 0
total = 0

random.seed(42)

for g in range(NUM_GAMES):
    board = chess.Board()
    sans = []

    # random short opening so greedy decoding produces distinct games
    for _ in range(random.randint(2, 6)):
        mv = random.choice(list(board.legal_moves))
        sans.append(board.san(mv))
        board.push(mv)

    for ply in range(MAX_PLIES):
        if board.is_game_over():
            break

        best = engine.play(board, chess.engine.Limit(depth=DEPTH)).move
        best_score = score_after(board, best)

        san = next_move(sans)
        mv = None
        if san:
            try:
                mv = board.parse_san(san)
            except Exception:
                mv = None

        total += 1
        if mv is None:
            illegal += 1
            mv = engine.play(board, chess.engine.Limit(depth=1)).move
        else:
            loss = max(best_score - score_after(board, mv), 0)
            buckets[phase(ply // 2 + 1)].append(loss)
            all_losses.append(loss)
            if loss >= 300:
                blunders += 1

        sans.append(board.san(mv))
        board.push(mv)

    print(f"Game {g + 1}/{NUM_GAMES} done")

engine.quit()

print(f"\nModel (greedy), {len(all_losses)} legal moves scored")
print(f"  Mean cp loss:   {statistics.mean(all_losses):.0f}")
print(f"  Median cp loss: {statistics.median(all_losses):.0f}")
print(f"  Blunders (300+ cp): {blunders}/{len(all_losses)} = {blunders / len(all_losses):.1%}")
print(f"  Illegal: {illegal}/{total} = {illegal / total:.1%}")
print("\nBy phase:")
for k, v in buckets.items():
    if v:
        print(f"  {k}: mean {statistics.mean(v):.0f} cp, median {statistics.median(v):.0f} cp (n={len(v)})")