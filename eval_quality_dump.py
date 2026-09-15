from unsloth import FastLanguageModel
import chess
import chess.engine
import random
import json
import sys

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "chess_lora_model_20k_r16"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42

STOCKFISH_PATH = "stockfish.exe"
NUM_POSITIONS = 300
DEPTH = 10

OUT = MODEL_PATH.rstrip("/\\") + f"_quality_seed{SEED}.jsonl"

random.seed(SEED)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH, max_seq_length = 4096, load_in_4bit = True,
)
FastLanguageModel.for_inference(model)
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)


def model_move(board):
    prompt = f"### Instruction:\nBoard (FEN): {board.fen()}\nWhat's your move?\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=0.7)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    san = text.split("### Response:")[-1].strip().split("\n")[0].split()[0]
    try:
        return board.parse_san(san)
    except Exception:
        return None


def score(board, move):
    """Evaluation after `move`, from the perspective of the side that just moved."""
    board.push(move)
    info = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
    cp = info["score"].pov(not board.turn).score(mate_score=10000)
    board.pop()
    return cp


records = []

for i in range(NUM_POSITIONS):
    # random position a few moves into a game
    board = chess.Board()
    for _ in range(random.randint(4, 30)):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))
    if board.is_game_over():
        continue

    best = engine.play(board, chess.engine.Limit(depth=DEPTH)).move
    best_score = score(board, best)

    mv = model_move(board)
    model_loss = None if mv is None else best_score - score(board, mv)

    rnd = random.choice(list(board.legal_moves))
    random_loss = best_score - score(board, rnd)

    records.append({
        "fen": board.fen(),
        "model_loss": model_loss,
        "random_loss": random_loss,
    })

    if (i + 1) % 20 == 0:
        print(f"{i+1} positions done...")

engine.quit()

with open(OUT, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

legal = [r["model_loss"] for r in records if r["model_loss"] is not None]
print(f"\nWrote {len(records)} positions to {OUT}")
print(f"Legal: {len(legal)}/{len(records)}")