from unsloth import FastLanguageModel
import chess
import chess.engine
import random
import statistics
import sys

random.seed(1337)

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "run_20000_r16_e2_model"
STOCKFISH_PATH = "stockfish.exe"
NUM_POSITIONS = 300
DEPTH = 10

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

model_losses, random_losses, illegal = [], [], 0

for i in range(NUM_POSITIONS):
    # random position a few moves into a game
    board = chess.Board()
    for _ in range(random.randint(4, 30)):
        if board.is_game_over(): break
        board.push(random.choice(list(board.legal_moves)))
    if board.is_game_over(): continue

    best = engine.play(board, chess.engine.Limit(depth=DEPTH)).move
    best_score = score(board, best)

    mv = model_move(board)
    if mv is None:
        illegal += 1
    else:
        model_losses.append(best_score - score(board, mv))

    rnd = random.choice(list(board.legal_moves))
    random_losses.append(best_score - score(board, rnd))

    if (i + 1) % 20 == 0:
        print(f"{i+1} positions done...")

engine.quit()
print(f"\nModel:  {statistics.mean(model_losses):.0f} cp loss (n={len(model_losses)})")
print(f"Random: {statistics.mean(random_losses):.0f} cp loss (n={len(random_losses)})")
print(f"Illegal outputs: {illegal}/{NUM_POSITIONS}")