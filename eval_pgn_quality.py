from unsloth import FastLanguageModel
import chess
import chess.engine
import random
import re
import json
import statistics
import sys

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "run_pgn_50000_r16_e1_b2_model"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
STOCKFISH_PATH = "stockfish.exe"
NUM_GAMES = 20
MAX_PLIES = 60
DEPTH = 10

random.seed(SEED)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH, max_seq_length = 1024, load_in_4bit = True,
)
FastLanguageModel.for_inference(model)
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

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


def next_move_text(sans):
    prompt = build_prompt(sans)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=8, do_sample=True, temperature=0.7)
    completion = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):]
    for tok in completion.split():
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


pairs = []

for g in range(NUM_GAMES):
    board = chess.Board()
    sans = []

    for ply in range(MAX_PLIES):
        if board.is_game_over():
            break

        best = engine.play(board, chess.engine.Limit(depth=DEPTH)).move
        best_score = score_after(board, best)

        san = next_move_text(sans)
        mv = None
        if san is not None:
            try:
                mv = board.parse_san(san)
            except Exception:
                mv = None

        rnd = random.choice(list(board.legal_moves))

        r_loss = best_score - score_after(board, rnd)
        m_loss = None if mv is None else best_score - score_after(board, mv)

        pairs.append({
            "fen": board.fen(),
            "model_loss": m_loss,
            "random_loss": r_loss,
        })

        played = mv if mv is not None else rnd
        sans.append(board.san(played))
        board.push(played)

    print(f"Game {g+1} done ({len(pairs)} positions so far)")

engine.quit()

OUT = f"{MODEL_PATH}_pgnquality_seed{SEED}.jsonl"
with open(OUT, "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")

both = [p for p in pairs if p["model_loss"] is not None]
diffs = [p["random_loss"] - p["model_loss"] for p in both]
sd = statistics.stdev(diffs) if len(diffs) > 1 else 0

print(f"\nWrote {len(pairs)} positions to {OUT}")
print(f"\nPaired on {len(both)} positions:")
print(f"  Model:  {statistics.mean(p['model_loss'] for p in both):.0f} cp")
print(f"  Random: {statistics.mean(p['random_loss'] for p in both):.0f} cp")
print(f"  Mean difference: {statistics.mean(diffs):.0f} cp")
print(f"  Std error:       {sd / len(diffs) ** 0.5:.0f} cp")
print(f"  Model better on: {sum(1 for d in diffs if d > 0)}/{len(diffs)}")