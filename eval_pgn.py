from unsloth import FastLanguageModel
import chess
import random
import re
import sys
import statistics

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "run_pgn_50000_r16_e1_b2_model"
NUM_GAMES = 20
MAX_PLIES = 100

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH, max_seq_length = 1024, load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

MOVE_NUM = re.compile(r"^\d+\.+$")

def build_prompt(sans):
    parts = []
    for i, san in enumerate(sans):
        if i % 2 == 0:
            parts.append(f"{i // 2 + 1}.")
        parts.append(san)
    if len(sans) % 2 == 0:              # white to move — supply the move number
        parts.append(f"{len(sans) // 2 + 1}.")
    return " ".join(parts)

def next_move_text(sans):
    prompt = build_prompt(sans)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=8, do_sample=True, temperature=0.7)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    completion = text[len(prompt):]
    for tok in completion.split():
        if MOVE_NUM.match(tok):          # model re-emitted the move number
            continue
        return tok
    return None

def phase(move_number):
    if move_number <= 10:
        return "opening (1-10)"
    if move_number <= 30:
        return "middle (11-30)"
    return "endgame (31+)"

buckets = {"opening (1-10)": [0, 0], "middle (11-30)": [0, 0], "endgame (31+)": [0, 0]}
game_rates = []

for g in range(NUM_GAMES):
    board = chess.Board()
    sans = []
    legal = total = 0

    for ply in range(MAX_PLIES):
        if board.is_game_over():
            break

        move_number = ply // 2 + 1
        b = buckets[phase(move_number)]

        san = next_move_text(sans)
        ok = False
        if san is not None:
            try:
                mv = board.parse_san(san)
                ok = True
            except Exception:
                ok = False

        total += 1
        b[1] += 1
        if ok:
            legal += 1
            b[0] += 1
        else:
            mv = random.choice(list(board.legal_moves))   # keep the game going

        sans.append(board.san(mv))
        board.push(mv)

    rate = legal / total if total else 0
    game_rates.append(rate)
    print(f"Game {g+1}: {rate:.1%} legal ({legal}/{total})")

print(f"\nMean legal-move rate: {statistics.mean(game_rates):.1%}")
print(f"Std dev: {statistics.stdev(game_rates):.1%}")
print("\nBy phase:")
for k, (ok, n) in buckets.items():
    if n:
        print(f"  {k}: {ok/n:.1%} legal ({ok}/{n})")