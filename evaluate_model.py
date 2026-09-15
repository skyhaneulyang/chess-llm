from unsloth import FastLanguageModel
import chess
import random
import sys
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "run_30000_r16_e2_model"
NUM_TRIALS = 20
MAX_MOVES_PER_TRIAL = 30

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    max_seq_length = 4096,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

def get_model_move(board):
    fen = board.fen()
    prompt = f"### Instruction:\nBoard (FEN): {fen}\nWhat's your move?\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=10)
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    move_text = full_text.split("### Response:")[-1].strip().split("\n")[0].split()[0]
    return move_text

def run_trial(random_opening_moves=0):
    board = chess.Board()

    # Optionally randomize the starting position slightly so we're not
    # testing the exact same opening sequence every single trial
    for _ in range(random_opening_moves):
        legal = list(board.legal_moves)
        if not legal:
            break
        board.push(random.choice(legal))

    legal_streak = 0
    for _ in range(MAX_MOVES_PER_TRIAL):
        move_text = get_model_move(board)
        try:
            move = board.parse_san(move_text)
            board.push(move)
            legal_streak += 1
        except Exception:
            break
    return legal_streak

def main():
    results = []
    for i in range(NUM_TRIALS):
        # vary starting point: some trials start from move 0, others a few random moves in
        random_start = random.choice([0, 0, 2, 4])
        streak = run_trial(random_opening_moves=random_start)
        results.append(streak)
        print(f"Trial {i+1}: legal streak = {streak} (random_start={random_start})")

    avg = sum(results) / len(results)
    print(f"\nAverage legal-move streak across {NUM_TRIALS} trials: {avg:.2f}")
    print(f"Min: {min(results)}, Max: {max(results)}")

if __name__ == "__main__":
    main()