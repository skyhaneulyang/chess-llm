from unsloth import FastLanguageModel
import chess
import random
import sys
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "run_30000_r16_e2_model" # points at your most recent trained model
MAX_RETRIES = 3  # how many times to ask the model again before falling back

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    max_seq_length = 4096,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

def get_model_move_candidate(board):
    fen = board.fen()
    prompt = f"### Instruction:\nBoard (FEN): {fen}\nWhat's your move?\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=0.7)
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    move_text = full_text.split("### Response:")[-1].strip().split("\n")[0].split()[0]
    return move_text

def get_legal_move(board):
    """Ask the model for a move, retrying on illegal output.
    Falls back to a random legal move if the model keeps failing."""
    for attempt in range(MAX_RETRIES):
        move_text = get_model_move_candidate(board)
        try:
            move = board.parse_san(move_text)
            return move, move_text, attempt + 1  # legal — return it
        except Exception:
            continue  # illegal — try asking again

    # Fallback: model failed every attempt, pick randomly from actually legal moves
    fallback_move = random.choice(list(board.legal_moves))
    return fallback_move, board.san(fallback_move), None  # None marks a fallback was used

def play_game(max_moves=100):
    board = chess.Board()
    fallback_count = 0
    total_moves = 0

    for i in range(max_moves):
        if board.is_game_over():
            print(f"Game over: {board.result()}")
            break

        move, move_text, attempts_used = get_legal_move(board)
        board.push(move)
        total_moves += 1

        if attempts_used is None:
            fallback_count += 1
            
    return fallback_count / total_moves if total_moves else 0

if __name__ == "__main__":
    import statistics
    rates = []
    for g in range(10):
        rate = play_game()
        rates.append(rate)
        print(f"Game {g+1}: {rate:.1%} fallback rate")
    print(f"\nMean fallback rate: {statistics.mean(rates):.1%}")
    print(f"Std dev: {statistics.stdev(rates):.1%}")