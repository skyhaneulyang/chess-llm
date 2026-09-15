from unsloth import FastLanguageModel
import chess

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "chess_lora_model",
    max_seq_length = 4096,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

board = chess.Board()  # starts at the standard opening position

for i in range(10):
    fen = board.fen()
    prompt = f"### Instruction:\nBoard (FEN): {fen}\nWhat's your move?\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=10)
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    move_text = full_text.split("### Response:")[-1].strip().split("\n")[0].split()[0]
    print(f"Move {i+1} — model said: '{move_text}'")

    try:
        move = board.parse_san(move_text)
        board.push(move)
        print(f"  -> Legal! Board updated.\n")
    except Exception as e:
        print(f"  -> ILLEGAL move or parse error: {e}\n")
        break