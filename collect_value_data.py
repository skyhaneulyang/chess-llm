from unsloth import FastLanguageModel
from transformers import logging as hf_logging
import warnings
import chess
import chess.pgn
import numpy as np
import torch
import sys

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "run_pgn_99186_r16_e1_b2_model"
PGN_FILE = "games_large.pgn"
OUT_X = "value_x.npy"
OUT_Y = "value_y.npy"

NUM_GAMES = 10000
MIN_PLY = 6
MAX_PLY = 70
STRIDE = 4
LAYER = 31

RESULT_TO_WHITE_SCORE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH, max_seq_length=1024, load_in_4bit=True
)
FastLanguageModel.for_inference(model)


def build_prompt(sans):
    parts = []
    for i, san in enumerate(sans):
        if i % 2 == 0:
            parts.append(f"{i // 2 + 1}.")
        parts.append(san)
    if len(sans) % 2 == 0:
        parts.append(f"{len(sans) // 2 + 1}.")
    return " ".join(parts)


def final_hidden(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return out.hidden_states[LAYER][0, -1, :].float().cpu().numpy().astype(np.float16)


xs = []
ys = []

with open(PGN_FILE, encoding="utf-8") as f:
    for g in range(NUM_GAMES):
        game = chess.pgn.read_game(f)
        if game is None:
            break

        result = game.headers.get("Result", "*")
        if result not in RESULT_TO_WHITE_SCORE:
            continue
        white_score = RESULT_TO_WHITE_SCORE[result]

        board = game.board()
        sans = []

        for ply, move in enumerate(game.mainline_moves()):
            if ply >= MAX_PLY:
                break

            if ply >= MIN_PLY and ply % STRIDE == 0:
                # value from the perspective of the side to move
                target = white_score if board.turn == chess.WHITE else 1.0 - white_score
                xs.append(final_hidden(build_prompt(sans)))
                ys.append(target)

            sans.append(board.san(move))
            board.push(move)

        if (g + 1) % 100 == 0:
            print(f"{g + 1} games, {len(xs)} positions")

X = np.stack(xs)
Y = np.array(ys, dtype=np.float32)
np.save(OUT_X, X)
np.save(OUT_Y, Y)
print(f"\nSaved {X.shape[0]} positions. X={X.shape} {X.dtype}, Y={Y.shape}")
print(f"Target mean {Y.mean():.3f} (0.5 = balanced)")