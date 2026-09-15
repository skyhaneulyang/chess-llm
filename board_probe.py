from unsloth import FastLanguageModel
from transformers import logging as hf_logging
import warnings
import chess
import chess.pgn
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
import sys

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "run_pgn_99186_r16_e1_b2_model"
PGN_FILE = "games_large.pgn"

NUM_GAMES = 300
MIN_PLY = 6
MAX_PLY = 60
STRIDE = 4
LAYERS = [8, 16, 24, 31]
TRAIN_FRAC = 0.8

PIECE_TO_CLASS = {
    None: 0,
    "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
    "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12,
}

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


def hidden_states(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return [out.hidden_states[L][0, -1, :].float().cpu().numpy() for L in LAYERS]


def board_labels(board):
    labels = np.zeros(64, dtype=np.int64)
    for sq in range(64):
        piece = board.piece_at(sq)
        labels[sq] = PIECE_TO_CLASS[piece.symbol() if piece else None]
    return labels


features = {L: [] for L in LAYERS}
labels = []

with open(PGN_FILE, encoding="utf-8") as f:
    collected = 0
    for g in range(NUM_GAMES):
        game = chess.pgn.read_game(f)
        if game is None:
            break

        board = game.board()
        sans = []

        for ply, move in enumerate(game.mainline_moves()):
            if ply >= MAX_PLY:
                break

            if ply >= MIN_PLY and ply % STRIDE == 0:
                hs = hidden_states(build_prompt(sans))
                for L, h in zip(LAYERS, hs):
                    features[L].append(h)
                labels.append(board_labels(board))
                collected += 1

            sans.append(board.san(move))
            board.push(move)

        if (g + 1) % 25 == 0:
            print(f"{g + 1} games scanned, {collected} positions collected")

labels = np.stack(labels)
n = labels.shape[0]
split = int(n * TRAIN_FRAC)
print(f"\nCollected {n} positions. Train {split}, test {n - split}.")

train_labels, test_labels = labels[:split], labels[split:]
baseline_correct = 0
for sq in range(64):
    counts = np.bincount(train_labels[:, sq], minlength=13)
    majority = counts.argmax()
    baseline_correct += (test_labels[:, sq] == majority).sum()
baseline = baseline_correct / (64 * (n - split))
print(f"Majority-class baseline: {baseline:.1%}\n")

for L in LAYERS:
    X = np.stack(features[L])
    X_train, X_test = X[:split], X[split:]

    correct = 0
    for sq in range(64):
        y_train = train_labels[:, sq]
        y_test = test_labels[:, sq]

        if len(np.unique(y_train)) == 1:
            correct += (y_test == y_train[0]).sum()
            continue

        clf = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
        clf.fit(X_train, y_train)
        correct += (clf.predict(X_test) == y_test).sum()

    acc = correct / (64 * (n - split))
    print(f"Layer {L:>2}: {acc:.1%} square accuracy   (baseline {baseline:.1%}, +{acc - baseline:.1%})")