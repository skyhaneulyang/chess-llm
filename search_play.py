from unsloth import FastLanguageModel
from transformers import logging as hf_logging
import warnings
import chess
import torch
import torch.nn as nn
import re
import sys

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "run_pgn_99186_r16_e1_b2_model"
LAYER = 31
NUM_CANDIDATES = 12
TWO_PLY = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH, max_seq_length=1024, load_in_4bit=True
)
FastLanguageModel.for_inference(model)

HIDDEN = model.config.hidden_size
value_head = nn.Sequential(
    nn.LayerNorm(HIDDEN),
    nn.Linear(HIDDEN, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 1),
    nn.Sigmoid(),
).cuda()
value_head.load_state_dict(torch.load("value_head.pt"))
value_head.eval()

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


def first_move_token(completion):
    for tok in completion.split():
        if MOVE_NUM.match(tok):
            continue
        return tok
    return None


def policy_move(sans):
    prompt = build_prompt(sans)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    comp = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):]
    return first_move_token(comp)


def candidates(board, sans, k=NUM_CANDIDATES):
    """Greedy move plus sampled alternatives. Returns legal, deduped moves."""
    found = {}

    san = policy_move(sans)
    if san:
        try:
            mv = board.parse_san(san)
            found[mv.uci()] = mv
        except Exception:
            pass

    prompt = build_prompt(sans)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=True,
                             temperature=0.8, num_return_sequences=k)
    for seq in out:
        comp = tokenizer.decode(seq, skip_special_tokens=True)[len(prompt):]
        tok = first_move_token(comp)
        if not tok:
            continue
        try:
            mv = board.parse_san(tok)
            found[mv.uci()] = mv
        except Exception:
            continue

    return list(found.values())


def evaluate(sans):
    """Value head output: win probability for the side to move."""
    prompt = build_prompt(sans)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        h = out.hidden_states[LAYER][0, -1, :].float().unsqueeze(0)
        return value_head(h).item()


def score_candidate(board, sans, mv):
    """Score one candidate. Leaves the board exactly as it found it."""
    san = board.san(mv)
    board.push(mv)
    line = sans + [san]

    try:
        if board.is_checkmate():
            return 1.0
        if board.is_game_over():
            return 0.5

        if not TWO_PLY:
            # opponent to move, so their win prob inverts to ours
            return 1.0 - evaluate(line)

        reply_san = policy_move(line)
        reply = None
        if reply_san:
            try:
                reply = board.parse_san(reply_san)
            except Exception:
                reply = None

        if reply is None:
            return 1.0 - evaluate(line)

        line = line + [board.san(reply)]
        board.push(reply)
        try:
            if board.is_checkmate():
                return 0.0          # we get mated after this move
            if board.is_game_over():
                return 0.5
            # our turn again, so the value is already from our perspective
            return evaluate(line)
        finally:
            board.pop()

    finally:
        board.pop()


def search_move(board, sans):
    cands = candidates(board, sans)
    if not cands:
        return None, None

    best_move, best_score = None, -1.0
    for mv in cands:
        score = score_candidate(board, sans, mv)
        if score > best_score:
            best_move, best_score = mv, score

    return best_move, best_score


if __name__ == "__main__":
    board = chess.Board()
    sans = []
    for ply in range(60):
        if board.is_game_over():
            break
        mv, score = search_move(board, sans)
        if mv is None:
            print("no legal candidate found")
            break
        san = board.san(mv)
        print(f"{len(sans) // 2 + 1}. {san}  (value {score:.3f}, {len(candidates(board, sans))} cands)")
        sans.append(san)
        board.push(mv)
    print()
    print(board)
    print(board.result())