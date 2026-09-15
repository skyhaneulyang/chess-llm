from unsloth import FastLanguageModel
from transformers import logging as hf_logging
import warnings
import chess
import chess.pgn
import chess.engine
import re
import statistics
import sys

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "run_pgn_99186_r16_e1_b2_model"
PGN_FILE = "games_large.pgn"
NUM_GAMES = 10
MAX_PLIES = 60
DEPTH = 10
BLUNDER = 300

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH, max_seq_length=1024, load_in_4bit=True
)
FastLanguageModel.for_inference(model)
engine = chess.engine.SimpleEngine.popen_uci("stockfish.exe")

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


def next_move(sans):
    prompt = build_prompt(sans)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    comp = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):]
    for tok in comp.split():
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


def phase(n):
    if n <= 10:
        return "opening (1-10)"
    if n <= 30:
        return "middle (11-30)"
    return "endgame (31+)"


model_losses = []
human_losses = []
buckets = {
    "opening (1-10)": ([], []),
    "middle (11-30)": ([], []),
    "endgame (31+)": ([], []),
}
model_blunders = 0
human_blunders = 0
illegal = 0
total = 0

with open(PGN_FILE, encoding="utf-8") as f:
    for g in range(NUM_GAMES):
        game = chess.pgn.read_game(f)
        if game is None:
            break

        board = game.board()
        sans = []

        for ply, human_move in enumerate(game.mainline_moves()):
            if ply >= MAX_PLIES or board.is_game_over():
                break

            best = engine.play(board, chess.engine.Limit(depth=DEPTH)).move
            best_score = score_after(board, best)
            h_loss = max(best_score - score_after(board, human_move), 0)

            san = next_move(sans)
            mv = None
            if san:
                try:
                    mv = board.parse_san(san)
                except Exception:
                    mv = None

            total += 1
            if mv is None:
                illegal += 1
            else:
                m_loss = max(best_score - score_after(board, mv), 0)
                model_losses.append(m_loss)
                human_losses.append(h_loss)
                mb, hb = buckets[phase(ply // 2 + 1)]
                mb.append(m_loss)
                hb.append(h_loss)
                if m_loss >= BLUNDER:
                    model_blunders += 1
                if h_loss >= BLUNDER:
                    human_blunders += 1

            sans.append(board.san(human_move))
            board.push(human_move)

        print(f"Game {g + 1}/{NUM_GAMES} done")

engine.quit()

n = len(model_losses)
diffs = [m - h for m, h in zip(model_losses, human_losses)]
sd = statistics.stdev(diffs) if n > 1 else 0

print(f"\nPaired on {n} identical positions (no search, {NUM_GAMES} games)")
print(f"  Illegal: {illegal}/{total} = {illegal / total:.1%}")
print()
print(f"  {'':<10}{'mean':>8}{'median':>9}{'blunders':>11}")
print(f"  {'Model':<10}{statistics.mean(model_losses):>8.0f}"
      f"{statistics.median(model_losses):>9.0f}{model_blunders / n:>10.1%}")
print(f"  {'Human':<10}{statistics.mean(human_losses):>8.0f}"
      f"{statistics.median(human_losses):>9.0f}{human_blunders / n:>10.1%}")
print()
print(f"  Mean difference: {statistics.mean(diffs):.0f} cp   SE {sd / n ** 0.5:.0f} cp")
print("\nBy phase (mean cp loss):")
for k, (mb, hb) in buckets.items():
    if mb:
        print(f"  {k}: model {statistics.mean(mb):.0f}, human {statistics.mean(hb):.0f} (n={len(mb)})")