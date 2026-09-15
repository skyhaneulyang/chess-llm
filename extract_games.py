import requests
import zstandard as zstd
import io
import chess.pgn

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2026-06.pgn.zst"
OUTPUT_FILE = "games_large.pgn"
MAX_GAMES = 100000
MIN_RATING = 1500

def stream_pgn_from_url(url):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(response.raw) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")
        yield from iter_games(text_stream)

def iter_games(text_stream):
    while True:
        game = chess.pgn.read_game(text_stream)
        if game is None:
            break
        yield game

def passes_filters(game):
    headers = game.headers
    try:
        white_elo = int(headers.get("WhiteElo", 0))
        black_elo = int(headers.get("BlackElo", 0))
    except ValueError:
        return False
    return white_elo >= MIN_RATING and black_elo >= MIN_RATING

def main():
    kept = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for game in stream_pgn_from_url(URL):
            if passes_filters(game):
                print(game, file=out, end="\n\n")
                kept += 1
                if kept % 5000 == 0:
                    print(f"Kept {kept} games so far...")
            if kept >= MAX_GAMES:
                break
    print(f"Done. Saved {kept} games to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()