import json
import random

CLEAN_FILE = "chess_dataset.jsonl"
RECOVERY_FILE = "recovery_dataset.jsonl"
OUTPUT_FILE = "combined_dataset.jsonl"

CLEAN_COUNT = 20000     # matches the size of your best previous run

def load(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rows.append(json.loads(line))
    return rows

clean = load(CLEAN_FILE, CLEAN_COUNT)
recovery = load(RECOVERY_FILE)

combined = clean + recovery
random.shuffle(combined)     # interleave so training doesn't see all-clean then all-recovery

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for ex in combined:
        f.write(json.dumps(ex) + "\n")

print(f"Clean: {len(clean)}, Recovery: {len(recovery)}, Total: {len(combined)}")