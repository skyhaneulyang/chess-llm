import json
import statistics
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        return {r["fen"]: r for r in map(json.loads, f)}


a_path, b_path = sys.argv[1], sys.argv[2]
a, b = load(a_path), load(b_path)

shared = [f for f in a if f in b
          and a[f]["model_loss"] is not None
          and b[f]["model_loss"] is not None]

if not shared:
    print("No positions where both models played legally — check the seeds match.")
    sys.exit()

a_scores = [a[f]["model_loss"] for f in shared]
b_scores = [b[f]["model_loss"] for f in shared]
diffs = [x - y for x, y in zip(a_scores, b_scores)]

sd = statistics.stdev(diffs) if len(diffs) > 1 else 0
se = sd / (len(diffs) ** 0.5) if diffs else 0

print(f"Positions where both played legally: {len(shared)}")
print(f"  A  {a_path}: {statistics.mean(a_scores):.0f} cp")
print(f"  B  {b_path}: {statistics.mean(b_scores):.0f} cp")
print(f"\nMean difference (A - B): {statistics.mean(diffs):.0f} cp")
print(f"Std dev of difference:   {sd:.0f} cp")
print(f"Std error:               {se:.0f} cp")
print(f"Positions where B was better: {sum(1 for d in diffs if d > 0)}/{len(diffs)}")