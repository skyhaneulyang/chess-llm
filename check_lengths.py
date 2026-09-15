import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B")

lens = []
with open("pgn_dataset.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 2000:
            break
        lens.append(len(tok(json.loads(line)["text"])["input_ids"]))

lens.sort()
n = len(lens)
print(f"median {lens[n//2]}")
print(f"p90    {lens[int(n*0.90)]}")
print(f"p99    {lens[int(n*0.99)]}")
print(f"max    {lens[-1]}")
print(f"over 1024: {sum(1 for x in lens if x > 1024)}/{n}")