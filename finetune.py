import os
import json
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

SUBSET_SIZE = 99186
LORA_RANK = 16
EPOCHS = 1
MAX_SEQ = 1024
BATCH_SIZE = 2

RUN_NAME = f"run_pgn_{SUBSET_SIZE}_r{LORA_RANK}_e{EPOCHS}_b{BATCH_SIZE}"

examples = []
with open("pgn_dataset.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= SUBSET_SIZE:
            break
        examples.append(json.loads(line))

with open("pgn_dataset_subset.jsonl", "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

dataset = load_dataset("json", data_files="pgn_dataset_subset.jsonl", split="train")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = MAX_SEQ,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_RANK,
    lora_alpha = LORA_RANK,
    lora_dropout = 0,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing = "unsloth",
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ,
    args = SFTConfig(
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = 4,
        num_train_epochs = EPOCHS,
        learning_rate = 2e-4,
        logging_steps = 10,
        save_steps = 100,
        output_dir = RUN_NAME,
    ),
)

has_checkpoint = os.path.isdir(RUN_NAME) and any(
    d.startswith("checkpoint-") for d in os.listdir(RUN_NAME)
)
trainer.train(resume_from_checkpoint=has_checkpoint)

model.save_pretrained(RUN_NAME + "_model")
tokenizer.save_pretrained(RUN_NAME + "_model")
print(f"Training complete. Model saved to {RUN_NAME}_model/")