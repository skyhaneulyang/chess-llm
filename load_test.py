from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = 4096,
    load_in_4bit = True,
)

FastLanguageModel.for_inference(model)

conversation = ""

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    conversation += f"\nYou: {user_input}\nAI:"
    inputs = tokenizer(conversation, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    conversation = response
    print(response[len(conversation) - len(response):])