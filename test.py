from transformers import pipeline , AutoTokenizer
local_model_path = "./deepseek_model"

tokenizer = AutoTokenizer.from_pretrained(local_model_path)
pipe = pipeline(
    "text-generation",
    model=local_model_path,
    tokenizer=tokenizer,
    device=0
)
output = pipe("Can you explain the theory of relativity?", max_length=1000, truncation=True)
print(output[0]["generated_text"])