from transformers import AutoModelForCausalLM, AutoTokenizer
local_model_path = "./deepseek_model"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path).to('cuda')

prompt = ("As an experienced physicist, could you explain the theory of relativity "
          "in simple terms for someone new to physics?")

inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

outputs = model.generate(
    **inputs,
    max_length=250,
    num_beams=5,
    no_repeat_ngram_size=2,
    repetition_penalty=1.2,
    early_stopping=True,
    eos_token_id=tokenizer.eos_token_id,
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
