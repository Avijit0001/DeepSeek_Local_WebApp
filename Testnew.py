from transformers import AutoModelForCausalLM, AutoTokenizer

local_model_path = "./deepseek_model"

tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path).to('cuda')

prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a expert with advanced knowledge in physics .Please answer the following medical question. 

### Question:
{}

### Response:
<think>{}"""

question = """Can you explain the theory of relativity?"""

inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda") 

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask, 
    max_new_tokens=120, 
    use_cache=True, 
)

response = tokenizer.batch_decode(outputs)

print(response[0].split("### Response:")[1])  