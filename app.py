from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load the model and tokenizer
local_model_path = "./deepseek_model"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path).to('cuda')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
    Write a response that appropriately completes the request. Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

    ### Instruction:
    You are a expert with advanced knowledge in physics. Please answer the following medical question. 

    ### Question:
    {}

    ### Response:
    <think>{}"""

    formatted_input = prompt_style.format(user_message, "")
    inputs = tokenizer([formatted_input], return_tensors="pt").to('cuda')

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=120,
        use_cache=True,
    )

    response = tokenizer.batch_decode(outputs)
    bot_response = response[0].split("### Response:")[1].strip()

    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
