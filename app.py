from flask import Flask, request, jsonify
import torch
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from flask_cors import CORS  # Enable CORS for frontend access

app = Flask(__name__)
CORS(app)  # Allow requests from any domain

# Use a small, free, lightweight model
MODEL_NAME = "google/flan-t5-small"

# Load the model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Tokenize input and generate response
        inputs = tokenizer(user_input, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=150)
        message = tokenizer.decode(output[0], skip_special_tokens=True)

        return jsonify({"response": message})
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    # Use PORT from environment variables for online deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
