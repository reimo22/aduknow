from flask import Flask, render_template, request, jsonify
from llama_cpp import Llama
import os

app = Flask(__name__)

# Initialize model
model_path = "models/unsloth.Q4_K_M.gguf"

if os.path.exists(model_path):
    llm = Llama(model_path=model_path, verbose=False)
else:
    print(f"Model file not found at {model_path}. Attempting to download...")
    llm = Llama.from_pretrained(
        repo_id="lucid-gunner/adullama3.1-v2-GGUF",
        filename="unsloth.Q4_K_M.gguf",
        verbose=False,
        local_dir="./models/",
    )
print("Model loaded successfully")

# List to store messages
system_prompt = "You are AduKnow, Adamson University's dedicated digital assistant. You are a domain expert in the university's student manual. You will not answer any questions unrelated to the student manual."
messages = [{"role": "system", "content": system_prompt}]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user's input from the request
    data = request.get_json()
    user_input = data.get('prompt')

    if not user_input:
        return jsonify({'response': 'Please provide a valid input.'}), 400

    try:
        # Append user message to messages list
        messages.append({"role": "user", "content": user_input})

        # Call the Llama model to generate a response
        response_generator = llm.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1,
            stream=True,
        )

        # Collect the full response from the generator
        full_response = ""
        for chunk in response_generator:
            full_response += chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")

        # Extract the assistant's response
        assistant_response = full_response.strip() or "No response"

        # Append assistant message to messages list
        messages.append({"role": "assistant", "content": assistant_response})

        return jsonify({'response': assistant_response})

    except Exception as e:
        print(f"Error during model inference: {e}")
        return jsonify({'response': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)