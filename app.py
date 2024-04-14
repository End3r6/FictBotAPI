from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import model
from flask_cors import CORS

model_paths = {
    # 'tony' : './models/tony_chatbot',
    'dia_small' : 'microsoft/DialoGPT-small',
    'dia_medium' : 'microsoft/DialoGPT-medium',
    'dia_large' : 'microsoft/DialoGPT-large', 
    'blender_400m' : 'facebook/blenderbot-400M',
    'blender_90m' : 'facebook/blenderbot-90M',
    'blender_small' : 'facebook/blenderbot-small'
}

app = Flask(__name__)
CORS(app)

step = 0
chat_model = None
chat_model_tokenizer = None
model_name = None


@app.route('/model', methods=['POST'])
def post_model():
    global chat_model, chat_model_tokenizer, model_name

    model_name = request.args.get('model_name')

    if model_name is None:
        model_name = 'dia_medium'

    path = model_paths[model_name]

    chat_model = AutoModelForCausalLM.from_pretrained(path)
    chat_model_tokenizer = AutoTokenizer.from_pretrained(path)

    # Add CORS headers to the response
    response_headers = {
        'Access-Control-Allow-Origin': '*',  # Change the '*' to the appropriate origin if needed
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'POST'
    }

    return "", 200, response_headers


@app.route('/chat', methods=['GET'])
def get_response():
    global step, chat_model, chat_model_tokenizer, model_name

    prompt = request.args.get('prompt')

    if prompt is None:
        prompt = 'Hello'

    response = model.generate_response(chat_model, chat_model_tokenizer, prompt, step, max_length=30)

    step += 1

    # Add CORS headers to the response
    response_headers = {
        'Access-Control-Allow-Origin': '*',  # Change the '*' to the appropriate origin if needed
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'GET'
    }

    return jsonify({'name': model_name, 'response': response}), 200, response_headers


@app.route('/test', methods=['GET'])
def get_test():
    return jsonify({'name': 'test', 'response': 'test'})


if __name__ == '__main__':
    app.run(port=5000)