from flask import Flask, jsonify, request

from transformers import AutoModelForCausalLM, AutoTokenizer
import model
from flask_cors import CORS

model_paths = {
    # 'tony' : './models/tony_chatbot',
    'dia_small' : 'microsoft/DialoGPT-small',
    'dia_medium' : 'microsoft/DialoGPT-medium',
    'dia_large' : 'microsoft/DialoGPT-large', 
    'blender-400m' : 'facebook/blenderbot-400M-distill' }


app = Flask(__name__)
CORS(app)

step = 0

@app.route('/chat', methods=['GET'])
def get_response():
    global step

    model_name = request.args.get('model_name')
    prompt = request.args.get('prompt')

    if model_name is None:
        model_name = 'dia_medium'
    if prompt is None:
        prompt = 'Hello'

    path =  model_paths[model_name]

    chat_model = AutoModelForCausalLM.from_pretrained(path)
    chat_model_tokenizer = AutoTokenizer.from_pretrained(path)

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
    return jsonify({'name': 'test', 'response' : 'test'})




if __name__ == '__main__':
   app.run(port=5000)