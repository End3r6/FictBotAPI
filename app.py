from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Conversation as Conversation
from flask_cors import CORS
from model import generate_response

app = Flask(__name__)
CORS(app)

model_paths = {
    'dia_medium': 'microsoft/DialoGPT-medium',
    'dia_small': 'microsoft/DialoGPT-small',
    'blender_90m': 'facebook/blenderbot-90M'

    # 'dia_large': 'microsoft/DialoGPT-large',
    # 'blender_400m': 'facebook/blenderbot-400M',
    # 'octo': 'NexaAIDev/Octopus-v2',
    # 'dia_base': 'microsoft/DialoGPT-base',
}

chat_model = None
chat_tokenizer = None
conversations = []
model_name = None

step = 0

@app.route('/model', methods=['GET', 'POST'])
def post_model():
    global chat_model, chat_tokenizer, model_name

    model_name = request.args.get('model_name')

    if model_name is None:
        model_name = 'dia_medium'

    path = model_paths[model_name]

    chat_model = AutoModelForCausalLM.from_pretrained(path)
    chat_tokenizer = AutoTokenizer.from_pretrained(path)
    chat_tokenizer.pad_token = chat_tokenizer.eos_token

    # Add CORS headers to the response
    response_headers = {
        'Access-Control-Allow-Origin': '*',  # Change the '*' to the appropriate origin if needed
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'POST'
    }

    return jsonify({'name': model_name, 'path': path}), 200, response_headers

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global step, tokenizer, last_output, conversations
    
    data = request.get_json()
    message = data['message']
    # conversation_id = data['conversation_id']
        
    reply = generate_response(chat_model, chat_tokenizer, message, step, max_length=250)
    
    # Add CORS headers to the response
    response_headers = {
        'Access-Control-Allow-Origin': '*',  # Change the '*' to the appropriate origin if needed
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'POST'
    }

    step += 1
    
    return jsonify({'reply': reply}), 200, response_headers

if __name__ == '__main__':
    # Add CORS headers to the response
    response_headers = {
        'Access-Control-Allow-Origin': '*',  # Change the '*' to the appropriate origin if needed
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'POST'
    }

    app.run(port=8000)