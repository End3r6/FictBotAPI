from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Conversation
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_paths = {
    'meena': 'google/meena-chatbot',
    'dia_small': 'microsoft/DialoGPT-small',
    'dia_medium': 'microsoft/DialoGPT-medium',
    'dia_large': 'microsoft/DialoGPT-large',
    'blender_400m': 'facebook/blenderbot-400M',
    'blender_90m': 'facebook/blenderbot-90M',
}

chat_model = None
chat_tokenizer = None
conversations = []
model_name = None

@app.route('/model', methods=['GET', 'POST'])
def post_model():
    global chat_model, chat_tokenizer, model_name

    model_name = request.args.get('model_name')

    if model_name is None:
        model_name = 'dia_medium'

    path = model_paths[model_name]

    chat_model = AutoModelForCausalLM.from_pretrained(path)
    chat_tokenizer = AutoTokenizer.from_pretrained(path)

    # Add CORS headers to the response
    response_headers = {
        'Access-Control-Allow-Origin': '*',  # Change the '*' to the appropriate origin if needed
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'POST'
    }

    return jsonify({'name': model_name, 'path': path}), 200, response_headers

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global conversations, chat_model, chat_tokenizer
    data = request.get_json()
    message = data['message']
    conversation_id = data['conversation_id']

    if len(conversations) <= conversation_id:
        conversations.append(Conversation())

    conversations[conversation_id].append_user(message)
    inputs = chat_tokenizer(conversations[conversation_id].messages, return_tensors='pt')
    reply = chat_model.generate(inputs.input_ids, max_length=50)

    conversations[conversation_id].append_system(reply)

    # Add CORS headers to the response
    response_headers = {
        'Access-Control-Allow-Origin': '*',  # Change the '*' to the appropriate origin if needed
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'POST'
    }

    return jsonify({'reply': conversations[conversation_id].messages[-1]['content']}), 200, response_headers

if __name__ == '__main__':
    app.run(port=5000)