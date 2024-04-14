from flask import Flask, jsonify, request

from transformers import AutoModelForCausalLM, AutoTokenizer
import model

chat_model = None
chat_model_tokenizer = None


model_paths = {
    # 'tony' : './models/tony_chatbot',
    'dia_small' : 'microsoft/DialoGPT-small',
    'dia_medium' : 'microsoft/DialoGPT-medium',
    'dia_large' : 'microsoft/DialoGPT-large' }


app = Flask(__name__)

step = 0

@app.route('/chat', methods=['GET'])
def get_response():
    global step

    model_name = request.args.get('model_name')
    prompt = request.args.get('prompt')

    path =  model_paths[model_name]

    chat_model = AutoModelForCausalLM.from_pretrained(path)
    chat_model_tokenizer = AutoTokenizer.from_pretrained(path)

    response = model.generate_response(chat_model, chat_model_tokenizer, prompt, step, max_length=30)

    step += 1

    return jsonify({'name': model_name, 'response' : response}).headers.add("Access-Control-Allow-Origin", "*")




if __name__ == '__main__':
   app.run(port=5000)