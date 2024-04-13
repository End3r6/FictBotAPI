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

    get_model_from_name(model_name)

    if chat_model is None:
        get_model_from_name('dia_medium')

    response = model.generate_response(chat_model, chat_model_tokenizer, prompt, step, max_length=30)

    step += 1
    
    response.headers.add("Access-Control-Allow-Origin", "*")
    return jsonify({'name': model_name, 'response' : response})

def get_model_from_name(name):
    global chat_model
    global chat_model_tokenizer


    path =  model_paths[name]

    chat_model = AutoModelForCausalLM.from_pretrained(path)
    chat_model_tokenizer = AutoTokenizer.from_pretrained(path)


if __name__ == '__main__':
   app.run(port=5000)