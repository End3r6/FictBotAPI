from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import model

model_paths = {
    # 'tony' : './models/tony_chatbot',
    'dia-small' : 'microsoft/DialoGPT-small',
    'dia-medium' : 'microsoft/DialoGPT-medium',
    'dia-large' : 'microsoft/DialoGPT-large', 
    'blender-400m' : 'facebook/blenderbot-400M-distill' }


chat_model = None
chat_model_tokenizer = None


def get_model_from_name(name):
    global chat_model
    global chat_model_tokenizer


    path =  model_paths[name]

    chat_model = AutoModelForCausalLM.from_pretrained(path)
    chat_model_tokenizer = AutoTokenizer.from_pretrained(path)



logging.get_logger("transformers").setLevel(logging.ERROR)

print("Welcome to the Chatbot!")
print("Please select a model:")
for key in model_paths.keys():
    print(key)

model_name = input("Enter the name: ")

get_model_from_name(model_name)

print(f"You can now chat with {model_name}.")

step = 0
while(True):
    prompt = input("You: ")
    response = model.generate_response(chat_model, chat_model_tokenizer, prompt, step, max_length=30)

    step += 1

    print(f"{model_name}: " + response)
