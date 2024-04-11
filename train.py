import torch
import os
import re
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments



def read_txt(file_path):
    text = ""
    with open(file_path, 'r') as file:
        text = file.read()
    return text

def read_documents_from_directory(dir_name):
    combined_text = ''
    for filename in os.listdir(dir_name):
        file_path = os.path.join(dir_name, filename)

        if filename.endswith('.txt'):
            combined_text += read_txt(file_path)

    return combined_text

def train_chatbot(directory, model_output_path, train_fraction=0.8):
    combined_text = read_documents_from_directory(directory)
    combined_text = re.sub(r'\n+', '\n', combined_text).strip()

    split_index = int(train_fraction * len(combined_text))
    train_text = combined_text[:split_index]
    test_text = combined_text[split_index:]

    with open('train.txt', 'w') as f:
        f.write(train_text)
    with open('val.txt', 'w') as f:
        f.write(test_text)

    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
    model  = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

    train_dataset = TextDataset(tokenizer=tokenizer, file_path='train.txt', block_size=128)
    val_dataset = TextDataset(tokenizer=tokenizer, file_path='val.txt', block_size=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=model_output_path,                   # output directory
        overwrite_output_dir=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=30,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',     # directory for storing logs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    trainer.save_model(model_output_path)

    tokenizer.save_pretrained(model_output_path)


train_chatbot('./data/processed/', './models/tony_chatbot')
