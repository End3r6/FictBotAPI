import torch

last_output = None
def generate_response(model, tokenizer, prompt, step, max_length=250):
    global last_output

    input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')

    # Check if last_output is None or empty string
    if last_output is None:
        bot_input_ids = input_ids
    else:
        # Convert last_output to tensor if it's not None
        last_output_tensor = torch.tensor(last_output)
        bot_input_ids = torch.cat([last_output_tensor, input_ids], dim=-1) if step > 0 else input_ids

    attention_mask = torch.ones_like(input_ids)
    pad_token_id = tokenizer.eos_token_id

    output = model.generate(
        bot_input_ids,
        max_length=max_length,
        attention_mask=attention_mask,
        num_beams=3,
        early_stopping=True,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.75,
        # num_return_sequences=5,
        pad_token_id=pad_token_id
    )

    return tokenizer.decode(output[:, bot_input_ids.shape[-1]:][0],  skip_special_tokens=True)


