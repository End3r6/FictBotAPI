import re
import spacy

nlp = spacy.load("en_core_web_sm")  # Load English tokenizer, tagger, parser, NER and word vectors

def tokenize_text(text):
    # Tokenization using spaCy
    tokens = [token.text for token in nlp(text)]
    return tokens

def clean_text(text):
    # Lowercase and remove punctuation and special characters
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    return cleaned_text

def preprocess_data(text_data):
    preprocessed_data = []
    for text in text_data:
        # Tokenize and clean each text
        tokens = tokenize_text(text)
        cleaned_tokens = [clean_text(token) for token in tokens]
        preprocessed_data.append(cleaned_tokens)
    return preprocessed_data
