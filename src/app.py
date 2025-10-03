import joblib
from flask import Flask, request, jsonify
import pandas as pd
import re
import emoji
import nltk
from nltk.tokenize import word_tokenize
import spacy
import string
from nltk.corpus import stopwords
# from src.utils import clean_tweet

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load the spaCy model (make sure it's installed: pip install spacy && python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Define stopwords list globally
stopwords_list = set(stopwords.words("english"))


def clean_tweet(text, demojize=True, hashtag_handling='remove_symbol', remove_short_tokens=True):
    if demojize:
        text = emoji.demojize(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    if hashtag_handling == 'remove_all':
        text = re.sub(r"#\w+", "", text)
    elif hashtag_handling == 'remove_symbol':
        text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\.{2,}", " ", text)
    text = re.sub(r"\^\^", "", text)
    text = re.sub(r"w/", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    if remove_short_tokens:
        tokens = text.split()
        tokens = [t for t in tokens if len(t) > 2]
        text = " ".join(tokens)
    return text

def preprocess_tokens(text, remove_stopwords=True, replace_numbers=True):
    text = clean_tweet(text)
    text = text.lower()
    if replace_numbers:
        text = re.sub(r"\b\d+\b", "", text)
    tokens = [t for t in word_tokenize(text) if t not in string.punctuation]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stopwords_list]
    return tokens

def pos_counts(text):
    doc = nlp(clean_tweet(text))
    noun = sum(1 for tok in doc if tok.pos_ == "NOUN")
    verb = sum(1 for tok in doc if tok.pos_ == "VERB")
    adj = sum(1 for tok in doc if tok.pos_ == "ADJ")
    return noun, verb, adj

# Load the saved model pipeline
try:
    model_pipeline = joblib.load(r'E:\DEPI\Techical Sills (AI)\Sentiment Analysis Project\models\sentiment_model.pkl')
except FileNotFoundError:
    print("Error: sentiment_model.pkl not found. Make sure you have saved the model.")
    model_pipeline = None

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if model_pipeline is None:
        return jsonify({'error': 'Model not loaded. Please train and save the model first.'}), 500

    data = request.get_json(force=True)
    tweet = data.get('tweet', '')

    if not tweet:
        return jsonify({'error': 'No tweet provided in the request body.'}), 400

    noun, verb, adj = pos_counts(tweet)
    input_data = pd.DataFrame({
        'text': [tweet],
        'noun_count': [noun],
        'verb_count': [verb],
        'adj_count': [adj]
    })

    prediction = model_pipeline.predict(input_data)[0]
    return jsonify({'sentiment': prediction})

if __name__ == '__main__':
    app.run(debug=True)
