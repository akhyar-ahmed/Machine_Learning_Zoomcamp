"""
To run it:
waitress-serve --host=0.0.0.0 --port=8001 main:app
"""
import gradio as gr
import joblib
import re
import nltk
import pandas as pd

from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer



# Download NLTK resources (if not already downloaded)
nltk.download("stopwords")
nltk.download("wordnet")


tokenizer = TweetTokenizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# Load the pre-trained model and other necessary preprocessing steps
with open("./models/LogisticRegression_best_model.pkl", "rb") as model_file:
    model = joblib.load(model_file)

# Load the pre-trained model and other necessary preprocessing steps
with open("./models/LogisticRegression_tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = joblib.load(vec_file)



def preprocess_tweet(tweet):
    # Convert to lowercase
    tweet = tweet.lower()

    # Remove URLs
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)

    # Remove mentions and hashtags
    tweet = re.sub(r"@\w+|\#", "", tweet)

    # Remove special characters and punctuation
    tweet = re.sub(r"[^\w\s]", "", tweet)

    # Remove spaces
    tweet = re.sub(r"\s+", " ", tweet)

    # Remove unnecessary dots
    tweet = re.sub(r"\.{2,}", ".", tweet)

    # Remove dots at the beginning or end of the sentence
    tweet = tweet.strip(".")

    # Remove spaces at the beginning or end of the sentence
    tweet = tweet.strip(" ")

    # Tokenization
    tokens = tokenizer.tokenize(tweet)

    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Join tokens back to text
    preprocessed_tweet = " ".join(lemmatized_tokens)
    return preprocessed_tweet


# Create a FastAPI application
app = FastAPI()

CUSTOM_PATH = "/predict"


# Function to predict sentiment from the tweet
def predict_sentiment(tweet):
    # processed_tweet = model.preprocess_tweet(tweet)
    # prediction = model.predict(processed_tweet)
    tweet = preprocess_tweet(tweet)
    encoded_tweet = vectorizer.transform([tweet])
    prediction = model.predict(encoded_tweet)
    return "Positive" if prediction == 1 else "Negative"

# Define the Gradio interface
tweet_input = gr.Textbox(placeholder="Enter your tweet here...")
prediction_output = gr.Label()


# Interface route for Gradio
interface = gr.Interface(fn=predict_sentiment, inputs=tweet_input, outputs=prediction_output)
app = gr.mount_gradio_app(app, interface, path=CUSTOM_PATH)
