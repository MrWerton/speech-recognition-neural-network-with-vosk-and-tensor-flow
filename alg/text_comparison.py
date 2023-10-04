import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity


embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

nltk.download("punkt")
nltk.download("stopwords")


def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word not in stopwords.words("english")]
    preprocessed_text = " ".join(words)
    return preprocessed_text


def calculate_similarity(text1, text2):
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)

    embeddings = embed([text1, text2])

    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    return similarity_score
