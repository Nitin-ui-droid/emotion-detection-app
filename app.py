# -------------------- IMPORTS --------------------
import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
from googletrans import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Emotion Detection", layout="centered")


# -------------------- DOWNLOAD NLTK STOPWORDS --------------------
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# -------------------- TEXT CLEANING FUNCTION --------------------
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters & numbers
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# -------------------- LOAD & TRAIN MODEL --------------------
@st.cache_resource
def load_data_and_train_model():
    # Load dataset
    train_df = pd.read_csv("train.txt", names=["text", "emotion"], sep=";")
    test_df = pd.read_csv("test.txt", names=["text", "emotion"], sep=";")

    # Clean text
    train_df["clean_text"] = train_df["text"].apply(clean_text)
    test_df["clean_text"] = test_df["text"].apply(clean_text)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["emotion"])
    y_test = label_encoder.transform(test_df["emotion"])

    # Create pipeline with TF-IDF and Naive Bayes
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('nb', MultinomialNB())
    ])

    model.fit(train_df["clean_text"], y_train)
    return model, label_encoder

# Load everything
model, label_encoder = load_data_and_train_model()

# -------------------- TRANSLATOR --------------------
translator = Translator()

# -------------------- EMOJI + SONG DICTIONARIES --------------------
emotion_emoji_dict = {
    'joy': '😊',
    'anger': '😠',
    'sadness': '😢',
    'fear': '😱',
    'love': '😍',
    'surprise': '😲',
    'neutral': '😐'
}

emotion_song_dict = {
    'joy': 'https://youtu.be/Gbi-2SwWs9Q?si=kI-TAW-k38mvvEkN',
    'anger': 'https://youtu.be/Vvwyg5NiGbI?si=vx-_WjD-8A02XTPP',
    'sadness': 'https://youtu.be/p5UaxR79fmQ?si=3C0CKnaTuAB3AtKC',
    'fear': 'https://youtu.be/1UvB8BzCp9Q?si=aflehFneyDd4xQxm',
    'love': 'https://youtu.be/pDTm7wACZqk?si=SDgLhDjHKjk5STr5',
    'surprise': 'https://youtu.be/fgbcH5DkUsk?si=I3HpHcwPB9FY87GO',
    'neutral': 'https://youtu.be/AYcxiROIktI?si=r_VOBBK6NFRwl255'
}

# -------------------- STREAMLIT APP UI --------------------
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 Emotion Detection from Text</h1>", unsafe_allow_html=True)
st.markdown("Detect the emotion behind your text using Machine Learning and NLP 🚀")

language = st.selectbox("Select Input Language:", ["English", "Hindi"])
user_input = st.text_area("💬 Enter your text here:")

if st.button("🔍 Predict Emotion"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        try:
            # Translation if Hindi selected
            if language == "Hindi":
                translated = translator.translate(user_input, src='hi', dest='en')
                text = translated.text
            else:
                text = user_input

            # Clean the text
            cleaned = clean_text(text)

            # Predict
            pred = model.predict([cleaned])
            emotion = label_encoder.inverse_transform(pred)[0]
            emoji_icon = emotion_emoji_dict.get(emotion.lower(), "")
            song = emotion_song_dict.get(emotion.lower(), "")

            # Output
            st.success(f"**Emotion Detected:** `{emotion.upper()}` {emoji_icon}")
            if song:
                st.markdown(f"🎧 [Click here to listen a song matching your mood]({song})")

        except Exception as e:
            st.error(f"❌ Error: {e}")
