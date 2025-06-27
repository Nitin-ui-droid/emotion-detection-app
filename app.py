# -------------------- IMPORTS --------------------
import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
from googletrans import Translator
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

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

# -------------------- LOAD & PREPROCESS DATA --------------------
@st.cache_resource
def load_data_and_train_model():
    # Load dataset
    train_df = pd.read_csv("train.txt", names=["text", "emotion"], sep=";")
    test_df = pd.read_csv("test.txt", names=["text", "emotion"], sep=";")

    # Clean text
    train_df["clean_text"] = train_df["text"].apply(clean_text)
    test_df["clean_text"] = test_df["text"].apply(clean_text)

    # Tokenization
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(train_df["clean_text"])
    X_train = tokenizer.texts_to_sequences(train_df["clean_text"])
    X_test = tokenizer.texts_to_sequences(test_df["clean_text"])

    # Padding
    X_train = pad_sequences(X_train, maxlen=100)
    X_test = pad_sequences(X_test, maxlen=100)

    # Labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["emotion"])
    y_test = label_encoder.transform(test_df["emotion"])
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Build Model
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train Model
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=1)

    return model, tokenizer, label_encoder

# Load everything
model, tokenizer, label_encoder = load_data_and_train_model()

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
    'joy': 'https://youtu.be/AYcxiROIktI?si=r_VOBBK6NFRwl255',
    'sadness': 'https://youtu.be/p5UaxR79fmQ?si=3C0CKnaTuAB3AtKC',
    'anger': 'https://youtu.be/Vvwyg5NiGbI?si=vx-_WjD-8A02XTPP',
    'fear': 'https://youtu.be/1UvB8BzCp9Q?si=aflehFneyDd4xQxm',
    'love': 'https://youtu.be/pDTm7wACZqk?si=SDgLhDjHKjk5STr5',
    'surprise': 'https://youtu.be/fgbcH5DkUsk?si=I3HpHcwPB9FY87GO',
    'neutral': 'https://youtu.be/Gbi-2SwWs9Q?si=kI-TAW-k38mvvEkN'
}

# -------------------- STREAMLIT APP UI --------------------
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 Emotion Detection from Text</h1>", unsafe_allow_html=True)
st.markdown("Detect the emotion behind your text using Deep Learning and NLP 🚀")

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

            # Clean, Tokenize, Pad
            cleaned = clean_text(text)
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=100)

            # Predict
            pred = model.predict(padded)
            emotion = label_encoder.inverse_transform([np.argmax(pred)])[0]
            emoji = emotion_emoji_dict.get(emotion.lower(), "")
            song = emotion_song_dict.get(emotion.lower(), "")

            # Output
            st.success(f"**Emotion Detected:** `{emotion.upper()}` {emoji}")
            if song:
                st.markdown(f"🎧 [Click here to listen a song matching your mood]({song})")

        except Exception as e:
            st.error(f"❌ Error: {e}")
