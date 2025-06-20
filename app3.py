import streamlit as st
import sqlite3
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
import torch
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
import nltk
from nltk.corpus import stopwords
import urllib.parse

# -------------------- DATABASE SETUP --------------------
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT
                )''')
    conn.commit()
    conn.close()

def signup_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    result = c.fetchone()
    conn.close()
    return result is not None

init_db()

# -------------------- AUTH UI --------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
        height: 100%;
        margin: 0;
        padding: 0;
    }

    .stApp {
        background: url("https://i.postimg.cc/7hk9nY46/Background.webp") no-repeat center center fixed;
        background-size: cover;
    }

    .login-box {
        background: transparent;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        max-width: 400px;
        margin: 4rem auto;
    }

    .login-title {
        text-align: center;
        font-size: 30px;
        font-weight: bold;
        color: #000000;
        margin-bottom: 1.5rem;
    }

    label {
        font-weight: 600;
        color: white;
    }

    div.stButton > button {
        width: 100%;
        background-color: #00bcd4;
        color: white;
        border-radius: 8px;
        padding: 0.6em;
        font-weight: bold;
        margin-top: 1rem;
        border: none;
        transition: all 0.3s ease-in-out;
    }

    div.stButton > button:hover {
        background-color: #008ba3;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }

    .stRadio > div > label {
        color: white !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Auth logic block
if not st.session_state.get("logged_in"):
    auth_mode = st.radio("Choose Mode", ["Login", "Signup"], horizontal=True, label_visibility="collapsed")

    with st.container():
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown(f'<div class="login-title">🔐 {auth_mode}</div>', unsafe_allow_html=True)

        if auth_mode == "Signup":
            new_user = st.text_input("👤 Create Username", key="signup_user")
            new_pass = st.text_input("🔒 Create Password", type="password", key="signup_pass")
            if st.button("Sign Up"):
                try:
                    signup_user(new_user, new_pass)
                    st.success("✅ Account created. Please login.")
                except:
                    st.error("❌ Username might already exist.")

        elif auth_mode == "Login":
            username = st.text_input("👤 Username", key="login_user")
            password = st.text_input("🔒 Password", type="password", key="login_pass")
            if st.button("Login"):
                if login_user(username, password):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.success(f"✅ Welcome, {username}")
                    st.stop()
                else:
                    st.error("❌ Invalid login credentials")

        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()  # Prevent further app execution until logged in

# -------------------- NLP + ASR SETUP --------------------
nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT("all-MiniLM-L6-v2")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

def transcribe(audio_file):
    audio, sr = librosa.load(audio_file, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.decode(predicted_ids[0])

def preprocess_text(text):
    doc = nlp(text.lower())
    cleaned_tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and token.text not in stop_words and token.pos_ in {"NOUN", "PROPN", "VERB"}
    ]
    return " ".join(cleaned_tokens)

def extract_keywords_tfidf(text, n=5):
    text = preprocess_text(text)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=10)
    tfidf_matrix = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()
    return keywords[:n]

def extract_keywords_keybert(text, n=5):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=n)
    return [kw[0] for kw in keywords]

def extract_keywords_ner(text):
    doc = nlp(text)
    entities = {ent.text.strip() for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PERSON", "PRODUCT", "EVENT"]}
    return list(entities)

def refine_keywords(keywords):
    refined_set = set()
    for kw in sorted(keywords, key=len, reverse=True):
        if not any(kw in existing_kw for existing_kw in refined_set):
            refined_set.add(kw)
    return list(refined_set)

def extract_combined_keywords(text):
    tfidf_keywords = extract_keywords_tfidf(text)
    keybert_keywords = extract_keywords_keybert(text)
    ner_keywords = extract_keywords_ner(text)
    all_keywords = set(tfidf_keywords) | set(keybert_keywords) | set(ner_keywords)
    return refine_keywords(all_keywords)

def create_google_search_link(query):
    query_encoded = urllib.parse.quote(query)
    return f"https://www.google.com/search?q={query_encoded}"

# -------------------- MAIN UI --------------------
st.title("🎙 Speech Recognition with Advanced Keyword Extraction")
st.markdown(f"👤 *Logged in as:* {st.session_state['username']}")
uploaded_file = st.file_uploader("🎵 Upload Audio (WAV/MP3)", type=["wav", "mp3"])

if st.button("🚀 Transcribe and Extract Keywords"):
    if uploaded_file:
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
        st.write("🔍 *Processing transcription...*")
        transcription = transcribe(uploaded_file)
        st.success("✅ *Transcription Complete!*")
        st.markdown("### 📜 Transcription:")
        st.markdown(f"""<div style='color: #FFFF00; font-size: 18px; font-weight: 500; background-color: transparent; padding: 10px;'>
        {transcription}
        </div>
        """,unsafe_allow_html=True)

        keywords = extract_combined_keywords(transcription)
        st.write("### 🔑 *Extracted Keywords & Search Links:*")

        st.markdown("""
        <style>
        .keyword-box {
            display: inline-block;
            background-color: #0078D4;
            color: white;
            padding: 8px 12px;
            margin: 5px;
            border-radius: 12px;
            font-size: 16px;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)

        for keyword in keywords:
            search_link = create_google_search_link(keyword)
            st.markdown(f'<a href="{search_link}" target="_blank"><span class="keyword-box">{keyword}</span></a>', unsafe_allow_html=True)