import streamlit as st
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
from collections import Counter

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="RationalMind AI", page_icon="🧠")

nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))

# ---------------- PREPROCESS ---------------- #
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    return " ".join([w for w in words if w not in STOP_WORDS])

# ---------------- LOAD MODEL (CACHED) ---------------- #
@st.cache_resource
def load_model():
    data = pd.read_csv("bias_dataset.csv")
    data['text'] = data['text'].apply(clean_text)

    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    X = vectorizer.fit_transform(data['text'])
    y = data['label']

    model = MultinomialNB()
    model.fit(X, y)

    return vectorizer, model

vectorizer, model = load_model()

# ---------------- SENTIMENT ---------------- #
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# ---------------- RULE ENGINE ---------------- #
def rule_engine(text):
    text = text.lower()

    rules = [
        ("sarcasm", ["yeah right", "as if"], "no_bias", "Sarcasm detected"),
        ("contrast", ["but", "however"], "no_bias", "Balanced thinking"),
        ("overgeneralization", ["always", "never"], "overgeneralization", "Extreme terms"),
        ("catastrophizing", ["worst", "ruined", "disaster"], "catastrophizing", "Worst-case thinking"),
        ("personalization", ["my fault", "because of me"], "personalization", "Self-blame"),
        ("emotional", ["i feel"], "emotional_reasoning", "Emotion-based reasoning"),
        ("overconfidence", ["know everything", "no need to study"], "overconfidence", "Overconfidence")
    ]

    for _, keywords, label, reason in rules:
        for word in keywords:
            if word in text:
                # context override
                if label == "overgeneralization" and ("try" in text or "improve" in text):
                    return None, "Positive context overrides extreme word"
                return label, reason

    return None, "No rule triggered"

# ---------------- ADVICE ---------------- #
def get_advice(bias):
    advice_map = {
        "overgeneralization": "Avoid extreme conclusions. Focus on specific situations.",
        "catastrophizing": "Think realistically. This situation is manageable.",
        "personalization": "Not everything is your fault. Consider other factors.",
        "emotional_reasoning": "Feelings are not facts. Think logically.",
        "overconfidence": "Stay open to learning and keep practicing.",
        "no_bias": "Your thinking is balanced and rational.",
        "uncertainty": "It's okay to be unsure. Gather more information."
    }
    return advice_map.get(bias, "Think carefully.")

# ---------------- ANALYSIS CORE ---------------- #
def analyze_text(user_input):

    sentences = [s.strip() for s in user_input.split('.') if s.strip()]
    results = []

    for sentence in sentences:
        sentiment = get_sentiment(sentence)

        rule_label, reason = rule_engine(sentence)

        if rule_label == "overgeneralization" and sentiment > 0:
            rule_label = None

        if rule_label:
            results.append((rule_label, reason, 1.0))
            continue

        clean = clean_text(sentence)
        vec = vectorizer.transform([clean])

        probs = model.predict_proba(vec)[0]
        confidence = max(probs)
        pred = model.predict(vec)[0]

        if confidence < 0.55:
            results.append(("uncertainty", "Low confidence prediction", confidence))
        else:
            results.append((pred, "ML prediction", confidence))

    # weighted decision
    bias_scores = {}
    for label, _, conf in results:
        bias_scores[label] = bias_scores.get(label, 0) + conf

    final_bias = max(bias_scores, key=bias_scores.get)
    final_reason = [r[1] for r in results if r[0] == final_bias][0]

    # global sentiment override
    if abs(get_sentiment(user_input)) < 0.2:
        final_bias = "no_bias"
        final_reason = "Neutral and balanced thinking"

    return final_bias, final_reason

# ---------------- UI ---------------- #

st.title("🧠 RationalMind AI")
st.caption("Cognitive Bias Detector & Intelligent Decision Advisor")

user_input = st.text_area("Enter your thought:")

if st.button("Analyze"):
    if user_input.strip():

        bias, reason = analyze_text(user_input)
        advice = get_advice(bias)

        st.success(f"Detected Bias: {bias}")
        st.info(f"Reason: {reason}")
        st.warning(f"Advice: {advice}")

    else:
        st.warning("Please enter a valid thought.")

# Footer
st.markdown("---")
st.caption("Developed as part of BYOP AI/ML Project")