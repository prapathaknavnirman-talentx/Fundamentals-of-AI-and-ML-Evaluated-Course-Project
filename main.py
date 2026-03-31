import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
from collections import Counter
# Download English stopwords (common filler words like "the", "is", "and")
nltk.download('stopwords')
# Load the labelled training dataset (columns: 'text', 'label')
dataset = pd.read_csv("bias_dataset.csv")
# Build a set of English stopwords for fast membership checks
english_stopwords = set(stopwords.words('english'))
def clean_text(text):
    """Lowercase the text and remove stopwords so only meaningful words remain."""
    words = text.lower().split()
    meaningful_words = [w for w in words if w not in english_stopwords]
    return " ".join(meaningful_words)
# Apply cleaning to every row in the dataset's text column
dataset['text'] = dataset['text'].apply(clean_text)
# TF-IDF converts text into numeric features (word importance scores)
vectorizer = TfidfVectorizer()
feature_matrix = vectorizer.fit_transform(dataset['text'])
labels = dataset['label']
# Train a Naive Bayes classifier on the TF-IDF features
bias_classifier = MultinomialNB()
bias_classifier.fit(feature_matrix, labels)
def get_sentiment_score(text):
    """Return a polarity score: negative (< 0), neutral (≈ 0), or positive (> 0)."""
    return TextBlob(text).sentiment.polarity
def detect_bias_by_rules(text):
    """
        Apply hand-crafted rules to catch common cognitive biases.
        Returns (bias_type, reason) or (None, reason) if no bias found.
    """
    text = text.lower()
    # Sarcastic phrases like "yeah right" or "as if" suggest the speaker
    # doesn't literally mean what they say — not a genuine bias.
    if "yeah right" in text or "as if" in text:
        return "no_bias", "Sarcasm detected"
    # "But" / "however" signal the speaker is considering both sides.
    if "but" in text or "however" in text:
        return "no_bias", "Balanced/contrasting statement"
    # Absolute words ("always", "never") can signal overgeneralisation,
    # unless they appear alongside growth-oriented words like "try" or "improve".
    if "always" in text or "never" in text:
        if "try" in text or "improve" in text:
            return None, "Positive context overrides extreme word"
        return "overgeneralization", "Extreme absolute terms detected"
    # Catastrophising: focusing on worst-case outcomes
    if "worst" in text or "ruined" in text or "disaster" in text:
        return "catastrophizing", "Worst-case thinking"
    # Personalisation: blaming yourself for external events
    if "my fault" in text or "because of me" in text:
        return "personalization", "Self-blame pattern"
    # Emotional reasoning: treating feelings as objective proof
    if "feel" in text and ("am" in text or "is" in text):
        return "emotional_reasoning", "Emotion used as evidence"
    # Overconfidence: believing you know everything already
    if "know everything" in text or "no need to study" in text:
        return "overconfidence", "Overconfidence detected"
    return None, "No rule triggered"
# Maps each bias type to a list of helpful reframing tips
ADVICE_BY_BIAS = {
    "overgeneralization": [
        "Avoid extreme conclusions and focus on specific situations.",
        "One event does not define everything.",
        "Look at exceptions to your belief.",
    ],
    "catastrophizing": [
        "Think of realistic outcomes instead of worst-case.",
        "This situation is manageable.",
        "Stay calm and evaluate logically.",
    ],
    "personalization": [
        "Not everything is your fault.",
        "Consider external factors.",
        "Be fair to yourself.",
    ],
    "emotional_reasoning": [
        "Feelings are not always facts.",
        "Separate emotions from reality.",
        "Think logically.",
    ],
    "overconfidence": [
        "Stay open to learning.",
        "Practice strengthens ability.",
        "Remain humble.",
    ],
    "no_bias": [
        "Your thinking is balanced and rational.",
        "Keep maintaining this mindset.",
        "You are thinking clearly.",
    ],
    "uncertainty": [
        "It's okay to be unsure.",
        "Gather more information before deciding.",
        "Uncertainty is a part of learning.",
    ],
}
def get_advice_for_bias(bias_type):
    """Look up advice tips for the given bias type. Falls back to a generic tip."""
    return ADVICE_BY_BIAS.get(bias_type, ["Think carefully."])
def pick_longest_advice(advice_list):
    """Pick the most detailed advice (longest string) as a greedy heuristic."""
    return max(advice_list, key=len)
user_input = input("Enter your thought: ")
# Append the raw input to a log file for later review or debugging
with open("user_log.txt", "a") as log_file:
    log_file.write(user_input + "\n")
# Split the input into individual sentences for finer-grained analysis
sentences = [s.strip() for s in user_input.split('.') if s.strip()]
sentence_results = []  # Each entry: (bias_type, reason)
for sentence in sentences:
    sentiment_score = get_sentiment_score(sentence)
    rule_bias, rule_reason = detect_bias_by_rules(sentence)
    # A positive sentiment suggests the "always/never" is used constructively,
    # so we override the overgeneralisation label.
    if rule_bias == "overgeneralization" and sentiment_score > 0:
        rule_bias = None
    # If a rule matched, use it directly — rules take priority over the ML model.
    if rule_bias:
        sentence_results.append((rule_bias, rule_reason))
        continue
    # Fallback: use the ML classifier when no rule applies
    cleaned_sentence = [clean_text(sentence)]
    feature_vector = vectorizer.transform(cleaned_sentence)
    class_probabilities = bias_classifier.predict_proba(feature_vector)
    top_confidence = max(class_probabilities[0])
    predicted_bias = bias_classifier.predict(feature_vector)[0]
    # Below 50% confidence the model is unsure — label it "uncertainty"
    if top_confidence < 0.5:
        sentence_results.append(("uncertainty", "Low-confidence ML prediction"))
    else:
        sentence_results.append((predicted_bias, "ML-based prediction"))
# Count how many sentences were labelled with each bias type
bias_frequency = Counter([result[0] for result in sentence_results])
# Choose the most frequent bias as the overall verdict
dominant_bias = bias_frequency.most_common(1)[0][0]
# Retrieve the reason from the first sentence that produced this bias
dominant_reason = [r[1] for r in sentence_results if r[0] == dominant_bias][0]
# Global sentiment override: if the full input is nearly neutral (|score| < 0.2),
# treat the thought as balanced regardless of individual sentence labels.
overall_sentiment = get_sentiment_score(user_input)
if abs(overall_sentiment) < 0.2:
    dominant_bias = "no_bias"
    dominant_reason = "Neutral and balanced thinking"
# Select and display the most informative advice tip
best_advice = pick_longest_advice(get_advice_for_bias(dominant_bias))
print("\nDetected Bias :", dominant_bias)
print("Reason        :", dominant_reason)
print("Best Advice   :", best_advice)