import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
from collections import Counter

nltk.download('stopwords')

# Load dataset
data = pd.read_csv("bias_dataset.csv")

# Stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    words = text.lower().split()
    return " ".join([w for w in words if w not in stop_words])

data['text'] = data['text'].apply(clean_text)

# ML model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

model = MultinomialNB()
model.fit(X, y)

# Sentiment
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Rule-based system
def rule_based_detection(text):
    text = text.lower()

    if "yeah right" in text or "as if" in text:
        return "no_bias", "Sarcasm detected"

    if "but" in text or "however" in text:
        return "no_bias", "Balanced/contrasting statement"

    if ("always" in text or "never" in text):
        if "try" in text or "improve" in text:
            return None, "Positive context overrides extreme word"
        return "overgeneralization", "Extreme absolute terms detected"

    if "worst" in text or "ruined" in text or "disaster" in text:
        return "catastrophizing", "Worst-case thinking"

    if "my fault" in text or "because of me" in text:
        return "personalization", "Self-blame pattern"

    if "feel" in text and ("am" in text or "is" in text):
        return "emotional_reasoning", "Emotion used as evidence"

    if "know everything" in text or "no need to study" in text:
        return "overconfidence", "Overconfidence detected"

    return None, "No rule triggered"

# Advice
def generate_advice(bias):
    advice_dict = {
        "overgeneralization": [
            "Avoid extreme conclusions and focus on specific situations.",
            "One event does not define everything.",
            "Look at exceptions to your belief."
        ],
        "catastrophizing": [
            "Think of realistic outcomes instead of worst-case.",
            "This situation is manageable.",
            "Stay calm and evaluate logically."
        ],
        "personalization": [
            "Not everything is your fault.",
            "Consider external factors.",
            "Be fair to yourself."
        ],
        "emotional_reasoning": [
            "Feelings are not always facts.",
            "Separate emotions from reality.",
            "Think logically."
        ],
        "overconfidence": [
            "Stay open to learning.",
            "Practice strengthens ability.",
            "Remain humble."
        ],
        "no_bias": [
            "Your thinking is balanced and rational.",
            "Keep maintaining this mindset.",
            "You are thinking clearly."
        ],
        "uncertainty": [
            "It's okay to be unsure.",
            "Gather more information before deciding.",
            "Uncertainty is a part of learning."
        ]
    }
    return advice_dict.get(bias, ["Think carefully."])

# Greedy search
def select_best_advice(advice_list):
    return max(advice_list, key=len)

# -------- INPUT -------- #

user_input = input("Enter your thought: ")

# Logging
with open("user_log.txt", "a") as f:
    f.write(user_input + "\n")

# Split sentences
sentences = [s.strip() for s in user_input.split('.') if s.strip()]

results = []

for sentence in sentences:
    sentiment = get_sentiment(sentence)
    rule_result, rule_reason = rule_based_detection(sentence)

    # Ignore false overgeneralization
    if rule_result == "overgeneralization" and sentiment > 0:
        rule_result = None

    if rule_result:
        results.append((rule_result, rule_reason))
        continue

    # ML fallback
    clean = [clean_text(sentence)]
    vec = vectorizer.transform(clean)

    probs = model.predict_proba(vec)
    confidence = max(probs[0])
    pred = model.predict(vec)[0]

    if confidence < 0.5:
        results.append(("uncertainty", "Uncertainty without distortion"))
    else:
        results.append((pred, "ML-based prediction"))

# Majority decision
bias_counts = Counter([r[0] for r in results])
final_bias = bias_counts.most_common(1)[0][0]

# Reason selection
final_reason = [r[1] for r in results if r[0] == final_bias][0]

# Global sentiment override
overall_sentiment = get_sentiment(user_input)
if abs(overall_sentiment) < 0.2:
    final_bias = "no_bias"
    final_reason = "Neutral and balanced thinking"

# Advice
advice = select_best_advice(generate_advice(final_bias))

print("\nDetected Bias:", final_bias)
print("Reason:", final_reason)
print("Best Advice:", advice)