"""
RationalMind AI — Cognitive Bias Detector & Thinking Companion
--------------------------------------------------------------
This app listens to your thoughts and gently helps you notice
patterns in your thinking — like catastrophizing, overgeneralizing,
or being too hard on yourself.

It's not here to judge. It's here to help you think more clearly.
"""

import streamlit as st
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob

# ------------------------------------------------------------------ #
#  Page setup
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="RationalMind — Your Thinking Companion",
    page_icon="🧠",
    layout="centered"
)

# Download stopwords quietly on first run
nltk.download("stopwords", quiet=True)
COMMON_WORDS = set(stopwords.words("english"))


# ------------------------------------------------------------------ #
#  Text cleaning
# ------------------------------------------------------------------ #

def clean_thought(text: str) -> str:
    """
    Strip punctuation and filler words so the model focuses
    on the meaningful words in what you wrote.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    meaningful_words = [word for word in text.split() if word not in COMMON_WORDS]
    return " ".join(meaningful_words)


# ------------------------------------------------------------------ #
#  Load and train the bias detection model (cached for speed)
# ------------------------------------------------------------------ #

@st.cache_resource(show_spinner="Getting the model ready…")
def load_bias_model():
    """
    Trains a simple Naive Bayes classifier on your dataset.
    Cached so it only runs once per session — not on every thought.
    """
    training_data = pd.read_csv("bias_dataset.csv")
    training_data["text"] = training_data["text"].apply(clean_thought)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    features = vectorizer.fit_transform(training_data["text"])
    labels = training_data["label"]

    classifier = MultinomialNB()
    classifier.fit(features, labels)

    return vectorizer, classifier


vectorizer, classifier = load_bias_model()


# ------------------------------------------------------------------ #
#  Sentiment helper
# ------------------------------------------------------------------ #

def measure_emotional_tone(text: str) -> float:
    """
    Returns a polarity score between -1 (very negative) and +1 (very positive).
    Near zero means the thought is emotionally neutral.
    """
    return TextBlob(text).sentiment.polarity


# ------------------------------------------------------------------ #
#  Rule-based pattern recognition
#  (catches obvious language patterns before the ML model weighs in)
# ------------------------------------------------------------------ #

THINKING_PATTERNS = [
    {
        "name":     "sarcasm",
        "triggers": ["yeah right", "as if"],
        "label":    "no_bias",
        "note":     "Sounds sarcastic — that's actually a sign of perspective."
    },
    {
        "name":     "balanced thinking",
        "triggers": ["but", "however"],
        "label":    "no_bias",
        "note":     "You're weighing both sides. That's healthy."
    },
    {
        "name":     "overgeneralization",
        "triggers": ["always", "never"],
        "label":    "overgeneralization",
        "note":     "Watch out for absolute language — life is rarely all-or-nothing."
    },
    {
        "name":     "catastrophizing",
        "triggers": ["worst", "ruined", "disaster"],
        "label":    "catastrophizing",
        "note":     "You might be imagining the worst-case scenario."
    },
    {
        "name":     "personalization",
        "triggers": ["my fault", "because of me"],
        "label":    "personalization",
        "note":     "You're taking on a lot of blame. Are you being fair to yourself?"
    },
    {
        "name":     "emotional reasoning",
        "triggers": ["i feel"],
        "label":    "emotional_reasoning",
        "note":     "Feelings are real — but they're not always facts."
    },
    {
        "name":     "overconfidence",
        "triggers": ["know everything", "no need to study"],
        "label":    "overconfidence",
        "note":     "Confidence is great. Just leave room for curiosity too."
    },
]


def check_for_obvious_patterns(text: str):
    """
    Scans for well-known cognitive distortion phrases before
    handing off to the ML model. Returns (label, explanation) or (None, reason).
    """
    lowered = text.lower()

    for pattern in THINKING_PATTERNS:
        for trigger in pattern["triggers"]:
            if trigger in lowered:

                # Positive context softens overgeneralization
                if pattern["label"] == "overgeneralization":
                    if any(word in lowered for word in ["try", "improve"]):
                        return None, "The context here is actually positive — good on you."

                return pattern["label"], pattern["note"]

    return None, "No clear pattern spotted by the rules."


# ------------------------------------------------------------------ #
#  Friendly advice for each bias type
# ------------------------------------------------------------------ #

ADVICE_FOR_EACH_BIAS = {
    "overgeneralization": (
        "Try swapping 'always' or 'never' with 'sometimes' or 'in this case'. "
        "One hard moment doesn't define every moment."
    ),
    "catastrophizing": (
        "Take a breath. Ask yourself: what's the most *realistic* outcome here? "
        "You've handled hard things before."
    ),
    "personalization": (
        "Not everything is your fault. Most situations involve many people and factors. "
        "Try listing what *wasn't* in your control."
    ),
    "emotional_reasoning": (
        "Feelings are valid — but they're not always the full picture. "
        "What evidence supports or contradicts what you're feeling?"
    ),
    "overconfidence": (
        "Confidence is a strength. Just stay curious and open to learning. "
        "Even experts have more to discover."
    ),
    "no_bias": (
        "Your thinking looks balanced and grounded. "
        "Keep questioning gently — that's what clear thinking looks like."
    ),
    "uncertainty": (
        "This one's a little mixed. That's okay — complex thoughts don't always fit neat boxes. "
        "Try breaking it into smaller pieces."
    ),
}


def get_advice(bias_label: str) -> str:
    return ADVICE_FOR_EACH_BIAS.get(bias_label, "Take a moment and think it through carefully.")


# ------------------------------------------------------------------ #
#  Core analysis — brings everything together
# ------------------------------------------------------------------ #

def analyze_thought(user_input: str) -> tuple[str, str]:
    """
    Breaks your input into sentences, checks each one for patterns,
    then combines the results into a single, weighted conclusion.

    Returns (bias_label, explanation).
    """
    sentences = [s.strip() for s in user_input.split(".") if s.strip()]
    findings = []  # list of (label, explanation, confidence_score)

    for sentence in sentences:
        emotional_tone = measure_emotional_tone(sentence)

        # Step 1: Check rule-based patterns first
        rule_label, rule_note = check_for_obvious_patterns(sentence)

        # Positive sentiment overrides overgeneralization flagging
        if rule_label == "overgeneralization" and emotional_tone > 0:
            rule_label = None

        if rule_label:
            findings.append((rule_label, rule_note, 1.0))
            continue

        # Step 2: Let the ML model weigh in
        cleaned = clean_thought(sentence)
        features = vectorizer.transform([cleaned])
        prediction_probs = classifier.predict_proba(features)[0]
        top_confidence = max(prediction_probs)
        predicted_label = classifier.predict(features)[0]

        if top_confidence < 0.55:
            findings.append((
                "uncertainty",
                "The model isn't confident enough to call this one.",
                top_confidence
            ))
        else:
            findings.append((predicted_label, "Spotted by the model.", top_confidence))

    # Step 3: Tally up weighted scores per label
    label_scores: dict[str, float] = {}
    for label, _, confidence in findings:
        label_scores[label] = label_scores.get(label, 0) + confidence

    winning_label = max(label_scores, key=label_scores.get)
    winning_note = next(note for lbl, note, _ in findings if lbl == winning_label)

    # Step 4: If the overall tone is very neutral, call it balanced
    overall_tone = measure_emotional_tone(user_input)
    if abs(overall_tone) < 0.2:
        winning_label = "no_bias"
        winning_note = "Your tone is calm and neutral — that's a good sign."

    return winning_label, winning_note


# ------------------------------------------------------------------ #
#  Friendly labels for display
# ------------------------------------------------------------------ #

DISPLAY_NAMES = {
    "overgeneralization":  "Overgeneralization",
    "catastrophizing":     "Catastrophizing",
    "personalization":     "Personalization",
    "emotional_reasoning": "Emotional Reasoning",
    "overconfidence":      "Overconfidence",
    "no_bias":             "Balanced Thinking ✓",
    "uncertainty":         "Hard to Say",
}


# ------------------------------------------------------------------ #
#  UI — what you actually see
# ------------------------------------------------------------------ #

st.title("🧠 RationalMind")
st.markdown(
    "**Your thinking companion.** Share what's on your mind and I'll help you "
    "notice any patterns that might be making things feel harder than they are."
)

st.divider()

user_thought = st.text_area(
    label="What's on your mind?",
    placeholder=(
        "Write freely — a worry, a decision, something that's been bugging you. "
        "There's no wrong answer here."
    ),
    height=160,
)

if st.button("Help me think this through", type="primary"):

    if not user_thought.strip():
        st.warning("It looks like you haven't written anything yet. Take your time — whenever you're ready.")

    else:
        with st.spinner("Reading your thought carefully…"):
            detected_bias, explanation = analyze_thought(user_thought)
            advice = get_advice(detected_bias)
            display_name = DISPLAY_NAMES.get(detected_bias, detected_bias.replace("_", " ").title())

        # Results
        st.subheader("Here's what I noticed")

        col1, col2 = st.columns([1, 2])

        with col1:
            if detected_bias == "no_bias":
                st.success(f"**{display_name}**")
            elif detected_bias == "uncertainty":
                st.info(f"**{display_name}**")
            else:
                st.warning(f"**{display_name}**")

        with col2:
            st.markdown(f"*{explanation}*")

        st.markdown("---")

        st.markdown("### A gentle reframe")
        st.info(advice)

        st.markdown("---")
        st.caption(
            "This isn't a diagnosis — it's just a nudge. "
            "If your thoughts are really weighing on you, talking to someone you trust (or a professional) always helps more than any app can."
        )

# ------------------------------------------------------------------ #
#  Footer
# ------------------------------------------------------------------ #

st.divider()
st.caption("Built with care as part of the BYOP AI/ML Project · RationalMind is a thinking tool, not a therapist.")