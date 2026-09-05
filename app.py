import streamlit as st
import pandas as pd
import numpy as np
import re

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🔐",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #777;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .safe-box {
        border: 2px solid #21a366;
        background-color: rgba(33, 163, 102, 0.08);
    }

    .danger-box {
        border: 2px solid #d9534f;
        background-color: rgba(217, 83, 79, 0.08);
    }

    .result-title {
        font-size: 28px;
        font-weight: 700;
    }

    .info-card {
        padding: 18px;
        border-radius: 10px;
        background-color: rgba(128, 128, 128, 0.08);
        margin-top: 15px;
    }

    .footer {
        text-align: center;
        color: #777;
        font-size: 13px;
        margin-top: 30px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔐 Phishing URL Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Cybersecurity Tool'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Analyze a URL and determine whether it is potentially "
    "phishing or legitimate using machine learning."
)

st.divider()


# ============================================================
# EXTRACT URL FEATURES
# ============================================================

def extract_features(urls):

    features = []

    suspicious_words = [
        "login",
        "signin",
        "verify",
        "verification",
        "secure",
        "account",
        "update",
        "password",
        "banking",
        "confirm",
        "authenticate",
        "credential",
        "payment",
        "wallet"
    ]

    for url in urls:

        url_lower = url.lower()

        url_length = len(url)

        dot_count = url.count(".")

        hyphen_count = url.count("-")

        digit_count = sum(
            char.isdigit() for char in url
        )

        special_count = sum(
            not char.isalnum() for char in url
        )

        subdomain_count = max(
            url_lower.count(".") - 1,
            0
        )

        has_at = int("@" in url)

        has_ip = int(
            bool(
                re.search(
                    r"https?://(?:\d{1,3}\.){3}\d{1,3}",
                    url_lower
                )
            )
        )

        has_https = int(
            url_lower.startswith("https://")
        )

        suspicious_word_count = sum(
            word in url_lower
            for word in suspicious_words
        )

        slash_count = url.count("/")

        question_count = url.count("?")

        features.append([
            url_length,
            dot_count,
            hyphen_count,
            digit_count,
            special_count,
            subdomain_count,
            has_at,
            has_ip,
            has_https,
            suspicious_word_count,
            slash_count,
            question_count
        ])

    return np.array(features)


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model():

    data = pd.read_csv(
        "dataset/PhiUSIIL_Phishing_URL_Dataset.csv"
    )

    data["URL"] = data["URL"].fillna("").astype(str)

    data["label"] = data["label"].astype(int)

    X = data["URL"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # TF-IDF
    # -----------------------------

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=100000
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)

    X_test_tfidf = vectorizer.transform(X_test)

    # -----------------------------
    # Additional URL features
    # -----------------------------

    X_train_extra = extract_features(X_train)

    X_test_extra = extract_features(X_test)

    scaler = StandardScaler()

    X_train_extra = scaler.fit_transform(
        X_train_extra
    )

    X_test_extra = scaler.transform(
        X_test_extra
    )

    X_train_extra = csr_matrix(
        X_train_extra
    )

    X_test_extra = csr_matrix(
        X_test_extra
    )

    # -----------------------------
    # Combine features
    # -----------------------------

    X_train_final = hstack([
        X_train_tfidf,
        X_train_extra
    ])

    X_test_final = hstack([
        X_test_tfidf,
        X_test_extra
    ])

    # -----------------------------
    # Logistic Regression
    # -----------------------------

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train_final,
        y_train
    )

    # -----------------------------
    # Accuracy
    # -----------------------------

    predictions = model.predict(
        X_test_final
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return (
        model,
        vectorizer,
        scaler,
        accuracy
    )


# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner(
    "🧠 Preparing machine learning model..."
):

    model, vectorizer, scaler, accuracy = train_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📊 Model Information")

    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    st.write("**Algorithm:** Logistic Regression")

    st.write(
        "**Features:** TF-IDF + URL characteristics"
    )

    st.divider()

    st.subheader("🔍 Features Used")

    st.write(
        "• URL length\n"
        "• Dot count\n"
        "• Hyphen count\n"
        "• Digit count\n"
        "• Special characters\n"
        "• Subdomain count\n"
        "• IP address detection\n"
        "• HTTPS detection\n"
        "• Suspicious keywords\n"
        "• Slash count\n"
        "• Query parameters"
    )

    st.divider()

    st.caption(
        "This project is intended for educational "
        "and cybersecurity research purposes."
    )


# ============================================================
# URL INPUT
# ============================================================

st.subheader("🌐 Analyze a URL")

url = st.text_input(
    "Enter the URL you want to check",
    placeholder="https://example.com",
    label_visibility="visible"
)


# ============================================================
# CHECK URL
# ============================================================

if st.button(
    "🔍 Check URL",
    use_container_width=True
):

    if not url.strip():

        st.warning(
            "⚠️ Please enter a URL before checking."
        )

    else:

        # -----------------------------
        # TF-IDF features
        # -----------------------------

        url_tfidf = vectorizer.transform([
            url
        ])

        # -----------------------------
        # URL features
        # -----------------------------

        url_extra = extract_features([
            url
        ])

        url_extra = scaler.transform(
            url_extra
        )

        url_extra = csr_matrix(
            url_extra
        )

        # -----------------------------
        # Combine features
        # -----------------------------

        url_final = hstack([
            url_tfidf,
            url_extra
        ])

        # -----------------------------
        # Prediction
        # -----------------------------

        prediction = model.predict(
            url_final
        )[0]

        probabilities = model.predict_proba(
            url_final
        )[0]

        # -----------------------------
        # Display result
        # -----------------------------

        st.divider()

        if prediction == 1:

            confidence = probabilities[1] * 100

            st.markdown(
                f"""
                <div class="result-box safe-box">
                    <div class="result-title">
                        🟢 LEGITIMATE URL
                    </div>
                    <p>
                        The model classified this URL
                        as legitimate.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

        else:

            confidence = probabilities[0] * 100

            st.markdown(
                f"""
                <div class="result-box danger-box">
                    <div class="result-title">
                        🔴 PHISHING URL
                    </div>
                    <p>
                        The model detected patterns
                        commonly associated with phishing.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.warning(
                "⚠️ Avoid entering passwords, payment "
                "details, or other sensitive information "
                "on suspicious websites."
            )

# ============================================================
# URL ANALYSIS
# ============================================================

if url.strip():

    st.subheader("🔎 URL Analysis")

    analysis_features = extract_features([url])[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "URL Length",
            int(analysis_features[0])
        )

        st.metric(
            "Dots",
            int(analysis_features[1])
        )

    with col2:
        st.metric(
            "Digits",
            int(analysis_features[3])
        )

        st.metric(
            "Subdomains",
            int(analysis_features[5])
        )

    with col3:
        st.metric(
            "Suspicious Keywords",
            int(analysis_features[9])
        )

        if analysis_features[8] == 1:
            st.success("🔒 HTTPS Detected")
        else:
            st.warning("⚠️ HTTPS Not Detected")

    if analysis_features[6] == 1:
        st.warning("⚠️ '@' symbol detected in URL")

    if analysis_features[7] == 1:
        st.warning("⚠️ IP address detected in URL")
        
# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

with st.expander("🧠 How does this detector work?"):

    st.write(
        "The system uses machine learning to analyze "
        "patterns within URLs."
    )

    st.write(
        "**Step 1 — TF-IDF:** "
        "Character-level TF-IDF extracts patterns from "
        "the URL text."
    )

    st.write(
        "**Step 2 — Feature Engineering:** "
        "The system calculates characteristics such as "
        "URL length, number of dots, digits, subdomains, "
        "suspicious keywords, and HTTPS usage."
    )

    st.write(
        "**Step 3 — Classification:** "
        "The extracted features are combined and passed "
        "to a Logistic Regression classifier."
    )

    st.write(
        "**Step 4 — Prediction:** "
        "The model predicts whether the URL is "
        "Phishing or Legitimate and provides a "
        "confidence score."
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "⚠️ This tool provides an ML-based prediction and "
    "is not a guarantee of website safety. "
    "Always verify suspicious links using trusted "
    "security resources."
)

st.caption(
    "🔐 Phishing URL Detector • Machine Learning + Cybersecurity"
)