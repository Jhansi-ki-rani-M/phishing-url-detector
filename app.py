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
# TITLE
# ============================================================

st.title("🔐 Phishing URL Detector")

st.write(
    "Enter a URL below to check whether it is "
    "legitimate or potentially phishing."
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

    X_train_extra = scaler.fit_transform(X_train_extra)
    X_test_extra = scaler.transform(X_test_extra)

    X_train_extra = csr_matrix(X_train_extra)
    X_test_extra = csr_matrix(X_test_extra)

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
    # Train Logistic Regression
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
    "🧠 Training machine learning model..."
):

    model, vectorizer, scaler, accuracy = train_model()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.sidebar.header("📊 Model Performance")

st.sidebar.metric(
    "Accuracy",
    f"{accuracy * 100:.2f}%"
)

st.sidebar.write(
    "Model: Logistic Regression"
)

st.sidebar.write(
    "Features: TF-IDF + URL characteristics"
)


# ============================================================
# URL INPUT
# ============================================================

url = st.text_input(
    "🌐 Enter URL",
    placeholder="https://example.com"
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
            "Please enter a URL."
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

            st.success(
                "🟢 LEGITIMATE URL"
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        else:

            confidence = probabilities[0] * 100

            st.error(
                "🔴 PHISHING URL"
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "⚠️ This tool is an ML-based prediction system and "
    "should not be treated as a guarantee of website safety."
)