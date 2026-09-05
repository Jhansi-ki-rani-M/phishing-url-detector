import re

import pandas as pd
import numpy as np

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv(
    "dataset/PhiUSIIL_Phishing_URL_Dataset.csv"
)

data["URL"] = data["URL"].fillna("").astype(str)
data["label"] = data["label"].astype(int)

X = data["URL"]
y = data["label"]

print("Dataset loaded!")
print("Legitimate URLs:", (y == 1).sum())
print("Phishing URLs:", (y == 0).sum())


# ============================================================
# 2. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 3. TF-IDF FEATURES
# ============================================================

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=100000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF features extracted!")


# ============================================================
# 4. EXTRACT URL-BASED FEATURES
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

        # URL length
        url_length = len(url)

        # Number of dots
        dot_count = url.count(".")

        # Number of hyphens
        hyphen_count = url.count("-")

        # Number of digits
        digit_count = sum(
            char.isdigit()
            for char in url
        )

        # Number of special characters
        special_count = sum(
            not char.isalnum()
            for char in url
        )

        # Number of subdomains
        subdomain_count = max(
            url_lower.count(".") - 1,
            0
        )

        # Contains @
        has_at = int("@" in url)

        # Contains IP address
        has_ip = int(
            bool(
                re.search(
                    r"https?://(?:\d{1,3}\.){3}\d{1,3}",
                    url_lower
                )
            )
        )

        # HTTPS
        has_https = int(
            url_lower.startswith("https://")
        )

        # Suspicious words
        suspicious_word_count = sum(
            word in url_lower
            for word in suspicious_words
        )

        # Number of slash characters
        slash_count = url.count("/")

        # Number of question marks
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


X_train_extra = extract_features(X_train)
X_test_extra = extract_features(X_test)

print("Additional URL features extracted!")


# ============================================================
# 5. SCALE NUMERICAL FEATURES
# ============================================================

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


# ============================================================
# 6. COMBINE TF-IDF + URL FEATURES
# ============================================================

X_train_final = hstack([
    X_train_tfidf,
    X_train_extra
])

X_test_final = hstack([
    X_test_tfidf,
    X_test_extra
])

print("All features combined!")


# ============================================================
# 7. TRAIN MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_final,
    y_train
)

print("Model trained successfully!")


# ============================================================
# 8. TEST MODEL
# ============================================================

y_pred = model.predict(
    X_test_final
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n===================================")
print("MODEL RESULTS")
print("===================================")

print(
    "\nModel Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "PHISHING",
            "LEGITIMATE"
        ]
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 9. PREDICT A NEW URL
# ============================================================

def predict_url(url):

    # Convert URL into TF-IDF features
    test_tfidf = vectorizer.transform(
        [url]
    )

    # Extract URL characteristics
    test_extra = extract_features(
        [url]
    )

    # Scale URL characteristics
    test_extra = scaler.transform(
        test_extra
    )

    test_extra = csr_matrix(
        test_extra
    )

    # Combine features
    test_final = hstack([
        test_tfidf,
        test_extra
    ])

    # Prediction
    prediction = model.predict(
        test_final
    )[0]

    # Prediction probabilities
    probability = model.predict_proba(
        test_final
    )[0]

    # Get confidence
    if prediction == 1:

        confidence = probability[1]

    else:

        confidence = probability[0]

    return prediction, confidence


# ============================================================
# 10. COMMAND-LINE TESTING
# ============================================================
# This section runs ONLY when this file is executed directly.
# It will NOT run when Streamlit imports this file.

if __name__ == "__main__":

    print("\nChoose an option:")
    print("1. Enter a URL manually")
    print("2. Test a phishing URL from the dataset")

    choice = input(
        "\nEnter your choice (1 or 2): "
    )

    if choice == "1":

        test_url = input(
            "\nEnter a URL to check: "
        )

    elif choice == "2":

        test_url = data[
            data["label"] == 0
        ]["URL"].iloc[0]

        print(
            "\nTesting a phishing URL from the dataset:"
        )

        print(test_url)

    else:

        print("\nInvalid choice.")
        exit()

    prediction, confidence = predict_url(
        test_url
    )

    print("\n===================================")

    if prediction == 1:

        print("Result: LEGITIMATE URL")

    else:

        print("Result: PHISHING URL")

    print(
        "Confidence:",
        round(confidence * 100, 2),
        "%"
    )

    print("===================================")