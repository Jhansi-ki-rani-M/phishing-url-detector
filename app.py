import streamlit as st
from phishing_detector import predict_url, extract_features


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

st.caption("Machine Learning Based Cybersecurity Tool")

st.write(
    "Enter a URL below to analyze whether it is likely to be "
    "a phishing or legitimate website."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🧠 Model Information")

    st.write("**Model:** Logistic Regression")

    st.write("**Accuracy:** 99.65%")

    st.write("**Features:** TF-IDF + URL Characteristics")

    st.write("**Dataset:** 235K+ URLs")

    st.divider()

    st.subheader("🔎 URL Features")

    st.write("• URL Length")
    st.write("• Number of Dots")
    st.write("• Number of Hyphens")
    st.write("• Number of Digits")
    st.write("• Special Characters")
    st.write("• Subdomains")
    st.write("• @ Symbol")
    st.write("• IP Address")
    st.write("• HTTPS")
    st.write("• Suspicious Keywords")
    st.write("• Number of Slashes")
    st.write("• Question Marks")


# ============================================================
# URL INPUT
# ============================================================

url = st.text_input(
    "🌐 Enter URL",
    placeholder="Example: https://www.google.com"
)


# ============================================================
# CHECK URL BUTTON
# ============================================================

if st.button(
    "🔍 Check URL",
    use_container_width=True
):

    if not url.strip():

        st.warning("⚠️ Please enter a URL first.")

    else:

        # ====================================================
        # PREDICTION
        # ====================================================

        prediction, confidence = predict_url(url)

        confidence_percentage = confidence * 100


        # ====================================================
        # RESULT
        # ====================================================

        if prediction == 1:

            st.success(
                "### ✅ LEGITIMATE URL\n\n"
                "This URL is classified as legitimate "
                "by the machine learning model."
            )

        else:

            st.error(
                "### 🚨 PHISHING URL\n\n"
                "This URL shows characteristics associated "
                "with phishing websites."
            )

            st.warning(
                "⚠️ Do not enter passwords, banking information, "
                "or other sensitive information on suspicious websites."
            )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        st.subheader("🤖 Prediction Confidence")

        st.progress(
            min(confidence, 1.0)
        )

        st.metric(
            "Model Confidence",
            f"{confidence_percentage:.2f}%"
        )


        # ====================================================
        # URL ANALYSIS
        # ====================================================

        st.divider()

        st.subheader("🔎 URL Analysis")

        analysis_features = extract_features([url])[0]

        col1, col2, col3 = st.columns(3)


        # ----------------------------------------------------
        # COLUMN 1
        # ----------------------------------------------------

        with col1:

            st.metric(
                "URL Length",
                int(analysis_features[0])
            )

            st.metric(
                "Dots",
                int(analysis_features[1])
            )


        # ----------------------------------------------------
        # COLUMN 2
        # ----------------------------------------------------

        with col2:

            st.metric(
                "Digits",
                int(analysis_features[3])
            )

            st.metric(
                "Subdomains",
                int(analysis_features[5])
            )


        # ----------------------------------------------------
        # COLUMN 3
        # ----------------------------------------------------

        with col3:

            st.metric(
                "Suspicious Keywords",
                int(analysis_features[9])
            )

            if analysis_features[8] == 1:

                st.success("🔒 HTTPS Detected")

            else:

                st.warning("⚠️ HTTPS Not Detected")


        # ====================================================
        # ADDITIONAL URL WARNINGS
        # ====================================================

        if analysis_features[6] == 1:

            st.warning(
                "⚠️ '@' symbol detected in URL"
            )


        if analysis_features[7] == 1:

            st.warning(
                "⚠️ IP address detected in URL"
            )


        # ====================================================
        # HOW THE DETECTOR WORKS
        # ====================================================

        st.divider()

        with st.expander("🧠 How does this detector work?"):

            st.write(
                """
                **Step 1 — URL Input**

                The user enters a website URL into the application.
                """
            )

            st.write(
                """
                **Step 2 — Feature Extraction**

                The system extracts characteristics from the URL,
                including URL length, dots, digits, subdomains,
                suspicious keywords, HTTPS usage, special
                characters, and other URL patterns.
                """
            )

            st.write(
                """
                **Step 3 — Feature Processing**

                Character-level TF-IDF converts the URL text into
                numerical features. These are combined with the
                engineered URL characteristics.
                """
            )

            st.write(
                """
                **Step 4 — Machine Learning**

                The combined features are given to a Logistic
                Regression machine learning model.
                """
            )

            st.write(
                """
                **Step 5 — Prediction**

                The model classifies the URL as either:

                🟢 Legitimate

                🔴 Phishing

                The application also displays the model's
                confidence for the prediction.
                """)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "⚠️ This tool is developed for educational and research "
    "purposes. Model predictions are not a guarantee of website safety."
)