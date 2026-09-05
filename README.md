#  Phishing URL Detection Using Machine Learning

##  Project Overview

Phishing attacks are one of the most common cybersecurity threats, where attackers use deceptive URLs to trick users into visiting malicious websites or revealing sensitive information.

This project implements a machine-learning-based system that analyzes URLs and classifies them as either **Phishing** or **Legitimate**. The system combines TF-IDF character-level features with manually engineered URL characteristics and uses Logistic Regression for classification.

A Streamlit web application provides an interactive interface where users can enter a URL and receive a prediction along with a confidence score.

---

## Objective

The main objective of this project is to develop an automated system for detecting potentially phishing URLs using machine learning.

The project aims to:

- Identify patterns commonly associated with phishing URLs
- Extract meaningful features from URLs
- Train a binary classification model
- Predict whether a URL is phishing or legitimate
- Provide an easy-to-use web interface for URL detection

---

## Features

-  URL-based phishing detection
- Machine learning classification
- TF-IDF character-level feature extraction
- URL feature engineering
- Prediction confidence score
- Interactive Streamlit web application
- Model evaluation using accuracy, precision, recall and F1-score

---

## Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **SciPy**
- **Scikit-learn**
- **TF-IDF Vectorization**
- **Logistic Regression**
- **Streamlit**
- **Git & GitHub**

---

## Dataset

The project uses the **PhiUSIIL Phishing URL Dataset**.

### Dataset Statistics

| Category | Number of URLs |
|----------|---------------:|
| Legitimate | 134,850 |
| Phishing | 100,945 |
| **Total** | **235,795** |

### Labels

- `0` → Phishing
- `1` → Legitimate

The dataset is not included in this repository because of its size. It is required locally to train the model.

---

##  Methodology

The detection system follows the following machine learning pipeline:

```text
URL Dataset
     ↓
Data Preprocessing
     ↓
Train/Test Split
     ↓
TF-IDF Character Features
     +
URL Feature Engineering
     ↓
Feature Scaling
     ↓
Feature Combination
     ↓
Logistic Regression
     ↓
URL Classification
     ↓
Confidence Score

**## 🔍 Feature Engineering**

<details>
<summary>Click to expand Feature Engineering</summary>

In addition to TF-IDF features, the system extracts characteristics directly from each URL.

The following features are used:

1. URL length
2. Number of dots
3. Number of hyphens
4. Number of digits
5. Number of special characters
6. Number of subdomains
7. Presence of `@`
8. Presence of an IP address
9. HTTPS usage
10. Number of suspicious keywords
11. Number of `/` characters
12. Number of `?` characters

The system also checks for suspicious keywords such as:

```text
login
signin
verify
verification
secure
account
update
password
banking
confirm
authenticate
credential
payment
wallet
```

These features help the model identify structural and lexical patterns commonly found in suspicious URLs.

</details>

---

**## Machine Learning Model**

<details>
<summary>Click to expand Machine Learning Model</summary>

### Logistic Regression

Logistic Regression is used as the primary classification algorithm because the task is a binary classification problem:

```text
Phishing  ↔  Legitimate
```

The model combines:

* Character-level TF-IDF features
* Engineered URL characteristics

The TF-IDF vectorizer uses character n-grams ranging from **2 to 5 characters**.

</details>

---

**## Results**

<details>
<summary>Click to expand Results</summary>

The trained model achieved:

### **99.65% Accuracy**

on the held-out test dataset.

### Confusion Matrix

```text
                 Predicted
              Phishing  Legitimate

Actual Phishing    20026       163
Actual Legitimate     1      26969
```

### Classification Performance

| Class      | Precision | Recall | F1-Score |
| ---------- | --------: | -----: | -------: |
| Phishing   |      1.00 |   0.99 |     1.00 |
| Legitimate |      0.99 |   1.00 |     1.00 |

The results demonstrate that the model performs strongly on the test dataset.

</details>

---

**## Web Application**

The trained model is integrated into a **Streamlit** web application.

Users can enter a URL into the application and receive:

* Predicted class
* Legitimate/Phishing result
* Confidence score

Example:

```text
Input:
https://www.google.com

Prediction:
LEGITIMATE URL

Confidence:
99.92%
```

> Screenshots of the Streamlit application will be added to this section.

---

**## How to Run**

<details>
<summary>Click to expand How to Run</summary>

### 1. Clone the repository

```bash
git clone <repository-url>
cd phishing-url-detector
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Add the dataset

Place the required dataset inside:

```text
dataset/
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

</details>

---

**## Project Structure**

```text
phishing-url-detector/
│
├── app.py
├── phishing_detector.py
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
│
├── dataset/
│   └── PhiUSIIL_Phishing_URL_Dataset.csv
│
├── project files/
│
└── src/
```

> The dataset and virtual environment are excluded from GitHub using `.gitignore`.

---

**## Limitations**

<details>
<summary>Click to expand Limitations</summary>

Although the model achieves high accuracy on the test dataset, it should not be considered a complete real-world phishing protection system.

Some limitations include:

* The system primarily analyzes URL characteristics.
* It does not inspect the complete webpage content.
* New phishing techniques may not be represented in the training dataset.
* Dataset performance may differ from real-world performance.
* A high confidence score does not guarantee that a website is safe.

This project is intended as a machine-learning-based research and educational cybersecurity project.

</details>

---

**## Future Improvements**

<details>
<summary>Click to expand Future Improvements</summary>

Possible improvements include:

* Real-time URL reputation checking
* WHOIS and domain-age analysis
* DNS-based features
* SSL certificate analysis
* Website content analysis
* Integration with threat intelligence APIs
* Browser extension implementation
* Comparison with advanced machine learning models
* Real-time phishing alerts

</details>

---

**## Author**

**Jhansi**

B.Tech – Computer Science Engineering (Artificial Intelligence & Machine Learning)

Cybersecurity Honors


