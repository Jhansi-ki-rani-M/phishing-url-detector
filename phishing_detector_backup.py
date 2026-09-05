import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load the real dataset
data = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")

# 2. Select URL and label
X = data["URL"]
y = data["label"]

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Dataset loaded!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# 4. Convert URLs into numerical features
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=100000
)

X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

print("URL features extracted!")


# 5. Train the machine learning model
model = LogisticRegression(max_iter=1000)

model.fit(X_train_features, y_train)

print("Model trained successfully!")


# 6. Test the model
y_pred = model.predict(X_test_features)


# 7. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)


# 8. Show detailed results
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 9. Show confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 10. Check a new URL
test_url = input("\nEnter a URL to check: ")

test_features = vectorizer.transform([test_url])

prediction = model.predict(test_features)[0]

if prediction == 1:
    print("Result: LEGITIMATE URL")
else:
    print("Result: PHISHING URL")