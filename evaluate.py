import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# LOAD
model = joblib.load(open("phishing_model.pkl", "rb"))
vectorizer = joblib.load(open("tfidf_vectorizer.pkl", "rb"))

df = pd.read_csv("phishing_email.csv")

print("Columns:", df.columns)

# ===== TEXT COLUMN =====
possible_cols = ["text", "email", "content", "message", "text_combined"]

col = None
for c in possible_cols:
    if c in df.columns:
        col = c
        break

if col is None:
    raise Exception("❌ Không tìm thấy cột nội dung email")

texts = df[col].fillna("")

# ===== LABEL =====
possible_labels = ["label", "target", "class", "spam"]

label_col = None
for c in possible_labels:
    if c in df.columns:
        label_col = c
        break

if label_col is None:
    raise Exception("❌ Không tìm thấy cột label")

y = df[label_col]

# ===== TF-IDF =====
X_tfidf = vectorizer.transform(texts).toarray()

# ===== EXTRA FEATURES =====
phishing_keywords = [
    "urgent","verify","login","account","bank",
    "password","click","limited","suspended",
    "update","confirm","security","alert",
    "immediately","action required",
    "reset","billing","payment","invoice",
    "policy","violation","evidence","activity"
]

keyword_scores = []
lengths = []

for text in texts:
    text_lower = text.lower()
    keyword_scores.append(sum(1 for w in phishing_keywords if w in text_lower))
    lengths.append(len(text.split()))

extra = np.array(list(zip(lengths, keyword_scores)))

# ===== COMBINE =====
X = np.hstack((X_tfidf, extra))

# ===== PREDICT =====
y_pred = model.predict(X)

# ===== RESULT =====
print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y, y_pred))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y, y_pred))