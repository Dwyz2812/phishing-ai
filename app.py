from flask import Flask, render_template, request
import pickle
import numpy as np
import re
from urllib.parse import urlparse

app = Flask(__name__)

# LOAD MODEL
model = pickle.load(open("phishing_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# =========================
# KEYWORDS
# =========================
phishing_keywords = [
    "urgent", "verify", "login", "account", "bank",
    "password", "click", "limited", "suspended",
    "update", "confirm", "security", "alert","document", "link", "attachment",
    "immediately", "action required", "unauthorized",
    "suspicious", "locked", "disable", "verify now",
    "reset", "billing", "payment", "invoice",
    "prize", "winner", "claim", "free",
    "gift", "bonus", "offer", "reward",
    "deadline", "expire", "final notice"
]

# =========================
# TRUSTED SERVICES 
# =========================
trusted_services = [
    "docs.google.com",
    "drive.google.com",
    "dropbox.com"
]

# =========================
# URL FUNCTIONS
# =========================
def extract_urls(text):
    return re.findall(r'https?://\S+|www\.\S+', text)

def analyze_url(url):
    score = 0
    domain = urlparse(url).netloc

    if "-" in domain: score += 1
    if len(domain) > 25: score += 1
    if domain.count('.') > 2: score += 1
    if any(char.isdigit() for char in domain): score += 1
    if not url.startswith("https"): score += 1

    if any(service in domain for service in trusted_services):
        score += 1

    return score

# =========================
# FEATURE
# =========================
def extract_features(text):
    X_tfidf = vectorizer.transform([text]).toarray()

    word_count = len(text.split())
    keyword_score = sum(1 for w in phishing_keywords if w in text.lower())

    extra = np.array([[word_count, keyword_score]])

    return np.hstack((X_tfidf, extra)), keyword_score

# =========================
# EXPLAIN
# =========================
def explain_text(text):
    return [w for w in phishing_keywords if w in text.lower()]

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["email"]

    # ===== AI =====
    X_input, keyword_score = extract_features(text)
    ai_prob = model.predict_proba(X_input)[0][1]

    prediction = 1 if ai_prob >= 0.6 else 0
    prob = ai_prob

    # ===== URL =====
    urls = extract_urls(text)
    url_results = []
    url_flag = False

    for url in urls:
        score = analyze_url(url)

        if any(service in url for service in trusted_services):
            status = "⚠️ Cần kiểm tra thêm (dịch vụ trung gian)"
            url_flag = True

        elif score >= 2:
            status = "⚠️ Đáng ngờ"
            url_flag = True

        else:
            status = "✅ An toàn"

        url_results.append((url, score, status))

    # ===== KEYWORD RULE =====
    keyword_flag = keyword_score >= 2

    # ===== COMBINE LOGIC (🔥 FIX QUAN TRỌNG) =====
    warning_message = None

    if url_flag:
        prediction = 1
        prob = max(prob, 0.85)

    elif keyword_flag:
        prediction = 1
        prob = max(prob, 0.75)

    # 🔥 NEW: nội dung nguy hiểm dù link an toàn
    elif ai_prob > 0.7:
        prediction = 1
        prob = ai_prob
        warning_message = "⚠️ Nội dung email có dấu hiệu lừa đảo mặc dù link có vẻ an toàn"

    # ===== EXPLAIN =====
    explain_words = explain_text(text)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=round(prob * 100, 2),
        urls=url_results,
        explain=explain_words,
        keyword_score=keyword_score,
        url_count=len(urls),
        warning=warning_message
    )

# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)