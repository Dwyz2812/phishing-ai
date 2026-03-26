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
    "update", "confirm", "security", "alert",
    "immediately", "action required",
    "reset", "billing", "payment", "invoice"
]

# =========================
# TRUSTED SERVICES
# =========================
trusted_services = [
    "google.com", "dropbox.com", "microsoft.com"
]

# =========================
# URL EXTRACT
# =========================
def extract_urls(text):
    urls = re.findall(r'https?://\S+|www\.\S+', text)

    # 🔥 detect anchor text (KHÔNG trùng)
    anchor_patterns = [
        "click here",
        "verify your account",
        "login now",
        "reset password"
    ]

    found = []
    for p in anchor_patterns:
        if p in text.lower():
            found.append(f"[ANCHOR] {p}")

    return urls + list(set(found))


# =========================
# URL ANALYZE
# =========================
def analyze_url(url):
    score = 0
    reasons = []

    # 🔥 anchor text
    if "[ANCHOR]" in url:
        return 3, ["Link ẩn dạng nút bấm"]

    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    if any(char.isdigit() for char in domain):
        score += 2
        reasons.append("Domain chứa số")

    if any(word in domain for word in ["bank", "login", "verify"]):
        score += 2
        reasons.append("Domain chứa từ nhạy cảm")

    if domain.count('.') > 2:
        score += 1
        reasons.append("Nhiều subdomain")

    if len(domain) > 25:
        score += 1
        reasons.append("Domain dài bất thường")

    if not url.startswith("https"):
        score += 1
        reasons.append("Không HTTPS")

    return score, reasons


# =========================
# 🔥 SENDER FIX MẠNH
# =========================
def analyze_sender(text):
    # Bắt email chuẩn trước (ưu tiên trong <>)
    match = re.search(r'<([\w\.-]+@[\w\.-]+)>', text)

    if not match:
        # fallback: bắt email bình thường
        match = re.search(r'[\w\.-]+@[\w\.-]+', text)

    if not match:
        return None, []

    email = match.group(1).lower() if "<" in match.group(0) else match.group(0).lower()
    domain = email.split("@")[-1]

    reasons = []
    score = 0

    brand_domains = {
        "microsoft": ["outlook.com", "hotmail.com", "microsoft.com"],
        "google": ["gmail.com", "google.com"],
        "apple": ["apple.com", "icloud.com"],
        "paypal": ["paypal.com"],
        "amazon": ["amazon.com"]
    }

    text_lower = text.lower()

    for brand, domains in brand_domains.items():
        if brand in text_lower:
            if not any(d in domain for d in domains):
                score += 3
                reasons.append(f"Giả danh {brand.upper()}")

    trusted_flat = [d for sub in brand_domains.values() for d in sub]
    if not any(d in domain for d in trusted_flat):
        score += 2
        reasons.append("Domain người gửi không uy tín")

    if any(x in domain for x in ["0", "1", "l", "rn"]):
        score += 1
        reasons.append("Domain có dấu hiệu giả mạo ký tự")

    if len(domain) > 25:
        score += 1
        reasons.append("Domain dài bất thường")

    return email, reasons
# =========================
# FEATURE
# =========================
def extract_features(text):
    X = vectorizer.transform([text]).toarray()
    keyword_score = sum(1 for w in phishing_keywords if w in text.lower())
    extra = np.array([[len(text.split()), keyword_score]])
    return np.hstack((X, extra)), keyword_score


# =========================
def explain_text(text):
    return [w for w in phishing_keywords if w in text.lower()]


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
        score, reasons = analyze_url(url)

        if score >= 3:
            status = "🚨 Nguy hiểm"
            url_flag = True
        elif score >= 1:
            status = "⚠️ Đáng ngờ"
        else:
            status = "✅ An toàn"

        url_results.append({
            "url": url,
            "score": score,
            "status": status,
            "reasons": reasons
        })

    # ===== SENDER =====
    sender_email, sender_reasons = analyze_sender(text)
    sender_flag = len(sender_reasons) > 0

    # ===== LOGIC =====
    warning = None

    if url_flag:
        prediction = 1
        prob = max(prob, 0.85)
        warning = "⚠️ Email chứa link nguy hiểm"

    elif sender_flag:
        prediction = 1
        prob = max(prob, 0.8)
        warning = "⚠️ Người gửi giả mạo"

    elif keyword_score >= 2:
        prediction = 1
        prob = max(prob, 0.75)

    # ===== RETURN =====
    return render_template(
        "index.html",
        prediction=prediction,
        probability=round(prob * 100, 2),
        urls=url_results,
        explain=explain_text(text),
        keyword_score=keyword_score,
        url_count=len(urls),
        warning=warning,
        sender=sender_email,
        sender_reasons=sender_reasons
    )


if __name__ == "__main__":
    app.run(debug=True)