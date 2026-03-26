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
    "reset", "billing", "payment", "invoice",

    # NEW
    "policy", "violation", "evidence", "activity"
]

# =========================
trusted_services = [
    "google.com", "dropbox.com", "microsoft.com",
    "hutech.edu.vn", "edu.vn"
]

# =========================
# 🔥 HOMOGLYPH DETECTION
# =========================
def normalize_domain(domain):
    replacements = {
        "0": "o",
        "1": "l",
        "3": "e",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s"
    }

    normalized = ""
    for c in domain:
        normalized += replacements.get(c.lower(), c.lower())

    normalized = normalized.replace("rn", "m")

    return normalized

# =========================
# URL EXTRACT
# =========================
def extract_urls(text):
    urls = re.findall(r'https?://\S+|www\.\S+', text)

    anchor_patterns = [
        "click here",
        "verify your account",
        "login now",
        "reset password",
        "view evidence",
        "download"
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

    if "[ANCHOR]" in url:
        return 3, ["Link ẩn dạng nút bấm"]

    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    # 🔥 check trusted
    if any(service in domain for service in trusted_services):
        return 0, ["Domain thuộc hệ thống uy tín"]

    # 🔥 HOMOGLYPH CHECK
    normalized = normalize_domain(domain)
    fake_brands = ["paypal.com", "google.com", "microsoft.com", "amazon.com"]

    for brand in fake_brands:
        if brand not in domain and brand in normalized:
            score += 3
            reasons.append(f"Domain giả mạo giống {brand}")

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
# SENDER
# =========================
def analyze_sender(text):
    match = re.search(r'<([\w\.-]+@[\w\.-]+)>', text)

    if not match:
        match = re.search(r'[\w\.-]+@[\w\.-]+', text)

    if not match:
        return None, [], 0

    email = match.group(1) if "<" in match.group(0) else match.group(0)
    email = email.lower()
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

    # 🔥 HOMOGLYPH CHECK (SENDER)
    normalized = normalize_domain(domain)

    for brand, domains in brand_domains.items():
        for real_domain in domains:
            if real_domain not in domain and real_domain in normalized:
                score += 3
                reasons.append(f"Domain giả mạo giống {brand.upper()}")

    trusted_flat = [d for sub in brand_domains.values() for d in sub] + ["edu.vn", "hutech.edu.vn"]

    if not any(d in domain for d in trusted_flat):
        score += 2
        reasons.append("Domain người gửi không uy tín")

    if any(x in domain for x in ["0", "1", "l", "rn"]):
        score += 1
        reasons.append("Domain có dấu hiệu giả mạo ký tự")

    if len(domain) > 25:
        score += 1
        reasons.append("Domain dài bất thường")

    return email, reasons, score

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
# 🔥 RISK SCORING SYSTEM
# =========================
def calculate_risk(ai_prob, keyword_score, url_results, sender_score, text):
    risk = 0
    reasons = []

    if ai_prob > 0.8:
        risk += 4
        reasons.append("AI đánh giá rất nguy hiểm")
    elif ai_prob > 0.6:
        risk += 2

    if keyword_score >= 3:
        risk += 2
        reasons.append("Nhiều từ khóa đáng ngờ")

    for u in url_results:
        if u["score"] >= 3:
            risk += 3
            reasons.append("Link nguy hiểm")
        elif u["score"] >= 1:
            risk += 1

    if sender_score >= 3:
        risk += 3
        reasons.append("Người gửi giả mạo")
    elif sender_score >= 1:
        risk += 1

    if "{{" in text and "}}" in text:
        risk += 4
        reasons.append("Email template (phishing kit)")

    if any(x in text.lower() for x in ["violation", "evidence", "urgent action"]):
        risk += 2
        reasons.append("Dấu hiệu gây áp lực")

    if "edu.vn" in text.lower() or "university" in text.lower():
        risk -= 2

    return risk, reasons

# =========================
@app.route("/")
def home():
    return render_template("index.html")

# =========================
@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["email"]

    X_input, keyword_score = extract_features(text)
    ai_prob = model.predict_proba(X_input)[0][1]

    urls = extract_urls(text)
    url_results = []

    for url in urls:
        score, reasons = analyze_url(url)

        if score >= 3:
            status = "🚨 Nguy hiểm"
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

    sender_email, sender_reasons, sender_score = analyze_sender(text)

    risk_score, risk_reasons = calculate_risk(
        ai_prob, keyword_score, url_results, sender_score, text
    )

    if risk_score >= 7:
        prediction = 1
        warning = "🚨 Nguy cơ cao (phishing)"
    elif risk_score >= 4:
        prediction = 1
        warning = "⚠️ Email đáng ngờ"
    else:
        prediction = 0
        warning = None

    prob = max(ai_prob, min(risk_score / 10, 1))

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
        sender_reasons=sender_reasons,
        risk_score=risk_score,
        risk_reasons=risk_reasons
    )

# =========================
if __name__ == "__main__":
    app.run(debug=True)