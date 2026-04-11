from flask import Flask, render_template, request
import joblib
import numpy as np
import re
from urllib.parse import urlparse
import requests
import os
from dotenv import load_dotenv

# =========================
# LOAD ENV
# =========================
load_dotenv()
API_KEY = os.getenv("URLSCAN_API_KEY")

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
model = joblib.load(open("phishing_model.pkl", "rb"))
vectorizer = joblib.load(open("tfidf_vectorizer.pkl", "rb"))

# =========================
# KEYWORDS
# =========================
phishing_keywords = [
    "urgent", "verify", "login", "account", "bank",
    "password", "click", "limited", "suspended",
    "update", "confirm", "security", "alert",
    "immediately", "action required",
    "reset", "billing", "payment", "invoice",
    "policy", "violation", "evidence", "activity"
]

# =========================
trusted_services = [
    "google.com", "dropbox.com", "microsoft.com",
    "hutech.edu.vn", "edu.vn"
]

# =========================
# SANDBOX (FIX)
# =========================
def test_url_in_sandbox(url):
    if not API_KEY:
        print("❌ Missing API KEY")
        return None

    headers = {
        'API-Key': API_KEY,
        'Content-Type': 'application/json'
    }

    clean_url = url.strip().rstrip('.,)')

    data = {
        "url": clean_url,
        "visibility": "public"
    }

    try:
        response = requests.post(
            "https://urlscan.io/api/v1/scan/",
            headers=headers,
            json=data,
            timeout=15
        )

        if response.status_code == 200:
            result = response.json()
            return f"https://urlscan.io/result/{result.get('uuid')}/"
        else:
            print("API ERROR:", response.status_code)

    except Exception as e:
        print("Sandbox error:", e)

    return None

# =========================
# HOMOGLYPH
# =========================
def normalize_domain(domain):
    replacements = {
        "0": "o", "1": "l", "3": "e",
        "5": "s", "7": "t", "@": "a", "$": "s"
    }

    normalized = ""
    for c in domain:
        normalized += replacements.get(c.lower(), c.lower())

    return normalized.replace("rn", "m")

# =========================
# URL EXTRACT
# =========================
def extract_urls(text):
    urls = re.findall(r'https?://\S+|www\.\S+', text)

    anchor_patterns = [
        "click here", "verify your account",
        "login now", "reset password",
        "view evidence", "download"
    ]

    found = []
    for p in anchor_patterns:
        if p in text.lower():
            found.append(f"[ANCHOR] {p}")

    return urls + list(set(found))

# =========================
# URL ANALYZE (UPGRADE)
# =========================
def analyze_url(url):
    score = 0
    reasons = []

    if "[ANCHOR]" in url:
        return 3, ["Link ẩn dạng nút bấm"]

    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path
    query = parsed.query

    # Trusted nhưng query lạ
    if any(service in domain for service in trusted_services):
        if len(query) > 30:
            score += 2
            reasons.append("Domain uy tín nhưng query bất thường")
        else:
            return 0, ["Domain thuộc hệ thống uy tín"]

    normalized = normalize_domain(domain)
    fake_brands = ["paypal.com", "google.com", "microsoft.com", "amazon.com"]

    for brand in fake_brands:
        if brand not in domain and brand in normalized:
            score += 3
            reasons.append(f"Domain giả mạo giống {brand}")

    # RULE CŨ
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

    # 🔥 RULE MỚI
    if len(query) > 40:
        score += 2
        reasons.append("Query dài bất thường")

    if query.count("&") >= 2:
        score += 2
        reasons.append("Nhiều tham số")

    if url.count("?") > 1:
        score += 2
        reasons.append("Double query bất thường")

    if "%" in url:
        score += 1
        reasons.append("URL encoding")

    if "@" in url:
        score += 3
        reasons.append("Có @ (redirect giả mạo)")

    if len(path) > 50:
        score += 1
        reasons.append("Path dài bất thường")

    if any(ext in url for ext in [".exe", ".zip", ".rar", ".html"]):
        score += 2
        reasons.append("File đáng ngờ")

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
    domain = email.split("@")[-1]

    score = 0
    reasons = []

    if "0" in domain or "1" in domain:
        score += 1
        reasons.append("Domain có ký tự giả mạo")

    return email, reasons, score

# =========================
# FEATURE
# =========================
def extract_features(text):
    X = vectorizer.transform([text]).toarray()
    keyword_score = sum(1 for w in phishing_keywords if w in text.lower())
    extra = np.array([[len(text.split()), keyword_score]])
    return np.hstack((X, extra)), keyword_score

def explain_text(text):
    return [w for w in phishing_keywords if w in text.lower()]

# =========================
# RISK
# =========================
def calculate_risk(ai_prob, keyword_score, url_results, sender_score, text):
    risk = 0
    reasons = []

    if ai_prob > 0.8:
        risk += 4
        reasons.append("AI đánh giá rất nguy hiểm")

    if keyword_score >= 3:
        risk += 2
        reasons.append("Nhiều từ khóa đáng ngờ")

    for u in url_results:
        if u["score"] >= 3:
            risk += 3
        elif u["score"] >= 1:
            risk += 1

    if sender_score >= 3:
        risk += 3

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

        sandbox_link = None
        if "[ANCHOR]" not in url:
            sandbox_link = test_url_in_sandbox(url)

        status = "✅ An toàn"
        if score >= 3:
            status = "🚨 Nguy hiểm"
        elif score >= 1:
            status = "⚠️ Đáng ngờ"

        url_results.append({
            "url": url,
            "score": score,
            "status": status,
            "reasons": reasons,
            "sandbox": sandbox_link
        })

    sender_email, sender_reasons, sender_score = analyze_sender(text)

    risk_score, risk_reasons = calculate_risk(
        ai_prob, keyword_score, url_results, sender_score, text
    )

    prediction = 1 if risk_score >= 4 else 0
    prob = max(ai_prob, min(risk_score / 10, 1))

    return render_template(
        "index.html",
        prediction=prediction,
        probability=round(prob * 100, 2),
        urls=url_results,
        explain=explain_text(text),
        keyword_score=keyword_score,
        url_count=len(urls),
        sender=sender_email,
        sender_reasons=sender_reasons,
        risk_score=risk_score,
        risk_reasons=risk_reasons
    )

# =========================
if __name__ == "__main__":
    app.run(debug=True)