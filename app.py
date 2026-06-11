# app.py
# The prediction web service. Loads the trained model and serves
# predictions over HTTP on port 8000.
# Run with: python app.py

from datetime import datetime, timezone
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "ml-prediction-service"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if data is None or "products" not in data:
        return jsonify({"message": "Request body must contain a products list"}), 400

    products = data["products"]
    today = datetime.now(timezone.utc)

    results = []

    for product in products:
        if "id" not in product or "expiryDate" not in product:
            return jsonify({"message": "Each product needs an id and an expiryDate"}), 400
        # date format example: "2024-07-01T00:00:00Z" (ISO 8601 with UTC timezone)
        expiry = datetime.fromisoformat(product["expiryDate"].replace("Z", "+00:00"))

        days_to_expiry = (expiry - today).days

        # Ask the trained model. predict()
        risk = model.predict([[days_to_expiry]])[0]

        results.append({
            "id": product["id"],
            "daysToExpiry": days_to_expiry,
            "riskStatus": risk,
        })

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(port=8000, debug=True)