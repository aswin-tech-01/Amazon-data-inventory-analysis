from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("inventory_model.pkl")

@app.route("/")
def index():
    return {"message": "📦 Inventory Data Analysis API is running"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        stock_level = data.get("stock_level", 0)
        price = data.get("price", 0)
        day_of_week = data.get("day_of_week", 1)

        # Prepare features
        features = np.array([[stock_level, price, day_of_week]])
        predicted_demand = model.predict(features)[0]

        return jsonify({"predicted_demand": round(float(predicted_demand), 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
