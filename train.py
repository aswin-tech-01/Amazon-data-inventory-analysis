import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Example dataset (replace with your real inventory dataset)
# Features: stock_level, price, day_of_week
# Target: demand (sales quantity)
data = {
    "stock_level": [50, 30, 60, 80, 20, 90],
    "price": [200, 180, 220, 210, 190, 230],
    "day_of_week": [1, 2, 3, 4, 5, 6],   # 1 = Monday, 7 = Sunday
    "demand": [20, 25, 35, 40, 15, 50]
}

df = pd.DataFrame(data)

# Features & Target
X = df[["stock_level", "price", "day_of_week"]]
y = df["demand"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "inventory_model.pkl")
print("✅ Model trained and saved as inventory_model.pkl")
