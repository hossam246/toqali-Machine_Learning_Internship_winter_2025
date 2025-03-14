import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Load the trained Logistic Regression model
with open("logistic_regression_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Expected feature names (from model)
expected_features = [
    'number_of_adults', 'number_of_children', 'number_of_weekend_nights',
    'number_of_week_nights', 'car_parking_space', 'lead_time', 'repeated', 'p-c',
    'p-not-c', 'average_price', 'special_requests', 'type_of_meal_Meal Plan 2',
    'type_of_meal_Not Selected', 'room_type_Room_Type 2', 'room_type_Room_Type 3',
    'room_type_Room_Type 4', 'room_type_Room_Type 5', 'room_type_Room_Type 6',
    'room_type_Room_Type 7', 'market_segment_type_Complementary',
    'market_segment_type_Corporate', 'market_segment_type_Offline',
    'market_segment_type_Online'
]
# Initialize Flask app
app = Flask(__name__)

#Define Preprocessing Function Before Using It
def preprocess_input(data):
    """Convert user input into a format suitable for the model."""

    # Convert input dictionary to DataFrame
    df = pd.DataFrame([data])

    # One-hot encode categorical features
    categorical_features = ["type_of_meal", "room_type", "market_segment_type"]
    df = pd.get_dummies(df, columns=categorical_features)

    # Ensure all expected features are present (set missing columns to 0)
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0

    # Convert all columns to float (ensuring numeric types match training data)
    df = df.astype(float)

    # Reorder columns to match training data
    df = df[expected_features]

    return df
# Health Check Route (Check if Flask is running)
@app.route("/")
def home():
    return render_template("index.html")

# Define prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        print("Received data:", data)  # Debugging line

        if not data:
            return jsonify({"error": "No data received"}), 400

        processed_data = preprocess_input(data)
        print("Processed Data:\n", processed_data)  # Debugging line

        prediction = model.predict(processed_data)
        result = "Not Canceled" if prediction[0] == "Not_Canceled" else "Canceled"
        
        return jsonify({"prediction": result})

    except Exception as e:
        print("Error:", str(e))  # Debugging line
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)  # Allow external connections