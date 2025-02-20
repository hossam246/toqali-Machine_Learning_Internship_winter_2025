from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")  # Serve the UI


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data as JSON
        data = request.get_json()

        # Convert to DataFrame
        df = pd.DataFrame([data])
        # Define the exact feature order used during training
        expected_features = [
            "number of adults",
            "number of children",
            "number of weekend nights",
            "number of week nights",
            "type of meal",
            "car parking space",
            "room type",
            "lead time",
            "market segment type",
            "repeated",
            "P-C",
            "P-not-C",
            "average price",
            "special requests",
            "adults_children",
            "weekend_weeknights",
            "lead_time_squared",
            "average_price_squared",
            "total_nights",
            "total_people",
        ]

        # Ensure all required features are in the input data
        for col in expected_features:
            if col not in df:
                df[col] = 0  # Fill missing features with 0

        # Apply the same feature transformations as in training
        df["adults_children"] = df["number of adults"] + df["number of children"]
        df["weekend_weeknights"] = (
            df["number of weekend nights"] + df["number of week nights"]
        )
        df["lead_time_squared"] = df["lead time"] ** 2
        df["average_price_squared"] = df["average price"] ** 2
        df["total_nights"] = (
            df["number of weekend nights"] + df["number of week nights"]
        )
        df["total_people"] = df["number of adults"] + df["number of children"]

        # Ensure columns are in the correct order
        df = df[expected_features]
        # print("Columns in incoming data:", df.columns.tolist())
        # print("Expected features:", expected_features)
        # # Apply the scaler
        # df = scaler.transform(df)
        # Select only the columns that scaler.pkl was trained on
        scaler_features = [
            "type of meal",
            "car parking space",
            "room type",
            "repeated",
            "P-C",
            "P-not-C",
            "special requests",
        ]
        df = df[scaler_features]  # Keep only these columns before transformation
        df = scaler.transform(df)  # Now apply scaling safely

        # Make prediction
        prediction = model.predict(df)
        prediction_label = "Canceled" if prediction[0] == 1 else "Not Canceled"

        return jsonify({"prediction": prediction_label})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
