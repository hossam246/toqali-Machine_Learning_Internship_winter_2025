import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model_data = pickle.load(open("model.pkl", "rb"))
model = model_data["model"]
feature_names = model_data["feature_names"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form.to_dict()

        processed_data = []
        for feature in feature_names:
            if feature in form_data:
                value = form_data[feature]
                try:
                    processed_data.append(float(value))  # Convert to float
                except ValueError:
                    return render_template("index.html", prediction_text="Invalid input: Ensure all inputs are numerical.")

            else:
                processed_data.append(0)  

        final_features = pd.DataFrame([processed_data], columns=feature_names)

        prediction = model.predict(final_features)[0]

        output = "Canceled" if prediction == 1 else "Not Canceled"
        return render_template("index.html", prediction_text=f"Hotel Booking Status Prediction: {output}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
