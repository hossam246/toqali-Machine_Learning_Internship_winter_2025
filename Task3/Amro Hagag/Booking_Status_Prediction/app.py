from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model, scaler, and encoders.
# The scaler was fitted on 12 features (all except "month" and "day").
model = joblib.load("models/knn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")  # Expected keys: "type of meal", "room type", "market segment type"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Retrieve all 14 inputs from the form.
        # (All inputs come as strings; convert as needed.)
        # For the 12 scaled features:
        type_of_meal = request.form["type_of_meal"]          # e.g., "Meal Plan 2"
        car_parking_space = request.form["car_parking_space"]   # "Yes" or "No"
        room_type = request.form["room_type"]                   # e.g., "Room_Type 1"
        lead_time = float(request.form["lead_time"])
        market_segment_type = request.form["market_segment_type"]  # e.g., "Corporate"
        repeated = request.form["repeated"]                     # "Yes" or "No"
        pc = float(request.form["pc"])
        p_not_c = float(request.form["p_not_c"])
        average_price = float(request.form["average_price"])
        special_requests = float(request.form["special_requests"])
        total_family_members = float(request.form["total_family_members"])
        total_number_of_days = float(request.form["total_number_of_days"])
        
        # For the 2 unscaled features:
        month = float(request.form["month"])
        day = float(request.form["day"])
        
        # 2. Process the 12 features that were scaled.
        # Apply label encoding to the categorical fields.
        type_of_meal_val = encoders["type of meal"].transform([type_of_meal])[0]
        room_type_val = encoders["room type"].transform([room_type])[0]
        market_segment_type_val = encoders["market segment type"].transform([market_segment_type])[0]
        
        # Convert Yes/No fields to numeric.
        car_parking_space_val = 1 if car_parking_space == "Yes" else 0
        repeated_val = 1 if repeated == "Yes" else 0
        
        # Build the array for the 12 features that were scaled.
        # The order (as used for scaling) is:
        # ['type of meal', 'car parking space', 'room type', 'lead time',
        #  'market segment type', 'repeated', 'P-C', 'P-not-C', 'average price',
        #  'special requests', 'total family members', 'total number of days']
        scaled_input = np.array([
            type_of_meal_val,
            car_parking_space_val,
            room_type_val,
            lead_time,
            market_segment_type_val,
            repeated_val,
            pc,
            p_not_c,
            average_price,
            special_requests,
            total_family_members,
            total_number_of_days
        ]).reshape(1, -1)
        
        # Scale these 12 features using the saved scaler.
        scaled_input_transformed = scaler.transform(scaled_input)
        
        # 3. Now, reconstruct the final 14-feature input in the order:
        # ['type of meal', 'car parking space', 'room type', 'lead time',
        #  'market segment type', 'repeated', 'P-C', 'P-not-C', 'average price',
        #  'special requests', 'month', 'day', 'total family members', 'total number of days']
        # We need to insert the unscaled "month" and "day" between index 9 and index 10.
        
        # Extract the first 10 scaled features (indices 0 to 9)
        first_part = scaled_input_transformed[:, :10]  # shape: (1, 10)
        # Extract the last 2 scaled features (indices 10 and 11)
        last_part = scaled_input_transformed[:, 10:]   # shape: (1, 2)
        # Prepare the unscaled features (month and day)
        unscaled_part = np.array([month, day]).reshape(1, -1)  # shape: (1, 2)
        
        # Combine them in order: first 10 scaled, then unscaled month/day, then last 2 scaled.
        final_features = np.hstack([first_part, unscaled_part, last_part])
        
        # Check the final shape (should be (1, 14))
        # print("Final features shape:", final_features.shape)
        
        # 4. Predict using the model.
        prediction = model.predict(final_features)[0]
        status = "Not Canceled" if prediction == 1 else "Canceled"
        
        # 5. Return the result.
        return render_template("index.html", prediction_text=f"Predicted Booking Status: {status}")
    
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)