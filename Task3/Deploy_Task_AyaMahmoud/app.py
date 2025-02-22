from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd  
import calendar
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

FLASK_API_URL = "http://127.0.0.1:3000/predict" 

with open("hotel_model.pkl", 'rb') as file:
    lr = pickle.load(file)

with open('booking_status_le.pkl', 'rb') as f:
    booking_status_le = pickle.load(f)
    
with open('date_of_reservation_le.pkl', 'rb') as f:
    date_of_reservation_le = pickle.load(f)
    
with open('one_hot_columns.pkl', 'rb') as f:  # Load the saved columns
    one_hot_columns = pickle.load(f)

formats = ["%m/%d/%Y", "%Y/%m/%d","%m-%d-%Y", "%Y-%m-%d"]
def extract_month(date_str):
    for fmt in formats:
        try:
            date_object = datetime.strptime(date_str, fmt)
            return calendar.month_name[date_object.month]  # Convert month number to name
        except ValueError:
            pass
    return None

@app.route('/')
def home():
    return render_template('interface.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Predict endpoint hit!")

        feature_names = ['number_of_adults', 'number_of_children', 'number_of_weekend_nights',
                         'number_of_week_nights', 'type_of_meal', 'car_parking_space',
                         'room_type', 'lead_time', 'market_segment_type', 'repeated', 
                         'P-C', 'P-not-C', 'average_price_', 'special_requests', 'date_of_reservation']

        # Convert received features list into a dictionary
        input_dict = dict(zip(feature_names, request.json['features']))

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])
        input_df['date_of_reservation'] = input_df['date_of_reservation'].apply(extract_month)
        input_df['date_of_reservation'] = date_of_reservation_le.transform([input_df['date_of_reservation'][0]])[0]  
        # Convert number columns to int or float
        int_cols = ['number_of_adults', 'number_of_children', 'number_of_weekend_nights', 
                    'number_of_week_nights', 'car_parking_space', 'lead_time', 
                    'repeated', 'P-C', 'P-not-C', 'special_requests']
        
        float_cols = ['average_price_']

        for col in int_cols:
            input_df[col] = input_df[col].astype(int)

        for col in float_cols:
            input_df[col] = input_df[col].astype(float)

        # Handle categorical encoding
        categorical_columns = ['type_of_meal', 'room_type', 'market_segment_type']
        input_df = pd.get_dummies(input_df, columns=categorical_columns)

        
        expected_columns = lr.feature_names_in_
        input_df = input_df.reindex(columns=expected_columns, fill_value=0)


        prediction = lr.predict(input_df)
        predicted_label = booking_status_le.inverse_transform(prediction)[0]

        return jsonify({'prediction': predicted_label})

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=3000)
