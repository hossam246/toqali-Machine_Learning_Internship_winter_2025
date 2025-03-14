import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# Load the model
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the feature names used during training
with open('feature_names.pkl', 'rb') as f:
    training_features = pickle.load(f)

# Try to load the scaler if it exists, otherwise create a new one
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Loaded existing scaler")
except FileNotFoundError:
    print("No scaler found, creating a new StandardScaler")
    # Create a new scaler - this is not ideal but will work for now
    scaler = StandardScaler()

@app.route('/')
def home():
    # Set current date as default
    today = datetime.now().strftime('%Y-%m-%d')
    return render_template('index.html', today=today)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Process form data
        if request.method == 'POST':
            # Get form data
            adults = int(request.form.get('adults', 0))
            children = int(request.form.get('children', 0))
            weekend_nights = int(request.form.get('weekend_nights', 0))
            week_nights = int(request.form.get('week_nights', 0))
            car_parking = int(request.form.get('car_parking', 0))
            lead_time = float(request.form.get('lead_time', 0))
            repeated = int(request.form.get('repeated', 0))
            prev_cancellations = int(request.form.get('prev_cancellations', 0))
            prev_non_cancellations = int(request.form.get('prev_non_cancellations', 0))
            avg_price = float(request.form.get('avg_price', 0))
            special_requests = int(request.form.get('special_requests', 0))
            
            # Get and process date
            reservation_date = request.form.get('reservation_date', '')
            year, month, day_of_week, is_weekend = process_date(reservation_date)
            
            # Get categorical values
            meal_type = request.form.get('meal_type', '')
            room_type = request.form.get('room_type', '')
            market_segment = request.form.get('market_segment', '')
            
            # Calculate total guests
            total_guests = adults + children
            
            # Apply log transformations as per your preprocessing
            lead_time_log = np.log1p(lead_time) if lead_time > 0 else 0
            avg_price_log = np.log1p(avg_price) if avg_price > 0 else 0
            
            # Create a DataFrame with zeros for all training features
            features_dict = {feature: 0 for feature in training_features}
            
            # Map the form inputs to the correct feature names
            # You'll need to adjust these mappings based on your actual training features
            
            # Map numeric features (assuming these column names from training)
            mapping = {
                'number_of_adults': adults,
                'number_of_children': children, 
                'number_of_weekend_nights': weekend_nights,
                'number_of_week_nights': week_nights,
                'car_parking_space': car_parking,
                'lead_time': lead_time_log,
                'repeated': repeated,
                'number_of_previous_cancelations': prev_cancellations,
                'number_of_previous_non_cancelations': prev_non_cancellations,
                'average_price': avg_price_log,
                'special_requests': special_requests,
                'total_guests': total_guests
            }
            
            # Fill in the features that exist in the training data
            for key, value in mapping.items():
                if key in features_dict:
                    features_dict[key] = value
            
            # Handle one-hot encoded day of week
            day_of_week_col = f'day_of_week_{float(day_of_week)}'
            if day_of_week_col in features_dict:
                features_dict[day_of_week_col] = 1
                
            # Handle is_weekend if it exists
            if 'is_weekend' in features_dict:
                features_dict['is_weekend'] = is_weekend
                
            # Handle year (may need adjusting based on your training data)
            if 'year' in features_dict:
                features_dict['year'] = year
                
            # Handle meal type one-hot encoding
            meal_col = f'type_of_meal_{meal_type}'
            if meal_col in features_dict:
                features_dict[meal_col] = 1
            
            # Handle room type one-hot encoding
            room_col = f'room_type_{room_type}'
            if room_col in features_dict:
                features_dict[room_col] = 1
                
            # Handle market segment one-hot encoding
            market_col = f'market_segment_type_{market_segment}'
            if market_col in features_dict:
                features_dict[market_col] = 1
            
            # Create DataFrame with exact feature names from training
            df = pd.DataFrame([features_dict])
            
            # Ensure the columns are in the same order as training
            df = df[training_features]
            
            # Make prediction
            prediction = model.predict(df)[0]
            probabilities = model.predict_proba(df)[0]
            
            # Get the probability of the predicted class
            probability = probabilities[1] if prediction == 1 else probabilities[0]
            probability_percentage = probability * 100
            
            # Interpret prediction
            if prediction == 1:
                result = "Booking Not Canceled"
                status_class = "not-canceled"
            else:
                result = "Booking Canceled"
                status_class = "canceled"
            
            # Generate insights based on the most important features
            insights = generate_insights(df.iloc[0].to_dict())
            
            return render_template('index.html', 
                                   prediction=result,
                                   status_class=status_class,
                                   confidence=f"{probability_percentage:.2f}%",
                                   insights=insights,
                                   form_data=request.form,
                                   today=reservation_date)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print("Error details:", error_details)
        return render_template('index.html', error=str(e), today=datetime.now().strftime('%Y-%m-%d'))

def process_date(date_str):
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        year = date_obj.year
        month = date_obj.month
        day_of_week = date_obj.weekday()  # 0=Monday, 6=Sunday
        is_weekend = 1 if day_of_week >= 5 else 0
        return year, month, day_of_week, is_weekend
    except:
        # Default values if date parsing fails
        now = datetime.now()
        return now.year, now.month, now.weekday(), 1 if now.weekday() >= 5 else 0

def generate_insights(features):
    # Based on the permutation importance you shared
    insights = []
    
    # Check if the features exist before using them
    if 'lead_time' in features and features['lead_time'] > 0:
        # Reverse log transformation if it was applied
        lead_time_orig = np.expm1(features['lead_time'])
        if lead_time_orig > 30:
            insights.append(f"Lead time of {lead_time_orig:.0f} days is significant. Longer lead times often have different cancellation patterns.")
    
    if 'average_price' in features and features['average_price'] > 0:
        # Reverse log transformation if it was applied
        avg_price_orig = np.expm1(features['average_price'])
        if avg_price_orig > 150:
            insights.append(f"The average price of ${avg_price_orig:.2f} is relatively high, which may influence cancellation likelihood.")
    
    if 'number_of_previous_cancelations' in features and features['number_of_previous_cancelations'] > 0:
        insights.append("Previous cancellations significantly increase the likelihood of another cancellation.")
    
    # Check for market segment features
    for feature in features:
        if feature.startswith('market_segment_type_') and features[feature] == 1:
            segment_name = feature.split('_')[-1]
            insights.append(f"The {segment_name} market segment has its own distinct cancellation patterns.")
    
    if 'special_requests' in features and features['special_requests'] > 1:
        insights.append("Multiple special requests can indicate a more committed booking.")
    
    # Calculate total nights if both features exist
    weekend_nights = features.get('number_of_weekend_nights', 0)
    week_nights = features.get('number_of_week_nights', 0)
    total_nights = weekend_nights + week_nights
    if total_nights > 7:
        insights.append("Longer stays tend to have different cancellation patterns than shorter stays.")
    
    return insights

if __name__ == '__main__':
    app.run(debug=True)