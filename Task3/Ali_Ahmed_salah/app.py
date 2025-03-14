import numpy as np
from flask import Flask, request, render_template
import pickle
from functions import extract_month, extract_year

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    predection = []
    predection.append(int(request.form['number_of_adults']))
    predection.append(int(request.form['number_of_children']))
    predection.append(int(request.form['number_of_weekend_nights']))
    predection.append(int(request.form['number_of_week_nights']))
    predection.append(int(request.form['car_parking_space']))
    predection.append(int(request.form['lead_time']))
    predection.append(int(request.form['repeated']))
    predection.append(int(request.form['P_C']))
    predection.append(int(request.form['P_not_C']))
    predection.append(float(request.form['average_price_']))
    predection.append(int(request.form['special_requests']))
    predection.append(extract_month(request.form['date_of_reservation']))
    predection.append(extract_year(request.form['date_of_reservation']))
    predection.append(int(request.form['number_of_adults']) + int(request.form['number_of_children']))
    predection.append(int(request.form['number_of_weekend_nights']) + int(request.form['number_of_week_nights']))

    meal_plan_mapping = {
    'Meal Plan 1': [1, 0, 0, 0],
    'Meal Plan 2': [0, 1, 0, 0],
    'Meal Plan 3': [0, 0, 1, 0],
    'Not Selected': [0, 0, 0, 1]
}
    meal_plan = request.form['type_of_meal']
    if meal_plan in meal_plan_mapping:
       predection += meal_plan_mapping[meal_plan]


    room_type_mapping = {
    'Room_Type 1': [1, 0, 0, 0, 0, 0, 0],
    'Room_Type 2': [0, 1, 0, 0, 0, 0, 0],
    'Room_Type 3': [0, 0, 1, 0, 0, 0, 0],
    'Room_Type 4': [0, 0, 0, 1, 0, 0, 0],
    'Room_Type 5': [0, 0, 0, 0, 1, 0, 0],
    'Room_Type 6': [0, 0, 0, 0, 0, 1, 0],
    'Room_Type 7': [0, 0, 0, 0, 0, 0, 1]
}
    room_type = request.form['room_type']
    if room_type in room_type_mapping:
        predection += room_type_mapping[room_type]

    market_segment_mapping = {
    'Aviation': [1, 0, 0, 0, 0],
    'Complementary': [0, 1, 0, 0, 0],
    'Corporate': [0, 0, 1, 0, 0],
    'Offline': [0, 0, 0, 1, 0],
    'Online': [0, 0, 0, 0, 1]
}
    market_segment_type = request.form['market_segment_type']
    if market_segment_type in market_segment_mapping:
        predection += market_segment_mapping[market_segment_type]


    predection = np.asarray(predection).reshape(1,-1)
    predection_scaled = scaler.transform(predection)

    final_prediction = model.predict(predection_scaled) 
    if (final_prediction == 1):
        cancelation_status = 'Not canceled'
    else:
        cancelation_status = 'Canceled'
    return render_template("index.html", prediction_text=f"Predicted Booking Status: {cancelation_status}")
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)