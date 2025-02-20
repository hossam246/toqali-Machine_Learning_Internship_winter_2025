import numpy as np
from flask import Flask, request, render_template
import pickle
from functions import extract_month, extract_year

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

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
    if request.form['type_of_meal'] == 'Meal Plan 1':
        ls = [1,0,0,0] 
        predection = predection + ls
    if request.form['type_of_meal'] == 'Meal Plan 2':
        ls = [0,1,0,0] 
        predection = predection + ls
    if request.form['type_of_meal'] == 'Meal Plan 3':
        ls = [0,0,1,0] 
        predection = predection + ls
    if request.form['type_of_meal'] == 'Not Selected':
        ls = [0,0,0,1] 
        predection = predection + ls
    if request.form['room_type'] == 'Room_Type 1':
        ls = [1,0,0,0,0,0,0] 
        predection = predection + ls
    if request.form['room_type'] == 'Room_Type 2':
        ls = [0,1,0,0,0,0,0] 
        predection = predection + ls
    if request.form['room_type'] == 'Room_Type 3':
        ls = [0,0,1,0,0,0,0] 
        predection = predection + ls
    if request.form['room_type'] == 'Room_Type 4':
        ls = [0,0,0,1,0,0,0] 
        predection = predection + ls
    if request.form['room_type'] == 'Room_Type 5':
        ls = [0,0,0,0,1,0,0] 
        predection = predection + ls
    if request.form['room_type'] == 'Room_Type 6':
        ls = [0,0,0,0,0,1,0] 
        predection = predection + ls
    if request.form['room_type'] == 'Room_Type 7':
        ls = [0,0,0,0,0,0,1]
        predection = predection + ls
    if request.form['market_segment_type'] == 'Aviation':
        ls = [1,0,0,0,0]
        predection = predection + ls    
    if request.form['market_segment_type'] == 'Complementary':
        ls = [0,1,0,0,0]
        predection = predection + ls
    if request.form['market_segment_type'] == 'Corporate':
        ls = [0,0,1,0,0]
        predection = predection + ls
    if request.form['market_segment_type'] == 'Offline':
        ls = [0,0,0,1,0]
        predection = predection + ls
    if request.form['market_segment_type'] == 'Online':
        ls = [0,0,0,0,1]
        predection = predection + ls

    predection = np.array([predection]).reshape(1,-1)

    final_prediction = model.predict(predection) 
    print(predection)
    return render_template("index.html", prediction_text=f"Predicted Booking Status: {final_prediction}")
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)