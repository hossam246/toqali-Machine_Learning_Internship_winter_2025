from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
model = joblib.load('Randomforestcalssifer.joblib')
class_names = np.array(["Not Canceled", "Canceled"])  


@app.route('/', methods=['GET'])
def hello_world():
    result = ''
    return render_template('index.html', result=result)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    
    #number_of_adults = int(request.form['number_of_adults'])
    #number_of_children = int(request.form['number_of_children'])
    #number_of_weekend_nights = int(request.form['number_of_weekend_nights'])
   # number_of_week_nights = int(request.form['number_of_week_nights'])
    #type_of_meal = request.form['type_of_meal']  # Assuming categorical, may need encoding
    car_parking_space = int(request.form['car_parking_space'])
   #room_type = request.form['room_type']  # Assuming categorical, may need encoding
    lead_time = float(request.form['lead_time'])
    market_segment_type = request.form['market_segment_type']  # Assuming categorical, may need encoding
  #  repeated = int(request.form['repeated'])
    #P_C = int(request.form['P_C'])
   # P_not_C = int(request.form['P_not_C'])
    average_price = float(request.form['average_price'])
    special_requests = int(request.form['special_requests'])
    month = int(request.form['month'])
    day = int(request.form['day'])
    #year = int(request.form['year'])
   # season = request.form['season']  # Assuming categorical, may need encoding

    
    labelencoder = LabelEncoder()
    #type_of_meal_encoded = labelencoder.fit_transform([type_of_meal])[0]
    #room_type_encoded = labelencoder.fit_transform([room_type])[0]
    market_segment_type_encoded = labelencoder.fit_transform([market_segment_type])[0]
    #season_encoded = labelencoder.fit_transform([season])[0]

    
    features = [[ car_parking_space,  lead_time,
        market_segment_type_encoded,  average_price,
        special_requests, month, day
    ]]

    prediction = model.predict(features)[0]  
    result = class_names[prediction]  
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(port=3000, debug=True)