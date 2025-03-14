from django.shortcuts import render
import numpy as np
import cupy as cp
import pickle

model = pickle.load(open('/home/ali/Cellula/Task_6/Task_6/Saved_models/model_XGB.pkl','rb'))
scaler = pickle.load(open('/home/ali/Cellula/Task_6/Task_6/Saved_models/scaler.pkl','rb'))
PCA_1 = pickle.load(open('/home/ali/Cellula/Task_6/Task_6/Saved_models/PCA1.pkl','rb'))
PCA_2 = pickle.load(open('/home/ali/Cellula/Task_6/Task_6/Saved_models/PCA2.pkl','rb'))

def predictor(request):
    predection = []
    if request.method == 'POST':
        car_condition_mapping = {'Very Good': 2, 'Bad': 0, 'Good': 1, 'Excellent': 3}
        car_condition = request.POST['Car_Condition']
        if car_condition in car_condition_mapping:
            predection.append(car_condition_mapping[car_condition])

        weather_mapping = {'sunny': 4, 'cloudy': 3, 'rainy': 1, 'stormy': 0, 'windy': 2}
        weather = request.POST['Weather']
        if weather in weather_mapping:
            predection.append(weather_mapping[weather])

        traffic_condition_mapping = {'Congested Traffic': 0, 'Dense Traffic': 1, 'Flow Traffic': 2}
        traffic_condition = request.POST['Traffic_Condition']
        if traffic_condition in traffic_condition_mapping:
            predection.append(traffic_condition_mapping[traffic_condition])        

        predection.append(int(request.POST['passenger_count']))
        predection.append(int(request.POST['day']))
        predection.append(int(request.POST['month']))
        predection.append(int(request.POST['weekday']))
        distance = float(request.POST['distance'])

        if distance > 0 :
            distance = np.log(distance)
            predection.append(distance)
        else :
            predection.append(distance)

        predection.append(float(request.POST['bearing']))

        pickup_longitude = float(request.POST['pickup_longitude'])
        pickup_latitude = float(request.POST['pickup_latitude'])
        dropoff_longitude = float(request.POST['dropoff_longitude'])
        dropoff_latitude = float(request.POST['dropoff_latitude'])
        jfk_dist = float(request.POST['jfk_dist'])
        ewr_dist = float(request.POST['ewr_dist'])
        lga_dist = float(request.POST['lga_dist'])
        sol_dist = float(request.POST['sol_dist'])
        nyc_dist = float(request.POST['nyc_dist'])
        location_features = np.array([pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, 
                                    jfk_dist, ewr_dist, lga_dist, sol_dist, nyc_dist]).reshape(1, -1)
        PC1 = PCA_1.transform(location_features)
        predection.append(PC1[0, 0])
        predection.append(PC1[0, 1])
        predection.append(PC1[0, 2])

        hour = int(request.POST['hour'])
        year = float(request.POST['year'])
        time_features = np.array([hour, year]).reshape(1, -1)
        PC2 = PCA_2.transform(time_features)
        predection.append(PC2[0, 0])

        predection = np.asarray(predection).reshape(1,-1)
        predection_scaled = scaler.transform(predection)

        final_prediction = model.predict(predection_scaled)
        final_prediction = np.exp(final_prediction)
        return render(request, 'index.html', {'result' : final_prediction})
    return render(request, 'index.html')