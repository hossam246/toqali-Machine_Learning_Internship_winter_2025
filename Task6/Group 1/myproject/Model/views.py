from django.shortcuts import render
import pickle
import numpy as np
import sklearn


car_conditions = {'Excellent': 0, 'Very Good': 1, 'Good': 2, 'Bad': 3}
traffic_conditions = {'Flow Traffic': 0, 'Dense Traffic': 1, 'Congested Traffic': 2}
weather_conditions = {'sunny': 0, 'windy': 1, 'cloudy': 2, 'rainy': 3, 'stormy': 4}
with open('./savedModels/Model1.pkl', 'rb') as file:
    model = pickle.load(file)
with open('./savedModels/pca.pkl', 'rb') as file:
    pca = pickle.load(file)
with open('./savedModels/Xscaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
def myrun(request):
    if request.method == 'POST':
        car_condition = int(car_conditions.get(request.POST.get('car_condition'), -1))
        weather = int(weather_conditions.get(request.POST.get('weather'), -1))
        traffic_condition = int(traffic_conditions.get(request.POST.get('traffic_condition'), -1))
        pickup_longitude = request.POST.get('pickup_longitude')
        pickup_latitude = request.POST.get('pickup_latitude')
        dropoff_longitude = request.POST.get('dropoff_longitude')
        dropoff_latitude = request.POST.get('dropoff_latitude')
        passenger_count = request.POST.get('passenger_count')
        hour = request.POST.get('hour')
        day = request.POST.get('day')
        month = request.POST.get('month')
        weekday = request.POST.get('weekday')
        year = request.POST.get('year')
        distance = request.POST.get('distance')
        bearing = request.POST.get('bearing')
        features = np.array([car_condition, weather, 
                    traffic_condition, pickup_longitude,
                    pickup_latitude, dropoff_longitude,
                    dropoff_latitude, passenger_count,
                    hour, day, month, weekday,
                    distance, bearing]).reshape(1, -1)
        scaled_features = scaler.transform(features)
        pca_features = pca.transform(scaled_features)
        y_transformed = model.predict(pca_features)
        y_pred = np.exp(y_transformed)
        return render(request, 'index.html', {'result' : y_pred})
    return render(request, 'index.html')

