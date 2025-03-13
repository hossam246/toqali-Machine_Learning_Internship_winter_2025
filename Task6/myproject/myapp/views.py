from django.shortcuts import render
import joblib
import numpy as np

def predict_fare(request):
    prediction = None
    error = None
    if request.method == 'POST':
        try:
            model = joblib.load('D:\\cellula\\myproject\\fare_model.joblib')
            scaler = joblib.load('D:\\cellula\\myproject\\fare_scaler.joblib')
            encoders = {
                col: joblib.load(f'D:\\cellula\\myproject\\{col}_encoder.joblib')
                for col in ['Car Condition', 'Weather', 'Traffic Condition']
            }
            features = [
                float(request.POST.get('hour')),
                float(request.POST.get('day')),
                float(request.POST.get('month')),
                float(request.POST.get('passenger_count')),
                float(request.POST.get('distance')),
                float(request.POST.get('jfk_dist')),
                encoders['Car Condition'].transform([request.POST.get('car_condition')])[0],
                encoders['Weather'].transform([request.POST.get('weather')])[0],
                encoders['Traffic Condition'].transform([request.POST.get('traffic')])[0],
            ]
            scaled_features = scaler.transform([features])
            prediction = round(model.predict(scaled_features)[0], 2)
        except Exception as e:
            error = f"Prediction error: {str(e)}"
    return render(request, 'D:\\cellula\\myproject\\templates\\index.html', {
        'prediction': prediction,
        'error': error
    })