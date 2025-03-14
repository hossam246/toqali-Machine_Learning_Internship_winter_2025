from django.shortcuts import render
import math
import numpy as np
import pandas as pd
import pickle


model = pickle.load(open(r'C:\Cellula\task 6\fare_prediction\pkl\rf.pkl', 'rb'))
features = ['distance', 'jfk_dist', 'ewr_dist', 'lga_dist', 'nyc_dist', 'sol_dist', 'passenger_count', 'bearing']
scaler = pickle.load(open(r'C:\Cellula\task 6\fare_prediction\pkl\scaler.pkl', 'rb'))

def home(request):
    return render(request, 'index.html')

def result(request):
    if request.method == 'POST':
        print(request.POST)  # Check if Django is receiving form data
        dictionary = {}
        for feat in features:
            dictionary[feat] = request.POST.get(feat, '')  
        df = pd.DataFrame([dictionary])
        non_scaled_feat = ['jfk_dist', 'ewr_dist', 'lga_dist', 'sol_dist', 'nyc_dist', 'distance', 'passenger_count']
        scaled_data = scaler.transform(df[non_scaled_feat])
        scaled_df = pd.DataFrame(scaled_data, columns= non_scaled_feat)
        df.drop(non_scaled_feat, axis= 1, inplace= True)
        df = pd.concat([scaled_df, df], axis=1)
        df = df[features]
        pred = model.predict(df)
        pred = round(math.exp(pred[0]), 2)
        return render(request, 'index.html', {'result' : f'Predicted Fare : {pred} dollars'})
    return render(request, 'index.html' )