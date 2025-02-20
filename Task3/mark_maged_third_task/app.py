#loading libraries
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle

#loading model and feature ordering
logit = pickle.load(open('logit.pkl', 'rb'))
reg_features = ['repeated', 'no_lead', 'mst_corporate', 'rs_winter', 'special_requests', 'mst_online', 'rs_summer', 'log_lead', 'average_price', 'total_individuals']

app = Flask(__name__)

#home page
@app.route('/', methods= ['GET'])
def hello():
    return render_template('logit.html')

#predection page
@app.route('/predict', methods=['POST'])
def predict():
    #transforming form values into a numerical list
    form_input = [float(x) for x in request.form.values()]

    #setting the proper values for the dummy variables using the input
    winter_summer= []
    corporate_online= []
    no_lead= [0]

    if form_input[0] == 1:
        winter_summer= [1, 0]
    elif form_input[0] == 2:
        winter_summer= [0, 1]
    else:
        winter_summer= [0, 0]

    if form_input[1] == 1:
        corporate_online= [1, 0]
    elif form_input[1] == 2:
        corporate_online= [0, 1]
    else:
        corporate_online= [0, 0]

    if form_input[4] == 0:
        no_lead= [1]
    form_input = form_input[2:]
    form_input = form_input + winter_summer + corporate_online + no_lead
    
    #rearanging the features to fit the model
    input_df = pd.DataFrame(np.array(form_input).reshape(1, -1), columns= ['repeated', 'special_requests', 'log_lead', 'average_price', 'total_individuals', 'rs_winter', 'rs_summer', 'mst_corporate', 'mst_online', 'no_lead'])
    input_df = input_df.loc[: ,reg_features]
    
    #integrating the input data into the preprocessed data the model was trained on for proper scaling
    pre = pd.read_csv(r'preprocessed.csv')
    pre.drop(columns= pre.columns[0], axis= 1, inplace= True)
    pre = pd.concat([pre, input_df])
    pre.reset_index(drop= True, inplace= True)
    
    #taking the log for lead time to handle outliers and decrease skewness
    pre['log_lead'] = np.log(pre['log_lead'].replace(0,1))

    #scaling data
    scaler = preprocessing.StandardScaler()
    scaled_data = scaler.fit_transform(pre[['special_requests', 'log_lead','average_price', 'total_individuals']])
    scaled_df = pd.DataFrame(scaled_data, columns= ['special_requests', 'log_lead','average_price', 'total_individuals'])
    pre.drop(columns= ['special_requests', 'log_lead','average_price', 'total_individuals'], axis= 1, inplace= True)
    reg_hs = pd.concat([pre, scaled_df], axis= 1)
    reg_hs = reg_hs.loc[:, reg_features]

    #extracting input row after scaling for predection
    input = pd.DataFrame(np.array(reg_hs.iloc[-1]).reshape(1,-1), columns= reg_hs.columns)
    prediction = logit.predict(input)
    probability = logit.predict_proba(input)[0]
    confidence = round(max(probability), 2)*100
    if prediction[0] == 0:
        return render_template('logit.html', prediction_text= 'Reservation will be canceled', confidence_text= f'Confidence: {confidence}%')
    else:
        return render_template('logit.html', prediction_text= 'Reservation will not be canceled', confidence_text= f'Confidence: {confidence}%')

if __name__ == '__main__':
    app.run(port= 3000, debug= True)