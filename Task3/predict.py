from flask import Flask, render_template, request
import numpy as np
import os
import pickle

app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'templates'))

model = pickle.load(open('model.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    form_data = request.form

    
    fields_to_encode = ['market_segment_type', 'room_type', 'type_of_meal']

    
    final_features = []

    
    for field, value in form_data.items():
        if field in fields_to_encode:
            
            encoded_value = label_encoder[field].transform([value])[0]  
            final_features.append(encoded_value)
        else:
            
            final_features.append(int(value))

    
    final_features = np.array(final_features).reshape(1, -1)

    
    prediction = model.predict(final_features)  
    class_name = prediction[0]  

    
    return render_template('index.html', classname=f'Predicted class is {class_name}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)