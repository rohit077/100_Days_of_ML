# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('RandomForestRegression_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    input_features = [int(x) for x in request.form.values()]
    Area=input_features[0]
    BHK=input_features[1]
    Bathroom=input_features[2]
    Parking=input_features[3]
    Per_Sqft=input_features[4]
    Type_Builder_Floor=input_features[5]
    final_features = [np.array([Area, BHK, Bathroom, Parking, Per_Sqft, Type_Builder_Floor])]
    Prediction = model.predict(final_features)

    output = round(Prediction[0], 2)

    return render_template('index.html', prediction_text='Housing Price would be ₹ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    Prediction = model.predict([np.array(list(data.values()))])

    output = Prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)