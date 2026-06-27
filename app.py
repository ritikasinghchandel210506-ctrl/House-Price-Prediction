import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model, scaler, and feature structural layout
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('features.pkl', 'rb') as f:
    model_features = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract raw data from form
        form_data = {
            'area': float(request.form['area']),
            'bedrooms': int(request.form['bedrooms']),
            'bathrooms': int(request.form['bathrooms']),
            'stories': int(request.form['stories']),
            'mainroad': int(request.form['mainroad']),
            'guestroom': int(request.form['guestroom']),
            'basement': int(request.form['basement']),
            'hotwaterheating': int(request.form['hotwaterheating']),
            'airconditioning': int(request.form['airconditioning']),
            'parking': int(request.form['parking']),
            'prefarea': int(request.form['prefarea']),
        }
        
        # Handle dummy variables for furnishingstatus
        furnishing = request.form['furnishingstatus']
        form_data['furnishingstatus_semi-furnished'] = 1 if furnishing == 'semi-furnished' else 0
        form_data['furnishingstatus_unfurnished'] = 1 if furnishing == 'unfurnished' else 0

        # Convert to DataFrame matching the exact structure of training features
        input_df = pd.DataFrame([form_data])
        input_df = input_df[model_features] # Ensure correct column order
        
        # Scale inputs
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        formatted_prediction = f"₹{prediction:,.2f}" # Formatted format

        return render_template('index.html', prediction_text=f'Estimated House Price: {formatted_prediction}')

if __name__ == '__main__':
    app.run(debug=True)