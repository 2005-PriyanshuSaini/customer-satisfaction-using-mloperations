from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import mlflow
import mlflow.sklearn

app = Flask(__name__)

# Load the model and preprocessing objects
def load_model():
    global model, scaler, label_encoders, feature_names
    
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('models/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

# Call load_model when the app starts
with app.app_context():
    load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form.to_dict()
    
    # Create a DataFrame with the input data
    input_df = pd.DataFrame([data])
    
    # Convert categorical variables
    for col in label_encoders.keys():
        if col in input_df.columns:
            try:
                # Ensure data type consistency
                input_df[col] = input_df[col].astype(str)
                
                # Check if all values are in the encoder's classes_
                if all(val in label_encoders[col].classes_ for val in input_df[col]):
                    input_df[col] = label_encoders[col].transform(input_df[col])
                else:
                    # Handle unknown values with a default
                    print(f"Warning: Unknown values in column {col}")
                    input_df[col] = -1
            except Exception as e:
                print(f"Error transforming column {col}: {e}")
                input_df[col] = -1
    
    # Ensure all feature columns are present
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match the training data
    input_df = input_df[feature_names]
    
    # Scale the features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    # Return the prediction
    return render_template('result.html', 
                          prediction=bool(prediction), 
                          probability=round(probability * 100, 2))

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Changed port to avoid conflict with AirPlay
