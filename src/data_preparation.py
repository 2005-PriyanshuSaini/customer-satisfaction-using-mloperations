import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle


def prepare_data():
    # Load the dataset
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Data cleaning
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Drop customerID as it's not relevant for prediction
    df = df.drop('customerID', axis=1)
    
    # Convert categorical variables to numeric
    categorical_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    # Save the label encoders for later use
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    # Split features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler for later use
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Save the column names for later use
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)
    
    return X_train, X_test, y_train, y_test, df

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, df = prepare_data()
    print("Data preparation completed.")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
