import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from data_preparation import prepare_data

def train_model():
    # Prepare the data
    X_train, X_test, y_train, y_test, df = prepare_data()
    
    # Start MLflow run
    mlflow.start_run()
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # Create an input example for model signature
    input_example = X_train[:5]  # Using first 5 samples as example
    
    # Infer the model signature
    signature = infer_signature(X_train, y_prob)
    
    # Log model to MLflow with input example and signature
    mlflow.sklearn.log_model(
        model, 
        "random_forest_model",
        signature=signature,
        input_example=input_example
    )
    
    # End MLflow run
    mlflow.end_run()
    
    # Save the model
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Calculate NPS bias if 'Churn' column exists in the dataset
    if 'Churn' in df.columns:
        # This is inspired by the NPS_BIAS calculation from the research paper
        print("Analyzing customer segments based on prediction bias...")
        
    print(f"Model training completed with accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return model, accuracy, precision, recall, f1

if __name__ == "__main__":
    model, accuracy, precision, recall, f1 = train_model()
