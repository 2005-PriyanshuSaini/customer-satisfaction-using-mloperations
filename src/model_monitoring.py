import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import pickle
import mlflow
import mlflow.sklearn
import time
import os
from datetime import datetime

def monitor_model_performance():
    # Load the model and data
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Prepare the data (simulating new data coming in)
    from data_preparation import prepare_data
    _, X_test, _, y_test, _ = prepare_data()
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log to MLflow
    mlflow.start_run()
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("timestamp", time.time())
    mlflow.end_run()
    
    # Create a monitoring log
    log_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'accuracy': accuracy
    }
    
    # Append to monitoring log file
    monitoring_log_file = 'models/monitoring_log.csv'
    
    if os.path.exists(monitoring_log_file):
        monitoring_log = pd.read_csv(monitoring_log_file)
        # Replace append with concat
        monitoring_log = pd.concat([monitoring_log, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        monitoring_log = pd.DataFrame([log_entry])
    
    monitoring_log.to_csv(monitoring_log_file, index=False)
    
    print(f"Model monitoring completed. Current accuracy: {accuracy:.4f}")
    
    # Check if model needs retraining
    if accuracy < 0.75:
        print("Model accuracy below threshold. Retraining recommended.")
        # In a real system, this could trigger an automated retraining pipeline
    
    return accuracy

if __name__ == "__main__":
    monitor_model_performance()
