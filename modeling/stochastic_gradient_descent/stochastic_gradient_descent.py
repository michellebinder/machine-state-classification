from sklearn.linear_model import SGDClassifier  
import pandas as pd
import sys
import os
import time
sys.path.append("/Users/bindiair/Desktop/machine-state-classification/modeling")
from helper import save_results_to_file, run_workflow

def safe_read_csv(filepath):
    try:
        print(f"Checking file: {filepath}")
        if not os.path.exists(filepath):
            print(f"Error: File {filepath} does not exist.")
            return None
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            print(f"Error: File {filepath} is empty.")
            return None
        print(f"File size: {file_size / 1e6:.2f} MB. Loading...")
        start_time = time.time()
        df = pd.read_csv(filepath)
        print(f"Loaded data. Shape: {df.shape}. Time taken: {time.time() - start_time:.2f} seconds.")
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

print('Loading training data...')
train_df = safe_read_csv("/Users/bindiair/Desktop/machine-state-classifications/data_preparation/processed_datasets/training_data_scaled_smote.csv")

print('Loading testing data...')
test_df = safe_read_csv("/Users/bindiair/Desktop/machine-state-classifications/data_preparation/processed_datasets/testing_data_scaled.csv")

X = train_df.drop(columns='Label')
y = train_df['Label']

X_test = test_df.drop(columns='Label')
y_test = test_df['Label']

model = SGDClassifier(max_iter=1000, random_state=42)

run_workflow(
    X=X,
    y=y,
    model=model,
    experiment_name="Stochastic_Gradient_Descent", 
    val_size=0.2,
    random_state=42,
    test_data=(X_test, y_test) 
)

save_results_to_file(filename='stochastic_gradient_descent_results.json') 
