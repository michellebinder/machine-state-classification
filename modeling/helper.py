# ------------------------------
# Imports
# ------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import mlflow
import mlflow.sklearn

# ------------------------------
# Global variables
# ------------------------------
global_results = {}

# ------------------------------
# Helper functions
# ------------------------------
def save_results_to_file(filename="results.json"):
    with open(filename, "w") as file:
        json.dump(global_results, file, indent=4)

def plot_confusion_matrix(cm, experiment_name):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix for {experiment_name}", fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.tight_layout()

    filename = f"{experiment_name.replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(filename)
    plt.close()
    return filename

# ------------------------------
# Workflow function
# ------------------------------
def run_workflow(X, y, model, experiment_name, param_grid=None, val_size=0.2, random_state=42, test_data=None):
    
    global global_results
    classifier_results = {}

    # Initialize mlflow experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="Train-Test-Split"):
        print("MLflow Run started...")

        # Log information
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_param("val_size", val_size)

        # Step 1: Train-Test Split 
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_state, stratify=y)

       # Step 2: Train and Evaluate on Validation Set
        # with mlflow.start_run(run_name="Train-Val-Split"):
        print("Training started...")
        model.fit(X_train, y_train)

        print("Validation started...")
        y_pred_val = model.predict(X_val)

        val_metrics_macro = {
            "accuracy_macro": accuracy_score(y_val, y_pred_val),
            "precision_macro": precision_score(y_val, y_pred_val, average="macro"),
            "recall_macro": recall_score(y_val, y_pred_val, average="macro"),
            "f1_macro": f1_score(y_val, y_pred_val, average="macro")
        }
        classifier_results["train_val_split_macro"] = val_metrics_macro

        for metric, value in val_metrics_macro.items():
            mlflow.log_metric(f"val_{metric}", value)

        # End the Train-Val-Split Run
        mlflow.end_run()

        # Step 3: Test on Unseen Data
        print("Testing on unseen data started...")
        if test_data is not None:
            with mlflow.start_run(run_name="Testing on Unseen Data"):
                X_test, y_test = test_data
                y_pred_test = model.predict(X_test)
                
                y_proba_test = model.predict_proba(X_test)

                test_metrics_macro = {
                    "accuracy_macro": accuracy_score(y_test, y_pred_test),
                    "precision_macro": precision_score(y_test, y_pred_test, average="macro"),
                    "recall_macro": recall_score(y_test, y_pred_test, average="macro"),
                    "f1_macro": f1_score(y_test, y_pred_test, average="macro")
                }

                test_metrics_micro = {
                    "precision_micro": precision_score(y_test, y_pred_test, average="micro"),
                    "recall_micro": recall_score(y_test, y_pred_test, average="micro"),
                    "f1_micro": f1_score(y_test, y_pred_test, average="micro")
                }

                classifier_results["testing_macro"] = test_metrics_macro
                classifier_results["testing_micro"] = test_metrics_micro

                for metric, value in test_metrics_macro.items():
                    mlflow.log_metric(f"test_{metric}", value)
                for metric, value in test_metrics_micro.items():
                    mlflow.log_metric(f"test_{metric}", value)

                # Log Confusion Matrix for Test Data
                cm = confusion_matrix(y_test, y_pred_test)
                filename = plot_confusion_matrix(cm, experiment_name)
                mlflow.log_artifact(filename)

                # Step 4: Save the Model to MLflow
                mlflow.sklearn.log_model(
                    sk_model=model, 
                    artifact_path=f"models/{experiment_name}",
                    registered_model_name=experiment_name 
                )

        # End the Testing on Unseen Data Run
        mlflow.end_run()

        global_results[experiment_name] = classifier_results
        print("Done")