# ------------------------------
# Imports
# ------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
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
def run_workflow(X, y, model, experiment_name, param_grid=None, val_size=0.2, random_state=42, test_data=None, log_classification_report=True):
    
    global global_results
    classifier_results = {}

    # -----------------------------
    # 1) Train-Val Split Run
    # -----------------------------

    # Initialize mlflow experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="Train-Test-Split"):
        print("MLflow Run started...")

        # Log information
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_param("val_size", val_size)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_state, stratify=y)

       # Model training
        print("Training started...")
        model.fit(X_train, y_train)

        # Validation
        print("Validation started...")
        y_pred_val = model.predict(X_val)

        # classification_report for validation
        report_val_dict = classification_report(y_val, y_pred_val, output_dict=True)
        report_val_text = classification_report(y_val, y_pred_val)

        # Log text report as an artifact
        with open("classification_report_val.txt", "w") as f:
            f.write(report_val_text)
        mlflow.log_artifact("classification_report_val.txt")

        # Log metrics from the dictionary
        # Accuracy
        mlflow.log_metric("val_accuracy", report_val_dict["accuracy"])

        # Macro metrics
        mlflow.log_metric("val_precision_macro", report_val_dict["macro avg"]["precision"])
        mlflow.log_metric("val_recall_macro", report_val_dict["macro avg"]["recall"])
        mlflow.log_metric("val_f1_macro", report_val_dict["macro avg"]["f1-score"])

        # Micro metrics 
        if "micro avg" in report_val_dict:
            mlflow.log_metric("val_precision_micro", report_val_dict["micro avg"]["precision"])
            mlflow.log_metric("val_recall_micro", report_val_dict["micro avg"]["recall"])
            mlflow.log_metric("val_f1_micro", report_val_dict["micro avg"]["f1-score"])

        # Per-class metrics
        for label, metrics in report_val_dict.items():
            # Typically, classes are labeled as strings ("0", "1", etc.) in the report
            if label.isdigit():
                mlflow.log_metric(f"val_precision_class_{label}", metrics["precision"])
                mlflow.log_metric(f"val_recall_class_{label}", metrics["recall"])
                mlflow.log_metric(f"val_f1_class_{label}", metrics["f1-score"])

        # Store metrics in classifier_results
        classifier_results["validation"] = {
            "accuracy": report_val_dict["accuracy"],
            "precision_macro": report_val_dict["macro avg"]["precision"],
            "recall_macro": report_val_dict["macro avg"]["recall"],
            "f1_macro": report_val_dict["macro avg"]["f1-score"],
            "precision_micro": report_val_dict["micro avg"]["precision"] if "micro avg" in report_val_dict else None,
            "recall_micro": report_val_dict["micro avg"]["recall"] if "micro avg" in report_val_dict else None,
            "f1_micro": report_val_dict["micro avg"]["f1-score"] if "micro avg" in report_val_dict else None
        }

        # End of Train-Val-Split run
        mlflow.end_run()

        # -----------------------------
        # 2) Test Run
        # -----------------------------
        if test_data is not None:
            print("Testing on unseen data started...")
            with mlflow.start_run(run_name="Testing on Unseen Data"):
                X_test, y_test = test_data
                y_pred_test = model.predict(X_test)

                # classification_report for test
                report_test_dict = classification_report(y_test, y_pred_test, output_dict=True)
                report_test_text = classification_report(y_test, y_pred_test)

                # Log text report as an artifact
                with open("classification_report_test.txt", "w") as f:
                    f.write(report_test_text)
                mlflow.log_artifact("classification_report_test.txt")

                # Log metrics from the dictionary
                # Accuracy
                mlflow.log_metric("test_accuracy", report_test_dict["accuracy"])

                # Macro
                mlflow.log_metric("test_precision_macro", report_test_dict["macro avg"]["precision"])
                mlflow.log_metric("test_recall_macro", report_test_dict["macro avg"]["recall"])
                mlflow.log_metric("test_f1_macro", report_test_dict["macro avg"]["f1-score"])

                # Micro 
                if "micro avg" in report_test_dict:
                    mlflow.log_metric("test_precision_micro", report_test_dict["micro avg"]["precision"])
                    mlflow.log_metric("test_recall_micro", report_test_dict["micro avg"]["recall"])
                    mlflow.log_metric("test_f1_micro", report_test_dict["micro avg"]["f1-score"])

                # Per-class metrics
                for label, metrics in report_test_dict.items():
                    if label.isdigit():
                        mlflow.log_metric(f"test_precision_class_{label}", metrics["precision"])
                        mlflow.log_metric(f"test_recall_class_{label}", metrics["recall"])
                        mlflow.log_metric(f"test_f1_class_{label}", metrics["f1-score"])

                # Store metrics in classifier_results
                classifier_results["test"] = {
                    "accuracy": report_test_dict["accuracy"],
                    "precision_macro": report_test_dict["macro avg"]["precision"],
                    "recall_macro": report_test_dict["macro avg"]["recall"],
                    "f1_macro": report_test_dict["macro avg"]["f1-score"],
                    "precision_micro": report_test_dict["micro avg"]["precision"] if "micro avg" in report_test_dict else None,
                    "recall_micro": report_test_dict["micro avg"]["recall"] if "micro avg" in report_test_dict else None,
                    "f1_micro": report_test_dict["micro avg"]["f1-score"] if "micro avg" in report_test_dict else None
                }

                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred_test)
                cm_filename = plot_confusion_matrix(cm, experiment_name)
                mlflow.log_artifact(cm_filename)

                # Log and save model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=f"models/{experiment_name}",
                    registered_model_name=experiment_name
                )

                # End of Test run
                mlflow.end_run()

        # -----------------------------
        # 3) Update global_results
        # -----------------------------
        global_results[experiment_name] = classifier_results
        print("Done")