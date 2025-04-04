# Machine State Classification

This repository contains all components required for the development of classifiers to predict machine states.

## Repository Structure

The repository consists of the following four main components:

### 1. **Labeling Tool**
- A graphical tool designed to label machine data.
- In this tool, machine data is visualized, and users can mark regions in the graph and assign them specific states.
- The labeled data can be exported as a CSV file.

### 2. **Data Preparation**
- Preprocessing of data, including:
  - Cleaning
  - Removing null values and duplicates
  - Normalizing the data
  - Encoding labels
- Splits the data into training, validation, and test sets, which are then exported as CSV files.
- Uses **SMOTE oversampling** to address class imbalance.

### 3. **Data Analysis**
- Performs initial analysis of the data:
  - Visualizes the distribution of variables.
  - Generates correlation matrices between variables and the target variable.

### 4. **Modeling**
- Trains, validates, and tests various machine learning models, all tracked using **MLflow**.
- Models used:
  - **[Logistic Regression](modeling/logistic_regression)**: Baseline model.
  - **[Random Forest](modeling/random_forest)**.
  - **[Multi-Layer Perceptron](modeling/multi_layer_perceptron)**.
  - **[Stochastic Gradient Descent](modeling/stochastic_gradient_descent)**.
  - **[ExtraTrees Classifier (Final Model)](modeling/extra_trees_classifier)**.

---

## Prerequisites

To run this project, ensure the following are installed on your system:

- Core Python libraries: `pandas`, `numpy`, `sklearn`, `matplotlib`, `seaborn`.
- **MLflow**: Used to track experiments and organize results. You can install it using:
  ```bash
  pip install mlflow
  ```

---

## Getting Started
Follow these steps to run the labeling tool or the MLflow UI locally on your machine.

---

### Run the Labeling Tool

1. Clone the Repository
```bash
git clone https://github.com/michellebinder/machine-state-classification.git
```

2. Navigate to the labeling_tool folder:
```bash
cd machine-state-classification/labeling_tool
```
3. Run the tool:
```bash
python tool.py
```
Alternatively, use your favorite IDE to execute the tool.py file.

---

### Run the MLflow UI
To explore the results of model experiments in MLflow:

1. Clone the Repository
```bash
git clone https://github.com/michellebinder/machine-state-classification.git
```

2. Install MLflow:
```bash
pip install mlflow
```

3. Start the MLflow UI by running the following command:
```bash
mlflow ui --backend-store-uri file:///<your-path>/machine-state-classification/mlruns
```
Replace <your-path> with the corresponding path to the mlruns folder in the cloned repository on your computer.

4. Open your browser and navigate to:
http://localhost:5000 (or your specified port).

You can now view the results for all the classifiers.
Click on a specific classifier to explore individual runs (e.g., train-test split or testing on unseen data).
Check detailed metrics and performance visualizations.
