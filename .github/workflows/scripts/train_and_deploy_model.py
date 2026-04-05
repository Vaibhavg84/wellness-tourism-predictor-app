%%writefile tourism_project/.github/workflows/scripts/train_and_deploy_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import mlflow
import mlflow.sklearn
from huggingface_hub import HfApi, create_repo, upload_file
from datasets import load_dataset
import joblib

# Set MLflow tracking URI (local tracking for GitHub Actions context)
mlflow.set_tracking_uri("file:./mlruns") # Changed to ./mlruns for CI/CD context
mlflow.set_experiment("Wellness_Tourism_Prediction")

# Ensure HF_TOKEN is available
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set.")

api = HfApi()

# Define Hugging Face repository ID for the split data
data_repo_id = "Vaibhav84/tourism_data_split"

# Load the dataset dictionary from Hugging Face
hf_dataset_reloaded = load_dataset(data_repo_id)
df_train_reloaded = hf_dataset_reloaded['train'].to_pandas()
df_test_reloaded = hf_dataset_reloaded['test'].to_pandas()

print("Train and Test data reloaded from Hugging Face.")

# Separate features (X) and target (y)
X_train = df_train_reloaded.drop('ProdTaken', axis=1)
y_train = df_train_reloaded['ProdTaken']
X_test = df_test_reloaded.drop('ProdTaken', axis=1)
y_test = df_test_reloaded['ProdTaken']

# Drop CustomerID as it's an identifier and not a feature
X_train = X_train.drop('CustomerID', axis=1)
X_test = X_test.drop('CustomerID', axis=1)

# Drop '__index_level_0__' if it exists, as it's an artifact from dataset conversion
if '__index_level_0__' in X_train.columns:
    X_train = X_train.drop('__index_level_0__', axis=1)
if '__index_level_0__' in X_test.columns:
    X_test = X_test.drop('__index_level_0__', axis=1)

# Identify numerical and categorical features
numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_train.select_dtypes(include='object').columns.tolist()

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

print("Preprocessing pipeline created successfully.")

with mlflow.start_run() as run:
    # Model Training and Hyperparameter Tuning
    rf_model = RandomForestClassifier(random_state=42)
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', rf_model)])

    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [5, 10, 15, None],
        'classifier__min_samples_split': [2, 5, 10]
    }

    print("Starting GridSearchCV...")
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print("GridSearchCV completed.")

    mlflow.log_params(grid_search.best_params_)
    print("Logged best parameters to MLflow.")

    # Log the best model to MLflow
    mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest_model")
    print("Logged best model to MLflow.")

    best_rf_model = grid_search.best_estimator_
    print("Best model found during tuning.")

    # Model Evaluation
    y_pred = best_rf_model.predict(X_test)
    y_proba = best_rf_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    })
    print(f"Logged evaluation metrics to MLflow: Accuracy={accuracy:.4f}, ROC_AUC={roc_auc:.4f}")
    print("Classification Report on Test Data:")
    print(classification_report(y_test, y_pred))

    # Upload Best Model to Hugging Face Hub
    HF_USERNAME = "Vaibhav84"  # Replace with your Hugging Face username
    HF_MODEL_NAME = "tourism_wellness_predictor"

    try:
        create_repo(repo_id=f"{HF_USERNAME}/{HF_MODEL_NAME}", repo_type="model", exist_ok=True, token=HF_TOKEN)
        print(f"Repository {HF_USERNAME}/{HF_MODEL_NAME} created or already exists.")
    except Exception as e:
        print(f"Error creating Hugging Face repository: {e}")
        print("Please ensure your HF_TOKEN has write access and your username is correct.")

    model_save_path = "best_random_forest_model.joblib" # Save locally in current working directory for CI/CD
    joblib.dump(best_rf_model, model_save_path)
    print(f"Best model saved locally to {model_save_path}.")

    try:
        upload_file(
            path_or_fileobj=model_save_path,
            path_in_repo="best_random_forest_model.joblib",
            repo_id=f"{HF_USERNAME}/{HF_MODEL_NAME}",
            repo_type="model",
            token=HF_TOKEN
        )
        print(f"Model successfully uploaded to Hugging Face Hub: https://huggingface.co/{HF_USERNAME}/{HF_MODEL_NAME}/blob/main/best_random_forest_model.joblib")

        model_card_content = f"""
---tags:
- classification
- tabular-data
- scikit-learn
---
# Wellness Tourism Predictor

This model predicts whether a customer will purchase a Wellness Tourism Package.

## Model Details
- **Algorithm**: RandomForestClassifier
- **Trained on**: Tourism customer data
- **Key Metrics (Test Set)**:
  - Accuracy: {accuracy:.4f}
  - ROC AUC: {roc_auc:.4f}

## Usage
This model can be used to identify potential customers for targeted marketing campaigns.
"""
        with open("README.md", "w") as f:
            f.write(model_card_content)
        upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=f"{HF_USERNAME}/{HF_MODEL_NAME}",
            repo_type="model",
            token=HF_TOKEN
        )
        print("Model card (README.md) uploaded to Hugging Face Hub.")

    except Exception as e:
        print(f"Error uploading model to Hugging Face Hub: {e}")
