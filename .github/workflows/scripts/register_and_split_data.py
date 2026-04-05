%%writefile tourism_project/.github/workflows/scripts/register_and_split_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
import os

# Ensure HF_TOKEN is available
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set.")

api = HfApi()

# 1. Load data from original Hugging Face dataset
ds = load_dataset("Vaibhav84/data")
df = ds['train'].to_pandas()

# Drop the 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print("Original data loaded and cleaned.")

# 2. Split data into training and testing sets
X = df.drop('ProdTaken', axis=1)
y = df['ProdTaken']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

print(f"Training set shape: {df_train.shape}")
print(f"Testing set shape: {df_test.shape}")

# 3. Convert pandas DataFrames to Hugging Face Dataset objects
hf_train_dataset = Dataset.from_pandas(df_train)
hf_test_dataset = Dataset.from_pandas(df_test)
hf_dataset_dict = DatasetDict({
    'train': hf_train_dataset,
    'test': hf_test_dataset
})

# 4. Upload the DatasetDict to the Hugging Face Hub
repo_id = "Vaibhav84/tourism_data_split"
hf_dataset_dict.push_to_hub(repo_id, token=HF_TOKEN)

print(f"Datasets successfully uploaded to Hugging Face Hub: {repo_id}")
