import os
from huggingface_hub import HfApi, create_repo, upload_file

# Ensure HF_TOKEN is available
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set.")

api = HfApi()

HF_USERNAME = "Vaibhav84"  # Replace with your Hugging Face username
HF_SPACE_NAME = "wellness-tourism-predictor-app"

DEPLOYMENT_DIR = "deployment/"
APP_FILE = os.path.join(DEPLOYMENT_DIR, "app.py")
REQUIREMENTS_FILE = os.path.join(DEPLOYMENT_DIR, "requirements.txt")
DOCKERFILE = os.path.join(DEPLOYMENT_DIR, "Dockerfile")

# Create a new Hugging Face Space repository if it doesn't exist
try:
    create_repo(repo_id=f"{HF_USERNAME}/{HF_SPACE_NAME}", repo_type="space", space_sdk="docker", exist_ok=True, token=HF_TOKEN)
    print(f"Hugging Face Space '{HF_USERNAME}/{HF_SPACE_NAME}' created or already exists.")
except Exception as e:
    print(f"Error creating Hugging Face Space: {e}")
    print("Please ensure your HF_TOKEN has write access and your username is correct. Also ensure the space name is valid.")

# Upload deployment files to the Hugging Face Space
try:
    upload_file(
        path_or_fileobj=APP_FILE,
        path_in_repo="app.py",
        repo_id=f"{HF_USERNAME}/{HF_SPACE_NAME}",
        repo_type="space",
        token=HF_TOKEN
    )
    print(f"Uploaded {APP_FILE} to Hugging Face Space.")

    upload_file(
        path_or_fileobj=REQUIREMENTS_FILE,
        path_in_repo="requirements.txt",
        repo_id=f"{HF_USERNAME}/{HF_SPACE_NAME}",
        repo_type="space",
        token=HF_TOKEN
    )
    print(f"Uploaded {REQUIREMENTS_FILE} to Hugging Face Space.")

    upload_file(
        path_or_fileobj=DOCKERFILE,
        path_in_repo="Dockerfile",
        repo_id=f"{HF_USERNAME}/{HF_SPACE_NAME}",
        repo_type="space",
        token=HF_TOKEN
    )
    print(f"Uploaded {DOCKERFILE} to Hugging Face Space.")

    print(f"\nAll deployment files successfully pushed to Hugging Face Space: https://huggingface.co/spaces/{HF_USERNAME}/{HF_SPACE_NAME}")

except Exception as e:
    print(f"Error uploading deployment files to Hugging Face Space: {e}")
