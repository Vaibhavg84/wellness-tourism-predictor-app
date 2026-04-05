# Wellness Tourism Predictor App

This repository contains a simple end-to-end pipeline for preparing tourism customer data, training a tabular classifier, registering artifacts with MLflow and Hugging Face, and deploying a Streamlit frontend to a Hugging Face Space.

The repository's GitHub Actions workflow automates three main steps:
- Data preparation & registration (splits the original dataset and pushes a split dataset to the Hugging Face Hub)
- Model training & registration (trains a scikit-learn model, logs metrics to MLflow, and uploads the model to the Hub)
- Deploy Streamlit Space (pushes the frontend app to a Hugging Face Space)

## Repo layout (important files)

- `.github/workflows/main.yml` — CI pipeline that runs the three jobs described above.
- `.github/workflows/scripts/workflow_requirements.txt` — Python dependencies used by CI jobs.
- `.github/workflows/scripts/register_and_split_data.py` — script that loads the original dataset, splits it, and uploads the split to the Hugging Face Hub.
- `.github/workflows/scripts/train_and_deploy_model.py` — trains the model, logs with MLflow, and uploads a model file to the Hub.
- `.github/workflows/scripts/deploy_streamlit_app.py` — uploads the Streamlit app files to a Hugging Face Space.
- `deployment/` — contains the Streamlit `app.py`, `requirements.txt`, and `Dockerfile` used for the Space (uploaded by the deploy script).

## High-level flow

1. The data preparation job loads the original dataset (expected at `Vaibhav84/data` on Hugging Face), cleans and splits it, then uploads a DatasetDict to `Vaibhav84/tourism_data_split`.
2. The training job downloads the split dataset `Vaibhav84/tourism_data_split`, builds a preprocessing pipeline, performs grid search for a RandomForest, logs parameters/metrics and saves the best model to the Hub at `Vaibhav84/tourism_wellness_predictor`.
3. The deploy job creates or updates a Hugging Face Space at `Vaibhav84/wellness-tourism-predictor-app` and uploads the app and requirements so the UI is available as a running Space.

## Hugging Face links (expected)
- Original dataset (used as input): https://huggingface.co/datasets/Vaibhav84/data
- Split dataset uploaded by CI: https://huggingface.co/datasets/Vaibhav84/tourism_data_split
- Trained model (model repo): https://huggingface.co/Vaibhav84/tourism_wellness_predictor
- Streamlit Space (frontend): https://huggingface.co/spaces/Vaibhav84/wellness-tourism-predictor-app

If any of these resources do not exist yet, the scripts and workflow attempt to create them (provided a valid `HF_TOKEN` with the right scope).

## Prerequisites

- Python 3.9 (CI is configured with Python 3.9)
- A Hugging Face account and an access token (see below)
- Recommended: create a virtual environment for local runs

Install dependencies (CI uses the workflow-specific requirements file):

PowerShell (from repo root):
```powershell
python -m pip install -r .github\workflows\scripts\workflow_requirements.txt
```

## Create and provide your Hugging Face token

1. Go to https://huggingface.co/settings/tokens and create a new access token. Give it the minimal scope required to push datasets/models/spaces (write/publish as needed).
2. Add the token to GitHub repository secrets so GitHub Actions can access it:
	 - Repository → Settings → Secrets and variables → Actions → New repository secret
	 - Name: `HF_TOKEN`
	 - Value: _paste token_

Alternative (CLI):
```powershell
# interactive, installs gh and authenticates first if needed
gh secret set HF_TOKEN --repo Vaibhavg84/wellness-tourism-predictor-app
```

## Run the pipeline locally (useful for debugging)

Set `HF_TOKEN` for the session (PowerShell):
```powershell
$env:HF_TOKEN = 'YOUR_HF_TOKEN_HERE'
# then run the scripts (from repo root)
python .github\workflows\scripts\register_and_split_data.py
python .github\workflows\scripts\train_and_deploy_model.py
python .github\workflows\scripts\deploy_streamlit_app.py
```

Notes:
- The scripts expect `HF_TOKEN` in the environment. If it is missing, some scripts will raise an error. Add the secret to GitHub Actions (see above) to make CI runs succeed.
- If you run the register script locally it will try to read the original dataset from `Vaibhav84/data` and push to `Vaibhav84/tourism_data_split`.

## Troubleshooting

- Error: "Could not open requirements file: ... workflow_requirements.txt"
	- Ensure the workflow step uses the correct repo-root-relative path. The workflow should install with `pip install -r .github/workflows/scripts/workflow_requirements.txt`.

- Error: `%%writefile` or `SyntaxError` at top of script
	- That indicates the script file contains Jupyter cell magic (was copied directly from a notebook). Remove the leading `%%writefile ...` line so the file is valid Python.

- Error: `HF_TOKEN environment variable not set.`
	- Add the `HF_TOKEN` secret to GitHub, or set `$env:HF_TOKEN` locally before running scripts.

- Error: `No such file or directory` when running a script in CI
	- The job runs at the repository root after checkout; make sure paths in `.github/workflows/*.yml` are repo-root-relative (for example: `.github/workflows/scripts/register_and_split_data.py`).

## Suggestions / Next steps

- Move reusable scripts out of the `.github` directory into a top-level `scripts/` or `ci/` directory and update `main.yml`. This makes local development easier and keeps `.github` focused on workflow definitions.
- Add a tiny `tests/` or `smoke_tests/` job that runs a very small portion of the pipeline with a tiny synthetic dataset to catch regressions quickly.
- If you want, I can:
	- move the scripts to `scripts/` and update the workflow for you, or
	- commit the path fixes we discussed and push them to `main`.

## Contributing

Open issues or PRs for improvements. For any change that touches workflows or secrets, run locally first and test in a fork or branch.

---
Last updated: 2026-04-05
