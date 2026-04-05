# Use a minimal base image with Python 3.9 installed
FROM python:3.9-slim-buster

# Set the working directory inside the container to /app
WORKDIR /app

# Copy requirements.txt and install dependencies first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all other application files to the working directory
COPY . .

# Define the command to run the Streamlit app on port "7860" (common for HF Spaces)
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
