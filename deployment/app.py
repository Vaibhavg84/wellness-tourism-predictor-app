import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

# Set page config test deployment
st.set_page_config(page_title="Wellness Tourism Package Predictor", layout="wide")

# Hugging Face Model Hub details
HF_MODEL_NAME = "tourism_wellness_predictor"
HF_USERNAME = "Vaibhav84"
MODEL_PATH_IN_REPO = "best_random_forest_model.joblib"

@st.cache_resource
def load_model():
    """Loads the pre-trained model from Hugging Face Hub."""
    try:
        # Download the model file from Hugging Face Hub
        model_local_path = hf_hub_download(
            repo_id=f"{HF_USERNAME}/{HF_MODEL_NAME}",
            filename=MODEL_PATH_IN_REPO,
            repo_type="model"
        )
        model = joblib.load(model_local_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from Hugging Face Hub: {e}")
        st.stop()

model = load_model()

st.title("🌴 Wellness Tourism Package Purchase Predictor 🧘‍♀️")
st.markdown("Enter customer details to predict if they will purchase the Wellness Tourism Package.")

# Input fields for customer details
st.header("Customer Information")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        typeofcontact = st.selectbox("Type of Contact", ['Self Inquiry', 'Company Invited'])
        citytier = st.slider("City Tier (1 is highest)", 1, 3, 2)
        occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Freelancer', 'Government'])
        gender = st.selectbox("Gender", ['Male', 'Female'])

    with col2:
        durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=10)
        numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
        preferredpropertystar = st.slider("Preferred Property Star", 1, 5, 3)
        maritalstatus = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
        numberoftrips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=5)

    with col3:
        passport = st.selectbox("Passport", [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
        pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        owncar = st.selectbox("Own Car", [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
        numberofchildrenvisiting = st.number_input("Number of Children Visiting (under 5)", min_value=0, max_value=5, value=0)
        monthlyincome = st.number_input("Monthly Income", min_value=0.0, value=50000.0)
        productpitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
        numberoffollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
        designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP', 'Director'])


    submitted = st.form_submit_button("Predict Purchase")

    if submitted:
        # Create a DataFrame from inputs
        input_data = pd.DataFrame([{
            'Age': age,
            'TypeofContact': typeofcontact,
            'CityTier': citytier,
            'DurationOfPitch': durationofpitch,
            'Occupation': occupation,
            'Gender': gender,
            'NumberOfPersonVisiting': numberofpersonvisiting,
            'PreferredPropertyStar': preferredpropertystar,
            'MaritalStatus': maritalstatus,
            'NumberOfTrips': numberoftrips,
            'Passport': passport,
            'PitchSatisfactionScore': pitchsatisfactionscore,
            'OwnCar': owncar,
            'NumberOfChildrenVisiting': numberofchildrenvisiting,
            'MonthlyIncome': monthlyincome,
            'ProductPitched': productpitched,
            'NumberOfFollowups': numberoffollowups,
            'Designation': designation
        }])

        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0, 1] # Probability of 'Yes'

            st.subheader("Prediction Result:")
            if prediction == 1:
                st.success(f"This customer is likely to purchase the Wellness Tourism Package! (Probability: {prediction_proba:.2f})")
            else:
                st.info(f"This customer is unlikely to purchase the Wellness Tourism Package. (Probability: {prediction_proba:.2f})")

            st.write("### Input Data Provided:")
            st.dataframe(input_data)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
