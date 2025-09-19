
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="subhash33/Tourism-Package-Model", filename="best_tourism_model.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("The Tourism Package Prediction App whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them based on their details.")
st.write("Kindly enter the customer details to check whether they are likely to purchase the new package.")

# Collect user input
Age = st.number_input("Age", min_value=18, max_value=100, value=20)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch", min_value=1, max_value=150, value=15)
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of People Visiting", min_value=1, max_value=10, value=3)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Unmarried", "Divorced", "Single"])
NumberOfTrips = st.number_input("Number of Trips", min_value=1, max_value=50, value=3)
Passport = st.selectbox("Passport", ["Yes", "No"])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
OwnCar = st.selectbox("Own Car", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=2)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=0, max_value=100000, value=25000)


# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict Button
if st.button("predict"):
    predict_proba = model.predict_proba(input_data)[0, 1]
    prediction = (predict_proba >= classification_threshold).astype(int)
    result = "Purchase the Package" if prediction == 1 else "Not likely to purchase the package"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
