import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the preprocessor and model from the file
model = pickle.load(open('Loan_appr_Pre_Proc_Cat_Boost.pkl','rb'))
# with open('Loan_appr_Pre_Proc_Cat_Boost.pkl', 'rb') as file:
#     model = pickle.load(file)
    

# Streamlit app
st.title("Loan Approval Prediction")

# Input fields for the new data
person_age = st.number_input('Person Age', min_value=18, max_value=100, value=30)
person_income = st.number_input('Person Income', min_value=0, value=50000)
person_home_ownership = st.selectbox('Person Home Ownership', ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
person_emp_length = st.number_input('Person Employment Length in months', min_value=0, max_value=50, value=5)
loan_intent = st.selectbox('Loan Intent', ['EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE'])
loan_grade = st.selectbox('Loan Grade', ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
loan_amnt = st.number_input('Loan Amount', min_value=0, value=10000)
loan_int_rate = st.number_input('Loan Interest Rate', min_value=0.0, value=10.0)
loan_percent_income = st.number_input('Loan Percent Income', min_value=0.0, max_value=1.0, value=0.2)
cb_person_default_on_file = st.selectbox('CB Person Default On File', ['Y', 'N'])
cb_person_cred_hist_length = st.number_input('CB Person Credit History Length', min_value=0, value=5)

# Create a DataFrame with the input values
new_data = pd.DataFrame({
    'person_age': [person_age],
    'person_income': [person_income],
    'person_home_ownership': [person_home_ownership],
    'person_emp_length': [person_emp_length],
    'loan_intent': [loan_intent],
    'loan_grade': [loan_grade],
    'loan_amnt': [loan_amnt],
    'loan_int_rate': [loan_int_rate],
    'loan_percent_income': [loan_percent_income],
    'cb_person_default_on_file': [cb_person_default_on_file],
    'cb_person_cred_hist_length': [cb_person_cred_hist_length]
})

# Make predictions
if st.button('Predict'):
    prediction = model.predict(new_data)
    st.write(f"Prediction: {'Approved' if prediction[0] == 1 else 'Rejected'}")
