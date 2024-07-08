import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app
st.set_page_config(page_title="Loan Approval Prediction", layout="wide")

# Custom CSS for background image
page_bg_img = '''
<style>
body {
background-image: url("https://c8.alamy.com/comp/WWW9FD/loan-approved-stamp-showing-credit-agreement-ok-WWW9FD.jpg");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation: Loan Approval Project")
page = st.sidebar.radio("Go to", ["Predict approval", "Details about Data", "Workflow and Steps"])

if page == "Predict approval":
    st.title("Loan Approval Prediction")

    # Load the preprocessor and model from the file
    with open('Loan_appr_Pre_Proc_Cat_Boost.pkl', 'rb') as file:
        model = pickle.load(file)

    st.header("Input Loan Applicant Details")

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

    st.markdown("***")
    st.markdown("### Note:")
    st.markdown("The prediction is based on the provided data and a pre-trained machine learning model. Ensure that the data is entered correctly for accurate results.")

elif page == "Details about Data":
    st.title("Exploratory Data Analysis (EDA)")
    st.markdown("Here you can view some visualizations and statistics of the dataset.")

    # Path to the HTML file of the notebook
    notebook_path = 'loan-approval.html'

    # Read the HTML content of the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_html = f.read()

    # Display the HTML content
    st.components.v1.html(notebook_html, height=800, scrolling=True)


elif page == "Workflow and Steps":
    st.title("Workflow and Model Details")
    st.markdown("### Workflow")
    st.markdown("""
    1. Data Collection: Gathered data from Kaggle.
    2. Data Preprocessing: Clean removed unneccesary columns and nulls, did EDA to understand rhe data better and prepared the data for modeling.
    3. Feature Engineering: Create new features and selected important features.
    4. Model Training: Train various machine learning models and hyperparameter tune the best performing.
    5. Model Evaluation: Evaluate the performance of the models.
    6. Model Selection: Select the best-performing model.
    7. Deployment: Deploy the model for prediction.
    """)

    st.markdown("### Model Details")
    st.markdown("""
    - **Model Used**: CatBoost Regressor
    - **Hyperparameters**:
        - Learning Rate: 0.1
        - Depth: 6
        - Iterations: 1000
    - **Preprocessing**:
        - Standard Scaling for numerical features
        - One-Hot Encoding for categorical features
    """)

    # st.header("Feature Importance")
    # # Assuming you have saved feature importances
    # feature_importances = model.get_feature_importance()
    # features = new_data.columns
    # importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    # st.write(importance_df.sort_values(by='Importance', ascending=False))
