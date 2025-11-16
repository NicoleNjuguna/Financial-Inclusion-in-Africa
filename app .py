import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


# Title and description
st.title("Financial Inclusion in Africa")
st.write("Predict whether an individual is likely to have a bank account based on demographic and socio-economic data.")


# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    # Load dataset
    df = pd.read_csv("Financial_inclusion_dataset.csv")


    # Handle missing values
    df = df.dropna()


    # Preserve original categorical data for Streamlit form
    df_original = df.copy()


    # Encode categorical variables
    categorical_columns = ['country', 'location_type', 'cellphone_access', 'gender_of_respondent',
                          'relationship_with_head', 'marital_status', 'education_level', 'job_type']
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le


    # Features and target
    X = df.drop(columns=['bank_account', 'uniqueid', 'year'])
    y = df['bank_account'].map({'Yes': 1, 'No': 0})


    return X, y, label_encoders, df, df_original


# Train model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)


    # Save model
    joblib.dump(model, 'model.pkl')


    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)


    return model, accuracy, report


# Load data and model
X, y, label_encoders, df, df_original = load_and_preprocess_data()
model, accuracy, report = train_model(X, y)


# Display model performance
st.subheader("Model Performance")
st.write(f"Accuracy: {accuracy:.2f}")
st.text("Classification Report:\n" + report)


# Input form
st.subheader("Predict Bank Account Ownership")
with st.form(key='prediction_form'):
    country = st.selectbox("Country", df_original['country'].unique())
    location_type = st.selectbox("Location Type", df_original['location_type'].unique())
    cellphone_access = st.selectbox("Cellphone Access", df_original['cellphone_access'].unique())
    household_size = st.number_input("Household Size", min_value=1, max_value=50, value=1)
    age = st.number_input("Age of Respondent", min_value=16, max_value=100, value=18)
    gender = st.selectbox("Gender", df_original['gender_of_respondent'].unique())
    relationship = st.selectbox("Relationship with Head", df_original['relationship_with_head'].unique())
    marital_status = st.selectbox("Marital Status", df_original['marital_status'].unique())
    education = st.selectbox("Education Level", df_original['education_level'].unique())
    job_type = st.selectbox("Job Type", df_original['job_type'].unique())


    submit_button = st.form_submit_button("Predict")


# Prediction logic
if submit_button:
    # Encode inputs
    input_data = {
        'country': label_encoders['country'].transform([country])[0],
        'location_type': label_encoders['location_type'].transform([location_type])[0],
        'cellphone_access': label_encoders['cellphone_access'].transform([cellphone_access])[0],
        'household_size': household_size,
        'age_of_respondent': age,
        'gender_of_respondent': label_encoders['gender_of_respondent'].transform([gender])[0],
        'relationship_with_head': label_encoders['relationship_with_head'].transform([relationship])[0],
        'marital_status': label_encoders['marital_status'].transform([marital_status])[0],
        'education_level': label_encoders['education_level'].transform([education])[0],
        'job_type': label_encoders['job_type'].transform([job_type])[0]
    }


    # Create DataFrame for prediction
    input_df = pd.DataFrame([input_data])


    # Make prediction
    model = joblib.load('model.pkl')
    prediction = model.predict(input_df)[0]
    prediction_text = "likely to have a bank account" if prediction == 1 else "unlikely to have a bank account"


    st.success(f"The individual is {prediction_text}.")

