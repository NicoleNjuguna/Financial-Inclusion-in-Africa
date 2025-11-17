import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

st.set_page_config(page_title="Financial Inclusion Prediction", layout="wide")

st.title("Financial Inclusion in Africa")
st.write("Use demographic and socio-economic data to estimate the likelihood of someone owning a bank account.")

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("Financial_inclusion_dataset.csv")
    df = df.dropna()

    df_original = df.copy()

    categorical_columns = [
        'country', 'location_type', 'cellphone_access',
        'gender_of_respondent', 'relationship_with_head',
        'marital_status', 'education_level', 'job_type'
    ]

    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop(columns=['bank_account', 'uniqueid', 'year'])
    y = df['bank_account'].map({'Yes': 1, 'No': 0})

    return X, y, label_encoders, df_original

# Train model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, 'model.pkl')

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return model, accuracy, report

X, y, label_encoders, df_original = load_and_preprocess_data()
model, accuracy, report = train_model(X, y)

# Better performance display
st.subheader("Model Performance")

col1, col2 = st.columns(2)
col1.metric("Accuracy", f"{accuracy:.2f}")
col2.write("Classification Report")
col2.code(report)

# Input section
st.subheader("Predict Bank Account Ownership")
st.write("Provide the individual's details below.")

with st.form(key="prediction_form", clear_on_submit=False):
    colA, colB = st.columns(2)

    with colA:
        country = st.selectbox("Country", df_original['country'].unique())
        location_type = st.selectbox("Location Type", df_original['location_type'].unique())
        cellphone_access = st.selectbox("Cellphone Access", df_original['cellphone_access'].unique())
        household_size = st.number_input("Household Size", 1, 50, 1)

    with colB:
        age = st.number_input("Age of Respondent", 16, 100, 18)
        gender = st.selectbox("Gender", df_original['gender_of_respondent'].unique())
        relationship = st.selectbox("Relationship with Head", df_original['relationship_with_head'].unique())
        marital_status = st.selectbox("Marital Status", df_original['marital_status'].unique())
        education = st.selectbox("Education Level", df_original['education_level'].unique())
        job_type = st.selectbox("Job Type", df_original['job_type'].unique())

    submitted = st.form_submit_button("Predict")

# Prediction
if submitted:
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
        'job_type': label_encoders['job_type'].transform([job_type])[0],
    }

    input_df = pd.DataFrame([input_data])
    model = joblib.load('model.pkl')

    prediction = model.predict(input_df)[0]
    result = "Likely to have a bank account" if prediction == 1 else "Unlikely to have a bank account"

    st.success(result)
