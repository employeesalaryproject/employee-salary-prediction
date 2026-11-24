import streamlit as st
import joblib
import pandas as pd
import sys
import sklearn.ensemble._forest as forest
import sklearn.tree._classes as tree_classes

sys.modules["sklearn.ensemble.forest"] = forest
sys.modules["sklearn.tree.tree"] = tree_classes



st.title("Employee Salary Prediction")
st.write("Welcome to Employee Salary Prediction App")

model = joblib.load("employee_salary_model_new.pkl")

company = st.selectbox("Company", ["TCS", "Infosys"])
age = st.number_input("Age", min_value=18, max_value=70, value=25)
place = st.selectbox("Place", ["Chennai", "Mumbai", "Calcutta", "Delhi"])
country = st.text_input("Country", "India")
gender = st.selectbox("Gender", [0, 1])

if st.button("Predict Salary"):

    input_data = pd.DataFrame([{
        "Company": company,
        "Age": age,
        "Place": place,
        "Country": country,
        "Gender": gender
    }])

    input_data = pd.get_dummies(input_data, drop_first=True)

    input_data = input_data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    prediction = model.predict(input_data)

    st.success(f"Predicted Salary: {prediction[0]:,.2f}")
