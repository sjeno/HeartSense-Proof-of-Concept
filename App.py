#!/usr/bin/env python
# coding: utf-8

# # Deploy on Streamlit

# In[2]:


import streamlit as st
import pandas as pd
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the model
model = load('random_forest.joblib')

# Load the data
data = pd.read_csv('https://raw.githubusercontent.com/sjeno/HeartSense-Proof-of-Concept/main/final_heart_disease_uci.csv')

# Define function to take user input
def get_user_input():
    age = st.sidebar.slider('Age', 30, 80, 50)
    sex = st.sidebar.selectbox('Sex', [0, 1])
    cp = st.sidebar.selectbox('Chest Pain Type', [0, 1, 2, 3])
    trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 94, 200, 130)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 126, 564, 250)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
    restecg = st.sidebar.selectbox('Resting ECG Results', [0, 1, 2])
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 71, 202, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', [0, 1])
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 1.0)
    
    # Create a data frame from the inputs
    user_data = pd.DataFrame({'age': [age],
                              'sex': [sex],
                              'cp': [cp],
                              'trestbps': [trestbps],
                              'chol': [chol],
                              'fbs': [fbs],
                              'restecg': [restecg],
                              'thalach': [thalach],
                              'exang': [exang],
                              'oldpeak': [oldpeak]})
    return user_data

# Main function to structure the web app
def main():
    st.title('Heart Disease Predictor')
    st.write('Please input the patient information on the left side of the page:')
    
    # Get user input
    user_input = get_user_input()

    # Make predictions
    prediction = model.predict(user_input)

    # Display prediction result and explanation
    if prediction[0] == 0:
        st.write("Prediction: No heart disease detected.")
        st.write("Explanation: The model predicts that the individual does not have heart disease.")
    elif prediction[0] == 1:
        st.write("Prediction: Heart disease detected.")
        st.write("Explanation: The model predicts that the individual has heart disease.")

# Call the main function to run the app
if __name__ == '__main__':
    main()


# In[ ]:




