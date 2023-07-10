#!/usr/bin/env python
# coding: utf-8

# # Streamlit App

# In[1]:


import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('https://raw.githubusercontent.com/sjeno/HeartSense-Proof-of-Concept/main/heart_disease_uci.csv')


# Split the data into features and target
X = data.drop('num', axis=1)
y = data['num']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline for numeric features
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing pipeline for categorical features
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder())
])

# Combine the numeric and categorical transformers
preprocessor = ColumnTransformer([
    ('numeric', numeric_transformer, ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']),
    ('categorical', categorical_transformer, ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'])
])

# Create the pipeline with preprocessor and classifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])


# Fit the model to the training data
pipeline.fit(X_train, y_train)

def main():
    st.title("Heart Disease Classification")
    st.sidebar.title("Features")
    
    # Get user inputs
    age = st.sidebar.slider("Age", 0, 100, 50)
    sex = st.sidebar.selectbox("Sex", ['Male', 'Female'])
    trestbps = st.sidebar.slider("Resting Blood Pressure", 0, 250, 120)
    chol = st.sidebar.slider("Cholesterol", 0, 600, 200)
    thalach = st.sidebar.slider("Maximum Heart Rate", 0, 250, 150)
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 10.0, 0.0)
    ca = st.sidebar.slider("Number of Major Vessels", 0, 4, 0)
    dataset = st.sidebar.selectbox("Dataset", ['Cleveland'])
    cp = st.sidebar.selectbox("Chest Pain", ['typical angina', 'asymptomatic', 'non-anginal', 'atypical angina'])
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [True, False])
    restecg = st.sidebar.selectbox("Resting Electrocardiographic Results", ['lv hypertrophy', 'normal'])
    exang = st.sidebar.selectbox("Exercise Induced Angina", [True, False])
    slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", ['downsloping', 'flat', 'upsloping'])
    thal = st.sidebar.selectbox("Thalassemia", ['fixed defect', 'normal', 'reversable defect'])

    # Create input data
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'trestbps': [trestbps],
        'chol': [chol],
        'thalach': [thalach],
        'oldpeak': [oldpeak],
        'ca': [ca],
        'dataset': [dataset],
        'cp': [cp],
        'fbs': [fbs],
        'restecg': [restecg],
        'exang': [exang],
        'slope': [slope],
        'thal': [thal]
    })
    
    # Make prediction
    prediction = pipeline.predict(input_data)[0]
    
    # Display prediction result and explanation
    if prediction == 0:
        st.write("Prediction: No heart disease detected.")
        st.write("Explanation: The model predicts that the individual does not have heart disease.")
    elif prediction == 1:
        st.write("Prediction: Heart disease detected.")
        st.write("Explanation: The model predicts that the individual has mild heart disease (Level 1).")
    elif prediction == 2:
        st.write("Prediction: Heart disease detected.")
        st.write("Explanation: The model predicts that the individual has moderate heart disease (Level 2).")
    elif prediction == 3:
        st.write("Prediction: Heart disease detected.")
        st.write("Explanation: The model predicts that the individual has severe heart disease (Level 3).")
    elif prediction == 4:
        st.write("Prediction: Heart disease detected.")
        st.write("Explanation: The model predicts that the individual has very severe heart disease (Level 4).")

if __name__ == '__main__':
    main()


# In[ ]:




