# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 18:50:32 2022

@author: santosh
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model
#loaded_model = pickle.load(open('C:/Users/santo/OneDrive/Desktop/Diabetes Prediction using Machine Learning/diabetes_model.sav', 'rb'))
loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))


# creating a function for Prediction

def diabetes_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
  
def main():
    
    
    # giving a title
    st.title('Diabetes Prediction Using Machine Learning')
    
    
    # getting the input data from the user
    
    
    Pregnancies = st.text_input('Number of Pregnancies (Ex: 0,17)')
    Glucose = st.text_input('Glucose Level (Ex: 0,199)')
    BloodPressure = st.text_input('Blood Pressure value (Ex: 0,122)')
    SkinThickness = st.text_input('Skin Thickness value (Ex: 0,99)')
    Insulin = st.text_input('Insulin Level (Ex: 0,846)')
    BMI = st.text_input('BMI value (Ex: 0,67.1)')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value (Ex: 0.078,2.42)')
    Age = st.text_input('Age of the Person (Ex: 21,81)')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()