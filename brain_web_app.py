#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 11:54:14 2023

@author: dbda1112
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
brain_model = pickle.load(open(r'/home/hpcap/Downloads/DBDA/brain_stroke_prediction_model.sav','rb'))

def brain_stroke_prediction(input_data):
    #changing the input_data to numpy array
    #input_data_as_numpy_array = np.asarray(input_data)
    input_data_as_numpy_array = np.array(input_data,dtype=object)
    
    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    prediction = brain_model.predict(input_data_reshaped)
    #print(prediction)
    
    if (prediction[0]==1):
        return 'The person has brain stroke'
    else:
        return 'The person does not have brain stroke'
    
def main():
    #giving a title
    st.title('Brain Stroke Prediction Web app')
    
    #getting the input data from the user
    
    #Ordinal features
    age = st.text_input('Age (in Years)')
    glucoseLevel = st.text_input('Average Glucose level (mg/dL)')
    BMI = st.text_input('BMI value')
    
    #Dummies start from here
    Sex = st.selectbox('Sex?',['Female','Male']) #Female,Male
    hypertension = st.selectbox('Hypertension?',['No','Yes']) #No,Yes
    HeartDisease = st.selectbox('Heart Disease?',['No','Yes']) #No,Yes
    everMarried = st.selectbox('Ever Married?',['No','Yes']) #No,Yes
    workType = st.selectbox('Work Type?',['Children','Never worked','Government Job', \
                                          'Private Job','Self employed']) #Government Job,Never worked,Private Job,Self employed,Children
    residence = st.selectbox('Residence type?',['Rural','Urban']) #Rural,Urban
    Smoking = st.selectbox('Smoking?',['Never Smoked','Formerly smoked','Smokes']) #Formerly smoked,Never Smoked,Smokes
    
    #Encoding the dummy features
    if Sex == 'Female':
        dum_Sex=[1,0]
    else:
        dum_Sex=[0,1]
        
    if hypertension == 'No':
        dum_hypertension=[1,0]
    else:
        dum_hypertension=[0,1]
    
    if HeartDisease == 'No':
        dum_HeartDisease=[1,0]
    else:
        dum_HeartDisease=[0,1]
        
    if everMarried == 'No':
        dum_everMarried=[1,0]
    else:
        dum_everMarried=[0,1]
    
    #Government Job,Never worked,Private Job,Self employed,Children
    if workType == 'Government Job':
        dum_workType =[1,0,0,0,0]
    elif workType == 'Never worked':
        dum_workType =[0,1,0,0,0]
    elif workType == 'Private Job':
        dum_workType =[0,0,1,0,0]
    elif workType == 'Self employed':
        dum_workType =[0,0,0,1,0]
    else:
        dum_workType =[0,0,0,0,1]
        
    if residence == 'Rural':
        dum_residence=[1,0]
    else:
        dum_residence=[0,1]
    
    #Formerly smoked,Never Smoked,Smokes
    if Smoking == 'Formerly smoked':
        dum_Smoking = [1,0,0]
    elif Smoking == 'Never Smoked':
        dum_Smoking = [0,1,0]
    else:
        dum_Smoking = [0,0,1]
    
        
    # code for Prediction
    diagnosis = ''
    
    #Defining list of features
    features = []
    ordinal_features = [age,glucoseLevel,BMI]
    dum_features = [dum_Sex,dum_hypertension,dum_HeartDisease,dum_everMarried,
                    dum_workType,dum_residence,dum_Smoking]
                    
                    
    for var in ordinal_features:
        features.append(var)
        
    for var in dum_features:
        features.extend(var)
        
    # creating a button for Prediction
    
    if st.button('Brain Stroke Test Result'):
        diagnosis = brain_stroke_prediction(features)
        
    st.success(diagnosis)
     
if __name__ == '__main__':
    main()