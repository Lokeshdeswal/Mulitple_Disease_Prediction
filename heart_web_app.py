#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 16:04:14 2023

@author: dbda1112
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
heart_model = pickle.load(open(r'/home/hpcap/Downloads/DBDA/heart_xgb_model1.sav','rb'))

def heart_disease_prediction(input_data):
    #changing the input_data to numpy array
    #input_data_as_numpy_array = np.asarray(input_data)
    input_data_as_numpy_array = np.array(input_data,dtype=object)
    
    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    prediction = heart_model.predict(input_data_reshaped)
    #print(prediction)
    
    if (prediction[0]==1):
        return 'The person has heart disease'
    else:
        return 'The person does not have heart disease'
    
def main():
    #giving a title
    st.title('Heart Disease Prediction Web app')
    
    #getting the input data from the user
    
    #Ordinal features
    BMI = st.text_input('BMI value')
    PhysicalHealth = st.slider('Physical illness and injury for how many days during past 30 days?', \
                               min_value=0.0,max_value=30.0,value=31.0,step=1.0)
    MentalHealth = st.slider('For how many days in the past 30 days was your Mental Health not good?', \
                                 min_value=0.0,max_value=30.0,value=31.0,step=1.0)
    SleepTime = st.slider('Average Sleep Time in hours',min_value=1.0,max_value=24.0,value=24.0,step=1.0)
    
    #Dummies start from here
    Smoking = st.selectbox('Smoking?',['No','Yes']) #No,Yes
    AlcoholDrinking = st.selectbox('Alcohol Consumption?',['No','Yes']) #No,Yes
    Stroke = st.selectbox('Ever had a heart Stroke?',['No','Yes']) #No,Yes
    DiffWalking = st.selectbox('Difficulty in Walking?',['No','Yes']) #No,Yes
    Sex = st.selectbox('Sex?',['Female','Male']) #Female,Male
    AgeCategory = st.selectbox('Age Category? (Young<30, 30<Adult<50, 50<Old<70, 70<Very Old)',['Young','Adult','Old','Very Old']) #Adult,Old,VeryOld,Young
    Race = st.selectbox('Race?',['White', 'Black', 'Asian', 'American Indian/Alaskan Native',
       'Other', 'Hispanic']) #American Indian/Alaskan Native,Asian,Black,Hispanic,Other,White
    Diabetic = st.selectbox('Diabetic?',['No','Yes']) #No,Yes
    PhysicalActivity = st.selectbox('Physical Activity',['No','Yes']) #No,Yes
    GenHealth = st.selectbox('Overall Health?',['Very good', 'Fair', 'Good', 'Poor', 'Excellent']) #Excellent,Fair,Good,Poor,Very good
    Asthma = st.selectbox('Asthma?',['No','Yes']) #No,Yes
    KidneyDisease = st.selectbox('Kidney Disease?',['No','Yes']) #No,Yes
    SkinCancer = st.selectbox('Skin Cancer?',['No','Yes'])
    
    #Encoding the dummy features
    if Smoking == 'No':
        dum_Smoking=[1,0]
    else:
        dum_Smoking=[0,1]
        
    if AlcoholDrinking == 'No':
        dum_AlcoholDrinking=[1,0]
    else:
        dum_AlcoholDrinking=[0,1]
    
    if Stroke == 'No':
        dum_Stroke=[1,0]
    else:
        dum_Stroke=[0,1]
        
    if DiffWalking == 'No':
        dum_DiffWalking=[1,0]
    else:
        dum_DiffWalking=[0,1]
        
    if Sex == 'Female':
        dum_Sex=[1,0]
    else:
        dum_Sex=[0,1]
    
    #Adult,Old,VeryOld,Young
    if AgeCategory == 'Adult':
        dum_AgeCategory=[1,0,0,0]
    elif AgeCategory == 'Old':
        dum_AgeCategory=[0,1,0,0]
    elif AgeCategory == 'Very Old':
        dum_AgeCategory=[0,0,1,0]
    else:
        dum_AgeCategory=[0,0,0,1]
    
    #American Indian/Alaskan Native,Asian,Black,Hispanic,Other,White
    if Race == 'American Indian/Alaskan Native':
        dum_Race = [1,0,0,0,0,0]
    elif Race == 'Asian':
        dum_Race = [0,1,0,0,0,0]
    elif Race == 'Black':
        dum_Race = [0,0,1,0,0,0]
    elif Race == 'Hispanic':
        dum_Race = [0,0,0,1,0,0]
    elif Race == 'Other':
        dum_Race = [0,0,0,0,1,0]
    else:
        dum_Race = [0,0,0,0,0,1]
    
    if Diabetic == 'No':
        dum_Diabetic=[1,0]
    else:
        dum_Diabetic=[0,1]
        
    if PhysicalActivity == 'No':
        dum_PhysicalActivity=[1,0]
    else:
        dum_PhysicalActivity=[0,1]
    
    #Excellent,Fair,Good,Poor,Very good
    if GenHealth == 'Excellent':
        dum_GenHealth = [1,0,0,0,0]
    elif GenHealth == 'Fair':
        dum_GenHealth = [0,1,0,0,0]
    elif GenHealth == 'Good':
        dum_GenHealth = [0,0,1,0,0]
    elif GenHealth == 'Poor':
        dum_GenHealth = [0,0,0,1,0]
    else:
        dum_GenHealth = [0,0,0,0,1]
        
    if Asthma == 'No':
        dum_Asthma=[1,0]
    else:
        dum_Asthma=[0,1]
        
    if KidneyDisease == 'No':
        dum_KidneyDisease=[1,0]
    else:
        dum_KidneyDisease=[0,1]
    
    if SkinCancer == 'No':
        dum_SkinCancer=[1,0]
    else:
        dum_SkinCancer=[0,1]
        
    # code for Prediction
    diagnosis = ''
    
    #Defining list of features
    features = []
    ordinal_features = [BMI,PhysicalHealth,MentalHealth,SleepTime]
    dum_features = [dum_Smoking,dum_AlcoholDrinking,dum_Stroke,dum_DiffWalking,dum_Sex,
                    dum_AgeCategory,dum_Race,dum_Diabetic,dum_PhysicalActivity,
                    dum_GenHealth,dum_Asthma,dum_KidneyDisease,dum_SkinCancer]
                    
    for var in ordinal_features:
        features.append(var)
        
    for var in dum_features:
        features.extend(var)
        
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        diagnosis = heart_disease_prediction(features)
        
    st.success(diagnosis)
     
if __name__ == '__main__':
    main()