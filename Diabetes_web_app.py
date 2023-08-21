# -*- coding: utf-8 -*-
import numpy as np
import pickle
import streamlit as st

#loading the saved model
diab_model = pickle.load(open(r'C:/Users/user/Downloads/diabetes_prediction_model.sav','rb'))

def diabatic_disease_prediction(input_data):
    #changing the input_data to numpy array
    input_data_as_numpy_array = np.array(input_data)
    
    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    prediction = diab_model.predict(input_data_reshaped)
    print(prediction)
    
    if (prediction[0]==1):
        return 'The person is Diabatic'
    else:
        return 'The person does not Diabaties'
    
    
def main():
    #giving a title
    st.title('Diabaties Prediction Web app')
    
    #getting the input data from the user
    
    #Ordinal features
    BMI = st.text_input('BMI value')
    
    MentalHealth = st.slider('For how many days in the past 30 days was your Mental Health not good?', \
                                 min_value=0.0,max_value=30.0,value=31.0,step=1.0)
    PhysicalHealth = st.slider('Physical illness and injury for how many days during past 30 days?', \
                               min_value=0.0,max_value=30.0,value=31.0,step=1.0)
    GenHlth = st.selectbox('General Health',['Excellent', 'Very good', 'Good', 'Fair','Poor'])
    Age = st.selectbox('Age(Years)',['18-24','24-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80 or older'])
    Education = st.selectbox('Education level',['Never Attended','Elementry','Secondory','Higher Secondory','Graduate','Post Graduate'])
    Income = st.selectbox('Income',['Less Than $10k','$10k-$15k','$15k-$20k','$20k-$25k','$25k-$30k','$30k-$35k','$35k-$40k','Above $40k'])
    HighBP = st.selectbox('High Blood Pressure',['No','Yes'])
    HighChol = st.selectbox('High Cholesterol',['No','Yes'])
    CholCheck = st.selectbox('Cholesterol check in 5 years',['No','Yes']) 
    Smoking = st.selectbox('Smoking?',['No','Yes']) #No,Yes
    Stroke = st.selectbox('Stroke?',['No','Yes']) #No,Yes
    HeartDiseaseorAttack = st.selectbox('Suffering from heart disease or Heart Attact',['No','Yes'])
    PhysicalActivity = st.selectbox('Physical Activity',['No','Yes']) #No,Yes
    Fruits = st.selectbox('Consume Fruit 1 or more times per day',['No','Yes'])
    Veggies = st.selectbox('Consume Vegetables 1 or more times per day',['No','Yes'])
    AlcoholDrinking = st.selectbox('Alcohol Consumption?',['No','Yes']) #No,Yes
    AnyHealthcare = st.selectbox('Health Insurance',['Yes','No'])
    NoDocbcCost = st.selectbox('Inaccessible healthcare due to cost recently ?',['Yes','No'])
    DiffWalking = st.selectbox('Difficulty in Walking?',['No','Yes']) #No,Yes
    Sex = st.selectbox('Sex?',['Female','Male']) #Female,Male
    
    gen_hlth_mapping = {'Excellent': 1, 'Very good': 2, 'Good': 3, 'Fair': 4}
    dum_GenHlth = gen_hlth_mapping.get(GenHlth, 5)
    age_mapping = {'18-24': 1,'24-29': 2,'30-34': 3,'35-39': 4,'40-44': 5,'45-49': 6,'50-54': 7,'55-59': 8,'60-64': 9,'65-69': 10,'70-74': 11,'75-79': 12}
    dum_Age = age_mapping.get(Age, 13)
    education_mapping = {'Never Attended': 1,'Elementary': 2,'Secondary': 3,'Higher Secondary': 4,'Graduate': 5}

    income_mapping = {'Less Than $10k': 1,'$10k-$15k': 2,'$15k-$20k': 3,'$20k-$25k': 4,'$25k-$30k': 5,'$30k-$35k': 6,'$35k-$40k': 7}
    dum_Education = education_mapping.get(Education,6)  
    dum_Income = income_mapping.get(Income,8)      
    
    
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
   
    
   
    if HighBP == 'No':
        dum_HighBP=[1,0]
    else:
        dum_HighBP=[0,1]
    
    if HighChol == 'No':
        dum_HighChol=[1,0]
    else:
        dum_HighChol=[0,1]

    if CholCheck == 'No':
        dum_CholCheck=[1,0]
    else:
        dum_CholCheck=[0,1]
    
    if HeartDiseaseorAttack == 'No':
        dum_HeartDiseaseorAttack=[1,0]
    else:
        dum_HeartDiseaseorAttack=[0,1]
    
    if PhysicalActivity == 'No':
        dum_PhysicalActivity=[1,0]
    else:
        dum_PhysicalActivity=[0,1]
        
    
    if Fruits == 'No':
        dum_Fruits=[1,0]
    else:
        dum_Fruits=[0,1]
    
    if Veggies == 'No':
        dum_Veggies=[1,0]
    else:
        dum_Veggies=[0,1]

    if AnyHealthcare == 'No':
        dum_AnyHealthcare=[1,0]
    else:
        dum_AnyHealthcare=[0,1]
    
    if NoDocbcCost == 'No':
        dum_NoDocbcCost=[1,0]
    else:
        dum_NoDocbcCost=[0,1]
    
    
    # code for Prediction
    diagnosis = ''
    
    #Defining list of features
    features = []
    dum_features = [dum_HighBP,dum_HighChol,dum_CholCheck,dum_Smoking,dum_Stroke,
                    dum_HeartDiseaseorAttack,dum_PhysicalActivity,dum_Fruits,dum_Veggies,
                    dum_AlcoholDrinking , dum_AnyHealthcare,dum_NoDocbcCost,dum_DiffWalking,dum_Sex ]
    features.append(BMI)
    features.append(dum_GenHlth)
    features.append(MentalHealth)
    features.append(PhysicalHealth)
    features.append(dum_Age)
    features.append(dum_Education)
    features.append(dum_Income)
    for var in dum_features:
        features.extend(var)
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabatic_disease_prediction(features)
        
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()    

    
    