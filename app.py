import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
# from keras.models import load_model
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
# from sklearn.preprocessing import MinMaxScaler


st.title('NCDs Prediction')


sections = ["Breast_Cancer", "Cervical_Cancer", "Heart_Diseases", "Diabetes"]

st.sidebar.title("Surported NCDs")
for each in sections:
    st.sidebar.markdown(f"<a href='#{''.join(each.split())}' style='text-decoration:none; font-style: normal; font-weight: 500; font-size: 18px; color: #012C3D;'>{each} </a>", unsafe_allow_html=True)



# section_1
st.markdown(f"<div id='Breast_Cancer'></div>", unsafe_allow_html=True)
st.header('You are making predictions for Breast Cancer')

concave_points_worst = st.number_input('Concave Points Worst - e.g 0.000, 0.2910 ')

concave_points_mean = st.number_input('Concave Points Mean - e.g 0.000, 0.2910 ')

perimeter_worst	 = st.number_input('Perimeter Worst - e.g 50.42, 251.1', step=1.0)

perimeter_mean	= st.number_input('Perimeter Mean - e.g 43.98, 188.79', step=1.0)

radius_worst	= st.number_input('Radius Worst - e.g 7.61, 36.35', step=1.0)

radius_mean	= st.number_input('Radius Mean - e.g  6.98, 28.11', step=1.0)

area_worst	= st.number_input('Area Worst - e.g 4254.0, 185.2', step=1.0)

area_mean = st.number_input('Area Mean - e.g 143.5, 2501.0', step=1.0)

df = pd.DataFrame(np.array([[concave_points_worst, concave_points_mean, perimeter_worst, perimeter_mean, radius_worst, radius_mean, area_worst, area_mean]]), columns=['concave points_worst', 'concave points_mean', 'perimeter_worst', 'perimeter_mean', 'radius_worst', 'radius_mean', 'area_worst', 'area_mean'])
# st.write(df)
bcancer_model = load_model('bcancer_model.h5')


if st.button('Predict Breast Cancer'): 
    pred = bcancer_model.predict(df)
    if pred > 0.5:
        st.info('This patient is at risk of Breast Cancer! With a probability of: {}'.format(pred[0][0]) )
    else:
        st.info('This patient is not at risk Diabetes! With a probability of: {}'.format(pred[0][0]) )

# section_2
st.markdown(f"<div id='Cervical_Cancer'></div>", unsafe_allow_html=True)
st.header('You are making prediction for cervical cancer')

dx	 = st.number_input('DX - Input is Either 1 (YES) or 0 (NO)', 0,1,0)
dx_CIN	= st.number_input('Dx:CIN - Input is Either 1 (YES) or 0 (NO)', 0,1,0)
dx_Cancer = st.number_input('DX:Cancer - Input is Either 1 (YES) or 0 (NO)', 0,1,0)
dx_HPV	 = st.number_input('DX:HPV - Input is Either 1 (YES) or 0 (NO)', 0,1,0)
STDs = st.number_input('STDs - Input is Either 1 (YES) or 0 (NO)', 0,1,0)	
no_STDs= st.number_input('STDs (number) - Number of STDs', 0, 20, 0)
genital_herpes = st.number_input('STDs:genital herpes - Input is Either 1 (YES) or 0 (NO)' , 0,1,0)
hiv = st.number_input('STDs:HIV - Input is Either 1 (YES) or 0 (NO)Input is Either 1 or 0', 0,1,0)

dfc = pd.DataFrame(np.array([[dx_Cancer,dx_HPV, dx, genital_herpes,	hiv, STDs, dx_CIN, no_STDs]]), columns=['Dx:Cancer','Dx:HPV','Dx','STDs:genital herpes','STDs:HIV','STDs','Dx:CIN','STDs (number)'])
# st.write(dfc)
ccancer_model = load_model('ccancer_model.h5')
if st.button('Predict Cervical Cancer'): 
    cpred = ccancer_model.predict(dfc)

    if cpred > 0.5:
        st.info('This patient is at risk of Cervical Cancer! With a probability of: {}'.format(cpred[0][0]) )
    else:
        st.info('This patient is not at risk of Cervical Cancer! With a probability of: {}'.format(cpred[0][0]) )
# section_3
st.markdown(f"<div id='Heart Diseases'></div>", unsafe_allow_html=True)
st.header('You are making prediction for Heart Disease')
exang = st.number_input('Exercise induced angina (1 = yes; 0 = no)' , 0,1,0)
st.header('Chest Pain Types')
st.write('0: Typical angina: chest pain related decrease blood supply to the heart')
st.write('1: Atypical angina: chest pain not related to heart')
st.write('2: Non-anginal pain: typically esophageal spasms (non heart related)')
st.write('3: Asymptomatic: chest pain not showing signs of disease')
cp = st.number_input('Chest Pain Type', 0,3,0)
oldpeak	 = st.number_input('ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will stress more', min_value=0.0, max_value= 1.0, step= 1.0)
thalach	= st.number_input('Maximum heart rate achieved - e.g 71, 202, 152')
ca	= st.number_input('number of major vessels colored by flourosopy - e.g 0 - 3')
st.header('Slope peak during exercise')
st.write('0: Upsloping: better heart rate with excercise (uncommon)')
st.write('1: Flatsloping: minimal change (typical healthy heart')
st.write('2: Downslopins: signs of unhealthy heart')
slope = st.number_input('The slope of the peak exercise ST segment', 0,2,0)	
thal = st.number_input('Thalium stress result -  Input an interger (Note:3 = normal; 6 = fixed defect; 7 = reversable defect)', step = 1.0)	
sex = st.number_input('Sex - 1 = male; 0 = female',0,1,0)
dfh = pd.DataFrame(np.array([[exang	,cp	,oldpeak	,thalach,	ca,	slope,thal,	sex]]), columns=['exang'	,'cp'	,'oldpeak'	,'thalach',	'ca',	'slope','thal',	'sex'])
# st.write(dfh)
heart_model = load_model('heart_model.h5')
if st.button('Predict Heart Disease'): 
    predh = heart_model.predict(dfh)

    if predh > 0.5:
        st.info('This patient is at risk of Heart disease! With a probability of: {}'.format(predh[0][0]) )
    else:
        st.info('This patient is not at risk of Heart disease! With a probability of: {}'.format(predh[0][0]) )
# section_4
st.markdown(f"<div id='Diabetes'></div>", unsafe_allow_html=True)
st.header('You are making prediction for Diabetes')
pregnancies	= st.number_input('Number of times pregnant- e.g 3, 1', step=1.0)
glucose	= st.number_input('Plasma glucose concentration a 2 hours in an oral glucose tolerance test - e.g 72, 145', step=1.0)
bloodPressure = st.number_input('Diastolic blood pressure (mm Hg) - e.g 72, 55', step=1.0)
SkinThickness = st.number_input('Triceps skin fold thickness (mm) - e.g 18, 90', step=1.0)
Insulin	 = st.number_input('2-Hour serum insulin (mu U/ml) - e.g 55, 130', step=1.0)
BMI	= st.number_input('Body mass index (weight in kg/(height in m)^2 - e.g 31.7, 67.7)', step=1.0)
DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function - e.g 0.08, 2.42')	
Age = st.number_input("Age in years", step=1.0)
dfd = pd.DataFrame(np.array([[pregnancies, glucose, bloodPressure, SkinThickness, Insulin, BMI,	DiabetesPedigreeFunction,Age]]), columns=['Pregnancies',	'Glucose',	'BloodPressure',	'SkinThickness',	'Insulin',	'BMI',	'DiabetesPedigreeFunction',	'Age'])
# st.write(dfc)
diabetes_model = load_model('diabetes_model.h5')
if st.button('Predict Diabetes'): 
    predd = diabetes_model.predict(dfd)
    if predd > 0.5:
        st.info('This patient is at risk of Diabetes! With a probability of: {}'.format(predd[0][0]) )
    else:
        st.info('This patient is not at risk of Diabetes! With a probability of: {}'.format(predd[0][0]) )
    # st.checkbox('I consent to give my data')