import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('best_model.h5')

# Define the symptom inputs
symptoms = [' continuous_sneezing', ' shivering', ' chills',
       ' watering_from_eyes', ' fatigue', ' weight_loss', ' restlessness',
       ' lethargy', ' cough', ' high_fever', ' headache', ' chest_pain',
       ' dizziness', ' loss_of_balance', ' vomiting', ' breathlessness',
       ' muscle_weakness', ' stiff_neck', ' swelling_joints',
       ' movement_stiffness']

# Title and description
st.title('Disease Prediction App')
st.write("""
    Select about 4 symptoms you feel to predict your disease.
""")

col1, col2, col3 = st.columns(3)

user_symptoms = {}
for i, symptom in enumerate(symptoms):
    if i % 3 == 0:
        user_symptoms[symptom] = col1.checkbox(symptom)
    elif i % 3 == 1:
        user_symptoms[symptom] = col2.checkbox(symptom)
    else:
        user_symptoms[symptom] = col3.checkbox(symptom)

# Convert user input to numpy array
input_data = np.array([[int(user_symptoms[s]) for s in symptoms]])
diseases = ['Allergy','Arthritis','Bronchial Asthma','Common Cold','Diabetes ','Heart attack','Hypertension ','Hypothyroidism','Tuberculosis']
# Prediction
if st.button('Predict'):
    prediction = model.predict(input_data)
    predicted_index = np.argmax(prediction)

    predicted_disease = diseases[predicted_index]
    
    st.write(f'Predicted Disease: {predicted_disease}')