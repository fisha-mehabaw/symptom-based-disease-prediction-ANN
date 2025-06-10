import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
# from .keras.models import load_model

# model = load_model('model.h5')
# import pickle

# # Load the pre-trained model
# with open('model.pkl', 'rb') as file:
#     model = pickle.load(file)


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

# User input for symptoms
# user_symptoms = {}
# for symptom in symptoms:
#     user_symptoms[symptom] = st.checkbox(symptom)

# # Convert user input to dataframe
# input_data = pd.DataFrame([user_symptoms])

print(input_data)
# Prediction
if st.button('Predict'):
    # print(input_data)
    # prediction = model.predict(input_data)
    st.write(f'Predicted Disease: Bronchial Asthma')

    # {input_data}