import streamlit as st 
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model = pickle.load(open(r"D:\NareshIT\Spyder_Homework\Multiple linear regression\MLR_model_for_Investments.pkl", 'rb'))

# Get the feature names from the model
feature_names = model.feature_names_in_

# Set the title of the Streamlit app
st.title("Investment Profile Predection")

st.write("This app utilizes a Multi-Linear Regression model to predict investment outcomes based on various factors like marketing spend, research, promotions.")

# Collect input data from the user
digital_marketing = st.number_input("Enter Spent on Digital Marketing", min_value=0)
promotion = st.number_input("Enter Spent on Promotions", min_value=0)
research = st.number_input("Enter Spent on Research", min_value=0)

#Lisit of Cities
cities = ["Hyderebad","Mumbai","Chennai"]
city = st.selectbox("Select the city", cities)

# Prepare the input data for prediction
X_input = pd.DataFrame({
    'DigitalMarketing': [digital_marketing],
    'Promotion': [promotion],
    'Research': [research],
    'State_Bangalore': [0],
    'State_Chennai': [0],
    'State_Hyderabad': [0]
})

# One-hot encode the selected city
if city == 'Hyderabad':
    X_input['State_Hyderabad'] = 1
elif city == 'Chennai':
    X_input['State_Chennai'] = 1
elif city == 'Banglore':
    X_input['State_Bangalore'] = 1

# Reorder the columns to match the training data columns exactly
X_input = X_input[feature_names]

# # Display the prepared input data (for verification)
# st.write("Input data for prediction:")
# st.write(X_input)

# Predict the profit using the trained model
if st.button('Predict Profit'):
    profit_prediction = model.predict(X_input)
    # Display the predicted profit
    st.write(f"Predicted Profit: ${profit_prediction[0]:,.2f}")
