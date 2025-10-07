import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title('Course Grade Prediction')

# Load the trained models and encoders
try:
  onehot_encoder = joblib.load('onehot_encoder.pkl')
  minmax_scaler = joblib.load('minmax_scaler.pkl')
  # *** IMPORTANT ***
  # To predict a numerical grade, you need to load a regression model here.
  # The currently loaded model ('best_model.pkl') is a classifier and will output 'si' or 'no'.
  prediction_model = joblib.load('best_model.pkl')
except FileNotFoundError:
  st.error("Error: Model files not found. Please make sure 'onehot_encoder.pkl', 'minmax_scaler.pkl', and 'knn_model.pkl' are in the same directory.")
  st.stop()

st.write("Enter the student's information to predict the grade.")


# Get user inputs
felder_options = ['activo', 'visual', 'equilibrio', 'intuitivo', 'reflexivo', 'secuencial', 'sensorial', 'verbal']
selected_felder = st.selectbox('Felder Learning Style', felder_options)
examen_admision = st.number_input('University Admission Exam Score', min_value=0.0, max_value=5.0, step=0.01)

# Create a DataFrame from user inputs
input_data = pd.DataFrame({
'Felder': [selected_felder],
'Examen_admisión_Universidad': [examen_admision]
})

# Preprocess the input data
# Scale the numerical feature
# Corrected syntax: Removed the extra square bracket
input_data['Examen_admisión_Universidad'] = minmax_scaler.transform(input_data[['Examen_admisión_Universidad']])

# Apply one-hot encoding to the 'Felder' column
try:
  felder_encoded = onehot_encoder.transform(input_data[['Felder']])
  feature_names = onehot_encoder.get_feature_names_out(['Felder'])
except AttributeError:
  feature_names = [f'Felder_{cat}' for cat in onehot_encoder.categories_[0]]


felder_encoded_df = pd.DataFrame(felder_encoded, columns=feature_names, index=input_data.index)

# Concatenate the encoded features with the original DataFrame (excluding the original 'Felder' column)
processed_input_data = pd.concat([input_data.drop('Felder', axis=1), felder_encoded_df], axis=1)

# Rename the scaled column to match the training data column name
processed_input_data = processed_input_data.rename(columns={'Examen_admisión_Universidad': 'Examen_admisión_Universidad_scaled'})

# Ensure the columns are in the same order as the training data
# Assuming the training data columns order can be retrieved from the loaded model or a saved list
# For this example, we'll manually list the expected order based on previous steps
expected_columns = ['Felder_activo', 'Felder_equilibrio', 'Felder_intuitivo', 'Felder_reflexivo',
'Felder_secuencial', 'Felder_sensorial', 'Felder_verbal', 'Felder_visual',
'Examen_admisión_Universidad_scaled']


processed_input_data = processed_input_data.reindex(columns=expected_columns, fill_value=0)

# Make prediction
if st.button('Predict Grade'):
  prediction = prediction_model.predict(processed_input_data)

  # Display only the predicted value
  # Check if the prediction is numerical or categorical
  if isinstance(prediction[0], (int, float)):
    # Display as a formatted number if numerical
    st.markdown(f"<h3 style='text-align:center; color:green;'>{float(prediction[0]):.2f}</h3>", unsafe_allow_html=True)
  else:
    # Display as text if categorical (e.g., 'si' or 'no')
    st.markdown(f"<h3 style='text-align:center; color:green;'>{prediction[0]}</h3>", unsafe_allow_html=True)}
