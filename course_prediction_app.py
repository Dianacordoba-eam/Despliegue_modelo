
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Define the path to your files in Google Drive
drive_path = "/content/drive/MyDrive/Despliegue EAM"

# Construct full paths to the model and scaler files
onehot_encoder_path = os.path.join(drive_path, 'onehot_encoder.pkl')
minmax_scaler_path = os.path.join(drive_path, 'minmax_scaler.pkl')
knn_model_path = os.path.join(drive_path, 'knn_model.pkl')


# Load the pre-trained objects
try:
    onehot_encoder = joblib.load(onehot_encoder_path)
    minmax_scaler = joblib.load(minmax_scaler_path)
    knn_model = joblib.load(knn_model_path)
except FileNotFoundError as e:
    st.error(f"Error loading model or scalers: {e}")
    st.stop() # Stop the app if files are not found

st.title("Aplicación de Predicción de Aprobación de Curso")

st.write("Esta aplicación predice si un estudiante aprobará un curso basado en su estilo de aprendizaje y puntaje de examen de admisión.")

# Input fields for user
st.header("Ingresa los datos del estudiante:")

felder_options = ['activo', 'visual', 'equilibrio', 'intuitivo', 'reflexivo', 'secuencial', 'sensorial', 'verbal'] # Add all possible felder categories
selected_felder = st.selectbox("Estilo de Aprendizaje (Felder):", felder_options)

examen_admision_score = st.number_input("Puntaje del Examen de Admisión:", min_value=0.0, max_value=5.0, step=0.01)


if st.button("Predecir"):
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'Felder': [selected_felder],
        'Examen_admisión': [examen_admision_score]
    })

    # Apply one-hot encoding to the 'Felder' column
    # Need to handle potential unseen categories during inference if they exist,
    # but for this example, we assume the input is within the trained categories.
    try:
        felder_encoded = onehot_encoder.transform(input_data[['Felder']])
        felder_encoded_df = pd.DataFrame(felder_encoded.toarray(), columns=onehot_encoder.get_feature_names_out(['Felder']))
    except ValueError as e:
        st.error(f"Error during one-hot encoding: {e}. Please ensure the selected Felder style is valid.")
        st.stop()


    # Apply min-max scaling to the 'Examen_admisión' column
    # Reshape the input for the scaler
    examen_admision_scaled = minmax_scaler.transform(input_data[['Examen_admisión']])
    input_data['Examen_admisión_Universidad_scaled'] = examen_admision_scaled


    # Prepare the final DataFrame for prediction
    # Drop the original 'Felder' and 'Examen_admisión' columns
    input_data_processed = input_data.drop(['Felder', 'Examen_admisión'], axis=1)

    # Concatenate with the one-hot encoded features
    input_data_processed = pd.concat([input_data_processed, felder_encoded_df], axis=1)

    # Ensure column order matches the training data
    # This is crucial. The columns must be in the same order as the model was trained on.
    # You might need to load the training column names or inspect the model.
    # For now, we'll assume a potential order based on previous steps, but verify this.
    # Example (replace with actual trained column order if known):
    # trained_columns = ['Felder_activo', 'Felder_equilibrio', 'Felder_intuitivo', 'Felder_reflexivo',
    #                    'Felder_secuencial', 'Felder_sensorial', 'Felder_verbal', 'Felder_visual',
    #                    'Examen_admisión_Universidad_scaled']
    # input_data_processed = input_data_processed[trained_columns]

    # A more robust way is to reindex based on the columns the encoder and scaler produced and the scaled numerical column
    # Get the columns from the one-hot encoder
    onehot_cols = list(onehot_encoder.get_feature_names_out(['Felder']))
    # Define the numerical scaled column name
    scaled_numerical_col = 'Examen_admisión_Universidad_scaled'
    # Create the expected column order (one-hot encoded columns first, then the scaled numerical)
    expected_cols = onehot_cols + [scaled_numerical_col]

    # Reindex the input data DataFrame to match the expected order
    try:
         input_data_processed = input_data_processed.reindex(columns=expected_cols, fill_value=0)
    except Exception as e:
         st.error(f"Error reordering columns: {e}. Please check the expected column names and order.")
         st.stop()


    # Make prediction
    prediction = knn_model.predict(input_data_processed)

    # Display the prediction
    st.header("Resultado de la Predicción:")
    if prediction[0] == 'si':
        st.success("El estudiante probablemente APROBARÁ el curso.")
    else:
        st.error("El estudiante probablemente NO APROBARÁ el curso.")
