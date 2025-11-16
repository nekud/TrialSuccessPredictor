import streamlit as st
import pandas as pd
import numpy as np # Make sure numpy is imported
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("Custom Clinical Trial Success Predictor AI")
st.write("Upload your own clinical trial data (CSV format) to train the model and make predictions.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data successfully loaded (first 5 rows):")
    st.dataframe(data.head())

    try:
        X = data[['Disease', 'Phase', 'Inclusion Criteria 1', 'Inclusion Criteria 2']]
        y = data['Status (Success/Fail)']
        X = pd.get_dummies(X)
        
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        st.success("Model trained successfully on uploaded data.")

        st.subheader("Predict a New Trial Outcome")
        disease_input = st.selectbox("Select Disease:", data['Disease'].unique())
        phase_input = st.selectbox("Select Trial Phase:", data['Phase'].unique())
        criteria1_input = st.selectbox("Select Criteria 1:", data['Inclusion Criteria 1'].unique())
        criteria2_input = st.selectbox("Select Criteria 2:", data['Inclusion Criteria 2'].unique())

        if st.button("Predict Trial Outcome"):
            # --- FIX IS HERE ---
            # Create a DataFrame pre-filled with zeros, then set the user's choices to 1
            zero_data = np.zeros(shape=(1, len(X.columns)))
            user_input_df = pd.DataFrame(zero_data, columns=X.columns)
            
            # Set the relevant user choices to 1 (using .iloc ensures correct assignment)
            if f'Disease_{disease_input}' in user_input_df.columns: user_input_df[f'Disease_{disease_input}'].iloc[0] = 1
            if f'Phase_{phase_input}' in user_input_df.columns: user_input_df[f'Phase_{phase_input}'].iloc[0] = 1
            if f'Inclusion Criteria 1_{criteria1_input}' in user_input_df.columns: user_input_df[f'Inclusion Criteria 1_{criteria1_input}'].iloc[0] = 1
            if f'Inclusion Criteria 2_{criteria2_input}' in user_input_df.columns: user_input_df[f'Inclusion Criteria 2_{criteria2_input}'].iloc[0] = 1

            # Make the prediction
            prediction = model.predict(user_input_df)
            prediction_proba = model.predict_proba(user_input_df)
            
            # Ensure 'Success' is the class we are looking for
            if 'Success' in model.classes_:
                success_prob = prediction_proba[0, list(model.classes_).index('Success')] * 100
            else:
                # Handle cases where 'Success' might not be in the training data classes
                success_prob = 0 


            st.subheader("Prediction Result:")
            if prediction == 'Success':
                st.success(f"Prediction: **Success** (Probability: {success_prob:.1f}%)")
            else:
                st.error(f"Prediction: **Fail** (Probability: {100-success_prob:.1f}%)")

    except KeyError as e:
        st.error(f"Error: Missing expected column in the uploaded file: {e}. Ensure your CSV has 'Disease', 'Phase', 'Inclusion Criteria 1', 'Inclusion Criteria 2', and 'Status (Success/Fail)' columns.")
        
else:
    st.info("Please upload a CSV file to begin.")
