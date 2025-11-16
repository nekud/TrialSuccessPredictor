import streamlit as st
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
# No need for train_test_split as we use full data for demo training

st.title("Custom Clinical Trial Success Predictor AI")
st.write("Upload your own clinical trial data (CSV format) to train the model and make predictions.")

# Initialize session state for the button if it doesn't exist
if 'clear_data' not in st.session_state:
    st.session_state.clear_data = False

# Function to reset the state when the button is clicked
def clear_session_data():
    st.session_state.clear_data = True

# --- The "Clear Data" Button ---
# This button is visible after a file is uploaded
if not st.session_state.clear_data:
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        st.button("Clear Data / Upload New File", on_click=clear_session_data)
        
        # Rest of the application logic moves inside this IF block
        data = pd.read_csv(uploaded_file)
        st.subheader("Data successfully loaded:")
        st.dataframe(data.head()) 

        try:
            # ... (Model training and prediction logic remains the same as previous example) ...
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
                zero_data = np.zeros(shape=(1, len(X.columns)))
                user_input_df = pd.DataFrame(zero_data, columns=X.columns)
                
                if f'Disease_{disease_input}' in user_input_df.columns: user_input_df[f'Disease_{disease_input}'].iloc[0] = 1
                if f'Phase_{phase_input}' in user_input_df.columns: user_input_df[f'Phase_{phase_input}'].iloc[0] = 1
                if f'Inclusion Criteria 1_{criteria1_input}' in user_input_df.columns: user_input_df[f'Inclusion Criteria 1_{criteria1_input}'].iloc[0] = 1
                if f'Inclusion Criteria 2_{criteria2_input}' in user_input_df.columns: user_input_df[f'Inclusion Criteria 2_{criteria2_input}'].iloc[0] = 1

                prediction = model.predict(user_input_df)
                prediction_proba = model.predict_proba(user_input_df)
                
                if 'Success' in model.classes_:
                    success_prob = prediction_proba[0, list(model.classes_).index('Success')] * 100
                else:
                    success_prob = 0 

                st.subheader("Prediction Result:")
                if prediction == 'Success':
                    st.success(f"Prediction: **Success** (Probability: {success_prob:.1f}%)")
                else:
                    st.error(f"Prediction: **Fail** (Probability: {100-success_prob:.1f}%)")

        except KeyError as e:
            st.error(f"Error: Missing expected column in the uploaded file: {e}.")
            
    else:
        st.info("Please upload a CSV file to begin.")

if st.session_state.clear_data:
    st.session_state.clear_data = False # Reset the state variable
    st.rerun() # Use the stable, non-experimental rerun function
