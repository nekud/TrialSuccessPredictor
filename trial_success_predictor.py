import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- 1. Model Training (happens when the app starts) ---
data = pd.read_csv('trial_data.csv')
X = data[['Disease', 'Phase', 'Inclusion Criteria 1', 'Inclusion Criteria 2']]
y = data['Status (Success/Fail)']
X = pd.get_dummies(X)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# --- 2. Building the Web App Interface ---
st.title("Clinical Trial Success Predictor AI")
st.write("Enter the trial parameters to predict success probability.")

# Get user input using Streamlit components
disease_input = st.selectbox("Select Disease:", X.columns.str.split('_').str[1].unique())
phase_input = st.selectbox("Select Trial Phase:", data['Phase'].unique())
criteria1_input = st.selectbox("Select Criteria 1:", data['Inclusion Criteria 1'].unique())
criteria2_input = st.selectbox("Select Criteria 2:", data['Inclusion Criteria 2'].unique())

# --- 3. Prediction Logic ---
if st.button("Predict Trial Outcome"):
    # Prepare user input into the correct format (dummy variables)
    user_input_df = pd.DataFrame([[0]*len(X.columns)], columns=X.columns)
    user_input_df[f'Disease_{disease_input}'] = 1
    user_input_df[f'Phase_{phase_input}'] = 1
    user_input_df[f'Inclusion Criteria 1_{criteria1_input}'] = 1
    user_input_df[f'Inclusion Criteria 2_{criteria2_input}'] = 1

    # Make the prediction
    prediction = model.predict(user_input_df)
    prediction_proba = model.predict_proba(user_input_df)[0]
    success_prob = prediction_proba[list(model.classes_).index('Success')] * 100

    # Display results
    st.subheader("Prediction Result:")
    if prediction == 'Success':
        st.success(f"Prediction: **Success** (Probability: {success_prob:.1f}%)")
    else:
        st.error(f"Prediction: **Fail** (Probability: {success_prob:.1f}%)")
