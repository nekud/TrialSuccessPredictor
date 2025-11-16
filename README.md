AI Clinical Trial Success Predictor 📊💊

This project is a simple, interactive web application built with Streamlit that uses a Random Forest machine learning model to predict the success probability of a clinical trial based on user-provided data. Users can upload their own CSV datasets to train a custom model instantly.
It serves as a beginner-friendly example of deploying a machine learning model to the web.

🚀 Getting Started
To run this application locally on your computer or deploy it yourself, follow these steps:

Prerequisites
You need Python 3.x installed on your system. You can install the required libraries using pip3:

bash
pip3 install streamlit pandas numpy scikit-learn

Installation and Usage
Clone the repository (if hosted on GitHub) or simply create a local folder.
Create the necessary files in your project directory:
trial_success_predictor.py: The main Python application file (copy the final code from the previous response).
trial_data.csv: A sample CSV file for testing (copy the sample data provided previously).
requirements.txt: A file listing the dependencies for deployment (see "Deployment" section).
Run the application from your terminal using streamlit:

bash
python3 -m streamlit run app.py

Open your browser to the local URL provided by Streamlit (usually http://localhost:8501).

📁 Project Structure
The project consists of three main files:

File Name	                        Description
trial_success_predictor.py	      The main Streamlit web application script. Contains UI logic, model training, and prediction functionality.
trial_data.csv	                  A sample dataset containing historical clinical trial data used for demonstration.
requirements.txt	                Lists Python dependencies required for cloud deployment (e.g., Streamlit Cloud).

✨ Features
Interactive Web UI: Built entirely using the Streamlit library.
Dynamic Model Training: The Random Forest model is trained instantly when a user uploads a CSV file.
User Input: Select trial parameters using dropdown menus generated from the uploaded data.
Success Probability: Provides a clear "Success" or "Fail" prediction along with a confidence probability percentage.
Data Upload Functionality: Allows users to return to the file upload screen easily using session state management.

🧠 The AI Model
The application uses the RandomForestClassifier from the scikit-learn library.
Data Preprocessing: It automatically uses pandas.get_dummies() to convert categorical data (like Disease names or Phase types) into numerical formats that the AI model can understand.
Training: The model.fit() command is used to train the model on the uploaded data.
Prediction: The model.predict() and model.predict_proba() functions are used to generate the final outcome and success probability.

☁️ Deployment
This application is designed to be easily deployed to the Streamlit Community Cloud by simply linking it to a GitHub repository containing the app.py file and the requirements.txt file (see "The Solution" in previous responses).
