import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_file_path = "heart_disease_model.pkl"
model = joblib.load(model_file_path)

# Function to preprocess input data (you should adapt this based on your preprocessing steps)
def preprocess_input(data):
    # Add your preprocessing steps here
    return data

# Function to predict
def predict(input_data):
    input_data_processed = preprocess_input(input_data)
    prediction = model.predict(input_data_processed)[0]
    return "Yes" if prediction == 1 else "No"

# Main function to run the Streamlit app
def main():
    # Set page config
    st.set_page_config(
        page_title="Heart Disease Prediction App",
        page_icon="❤️",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Define custom CSS styles
    st.markdown(
        """
        <style>
        body {
            background-color: #F0F2F6; /* Light gray background */
            color: #333333; /* Dark gray text */
        }
        .st-d3 .st-cc {
            color: #FF5733; /* Red accent color */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Heart Disease Prediction")

    # Sidebar with user inputs
    st.sidebar.header("Enter Patient Details")
    age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=25)
    sex = st.sidebar.radio("Sex", ["Male", "Female"])
    cp = st.sidebar.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    trestbps = st.sidebar.number_input("Resting BP (mm Hg)", min_value=0, max_value=300, value=120)
    chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
    fbs = st.sidebar.radio("Fasting Blood Sugar (> 120 mg/dl)", ["False", "True"])
    restecg = st.sidebar.selectbox("Resting ECG Results", ["Normal", "Abnormal", "Ventricular Hypertrophy"])
    thalach = st.sidebar.number_input("Max Heart Rate", min_value=0, max_value=300, value=150)
    exang = st.sidebar.radio("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.sidebar.number_input("ST Depression", min_value=0.0, max_value=10.0, value=0.0)
    slope = st.sidebar.selectbox("Slope of Peak ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.sidebar.selectbox("Number of Vessels Colored", ["0", "1", "2", "3"])
    thal = st.sidebar.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    # Prepare input data as a dictionary
    input_data = {
        'age': age,
        'sex': 1 if sex == "Male" else 0,
        'cp': ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(cp) + 1,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': 1 if fbs == "True" else 0,
        'restecg': ["Normal", "Abnormal", "Ventricular Hypertrophy"].index(restecg),
        'thalach': thalach,
        'exang': 1 if exang == "Yes" else 0,
        'oldpeak': oldpeak,
        'slope': ["Upsloping", "Flat", "Downsloping"].index(slope) + 1,
        'ca': int(ca),
        'thal': ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 1
    }

    # When the user clicks the Predict button
    if st.sidebar.button("Predict"):
        prediction = predict(pd.DataFrame([input_data]))
        st.subheader("Prediction")
        st.write(f"The model predicts: {prediction}")

        # Display more details about the prediction
        st.subheader("Prediction Details")
        st.write("Based on the entered details, the model predicts whether the patient is likely to have heart disease.")

        # Show the input data used for prediction
        st.subheader("Input Details")
        st.write(pd.DataFrame([input_data]))

    # Main content area
    st.subheader("About")
    st.markdown("""
    This web application predicts whether a patient is likely to have heart disease based on various medical attributes.

    **Data Source**: [Heart Disease UCI](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)  
    **Model**: Random Forest Classifier  
    **Accuracy**: Training - 1.00, Testing - 0.84  
    """)

if __name__ == "__main__":
    main()
