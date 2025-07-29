import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression  # Example model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
from datetime import datetime
import warnings
import pickle  # For loading the model
import joblib  # For loading the saved models
import os  # For file path management

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Loan Default Risk Prediction", page_icon="💰", layout="wide")

# Custom CSS for a more visually appealing layout
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to right, #f0f2f5, #e1eaf2);
    }
    .main .block-container {
        padding-top: 20px;
        padding-bottom: 20px;
    }
    .stButton>button {
        color: white;
        background-color: #2E8B57; /* Sea Green */
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #228B22; /* Forest Green */
    }
    .stTextInput>label, .stNumberInput>label, .stSelectbox>label, .stSlider>label {
        color: #333;
        font-weight: bold;
    }
    .streamlit-expanderHeader {
        font-weight: bold;
    }
    .help-icon {
        color: #888;
        cursor: help;
    }
    /* Add some color to the expander headers */
    .streamlit-expanderHeader {
        background-color: #f0f8ff; /* AliceBlue */
        border-radius: 5px;
        padding: 5px;
    }
    /* Style for the prediction results */
    .prediction-results {
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .high-risk {
        background-color: #ffe6e6; /* LightRed */
        border: 1px solid #ff3333; /* Red */
        color: #ff3333;
    }
    .low-risk {
        background-color: #e6ffe6; /* LightGreen */
        border: 1px solid #33cc33; /* Green */
        color: #33cc33;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Branding ---
st.image("https://via.placeholder.com/150", width=100)  # Replace with your logo
st.title("💰 Loan Default Risk Prediction")
st.markdown("Enter customer data to predict loan default risk.")
st.markdown("---")

# --- Load Model and Data ---
#@st.cache_resource  # Removed cache_resource decorator
def load_model_and_data():
    """Loads the trained model and necessary data from session state."""
    if 'user_prediction_data' not in st.session_state:
        st.error("Please train and save the model first.")
        return None, None, None, None, None  # Return 5 values

    data = st.session_state.user_prediction_data
    model_filenames = data['models']
    feature_filenames = data['feature_names']
    df = data['df']
    target_col = data['target_col']

    models = {}
    feature_names = {}

    for model_name, model_filename in model_filenames.items():
        try:
            models[model_name] = {'model': joblib.load(model_filename)}  # Load the model
            st.success(f"Model '{model_name}' loaded from '{model_filename}'")
        except Exception as e:
            st.error(f"Error loading model '{model_name}' from '{model_filename}': {e}")
            return None, None, None, None, None

        try:
            with open(feature_filenames[model_name], 'rb') as f:
                feature_names[model_name] = pickle.load(f)  # Load feature names
            st.success(f"Features for '{model_name}' loaded from '{feature_filenames[model_name]}'")
        except Exception as e:
            st.error(f"Error loading features for '{model_name}' from '{feature_filenames[model_name]}': {e}")
            return None, None, None, None, None

    return models, feature_names, df, target_col, None  # No preprocessor saved


models, feature_names, df, target_col, preprocessor = load_model_and_data()

# ✅ Move this check right after loading
if models is None:
    st.error("Failed to load models. Please check the training page and ensure models are saved correctly.")
    st.stop()

# Now it’s safe to do this:
selected_model_name = 'All Features'

# Check if models is a dictionary before attempting to iterate
if not isinstance(models, dict):
    st.error("The 'models' variable is not a dictionary. Please check the training page and ensure the models are being saved correctly.")
    st.stop()

if selected_model_name not in models:
    st.error(f"Model '{selected_model_name}' not found. Please check the training page.")
    st.stop()

if models[selected_model_name] is None:
    st.error(f"The model '{selected_model_name}' is not properly loaded. Please check the training page.")
    st.stop()

# --- Default Input Fields ---
default_fields = {}  # Store default field values

with st.form("loan_application_form"):
    st.header("Customer Information")

    with st.expander("Personal Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            default_fields['Customer Full Name'] = st.text_input("Customer Full Name", help="Enter the customer's full name.")
            default_fields['Gender'] = st.selectbox("Gender", options=["Male", "Female", "Other"], help="Select the customer's gender.")
            default_fields['Age'] = st.number_input("Age", min_value=18, max_value=100, value=30, help="Enter the customer's age.")
        with col2:
            default_fields['Location / Region'] = st.text_input("Location / Region", help="Enter the customer's location or region.")
            default_fields['Marital Status'] = st.selectbox("Marital Status", options=["Single", "Married", "Divorced", "Widowed"], help="Select the customer's marital status.")

    with st.expander("Employment Details"):
        col1, col2 = st.columns(2)
        with col1:
            default_fields['Employment Status'] = st.selectbox("Employment Status", options=["Employed", "Self-employed", "Unemployed"], help="Select the customer's employment status.")
            default_fields['Occupation'] = st.text_input("Occupation", help="Enter the customer's occupation.")
        with col2:
            default_fields['Monthly Income'] = st.number_input("Monthly Income", min_value=0, value=3000, help="Enter the customer's monthly income.")
            default_fields['Years of Experience'] = st.number_input("Years of Experience", min_value=0, value=5, help="Enter the customer's years of work experience.")

    with st.expander("Financial & Loan Information"):
        col1, col2 = st.columns(2)
        with col1:
            default_fields['Credit History Length (in years)'] = st.number_input("Credit History Length (in years)", min_value=0, value=5, help="Enter the length of the customer's credit history in years.")
            default_fields['Number of Existing Loans'] = st.number_input("Number of Existing Loans", min_value=0, value=1, help="Enter the number of existing loans the customer has.")
            default_fields['Loan Amount Requested'] = st.number_input("Loan Amount Requested", min_value=1000, value=10000, help="Enter the amount of loan the customer is requesting.")
        with col2:
            default_fields['Loan Purpose'] = st.selectbox("Loan Purpose", options=["Debt Consolidation", "Home Improvement", "Car Purchase", "Education", "Other"], help="Select the purpose of the loan.")
            default_fields['Loan Term (months)'] = st.selectbox("Loan Term (months)", options=[6, 12, 24, 36], help="Select the desired loan term in months.")

    # --- Load Model and Feature Names ---

    # Add a check to ensure selected_model_name exists before accessing models[selected_model_name]
    if selected_model_name in models and models[selected_model_name] is not None:
        try:
            selected_model = models[selected_model_name]['model']
        except KeyError:
            st.error(f"KeyError: Model '{selected_model_name}' not found in models dictionary.  Check the training page.")
            st.stop()
    else:
        st.error(f"Model '{selected_model_name}' not found or is None. Please check the training page.")
        st.stop()

    # Try to get feature names from the model, or use the stored feature names
    try:
        required_features = list(selected_model.feature_names_in_)  # Works for some models
    except AttributeError:
        if feature_names and selected_model_name != 'All Features':
            required_features = feature_names[selected_model_name]
        elif feature_names:
            required_features = feature_names['All Features']
        else:
            required_features = []

    # --- Dynamically Generate Missing Input Fields ---
    missing_features = [f for f in required_features if f not in default_fields]

    st.subheader("Additional Required Information")  # Section for dynamic fields

    dynamic_fields = {}  # Store dynamically generated field values

    for feature in missing_features:
        # Determine the input type based on the feature name (you might need a better mapping)
        if 'income' in feature.lower() or 'amount' in feature.lower() or 'credit' in feature.lower() or 'age' in feature.lower() or 'experience' in feature.lower():
            dynamic_fields[feature] = st.number_input(f"❌ Please enter: {feature}", help=f"This field is required for the model.", key=f"dynamic_{feature}")
        elif 'status' in feature.lower() or 'purpose' in feature.lower() or 'gender' in feature.lower():
            dynamic_fields[feature] = st.selectbox(f"❌ Please enter: {feature}", options=['Option 1', 'Option 2', 'Option 3'], help=f"This field is required for the model.", key=f"dynamic_{feature}")  # Replace options
        else:
            dynamic_fields[feature] = st.text_input(f"❌ Please enter: {feature}", help=f"This field is required for the model.", key=f"dynamic_{feature}")  # Unique key

    reset_form = st.form_submit_button("Reset Form")
    submit_button = st.form_submit_button("Predict Loan Default Risk")

    # --- Validation and Prediction Logic ---
    if submit_button:
        # --- Validate Required Fields ---
        missing_input = [f for f in missing_features if not str(dynamic_fields.get(f))]  # Check for empty dynamic fields

        if missing_input:
            st.error(f"Please fill in all required fields: {', '.join(missing_input)}")
            st.stop()

        # --- Data Preprocessing ---
        input_data = default_fields.copy()  # Start with default fields
        input_data.update(dynamic_fields)  # Add dynamic fields

        input_df = pd.DataFrame([input_data])

        # --- Feature Selection and Preprocessing ---
        try:
            # Ensure that the input DataFrame only contains the features used during training
            common_columns = [col for col in required_features if col in input_df.columns]

            if not common_columns:
                st.error("Error: No common features between input data and model features. Please check the training page and ensure all required features are provided.")
                st.stop()

            input_df = input_df[common_columns]

            if input_df.empty:
                st.error("Error: The resulting DataFrame is empty after selecting common features. Please check the training page and ensure all required features are provided.")
                st.stop()

            # Apply the preprocessing pipeline if available
            if preprocessor:
                # Ensure input_df is a DataFrame before transforming
                if not isinstance(input_df, pd.DataFrame):
                    st.error("Error: input_df is not a DataFrame.")
                    st.stop()
                input_df = preprocessor.transform(input_df)
            else:
                # Convert Categorical Features to the Correct Type if no preprocessor
                for col in input_df.columns:
                    if col in ['Gender', 'Employment Status', 'Loan Purpose', 'Marital Status']:
                        input_df[col] = input_df[col].astype('category')

        except KeyError as e:
            st.error(f"Error: Missing required feature(s) in input data: {e}. Please check the training page and ensure all required features are provided.")
            st.stop()
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")
            st.stop()

        # --- Make Prediction ---
        try:
            # Ensure input_df is in the correct format for prediction
            if not isinstance(input_df, pd.DataFrame) and not isinstance(input_df, np.ndarray):
                st.error("Error: input_df is not a DataFrame or NumPy array.")
                st.stop()

            # Check if the model has a predict_proba method (classification)
            if hasattr(selected_model, "predict_proba"):
                prediction = selected_model.predict(input_df)
                probability = selected_model.predict_proba(input_df)[0, 1]  # Probability of default
                risk_level = "High" if prediction[0] == 1 else "Low"  # Assuming 1 is High Risk

                # --- Display Results ---
                st.header("Prediction Results")

                # Use HTML to style the prediction results
                if risk_level == "High":
                    st.markdown(f"""
                        <div class="prediction-results high-risk">
                        ❗ Risk: {risk_level}
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("Based on the provided information, the loan application is considered high risk.  Further investigation is recommended.")
                else:
                    st.markdown(f"""
                        <div class="prediction-results low-risk">
                        ❗ Risk: {risk_level}
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("Based on the provided information, the loan application is considered low risk.")

                st.markdown(f"Confidence Level: {probability:.2f}")

            # If the model doesn't have predict_proba, assume it's a regression model
            else:
                predicted_amount = selected_model.predict(input_df)[0]

                # --- Helper function to translate predicted amount into risk level ---
                def amount_to_risk(amount):
                    if amount > 15000:
                        return "High"
                    elif amount > 8000:
                        return "Medium"
                    else:
                        return "Low"

                risk_level = amount_to_risk(predicted_amount)

                # --- Display Results ---
                st.header("Prediction Results")

                # Use HTML to style the prediction results
                if risk_level == "High":
                    st.markdown(f"""
                        <div class="prediction-results high-risk">
                        ❗ Risk: {risk_level}
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown(f"Predicted Default Amount: ${predicted_amount:.2f}")
                    st.markdown("Based on the predicted default amount, the loan application is considered high risk.  Further investigation is recommended.")
                elif risk_level == "Medium":
                    st.markdown(f"""
                        <div class="prediction-results">
                        ❗ Risk: {risk_level}
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown(f"Predicted Default Amount: ${predicted_amount:.2f}")
                    st.markdown("Based on the predicted default amount, the loan application is considered medium risk.")
                else:
                    st.markdown(f"""
                        <div class="prediction-results low-risk">
                        ❗ Risk: {risk_level}
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown(f"Predicted Default Amount: ${predicted_amount:.2f}")
                    st.markdown("Based on the predicted default amount, the loan application is considered low risk.")


        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.stop()

    if reset_form:
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

# --- Footer ---
st.markdown("---")
st.markdown("© 2024 Loan Prediction Services. Contact: info@loanpredictions.com")