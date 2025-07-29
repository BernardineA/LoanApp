import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import json
from datetime import datetime
import warnings
import joblib  # For saving and loading models
import pickle  # Alternative for saving and loading

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Live Prediction", page_icon="🔮", layout="wide")

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
        background-color: #007bff;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stTextInput>label, .stNumberInput>label, .stSelectbox>label, .stSlider>label {
        color: #333;
        font-weight: bold;
    }
    .streamlit-expanderHeader {
        font-weight: bold;
    }
    /* Style for highlighting missing input fields */
    .missing-input {
        border: 2px solid red !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔮 Live Loan Default Prediction")
st.markdown("---")


def validate_input_data(input_data, feature_names):
    """Validate input data for predictions"""
    errors = []
    # Check for missing features
    missing_features = set(feature_names) - set(input_data.keys())
    if missing_features:
        errors.append(f"Missing required features: {', '.join(missing_features)}")
    return errors


def make_prediction(models, input_data, selected_models, feature_names):
    """Make predictions using selected models"""
    predictions = {}

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    for model_name in selected_models:
        if model_name in models:
            # Check if 'model' key exists before accessing it
            if 'model' not in models[model_name]:
                st.error(f"Model '{model_name}' is missing the 'model' key.  Check the evaluation page.")
                continue  # Skip this model

            model = models[model_name]['model']  # Access the model from the result

            try:
                # Subset the input data to only include the features used by the model
                if model_name != 'All Features':
                    # Ensure all required features are present in input_data
                    required_model_features = feature_names[model_name]
                    if not all(feature in input_data for feature in required_model_features):
                        missing = ", ".join(set(required_model_features) - set(input_data.keys()))
                        st.error(f"Missing features for model {model_name}.  Required features: {missing}")
                        continue  # Skip this model if features are missing

                    input_df_subset = input_df[required_model_features]
                else:
                    required_model_features = feature_names['All Features']
                    if not all(feature in input_data for feature in required_model_features):
                        missing = ", ".join(set(required_model_features) - set(input_data.keys()))
                        st.error(f"Missing features for model All Features.  Required features: {missing}")
                        continue  # Skip this model if features are missing
                    input_df_subset = input_df[required_model_features]

                prediction = model.predict(input_df_subset)[0]

                # Ensure prediction is non-negative
                prediction = max(0, prediction)

                # Assuming all models are LinearRegression for this example
                r2_score = models[model_name]['r2']  # Get R-squared from the model's evaluation result

                predictions[model_name] = {
                    'prediction': prediction,
                    'confidence': r2_score
                }
            except Exception as e:
                st.error(f"Error making prediction with {model_name}: {str(e)}")

    return predictions


def calculate_risk_level(prediction, loan_amount):
    """Calculate risk level based on prediction and loan amount"""
    if loan_amount == 0:
        return "N/A", "light"  # Or any other appropriate default

    risk_ratio = prediction / loan_amount if loan_amount > 0 else 0

    if risk_ratio < 0.05:
        return "Low", "success"
    elif risk_ratio < 0.15:
        return "Medium", "warning"
    else:
        return "High", "error"


def generate_prediction_explanation(input_data, predictions):
    """Generate explanation for the predictions"""
    explanations = []

    # Risk factors analysis
    risk_factors = []
    protective_factors = []

    if input_data.get('Credit_Score', 0) < 600:
        risk_factors.append("Low credit score")
    elif input_data.get('Credit_Score', 0) > 750:
        protective_factors.append("High credit score")

    if input_data.get('debt_to_income', 0) > 0.4:
        risk_factors.append("High debt-to-income ratio")
    elif input_data.get('debt_to_income', 0) < 0.2:
        protective_factors.append("Low debt-to-income ratio")

    if input_data.get('delinquencies', 0) > 2:
        risk_factors.append("Multiple past delinquencies")
    elif input_data.get('delinquencies', 0) == 0:
        protective_factors.append("No past delinquencies")

    if input_data.get('employment_years', 0) > 5:
        protective_factors.append("Stable employment history")
    elif input_data.get('employment_years', 0) < 2:
        risk_factors.append("Short employment history")

    explanations.append("**Risk Factors:**")
    if risk_factors:
        for factor in risk_factors:
            explanations.append(f"- {factor}")
    else:
        explanations.append("- None identified")

    explanations.append("\n**Protective Factors:**")
    if protective_factors:
        for factor in protective_factors:
            explanations.append(f"- {factor}")
    else:
        explanations.append("- None identified")

    return explanations


def main():
    st.markdown("""
    Make real-time loan default predictions using your trained machine learning models. 
    Input loan application details and get instant predictions with confidence intervals 
    and risk assessments.
    """)

    # Initialize session state
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []

    # Load models and feature names from the evaluation page
    st.markdown("## 🤖 Model Loading")

    # Check if saved_evaluation exists in session state
    if 'saved_evaluation' not in st.session_state:
        st.warning("Please complete model training and evaluation first.")
        return

    # Load data from saved_evaluation
    saved_evaluation = st.session_state.saved_evaluation
    evaluation_results = saved_evaluation['evaluation_results']
    target_col = saved_evaluation['target_col']
    df = saved_evaluation['df']

    # Extract models and feature names
    models = {}
    feature_names = {}
    for model_name, result in evaluation_results.items():
        # Assuming the model is stored in the 'model' key of the evaluation result
        # and feature names are stored in 'feature_names'
        models[model_name] = result  # Store the entire result
        feature_names[model_name] = list(df.drop(columns=[target_col]).columns)  # All features

    st.success("✅ Models and feature names loaded successfully!")

    # Model selection
    st.markdown("## 🎯 Model Selection")

    available_models = list(models.keys())
    selected_models = st.multiselect(
        "Select models for prediction:",
        available_models,
        default=available_models,
        help="Choose which models to use for making predictions"
    )

    # Ensure 'loan_amount' is always a required feature
    required_features = set()
    for model_name in selected_models:
        if model_name != 'All Features':
            required_features.update(feature_names[model_name])
        else:
            required_features.update(feature_names['All Features'])
    required_features.add('loan_amount')  # Ensure loan_amount is always required

    if not selected_models:
        st.warning("Please select at least one model for predictions.")
        return

    # Display model information
    st.markdown("### 📊 Model Performance Overview")

    model_info_data = []
    for model_name in selected_models:
        model_info_data.append({
            'Model': model_name,
            'R² Score': evaluation_results[model_name]['r2'],  # Get R² from evaluation results
            'Status': '✅ Ready'
        })

    model_df = pd.DataFrame(model_info_data)
    st.dataframe(model_df)

    # Input Section
    st.markdown("## 📝 Loan Application Input")
    st.markdown("### Enter Loan Application Details")

    # Create a dictionary to store input values
    input_data = {}
    missing_input_fields = []  # To track missing fields for highlighting

    # Add missing features to required_features
    if 'term' in df.columns:
        required_features.add('term')
    if 'ID' in df.columns:
        required_features.add('ID')
    if 'Interest_rate_spread' in df.columns:
        required_features.add('Interest_rate_spread')

    # Group features into sections for better UI
    with st.expander("Loan Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            if 'loan_amount' in required_features:
                input_data['loan_amount'] = st.number_input(
                    "Loan Amount ($)",
                    min_value=1000,
                    max_value=100000,
                    value=15000,
                    step=500,
                    help="Total loan amount requested",
                    key='loan_amount',  # Unique key for highlighting
                    format="%d"
                )
            if 'loan_term' in required_features:
                input_data['loan_term'] = st.selectbox(
                    "Loan Term (months)",
                    [12, 24, 36, 48, 60, 72, 84],
                    index=2,
                    help="Length of the loan in months",
                    key='loan_term'
                )
            if 'property_value' in required_features:
                input_data['property_value'] = st.number_input(
                    "Property Value ($)",
                    min_value=50000,
                    max_value=10000000,
                    value=500000,
                    step=10000,
                    help="Value of the property",
                    key='property_value',
                    format="%d"
                )
            if 'term' in required_features:
                input_data['term'] = st.number_input(
                    "Term",
                    min_value=1,
                    max_value=1000,
                    value=360,
                    step=1,
                    help="Term of the loan",
                    key='term',
                    format="%d"
                )
        with col2:
            if 'income' in required_features:
                input_data['income'] = st.number_input(
                    "Income ($)",
                    min_value=10000,
                    max_value=1000000,
                    value=60000,
                    step=1000,
                    help="Borrower's income",
                    key='income',
                    format="%d"
                )
            if 'LTV' in required_features:
                input_data['LTV'] = st.number_input(
                    "Loan-to-Value Ratio (LTV)",
                    min_value=0.0,
                    max_value=100.0,
                    value=70.0,
                    step=0.1,
                    help="Loan amount as a percentage of property value",
                    key='LTV'
                )
            if 'Interest_rate_spread' in required_features:
                input_data['Interest_rate_spread'] = st.number_input(
                    "Interest Rate Spread",
                    min_value=-10.0,
                    max_value=10.0,
                    value=2.0,
                    step=0.1,
                    help="Interest rate spread",
                    key='Interest_rate_spread'
                )

    with st.expander("Borrower Information", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if 'annual_income' in required_features:
                input_data['annual_income'] = st.number_input(
                    "Annual Income ($)",
                    min_value=10000,
                    max_value=500000,
                    value=50000,
                    step=1000,
                    help="Borrower's annual gross income",
                    key='annual_income',
                    format="%d"
                )
            if 'credit_score' in required_features:
                input_data['credit_score'] = st.slider(
                    "Credit Score",
                    min_value=300,
                    max_value=850,
                    value=650,
                    step=5,
                    help="FICO credit score",
                    key='credit_score'
                )
        with col2:
            if 'employment_years' in required_features:
                input_data['employment_years'] = st.slider(
                    "Employment Years",
                    min_value=0.0,
                    max_value=40.0,
                    value=5.0,
                    step=0.5,
                    help="Years at current employment",
                    key='employment_years'
                )
            if 'debt_to_income' in required_features:
                input_data['debt_to_income'] = st.slider(
                    "Debt-to-Income Ratio",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.01,
                    help="Monthly debt payments / monthly income",
                    key='debt_to_income'
                )

    with st.expander("Credit History", expanded=False):
        if 'delinquencies' in required_features:
            input_data['delinquencies'] = st.number_input(
                "Past Delinquencies",
                min_value=0,
                max_value=20,
                value=1,
                help="Number of past payment delinquencies",
                key='delinquencies',
                format="%d"
            )
        if 'Credit_Score' in required_features:
            input_data['Credit_Score'] = st.slider(
                "Credit_Score",
                min_value=300,
                max_value=850,
                value=650,
                step=5,
                help="FICO credit score",
                key='Credit_Score'
            )
        if 'ID' in required_features:
            input_data['ID'] = st.number_input(
                "ID",
                min_value=1,
                max_value=1000000,
                value=1,
                step=1,
                help="Loan ID",
                key='ID',
                format="%d"
            )

    # Prediction button
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        predict_button = st.button("🔮 Make Prediction", type="primary", use_container_width=True)

    if predict_button:
        # Validate input
        # Check for missing input fields
        for feature in required_features:
            if feature not in input_data:
                missing_input_fields.append(feature)

        errors = validate_input_data(input_data, required_features)

        if missing_input_fields or errors:
            st.error("❌ Please fix the following errors:")
            if missing_input_fields:
                st.error(f"• Missing required features: {', '.join(missing_input_fields)}")
            for error in errors:
                st.error(f"• {error}")

            # Apply CSS to highlight missing input fields
            for feature in missing_input_fields:
                st.markdown(
                    f"""
                    <script>
                        var labels = document.querySelectorAll('label');
                        labels.forEach(label => {{
                            if (label.textContent.trim() === '{feature}') {{
                                label.style.border = '2px solid red';
                            }}
                        }});
                    </script>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            # Make predictions
            with st.spinner("Making predictions..."):
                # Adjust the make_prediction function call to pass the actual models
                # and feature names from the evaluation results
                predictions = make_prediction(
                    models,  # Pass the entire models dictionary
                    input_data,
                    selected_models,
                    feature_names
                )

            if predictions:
                # Display results
                st.markdown("## 🎯 Prediction Results")

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)

                # Now we are sure that loan_amount is in input_data
                avg_prediction = np.mean([p['prediction'] for p in predictions.values()])
                risk_level, risk_color = calculate_risk_level(avg_prediction, input_data['loan_amount'])

                with col1:
                    st.metric("Average Prediction", f"${avg_prediction:,.2f}")

                with col2:
                    st.metric("Risk Level", risk_level)

                with col3:
                    risk_percentage = (avg_prediction / input_data['loan_amount']) * 100
                    st.metric("Risk Percentage", f"{risk_percentage:.1f}%")

                with col4:
                    model_agreement = len(predictions)
                    st.metric("Model Agreement", f"{model_agreement}/{len(selected_models)}")

                # Detailed predictions
                st.markdown("### 📊 Individual Model Predictions")

                prediction_data = []
                for model_name, pred_info in predictions.items():
                    prediction_data.append({
                        'Model': model_name,
                        'Prediction ($)': f"${pred_info['prediction']:,.2f}",
                        'Model R² Score': pred_info['confidence'],
                        'Risk Level': calculate_risk_level(pred_info['prediction'], input_data['loan_amount'])[0]
                    })

                pred_df = pd.DataFrame(prediction_data)
                st.dataframe(pred_df, use_container_width=True)

                # Visualization
                fig = go.Figure()

                model_names = list(predictions.keys())
                pred_values = [predictions[m]['prediction'] for m in model_names]

                fig.add_trace(go.Bar(
                    x=model_names,
                    y=pred_values,
                    name='Predictions',
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(model_names)]
                ))

                fig.add_hline(
                    y=avg_prediction,
                    line_dash="dash",
                    annotation_text=f"Average: ${avg_prediction:,.2f}",
                    annotation_position="bottom right"
                )

                fig.update_layout(
                    title='Model Predictions Comparison',
                    xaxis_title='Models',
                    yaxis_title='Predicted Default Amount ($)',
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

                # Explanation
                st.markdown("### 🔍 Prediction Explanation")

                explanations = generate_prediction_explanation(input_data, predictions)
                for explanation in explanations:
                    st.markdown(explanation)

                # Save to history
                prediction_record = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'input_data': input_data.copy(),
                    'predictions': predictions.copy(),
                    'average_prediction': avg_prediction,
                    'risk_level': risk_level
                }

                st.session_state.prediction_history.append(prediction_record)

                # Download results
                st.markdown("### 💾 Export Results")

                col1, col2 = st.columns(2)

                with col1:
                    # Prepare export data
                    export_data = {
                        'Timestamp': prediction_record['timestamp'],
                        'Loan_Amount': input_data.get('loan_amount', 0),
                        'Annual_Income': input_data.get('annual_income', 0),
                        'Credit_Score': input_data.get('credit_score', 0),
                        'Average_Prediction': avg_prediction,
                        'Risk_Level': risk_level
                    }

                    # Add individual model predictions
                    for model_name, pred_info in predictions.items():
                        export_data[f'{model_name}_Prediction'] = pred_info['prediction']

                    export_df = pd.DataFrame([export_data])
                    csv = export_df.to_csv(index=False)

                    st.download_button(
                        label="📁 Download Results CSV",
                        data=csv,
                        file_name=f"loan_prediction_{prediction_record['timestamp'].replace(':', '-').replace(' ', '_')}.csv",
                        mime="text/csv"
                    )

                with col2:
                    if st.button("📊 Add to Comparison"):
                        st.success("✅ Results added to prediction history for comparison!")

            else:
                st.error("❌ Failed to make predictions. Please check your models and input data.")

    # Prediction History
    if st.session_state.prediction_history:
        st.markdown("## 📈 Prediction History & Analysis")

        tab1, tab2 = st.tabs(["History", "Trends"])

        with tab1:
            st.markdown(f"### Recent Predictions ({len(st.session_state.prediction_history)} total)")

            # Display recent predictions
            history_data = []
            for record in st.session_state.prediction_history[-10:]:  # Last 10 predictions
                history_data.append({
                    'Timestamp': record['timestamp'],
                    'Loan Amount': f"${record['input_data'].get('loan_amount', 0):,}",
                    'Credit Score': f"${record['input_data'].get('credit_score', 0):,}",
                    'Average Prediction': f"${record['average_prediction']:,.2f}",
                    'Risk Level': record['risk_level']
                })

            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)

            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.success("Prediction history cleared!")
                st.rerun()

        with tab2:
            st.markdown("### 📊 Prediction Trends")

            if len(st.session_state.prediction_history) >= 3:
                # Extract data for plotting
                timestamps = [record['timestamp'] for record in st.session_state.prediction_history]
                predictions = [record['average_prediction'] for record in st.session_state.prediction_history]

                # Time series plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=predictions,
                    mode='lines+markers',
                    name='Predictions'
                ))

                fig.update_layout(
                    title='Prediction History Over Time',
                    xaxis_title='Timestamp',
                    yaxis_title='Predicted Default Amount ($)',
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

                # Risk level distribution
                risk_levels = [record['risk_level'] for record in st.session_state.prediction_history]
                risk_counts = pd.Series(risk_levels).value_counts()

                fig_pie = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title='Risk Level Distribution'
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Make at least 3 predictions to see trends analysis.")

    # Save data to user prediction page
    st.markdown("#### Save Data for User Prediction Page")
    if st.button("💾 Save Data for User Prediction"):
        # Save models and feature names
        for model_name, model_data in models.items():
            model_filename = f"{model_name.replace(' ', '_')}_model.joblib"
            joblib.dump(model_data['model'], model_filename)  # Save the model
            st.success(f"Model '{model_name}' saved as '{model_filename}'")

            feature_filename = f"{model_name.replace(' ', '_')}_features.pkl"
            with open(feature_filename, 'wb') as f:
                pickle.dump(feature_names[model_name], f)  # Save feature names
            st.success(f"Features for '{model_name}' saved as '{feature_filename}'")

        st.session_state.user_prediction_data = {
            'models': {model_name: f"{model_name.replace(' ', '_')}_model.joblib" for model_name in models},  # Save model filenames
            'feature_names': {model_name: f"{model_name.replace(' ', '_')}_features.pkl" for model_name in models},  # Save feature filenames
            'df': df,
            'target_col': target_col
        }
        st.success("Data saved for User Prediction Page!")


if __name__ == "__main__":
    main()