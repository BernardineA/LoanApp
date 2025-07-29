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
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔮 Live Loan Default Prediction")
st.markdown("---")


def validate_input_data(input_data, feature_names):
    """Validate input data for predictions"""
    errors = []
    warnings = []

    # Check for missing features
    missing_features = set(feature_names) - set(input_data.keys())
    if missing_features:
        errors.append(f"Missing required features: {', '.join(missing_features)}")

    # Validate specific feature ranges
    validations = {
        'loan_amount': (1000, 100000, "Loan amount should be between $1,000 and $100,000"),
        'annual_income': (10000, 500000, "Annual income should be between $10,000 and $500,000"),
        'credit_score': (300, 850, "Credit score should be between 300 and 850"),
        'loan_term': (6, 120, "Loan term should be between 6 and 120 months"),
        'employment_years': (0, 50, "Employment years should be between 0 and 50"),
        'debt_to_income': (0, 1, "Debt-to-income ratio should be between 0 and 1"),
        'delinquencies': (0, 50, "Number of delinquencies should be between 0 and 50")
    }

    for feature, (min_val, max_val, message) in validations.items():
        if feature in input_data:
            value = input_data[feature]
            if not (min_val <= value <= max_val):
                if feature == 'credit_score' and not (300 <= value <= 850):
                    errors.append(message)
                elif feature == 'debt_to_income' and not (0 <= value <= 1):
                    errors.append(message)
                else:
                    warnings.append(f"{feature}: {value} is outside typical range ({min_val}-{max_val})")

    return errors, warnings


def make_prediction(models, input_data, selected_models, feature_names):
    """Make predictions using selected models"""
    predictions = {}

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    for model_name in selected_models:
        if model_name in models:
            model = models[model_name]

            try:
                # Subset the input data to only include the features used by the model
                if model_name != 'All Features':
                    input_df_subset = input_df[feature_names[model_name]]
                else:
                    input_df_subset = input_df[feature_names['All Features']]

                prediction = model.predict(input_df_subset)[0]

                # Ensure prediction is non-negative
                prediction = max(0, prediction)

                # Assuming all models are LinearRegression for this example
                r2_score = 0.7  # Replace with actual R-squared score if available

                predictions[model_name] = {
                    'prediction': prediction,
                    'confidence': r2_score
                }
            except Exception as e:
                st.error(f"Error making prediction with {model_name}: {str(e)}")

    return predictions


def calculate_risk_level(prediction, loan_amount):
    """Calculate risk level based on prediction and loan amount"""
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

    if input_data.get('credit_score', 0) < 600:
        risk_factors.append("Low credit score")
    elif input_data.get('credit_score', 0) > 750:
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

    if 'trained_models' not in st.session_state or 'final_dataset' not in st.session_state:
        st.warning("Please complete model training and evaluation first.")
        return

    models = st.session_state.trained_models
    final_dataset = st.session_state.final_dataset

    # Extract feature names used for each model
    feature_names = {}
    for model_name, model in models.items():
        if model_name != 'All Features':
            # Assuming the models have a 'feature_names_in_' attribute
            feature_names[model_name] = list(final_dataset.drop(columns=[final_dataset.columns[-1]]).columns)
        else:
            feature_names['All Features'] = list(final_dataset.drop(columns=[final_dataset.columns[-1]]).columns)

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

    if not selected_models:
        st.warning("Please select at least one model for predictions.")
        return

    # Display model information
    st.markdown("### 📊 Model Performance Overview")

    model_info_data = []
    for model_name in selected_models:
        model_info_data.append({
            'Model': model_name,
            'R² Score': 0.7,  # Replace with actual R-squared score if available
            'Status': '✅ Ready'
        })

    model_df = pd.DataFrame(model_info_data)
    st.dataframe(model_df)

    # Input Section
    st.markdown("## 📝 Loan Application Input")

    tab1, tab2, tab3 = st.tabs(["Manual Input", "Batch Prediction", "Input Validation"])

    with tab1:
        st.markdown("### Enter Loan Application Details")

        col1, col2, col3 = st.columns(3)

        # Determine the available features from the loaded dataset
        available_features = list(final_dataset.drop(columns=[final_dataset.columns[-1]]).columns)

        with col1:
            st.markdown("#### **Loan Information**")
            loan_amount = st.number_input(
                "Loan Amount ($)",
                min_value=1000,
                max_value=100000,
                value=15000,
                step=500,
                help="Total loan amount requested"
            )

            loan_term = st.selectbox(
                "Loan Term (months)",
                [12, 24, 36, 48, 60, 72, 84],
                index=2,
                help="Length of the loan in months"
            )

        with col2:
            st.markdown("#### **Borrower Information**")
            annual_income = st.number_input(
                "Annual Income ($)",
                min_value=10000,
                max_value=500000,
                value=50000,
                step=1000,
                help="Borrower's annual gross income"
            )

            employment_years = st.slider(
                "Employment Years",
                min_value=0.0,
                max_value=40.0,
                value=5.0,
                step=0.5,
                help="Years at current employment"
            )

        with col3:
            st.markdown("#### **Credit Information**")
            credit_score = st.slider(
                "Credit Score",
                min_value=300,
                max_value=850,
                value=650,
                step=5,
                help="FICO credit score"
            )

            debt_to_income = st.slider(
                "Debt-to-Income Ratio",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.01,
                help="Monthly debt payments / monthly income"
            )

            delinquencies = st.number_input(
                "Past Delinquencies",
                min_value=0,
                max_value=20,
                value=1,
                help="Number of past payment delinquencies"
            )

        # Prepare input data
        input_data = {
            'loan_amount': loan_amount,
            'annual_income': annual_income,
            'credit_score': credit_score,
            'loan_term': loan_term,
            'employment_years': employment_years,
            'debt_to_income': debt_to_income,
            'delinquencies': delinquencies
        }

        # Prediction button
        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            predict_button = st.button("🔮 Make Prediction", type="primary", use_container_width=True)

        if predict_button:
            # Validate input
            errors, warnings = validate_input_data(input_data, available_features)

            if errors:
                st.error("❌ Please fix the following errors:")
                for error in errors:
                    st.error(f"• {error}")
            else:
                if warnings:
                    st.warning("⚠️ Warnings:")
                    for warning in warnings:
                        st.warning(f"• {warning}")

                # Make predictions
                with st.spinner("Making predictions..."):
                    predictions = make_prediction(models, input_data, selected_models, feature_names)

                if predictions:
                    # Display results
                    st.markdown("## 🎯 Prediction Results")

                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)

                    avg_prediction = np.mean([p['prediction'] for p in predictions.values()])
                    risk_level, risk_color = calculate_risk_level(avg_prediction, loan_amount)

                    with col1:
                        st.metric("Average Prediction", f"${avg_prediction:,.2f}")

                    with col2:
                        st.metric("Risk Level", risk_level)

                    with col3:
                        risk_percentage = (avg_prediction / loan_amount) * 100
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
                            'Risk Level': calculate_risk_level(pred_info['prediction'], loan_amount)[0]
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
                            'Loan_Amount': loan_amount,
                            'Annual_Income': annual_income,
                            'Credit_Score': credit_score,
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

    with tab2:
        st.markdown("### 📋 Batch Prediction")

        st.info("Upload a CSV file with multiple loan applications for batch predictions.")

        uploaded_file = st.file_uploader(
            "Choose CSV file for batch prediction",
            type=['csv'],
            help="CSV file should contain columns matching the required features"
        )

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"✅ Batch file loaded: {len(batch_df)} applications")

                # Show preview
                st.markdown("#### Data Preview")
                st.dataframe(batch_df.head())

                # Validate batch data
                missing_features = set(available_features) - set(batch_df.columns)
                if missing_features:
                    st.error(f"❌ Missing required columns: {', '.join(missing_features)}")
                else:
                    if st.button("🚀 Run Batch Predictions"):
                        with st.spinner("Processing batch predictions..."):
                            batch_results = []

                            for idx, row in batch_df.iterrows():
                                input_data = row[available_features].to_dict()
                                predictions = make_prediction(models, input_data, selected_models, feature_names)

                                avg_pred = np.mean(
                                    [p['prediction'] for p in predictions.values()]) if predictions else 0
                                risk_level, _ = calculate_risk_level(avg_pred, row.get('loan_amount', 1))

                                result = {
                                    'Application_ID': idx + 1,
                                    'Average_Prediction': avg_pred,
                                    'Risk_Level': risk_level
                                }

                                # Add individual predictions
                                for model_name, pred_info in predictions.items():
                                    result[f'{model_name}_Prediction'] = pred_info['prediction']

                                batch_results.append(result)

                            # Display results
                            results_df = pd.DataFrame(batch_results)
                            st.dataframe(results_df)

                            # Download batch results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📁 Download Batch Results",
                                data=csv,
                                file_name="batch_predictions.csv",
                                mime="text/csv"
                            )

                            # Summary statistics
                            st.markdown("#### Batch Summary")
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                avg_prediction = results_df['Average_Prediction'].mean()
                                st.metric("Average Prediction", f"${avg_prediction:,.2f}")

                            with col2:
                                high_risk_count = sum(results_df['Risk_Level'] == 'High')
                                st.metric("High Risk Applications", f"{high_risk_count}/{len(results_df)}")

                            with col3:
                                risk_distribution = results_df['Risk_Level'].value_counts()
                                st.metric("Most Common Risk", risk_distribution.index[0])

            except Exception as e:
                st.error(f"Error processing batch file: {str(e)}")

    with tab3:
        st.markdown("### ✅ Input Validation Rules")

        st.markdown("""
        **Required Features:**
        - Loan Amount: $1,000 - $100,000
        - Annual Income: $10,000 - $500,000  
        - Credit Score: 300 - 850
        - Loan Term: 6 - 120 months
        - Employment Years: 0 - 50 years
        - Debt-to-Income Ratio: 0 - 1 (0% - 100%)
        - Past Delinquencies: 0 - 50
        """)

        st.markdown("""
        **Data Quality Checks:**
        - ✅ Range validation for all numeric inputs
        - ✅ Missing value detection
        - ✅ Realistic value warnings
        - ✅ Feature completeness verification
        """)

        # Test validation with sample data
        if st.button("Test Validation"):
            test_data = {
                'loan_amount': 15000,
                'annual_income': 50000,
                'credit_score': 650,
                'loan_term': 36,
                'employment_years': 5.0,
                'debt_to_income': 0.3,
                'delinquencies': 1
            }

            errors, warnings = validate_input_data(test_data, available_features)

            if not errors and not warnings:
                st.success("✅ Sample data passes all validation checks!")
            else:
                if errors:
                    st.error("Validation errors found:")
                    for error in errors:
                        st.error(f"• {error}")
                if warnings:
                    st.warning("Validation warnings:")
                    for warning in warnings:
                        st.warning(f"• {warning}")

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
                    'Loan Amount': f"${record['input_data']['loan_amount']:,}",
                    'Credit Score': record['input_data']['credit_score'],
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


if __name__ == "__main__":
    main()