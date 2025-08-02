import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import time
import warnings
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Model Evaluation", page_icon="📊", layout="wide")

st.title("📊 Model Evaluation with K-Fold Cross-Validation")
st.markdown("---")


@st.cache_resource
def evaluate_model(_model, X, y, cv=5):
    """Evaluate model using K-Fold cross-validation and return metrics and predictions."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    start_time = time.time()

    # Perform cross-validation and get predictions
    y_pred = cross_val_predict(_model, X, y, cv=kf, n_jobs=-1)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    evaluation_time = time.time() - start_time

    return {
        'rmse': rmse,
        'r2': r2,
        'y_pred': y_pred,
        'evaluation_time': evaluation_time
    }


def plot_predicted_vs_actual(y_true, y_pred, model_name):
    """Plot predicted vs actual values."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(opacity=0.7)
    ))

    # Add a line for perfect predictions
    fig.add_trace(go.Scatter(
        x=[min(y_true), max(y_true)],
        y=[min(y_true), max(y_true)],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title=f'Predicted vs Actual - {model_name}',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        height=500,
        template="plotly_white"  # Use a white background for better readability
    )

    # Add description
    st.markdown("This chart visualizes the relationship between the actual and predicted values for the selected model.  Ideally, the points should cluster closely around the red dashed line, which represents perfect predictions.  Deviations from this line indicate prediction errors.")

    return fig


def generate_evaluation_report(evaluation_results, target_col, cv_folds):
    """Generates a report explaining model evaluation results."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Model Evaluation Report", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))

    # Introduction
    story.append(Paragraph("This report summarizes the model evaluation process using K-Fold Cross-Validation.", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Configuration Information
    story.append(Paragraph("<b>Configuration Information:</b>", styles['h2']))
    story.append(Paragraph(f"Target Variable: {target_col}", styles['Normal']))
    story.append(Paragraph(f"Cross-Validation Folds: {cv_folds}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Evaluation Results
    story.append(Paragraph("<b>Evaluation Results:</b>", styles['h2']))
    story.append(Paragraph("The following table shows the evaluation metrics for each model:", styles['Normal']))

    # Prepare data for the table
    table_data = []
    for model_name, result in evaluation_results.items():
        table_data.append({
            'Model': model_name,
            'RMSE': result['rmse'],
            'R²': result['r2'],
            'Evaluation Time (s)': result['evaluation_time']
        })

    # Convert DataFrame to string for report
    results_df = pd.DataFrame(table_data)
    table_string = results_df.to_string()
    story.append(Paragraph(table_string, styles['Code']))
    story.append(Spacer(1, 0.2 * inch))

    # Best Model Highlight
    best_model_name = results_df.loc[results_df['R²'].idxmax(), 'Model']
    best_score = results_df['R²'].max()
    story.append(Paragraph(f"<b>Best Model:</b> {best_model_name} with R² of {best_score:.4f}", styles['h3']))
    story.append(Spacer(1, 0.2 * inch))

    doc.build(story)
    buffer.seek(0)
    return buffer


def main():
    st.markdown("""
    This section evaluates the performance of your trained models using K-Fold Cross-Validation. 
    We calculate Root Mean Squared Error (RMSE) and R-Squared to assess model accuracy. 
    Visualizations of predicted vs actual values provide insights into model performance.
    """)

    # Initialize session state
    if 'final_dataset' not in st.session_state:
        st.session_state.final_dataset = None
    if 'model_results' not in st.session_state:
        st.session_state.model_results = []
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = {}

    # Data Loading Section
    st.markdown("## 📁 Data Loading")

    # Load data from session state if available
    df = st.session_state.final_dataset
    if df is not None:
        st.success("✅ Loaded selected features dataset from previous step.")
    else:
        st.info("👆 Please complete model training first.")
        return

    # Configuration
    st.markdown("## ⚙️ Evaluation Configuration")

    numeric_cols = []
    if df is not None:  # Check if df is not None before proceeding
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        st.warning("No dataset loaded. Please complete model training first.")
        return

    col1, col2 = st.columns(2)

    with col1:
        target_col = st.selectbox(
            "Select target variable:",
            numeric_cols,
            index=0,
            help="Choose the target variable for evaluation. This should be the same target used during model training."
        )

    with col2:
        cv_folds = st.slider(
            "Cross-validation folds:",
            3, 10, 5,
            help="The number of folds to use for K-Fold Cross-Validation.  A higher number of folds generally provides a more reliable estimate of model performance."
        )

    if target_col:
        # Prepare data
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X = X.select_dtypes(include=[np.number])

        # Scaling option
        scale_features = st.checkbox(
            "Scale features",
            value=True,
            help="Scale the features using StandardScaler. This is generally recommended for models that are sensitive to feature scaling, such as linear regression."
        )

        if scale_features:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values

        st.success(f"Features: **{len(X.columns)}** | Target: **{target_col}** | Samples: **{len(df)}**")

        # Model Evaluation Section
        st.markdown("## 📊 Model Evaluation")

        # Check if model_results exists in session state
        if 'model_results' in st.session_state and st.session_state.model_results:
            # Extract model names from model_results
            model_names = [result['Feature Selection'] for result in st.session_state.model_results]

            selected_models = st.multiselect(
                "Select models to evaluate:",
                model_names,
                default=model_names[:min(2, len(model_names))],
                help="Choose the models you want to evaluate using K-Fold Cross-Validation."
            )

            if st.button("🚀 Start Evaluation", type="primary"):
                evaluation_results = {}

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, model_name in enumerate(selected_models):
                    status_text.text(f"Evaluating {model_name}...")

                    with st.spinner(f"Evaluating {model_name}..."):
                        # Find the model result that matches the selected model name
                        model_result = next((result for result in st.session_state.model_results if result['Feature Selection'] == model_name), None)

                        if model_result is None:
                            st.error(f"Model '{model_name}' not found in trained models.")
                            continue

                        model = model_result['Model']
                        feature_names = model_result['Feature Names']

                        # Determine the data to use based on the model type
                        if model_name == 'All Features':
                            X_eval = X.values
                        else:
                            X_eval = df[feature_names].values

                        # Evaluate the model
                        evaluation = evaluate_model(model, X_eval, y, cv_folds)

                        evaluation_results[model_name] = {
                            'model': model,  # Store the model itself
                            'rmse': evaluation['rmse'],
                            'r2': evaluation['r2'],
                            'y_pred': evaluation['y_pred'],
                            'evaluation_time': evaluation['evaluation_time']
                        }

                    # Update progress
                    progress_bar.progress((i + 1) / len(selected_models))

                # Store results
                st.session_state.evaluation_results = evaluation_results

                status_text.text("✅ Model evaluation completed!")
                st.success(f"Completed evaluation for {len(selected_models)} models!")

            # Display Evaluation Results
            if st.session_state.evaluation_results:
                st.markdown("### 📈 Evaluation Metrics")

                results_data = []
                for model_name, result in st.session_state.evaluation_results.items():
                    results_data.append({
                        'Model': model_name,
                        'RMSE': result['rmse'],
                        'R²': result['r2'],
                        'Evaluation Time (s)': result['evaluation_time']
                    })

                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df.round(4))

                # Best model highlight
                best_model_name = results_df.loc[results_df['R²'].idxmax(), 'Model']
                best_score = results_df['R²'].max()
                st.success(f"🏆 Best Model: **{best_model_name}** with R² of **{best_score:.4f}**")

                # Predicted vs Actual Plots
                st.markdown("### 📊 Predicted vs Actual")

                plot_model = st.selectbox(
                    "Select model for predicted vs actual plot:",
                    list(st.session_state.evaluation_results.keys()),
                    help="Choose the model for which you want to visualize the predicted vs actual values."
                )

                if plot_model:
                    result = st.session_state.evaluation_results[plot_model]
                    fig = plot_predicted_vs_actual(y, result['y_pred'], plot_model)
                    st.plotly_chart(fig, use_container_width=True)

                # Report Generation
                st.markdown("#### Generate Report")
                read_report = st.checkbox("Display Report", value=False)
                report_buffer = None  # Initialize report_buffer

                if st.button("Generate and Download Report"):
                    report_buffer = generate_evaluation_report(st.session_state.evaluation_results, target_col, cv_folds)
                    st.download_button(
                        label="Download Model Evaluation Report",
                        data=report_buffer,
                        file_name="model_evaluation_report.pdf",
                        mime="application/pdf"
                    )

                if read_report and report_buffer:
                    report_content = report_buffer.read().decode('latin-1')
                    st.markdown(report_content, unsafe_allow_html=True)

        else:
            st.info("No trained models available. Please complete model training first.")

        # Save evaluation for the next page, prediction
        st.markdown("#### Save Evaluation for Prediction Page")
        if st.button("💾 Save Evaluation for Prediction"):
            st.session_state.saved_evaluation = {
                'evaluation_results': st.session_state.evaluation_results,
                'target_col': target_col,
                'cv_folds': cv_folds,
                'df': df
            }
            st.success("Evaluation data saved for Prediction Page!")

        # Next Steps
        st.markdown("---")
        st.markdown("### 🎯 Next Steps")

        st.markdown("""
        **Ready for Live Predictions?**

        Your evaluated models are now ready for making predictions on new loan applications. 
        Navigate to the **Live Prediction** page to:

        - Input new loan application data
        - Get instant default amount predictions
        - Compare predictions across different models
        - Export prediction results
        """)


if __name__ == "__main__":
    main()
