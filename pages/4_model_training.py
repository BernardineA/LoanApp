import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import time
import warnings
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Model Training", page_icon="🧠", layout="wide")

st.title("🧠 Model Training")
st.markdown("---")


@st.cache_resource
def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_type='linear', alpha=1.0, task_type='regression'):
    """
    Trains and evaluates a model based on the specified type.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target.
        y_test (pd.Series): Testing target.
        model_type (str): Type of model ('linear', 'lasso', 'random_forest', 'logistic').
        alpha (float): Regularization strength for Lasso.
        task_type (str): 'regression' or 'classification'

    Returns:
        tuple: Trained model, predictions, metrics, and training time.
    """
    start_time = time.time()

    if task_type == 'regression':
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'lasso':
            model = Lasso(alpha=alpha)
        elif model_type == 'random_forest':
            model = RandomForestRegressor(random_state=42)
        else:
            raise ValueError("Invalid regression model type specified.")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics = {'MSE': mse, 'R-squared': r2}

    elif task_type == 'classification':
        if model_type == 'logistic':
            model = LogisticRegression(random_state=42)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(random_state=42)
        else:
            raise ValueError("Invalid classification model type specified.")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC AUC

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'AUC': auc}

    else:
        raise ValueError("Invalid task type specified.")

    end_time = time.time()
    training_time = end_time - start_time

    return model, y_pred, metrics, training_time


def visualize_coefficients(model, feature_names, model_type):
    """
    Visualizes the coefficients of a linear, Lasso, or Logistic Regression model.

    Args:
        model: Trained model.
        feature_names (list): List of feature names.
        model_type (str): Type of model ('linear', 'lasso', 'logistic').
    """
    if model_type not in ['linear', 'lasso', 'logistic', 'random_forest']:
        st.warning("Coefficient visualization is only available for linear, Lasso, Logistic Regression, and Random Forest models.")
        return

    if model_type == 'random_forest':
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

            fig = px.bar(
                feature_importance_df,
                x='Importance', y='Feature',
                orientation='h',
                title='Random Forest Feature Importances',
                labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Random Forest model does not have feature importances to visualize.")
        return

    if hasattr(model, 'coef_'):
        coefficients = model.coef_[0] if model_type == 'logistic' else model.coef_
    else:
        st.warning("Model does not have coefficients to visualize.")
        return

    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)

    fig = px.bar(
        coef_df,
        x='Coefficient', y='Feature',
        orientation='h',
        title=f'Regression Coefficients - {model_type.capitalize()}',
        labels={'Coefficient': 'Coefficient Value', 'Feature': 'Feature Name'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interpretation Hints
    st.markdown("#### Interpretation Hints:")
    st.markdown("- **Positive Coefficient:** An increase in the feature's value is associated with an increase in the target variable.")
    st.markdown("- **Negative Coefficient:** An increase in the feature's value is associated with a decrease in the target variable.")
    st.markdown("- **Magnitude:** The larger the absolute value of the coefficient, the stronger the association.")


def generate_model_report(model_results, target_col, test_size):
    """Generates a report explaining model training and results."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Model Training Report", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))

    # Introduction
    story.append(Paragraph("This report summarizes the model training process, comparing different models and highlighting the best choice.", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # General Information
    story.append(Paragraph("<b>General Information:</b>", styles['h2']))
    story.append(Paragraph(f"Target Variable: {target_col}", styles['Normal']))
    story.append(Paragraph(f"Test Set Size: {test_size}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Model Comparison
    story.append(Paragraph("<b>Model Comparison:</b>", styles['h2']))
    story.append(Paragraph("The following table compares the performance of different models:", styles['Normal']))

    # Prepare data for the table
    table_data = []
    for result in model_results:
        row = {
            'Feature Selection': result['Feature Selection'],
            'Model Type': result['Model Type'],
            'Task Type': result['Task Type'],
            'Training Time': result['Training Time']
        }
        row.update(result['Metrics'])  # Add metrics to the row
        table_data.append(row)

    # Convert DataFrame to string for report
    results_df = pd.DataFrame(table_data)
    table_string = results_df.to_string()
    story.append(Paragraph(table_string, styles['Code']))
    story.append(Spacer(1, 0.2 * inch))

    # Model Details
    for result in model_results:
        story.append(Paragraph(f"<b>Model Details - {result['Feature Selection']} - {result['Model Type']}:</b>", styles['h3']))
        story.append(Paragraph(f"Task Type: {result['Task Type']}", styles['Normal']))
        story.append(Paragraph(f"Training Time: {result['Training Time']:.4f} seconds", styles['Normal']))
        story.append(Paragraph("Metrics:", styles['Normal']))
        for metric, value in result['Metrics'].items():
            story.append(Paragraph(f"- {metric}: {value:.4f}", styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))

    doc.build(story)
    buffer.seek(0)
    return buffer


def main():
    st.markdown("""
    This section focuses on training machine learning models using the features selected in the previous step.
    You can train Linear Regression, Lasso Regression, Logistic Regression, or Random Forest models and evaluate their performance.
    """)

    # Initialize session state for model results
    if 'model_results' not in st.session_state:
        st.session_state.model_results = []

    # Data Loading Section
    st.markdown("## 📁 Data Loading")

    # Load data from session state if available
    if 'modeling_data' in st.session_state:
        df = st.session_state.modeling_data
        st.success("✅ Loaded final dataset from Feature Selection.")
    else:
        df = None

    # Option to upload dataset
    uploaded_file = st.file_uploader(
        "Alternatively, upload a dataset (CSV format)",
        type=['csv'],
        help="Upload a CSV file to use for model training."
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
            st.session_state.modeling_data = df  # Save to session state
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return

    if df is None:
        st.info("👆 Please complete the feature selection step or upload a dataset to begin model training.")
        return

    # Target Variable Selection
    st.markdown("## 🎯 Target Variable Selection")

    target_col = st.selectbox(
        "Select target variable:",
        df.columns.tolist(),
        index=len(df.columns) - 1,
        help="Choose the column that represents the target variable."
    )

    # Determine task type based on target variable characteristics
    if df[target_col].dtype == 'object' or len(df[target_col].unique()) <= 10:
        task_type = 'classification'
        st.info("Target variable appears to be categorical. Setting task type to classification.")
    else:
        task_type = 'regression'
        st.info("Target variable appears to be numerical. Setting task type to regression.")

    # Display the determined task type
    st.write(f"**Task Type:** {task_type}")

    # Model Selection
    st.markdown("### 🤖 Model Selection and Training")

    if task_type == 'regression':
        model_type = st.selectbox(
            "Select regression model type:",
            ['linear', 'lasso', 'random_forest'],
            index=0,
            help="Choose the type of regression model to train."
        )
    else:
        model_type = st.selectbox(
            "Select classification model type:",
            ['logistic', 'random_forest'],
            index=0,
            help="Choose the type of classification model to train."
        )

    # Lasso Regularization Strength
    if model_type == 'lasso':
        alpha = st.slider(
            "Lasso regularization strength (alpha):",
            min_value=0.01, max_value=1.0, value=0.1, step=0.01,
            help="The regularization strength; must be a positive float. Higher values specify stronger regularization."
        )
    else:
        alpha = None

    # Split data
    test_size = st.slider(
        "Test set size:",
        min_value=0.1, max_value=0.9, value=0.2, step=0.05,
        help="The proportion of the dataset to include in the test split."
    )

    # Feature Selection Choice
    st.markdown("### ⚙️ Feature Selection Choice")

    # Load available feature selection methods from session state
    default_feature_selections = ["Best Subset", "Forward Stepwise", "Lasso", "Random Forest"]
    available_feature_selections = []

    if 'feature_selection_results' in st.session_state:
        # Only add the keys if the corresponding value has 'features'
        for key, value in st.session_state.feature_selection_results.items():
            if 'features' in value:
                available_feature_selections.append(key)
    else:
        available_feature_selections = default_feature_selections

    selected_feature_selections = st.multiselect(
        "Select Feature Selection Methods to Use:",
        options=available_feature_selections,
        default=available_feature_selections if not 'feature_selection_results' in st.session_state else [],
        help="Choose which feature selection methods to use for model training.",
    )

    # Store feature selection results in session state
    st.session_state.selected_feature_selections = selected_feature_selections

    # Train Models and Store Results
    if st.button("Train Models"):
        with st.spinner("Training models..."):
            model_results = []  # Reset model_results for each training run
            for selection_name in selected_feature_selections:
                if 'feature_selection_results' in st.session_state and selection_name in st.session_state.feature_selection_results:
                    selected_features = st.session_state.feature_selection_results[selection_name]['features']
                    X = df[selected_features]
                    y = df[target_col]
                    feature_names = selected_features
                else:
                    # Handle the case where feature_selection_results is not in st.session_state
                    # or the selection_name is not a key in feature_selection_results
                    if selection_name in default_feature_selections:
                        st.warning(f"'{selection_name}' was selected, but no feature selection results were found.  Using all features instead.")
                        X = df.drop(columns=[target_col])
                        y = df[target_col]
                        feature_names = X.columns.tolist()
                    else:
                        st.error(f"Feature selection method '{selection_name}' not found in session state or does not have 'features' key.")
                        continue  # Skip to the next selection

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                if model_type == 'lasso' and task_type == 'regression':
                    model, y_pred, metrics, training_time = train_and_evaluate_model(
                        X_train, X_test, y_train, y_test, model_type, alpha, task_type
                    )
                else:
                    model, y_pred, metrics, training_time = train_and_evaluate_model(
                        X_train, X_test, y_train, y_test, model_type, task_type=task_type
                    )

                model_results.append({
                    'Feature Selection': selection_name,
                    'Model Type': model_type,
                    'Task Type': task_type,
                    'Metrics': metrics,
                    'Training Time': training_time,
                    'Model': model,
                    'Feature Names': feature_names,
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_pred': y_pred
                })

            st.success("Models trained!")

            # Store model results in session state
            st.session_state.model_results = model_results

        # Display Comparison Table
        st.markdown("### 📊 Model Comparison")
        results_data = []
        for result in model_results:
            row = {
                'Feature Selection': result['Feature Selection'],
                'Model Type': result['Model Type'],
                'Task Type': result['Task Type'],
                'Training Time': result['Training Time']
            }
            row.update(result['Metrics'])  # Add metrics to the row
            results_data.append(row)

        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df)

        # Visualize Coefficients for the First Model
        st.markdown("### 📈 Regression Coefficients (First Model)")
        if model_results:
            if model_results[0]['Model Type'] in ['linear', 'lasso', 'logistic', 'random_forest']:
                visualize_coefficients(model_results[0]['Model'], model_results[0]['Feature Names'], model_results[0]['Model Type'])
            else:
                st.warning("Coefficient visualization is not available for this model type.")

        # Predictions vs Actual for Each Model (Regression)
        if task_type == 'regression':
            st.markdown("#### 📉 Predictions vs Actual Values")
            for i, result in enumerate(model_results):
                st.markdown(f"##### {result['Feature Selection']} - {result['Model Type']}")
                predictions_df = pd.DataFrame({'Actual': result['y_test'], 'Predicted': result['y_pred']})
                fig = px.scatter(
                    predictions_df, x='Actual', y='Predicted',
                    title='Actual vs Predicted Values',
                    labels={'Actual': 'Actual Value', 'Predicted': 'Predicted Value'}
                )
                fig.update_traces(marker=dict(color=['blue'] * len(predictions_df)), name='Data Points')  # Set color to blue
                st.plotly_chart(fig, use_container_width=True, key=f"predictions_plot_{i}") # Add key to fix duplicate element ID

        # ROC Curve for the First Model (Classification)
        if task_type == 'classification':
            st.markdown("#### 📊 ROC Curve (First Model)")
            if model_results:
                X = df[model_results[0]['Feature Names']]
                X_train, X_test, y_train, y_test = train_test_split(X, df[target_col], test_size=test_size, random_state=42)
                y_proba = model_results[0]['Model'].predict_proba(X_test)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                fig = px.area(
                    x=fpr, y=tpr,
                    title=f'ROC Curve (AUC={model_results[0]["Metrics"]["AUC"]:.4f})',
                    labels=dict(x='False Positive Rate', y='True Positive Rate'),
                    width=800, height=400
                )
                fig.add_shape(
                    type='line', line=dict(dash='dash'),
                    x0=0, x1=1, y0=0, y1=1
                )
                fig.update_yaxes(scaleanchor="x", scaleratio=1)
                fig.update_xaxes(constrain='domain')
                st.plotly_chart(fig, use_container_width=True)

    # Report Generation
    if st.session_state.model_results:
        st.markdown("#### Generate Report")
        read_report = st.checkbox("Display Report", value=False)
        report_buffer = None  # Initialize report_buffer
        if st.button("Generate and Download Report"):
            report_buffer = generate_model_report(st.session_state.model_results, target_col, test_size)
            st.download_button(
                label="Download Model Training Report",
                data=report_buffer,
                file_name="model_training_report.pdf",
                mime="application/pdf"
            )

        if read_report and report_buffer:
            report_content = report_buffer.read().decode('latin-1')
            st.markdown(report_content, unsafe_allow_html=True)

    # Save data for evaluation Page
    if st.session_state.model_results:
        st.markdown("#### Save for Evaluation Page")
        if st.button("💾 Save for Evaluation"):
            st.session_state.evaluation_data = {
                'model_results': st.session_state.model_results,
                'target_col': target_col,
                'test_size': test_size,
                'task_type': task_type,
                'df': df  # Save the DataFrame
            }
            st.success("Data saved for Evaluation Page!")

    st.markdown("---")
    st.markdown("Are you ready to evaluate your model?")
    if st.button("Click here and load the data to Model Evaluation"):
        st.session_state['model_evaluation_data'] = df.copy()
        st.success("Data saved for Model Evaluation!")
        st.info("You can now navigate to the Model Evaluation page to use this data.")


if __name__ == "__main__":
    main()
