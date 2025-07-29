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

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Model Training", page_icon="🧠", layout="wide")

st.title("🧠 Model Training")
st.markdown("---")


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
    if model_type not in ['linear', 'lasso', 'logistic']:
        st.warning("Coefficient visualization is only available for linear, Lasso, and Logistic Regression models.")
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
        title=f'Regression Coefficients - {model_type.capitalize()}'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interpretation Hints
    st.markdown("#### Interpretation Hints:")
    st.markdown("- **Positive Coefficient:** An increase in the feature's value is associated with an increase in the target variable.")
    st.markdown("- **Negative Coefficient:** An increase in the feature's value is associated with a decrease in the target variable.")
    st.markdown("- **Magnitude:** The larger the absolute value of the coefficient, the stronger the association.")


def main():
    st.markdown("""
    This section focuses on training machine learning models using the features selected in the previous step.
    You can train Linear Regression, Lasso Regression, Logistic Regression, or Random Forest models and evaluate their performance.
    """)

    # Check if final dataset exists in session state
    if 'final_dataset' not in st.session_state and 'feature_selection_results' not in st.session_state:
        st.info("👆 Please complete the feature selection step first.")
        return

    # Load feature selection results from session state
    feature_selection_results = st.session_state.get('feature_selection_results', {})

    # Prepare data
    if 'final_dataset' in st.session_state:
        final_dataset = st.session_state.final_dataset
        target_col = st.selectbox(
            "Select target variable:",
            final_dataset.columns.tolist(),
            index=len(final_dataset.columns) - 1,
            help="Choose the column that represents the target variable."
        )
    else:
        st.warning("No final dataset found. Please select a target variable from one of the feature selection results.")
        target_col = None
        return  # Exit if no target column is selected

    # Determine task type based on target variable characteristics
    if st.session_state.final_dataset[target_col].dtype == 'object' or len(st.session_state.final_dataset[target_col].unique()) <= 10:
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
    feature_selection_options = list(st.session_state.get('feature_selection_results', {}).keys())
    feature_selection_options.append("All Features")

    # Create a dictionary to map numbers to feature selection options
    option_mapping = {i + 1: option for i, option in enumerate(feature_selection_options)}

    # Display the options with numbers
    st.write("Select Feature Selection Methods to Use:")
    for number, option in option_mapping.items():
        st.write(f"{number}: {option}")

    # Get the selected numbers from the user
    selected_numbers = st.multiselect(
        "Enter the numbers corresponding to the feature selection methods you want to use:",
        options=list(option_mapping.keys()),
        default=[1] if option_mapping else [],
        help="Choose which feature selection methods to use for model training. 'All Features' will use all columns in the original dataset."
    )

    # Map the selected numbers back to the feature selection options
    selected_feature_selections = [option_mapping[number] for number in selected_numbers]

    # Store feature selection results in session state
    st.session_state.selected_feature_selections = selected_feature_selections

    # Train Models and Store Results
    model_results = []

    if st.button("Train Models"):
        with st.spinner("Training models..."):
            for selection_name in selected_feature_selections:
                if selection_name == "All Features":
                    X = st.session_state.final_dataset.drop(columns=[target_col])
                    y = st.session_state.final_dataset[target_col]
                    feature_names = X.columns.tolist()
                else:
                    selected_features = st.session_state.feature_selection_results[selection_name]['features']
                    X = st.session_state.final_dataset[selected_features]
                    y = st.session_state.final_dataset[target_col]
                    feature_names = selected_features

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
                    'Feature Names': feature_names
                })

        st.success("Models trained!")

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
            if model_results[0]['Model Type'] in ['linear', 'lasso', 'logistic']:
                visualize_coefficients(model_results[0]['Model'], model_results[0]['Feature Names'], model_results[0]['Model Type'])
            else:
                st.warning("Coefficient visualization is not available for this model type.")

        # Predictions vs Actual for the First Model (Regression)
        if task_type == 'regression':
            st.markdown("#### 📉 Predictions vs Actual Values (First Model)")
            if model_results:
                X = st.session_state.final_dataset[model_results[0]['Feature Names']]
                X_train, X_test, y_train, y_test = train_test_split(X, st.session_state.final_dataset[target_col], test_size=test_size, random_state=42)
                y_pred = model_results[0]['Model'].predict(X_test)
                predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                fig = px.scatter(
                    predictions_df, x='Actual', y='Predicted',
                    title='Actual vs Predicted Values'
                )
                st.plotly_chart(fig, use_container_width=True)

        # ROC Curve for the First Model (Classification)
        if task_type == 'classification':
            st.markdown("#### 📊 ROC Curve (First Model)")
            if model_results:
                X = st.session_state.final_dataset[model_results[0]['Feature Names']]
                X_train, X_test, y_train, y_test = train_test_split(X, st.session_state.final_dataset[target_col], test_size=test_size, random_state=42)
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

        # Store model results in session state
        st.session_state.model_results = model_results

    # Button to navigate to the next step (Model Deployment)
    if st.button("Proceed to Model Deployment"):
        st.session_state.next_page = "model_deployment"
        st.rerun()


if __name__ == "__main__":
    main()