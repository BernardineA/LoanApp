import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from itertools import combinations
import time
import warnings
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from io import BytesIO
import hashlib  # Import the hashlib library

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Feature Selection", page_icon="🎯", layout="wide")

st.title("🎯 Feature Selection")
st.markdown("---")


@st.cache_data
def calculate_feature_importance_stats(X, y):
    """Calculate various feature importance statistics"""
    results = {}

    # Correlation with target
    correlations = X.corrwith(pd.Series(y)).abs().sort_values(ascending=False)
    results['correlation'] = correlations

    # Mutual Information
    mi_scores = mutual_info_regression(X, y, random_state=42)
    results['mutual_info'] = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    # Random Forest Feature Importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    results['rf_importance'] = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    # Lasso Feature Importance
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X, y)
    results['lasso_importance'] = pd.Series(np.abs(lasso.coef_), index=X.columns).sort_values(ascending=False)

    return results


@st.cache_data
def calculate_model_metrics(X, y, _model):  # Changed 'model' to '_model'
    """Calculate R2, Adjusted R2, RMSE, AIC, and BIC for a given model."""
    X_sm = sm.add_constant(X)  # Add constant for statsmodels
    model_sm = sm.OLS(y, X_sm).fit()
    y_pred = model_sm.predict(X_sm)
    r2 = model_sm.rsquared
    adjusted_r2 = model_sm.rsquared_adj
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    aic = model_sm.aic
    bic = model_sm.bic
    return r2, adjusted_r2, rmse, aic, bic


@st.cache_data
def best_subset_selection(X, y, max_features=10):
    """
    Implements Best Subset Selection to identify the most predictive subset of features.
    """
    best_score = -np.inf
    best_features = None
    results = []

    for k in range(1, min(max_features + 1, X.shape[1] + 1)):
        for subset in combinations(X.columns, k):
            subset = list(subset)
            X_subset = X[subset]

            # Train a linear regression model
            model = LinearRegression()
            model.fit(X_subset, y)

            # Calculate metrics
            r2, adjusted_r2, rmse, aic, bic = calculate_model_metrics(X_subset, y, model)
            score = adjusted_r2  # Use Adjusted R-squared for evaluation

            results.append({'features': subset, 'r2': r2, 'adjusted_r2': adjusted_r2, 'rmse': rmse, 'aic': aic, 'bic': bic, 'num_features': k})

            # Update the best score and features if the current subset is better
            if score > best_score:
                best_score = score
                best_features = subset

    results_df = pd.DataFrame(results)
    return best_features, None, results_df


@st.cache_data
def forward_stepwise_selection(X, y, max_features=10):
    """
    Implements Forward Stepwise Selection to identify the most predictive subset of features.
    """
    selected_features = []
    available_features = list(X.columns)
    best_score = -np.inf
    results = []

    while available_features and len(selected_features) < max_features:
        current_best_feature = None
        current_best_score = -np.inf
        current_r2 = None
        current_adjusted_r2 = None
        current_rmse = None
        current_aic = None
        current_bic = None

        for feature in available_features:
            # Add the feature to the selected features
            current_features = selected_features + [feature]
            X_subset = X[current_features]

            # Train a linear regression model
            model = LinearRegression()
            model.fit(X_subset, y)

            # Calculate metrics
            r2, adjusted_r2, rmse, aic, bic = calculate_model_metrics(X_subset, y, model)
            score = adjusted_r2  # Use Adjusted R-squared for evaluation

            # Update the best score and feature if the current feature is better
            if score > current_best_score:
                current_best_score = score
                current_best_feature = feature
                current_r2 = r2
                current_adjusted_r2 = adjusted_r2
                current_rmse = rmse
                current_aic = aic
                current_bic = bic

        # If a better feature was found, add it to the selected features
        if current_best_feature is not None:
            selected_features.append(current_best_feature)
            available_features.remove(current_best_feature)
            results.append({'features': selected_features, 'r2': current_r2, 'adjusted_r2': current_adjusted_r2, 'rmse': current_rmse, 'aic': current_aic, 'bic': bic, 'num_features': len(selected_features)})
            best_score = current_best_score
        else:
            break  # No improvement, stop adding features

    results_df = pd.DataFrame(results)
    return selected_features, None, results_df


def generate_report(comparison_df, best_method, X, y, final_features):
    """Generates a report explaining feature selection methods and results."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Feature Selection Report", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))

    # Introduction
    story.append(Paragraph("This report summarizes the feature selection process, comparing different methods and highlighting the best choice.", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Method Comparison
    story.append(Paragraph("<b>Method Comparison:</b>", styles['h2']))
    story.append(Paragraph("The following table compares the performance of different feature selection methods:", styles['Normal']))

    # Convert DataFrame to string for report
    table_data = comparison_df.to_string()
    story.append(Paragraph(table_data, styles['Code']))
    story.append(Spacer(1, 0.2 * inch))

    # Best Method Explanation
    story.append(Paragraph("<b>Best Method:</b>", styles['h2']))
    story.append(Paragraph(f"The best feature selection method was <b>{best_method}</b>.", styles['Normal']))

    # Add explanation based on the best method
    if best_method == 'Best Subset':
        story.append(Paragraph("Best Subset Selection was chosen for its ability to evaluate all possible feature combinations, optimizing for Adjusted R-squared and AIC.", styles['Normal']))
    elif best_method == 'Forward Stepwise':
        story.append(Paragraph("Forward Stepwise Selection was selected for its efficiency in adding features incrementally, providing a balance between performance and computational cost.", styles['Normal']))
    elif best_method == 'Lasso':
        story.append(Paragraph("Lasso Regression was chosen for its ability to perform feature selection by shrinking the coefficients of less important features to zero, resulting in a sparse and robust model.", styles['Normal']))
    elif best_method == 'Random Forest':
        story.append(Paragraph("Random Forest was selected for its ability to capture non-linear relationships and provide robust feature importance scores, optimizing for RMSE.", styles['Normal']))

    story.append(Spacer(1, 0.2 * inch))

    # Final Features
    story.append(Paragraph("<b>Final Features:</b>", styles['h2']))
    story.append(Paragraph(f"The final selected features are: {', '.join(final_features)}.", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Model Performance with Final Features
    story.append(Paragraph("<b>Model Performance with Final Features:</b>", styles['h2']))
    model = LinearRegression()
    model.fit(X[final_features], y)
    r2, adjusted_r2, rmse, aic, bic = calculate_model_metrics(X[final_features], y, model)
    story.append(Paragraph(f"R-squared: {r2:.4f}", styles['Normal']))
    story.append(Paragraph(f"Adjusted R-squared: {adjusted_r2:.4f}", styles['Normal']))
    story.append(Paragraph(f"RMSE: {rmse:.4f}", styles['Normal']))
    story.append(Paragraph(f"AIC: {aic:.4f}", styles['Normal']))
    story.append(Paragraph(f"BIC: {bic:.4f}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Conclusion
    story.append(Paragraph("In conclusion, the selected features provide a strong foundation for building a predictive model. The chosen feature selection method balances performance, interpretability, and computational efficiency.", styles['Normal']))

    # Monkey-patch reportlab to use hashlib
    import reportlab.pdfbase.pdfdoc as pdfdoc
    pdfdoc.md5 = hashlib.md5

    doc.build(story)
    buffer.seek(0)
    return buffer


def main():
    st.markdown("""
    This section provides comprehensive feature selection capabilities using statistical methods, 
    machine learning-based approaches, and domain knowledge to identify the most predictive features 
    for loan default prediction.
    """)

    # Initialize session state
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = None
    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = None
    if 'final_dataset' not in st.session_state:
        st.session_state.final_dataset = None
    if 'feature_selection_results' not in st.session_state:
        st.session_state.feature_selection_results = {}

    # Data Loading Section
    st.markdown("## 📁 Data Loading")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Load data from session state if available
        if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
            df = st.session_state.processed_data
            st.success("✅ Loaded preprocessed data from previous step.")
        else:
            df = None  # Initialize df to None

        uploaded_file = st.file_uploader(
            "Upload preprocessed dataset (CSV format)",
            type=['csv'],
            help="Upload a preprocessed CSV file for feature selection"
        )

        # Load data from uploaded file if available
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
                st.session_state.preprocessed_data = df  # Save to session state
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return

        if df is None:
            st.info("👆 Please upload a preprocessed dataset or load from the previous step to begin feature selection.")
            return

    # Target Variable Selection
    st.markdown("## 🎯 Target Variable Selection")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    target_col = st.selectbox(
        "Select target variable (what you want to predict):",
        numeric_cols,
        index=0,
        help="Choose the column that represents the loan default amount"
    )

    if target_col:
        # Prepare features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Only keep numeric features for now
        X = X.select_dtypes(include=[np.number])

        st.success(f"Target variable: **{target_col}** | Features: **{len(X.columns)}**")

        # Feature Selection Methods
        st.markdown("## 🔍 Feature Selection Methods")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Statistical Analysis", "Subset Selection", "Embedded Methods", "Final Selection"
        ])

        with tab1:
            st.markdown("### 📊 Statistical Feature Analysis")
            st.markdown("Use statistical measures to assess the relationship between each feature and the target variable.")

            # Calculate feature importance statistics
            with st.spinner("Calculating feature importance statistics..."):
                importance_stats = calculate_feature_importance_stats(X, y)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Feature Importance Comparison")
                st.markdown("Compare feature importance scores from different statistical methods.")

                # Create comparison dataframe
                comparison_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Correlation': importance_stats['correlation'][X.columns],
                    'Mutual Info': importance_stats['mutual_info'][X.columns],
                })

                st.dataframe(comparison_df.round(4))

            with col2:
                st.markdown("#### Feature Importance Visualization")
                st.markdown("Visualize the top features based on selected statistical method.")

                method = st.selectbox(
                    "Select method to visualize:",
                    ['Correlation', 'Mutual Info']
                )

                if method:
                    scores = importance_stats[method.lower().replace(' ', '_')]

                    fig = px.bar(
                        x=scores.values[:15],
                        y=scores.index[:15],
                        orientation='h',
                        title=f'Top 15 Features - {method}',
                        labels={'x': 'Importance Score', 'y': 'Features'}
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

            # Heatmap of all importance methods
            st.markdown("#### Feature Importance Heatmap")
            st.markdown("Visualize the normalized feature importance across different methods using a heatmap.")

            heatmap_data = comparison_df[
                ['Feature', 'Correlation', 'Mutual Info']].set_index(
                'Feature')

            # Normalize for heatmap
            heatmap_normalized = heatmap_data.div(heatmap_data.max(axis=0), axis=1)

            fig = px.imshow(
                heatmap_normalized.T,
                title="Normalized Feature Importance Across Methods",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("### 💡 Subset Selection Methods")
            st.markdown("Evaluate all possible subsets of features to find the best performing combination.")

            max_features_bss = st.slider(
                "Maximum number of features to consider (Subset Selection):",
                1, len(X.columns), min(10, len(X.columns)),
                key="max_features_bss",
                help="The maximum number of features to include in the best subset."
            )

            max_features_fss = st.slider(
                "Maximum number of features to select (Forward Stepwise):",
                1, len(X.columns), min(10, len(X.columns)),
                key="max_features_fss",
                help="The maximum number of features to select using forward stepwise selection."
            )

            subset_methods = st.multiselect(
                "Select Subset Selection Methods to Run:",
                ['Best Subset Selection', 'Forward Stepwise Selection'],
                ['Best Subset Selection', 'Forward Stepwise Selection']
            )

            if st.button("Run Subset Selection Methods"):
                with st.spinner("Running Subset Selection Methods..."):
                    start_time = time.time()

                    # Best Subset Selection
                    if 'Best Subset Selection' in subset_methods:
                        best_features, best_model, results_df_bss = best_subset_selection(X, y, max_features_bss)
                        st.session_state.feature_selection_results['Best Subset'] = {'features': best_features}  # Store in session state
                        st.session_state.best_features = best_features
                        st.session_state.results_df_bss = results_df_bss
                    else:
                        st.session_state.best_features = None
                        st.session_state.results_df_bss = None

                    # Forward Stepwise Selection
                    if 'Forward Stepwise Selection' in subset_methods:
                        selected_features, best_model, results_df_fss = forward_stepwise_selection(X, y, max_features_fss)
                        st.session_state.feature_selection_results['Forward Stepwise'] = {'features': selected_features}  # Store in session state
                        st.session_state.selected_features = selected_features
                        st.session_state.results_df_fss = results_df_fss
                    else:
                        st.session_state.selected_features = None
                        st.session_state.results_df_fss = None

                    end_time = time.time()
                    elapsed_time = end_time - start_time

                st.success(f"Subset Selection Methods completed in {elapsed_time:.2f} seconds!")

        with tab3:
            st.markdown("### 🎯 Embedded Methods")
            st.markdown("Embedded methods perform feature selection as part of the model training process.")

            embedded_methods = st.multiselect(
                "Select Embedded Methods to Run:",
                ['Lasso', 'Random Forest'],
                ['Lasso', 'Random Forest']
            )

            if st.button("Run Embedded Methods"):
                with st.spinner("Running Embedded Methods..."):
                    start_time = time.time()

                    # Lasso Regression
                    if 'Lasso' in embedded_methods:
                        scaler = StandardScaler()
                        X_scaled = pd.DataFrame(
                            scaler.fit_transform(X),
                            columns=X.columns
                        )
                        lasso_cv = LassoCV(cv=5, random_state=42)
                        lasso_cv.fit(X_scaled, y)
                        selected_mask = np.abs(lasso_cv.coef_) > 1e-6
                        selected_features_lasso = X.columns[selected_mask].tolist()
                        st.session_state.feature_selection_results['Lasso'] = {'features': selected_features_lasso}  # Store in session state
                        st.session_state.selected_features_lasso = selected_features_lasso
                    else:
                        st.session_state.selected_features_lasso = None

                    # Random Forest
                    if 'Random Forest' in embedded_methods:
                        rf = RandomForestRegressor(n_estimators=100, random_state=42)
                        rf.fit(X, y)
                        importance_threshold = 0.01  # Fixed threshold for simplicity
                        selected_mask = rf.feature_importances_ > importance_threshold
                        selected_features_rf = X.columns[selected_mask].tolist()
                        st.session_state.feature_selection_results['Random Forest'] = {'features': selected_features_rf}  # Store in session state
                        st.session_state.selected_features_rf = selected_features_rf
                    else:
                        st.session_state.selected_features_rf = None

                    end_time = time.time()
                    elapsed_time = end_time - start_time

                st.success(f"Embedded Methods completed in {elapsed_time:.2f} seconds!")

        with tab4:
            st.markdown("### ✅ Final Feature Selection")
            st.markdown("Combine insights from different methods to create the final feature set.")

            # Prepare data for comparison table
            comparison_data = []

            # Best Subset Selection
            if 'best_features' in st.session_state and st.session_state.best_features:
                best_features = st.session_state.best_features
                model = LinearRegression()
                model.fit(X[best_features], y)
                r2, adjusted_r2, rmse, aic, bic = calculate_model_metrics(X[best_features], y, model)
                comparison_data.append({
                    'Method': 'Best Subset',
                    'Features Selected': best_features,
                    'R²': r2,
                    'Adj R²': adjusted_r2,
                    'RMSE': rmse,
                    'AIC': aic,
                    'BIC': bic,
                    '#Features': len(best_features),
                    'Notes': 'Best on adj R² & AIC',
                    'Algorithm': 'Subset Selection'
                })

            # Forward Stepwise Selection
            if 'selected_features' in st.session_state and st.session_state.selected_features:
                selected_features = st.session_state.selected_features
                model = LinearRegression()
                model.fit(X[selected_features], y)
                r2, adjusted_r2, rmse, aic, bic = calculate_model_metrics(X[selected_features], y, model)
                comparison_data.append({
                    'Method': 'Forward Stepwise',
                    'Features Selected': selected_features,
                    'R²': r2,
                    'Adj R²': adjusted_r2,
                    'RMSE': rmse,
                    'AIC': aic,
                    'BIC': bic,
                    '#Features': len(selected_features),
                    'Notes': '',
                    'Algorithm': 'Stepwise Selection'
                })

            # Lasso
            if 'selected_features_lasso' in st.session_state and st.session_state.selected_features_lasso:
                selected_features_lasso = st.session_state.selected_features_lasso
                model = LinearRegression()
                model.fit(X[selected_features_lasso], y)
                r2, adjusted_r2, rmse, aic, bic = calculate_model_metrics(X[selected_features_lasso], y, model)
                comparison_data.append({
                    'Method': 'Lasso',
                    'Features Selected': selected_features_lasso,
                    'R²': r2,
                    'Adj R²': adjusted_r2,
                    'RMSE': rmse,
                    'AIC': aic,
                    'BIC': bic,
                    '#Features': len(selected_features_lasso),
                    'Notes': 'Sparse, robust',
                    'Algorithm': 'Embedded'
                })

            # Random Forest
            if 'selected_features_rf' in st.session_state and st.session_state.selected_features_rf:
                selected_features_rf = st.session_state.selected_features_rf
                model = LinearRegression()
                model.fit(X[selected_features_rf], y)
                r2, adjusted_r2, rmse, aic, bic = calculate_model_metrics(X[selected_features_rf], y, model)
                comparison_data.append({
                    'Method': 'Random Forest',
                    'Features Selected': selected_features_rf,
                    'R²': r2,
                    'Adj R²': adjusted_r2,
                    'RMSE': rmse,
                    'AIC': aic,
                    'BIC': bic,
                    '#Features': len(selected_features_rf),
                    'Notes': 'Best RMSE, nonlinear',
                    'Algorithm': 'Embedded'
                })

            # Determine best method
            best_method = None
            best_adj_r2 = -np.inf
            best_rmse = np.inf

            for i, method_data in enumerate(comparison_data):
                if method_data['Method'] != 'Random Forest' and method_data['Adj R²'] > best_adj_r2:
                    best_adj_r2 = method_data['Adj R²']
                    best_method = method_data['Method']
                elif method_data['Method'] == 'Random Forest' and method_data['RMSE'] < best_rmse:
                    best_rmse = method_data['RMSE']
                    best_method = method_data['Method']

            # Update Notes to highlight the best method
            for i, method_data in enumerate(comparison_data):
                if method_data['Method'] == best_method:
                    if method_data['Method'] != 'Random Forest':
                        comparison_data[i]['Notes'] = 'Best on adj R² & AIC'
                    else:
                        comparison_data[i]['Notes'] = 'Best RMSE, nonlinear'
                else:
                    comparison_data[i]['Notes'] = ''

            # Display comparison table
            if comparison_data:
                st.markdown("#### Feature Selection Method Comparison")
                comparison_df = pd.DataFrame(comparison_data)

                # Highlight the best method
                def highlight_best(s):
                    if s['Notes'] == 'Best on adj R² & AIC' or s['Notes'] == 'Best RMSE, nonlinear':
                        return ['background-color: yellow'] * len(s)
                    else:
                        return [''] * len(s)

                styled_df = comparison_df.style.apply(highlight_best, axis=1)
                st.dataframe(styled_df)

            # Visualization of results
            if comparison_data:
                st.markdown("#### Method Performance Visualization")

                # R² Comparison
                fig_r2 = px.bar(comparison_df, x='Method', y='R²', title='R² Comparison')
                st.plotly_chart(fig_r2, use_container_width=True)

                # RMSE Comparison
                fig_rmse = px.bar(comparison_df, x='Method', y='RMSE', title='RMSE Comparison')
                st.plotly_chart(fig_rmse, use_container_width=True)

            # Final selection interface
            st.markdown("#### Create Final Feature Set")

            col1, col2 = st.columns(2)

            with col1:
                selection_methods = st.multiselect(
                    "Select feature selection algorithms:",
                    [
                        'Best Subset',
                        'Forward Stepwise',
                        'Lasso',
                        'Random Forest',
                        'Custom Selection'
                    ],
                    default=['Best Subset']
                )

                final_features = []
                for method in selection_methods:
                    if method == 'Best Subset':
                        if 'best_features' in st.session_state and st.session_state.best_features:
                            final_features.extend(st.session_state.best_features)
                        else:
                            st.warning("Run Best Subset Selection first.")

                    elif method == 'Forward Stepwise':
                        if 'selected_features' in st.session_state and st.session_state.selected_features:
                            final_features.extend(st.session_state.selected_features)
                        else:
                            st.warning("Run Forward Stepwise Selection first.")

                    elif method == 'Lasso':
                        if 'selected_features_lasso' in st.session_state and st.session_state.selected_features_lasso:
                            final_features.extend(st.session_state.selected_features_lasso)
                        else:
                            st.warning("Run Lasso Feature Selection first.")

                    elif method == 'Random Forest':
                        if 'selected_features_rf' in st.session_state and st.session_state.selected_features_rf:
                            final_features.extend(st.session_state.selected_features_rf)
                        else:
                            st.warning("Run Random Forest Feature Selection first.")

                    elif method == 'Custom Selection':
                        all_features = list(X.columns)
                        custom_features = st.multiselect(
                            "Select features manually:",
                            all_features,
                            default=all_features[:min(5, len(all_features))]
                        )
                        final_features.extend(custom_features)

                # Remove duplicates
                final_features = list(set(final_features))

            with col2:
                st.markdown("#### Selected Features Summary")

                if final_features:
                    st.metric("Final Feature Count", len(final_features))

                    # Save selection
                    if st.button("💾 Save Feature Selection"):
                        st.session_state.selected_features = final_features
                        st.success("✅ Feature selection saved!")

                        # Create final dataset
                        final_dataset = df[final_features + [target_col]]
                        st.session_state.final_dataset = final_dataset  # Save to session state

                        csv = final_dataset.to_csv(index=False)
                        st.download_button(
                            label="📁 Download Selected Features Dataset",
                            data=csv,
                            file_name="selected_features_dataset.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("Please select at least one feature.")

            # Visualization of final features
            if final_features:
                st.markdown("#### Final Features Visualization")

                # Correlation matrix of selected features
                st.markdown("##### Correlation Matrix")
                st.markdown("This heatmap shows the correlation between the selected features.  Strong positive correlations are shown in blue, strong negative correlations in red, and weak correlations are closer to white.")
                selected_data = X[final_features]
                corr_matrix = selected_data.corr()

                fig = px.imshow(
                    corr_matrix,
                    title="Correlation Matrix - Selected Features",
                    color_continuous_scale="RdBu_r"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

                # Feature importance comparison for selected features
                st.markdown("##### Feature Importance Comparison")
                st.markdown("This chart compares the feature importance scores (Correlation and Mutual Information) for the selected features.")
                selected_importance = pd.DataFrame({
                    'Feature': final_features,
                    'Correlation': [importance_stats['correlation'][f] for f in final_features],
                    'Mutual Info': [importance_stats['mutual_info'][f] for f in final_features],
                })

                fig = go.Figure()
                fig.add_trace(
                    go.Bar(name='Correlation', x=selected_importance['Feature'], y=selected_importance['Correlation']))
                fig.add_trace(
                    go.Bar(name='Mutual Info', x=selected_importance['Feature'], y=selected_importance['Mutual Info']))

                fig.update_layout(
                    title='Feature Importance Comparison - Selected Features',
                    xaxis_title='Features',
                    yaxis_title='Importance Score',
                    barmode='group',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

            # Report Generation
            if comparison_data and final_features:
                st.markdown("#### Generate Report")
                read_report = st.checkbox("Display Report", value=False)
                report_content = ""  # Initialize report_content
                if st.button("Generate and Download Report"):
                    report_buffer = generate_report(comparison_df, best_method, X, y, final_features)
                    st.download_button(
                        label="Download Feature Selection Report",
                        data=report_buffer,
                        file_name="feature_selection_report.pdf",
                        mime="application/pdf"
                    )
                    # Generate report content immediately after generating the report
                    report_buffer = generate_report(comparison_df, best_method, X, y, final_features)
                    try:
                        report_content = report_buffer.read().decode('utf-8')
                    except UnicodeDecodeError:
                        report_content = report_buffer.read().decode('latin-1')  # Try latin-1 encoding

                if read_report:
                    st.markdown(report_content, unsafe_allow_html=True)

            # Save for Model Training
            if 'final_dataset' in st.session_state and st.session_state.final_dataset is not None:
                st.markdown("#### Save for Model Training")
                if st.button("💾 Save for Model Training"):
                    st.session_state.modeling_data = st.session_state.final_dataset.copy()
                    st.success("Data saved for Model Training!")

    st.markdown("---")
    st.markdown("Are you ready to train your data?")
    if st.button("Click here and load the data to Model Training"):
        st.session_state['model_training_data'] = df.copy()
        st.success("Data saved for Model Training!")
        st.info("You can now navigate to the Model Training page to use this data.")


def r2_score(y_true, y_pred):
    """Calculate R-squared score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0


if __name__ == "__main__":
    main()
