import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import KNNImputer
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Data Preprocessing", page_icon="🔧", layout="wide")

st.title("🔧 Data Preprocessing")
st.markdown("---")

DEFAULT_DATASET_PATH = "data/Loan_Default.csv"

@st.cache_data
def load_default_data(path=DEFAULT_DATASET_PATH):
    """Load the default dataset from the specified path."""
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file {path} was not found.")
        return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

@st.cache_data
def calculate_missing_data(df):
    """Calculates and returns missing data summary."""
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing %': missing_percent.values
    }).query('`Missing Count` > 0')
    return missing_df

@st.cache_data
def detect_outliers_iqr(df, selected_cols):
    """Detects outliers using the IQR method."""
    Q1 = df[selected_cols].quantile(0.25)
    Q3 = df[selected_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df[selected_cols] < lower_bound) | (df[selected_cols] > upper_bound)).any(axis=1)
    return outliers

@st.cache_data
def detect_outliers_zscore(df, selected_cols, z_threshold):
    """Detects outliers using the Z-score method."""
    z_scores = np.abs((df[selected_cols] - df[selected_cols].mean()) / df[selected_cols].std())
    outliers = (z_scores > z_threshold).any(axis=1)
    return outliers

@st.cache_data
def detect_outliers_modified_zscore(df, selected_cols, modified_z_threshold):
    """Detects outliers using the Modified Z-score method."""
    median = df[selected_cols].median()
    mad = np.median(np.abs(df[selected_cols] - median))
    modified_z_scores = 0.6745 * (df[selected_cols] - median) / mad
    outliers = (np.abs(modified_z_scores) > modified_z_threshold).any(axis=1)
    return outliers

def main():
    st.markdown("""
    This section provides comprehensive data preprocessing capabilities including missing value imputation, 
    outlier detection and treatment, feature scaling, and encoding categorical variables.
    """)

    # Initialize session state for processed data
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None

    # Initialize session state for data source
    if 'data_source' not in st.session_state:
        st.session_state.data_source = None

    # Data Loading Section
    st.markdown("## 📁 Data Loading")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload your dataset (CSV format)",
            type=['csv'],
            help="Upload a CSV file for preprocessing"
        )

    with col2:
        use_default = st.button("Use Default Dataset")

    # Load data
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
            st.session_state.data_source = 'upload'
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return
    elif use_default:
        df = load_default_data()
        if df is not None:
            st.success(f"✅ Default dataset loaded! Shape: {df.shape}")
            st.session_state.data_source = 'default'
        else:
            return
    else:
        if st.session_state.data_source is None:
            st.info("👆 Please upload a dataset or use the default data to begin preprocessing.")
            return

    # Store original data
    if 'original_df' not in st.session_state:
        st.session_state.original_df = df.copy()
    original_df = st.session_state.original_df

    # Initialize processed_data in session state if it's None
    if st.session_state.processed_data is None:
        st.session_state.processed_data = df.copy()

    # Use the processed data from session state
    df = st.session_state.processed_data

    # Data Overview
    st.markdown("## 📋 Data Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{len(df):,}")

    with col2:
        st.metric("Features", len(df.columns))

    with col3:
        missing_count = df.isnull().sum().sum()
        st.metric("Missing Values", missing_count)

    with col4:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        st.metric("Numeric/Categorical", f"{len(numeric_cols)}/{len(categorical_cols)}")

    # Preprocessing Pipeline
    st.markdown("## 🔧 Preprocessing Pipeline")

    TABS = ["Missing Values", "Outlier Treatment", "Feature Scaling", "Categorical Encoding", "Handling Duplicates",
            "Data Type Conversion", "Feature Engineering", "Summary"]

    tabs = st.tabs(TABS)

    # --- Tab 1: Missing Values ---
    with tabs[0]:
        st.markdown("### 🕳️ Missing Value Analysis & Treatment")

        # Missing value analysis
        missing_df = calculate_missing_data(df)

        if not missing_df.empty:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Missing Data Summary")
                st.dataframe(missing_df)

                # Visualization
                fig = px.bar(
                    missing_df, x='Column', y='Missing %',
                    title='Missing Data Percentage by Column'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### Missing Value Treatment")

                columns_with_missing = missing_df['Column'].tolist()
                selected_cols = st.multiselect(
                    "Select columns to impute:",
                    columns_with_missing,
                    key="missing_cols"
                )

                imputation_methods = ['None', 'Mean', 'Median', 'Mode', 'Forward Fill', 'Backward Fill', 'KNN']
                methods = {}

                for col in selected_cols:
                    methods[col] = st.selectbox(
                        f"Imputation method for {col}:",
                        imputation_methods,
                        key=f"method_{col}"
                    )

                if st.button("Apply Imputation"):
                    if not selected_cols:
                        st.error("Please select columns to impute.")
                    else:
                        for col in selected_cols:
                            method = methods[col]
                            if method != 'None':
                                if method == 'Mean':
                                    df[col].fillna(df[col].mean(), inplace=True)
                                elif method == 'Median':
                                    df[col].fillna(df[col].median(), inplace=True)
                                elif method == 'Mode':
                                    df[col].fillna(df[col].mode()[0], inplace=True)
                                elif method == 'Forward Fill':
                                    df[col].fillna(method='ffill', inplace=True)
                                elif method == 'Backward Fill':
                                    df[col].fillna(method='bfill', inplace=True)
                                elif method == 'KNN':
                                    imputer = KNNImputer(n_neighbors=5)
                                    df[col] = imputer.fit_transform(df[[col]]).flatten()
                        st.success("Imputation applied successfully!")
                        st.session_state.processed_data = df.copy()  # Update session state
                        st.dataframe(df.head())  # Show preview
                        st.rerun()
        else:
            st.success("✅ No missing values found in the dataset!")

        st.markdown("#### Current Data Preview")
        st.dataframe(df.head())

        # Buttons at the bottom
        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📁 Download CSV",
                data=csv,
                file_name="preprocessed_data_tab1.csv",
                mime="text/csv"
            )
        with col2:
            if st.button("💾 Save for Next Steps", key="save_tab1"):
                st.session_state.processed_data = df.copy()
                st.success("Data saved!")

    # --- Tab 2: Outlier Treatment ---
    with tabs[1]:
        st.markdown("### 📊 Outlier Detection & Treatment")

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            selected_cols = st.multiselect(
                "Select columns for outlier analysis:",
                numeric_cols,
                key="outlier_cols"
            )

            # Boxplot Visualization
            for col in selected_cols:
                fig = px.box(df, y=col, points="all", title=f"Boxplot of {col}")
                st.plotly_chart(fig, use_container_width=True)

            outlier_method = st.selectbox(
                "Outlier detection method:",
                ['None', 'IQR Method', 'Z-Score', 'Modified Z-Score'],
                key="outlier_method"
            )

            if outlier_method != 'None':
                if not selected_cols:
                    st.error("Please select columns for outlier detection.")
                else:
                    # Outlier detection logic (caching could be added here for performance)
                    if outlier_method == 'IQR Method':
                        outliers = detect_outliers_iqr(df, selected_cols)

                    elif outlier_method == 'Z-Score':
                        z_threshold = st.slider("Z-Score threshold:", 1.0, 4.0, 3.0, 0.1)
                        outliers = detect_outliers_zscore(df, selected_cols, z_threshold)

                    else:  # Modified Z-Score
                        modified_z_threshold = st.slider("Modified Z-Score threshold:", 1.0, 4.0, 3.5, 0.1)
                        outliers = detect_outliers_modified_zscore(df, selected_cols, modified_z_threshold)

                    st.metric("Outliers Detected", outliers.sum())
                    st.metric("Outlier Percentage", f"{outliers.sum() / len(df) * 100:.1f}%")

                    treatment = st.selectbox(
                        "Select treatment method:",
                        ['None', 'Remove Outliers', 'Cap at Percentiles', 'Transform (Log)', 'Winsorize'],
                        key="outlier_treatment"
                    )

                    if treatment != 'None':
                        if treatment == 'Remove Outliers':
                            if st.button("Apply Outlier Removal"):
                                df = df[~outliers]
                                st.success(f"Removed {outliers.sum()} outliers")
                                st.session_state.processed_data = df.copy()
                                st.dataframe(df.head())
                                st.rerun()

                        elif treatment == 'Cap at Percentiles':
                            lower_pct = st.slider("Lower percentile:", 0.0, 10.0, 5.0)
                            upper_pct = st.slider("Upper percentile:", 90.0, 100.0, 95.0)

                            if st.button("Apply Capping"):
                                lower_cap = df[selected_cols].quantile(lower_pct / 100)
                                upper_cap = df[selected_cols].quantile(upper_pct / 100)
                                df[selected_cols] = np.clip(df[selected_cols], lower_cap, upper_cap)
                                st.success("Outliers capped successfully")
                                st.session_state.processed_data = df.copy()
                                st.dataframe(df.head())
                                st.rerun()

                        elif treatment == 'Transform (Log)':
                            if st.button("Apply Log Transformation"):
                                df[selected_cols] = np.log1p(df[selected_cols])
                                st.success("Log transformation applied")
                                st.session_state.processed_data = df.copy()
                                st.dataframe(df.head())
                                st.rerun()
        else:
            st.warning("No numeric columns found for outlier analysis.")

        st.markdown("#### Current Data Preview")
        st.dataframe(df.head())

        # Buttons at the bottom
        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📁 Download CSV",
                data=csv,
                file_name="preprocessed_data_tab2.csv",
                mime="text/csv"
            )
        with col2:
            if st.button("💾 Save for Next Steps", key="save_tab2"):
                st.session_state.processed_data = df.copy()
                st.success("Data saved!")

    # --- Tab 3: Feature Scaling ---
    with tabs[2]:
        st.markdown("### ⚖️ Feature Scaling")

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Scaling Configuration")

                columns_to_scale = st.multiselect(
                    "Select columns to scale:",
                    numeric_cols.tolist(),
                    default=list(numeric_cols),
                    key="scale_cols"
                )

                scaling_method = st.selectbox(
                    "Select scaling method:",
                    ['None', 'StandardScaler', 'MinMaxScaler', 'RobustScaler'],
                    help="""
                    - StandardScaler: Mean=0, Std=1
                    - MinMaxScaler: Range=[0,1]
                    - RobustScaler: Uses median and IQR
                    """
                )

                if scaling_method != 'None':
                    if not columns_to_scale:
                        st.error("Please select columns to scale.")
                    else:
                        if st.button("Apply Scaling"):
                            if scaling_method == 'StandardScaler':
                                scaler = StandardScaler()
                            elif scaling_method == 'MinMaxScaler':
                                scaler = MinMaxScaler()
                            else:  # RobustScaler
                                scaler = RobustScaler()

                            df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
                            st.success(f"{scaling_method} applied to selected columns")
                            st.session_state.processed_data = df.copy()
                            st.dataframe(df.head())
                            st.rerun()

            with col2:
                st.markdown("#### Before/After Comparison")
                comparison_col = st.selectbox(
                    "Select column to compare:",
                    columns_to_scale,
                    key="compare_col"
                )
                if comparison_col:
                    before = original_df[comparison_col] if comparison_col in original_df.columns else None
                    after = df[comparison_col]
                    st.write("Before Scaling:")
                    if before is not None:
                        st.line_chart(before)
                    else:
                        st.write("Original data not available for comparison.")
                    st.write("After Scaling:")
                    st.line_chart(after)
        else:
            st.warning("No numeric columns found for scaling.")

        st.markdown("#### Current Data Preview")
        st.dataframe(df.head())

        # Buttons at the bottom
        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📁 Download CSV",
                data=csv,
                file_name="preprocessed_data_tab3.csv",
                mime="text/csv"
            )
        with col2:
            if st.button("💾 Save for Next Steps", key="save_tab3"):
                st.session_state.processed_data = df.copy()
                st.success("Data saved!")

    # --- Tab 4: Categorical Encoding ---
    with tabs[3]:
        st.markdown("### 🔤 Categorical Variable Encoding")

        categorical_cols = df.select_dtypes(include=['object']).columns

        if len(categorical_cols) > 0:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Encoding Configuration")

                selected_cols = st.multiselect(
                    "Select columns to encode:",
                    categorical_cols,
                    key="encode_cols"
                )

                encoding_methods = {}

                for col in selected_cols:
                    st.markdown(f"**{col}** ({df[col].nunique()} unique values)")

                    unique_values = df[col].value_counts()
                    st.write(f"Top values: {', '.join(unique_values.head(5).index.tolist())}")

                    encoding_method = st.selectbox(
                        f"Encoding method for {col}:",
                        ['None', 'Label Encoding', 'One-Hot Encoding'],  # Removed Binary Encoding
                        key=f"encode_{col}"
                    )
                    encoding_methods[col] = encoding_method

                if st.button("Apply Encoding"):
                    if not selected_cols:
                        st.error("Please select columns to encode.")
                    else:
                        for col in selected_cols:
                            encoding_method = encoding_methods[col]
                            if encoding_method == 'Label Encoding':
                                le = LabelEncoder()
                                df[col] = le.fit_transform(df[col].astype(str))
                                st.success(f"✅ Label encoding applied to {col}")
                                st.session_state.processed_data = df.copy()
                                st.dataframe(df.head())
                                st.rerun()

                            elif encoding_method == 'One-Hot Encoding':
                                dummies = pd.get_dummies(df[col], prefix=col, dtype=int)  # Ensure 0 and 1
                                df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
                                st.success(f"One-hot encoding applied to {col}")
                                st.session_state.processed_data = df.copy()
                                st.dataframe(df.head())
                                st.rerun()

            with col2:
                st.markdown("#### Categorical Variable Summary")
                if selected_cols:
                    st.write("Encoded Columns Preview:")
                    st.dataframe(df[selected_cols].head())
        else:
            st.success("✅ No categorical columns found that need encoding.")

        st.markdown("#### Current Data Preview")
        st.dataframe(df.head())

        # Buttons at the bottom
        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📁 Download CSV",
                data=csv,
                file_name="preprocessed_data_tab4.csv",
                mime="text/csv"
            )
        with col2:
            if st.button("💾 Save for Next Steps", key="save_tab4"):
                st.session_state.processed_data = df.copy()
                st.success("Data saved!")

    # --- Tab 5: Handling Duplicates ---
    with tabs[4]:
        st.markdown("### 👯 Handling Duplicates")

        duplicates = df.duplicated()
        num_duplicates = duplicates.sum()

        st.markdown(f"#### Duplicate Records: {num_duplicates}")

        if num_duplicates > 0:
            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(df[duplicates].head())

            with col2:
                st.markdown("#### Duplicate Handling Options")
                if st.button("Remove Duplicates"):
                    df.drop_duplicates(inplace=True)
                    st.success("Duplicates removed successfully!")
                    st.session_state.processed_data = df.copy()
                    st.dataframe(df.head())
                    st.rerun()
        else:
            st.success("✅ No duplicate records found!")

        st.markdown("#### Current Data Preview")
        st.dataframe(df.head())

        # Buttons at the bottom
        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📁 Download CSV",
                data=csv,
                file_name="preprocessed_data_tab5.csv",
                mime="text/csv"
            )
        with col2:
            if st.button("💾 Save for Next Steps", key="save_tab5"):
                st.session_state.processed_data = df.copy()
                st.success("Data saved!")

    # --- Tab 6: Data Type Conversion ---
    with tabs[6]:
        st.markdown("### ✨ Feature Engineering")

        st.markdown("#### Create New Features")

        col1, col2 = st.columns(2)

        with col1:
            feature_name = st.text_input("New Feature Name:")
            feature_type = st.selectbox("Feature Type:", ['None', 'Numeric', 'Categorical', 'Binning'])

            if feature_type == 'Numeric':
                operation = st.selectbox("Operation:", ['Addition', 'Subtraction', 'Multiplication', 'Division'])
                col1_name = st.selectbox("Column 1:", df.columns)
                col2_name = st.selectbox("Column 2:", df.columns)

                if st.button("Create Feature"):
                    if not feature_name or not col1_name or not col2_name:
                        st.error("Please provide all required inputs for feature creation.")
                    else:
                        try:
                            if operation == 'Addition':
                                df[feature_name] = df[col1_name] + df[col2_name]
                            elif operation == 'Subtraction':
                                df[feature_name] = df[col1_name] - df[col2_name]
                            elif operation == 'Multiplication':
                                df[feature_name] = df[col1_name] * df[col2_name]
                            elif operation == 'Division':
                                df[feature_name] = df[col1_name] / df[col2_name]
                            st.success(f"✅ Created new feature: {feature_name}")
                            st.session_state.processed_data = df.copy()
                            st.dataframe(df.head())
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Feature creation failed: {e}")

            elif feature_type == 'Categorical':
                # Example: Combining two categorical columns
                col1_name = st.selectbox("Column 1:", df.columns)
                col2_name = st.selectbox("Column 2:", df.columns)
                separator = st.text_input("Separator:", "_")

                if st.button("Create Combined Feature"):
                    if not feature_name or not col1_name or not col2_name:
                        st.error("Please provide all required inputs for feature creation.")
                    else:
                        try:
                            df[feature_name] = df[col1_name].astype(str) + separator + df[col2_name].astype(str)
                            st.success(f"✅ Created new combined feature: {feature_name}")
                            st.session_state.processed_data = df.copy()
                            st.dataframe(df.head())
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Feature creation failed: {e}")

            elif feature_type == 'Binning':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                col_to_bin = st.selectbox("Select a numeric column to bin:", numeric_cols)
                num_bins = st.slider("Number of bins:", 2, 10, 5)

                if st.button("Create Binned Feature"):
                    if not col_to_bin:
                        st.error("Please select a column to bin.")
                    else:
                        try:
                            new_col_name = f"{col_to_bin}_Binned"
                            df[new_col_name] = pd.cut(df[col_to_bin], bins=num_bins, labels=[f"Group {i+1}" for i in range(num_bins)])
                            st.success(f"✅ Created new binned feature: {new_col_name}")
                            st.session_state.processed_data = df.copy()
                            st.dataframe(df.head())
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Feature creation failed: {e}")

        st.markdown("#### Current Data Preview")
        st.dataframe(df.head())

        # Buttons at the bottom
        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📁 Download CSV",
                data=csv,
                file_name="preprocessed_data_tab6.csv",
                mime="text/csv"
            )
        with col2:
            if st.button("💾 Save for Next Steps", key="save_tab6"):
                st.session_state.processed_data = df.copy()
                st.success("Data saved!")

    # --- Tab 7: Data Type Conversion ---
    with tabs[7]:
        st.markdown("### 📊 Preprocessing Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Original vs Processed Data")

            comparison_stats = pd.DataFrame({
                'Metric': ['Rows', 'Columns', 'Missing Values', 'Numeric Columns', 'Categorical Columns'],
                'Original': [
                    len(original_df),
                    len(original_df.columns),
                    original_df.isnull().sum().sum(),
                    len(original_df.select_dtypes(include=[np.number]).columns),
                    len(original_df.select_dtypes(include=['object']).columns)
                ],
                'Processed': [
                    len(df),
                    len(df.columns),
                    df.isnull().sum().sum(),
                    len(df.select_dtypes(include=[np.number]).columns),
                    len(df.select_dtypes(include=['object']).columns)
                ]
            })

            st.dataframe(comparison_stats)

        with col2:
            st.markdown("#### Data Quality Metrics")

            # Calculate data quality score
            missing_score = max(0, 100 - (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100))

            # Completeness
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100

            st.metric("Data Completeness", f"{completeness:.1f}%")
            st.metric("Rows Retained", f"{len(df)} / {len(original_df)}")
            st.metric("Features Generated", len(df.columns) - len(original_df.columns))

        st.markdown("#### Processed Dataset Preview")
        st.dataframe(df)

        # Store processed data in session state
        st.session_state.processed_data = df

        st.markdown("#### Export Processed Data")
        col1, col2, col3 = st.columns(3)

        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📁 Download CSV",
                data=csv,
                file_name="preprocessed_data.csv",
                mime="text/csv"
            )

        with col2:
            if st.button("💾 Save for Feature Selection"):
                st.session_state.preprocessed_for_feature_selection = df.copy()
                st.success("Data saved for Feature Selection!")

        with col3:
            if st.button("🔄 Reset to Original"):
                st.session_state.processed_data = original_df.copy()
                st.success("Data reset to original state")
                st.rerun()

if __name__ == "__main__":
    main()