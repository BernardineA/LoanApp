import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Data Exploration", page_icon="📊", layout="wide")

# Assuming you want to default to light mode if no theme selection is available
theme_mode = "Light"  # Default to light mode

if theme_mode == "Dark":
    st.markdown("""
        <style>
        body {
            background-color: #111 !important;
            color: #FFFFFF !important;
        }
        .stApp {
            background-color: #111 !important;
        }
        div.block-container.css-91ifnm.eqr7zpz1 {
            background-color: #111 !important;
            color: #FFFFFF !important;
        }
        .css-18e3th9, .css-1d391kg, .css-1v0mbdj, .block-container {
            color: #FFFFFF !important;
        }
        .streamlit-expanderHeader {
            color: #FFFFFF !important;
        }
        </style>
    """, unsafe_allow_html=True)

st.title("📊 Data Exploration")


@st.cache_data
def load_default_data():
    try:
        df = pd.read_csv("data/Loan_Default.csv")
        return df
    except Exception as e:
        return None


@st.cache_data
def load_uploaded_file(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df


@st.cache_data
def get_numeric_summary(df):
    return df.describe().T


@st.cache_data
def get_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr()


@st.cache_data
def get_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers


def show_metadata():
    st.markdown("## 🔍 Dataset Metadata")
    st.markdown("""
    - **Source**: [Kaggle - Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset)
    - **Creator**: Yasser H.
    - **Context**: The dataset contains financial information about individuals and their history of loan payments.
    - **Size**: ~10,000 records, 27 features
    - **Data Types**: Numerical, Categorical
    - **Usage**: Loan default prediction, credit risk modeling
    - **License**: CC0: Public Domain
    - **Last Updated**: 2021

    ### Column Descriptions:
    - **Customer_ID**: Unique identifier for each customer.
    - **Month**: Month of the transaction record.
    - **Age**: Age of the customer.
    - **Occupation**: Customer's occupation type.
    - **Annual_Income**: Annual income of the customer.
    - **Monthly_Inhand_Salary**: Monthly income in hand.
    - **Num_Bank_Accounts**: Number of bank accounts held.
    - **Num_Credit_Card**: Number of credit cards owned.
    - **Interest_Rate**: Interest rate on loans.
    - **Num_of_Loan**: Number of loans taken.
    - **Type_of_Loan**: Types of loans taken.
    - **Delay_from_due_date**: Average delay in paying EMIs.
    - **Num_of_Delayed_Payment**: Number of delayed EMI payments.
    - **Changed_Credit_Limit**: Whether credit limit changed.
    - **Num_Credit_Inquiries**: Number of credit inquiries.
    - **Credit_Mix**: Type of credit used (Good/Bad/Standard).
    - **Outstanding_Debt**: Total outstanding debt.
    - **Credit_Utilization_Ratio**: Used credit as a ratio.
    - **Credit_History_Age**: Age of credit history.
    - **Payment_of_Min_Amount**: Whether minimum amount paid.
    - **Total_EMI_per_month**: EMI amount per month.
    - **Amount_invested_monthly**: Monthly investment amount.
    - **Payment_Behaviour**: EMI payment behavior.
    - **Monthly_Balance**: Monthly account balance.
    - **Loan_Default**: Target variable. 1 = Defaulted, 0 = Not Defaulted.
    """)


def main():
    st.markdown("""
    This section provides comprehensive exploratory data analysis (EDA) capabilities for understanding 
    loan default patterns and relationships between different variables.
    """)

    st.markdown("## 📁 Data Loading")
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload your loan dataset (CSV format)",
            type=['csv'],
            help="Upload a CSV file containing loan data for analysis"
        )

    if "use_default_clicked" not in st.session_state:
        st.session_state.use_default_clicked = False

    with col2:
        if st.button("Use Default Dataset"):
            st.session_state.use_default_clicked = True

    metadata_shown = False

    if uploaded_file is not None:
        df = load_uploaded_file(uploaded_file)
        st.session_state['data'] = df  # Store the dataframe in session state
        st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
    elif st.session_state.use_default_clicked:
        df = load_default_data()
        if df is not None:
            st.session_state['data'] = df  # Store the dataframe in session state
            st.success(f"✅ Default dataset loaded! Shape: {df.shape}")
            st.info("This is the default dataset from our system.")
            if st.checkbox("🔍 Show Dataset Metadata"):
                show_metadata()
                metadata_shown = True
        else:
            st.error("⚠️ Could not load default dataset.")
            return
    else:
        st.info("👇 Please upload a dataset or click 'Use Default Dataset' to begin exploration.")
        return

    # Access the dataframe from session state
    if 'data' not in st.session_state:
        return  # Exit if no data is loaded
    df = st.session_state['data']

    with st.expander("📊 Quick Summary Panel", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
            st.markdown("Total number of records in the dataset.")
        with col2:
            st.metric("Features", len(df.columns))
            st.markdown("Number of columns (features) in the dataset.")
        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Features", len(numeric_cols))
            st.markdown("Number of numeric columns in the dataset.")
        with col4:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Missing Data %", f"{missing_pct:.1f}%")
            st.markdown("Percentage of missing values in the entire dataset.")

    st.sidebar.markdown("### 🧽 Column Filter")
    selected_columns = st.sidebar.multiselect("Select columns to include:", options=df.columns,
                                              default=list(df.columns))
    df = df[selected_columns]

    tab1, tab2, tab3, tab4 = st.tabs(["🔍 View Data", "📈 Statistical Summary", "📊 Visualizations", "🧪 Data Quality"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### View Data Rows")
            row_view_option = st.selectbox("Choose row display option:", ["First", "Last", "Custom"])
            num_rows = st.number_input("Number of rows to display:", min_value=1, max_value=len(df), value=5)
            if row_view_option == "First":
                st.dataframe(df.head(num_rows))
                st.markdown("Displaying the first N rows of the dataset.")
            elif row_view_option == "Last":
                st.dataframe(df.tail(num_rows))
                st.markdown("Displaying the last N rows of the dataset.")
            else:
                start_idx = st.number_input("Start index (0-based):", min_value=0, max_value=len(df) - num_rows, value=0) # Corrected max_value
                st.dataframe(df.iloc[int(start_idx):int(start_idx) + int(num_rows)])
                st.markdown("Displaying a custom range of rows from the dataset.")
        with col2:
            st.markdown("### Data Types & Missing Values")
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Missing Values': df.isnull().sum(),
                'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(info_df)
            st.markdown("Information about each column, including data type and missing value counts.")

    with tab2:
        st.subheader("📈 Statistical Summary")
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            st.dataframe(get_numeric_summary(df), use_container_width=True)
            st.markdown("Descriptive statistics for numeric columns (mean, std, min, max, etc.).")
        else:
            st.warning("No numeric columns found for statistical summary.")

    with tab3:
        st.subheader("📊 Data Visualization")
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            subtab1, subtab2, subtab3, subtab4 = st.tabs([
                "Distribution", "Correlation", "Scatter", "Boxplot"
            ])
            with subtab1:
                # Improved default for distribution plot
                default_dist_col = numeric_df.columns[0] if len(numeric_df.columns) > 0 else None
                dist_col = st.selectbox("Select column for distribution plot:", numeric_df.columns, index=0 if default_dist_col else 0)
                fig = px.histogram(df, x=dist_col, title=f"Distribution of {dist_col}")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"Distribution plot of the '{dist_col}' column.")

            with subtab2:
                corr = get_correlation_matrix(df)
                fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("Correlation matrix showing the relationships between numeric columns.")

            with subtab3:
                # Improved defaults for scatter plot
                x_col_index = 0 if len(numeric_df.columns) > 0 else None
                y_col_index = 1 if len(numeric_df.columns) > 1 else None

                x_col = st.selectbox("Select X-axis feature:", numeric_df.columns, index=x_col_index if x_col_index is not None else 0)
                y_col = st.selectbox("Select Y-axis feature:", numeric_df.columns, index=y_col_index if y_col_index is not None else 0)
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"Scatter plot showing the relationship between '{x_col}' and '{y_col}'.")

            with subtab4:
                # Improved default for box plot
                selected_col_index = 0 if len(numeric_df.columns) > 0 else None
                selected_col = st.selectbox("Select Column:", numeric_df.columns, index=selected_col_index if selected_col_index is not None else 0)
                fig = px.box(df, y=selected_col, title=f"Boxplot of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"Box plot of the '{selected_col}' column, highlighting potential outliers.")
        else:
            st.warning("No numeric data available for visualizations.")

    with tab4:
        st.subheader("🧪 Data Quality Assessment")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Missing Data Pattern")
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                fig = px.bar(x=missing_data.index, y=missing_data.values, title="Missing Values by Column")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("Bar chart showing the number of missing values in each column.")
            else:
                st.success("✅ No missing data found!")

        with col2:
            st.markdown("### Outlier Detection")
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                outlier_column = st.selectbox("Select column for outlier detection:", numeric_df.columns)
                outliers = get_outliers(df, outlier_column)
                st.metric("Outliers Found", len(outliers))
                st.metric("Outlier %", f"{len(outliers) / len(df) * 100:.1f}%")
                st.markdown(f"Number and percentage of outliers detected in the '{outlier_column}' column.")
                if not outliers.empty:
                    st.dataframe(outliers)
                    st.markdown("DataFrame containing the detected outliers.")

    st.markdown("---")
    st.markdown("## 📀 Export Options")
    col1, col2, col3, col4 = st.columns(4)  # Added a fourth column

    with col1:
        if st.button("📊 Download EDA Report"):
            # Placeholder for EDA report generation
            st.info("Feature coming soon: Automated EDA report generation")

    with col2:
        if st.button("📈 Download Visualizations"):
            # Placeholder for downloading visualizations
            st.info("Feature coming soon: Export all charts as PDF/Images")

    with col3:
        csv = df.to_csv(index=False)
        st.download_button("📁 Download Processed Data", data=csv, file_name="explored_data.csv", mime="text/csv", help="Download the processed data as a CSV file.")

    with col4:
        # Button to save data for preprocessing
        if st.button("💾 Save Data for Preprocessing"):
            st.session_state['preprocessing_data'] = df  # Save to a different key
            st.success("✅ Data saved for preprocessing!")
            st.info("You can now navigate to the preprocessing page to use this data.")

    # Example of how to access the data on the preprocessing page (This would be in your preprocessing page code)
    # if 'preprocessing_data' in st.session_state:
    #     preprocessing_df = st.session_state['preprocessing_data']
    #     st.write("Data loaded for preprocessing:")
    #     st.dataframe(preprocessing_df.head())
    # else:
    #     st.info("No data saved for preprocessing yet.")


if __name__ == "__main__":
    main()