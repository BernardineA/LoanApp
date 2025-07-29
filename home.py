import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from PIL import Image
import time
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Loan Default Prediction App",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- CSS Styling -------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.main .block-container {padding-top: 1rem; max-width: 100%;}
.hero-section {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 4rem 2rem;
    border-radius: 20px;
    text-align: center;
    margin: 2rem 0;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
}
.hero-title {font-size: 3.5rem; font-weight: 700; margin-bottom: 1rem;}
.hero-subtitle {font-size: 1.3rem; margin-bottom: 2rem;}
.company-logo {display: flex; align-items: center; justify-content: center; padding: 1rem 2rem;}
.logo-text {font-size: 2rem; font-weight: 700; margin-left: 0.5rem;}
.feature-card, .team-card, .tech-item {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    margin: 0.5rem;
}
.team-avatar {
    width: 100px; height: 100px; border-radius: 50%;
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem; color: white; font-weight: 600; margin: auto;
}
.section-title {text-align: center; font-size: 2.5rem; font-weight: 700; margin-top: 3rem;}
.section-subtitle {text-align: center; font-size: 1.1rem; color: #7f8c8d;}
.stats-container {background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 3rem 2rem; border-radius: 20px; margin: 3rem 0; color: white;}
.stats-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 2rem; text-align: center;}
.stat-number {font-size: 3rem; font-weight: 700;}
.stat-label {font-size: 1rem;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Function to generate sample analytics data
def generate_sample_analytics():
    models = ['Linear Regression', 'Random Forest', 'XGBoost']
    r2_scores = [0.78, 0.85, 0.87]
    features = ['Credit Score', 'Income', 'Debt Ratio', 'Loan Amount', 'Employment Years']
    importance = [0.35, 0.28, 0.22, 0.15, 0.10]
    np.random.seed(42)
    credit_scores = np.random.normal(650, 100, 200)
    default_risk = 10 - (credit_scores - 300) / 55 + np.random.normal(0, 1, 200)
    default_risk = np.clip(default_risk, 0, 12)
    return {
        'models': models,
        'r2_scores': r2_scores,
        'features': features,
        'importance': importance,
        'credit_scores': credit_scores,
        'default_risk': default_risk
    }
 # Analytics Section
    analytics = generate_sample_analytics()
    st.markdown("""
        <div class="section-header">
            <div class="section-title">📊 Sample Analytics</div>
            <div class="section-subtitle">Insights from model performance and feature impact</div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.bar(
            x=analytics['models'],
            y=analytics['r2_scores'],
            color=analytics['r2_scores'],
            title="Model Performance (R² Score)"
        ), use_container_width=True)

        st.plotly_chart(px.bar(
            x=analytics['importance'],
            y=analytics['features'],
            orientation='h',
            title="Feature Importance"
        ), use_container_width=True)

    with col2:
        st.plotly_chart(px.scatter(
            x=analytics['credit_scores'],
            y=analytics['default_risk'],
            title="Credit Score vs Default Risk"
        ), use_container_width=True)

        st.plotly_chart(px.bar(
            x=["$0–10k", "$10–25k", "$25–50k", "$50k+"],
            y=[15, 22, 28, 12],
            title="Default Rate by Loan Range"
        ), use_container_width=True)


def main():
    import streamlit as st
    import base64

    # Set page config
    st.set_page_config(
        page_title="Loan Default Prediction App",
        page_icon="🚀",
        layout="wide"
    )

    # Function to convert image to base64
    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        return f"data:image/jpeg;base64,{encoded}"

    # Convert your local image (ensure the path is correct)
    logo_base64 = get_base64_image("images/Logo.jpg")

    # Main header with embedded logo
    st.markdown(
        f'''
        <div class="hero-section" style="text-align: center; padding: 1rem 0;">
            <div class="company-logo" style="display: flex; align-items: center; justify-content: center;">
                <img src="{logo_base64}" alt="Company Logo" style="height: 100px; margin-right: 15px;">
                <span class="logo-text" style="font-size: 3rem; font-weight: bold; color: #f7f5f5;">Data Insights</span>
            </div>
            <h1 class="hero-title" style="margin-top: 0.5rem;">Loan Default Prediction System</h1>
            <p class="hero-subtitle" style="font-size: 1.2rem; color: #333;">Transforming Information Into Actions</p>
        </div>
        ''',
        unsafe_allow_html=True
    )



    # Overview Section
    st.markdown('<h2 class="sub-header">🏠 Overview</h2>', unsafe_allow_html=True)
    st.markdown("""
    Welcome to our **Loan Default Prediction Web App** – an interactive machine learning application designed for 
    academic purposes in Applied Regression & Machine Learning. This application simulates a real-world loan system 
    and predicts loan default amounts using comprehensive borrower history and financial behavior analysis.

    Our app provides hands-on experience with the complete machine learning pipeline, from data exploration to 
    model deployment, making it an excellent educational tool for understanding practical ML applications in finance.
    """)

    # Project Info Section
    st.markdown('<h2 class="sub-header">📌 Project Information</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎓 Educational Purpose")
        st.markdown("""
        This application allows users to:
        - **Explore and visualize** comprehensive loan datasets
        - **Preprocess and clean** data for optimal model performance
        - **Perform feature selection** using statistical methods
        - **Train and compare models** (Linear Regression, Random Forest, XGBoost)
        - **Tune models** with GridSearchCV for optimal hyperparameters
        - **Generate live predictions** for new loan applications
        """)

    with col2:
        st.markdown("### 🔧 Technical Implementation")
        st.markdown("""
        Built using cutting-edge technologies:
        - **Python & Streamlit** for web application framework
        - **Pandas & NumPy** for efficient data manipulation
        - **Scikit-learn** for machine learning algorithms
        - **XGBoost** for advanced gradient boosting
        - **Matplotlib & Seaborn** for static visualizations
        - **Plotly** for interactive charts and dashboards
        """)

    # Meet the Team Section
    st.markdown('<h2 class="sub-header">🧑‍💻 Meet the Team</h2>', unsafe_allow_html=True)

    # Define team members with image paths
    team_members = [
        {"name": "Bernardine Akorfa Gawu", "id": "22253324", "role": "Data Scientist", "image": "Images/Bernardine.jpg"},
        {"name": "Abigail Amissah", "id": "22253929", "role": "ML Engineer", "image": "Images/Abigail.jpg"},
        {"name": "Samuel Asare", "id": "22253156", "role": "Full Stack Developer", "image": "Images/Samuel.jpg"},
        {"name": "Afful Francis Gyan", "id": "22253332", "role": "Data Analyst", "image": "Images/Francis.jpg"},
        {"name": "Gloria Odamtten", "id": "22252377", "role": "Project Manager", "image": "Images/Gloria.jpg"}
    ]

    # Display cards in columns
    cols = st.columns(5)
    for i, member in enumerate(team_members):
        with cols[i]:
            try:
                st.image(member["image"], width=120, caption=None)
            except FileNotFoundError:
                st.warning(f"Image not found: {member['image']}")

            st.markdown(f"""
                <div style="text-align: left;">
                    <h4 style="margin: 0.5rem 0;">{member['name']}</h4>
                    <p style="color: #666; margin: 0;">{member['role']}</p>
                    <p style="font-weight: bold; color: #1f77b4; margin: 0.3rem 0;">ID: {member['id']}</p>
                </div>
            """, unsafe_allow_html=True)

    # Project Objective Section
    st.markdown('<h2 class="sub-header">🎯 Project Objective</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 2rem; border-radius: 10px; border-left: 5px solid #1f77b4;">
            <p style="font-size: 1.1rem; line-height: 1.6;">
            Our project simulates a comprehensive <strong>loan management system</strong> using regression-based 
            machine learning techniques. The application predicts default amounts based on historical borrower data, 
            helping financial institutions assess loan risks more effectively and make informed lending decisions.
            </p>
            <p style="font-size: 1.1rem; line-height: 1.6;">
            By leveraging advanced ML algorithms and statistical analysis, we aim to demonstrate how data science 
            can revolutionize traditional financial risk assessment processes.
            </p>
        </div>
        """, unsafe_allow_html=True)



    # Tools & Technologies Section
    st.markdown('<h2 class="sub-header">🛠 Tools & Technologies</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 3rem;">🐍</div>
            <h4>Python & Streamlit</h4>
            <p>Core development framework</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 3rem;">📊</div>
            <h4>Data Science Stack</h4>
            <p>Pandas, NumPy, Seaborn</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 3rem;">🧠</div>
            <h4>Machine Learning</h4>
            <p>Scikit-learn, XGBoost</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 3rem;">📈</div>
            <h4>Visualization</h4>
            <p>Matplotlib, Plotly</p>
        </div>
        """, unsafe_allow_html=True)

   # Deliverables Section
    st.markdown('<h2 class="sub-header">📍 Deliverables</h2>', unsafe_allow_html=True)

    deliverables = [
        "Multi-page interactive Streamlit web application",
        "Well-documented and modular source code",
        "Comprehensive data exploration and visualization tools",
        "Multiple ML model implementations and comparisons",
        "Interactive prediction interface for real-time results",
        "Professional presentation materials for class demo"
    ]

    col1, col2 = st.columns(2)
    for i, deliverable in enumerate(deliverables):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
            <div class="deliverable-item">
                ✅ {deliverable}
            </div>
            """, unsafe_allow_html=True)

    # Call to Action
    st.markdown("""
        <div class="section-header">
            <div class="section-title">📍 Ready to Dive In?</div>
        </div>
    """, unsafe_allow_html=True)

    # Navigation Instructions
    st.markdown('<h2 class="sub-header">🚀 Get Started</h2>', unsafe_allow_html=True)
    st.info("""
    Use the sidebar navigation to access different sections of the application:

    1. **Data Exploration** - Discover insights in your loan dataset
    2. **Data Preprocessing** - Clean and prepare data for modeling
    3. **Feature Selection** - Identify the most important predictive features
    4. **Model Training** - Train and compare multiple ML algorithms
    5. **Model Tuning** - Optimize hyperparameters for best performance
    6. **Live Prediction** - Make real-time loan default predictions
    """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>Applied Regression & Machine Learning Project</strong></p>
        <p>Academic Year 2024-2025 | Built with ❤️ using Streamlit and Python</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()


