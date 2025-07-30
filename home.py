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
import base64
import os  # Import the 'os' module

# Set page configuration
st.set_page_config(
    page_title="Loan Default Prediction App",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)
warnings.filterwarnings('ignore')

# ------------------- CSS Styling -------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

body {
    font-family: 'Inter', sans-serif;
    background-color: #f4f4f4;
    color: #333;
}

.main .block-container {
    padding-top: 2rem;
    max-width: 90%;
}

.hero-section {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 4rem 2rem;
    border-radius: 20px;
    text-align: center;
    margin: 2rem 0;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
}

.hero-subtitle {
    font-size: 1.3rem;
    margin-bottom: 2rem;
    color: #eee;
}

.company-logo {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1rem 2rem;
}

.logo-text {
    font-size: 2rem;
    font-weight: 700;
    margin-left: 0.5rem;
    color: #fff;
}

.feature-card, .team-card, .tech-item {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    margin: 0.5rem;
    transition: transform 0.3s ease-in-out;
}

.feature-card:hover, .team-card:hover, .tech-item:hover {
    transform: translateY(-5px);
}

.team-avatar {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    color: white;
    font-weight: 600;
    margin: auto;
    margin-bottom: 1rem;
}

.section-title {
    text-align: center;
    font-size: 2.5rem;
    font-weight: 700;
    margin-top: 3rem;
    color: #333;
}

.section-subtitle {
    text-align: center;
    font-size: 1.1rem;
    color: #7f8c8d;
    margin-bottom: 2rem;
}

.stats-container {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 3rem 2rem;
    border-radius: 20px;
    margin: 3rem 0;
    color: white;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 2rem;
    text-align: center;
}

.stat-number {
    font-size: 3rem;
    font-weight: 700;
}

.stat-label {
    font-size: 1rem;
}

.sub-header {
    font-size: 2rem;
    font-weight: 600;
    color: #333;
    margin-top: 2rem;
    margin-bottom: 1rem;
    border-bottom: 2px solid #ddd;
    padding-bottom: 0.5rem;
}

.deliverable-item {
    background-color: #fff;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
    margin-bottom: 0.5rem;
    transition: background-color 0.3s ease;
}

.deliverable-item:hover {
    background-color: #f9f9f9;
}

.team-member-card {
    background-color: #fff;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
    text-align: center;
    transition: transform 0.3s ease;
}

.team-member-card:hover {
    transform: translateY(-5px);
}

.team-member-image {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    object-fit: cover;
    margin-bottom: 1rem;
}

.team-member-name {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
}

.team-member-role {
    color: #777;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
}

.team-member-id {
    font-weight: bold;
    color: #007bff;
}

.project-objective-box {
    background-color: #f8f9fa;
    padding: 2rem;
    border-radius: 10px;
    border-left: 5px solid #007bff;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
}

.project-objective-text {
    font-size: 1.1rem;
    line-height: 1.6;
    color: #555;
}

.tools-tech-container {
    display: flex;
    justify-content: space-around;
    margin-top: 2rem;
}

.tools-tech-item {
    text-align: center;
    padding: 1rem;
    border-radius: 10px;
    transition: transform 0.3s ease;
}

.tools-tech-item:hover {
    transform: translateY(-5px);
}

.tools-tech-icon {
    font-size: 3rem;
    margin-bottom: 0.5rem;
}

.tools-tech-title {
    font-size: 1.1rem;
    font-weight: 600;
}

.tools-tech-description {
    color: #777;
    font-size: 0.9rem;
}

.navigation-instructions {
    background-color: #e9ecef;
    padding: 1.5rem;
    border-radius: 10px;
    margin-top: 2rem;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
}

.navigation-instructions h3 {
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.navigation-instructions ul {
    list-style-type: none;
    padding-left: 0;
}

.navigation-instructions li {
    margin-bottom: 0.5rem;
}

.footer {
    text-align: center;
    color: #777;
    padding: 2rem;
    border-top: 1px solid #ddd;
    margin-top: 3rem;
}

/* Hide Streamlit Menu and Footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Function to convert image to base64
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        return f"data:image/jpeg;base64,{encoded}"
    except FileNotFoundError:
        st.error(f"Image not found: {image_path}")
        return None  # Return None if the image is not found

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

def main():
    # Define image paths
    logo_path = "images/logo.jpg"
    team_images = {
        "Bernardine Akorfa Gawu": "images/Bernardine.jpeg",
        "Abigail Amissah": "images/Abigail.jpeg",
        "Samuel Asare": "images/Samuel.jpeg",
        "Afful Francis Gyan": "images/Francis.jpeg",
        "Gloria Odamtten": "images/Gloria.jpeg"
    }

    # Check if the images directory exists
    if not os.path.exists("images"):
        st.error("The 'images' directory does not exist. Please create it and add the required images.")
        return

    # Convert logo to base64
    logo_base64 = get_base64_image(logo_path)

    # Main header with embedded logo
    st.markdown(
        f'''
        <div class="hero-section" style="text-align: center; padding: 1rem 0;">
            <span class="logo-text" style="font-size: 3rem; font-weight: bold; color: #f7f5f5;">Data Insights</span>
            <h1 class="hero-title" style="margin-top: 0.5rem;">Loan Default Prediction System</h1>
            <p class="hero-subtitle" style="font-size: 1.2rem; color: #eee;">Transforming Information Into Actions</p>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # Stats section
    st.markdown(
        '<div class="stats-container"><div class="stats-grid"><div><div class="stat-number">5</div><div class="stat-label">Team Members</div></div><div><div class="stat-number">3</div><div class="stat-label">ML Models</div></div><div><div class="stat-number">6</div><div class="stat-label">App Pages</div></div><div><div class="stat-number">87%</div><div class="stat-label">Best Accuracy</div></div></div></div>',
        unsafe_allow_html=True)

    # Overview Section
    st.markdown('<h2 class="sub-header">🏠 Overview</h2>', unsafe_allow_html=True)
    st.markdown("""
    Welcome to our **Loan Default Prediction Web App** – an interactive machine learning application designed for 
    academic purposes in Regression & Machine Learning. This application simulates a real-world loan system 
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
        - **Train and compare models** (Linear Regression, Random Forest)
        - **Generate live predictions** for new loan applications
        """)

    with col2:
        st.markdown("### 🔧 Technical Implementation")
        st.markdown("""
        Built using cutting-edge technologies:
        - **Python & Streamlit** for web application framework
        - **Pandas & NumPy** for efficient data manipulation
        - **Scikit-learn** for machine learning algorithms
        - **Matplotlib & Seaborn** for static visualizations
        - **Plotly** for interactive charts and dashboards
        """)

    # Meet the Team Section
    st.markdown('<h2 class="sub-header">🧑‍💻 Meet the Team</h2>', unsafe_allow_html=True)

    # Define team members with image paths
    team_members = [
        {"name": "Bernardine Akorfa Gawu", "id": "22253324", "role": "Data Scientist", "image": "images/Bernardine.jpg"},
        {"name": "Abigail Amissah", "id": "22253929", "role": "ML Engineer", "image": "images/Abigail.jpg"},
        {"name": "Samuel Asare", "id": "22253156", "role": "Full Stack Developer", "image": "images/Samuel.jpg"},
        {"name": "Afful Francis Gyan", "id": "22253332", "role": "Data Analyst", "image": "images/Francis.jpg"},
        {"name": "Gloria Odamtten", "id": "22252377", "role": "Project Manager", "image": "images/Gloria.jpg"}
    ]

    # Display cards in columns
    cols = st.columns(5)
    for i, member in enumerate(team_members):
        with cols[i]:
            image_base64 = get_base64_image(member['image'])
            if image_base64:
                st.markdown(f"""
                    <div class="team-member-card">
                        <img src="{image_base64}" alt="{member['name']}" class="team-member-image">
                        <h3 class="team-member-name">{member['name']}</h3>
                        <p class="team-member-role">{member['role']}</p>
                        <p class="team-member-id">ID: {member['id']}</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"Could not display image for {member['name']}")

    # Project Objective Section
    st.markdown('<h2 class="sub-header">🎯 Project Objective</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="project-objective-box">
            <p class="project-objective-text">
            Our project simulates a comprehensive <strong>loan prediction system</strong> using regression-based 
            machine learning techniques. The application predicts default amounts based on historical borrower data, 
            helping financial institutions assess loan risks more effectively and make informed lending decisions.
            </p>
            <p class="project-objective-text">
            By leveraging advanced ML algorithms and statistical analysis, we aim to demonstrate how data science 
            can revolutionize traditional financial risk assessment processes.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Tools & Technologies Section
    st.markdown('<h2 class="sub-header">🛠 Tools & Technologies</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="tools-tech-container">
        <div class="tools-tech-item">
            <div class="tools-tech-icon">🐍</div>
            <h4 class="tools-tech-title">Python & Streamlit</h4>
            <p class="tools-tech-description">Core development framework</p>
        </div>
        <div class="tools-tech-item">
            <div class="tools-tech-icon">📊</div>
            <h4 class="tools-tech-title">Data Science Stack</h4>
            <p class="tools-tech-description">Pandas, NumPy, Seaborn</p>
        </div>
        <div class="tools-tech-item">
            <div class="tools-tech-icon">🧠</div>
            <h4 class="tools-tech-title">Machine Learning</h4>
            <p class="tools-tech-description">Scikit-learn, XGBoost</p>
        </div>
        <div class="tools-tech-item">
            <div class="tools-tech-icon">📈</div>
            <h4 class="tools-tech-title">Visualization</h4>
            <p class="tools-tech-description">Matplotlib, Plotly</p>
        </div>
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

    # Call to Action
    st.markdown("""
        <div class="section-header">
            <div class="section-title">📍 Ready to Dive In?</div>
        </div>
    """, unsafe_allow_html=True)

    # Navigation Instructions
    st.markdown('<h2 class="sub-header">🚀 Get Started</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="navigation-instructions">
        <h3>Navigation Instructions</h3>
        <p>Use the sidebar navigation to access different sections of the application:</p>
        <ul>
            <li><strong>Data Exploration</strong> - Discover insights in your loan dataset</li>
            <li><strong>Data Preprocessing</strong> - Clean and prepare data for modeling</li>
            <li><strong>Feature Selection</strong> - Identify the most important predictive features</li>
            <li><strong>Model Training</strong> - Train and compare multiple ML algorithms</li>
            <li><strong>Live Prediction</strong> - Make real-time loan default predictions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>Applied Regression & Machine Learning Project</strong></p>
        <p>UGBS | Academic Year 2025-2026 | Built with ❤️ using Streamlit and Python</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
