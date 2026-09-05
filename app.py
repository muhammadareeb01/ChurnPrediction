import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import time
import sys
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# -----------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION & REACT / NEXT.JS MODERN SAAS CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="E-Commerce | AI Churn Intelligence & Decision Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CDN FontAwesome & Inter/Plus Jakarta Sans Modern Design System
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Typography & Reset */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Core Streamlit Overrides */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #111827 0%, #030712 100%) !important;
        color: #F8FAFC !important;
    }
    
    /* Hide top white header */
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    .stApp > header {
        background-color: transparent !important;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 1440px;
    }

    /* Animations */
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    @keyframes slideUp { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }

    /* ---------------------------------------------------
       PREMIUM HEADER (.saas-header)
    --------------------------------------------------- */
    .saas-header {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.5rem 2rem;
        color: white;
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        animation: fadeIn 0.8s ease-out, slideUp 0.6s ease-out;
    }
    .saas-title {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        background: linear-gradient(to right, #F8FAFC, #94A3B8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .saas-subtitle {
        font-size: 0.9rem;
        color: #94A3B8;
        margin-top: 0.4rem;
        font-weight: 400;
    }

    /* ---------------------------------------------------
       METRIC CARDS (.metric-card)
    --------------------------------------------------- */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.25rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -2px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.05);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: fadeIn 0.6s ease-out, slideUp 0.5s ease-out;
        animation-fill-mode: both;
    }
    .metric-card:nth-child(1) { animation-delay: 0.05s; }
    .metric-card:nth-child(2) { animation-delay: 0.1s; }
    .metric-card:nth-child(3) { animation-delay: 0.15s; }
    .metric-card:nth-child(4) { animation-delay: 0.2s; }
    .metric-card:nth-child(5) { animation-delay: 0.25s; }
    
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 20px 25px -5px rgba(0,0,0,0.3), 0 8px 10px -6px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.1);
        border-color: rgba(255,255,255,0.15);
        background: rgba(30, 41, 59, 0.7);
    }
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        opacity: 0.8;
    }
    .metric-blue::before { background: linear-gradient(90deg, #3B82F6, #06B6D4); }
    .metric-rose::before { background: linear-gradient(90deg, #F43F5E, #E11D48); }
    .metric-emerald::before { background: linear-gradient(90deg, #10B981, #059669); }
    .metric-amber::before { background: linear-gradient(90deg, #F59E0B, #D97706); }
    .metric-indigo::before { background: linear-gradient(90deg, #6366F1, #8B5CF6); }

    .metric-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        font-weight: 600;
        color: #94A3B8;
        letter-spacing: 0.05em;
    }
    .metric-icon {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
    }
    .icon-blue { background: rgba(59, 130, 246, 0.15); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.2); }
    .icon-rose { background: rgba(244, 63, 94, 0.15); color: #FB7185; border: 1px solid rgba(244, 63, 94, 0.2); }
    .icon-emerald { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.2); }
    .icon-amber { background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.2); }
    .icon-indigo { background: rgba(99, 102, 241, 0.15); color: #818CF8; border: 1px solid rgba(99, 102, 241, 0.2); }

    .metric-num {
        font-size: 1.8rem;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.2;
        margin-bottom: 0.25rem;
    }
    .metric-detail {
        font-size: 0.8rem;
        color: #64748B;
        font-weight: 500;
    }

    /* ---------------------------------------------------
       GLASS CARDS (.react-card)
    --------------------------------------------------- */
    .react-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
        animation: fadeIn 0.8s ease-out, slideUp 0.7s ease-out;
        transition: all 0.3s ease;
    }
    .react-card:hover {
        border-color: rgba(255, 255, 255, 0.1);
        background: rgba(30, 41, 59, 0.6);
    }
    .react-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #E2E8F0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1.25rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        padding-bottom: 0.75rem;
    }

    /* Insight Chip */
    .chart-insight {
        background: rgba(59, 130, 246, 0.1);
        border-left: 3px solid #3B82F6;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #CBD5E1;
        margin-top: 0;
        margin-bottom: 1.25rem;
        box-shadow: inset 0 0 20px rgba(59,130,246,0.02);
    }

    /* ---------------------------------------------------
       STATUS BADGES
    --------------------------------------------------- */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.6rem;
        border-radius: 8px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    .badge-churn {
        background: rgba(225, 29, 72, 0.15);
        color: #FDA4AF;
        border: 1px solid rgba(225, 29, 72, 0.3);
    }
    .badge-safe {
        background: rgba(16, 185, 129, 0.15);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    .pulse-red { background: #EF4444; box-shadow: 0 0 10px #EF4444, 0 0 0 3px rgba(239,68,68,0.2); }
    .pulse-green { background: #10B981; box-shadow: 0 0 10px #10B981, 0 0 0 3px rgba(16,185,129,0.2); }

    /* ---------------------------------------------------
       TABLES (.saas-table)
    --------------------------------------------------- */
    .saas-table-container {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        overflow: hidden;
        background: rgba(15, 23, 42, 0.5);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
        animation: fadeIn 0.7s ease-out;
    }
    .saas-table {
        width: 100%;
        border-collapse: collapse;
        text-align: left;
        font-size: 0.85rem;
    }
    .saas-table th {
        background: rgba(30, 41, 59, 0.8);
        color: #94A3B8;
        font-weight: 600;
        padding: 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.75rem;
    }
    .saas-table td {
        padding: 0.85rem 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.03);
        color: #E2E8F0;
    }
    .saas-table tr:hover td {
        background: rgba(59, 130, 246, 0.05);
    }

    /* ---------------------------------------------------
       SIDEBAR OVERRIDES
    --------------------------------------------------- */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.85) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 0.5rem !important;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.5rem 0.5rem 1rem 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
    }
    .sidebar-brand-icon {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.25rem;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    .sidebar-brand span {
        color: #F8FAFC !important;
        font-weight: 800;
        font-size: 1.1rem;
        letter-spacing: -0.02em;
    }
    .sidebar-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #10B981;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.3rem 0.75rem;
        border-radius: 20px;
        margin-bottom: 1rem;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.1);
    }

    /* Style Streamlit Navigation Radio Buttons inside Sidebar */
    [data-testid="stSidebar"] .stRadio > div { gap: 0.5rem; }
    [data-testid="stSidebar"] .stRadio label {
        background: transparent !important;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255, 255, 255, 0.03) !important;
        border-color: rgba(255,255,255,0.05);
    }
    [data-testid="stSidebar"] .stRadio div[data-baseweb="radio"] div {
        color: #94A3B8 !important;
        font-weight: 500 !important;
        font-size: 0.95rem;
    }
    /* Hide the actual radio circle */
    [data-testid="stSidebar"] .stRadio div[data-baseweb="radio"] > div:first-child {
        display: none;
    }

    /* ---------------------------------------------------
       STREAMLIT DEFAULT COMPONENT OVERRIDES
    --------------------------------------------------- */
    
    /* Input Fields (Selectbox, TextInput, FileUploader) */
    .stSelectbox > div > div, 
    .stMultiSelect > div > div, 
    .stTextInput > div > div, 
    .stNumberInput > div > div,
    .stFileUploader {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2) !important;
    }
    .stSelectbox > div > div:focus-within, 
    .stTextInput > div > div:focus-within {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.3) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 1.25rem !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.4), inset 0 1px 0 rgba(255,255,255,0.2) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 12px -2px rgba(37, 99, 235, 0.5), inset 0 1px 0 rgba(255,255,255,0.2) !important;
        background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 100%) !important;
    }
    
    .stDownloadButton button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.4) !important;
    }
    .stDownloadButton button:hover {
        background: linear-gradient(135deg, #34D399 0%, #10B981 100%) !important;
        box-shadow: 0 8px 12px -2px rgba(16, 185, 129, 0.5) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(15, 23, 42, 0.4);
        padding: 0.5rem;
        border-radius: 12px;
        border-bottom: none !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.25rem;
        font-weight: 600;
        font-size: 0.9rem;
        color: #94A3B8;
        border: 1px solid transparent !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(30, 41, 59, 0.8) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.4) !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
    }
    
    /* Streamlit Metric Overrides */
    [data-testid="stMetricValue"] { color: #F8FAFC !important; font-weight: 800 !important; }
    [data-testid="stMetricLabel"] { color: #94A3B8 !important; font-weight: 600 !important; }

    /* Tooltip / Warning / Info */
    .stAlert {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        backdrop-filter: blur(10px) !important;
        color: #E2E8F0 !important;
        border-radius: 12px !important;
    }

    /* Badge Primary (for Feature Dictionary etc) */
    .badge-primary {
        background: rgba(59, 130, 246, 0.12);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.25);
    }

    /* Streamlit Dataframe Dark Mode Override */
    .stDataFrame, [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stDataFrame"] > div {
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
    }

    /* Progress Bar Override */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3B82F6 0%, #06B6D4 100%) !important;
        border-radius: 10px !important;
    }
    .stProgress > div > div {
        background: rgba(30, 41, 59, 0.6) !important;
        border-radius: 10px !important;
    }

    /* Caption / Small text */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #64748B !important;
    }

    /* Plotly Chart Container Styling */
    .js-plotly-plot, .plotly {
        border-radius: 12px;
    }

    /* Markdown text paragraphs */
    .stMarkdown p {
        color: #CBD5E1;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #F8FAFC !important;
    }
    .stMarkdown strong, .stMarkdown b {
        color: #E2E8F0;
    }
    .stMarkdown code {
        background: rgba(59, 130, 246, 0.1) !important;
        color: #60A5FA !important;
        padding: 0.15rem 0.4rem !important;
        border-radius: 6px !important;
        font-size: 0.85em !important;
    }

    /* Horizontal Rule */
    .stMarkdown hr {
        border-color: rgba(255,255,255,0.08) !important;
    }

    /* Text Input Placeholder */
    .stTextInput input::placeholder {
        color: #64748B !important;
    }

    /* Expander Content */
    .streamlit-expanderContent {
        background: rgba(15, 23, 42, 0.4) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
    }

    /* Streamlit Table (st.table, st.dataframe markdown tables) */
    .stMarkdown table {
        background: rgba(30, 41, 59, 0.4);
        border-collapse: collapse;
        border-radius: 10px;
        overflow: hidden;
    }
    .stMarkdown table th {
        background: rgba(30, 41, 59, 0.8) !important;
        color: #94A3B8 !important;
        border-bottom: 1px solid rgba(255,255,255,0.08) !important;
        padding: 0.75rem 1rem !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        font-size: 0.8rem;
    }
    .stMarkdown table td {
        color: #E2E8F0 !important;
        border-bottom: 1px solid rgba(255,255,255,0.03) !important;
        padding: 0.6rem 1rem !important;
    }
    .stMarkdown table tr:hover td {
        background: rgba(59, 130, 246, 0.05) !important;
    }

    /* Scrollbar Styling */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.5); }
    ::-webkit-scrollbar-thumb { background: rgba(148, 163, 184, 0.2); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(148, 163, 184, 0.4); }

    /* Add Icon Hover Animations */
    .fa-solid:hover {
        transform: scale(1.1);
        transition: transform 0.2s ease;
    }
    .metric-icon:hover .fa-solid {
        animation: fa-beat 1s infinite;
    }
    .sidebar-brand-icon .fa-solid {
        animation: fa-bounce 2s infinite;
    }
    
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA & MODEL LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_app_model():
    if os.path.exists("final_model.sav"):
        try:
            return pickle.load(open("final_model.sav", "rb"))
        except Exception:
            return None
    return None

@st.cache_data
def process_uploaded_file(uploaded_file, _model):
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        # Median imputation for continuous
        candidate_continuous = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'COD_Amount']
        for col in candidate_continuous:
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median(numeric_only=True))
                
        # Clean string formatting
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].astype(str).str.strip()
            
        # Engineer RFM features
        try:
            from src.preprocess import engineer_rfm_features
            df = engineer_rfm_features(df)
        except Exception:
            pass
            
        # Generate predictions
        if _model is not None:
            model_features = getattr(_model, 'feature_names_in_', None) or getattr(_model, 'feature_names', None)
            if model_features is not None:
                X = df.copy()
                for mf in model_features:
                    if mf not in X.columns:
                        X[mf] = 'Unknown' if mf == 'ProductName' else 0
                    else:
                        if mf != 'ProductName':
                            X[mf] = pd.to_numeric(X[mf], errors='coerce').fillna(0)
                        else:
                            X[mf] = X[mf].astype(str)
                X = X[model_features]
                for col in X.select_dtypes(include=['object']).columns:
                    X[col] = X[col].astype(str)
                
                df['Churn AI ML'] = _model.predict(X)
                if hasattr(_model, 'predict_proba'):
                    try:
                        df['Churn_Probability'] = np.round(_model.predict_proba(X)[:, 1] * 100, 1)
                    except Exception:
                        pass
        return df
    except Exception as e:
        st.sidebar.error(f"Error processing file: {e}")
        return pd.DataFrame()

model = load_app_model()

@st.cache_data
def load_kaggle_training_data():
    """Load the Kaggle E-Commerce Churn dataset used for model training."""
    default_path = 'E Commerce Dataset Updated.xlsx'
    try:
        if os.path.exists(default_path):
            xf = pd.ExcelFile(default_path)
            sheet = 'E Comm' if 'E Comm' in xf.sheet_names else xf.sheet_names[0]
            return pd.read_excel(default_path, sheet_name=sheet)
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

df_kaggle = load_kaggle_training_data()

# -----------------------------------------------------------------------------
# 3. SAAS SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">
            <i class="fa-solid fa-cart-shopping"></i>
        </div>
        <div>
            <div style="font-weight: 800; font-size: 1.05rem; color: #F8FAFC; letter-spacing: -0.01em;">E-Commerce Churn AI</div>
            <div style="font-size: 0.7rem; color: #94A3B8; font-weight: 500;">Decision Support Platform</div>
        </div>
    </div>
    <div class="sidebar-status-pill">
        <span style="width:6px;height:6px;background:#10B981;border-radius:50%;display:inline-block;"></span>
        EBM Inference Engine: Online
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='font-size: 0.72rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.6rem;'><i class='fa-solid fa-cloud-arrow-up' style='margin-right: 4px;'></i> Upload Test Dataset</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"], label_visibility="collapsed")
    df_master = process_uploaded_file(uploaded_file, model)
    
    if len(df_master) == 0:
        st.sidebar.warning("⚠️ Please upload a dataset to view dashboard insights.")
    else:
        st.markdown(f"""
        <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 10px; padding: 0.6rem 0.75rem; margin-bottom: 0.8rem;">
            <div style="font-size: 0.7rem; font-weight: 700; color: #38BDF8; margin-bottom: 4px;"><i class="fa-solid fa-database" style="margin-right: 4px;"></i> UPLOADED DATASET</div>
            <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 2px;">📊 <b>{len(df_master):,}</b> rows × <b>{len(df_master.columns)}</b> columns</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Always show Kaggle training data info
    if not df_kaggle.empty:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 10px; padding: 0.6rem 0.75rem; margin-bottom: 0.8rem;">
            <div style="font-size: 0.7rem; font-weight: 700; color: #34D399; margin-bottom: 4px;"><i class="fa-solid fa-graduation-cap" style="margin-right: 4px;"></i> TRAINING DATA (Kaggle)</div>
            <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 2px;">📊 <b>{len(df_kaggle):,}</b> rows × <b>{len(df_kaggle.columns)}</b> columns</div>
        </div>
        """, unsafe_allow_html=True)
    
    navigation = st.radio(
        "NAVIGATION",
        options=[
            "📊 Executive AI Dashboard",
            "🎯 Retargeting Action Center",
            "🤖 AI Model Performance",
            "📖 Project Methodology & Architecture",
            "📁 Complete Dataset Explorer",
            "⚙️ Data Hub & AI Retraining"
        ],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("<div style='border-top: 1px solid #1E293B; margin: 1.2rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.72rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.6rem;'><i class='fa-solid fa-filter' style='margin-right: 4px;'></i> Data Filters</div>", unsafe_allow_html=True)
    
    city_options = sorted(df_master['CityTier'].dropna().unique().tolist()) if 'CityTier' in df_master.columns else []
    sel_cities = st.multiselect("City Tier", options=city_options, default=city_options)
    
    prod_options = sorted(df_master['ProductName'].dropna().unique().tolist()) if 'ProductName' in df_master.columns else []
    sel_prods = st.multiselect("Product Category", options=prod_options, default=[])
    
    risk_filter = st.selectbox("Customer Risk Segment", ["All Customers", "🚨 At-Risk Churners Only", "🛡️ Retained/Safe Only"])

    st.markdown("<div style='border-top: 1px solid #1E293B; margin: 1.2rem 0;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid #334155; border-radius: 10px; padding: 0.75rem; font-size: 0.74rem; color: #94A3B8;">
        <div style="color: #F8FAFC; font-weight: 700; margin-bottom: 4px;"><i class="fa-solid fa-graduation-cap" style="color: #38BDF8; margin-right: 4px;"></i> MUET Master Thesis</div>
        <div>Author: <b>M. Tehmas (25MEIT003)</b></div>
        <div>Model: <b>Explainable Boosting (EBM)</b></div>
    </div>
    """, unsafe_allow_html=True)

# Apply global filters
df_filtered = df_master.copy()
if sel_cities and 'CityTier' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['CityTier'].isin(sel_cities)]
if sel_prods and 'ProductName' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['ProductName'].isin(sel_prods)]

pred_col = 'Churn AI ML' if 'Churn AI ML' in df_filtered.columns else ('Churn' if 'Churn' in df_filtered.columns else None)

if risk_filter == "🚨 At-Risk Churners Only" and pred_col:
    df_filtered = df_filtered[df_filtered[pred_col] == 1]
elif risk_filter == "🛡️ Retained/Safe Only" and pred_col:
    df_filtered = df_filtered[df_filtered[pred_col] == 0]

# -----------------------------------------------------------------------------
# 4. PAGE 1: EXECUTIVE AI DASHBOARD
# -----------------------------------------------------------------------------
if navigation == "📊 Executive AI Dashboard":
    st.markdown("""
    <div class="saas-header">
        <div>
            <div class="saas-title"><i class="fa-solid fa-chart-line" style="color:#38BDF8;"></i> Executive AI & BI Analytics</div>
            <div class="saas-subtitle">Real-time Glassbox Machine Learning predictions detecting customer departure and revenue risk.</div>
        </div>
        <div style="text-align: right;">
            <span class="badge badge-primary"><i class="fa-solid fa-shield-halved"></i> Active Glassbox</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    total_eval = len(df_filtered)
    if pred_col:
        total_churn = int((df_filtered[pred_col] == 1).sum())
        churn_rate = round((total_churn / total_eval * 100), 2) if total_eval > 0 else 0
        retention_rate = round(100 - churn_rate, 2)
    else:
        total_churn = 0
        churn_rate = 0.0
        retention_rate = 100.0

    at_risk_cod = df_filtered[df_filtered[pred_col] == 1]['COD_Amount'].sum() if ('COD_Amount' in df_filtered.columns and pred_col) else 0
    complain_count = int(df_filtered['Complain'].sum()) if 'Complain' in df_filtered.columns else 0

    # 3D SaaS Metric Cards
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card metric-blue">
            <div class="metric-header">
                <span class="metric-label">Evaluated Users</span>
                <div class="metric-icon icon-blue"><i class="fa-solid fa-users"></i></div>
            </div>
            <div class="metric-num">{total_eval:,}</div>
            <div class="metric-detail">Active customer base</div>
        </div>
        <div class="metric-card metric-rose">
            <div class="metric-header">
                <span class="metric-label">At-Risk Churners</span>
                <div class="metric-icon icon-rose"><i class="fa-solid fa-user-xmark"></i></div>
            </div>
            <div class="metric-num">{total_churn:,}</div>
            <div class="metric-detail">{churn_rate}% Predicted Churn Rate</div>
        </div>
        <div class="metric-card metric-emerald">
            <div class="metric-header">
                <span class="metric-label">Retention Outlook</span>
                <div class="metric-icon icon-emerald"><i class="fa-solid fa-user-check"></i></div>
            </div>
            <div class="metric-num">{retention_rate}%</div>
            <div class="metric-detail">Safe customer ratio</div>
        </div>
        <div class="metric-card metric-amber">
            <div class="metric-header">
                <span class="metric-label">At-Risk COD Volume</span>
                <div class="metric-icon icon-amber"><i class="fa-solid fa-money-bill-wave"></i></div>
            </div>
            <div class="metric-num">PKR {at_risk_cod:,.0f}</div>
            <div class="metric-detail">Total financial exposure</div>
        </div>
        <div class="metric-card metric-indigo">
            <div class="metric-header">
                <span class="metric-label">Complaints Filed</span>
                <div class="metric-icon icon-indigo"><i class="fa-solid fa-triangle-exclamation"></i></div>
            </div>
            <div class="metric-num">{complain_count:,}</div>
            <div class="metric-detail">Direct dissatisfaction signal</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        if 'ProductName' in df_filtered.columns and pred_col:
            prod_summary = df_filtered.groupby('ProductName')[pred_col].agg(Total='count', AtRisk='sum').reset_index()
            prod_summary['Retained'] = prod_summary['Total'] - prod_summary['AtRisk']
            prod_summary = prod_summary.sort_values(by='Total', ascending=True)
            
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(y=prod_summary['ProductName'], x=prod_summary['Retained'], name='Retained (Safe)', orientation='h', marker=dict(color='#10B981')))
            fig1.add_trace(go.Bar(y=prod_summary['ProductName'], x=prod_summary['AtRisk'], name='At-Risk (Churn)', orientation='h', marker=dict(color='#EF4444')))
            fig1.update_layout(
                title="<b>1. Product Retention vs. Churn Breakdown</b>",
                barmode='stack',
                xaxis_title="Number of Customers",
                yaxis_title="",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#94A3B8')),
                height=340,
                margin=dict(l=10, r=10, t=45, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E2E8F0'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("<div class='chart-insight'><i class='fa-solid fa-lightbulb' style='color:#3B82F6;'></i> <b>Insight:</b> Shows customer retention volume across each product. Red bars highlight critical at-risk customers needing immediate retargeting.</div>", unsafe_allow_html=True)
            
        if 'COD_Amount' in df_filtered.columns and pred_col:
            fig3 = px.box(
                df_filtered,
                x=pred_col,
                y='COD_Amount',
                color=pred_col,
                color_discrete_map={0: '#10B981', 1: '#EF4444'},
                labels={pred_col: 'Prediction (0=Safe, 1=Churn)', 'COD_Amount': 'COD Order Value (PKR)'},
                title="<b>3. Order Monetary Value (COD) vs. Churn Risk</b>"
            )
            fig3.update_layout(height=340, showlegend=False, margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#E2E8F0'), xaxis=dict(gridcolor='rgba(255,255,255,0.05)'), yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("<div class='chart-insight'><i class='fa-solid fa-lightbulb' style='color:#3B82F6;'></i> <b>Insight:</b> Box plot compares purchasing power between loyal and churn-risk buyers to prevent high-ticket revenue loss.</div>", unsafe_allow_html=True)

    with col_c2:
        if 'CityTier' in df_filtered.columns and pred_col:
            tier_df = df_filtered.groupby('CityTier')[pred_col].agg(Total='count', ChurnCount='sum').reset_index()
            tier_df['TierLabel'] = [f"City Tier {t}" for t in tier_df['CityTier']]
            
            fig2 = px.pie(tier_df, names='TierLabel', values='ChurnCount', hole=0.45, color_discrete_sequence=['#3B82F6', '#8B5CF6', '#EC4899'], title="<b>2. Churn Risk Share by City Tier</b>")
            fig2.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#E2E8F0'), legend=dict(font=dict(color='#94A3B8')))
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("<div class='chart-insight'><i class='fa-solid fa-lightbulb' style='color:#3B82F6;'></i> <b>Insight:</b> Identifies geographic regions with highest dissatisfaction, pinpointing logistics latency bottlenecks.</div>", unsafe_allow_html=True)
            
        if 'Tenure' in df_filtered.columns and 'DaySinceLastOrder' in df_filtered.columns:
            fig4 = px.scatter(
                df_filtered,
                x='Tenure',
                y='DaySinceLastOrder',
                color=pred_col if pred_col else None,
                color_discrete_map={0: '#10B981', 1: '#EF4444'},
                labels={'Tenure': 'Tenure (Months)', 'DaySinceLastOrder': 'Days Inactive (Recency)'},
                title="<b>4. RFM Recency vs. Tenure Risk Quadrants</b>",
                hover_data=['CustomerName', 'ProductName'] if 'CustomerName' in df_filtered.columns else None
            )
            fig4.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#E2E8F0'), legend=dict(font=dict(color='#94A3B8')), xaxis=dict(gridcolor='rgba(255,255,255,0.05)'), yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("<div class='chart-insight'><i class='fa-solid fa-lightbulb' style='color:#3B82F6;'></i> <b>Insight:</b> Upper-left cluster represents newly registered users who went dormant quickly—prime candidates for re-engagement.</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. PAGE 2: RETARGETING ACTION CENTER
# -----------------------------------------------------------------------------
elif navigation == "🎯 Retargeting Action Center":
    st.markdown("""
    <div class="saas-header">
        <div>
            <div class="saas-title"><i class="fa-solid fa-crosshairs" style="color:#F43F5E;"></i> Retargeting Action Center</div>
            <div class="saas-subtitle">Prescriptive Decision Engine: Converting AI predictions into automated shopkeeper interventions.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tab_list, tab_diag = st.tabs(["📋 Filterable Customer Risk Matrix", "⚡ 1-Click Single Customer Diagnosis & Outreach"])
    
    preferred_cols = [
        'CustomerID', 'CustomerName', 'ProductName', 'COD_Amount', 
        'Tenure', 'DaySinceLastOrder', 'OrderCount', 'Complain', 
        'RFM_Recency', 'Engagement_Score', 'Churn AI ML', 'Churn_Probability'
    ]
    show_cols = [c for c in preferred_cols if c in df_filtered.columns]
    
    with tab_list:
        # Header with Download Button Right-Aligned
        head_col, btn_col = st.columns([3.5, 1])
        with head_col:
            st.markdown("<div style='font-size:1rem; font-weight:700; color:#F8FAFC; margin-top:4px;'><i class='fa-solid fa-table-list' style='color:#3B82F6; margin-right:6px;'></i> Customer Risk Registry</div>", unsafe_allow_html=True)
        with btn_col:
            csv_data = df_filtered[show_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export to CSV",
                data=csv_data,
                file_name="Retargeting_List.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        c_search, c_sort = st.columns([2.5, 1])
        with c_search:
            search_query = st.text_input("Search Customer Name, ID, or Product:", "", label_visibility="collapsed", placeholder="🔎 Search by Customer Name, ID, or Product Category...")
        with c_sort:
            sort_by = st.selectbox("Sort Table By:", ["Highest Churn Probability", "Highest COD Amount", "Most Inactive Days"], label_visibility="collapsed")
            
        display_df = df_filtered.copy()
        if search_query:
            mask = display_df.astype(str).apply(lambda x: x.str.contains(search_query, case=False)).any(axis=1)
            display_df = display_df[mask]
            
        if sort_by == "Highest Churn Probability" and 'Churn_Probability' in display_df.columns:
            display_df = display_df.sort_values(by='Churn_Probability', ascending=False)
        elif sort_by == "Highest COD Amount" and 'COD_Amount' in display_df.columns:
            display_df = display_df.sort_values(by='COD_Amount', ascending=False)
        elif sort_by == "Most Inactive Days" and 'DaySinceLastOrder' in display_df.columns:
            display_df = display_df.sort_values(by='DaySinceLastOrder', ascending=False)

        # Styled HTML Table for Preview
        table_html = "<div class='saas-table-container'><table class='saas-table'><thead><tr>"
        for col_name in show_cols[:7]:
            table_html += f"<th>{col_name}</th>"
        table_html += "<th>Status</th><th>Churn Prob</th></tr></thead><tbody>"
        
        for _, r in display_df.head(25).iterrows():
            is_risk = r.get('Churn AI ML', 0) == 1
            status_badge = '<span class="badge badge-churn"><span class="pulse-dot pulse-red"></span> At Risk</span>' if is_risk else '<span class="badge badge-safe"><span class="pulse-dot pulse-green"></span> Safe</span>'
            prob_val = f"{r.get('Churn_Probability', 0)}%" if 'Churn_Probability' in r else ("High" if is_risk else "Low")
            
            table_html += f"<tr>"
            table_html += f"<td><b>{r.get('CustomerID', 'N/A')}</b></td>"
            table_html += f"<td>{r.get('CustomerName', 'N/A')}</td>"
            table_html += f"<td>{r.get('ProductName', 'N/A')}</td>"
            table_html += f"<td>PKR {r.get('COD_Amount', 0):,}</td>"
            table_html += f"<td>{r.get('Tenure', 0)} mo</td>"
            table_html += f"<td>{r.get('DaySinceLastOrder', 0)} days</td>"
            table_html += f"<td>{r.get('OrderCount', 0)} orders</td>"
            table_html += f"<td>{status_badge}</td>"
            table_html += f"<td><b>{prob_val}</b></td>"
            table_html += f"</tr>"
            
        table_html += "</tbody></table></div>"
        st.markdown(table_html, unsafe_allow_html=True)
        st.caption(f"Displaying top 25 of {len(display_df)} matching records. Use download button above to export full matrix.")

    with tab_diag:
        if len(df_filtered) > 0:
            if 'CustomerName' in df_filtered.columns:
                cust_options = df_filtered.apply(
                    lambda r: f"{r['CustomerID']} | {r['CustomerName']} ({r.get('ProductName', 'N/A')}) - Risk: {r.get('Churn_Probability', 0)}%", 
                    axis=1
                ).tolist()
            else:
                cust_options = df_filtered['CustomerID'].astype(str).tolist()

            selected_str = st.selectbox("Select Customer to Diagnose & Retarget:", options=cust_options)
            selected_cid = selected_str.split(" | ")[0] if " | " in selected_str else selected_str
            
            cust_data = df_master[df_master['CustomerID'] == selected_cid].iloc[0]
            
            c_left, c_right = st.columns([1, 1.3])
            
            with c_left:
                st.markdown("<div class='react-card'>", unsafe_allow_html=True)
                st.markdown("<div class='react-card-title'><i class='fa-solid fa-id-card' style='color:#2563EB;'></i> Customer Snapshot</div>", unsafe_allow_html=True)
                
                is_churner = cust_data.get('Churn AI ML', 0) == 1
                badge_html = '<span class="badge badge-churn"><span class="pulse-dot pulse-red"></span> AT RISK (CHURNING)</span>' if is_churner else '<span class="badge badge-safe"><span class="pulse-dot pulse-green"></span> SAFE (RETAINED)</span>'
                
                st.markdown(f"**Status:** {badge_html}", unsafe_allow_html=True)
                st.markdown(f"**Customer ID:** `{cust_data.get('CustomerID', 'N/A')}`")
                st.markdown(f"**Full Name:** **{cust_data.get('CustomerName', 'Valued Customer')}**")
                st.markdown(f"**Product Category:** {cust_data.get('ProductName', 'N/A')}")
                st.markdown(f"**Total COD Value:** PKR {cust_data.get('COD_Amount', 0):,}")
                st.markdown(f"**Tenure:** {cust_data.get('Tenure', 0)} months")
                st.markdown(f"**Inactivity:** {cust_data.get('DaySinceLastOrder', 0)} days since last order")
                st.markdown(f"**Total Orders:** {cust_data.get('OrderCount', 0)}")
                st.markdown(f"**Complaint History:** {'🚨 Registered Complaint' if cust_data.get('Complain', 0) == 1 else '✅ Clean History'}")
                if 'Churn_Probability' in cust_data:
                    st.progress(float(cust_data['Churn_Probability']) / 100, text=f"AI Churn Probability: {cust_data['Churn_Probability']}%")
                st.markdown("</div>", unsafe_allow_html=True)

            with c_right:
                st.markdown("<div class='react-card'>", unsafe_allow_html=True)
                st.markdown("<div class='react-card-title'><i class='fa-solid fa-wand-magic-sparkles' style='color:#6366F1;'></i> Prescriptive Next Best Action</div>", unsafe_allow_html=True)
                
                has_comp = cust_data.get('Complain', 0) == 1
                inactivity = cust_data.get('DaySinceLastOrder', 0)
                orders = cust_data.get('OrderCount', 0)
                
                if has_comp:
                    st.error("⚠️ **Primary Friction Driver:** Customer registered an unresolved complaint.")
                    strategy_badge = "Priority Apology Email & 20% Discount Voucher"
                    msg_draft = f"Dear {cust_data.get('CustomerName', 'Customer')}, we noticed your recent complaint regarding your order. We sincerely apologize for the inconvenience. Please accept this exclusive 20% discount code [EASYCARE20] for your next purchase."
                elif inactivity > 15:
                    st.warning(f"⚠️ **Primary Friction Driver:** High Dormancy Risk ({inactivity} days inactive).")
                    strategy_badge = "Cart Re-engagement & Free Delivery Incentive"
                    msg_draft = f"Assalam-o-Alaikum {cust_data.get('CustomerName', 'Customer')}! We miss you at our store. Your favorite items in {cust_data.get('ProductName', 'our store')} are waiting. Use code [FREESHIP] today for Free Cash-on-Delivery!"
                elif orders <= 1:
                    st.info("💡 **Primary Friction Driver:** Single-order customer requiring onboarding trust.")
                    strategy_badge = "2nd Order Loyalty Bonus Reward"
                    msg_draft = f"Hi {cust_data.get('CustomerName', 'Customer')}! Thanks for your first purchase. Complete your 2nd order this week and unlock 2x Reward Points + 10% cashback."
                else:
                    st.success("✅ **Primary Driver:** Active regular customer.")
                    strategy_badge = "VIP Exclusive Catalog Preview"
                    msg_draft = f"Dear VIP {cust_data.get('CustomerName', 'Customer')}, explore our newly arrived top trending collection in {cust_data.get('ProductName', 'Fashion & Tech')} tailored for you!"

                st.markdown(f"**Recommended Strategy:** `{strategy_badge}`")
                st.text_area("Generated Outreach Copy (Ready to Dispatch):", value=msg_draft, height=95)
                
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("📲 Simulate WhatsApp / SMS Notification", use_container_width=True):
                        st.success(f"SMS queued for {cust_data.get('CustomerName', 'Customer')}!")
                with b2:
                    if st.button("📧 Dispatch Email Campaign", use_container_width=True):
                        st.success(f"Retention email sent to {cust_data.get('CustomerID', 'Customer')}!")
                st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. PAGE 3: AI MODEL PERFORMANCE
# -----------------------------------------------------------------------------
elif navigation == "🤖 AI Model Performance":
    st.markdown("""
    <div class="saas-header">
        <div>
            <div class="saas-title"><i class="fa-solid fa-brain" style="color:#A78BFA;"></i> AI Model Performance & Evaluation</div>
            <div class="saas-subtitle">Live evaluation metrics computed on held-out test set (400 records) using the trained Explainable Boosting Machine (EBM).</div>
        </div>
        <div style="text-align:right;">
            <span class="badge" style="background:rgba(139,92,246,0.15);color:#A78BFA;border:1px solid rgba(139,92,246,0.3);"><i class="fa-solid fa-flask"></i> Kaggle Test Set</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data
    def compute_model_metrics():
        """Compute classification metrics from saved model + test split."""
        test_X_path = os.path.join('data', 'processed', 'X_test.csv')
        test_y_path = os.path.join('data', 'processed', 'y_test.csv')
        if not (os.path.exists(test_X_path) and os.path.exists(test_y_path) and os.path.exists("final_model.sav")):
            return None
        try:
            X_test_df = pd.read_csv(test_X_path)
            y_test_df = pd.read_csv(test_y_path)
            y_test_vals = y_test_df['Churn'] if 'Churn' in y_test_df.columns else y_test_df.iloc[:, 0]
            clf = pickle.load(open("final_model.sav", "rb"))
            # Align features
            feat_names = getattr(clf, 'feature_names_in_', None) or getattr(clf, 'feature_names', None)
            if feat_names is not None:
                for f in feat_names:
                    if f not in X_test_df.columns:
                        X_test_df[f] = 0
                X_test_df = X_test_df[feat_names]
            for col in X_test_df.select_dtypes(include=['object']).columns:
                X_test_df[col] = X_test_df[col].astype(str)
            y_pred = clf.predict(X_test_df)
            y_prob = clf.predict_proba(X_test_df)[:, 1] if hasattr(clf, 'predict_proba') else None
            acc   = round(accuracy_score(y_test_vals, y_pred) * 100, 2)
            prec  = round(precision_score(y_test_vals, y_pred, zero_division=0) * 100, 2)
            rec   = round(recall_score(y_test_vals, y_pred, zero_division=0) * 100, 2)
            f1    = round(f1_score(y_test_vals, y_pred, zero_division=0) * 100, 2)
            auc   = round(roc_auc_score(y_test_vals, y_prob) * 100, 2) if y_prob is not None else None
            cm    = confusion_matrix(y_test_vals, y_pred).tolist()
            # Feature importance
            feat_imp = None
            if hasattr(clf, 'term_importances_') and feat_names is not None:
                imp_vals = clf.term_importances()
                feat_imp = sorted(zip(feat_names, imp_vals), key=lambda x: x[1], reverse=True)[:15]
            elif hasattr(clf, 'feature_importances_') and feat_names is not None:
                feat_imp = sorted(zip(feat_names, clf.feature_importances_), key=lambda x: x[1], reverse=True)[:15]
            # ROC curve data
            roc_data = None
            if y_prob is not None:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_test_vals, y_prob)
                roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            return {
                "accuracy": acc, "precision": prec, "recall": rec,
                "f1": f1, "auc": auc, "confusion_matrix": cm,
                "feature_importance": feat_imp, "roc_data": roc_data,
                "n_test": len(y_test_vals), "n_features": X_test_df.shape[1]
            }
        except Exception as e:
            return {"error": str(e)}

    metrics = compute_model_metrics()

    if metrics is None:
        st.warning("⚠️ Test data or model not found. Make sure `data/processed/X_test.csv`, `y_test.csv`, and `final_model.sav` exist.")
    elif "error" in metrics:
        st.error(f"❌ Error computing metrics: {metrics['error']}")
    else:
        # ── KPI Metric Cards ──────────────────────────────────────────────
        st.markdown(f"""
        <div class="metric-grid" style="grid-template-columns: repeat(5, 1fr);">
            <div class="metric-card metric-emerald">
                <div class="metric-header">
                    <span class="metric-label">Accuracy</span>
                    <div class="metric-icon icon-emerald"><i class="fa-solid fa-bullseye"></i></div>
                </div>
                <div class="metric-num">{metrics['accuracy']}%</div>
                <div class="metric-detail">Overall correct predictions</div>
            </div>
            <div class="metric-card metric-blue">
                <div class="metric-header">
                    <span class="metric-label">Precision</span>
                    <div class="metric-icon icon-blue"><i class="fa-solid fa-crosshairs"></i></div>
                </div>
                <div class="metric-num">{metrics['precision']}%</div>
                <div class="metric-detail">Of predicted churners, truly churned</div>
            </div>
            <div class="metric-card metric-amber">
                <div class="metric-header">
                    <span class="metric-label">Recall</span>
                    <div class="metric-icon icon-amber"><i class="fa-solid fa-magnet"></i></div>
                </div>
                <div class="metric-num">{metrics['recall']}%</div>
                <div class="metric-detail">Actual churners correctly caught</div>
            </div>
            <div class="metric-card metric-rose">
                <div class="metric-header">
                    <span class="metric-label">F1 Score</span>
                    <div class="metric-icon icon-rose"><i class="fa-solid fa-scale-balanced"></i></div>
                </div>
                <div class="metric-num">{metrics['f1']}%</div>
                <div class="metric-detail">Precision–Recall harmonic mean</div>
            </div>
            <div class="metric-card metric-indigo">
                <div class="metric-header">
                    <span class="metric-label">ROC-AUC</span>
                    <div class="metric-icon icon-indigo"><i class="fa-solid fa-chart-area"></i></div>
                </div>
                <div class="metric-num">{metrics['auc'] if metrics['auc'] else 'N/A'}%</div>
                <div class="metric-detail">Discrimination ability score</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── What each metric means ─────────────────────────────────────────
        with st.expander("📚 What do these metrics mean? (Click to expand)", expanded=False):
            st.markdown("""
            | Metric | What it means in plain Urdu/English |
            |--------|--------------------------------------|
            | **Accuracy** | Out of 100 customers, kitne sahi predict hue (churn ya nahi)? |
            | **Precision** | Jinhe AI ne "Churn" kaha, unme se kitne actually churn hue? (False Alarm rate) |
            | **Recall** | Jo actually churn hue, unme se AI ne kitno ko pakra? (Miss rate) |
            | **F1 Score** | Precision aur Recall ka balance — agar dono high hain tou F1 bhi high hoga |
            | **ROC-AUC** | 100% = perfect model, 50% = coin toss. Humare EBM ka score dekho! |
            """)

        st.markdown("---")
        col_cm, col_roc = st.columns(2)

        # ── Confusion Matrix ───────────────────────────────────────────────
        with col_cm:
            cm = metrics["confusion_matrix"]
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            cm_labels = ["Retained (0)", "Churned (1)"]
            fig_cm = go.Figure(data=go.Heatmap(
                z=[[tn, fp], [fn, tp]],
                x=["Predicted: Safe", "Predicted: Churn"],
                y=["Actual: Safe", "Actual: Churn"],
                text=[[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]],
                texttemplate="<b>%{text}</b>",
                colorscale=[
                    [0.0, "#0F172A"], [0.3, "#1E3A5F"],
                    [0.6, "#2563EB"], [1.0, "#3B82F6"]
                ],
                showscale=False
            ))
            fig_cm.update_layout(
                title="<b>Confusion Matrix (Test Set)</b>",
                height=340,
                margin=dict(l=10, r=10, t=50, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(size=13, color='#E2E8F0')
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            st.markdown(f"<div class='chart-insight'><i class='fa-solid fa-lightbulb' style='color:#3B82F6;'></i> <b>TN={tn}</b> sahi Safe, <b>TP={tp}</b> sahi Churn pakray, <b>FP={fp}</b> ghalat alarm, <b>FN={fn}</b> miss hue churners.</div>", unsafe_allow_html=True)

        # ── ROC Curve ──────────────────────────────────────────────────────
        with col_roc:
            if metrics["roc_data"]:
                fpr = metrics["roc_data"]["fpr"]
                tpr = metrics["roc_data"]["tpr"]
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode='lines', name=f'EBM (AUC={metrics["auc"]}%)',
                    line=dict(color='#6366F1', width=3),
                    fill='tozeroy', fillcolor='rgba(99,102,241,0.1)'
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier',
                    line=dict(color='#94A3B8', dash='dash', width=1.5)
                ))
                fig_roc.update_layout(
                    title="<b>ROC Curve — EBM vs Random Baseline</b>",
                    xaxis_title="False Positive Rate (FPR)",
                    yaxis_title="True Positive Rate (Recall)",
                    height=340,
                    margin=dict(l=10, r=10, t=50, b=10),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(color='#94A3B8')),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#E2E8F0'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.1)')
                )
                st.plotly_chart(fig_roc, use_container_width=True)
                st.markdown("<div class='chart-insight'><i class='fa-solid fa-lightbulb' style='color:#3B82F6;'></i> Curve jitni upper-left corner ke qareeb ho, utna behtar model. Purple = EBM, grey dashed = coin toss baseline.</div>", unsafe_allow_html=True)

        st.markdown("---")

        # ── Feature Importance ─────────────────────────────────────────────
        if metrics["feature_importance"]:
            feat_names_list = [x[0] for x in metrics["feature_importance"]]
            feat_vals_list  = [float(x[1]) for x in metrics["feature_importance"]]
            fig_fi = go.Figure(go.Bar(
                x=feat_vals_list[::-1],
                y=feat_names_list[::-1],
                orientation='h',
                marker=dict(
                    color=feat_vals_list[::-1],
                    colorscale=[[0, '#EDE9FE'], [0.5, '#8B5CF6'], [1, '#4C1D95']],
                    showscale=False
                )
            ))
            fig_fi.update_layout(
                title="<b>Top Feature Importances (EBM Glassbox)</b>",
                xaxis_title="Importance Score",
                yaxis_title="",
                height=420,
                margin=dict(l=10, r=10, t=50, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E2E8F0'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig_fi, use_container_width=True)
            st.markdown("<div class='chart-insight'><i class='fa-solid fa-lightbulb' style='color:#3B82F6;'></i> <b>Insight:</b> Yeh chart dikhata hai AI ka churn prediction karte waqt kaunsa feature sabse zyada matter karta hai. Lambi bar = zyada ehmiyat.</div>", unsafe_allow_html=True)
        else:
            st.info("Feature importance chart EBM model se automatically extract ho raha hai. Agar nazar nahi aa raha tou model retraining karein.")

        # ── Summary Stats Bar ──────────────────────────────────────────────
        st.markdown(f"""
        <div class="react-card" style="background:rgba(15,23,42,0.6); border:1px solid rgba(255,255,255,0.05); margin-top:0.5rem; box-shadow:none;">
            <div style="display:flex; gap:2rem; flex-wrap:wrap; align-items:center;">
                <div style="color:#94A3B8; font-size:0.8rem;"><i class="fa-solid fa-database" style="color:#38BDF8;"></i> <b style="color:#F8FAFC;">Test Records:</b> {metrics['n_test']:,}</div>
                <div style="color:#94A3B8; font-size:0.8rem;"><i class="fa-solid fa-layer-group" style="color:#A78BFA;"></i> <b style="color:#F8FAFC;">Features Used:</b> {metrics['n_features']}</div>
                <div style="color:#94A3B8; font-size:0.8rem;"><i class="fa-solid fa-split" style="color:#34D399;"></i> <b style="color:#F8FAFC;">Split Strategy:</b> 80/20 Stratified</div>
                <div style="color:#94A3B8; font-size:0.8rem;"><i class="fa-solid fa-robot" style="color:#FBBF24;"></i> <b style="color:#F8FAFC;">Algorithm:</b> Explainable Boosting Machine (EBM)</div>
                <div style="color:#94A3B8; font-size:0.8rem;"><i class="fa-solid fa-code-branch" style="color:#F43F5E;"></i> <b style="color:#F8FAFC;">Library:</b> InterpretML (Microsoft)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. PAGE 4: PROJECT METHODOLOGY & ARCHITECTURE
# -----------------------------------------------------------------------------
elif navigation == "📖 Project Methodology & Architecture":
    st.markdown("""
    <div class="saas-header">
        <div>
            <div class="saas-title"><i class="fa-solid fa-book-bookmark" style="color:#38BDF8;"></i> Master Thesis Framework & Methodology</div>
            <div class="saas-subtitle">A User-Centric Business Intelligence Framework for Predictive Customer Churn in E-Commerce (Author: M. Tehmas - 25MEIT003, MUET).</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 1. Problem Statement & Research Objectives")
    st.markdown("""
    E-Commerce SMEs in emerging markets face fierce competition and high customer acquisition costs. Without predictive decision support, shopkeepers only discover customer departure **after** they churn.
    
    **Thesis Contribution:**
    * Design an interpretable **Glassbox AI Engine (Explainable Boosting Machine - EBM)** predicting churn risk beforehand.
    * Deliver a prescriptive **"Next Best Action"** instead of unreadable blackbox model scores.
    """)
    
    st.markdown("---")
    st.markdown("### 📊 2. Data Strategy: Transfer Learning & Domain Adaptation")
    
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        kaggle_rows = f"{len(df_kaggle):,}" if not df_kaggle.empty else "N/A"
        kaggle_cols = len(df_kaggle.columns) if not df_kaggle.empty else 0
        st.markdown(f"""
        <div class='react-card' style='border-top: 3px solid #3B82F6;'>
            <div style='font-weight: 700; color: #F8FAFC; margin-bottom: 8px;'><i class='fa-solid fa-graduation-cap' style='color:#3B82F6;'></i> Phase 1: Training Data (Model Learning)</div>
            <ul style='font-size: 0.85rem; color: #94A3B8;'>
                <li><b>Dataset:</b> <code>E Commerce Dataset Updated.xlsx</code> (Kaggle Benchmark)</li>
                <li><b>Total Records:</b> <code>{kaggle_rows}</code> rows × <code>{kaggle_cols}</code> columns</li>
                <li><b>Purpose:</b> Teaches the Glassbox AI the fundamental global patterns of customer churn.</li>
                <li><b>Algorithm Input:</b> 15+ Features + Target <code>Churn</code> (Ground Truth).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with t_col2:
        if not df_master.empty:
            inference_info = f"<code>{len(df_master):,}</code> Records Uploaded"
        else:
            inference_info = "<span style='color:#F59E0B;'>No dataset uploaded yet</span>"
        st.markdown(f"""
        <div class='react-card' style='border-top: 3px solid #10B981;'>
            <div style='font-weight: 700; color: #F8FAFC; margin-bottom: 8px;'><i class='fa-solid fa-rocket' style='color:#10B981;'></i> Phase 2: Inference Data (Real-world Application)</div>
            <ul style='font-size: 0.85rem; color: #94A3B8;'>
                <li><b>Dataset:</b> User-uploaded test dataset</li>
                <li><b>Volume Executed:</b> {inference_info}</li>
                <li><b>Purpose:</b> Deploying trained intelligence on unseen data.</li>
                <li><b>AI Output:</b> <code>Churn AI ML</code> (Predicted Future Risk)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Feature Dictionary (E-Commerce Churn Data)")
    col_dict_data = [
        {"Column Name": "CustomerID", "Type": "Identifier", "Description": "Unique customer account ID"},
        {"Column Name": "CustomerName", "Type": "Demographic", "Description": "Full name of the shopper"},
        {"Column Name": "ProductName", "Type": "Behavioral", "Description": "Most frequently purchased product category"},
        {"Column Name": "COD_Amount", "Type": "Monetary", "Description": "Total Cash-on-Delivery order amount"},
        {"Column Name": "Tenure", "Type": "Engagement", "Description": "Duration of customer relationship in months"},
        {"Column Name": "DaySinceLastOrder", "Type": "Recency (RFM)", "Description": "Number of days elapsed since the customer's last order"},
        {"Column Name": "OrderCount", "Type": "Frequency (RFM)", "Description": "Total lifetime completed orders"},
        {"Column Name": "Complain", "Type": "Friction Signal", "Description": "Binary flag (1 = customer filed complaint, 0 = no complaint)"},
        {"Column Name": "CityTier", "Type": "Geographic", "Description": "Logistics tier (Tier 1, 2, or 3) representing delivery distance"},
        {"Column Name": "Engagement_Score", "Type": "Engineered RFM", "Description": "Composite metric: Tenure × (OrderCount + 1)"},
        {"Column Name": "Friction_Risk", "Type": "Engineered RFM", "Description": "Composite metric: Complain × (DaySinceLastOrder + 1)"},
        {"Column Name": "Churn AI ML", "Type": "AI Output", "Description": "Predicted label (1 = At-Risk Churner, 0 = Retained)"},
        {"Column Name": "Churn_Probability", "Type": "AI Output", "Description": "Estimated risk score from 0.0% to 100.0%"}
    ]
    
    # Render clean HTML dictionary table
    dict_table_html = "<div class='saas-table-container'><table class='saas-table'><thead><tr><th>Column Name</th><th>Feature Type</th><th>Description & Business Context</th></tr></thead><tbody>"
    for d in col_dict_data:
        dict_table_html += f"<tr><td><code>{d['Column Name']}</code></td><td><span class='badge badge-primary'>{d['Type']}</span></td><td>{d['Description']}</td></tr>"
    dict_table_html += "</tbody></table></div>"
    st.markdown(dict_table_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔬 3. Methodological Pipeline & Engineering Rigor")
    st.markdown("""
    1. **Modular Cleaning & PyTorch Imputation:** Advanced Denoising Autoencoder to learn multi-dimensional continuous distributions with median fallback.
    2. **Stratified Train-Test Splitting (80/20):** Guarantees exact representation of minority churners across both training and testing datasets.
    3. **Glassbox Interpretability vs Blackbox Proof:** EBM builds generalized additive models with pairwise interactions, allowing store managers to inspect exact feature scorecards.
    """)

# -----------------------------------------------------------------------------
# 7. PAGE 4: COMPLETE DATASET EXPLORER
# -----------------------------------------------------------------------------
elif navigation == "📁 Complete Dataset Explorer":
    st.markdown("""
    <div class="saas-header">
        <div>
            <div class="saas-title"><i class="fa-solid fa-database" style="color:#38BDF8;"></i> Complete Dataset Explorer</div>
            <div class="saas-subtitle">Interactive inspection of uploaded and training datasets with column sorting, statistics, and 1-click export.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tab_uploaded, tab_kaggle = st.tabs(["📂 Uploaded Dataset", "🎓 Kaggle Training Data"])
    
    with tab_uploaded:
        if df_filtered.empty:
            st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📭</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;">No Dataset Uploaded</div>
                <div style="font-size: 0.9rem; color: #94A3B8; max-width: 400px; margin: 0 auto;">Upload a CSV or Excel file from the sidebar to explore your dataset here. The AI model will automatically generate churn predictions.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            head_col, btn_col = st.columns([3.8, 1.2])
            with head_col:
                st.markdown(f"<div style='font-size:1rem; font-weight:700; color:#F8FAFC; margin-top:4px;'><i class='fa-solid fa-table' style='color:#3B82F6; margin-right:6px;'></i> Uploaded Data (Showing {len(df_filtered)} rows × {len(df_filtered.columns)} columns)</div>", unsafe_allow_html=True)
            with btn_col:
                csv_full = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Dataset (CSV)",
                    data=csv_full,
                    file_name="Full_Dataset.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            st.dataframe(df_filtered, use_container_width=True, height=480)
            
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            st.markdown("#### 📊 Descriptive Numerical Statistics:")
            try:
                st.dataframe(df_filtered.describe().round(2), use_container_width=True)
            except ValueError:
                st.info("No numerical columns to describe.")
    
    with tab_kaggle:
        if df_kaggle.empty:
            st.warning("Kaggle training dataset not found on server.")
        else:
            st.markdown(f"<div style='font-size:1rem; font-weight:700; color:#F8FAFC; margin-top:4px;'><i class='fa-solid fa-graduation-cap' style='color:#3B82F6; margin-right:6px;'></i> Kaggle E-Commerce Churn Dataset ({len(df_kaggle):,} rows × {len(df_kaggle.columns)} columns)</div>", unsafe_allow_html=True)
            st.markdown("<div class='chart-insight'>This is the benchmark dataset used to <b>train</b> the EBM model. It contains ground-truth <code>Churn</code> labels for supervised learning.</div>", unsafe_allow_html=True)
            st.dataframe(df_kaggle, use_container_width=True, height=480)
            
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            st.markdown("#### 📊 Training Data Statistics:")
            try:
                st.dataframe(df_kaggle.describe().round(2), use_container_width=True)
            except ValueError:
                st.info("No numerical columns to describe.")

# -----------------------------------------------------------------------------
# 8. PAGE 5: DATA HUB & AI RETRAINING
# -----------------------------------------------------------------------------
elif navigation == "⚙️ Data Hub & AI Retraining":
    st.markdown("""
    <div class="saas-header">
        <div>
            <div class="saas-title"><i class="fa-solid fa-cloud-arrow-up" style="color:#10B981;"></i> Data Hub & AI Inference Engine</div>
            <div class="saas-subtitle">Upload new monthly data, merge with the existing dataset, and generate new predictions seamlessly.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_up, col_info = st.columns([1, 1])
    
    with col_up:
        st.markdown("<div class='react-card'>", unsafe_allow_html=True)
        st.markdown("<div class='react-card-title'><i class='fa-solid fa-file-excel' style='color:#10B981;'></i> Upload New Dataset</div>", unsafe_allow_html=True)
        st.markdown("<span style='font-size: 0.8rem; color: #94A3B8;'>Upload the latest `.xlsx` or `.csv` dataset. The system will automatically merge it with the master record and handle duplicates based on CustomerID.</span>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Drag and drop your file here", type=['csv', 'xlsx'], label_visibility="collapsed")
        
        if uploaded_file is not None:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                btn_replace = st.button("🔄 Replace Entire Database", use_container_width=True)
            with col_b2:
                btn_merge = st.button("➕ Merge with Existing Data", type="primary", use_container_width=True)
                
            if btn_replace or btn_merge:
                action_name = "Replacement" if btn_replace else "Merge"
                with st.status(f"Initializing Data {action_name} & AI Predictions...", expanded=True) as status:
                    try:
                        # 1. Read New Data
                        st.write("📂 Reading new uploaded data...")
                        if uploaded_file.name.endswith('.csv'):
                            df_new = pd.read_csv(uploaded_file)
                        else:
                            df_new = pd.read_excel(uploaded_file)
                            
                        master_path = "E Commerce Dataset Updated.xlsx"
                        
                        if btn_replace:
                            st.write("🗑️ Overwriting master database with new data...")
                            final_df = df_new
                        else:
                            # 2. Load Existing Data
                            st.write("💾 Loading master database...")
                            if os.path.exists(master_path):
                                try:
                                    df_existing = pd.read_excel(master_path, sheet_name="E Comm")
                                except Exception:
                                    df_existing = pd.read_excel(master_path)
                            else:
                                df_existing = pd.DataFrame()
                                
                            # 3. Merge and drop duplicates
                            st.write("🔗 Merging and resolving duplicates (keeping latest)...")
                            final_df = pd.concat([df_existing, df_new], ignore_index=True)
                            if 'CustomerID' in final_df.columns:
                                final_df = final_df.drop_duplicates(subset=['CustomerID'], keep='last')
                                
                        # 4. Save Master File
                        st.write("💾 Saving updated master database...")
                        final_df.to_excel(master_path, index=False, sheet_name="EBM_Churn_Data")
                        
                        # 5. Trigger AI Prediction
                        st.write("🧠 Generating new AI Churn Predictions via Kaggle Model...")
                        
                        process = subprocess.Popen([sys.executable, "generate_dashboard_dataset.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        for line in process.stdout:
                            if "[*]" in line or "[+]" in line or "[!]" in line:
                                st.write(f"<i style='color:#64748B; font-size:0.8rem;'>{line.strip()}</i>", unsafe_allow_html=True)
                        process.wait()
                        
                        if process.returncode == 0:
                            status.update(label=f"AI Predictions Generated ({action_name} Complete)!", state="complete", expanded=False)
                            st.success(f"✅ Dataset {action_name.lower()}ed and AI predictions updated successfully! The dashboard is live.")
                            st.balloons()
                            time.sleep(2)
                            st.cache_data.clear() # Clear memory so UI pulls the new CSV!
                            st.rerun()
                        else:
                            status.update(label="Prediction Failed", state="error", expanded=True)
                            st.error(f"Error during AI Prediction: {process.stderr.read()}")
                            
                    except Exception as e:
                        status.update(label="Process Failed", state="error", expanded=True)
                        st.error(f"An error occurred: {str(e)}")
                        
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_info:
        st.markdown("<div class='react-card'>", unsafe_allow_html=True)
        st.markdown("<div class='react-card-title'><i class='fa-solid fa-circle-info' style='color:#3B82F6;'></i> How Incremental Learning Works</div>", unsafe_allow_html=True)
        st.markdown("""
        <ul style="font-size: 0.82rem; color: #94A3B8; line-height: 1.6; padding-left: 1.2rem;">
            <li><b style='color:#CBD5E1;'>1. Upload:</b> Drop your new month's transaction data (CSV or Excel).</li>
            <li><b style='color:#CBD5E1;'>2. Choose Action:</b> Decide whether to <b>Replace</b> the entire database (start fresh) or <b>Merge</b> with existing records.</li>
        <ul style="font-size: 0.85rem; color: #94A3B8; line-height: 1.6; padding-left: 1.2rem;">
            <li><b style='color:#CBD5E1;'>Real-time Merging:</b> When you upload a new file, it detects existing users (by CustomerID) and updates their stats, while appending completely new users.</li>
            <li><b style='color:#CBD5E1;'>Live Prediction:</b> The EBM AI model (trained globally on the Kaggle set) is immediately executed on the merged dataset to predict churn risks.</li>
            <li><b style='color:#CBD5E1;'>Dashboards Update:</b> All insights, metric cards, and charts recalculate instantly.</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Show Current Master Data Info
        if not df_master.empty:
            st.markdown("<div class='react-card'>", unsafe_allow_html=True)
            st.markdown("<div class='react-card-title'><i class='fa-solid fa-server' style='color:#8B5CF6;'></i> Master Database Status</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.2); padding: 1rem; border-radius: 12px; display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <div style="font-size: 0.75rem; color: #8B5CF6; font-weight: 700; text-transform: uppercase;">Current Volume</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: #F8FAFC;">{len(df_master):,}</div>
                </div>
                <i class="fa-solid fa-database" style="font-size: 2.5rem; color: rgba(139, 92, 246, 0.4);"></i>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 9. COMPACT SAAS FOOTER
# -----------------------------------------------------------------------------
st.markdown("<div style='margin-top: 2rem; border-top: 1px solid rgba(255,255,255,0.08); padding: 1rem 0; text-align: center; color: #64748B; font-size: 0.78rem;'><b style='color:#94A3B8;'>E-Commerce Churn AI Decision Support Framework</b> • Master of Engineering in Information Technology Thesis • MUET Jamshoro</div>", unsafe_allow_html=True)
