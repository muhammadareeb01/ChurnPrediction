import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION & REACT / NEXT.JS MODERN SAAS CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EasyBazar | AI Churn Intelligence & Decision Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CDN FontAwesome & Inter/Plus Jakarta Sans Modern Design System
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Streamlit block compacting */
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }

    /* Next.js Top Navigation Header Banner */
    .saas-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.2rem 1.6rem;
        color: white;
        margin-bottom: 1.2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.15);
    }
    .saas-title {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .saas-subtitle {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-top: 0.2rem;
    }

    /* 3D Modern SaaS Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 0.9rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02), 0 1px 2px rgba(0,0,0,0.03);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -5px rgba(15, 23, 42, 0.08);
        border-color: #CBD5E1;
    }
    .metric-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
    }
    .metric-blue::before { background: linear-gradient(90deg, #3B82F6, #60A5FA); }
    .metric-rose::before { background: linear-gradient(90deg, #F43F5E, #FB7185); }
    .metric-emerald::before { background: linear-gradient(90deg, #10B981, #34D399); }
    .metric-amber::before { background: linear-gradient(90deg, #F59E0B, #FBBF24); }
    .metric-indigo::before { background: linear-gradient(90deg, #6366F1, #818CF8); }

    .metric-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.4rem;
    }
    .metric-label {
        font-size: 0.76rem;
        text-transform: uppercase;
        font-weight: 700;
        color: #64748B;
        letter-spacing: 0.04em;
    }
    .metric-icon {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.95rem;
    }
    .icon-blue { background: #EFF6FF; color: #2563EB; }
    .icon-rose { background: #FFF1F2; color: #E11D48; }
    .icon-emerald { background: #ECFDF5; color: #059669; }
    .icon-amber { background: #FFFBEB; color: #D97706; }
    .icon-indigo { background: #EEF2FF; color: #4F46E5; }

    .metric-num {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }
    .metric-detail {
        font-size: 0.76rem;
        color: #64748B;
        font-weight: 500;
    }

    /* React-like Card Container */
    .react-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .react-card-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.8rem;
    }

    /* Chart Explanation Mini-Chip */
    .chart-insight {
        background: #F8FAFC;
        border-left: 3px solid #3B82F6;
        padding: 0.5rem 0.8rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.78rem;
        color: #475569;
        margin-top: -0.3rem;
        margin-bottom: 0.8rem;
    }

    /* Status Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
    }
    .badge-churn {
        background: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FECACA;
    }
    .badge-safe {
        background: #DCFCE7;
        color: #15803D;
        border: 1px solid #BBF7D0;
    }
    .pulse-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        display: inline-block;
    }
    .pulse-red { background: #DC2626; box-shadow: 0 0 0 2px rgba(220, 38, 38, 0.2); }
    .pulse-green { background: #16A34A; box-shadow: 0 0 0 2px rgba(22, 163, 74, 0.2); }

    /* Next.js Styled Table Container */
    .saas-table-container {
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        overflow: hidden;
        background: #FFFFFF;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .saas-table {
        width: 100%;
        border-collapse: collapse;
        text-align: left;
        font-size: 0.82rem;
    }
    .saas-table th {
        background: #F8FAFC;
        color: #475569;
        font-weight: 700;
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #E2E8F0;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        font-size: 0.72rem;
    }
    .saas-table td {
        padding: 0.7rem 1rem;
        border-bottom: 1px solid #F1F5F9;
        color: #1E293B;
    }
    .saas-table tr:last-child td {
        border-bottom: none;
    }
    .saas-table tr:hover td {
        background: #F8FAFC;
    }

    /* Sidebar Dark SaaS Theme */
    [data-testid="stSidebar"] {
        background-color: #0B1120;
        border-right: 1px solid #1E293B;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.5rem 0.2rem 1rem 0.2rem;
        border-bottom: 1px solid #1E293B;
        margin-bottom: 1rem;
    }
    .sidebar-brand-icon {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #38BDF8 0%, #2563EB 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.1rem;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.3);
    }
    .sidebar-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        color: #34D399;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        margin-bottom: 0.8rem;
    }

    /* Sleek Streamlit Widgets Tweaks */
    .stDownloadButton button {
        background: #2563EB !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        padding: 0.4rem 0.9rem !important;
        transition: background 0.15s ease !important;
    }
    .stDownloadButton button:hover {
        background: #1D4ED8 !important;
    }
    
    /* Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 6px 14px;
        font-weight: 600;
        font-size: 0.85rem;
        color: #64748B;
    }
    .stTabs [aria-selected="true"] {
        background: #EFF6FF !important;
        color: #2563EB !important;
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
def load_app_data():
    if os.path.exists("Final_Dashboard_Data.csv"):
        return pd.read_csv("Final_Dashboard_Data.csv")
    elif os.path.exists("EasyBazar_EBM_Dataset.xlsx"):
        return pd.read_excel("EasyBazar_EBM_Dataset.xlsx", sheet_name="EBM_Churn_Data")
    return pd.DataFrame()

model = load_app_model()
df_master = load_app_data()

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
            <div style="font-weight: 800; font-size: 1.05rem; color: #F8FAFC; letter-spacing: -0.01em;">EasyBazar AI</div>
            <div style="font-size: 0.7rem; color: #94A3B8; font-weight: 500;">Decision Support Platform</div>
        </div>
    </div>
    <div class="sidebar-status-pill">
        <span style="width:6px;height:6px;background:#10B981;border-radius:50%;display:inline-block;"></span>
        EBM Inference Engine: Online
    </div>
    """, unsafe_allow_html=True)
    
    navigation = st.radio(
        "NAVIGATION",
        options=[
            "📊 Executive AI Dashboard",
            "🎯 Retargeting Action Center",
            "📖 Project Methodology & Architecture",
            "📁 Complete Dataset Explorer"
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

pred_col = 'AI_Churn_Prediction' if 'AI_Churn_Prediction' in df_filtered.columns else ('Churn' if 'Churn' in df_filtered.columns else None)

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
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=340,
                margin=dict(l=10, r=10, t=45, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
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
            fig3.update_layout(height=340, showlegend=False, margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("<div class='chart-insight'><i class='fa-solid fa-lightbulb' style='color:#3B82F6;'></i> <b>Insight:</b> Box plot compares purchasing power between loyal and churn-risk buyers to prevent high-ticket revenue loss.</div>", unsafe_allow_html=True)

    with col_c2:
        if 'CityTier' in df_filtered.columns and pred_col:
            tier_df = df_filtered.groupby('CityTier')[pred_col].agg(Total='count', ChurnCount='sum').reset_index()
            tier_df['TierLabel'] = [f"City Tier {t}" for t in tier_df['CityTier']]
            
            fig2 = px.pie(tier_df, names='TierLabel', values='ChurnCount', hole=0.45, color_discrete_sequence=['#3B82F6', '#8B5CF6', '#EC4899'], title="<b>2. Churn Risk Share by City Tier</b>")
            fig2.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
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
            fig4.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
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
        'RFM_Recency', 'Engagement_Score', 'AI_Churn_Prediction', 'Churn_Probability'
    ]
    show_cols = [c for c in preferred_cols if c in df_filtered.columns]
    
    with tab_list:
        # Header with Download Button Right-Aligned
        head_col, btn_col = st.columns([3.5, 1])
        with head_col:
            st.markdown("<div style='font-size:1rem; font-weight:700; color:#1E293B; margin-top:4px;'><i class='fa-solid fa-table-list' style='color:#2563EB; margin-right:6px;'></i> Customer Risk Registry</div>", unsafe_allow_html=True)
        with btn_col:
            csv_data = df_filtered[show_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export to CSV",
                data=csv_data,
                file_name="EasyBazar_Retargeting_List.csv",
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
            is_risk = r.get('AI_Churn_Prediction', 0) == 1
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
                
                is_churner = cust_data.get('AI_Churn_Prediction', 0) == 1
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
                    msg_draft = f"Dear {cust_data.get('CustomerName', 'Customer')}, we noticed your recent complaint regarding your order on EasyBazar. We sincerely apologize for the inconvenience. Please accept this exclusive 20% discount code [EASYCARE20] for your next purchase."
                elif inactivity > 15:
                    st.warning(f"⚠️ **Primary Friction Driver:** High Dormancy Risk ({inactivity} days inactive).")
                    strategy_badge = "Cart Re-engagement & Free Delivery Incentive"
                    msg_draft = f"Assalam-o-Alaikum {cust_data.get('CustomerName', 'Customer')}! We miss you on EasyBazar. Your favorite items in {cust_data.get('ProductName', 'our store')} are waiting. Use code [FREESHIP] today for Free Cash-on-Delivery!"
                elif orders <= 1:
                    st.info("💡 **Primary Friction Driver:** Single-order customer requiring onboarding trust.")
                    strategy_badge = "2nd Order Loyalty Bonus Reward"
                    msg_draft = f"Hi {cust_data.get('CustomerName', 'Customer')}! Thanks for your first purchase on EasyBazar. Complete your 2nd order this week and unlock 2x EasyReward Points + 10% cashback."
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
# 6. PAGE 3: PROJECT METHODOLOGY & ARCHITECTURE
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
    E-Commerce SMEs in emerging markets (like EasyBazar) face fierce competition and high customer acquisition costs. Without predictive decision support, shopkeepers only discover customer departure **after** they churn.
    
    **Thesis Contribution:**
    * Design an interpretable **Glassbox AI Engine (Explainable Boosting Machine - EBM)** predicting churn risk beforehand.
    * Deliver a prescriptive **"Next Best Action"** instead of unreadable blackbox model scores.
    """)
    
    st.markdown("---")
    st.markdown("### 📊 2. EasyBazar Dataset Profile & Feature Dictionary")
    
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown(f"""
        * **Dataset Name:** `EasyBazar_EBM_Dataset.xlsx` (Sheet: `EBM_Churn_Data`)
        * **Total Customer Records:** `{len(df_master):,}` rows
        * **Total Features:** `{len(df_master.columns)}` columns
        * **Target Variable:** `Churn` (1 = Churned, 0 = Retained)
        * **Base Churn Rate:** `{round(df_master['Churn'].mean() * 100, 2) if 'Churn' in df_master.columns else 'N/A'}%`
        """)
    with t_col2:
        st.markdown("""
        * **E-Commerce Context:** Cash-on-Delivery (COD), Electronics & Apparel SME
        * **Key Signals:** Inactivity Recency, Complaints, Tenure, Order Count, COD Value
        * **AI Framework:** Microsoft InterpretML (Explainable Boosting Machine)
        """)

    col_dict_data = [
        {"Column Name": "CustomerID", "Type": "Identifier", "Description": "Unique customer account ID (e.g. EZB-10682)"},
        {"Column Name": "CustomerName", "Type": "Demographic", "Description": "Full name of the shopper"},
        {"Column Name": "ProductName", "Type": "Behavioral", "Description": "Most frequently purchased product category"},
        {"Column Name": "COD_Amount", "Type": "Monetary", "Description": "Total Cash-on-Delivery order amount in PKR"},
        {"Column Name": "Tenure", "Type": "Engagement", "Description": "Duration of relationship with EasyBazar in months"},
        {"Column Name": "DaySinceLastOrder", "Type": "Recency (RFM)", "Description": "Number of days elapsed since the customer's last order"},
        {"Column Name": "OrderCount", "Type": "Frequency (RFM)", "Description": "Total lifetime completed orders"},
        {"Column Name": "Complain", "Type": "Friction Signal", "Description": "Binary flag (1 = customer filed complaint, 0 = no complaint)"},
        {"Column Name": "CityTier", "Type": "Geographic", "Description": "Logistics tier (Tier 1, 2, or 3) representing delivery distance"},
        {"Column Name": "Engagement_Score", "Type": "Engineered RFM", "Description": "Composite metric: Tenure × (OrderCount + 1)"},
        {"Column Name": "Friction_Risk", "Type": "Engineered RFM", "Description": "Composite metric: Complain × (DaySinceLastOrder + 1)"},
        {"Column Name": "AI_Churn_Prediction", "Type": "AI Output", "Description": "Predicted label (1 = At-Risk Churner, 0 = Retained)"},
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
            <div class="saas-title"><i class="fa-solid fa-database" style="color:#38BDF8;"></i> EasyBazar Complete Dataset Explorer</div>
            <div class="saas-subtitle">Interactive inspection of all 2,000 records with column sorting, statistics, and 1-click export.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Download Button directly aligned to the right of the table heading
    head_col, btn_col = st.columns([3.8, 1.2])
    with head_col:
        st.markdown(f"<div style='font-size:1rem; font-weight:700; color:#1E293B; margin-top:4px;'><i class='fa-solid fa-table' style='color:#2563EB; margin-right:6px;'></i> Master Registry (Showing {len(df_filtered)} rows × {len(df_filtered.columns)} columns)</div>", unsafe_allow_html=True)
    with btn_col:
        csv_full = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Dataset (CSV)",
            data=csv_full,
            file_name="EasyBazar_Full_Dataset.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    st.dataframe(df_filtered, use_container_width=True, height=480)
    
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("#### 📊 Descriptive Numerical Statistics:")
    st.dataframe(df_filtered.describe().round(2), use_container_width=True)

# -----------------------------------------------------------------------------
# 8. COMPACT SAAS FOOTER
# -----------------------------------------------------------------------------
st.markdown("<div style='margin-top: 2rem; border-top: 1px solid #E2E8F0; padding: 1rem 0; text-align: center; color: #64748B; font-size: 0.78rem;'><b>EasyBazar AI Decision Support Framework</b> • Master of Engineering in Information Technology Thesis • MUET Jamshoro</div>", unsafe_allow_html=True)
