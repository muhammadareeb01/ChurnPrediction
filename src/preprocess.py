"""
Phase 1: Data Preprocessing & Exploratory Analysis
Master Thesis: A User-Centric Business Intelligence Framework for Predictive Customer Churn in E-Commerce
Author: Muhammad Tehmas (25MEIT003) - MUET
Supervisor: Prof. Dr. Shahnawaz Talpur | Co-Supervisor: Engr. Madeha Memon

This module performs modular data cleaning, missing value imputation, and stratified train-test splitting.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add current and src dir to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath('.'))

def engineer_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    AS&RB Requirement: Engineers Direct/Indirect RFM Signals for the EBM Glassbox.
    Transforms raw interactions into 'Recency, Frequency' and an 'Engagement Score'.
    Works dynamically for both Kaggle and upcoming Local SME datasets.
    """
    print("[*] Engineering User-Centric RFM Features (Direct & Indirect signals)...")
    df_feat = df.copy()
    
    # 1. Recency & Frequency Extraction
    if 'DaySinceLastOrder' in df_feat.columns:
        df_feat['RFM_Recency'] = df_feat['DaySinceLastOrder']
    if 'OrderCount' in df_feat.columns:
        df_feat['RFM_Frequency'] = df_feat['OrderCount']
        
    # 2. Indirect Signal: Engagement Score (Tenure combined with App usage or Orders)
    if 'Tenure' in df_feat.columns and 'OrderCount' in df_feat.columns:
        df_feat['Engagement_Score'] = df_feat['Tenure'] * (df_feat['OrderCount'] + 1)
        
    # 3. Friction Risk Signal
    if 'Complain' in df_feat.columns and 'DaySinceLastOrder' in df_feat.columns:
        df_feat['Friction_Risk'] = df_feat['Complain'] * (df_feat['DaySinceLastOrder'] + 1)
        
    return df_feat

def load_data(file_path: str = None) -> pd.DataFrame:
    """Loads the raw Kaggle e-commerce dataset for Training."""
    if file_path is None:
        file_path = 'data_ecommerce_customer_churn.csv'

    print(f"[*] Loading training dataset from: {file_path} ...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    df = pd.read_csv(file_path)
    print(f"[+] Dataset loaded successfully! Initial Shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df

def analyze_missing_values(df: pd.DataFrame) -> dict:
    """Analyzes and returns missing value counts and percentages."""
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    missing_stats = {}
    for col, count in missing_cols.items():
        pct = (count / len(df)) * 100
        missing_stats[col] = {"count": int(count), "percentage": round(pct, 2)}
    return missing_stats

def clean_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans data and imputes missing values.
    
    Decision Rationale (Why Median Imputation?):
    In e-commerce customer behavior metrics (Tenure, OrderCount, Distance, DaySinceLastOrder),
    distributions are typically right-skewed due to outliers (heavy buyers or extreme distances).
    Using Mean imputation would distort the data by pulling values towards extreme outliers.
    Median imputation captures the true central tendency (50th percentile) of a typical customer.
    
    Decision Rationale (Why No One-Hot Encoding?):
    We intend to use Explainable Boosting Machines (EBM) in Phase 2. EBM natively handles categorical
    strings and produces human-readable rules (e.g., 'PreferredLoginDevice == Mobile').
    One-Hot Encoding would fragment features into binary dummy columns, destroying interpretability
    for our SME Business Intelligence dashboard.
    """
    df_clean = df.copy()
    
    # Check for duplicates by CustomerID if present
    if 'CustomerID' in df_clean.columns:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['CustomerID'])
        if len(df_clean) < initial_rows:
            print(f"[*] Removed {initial_rows - len(df_clean)} duplicate CustomerIDs.")

    # Identify continuous columns with missing values
    candidate_continuous = [
        'Tenure', 'WarehouseToHome', 'HourSpendOnApp', 
        'OrderAmountHikeFromlastYear', 'CouponUsed', 
        'OrderCount', 'DaySinceLastOrder', 'COD_Amount'
    ]
    continuous_cols = [c for c in candidate_continuous if c in df_clean.columns]
    
    print("[*] Performing Advanced PyTorch Imputation for continuous variables...")
    imputation_log = {}
    
    # Try importing PyTorch and running Advanced Imputation (Glassbox Compliant)
    try:
        try:
            from pytorch_imputer import pytorch_impute
        except ImportError:
            from src.pytorch_imputer import pytorch_impute
        df_clean, imputation_log = pytorch_impute(df_clean, continuous_cols)
    except Exception as e:
        print(f"[!] PyTorch Imputation Failed ({e}). Falling back to simple Median Imputation...")
        for col in continuous_cols:
            if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
                median_val = float(df_clean[col].median())
                missing_before = int(df_clean[col].isnull().sum())
                df_clean[col] = df_clean[col].fillna(median_val)
                imputation_log[col] = {"imputed_value": median_val, "missing_filled": missing_before}
                print(f"    -> {col}: Filled {missing_before} missing values with Median ({median_val})")
            
    # Clean string formatting in categorical columns (strip trailing whitespaces)
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        
    # Check if any missing values remain
    remaining_missing = int(df_clean.isnull().sum().sum())
    print(f"[+] Cleaning complete! Remaining missing values across entire dataset: {remaining_missing}")
    
    return df_clean, imputation_log

def perform_stratified_split(df: pd.DataFrame, target_col: str = 'Churn', test_size: float = 0.20, random_state: int = 42):
    """
    Splits data into Train and Test sets using Stratification.
    """
    print(f"[*] Splitting dataset into Train ({int((1-test_size)*100)}%) and Test ({int(test_size*100)}%) with Stratification on '{target_col}'...")
    
    # Drop identifier and metadata columns for feature matrix
    drop_candidates = [target_col, 'CustomerID', 'CustomerName', 'OrderDate', 'ShipmentID']
    drop_cols = [c for c in drop_candidates if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    train_churn_rate = round(y_train.mean() * 100, 2)
    test_churn_rate = round(y_test.mean() * 100, 2)
    
    print(f"[+] Split successful!")
    print(f"    -> X_train shape: {X_train.shape} | Churn Rate: {train_churn_rate}%")
    print(f"    -> X_test shape : {X_test.shape}  | Churn Rate: {test_churn_rate}%")
    
    return X_train, X_test, y_train, y_test

def main():
    processed_dir = os.path.join('data', 'processed')
    outputs_dir = 'outputs'
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 1. Load Data (EasyBazar dataset prioritized)
    df_raw = load_data()
    
    # 2. Analyze Missing Values before cleaning
    missing_stats = analyze_missing_values(df_raw)
    
    # 3. Clean & Impute
    df_clean, imputation_log = clean_and_impute(df_raw)
    
    # 3.5 Feature Engineering (RFM Metrics)
    df_engineered = engineer_rfm_features(df_clean)
    
    # Save complete engineered dataset
    clean_csv_path = os.path.join(processed_dir, 'cleaned_full_dataset.csv')
    df_engineered.to_csv(clean_csv_path, index=False)
    print(f"[+] Saved engineered full dataset to: {clean_csv_path}")
    
    # 4. Stratified Split
    X_train, X_test, y_train, y_test = perform_stratified_split(df_engineered)
    
    # Save split datasets
    X_train.to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(processed_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)
    print(f"[+] Saved X_train, X_test, y_train, y_test to: {processed_dir}")
    
    # 5. Generate and Save Summary Report JSON for PDF generation
    summary_report = {
        "project_title": "A User-Centric Business Intelligence Framework for Predictive Customer Churn in E-Commerce",
        "student": "Muhammad Tehmas (25MEIT003)",
        "university": "Mehran University of Engineering and Technology (MUET)",
        "supervisors": "Prof. Dr. Shahnawaz Talpur | Co-Supervisor: Engr. Madeha Memon",
        "dataset_name": "Kaggle E-Commerce Customer Churn Dataset (Full Train)",
        "initial_records": int(df_raw.shape[0]),
        "total_features": int(X_train.shape[1]),
        "target_variable": "Churn (1 = Churned, 0 = Retained)",
        "overall_churn_rate_pct": round(float(df_raw['Churn'].mean() * 100), 2),
        "missing_values_detected": missing_stats,
        "imputation_strategy": "Hybrid PyTorch Denoising Autoencoder (Learn distributions, fallback to Median). Retains glassbox original features.",
        "imputation_log": imputation_log,
        "encoding_strategy": "No One-Hot Encoding (Categorical text preserved for Explainable Boosting Machine interpretability)",
        "train_test_split_strategy": "80/20 Stratified Split",
        "train_records": int(X_train.shape[0]),
        "test_records": int(X_test.shape[0]),
        "train_churn_rate_pct": round(float(y_train.mean() * 100), 2),
        "test_churn_rate_pct": round(float(y_test.mean() * 100), 2)
    }
    
    summary_path = os.path.join(outputs_dir, 'phase1_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=4)
    print(f"[+] Saved Phase 1 summary report data to: {summary_path}")

if __name__ == "__main__":
    main()
