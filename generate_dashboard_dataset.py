import pandas as pd
import pickle
import os
import numpy as np

def create_dashboard_data():
    print("[*] Loading EasyBazar dataset for dashboard prediction...")
    if os.path.exists('EasyBazar_EBM_Dataset.xlsx'):
        file_path = 'EasyBazar_EBM_Dataset.xlsx'
        sheet_name = 'EBM_Churn_Data'
    elif os.path.exists(os.path.join('data', 'raw', 'EasyBazar_EBM_Dataset.xlsx')):
        file_path = os.path.join('data', 'raw', 'EasyBazar_EBM_Dataset.xlsx')
        sheet_name = 'EBM_Churn_Data'
    else:
        print("[!] EasyBazar dataset not found. Please provide 'EasyBazar_EBM_Dataset.xlsx'.")
        return
        
    try:
        dataset = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception:
        print(f"[*] Sheet '{sheet_name}' not found, reading default/first sheet...")
        dataset = pd.read_excel(file_path)
    
    # Load the retrained model (trained on Kaggle)
    model_path = "final_model.sav"
    model = pickle.load(open(model_path, "rb"))
    
    print("[*] Processing missing values for prediction...")
    candidate_continuous = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'COD_Amount']
    for col in candidate_continuous:
        if col in dataset.columns and dataset[col].isnull().sum() > 0:
            dataset[col] = dataset[col].fillna(dataset[col].median(numeric_only=True))

    cat_cols = dataset.select_dtypes(include=['object']).columns
    for col in cat_cols:
        dataset[col] = dataset[col].astype(str).str.strip()
        
    print("[*] Generating RFM and Indirect Business Signals...")
    try:
        from src.preprocess import engineer_rfm_features
        dataset = engineer_rfm_features(dataset)
    except Exception as e:
        print(f"[!] Warning: Could not generate RFM signals. Error: {e}")

    # Align features specifically for model prediction
    model_features = getattr(model, 'feature_names', None) or getattr(model, 'feature_names_in_', None)
    if model_features:
        for mf in model_features:
            if mf not in dataset.columns:
                if mf == 'ProductName':
                    dataset[mf] = 'Unknown'
                else:
                    dataset[mf] = 0
            else:
                if mf != 'ProductName':
                    dataset[mf] = pd.to_numeric(dataset[mf], errors='coerce').fillna(0)
                else:
                    dataset[mf] = dataset[mf].astype(str)
        X = dataset[model_features].copy()
    else:
        drop_candidates = ['CustomerID', 'CustomerName', 'OrderDate', 'ShipmentID', 'Churn']
        drop_cols = [c for c in drop_candidates if c in dataset.columns]
        X = dataset.drop(columns=drop_cols, errors='ignore')
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype(str)

    print("[*] Generating AI Predictions (using Kaggle-trained patterns)...")
    dataset['Churn AI ML'] = model.predict(X)
    if hasattr(model, 'predict_proba'):
        try:
            dataset['Churn_Probability'] = np.round(model.predict_proba(X)[:, 1] * 100, 1)
        except Exception:
            pass
    
    # Save the ready dataset
    out_path = "Final_Dashboard_Data.csv"
    dataset.to_csv(out_path, index=False)
    print(f"[+] Final Dashboard Data saved at {out_path}! Total rows: {len(dataset)}")

if __name__ == "__main__":
    create_dashboard_data()

