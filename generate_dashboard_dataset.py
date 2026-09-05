import pandas as pd
import pickle
import os
import numpy as np

def create_dashboard_data():
    print("[*] Loading E-Commerce dataset for dashboard prediction...")
    if os.path.exists('E Commerce Dataset Updated.xlsx'):
        file_path = 'E Commerce Dataset Updated.xlsx'
        sheet_name = 'EBM_Churn_Data'
    # Fallback to older ones if needed, or Kaggle CSV
    elif os.path.exists(os.path.join('data', 'raw', 'E Commerce Dataset Updated.xlsx')):
        file_path = os.path.join('data', 'raw', 'E Commerce Dataset Updated.xlsx')
        sheet_name = 'EBM_Churn_Data'
    else:
        print("[!] E-Commerce dataset not found. Please provide 'E Commerce Dataset Updated.xlsx'.")
        return
        
    try:
        dataset = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception:
        print(f"[*] Sheet '{sheet_name}' not found, reading default/first sheet...")
        dataset = pd.read_excel(file_path)
    
    model_path = "final_model.sav"
    model = pickle.load(open(model_path, "rb"))

    print("[*] Generating AI Predictions (using Kaggle-trained patterns)...")
    from src.inference import score_uploaded_dataset
    dataset = score_uploaded_dataset(dataset, model)
    
    # Save the ready dataset
    out_path = "Final_Dashboard_Data.csv"
    dataset.to_csv(out_path, index=False)
    print(f"[+] Final Dashboard Data saved at {out_path}! Total rows: {len(dataset)}")

if __name__ == "__main__":
    create_dashboard_data()

