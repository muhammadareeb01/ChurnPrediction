import pandas as pd
import pickle
import os
from interpret.glassbox import ExplainableBoostingClassifier

def retrain_model():
    print("[*] Loading training data...")
    X_train_path = os.path.join('data', 'processed', 'X_train.csv')
    y_train_path = os.path.join('data', 'processed', 'y_train.csv')
    
    if not os.path.exists(X_train_path) or not os.path.exists(y_train_path):
        print("[!] Missing training data. Running preprocess first...")
        from src.preprocess import main as run_preprocess
        run_preprocess()
        
    X_train = pd.read_csv(X_train_path)
    y_df = pd.read_csv(y_train_path)
    y_train = y_df['Churn'] if 'Churn' in y_df.columns else y_df.iloc[:, 0]
    
    # Ensure categorical columns are strings for EBM
    cat_cols = X_train.select_dtypes(include=['object']).columns
    for col in cat_cols:
        X_train[col] = X_train[col].astype(str)
        
    print(f"[*] Training Explainable Boosting Machine (EBM) on {X_train.shape[0]} records and {X_train.shape[1]} features...")
    clf = ExplainableBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    model_path = "final_model.sav"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
        
    print(f"[+] Model successfully trained and saved as {model_path}!")

if __name__ == "__main__":
    retrain_model()

