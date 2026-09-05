import pandas as pd
import pickle
import os
from interpret.glassbox import ExplainableBoostingClassifier

GLASSBOX_FEATURES = ['Tenure', 'OrderCount', 'DaySinceLastOrder', 'Complain']
TARGET_COL = 'Churn'


def _print_churn_patterns(X_train, y_train):
    """Show how each input column relates to the Kaggle Churn label."""
    frame = X_train.copy()
    frame[TARGET_COL] = y_train.values
    print("[*] Supervised glassbox target: Kaggle Churn (1 = left, 0 = stayed)")
    print(f"    -> Churners: {int(y_train.sum()):,} | Retained: {int((y_train == 0).sum()):,} | Rate: {y_train.mean() * 100:.2f}%")
    complain_rate = frame.groupby('Complain')[TARGET_COL].mean() * 100
    for complain_val, rate in complain_rate.items():
        print(f"    -> Complain={int(complain_val)} churn rate: {rate:.2f}%")
    print(f"    -> Mean Tenure | churners={frame.loc[frame[TARGET_COL] == 1, 'Tenure'].mean():.2f} vs retained={frame.loc[frame[TARGET_COL] == 0, 'Tenure'].mean():.2f}")
    print(f"    -> Mean OrderCount | churners={frame.loc[frame[TARGET_COL] == 1, 'OrderCount'].mean():.2f} vs retained={frame.loc[frame[TARGET_COL] == 0, 'OrderCount'].mean():.2f}")
    print(f"    -> Mean DaySinceLastOrder | churners={frame.loc[frame[TARGET_COL] == 1, 'DaySinceLastOrder'].mean():.2f} vs retained={frame.loc[frame[TARGET_COL] == 0, 'DaySinceLastOrder'].mean():.2f}")


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
    if TARGET_COL not in y_df.columns:
        raise ValueError("Kaggle Churn labels are missing. EBM cannot learn churn patterns without the target column.")
    y_train = y_df[TARGET_COL].astype(int)

    keep_cols = [c for c in GLASSBOX_FEATURES if c in X_train.columns]
    if len(keep_cols) != len(GLASSBOX_FEATURES):
        missing = [c for c in GLASSBOX_FEATURES if c not in X_train.columns]
        raise ValueError(f"Missing glassbox input columns: {missing}")
    X_train = X_train[keep_cols]
    if TARGET_COL in X_train.columns:
        X_train = X_train.drop(columns=[TARGET_COL])

    _print_churn_patterns(X_train, y_train)

    print(f"[*] Training glassbox EBM (InterpretML) on {X_train.shape[0]} records and {X_train.shape[1]} features...")
    print("    Algorithm lock: ExplainableBoostingClassifier only — no blackbox models in the live pipeline.")
    clf = ExplainableBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    model_path = "final_model.sav"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    pipeline = {
        "feature_columns": GLASSBOX_FEATURES,
        "impute_values": {c: float(X_train[c].median()) for c in GLASSBOX_FEATURES},
        "threshold": 0.5,
        "target": TARGET_COL,
        "algorithm": "ExplainableBoostingClassifier",
    }
    with open("inference_pipeline.sav", "wb") as f:
        pickle.dump(pipeline, f)
        
    print(f"[+] Glassbox EBM trained against Kaggle Churn labels and saved as {model_path}!")
    print("[+] Saved inference_pipeline.sav (feature list, impute values, 0.5 threshold).")

if __name__ == "__main__":
    retrain_model()

