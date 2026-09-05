"""Production inference: Kaggle-trained glassbox EBM -> predicted Churn columns."""

import os
import pickle
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ['Tenure', 'OrderCount', 'DaySinceLastOrder', 'Complain']
MODEL_PATH = 'final_model.sav'
PIPELINE_PATH = 'inference_pipeline.sav'
CHURN_COL = 'Churn'
PCT_COL = 'Churn_Prediction_Percentage'
HIGH_RISK_PCT = 70.0
THRESHOLD = 0.5
HELPER_COLS = ['Churn AI ML', 'Churn_Probability', 'Churn_IsPredicted']


def load_pipeline():
    if os.path.exists(PIPELINE_PATH):
        try:
            return pickle.load(open(PIPELINE_PATH, 'rb'))
        except Exception:
            pass
    return {
        'feature_columns': FEATURE_COLUMNS,
        'impute_values': {c: 0.0 for c in FEATURE_COLUMNS},
        'threshold': THRESHOLD,
        'target': CHURN_COL,
    }


def resolve_column(df: pd.DataFrame, name: str):
    if name in df.columns:
        return name
    lookup = {str(c).lower(): c for c in df.columns}
    return lookup.get(name.lower())


def build_feature_matrix(df: pd.DataFrame, feature_columns, impute_values) -> pd.DataFrame:
    """Build model X from a copy of source columns. Does not mutate the original frame."""
    X = pd.DataFrame(index=df.index)
    for feat in feature_columns:
        src = resolve_column(df, feat)
        fill = impute_values.get(feat, 0)
        if src is None:
            X[feat] = fill
        else:
            X[feat] = pd.to_numeric(df[src], errors='coerce').fillna(fill)
    return X[list(feature_columns)]


def score_uploaded_dataset(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Keep every original uploaded column.
    Always write predicted Churn (0/1) and Churn_Prediction_Percentage.
    Never requires a ground-truth Churn column.
    """
    if df is None or df.empty or model is None:
        return df if df is not None else pd.DataFrame()

    pipeline = load_pipeline()
    features = pipeline.get('feature_columns') or FEATURE_COLUMNS
    impute_values = pipeline.get('impute_values') or {}
    threshold = float(pipeline.get('threshold', THRESHOLD))

    X = build_feature_matrix(df, features, impute_values)
    proba = model.predict_proba(X)[:, 1]
    pct = np.round(proba * 100, 1)
    churn = (proba >= threshold).astype(int)

    out = df.copy()
    out = out.drop(columns=[c for c in HELPER_COLS if c in out.columns], errors='ignore')
    out[CHURN_COL] = churn
    out[PCT_COL] = pct
    return out
