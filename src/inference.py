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


CAT_FEATURES = {
    'PreferedOrderCat', 'MaritalStatus', 'ProductName',
    'PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
}


def model_feature_list(model, fallback):
    names = getattr(model, 'feature_names_in_', None) or getattr(model, 'feature_names', None)
    if names is None:
        return list(fallback)
    return list(names)


def build_feature_matrix(df: pd.DataFrame, feature_columns, impute_values) -> pd.DataFrame:
    """Build model X from a copy of source columns. Does not mutate the original frame."""
    work = df.copy()
    dsl = resolve_column(work, 'DaySinceLastOrder')
    comp = resolve_column(work, 'Complain')
    if 'RFM_Recency' in feature_columns and resolve_column(work, 'RFM_Recency') is None and dsl:
        work['RFM_Recency'] = pd.to_numeric(work[dsl], errors='coerce')
    if 'Friction_Risk' in feature_columns and resolve_column(work, 'Friction_Risk') is None and dsl and comp:
        complain = pd.to_numeric(work[comp], errors='coerce').fillna(0)
        recency = pd.to_numeric(work[dsl], errors='coerce').fillna(0)
        work['Friction_Risk'] = complain * (recency + 1)

    X = pd.DataFrame(index=df.index)
    for feat in feature_columns:
        src = resolve_column(work, feat)
        fill = impute_values.get(feat, 0)
        if feat in CAT_FEATURES:
            if src is None:
                X[feat] = 'Unknown'
            else:
                X[feat] = work[src].astype(str).replace({'nan': 'Unknown', 'None': 'Unknown'})
        else:
            if src is None:
                X[feat] = fill
            else:
                X[feat] = pd.to_numeric(work[src], errors='coerce').fillna(fill)
    return X[list(feature_columns)]


def score_uploaded_dataset(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Keep every original uploaded column.
    Always write predicted Churn (0/1) and Churn_Prediction_Percentage.
    Never requires a ground-truth Churn column.
    Aligns columns to whatever the loaded EBM was trained on (4-feature or 12-feature).
    """
    if df is None or df.empty or model is None:
        return df if df is not None else pd.DataFrame()

    pipeline = load_pipeline()
    fallback = pipeline.get('feature_columns') or FEATURE_COLUMNS
    impute_values = pipeline.get('impute_values') or {}
    threshold = float(pipeline.get('threshold', THRESHOLD))
    features = model_feature_list(model, fallback)

    X = build_feature_matrix(df, features, impute_values)
    proba = model.predict_proba(X)[:, 1]
    pct = np.round(proba * 100, 1)
    churn = (proba >= threshold).astype(int)

    out = df.copy()
    out = out.drop(columns=[c for c in HELPER_COLS if c in out.columns], errors='ignore')
    out[CHURN_COL] = churn
    out[PCT_COL] = pct
    return out
