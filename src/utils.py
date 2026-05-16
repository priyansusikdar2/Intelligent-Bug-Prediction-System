"""
utils.py - Utility functions
"""

import os
import pandas as pd
import numpy as np
import joblib
import logging

def setup_logging(log_file='bug_prediction.log'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_model(model_name):
    """Load trained model from disk"""
    filepath = os.path.join('models', model_name)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found: {filepath}")
    return joblib.load(filepath)

def load_scaler(scaler_name='scaler.pkl'):
    """Load scaler from disk"""
    filepath = os.path.join('models', scaler_name)
    if not os.path.exists(filepath):
        return None
    return joblib.load(filepath)

def identify_target_column(df):
    """Identify target column"""
    target_candidates = ['defects', 'Defective', 'bug', 'label', 'class', 'defect']
    for col in target_candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        if df[col].nunique() <= 2 and df[col].dtype in ['int64', 'float64']:
            return col
    return None

def get_feature_columns(df, target_col):
    """Get feature columns"""
    feature_cols = [col for col in df.columns if col != target_col]
    feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
    return feature_cols