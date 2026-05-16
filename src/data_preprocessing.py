"""
data_preprocessing.py - Data preprocessing and cleaning module
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Simple logger setup
def setup_logging():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

logger = setup_logging()

# Configuration
CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'target_column': 'defects',
    'alternative_targets': ['Defective', 'bug', 'label', 'class', 'defect']
}

def identify_target_column(df):
    """Automatically identify the target column"""
    if 'defects' in df.columns:
        return 'defects'
    for alt in ['Defective', 'bug', 'label', 'class', 'defect']:
        if alt in df.columns:
            return alt
    for col in df.columns:
        if df[col].nunique() <= 2 and df[col].dtype in ['int64', 'float64']:
            if set(df[col].unique()).issubset({0, 1, 0.0, 1.0}):
                return col
    raise ValueError("Could not identify target column")

def get_feature_columns(df, target_col):
    """Get feature columns"""
    feature_cols = [col for col in df.columns if col != target_col]
    feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
    return feature_cols

def save_scaler(scaler):
    """Save scaler to disk"""
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    logger.info("Scaler saved to models/scaler.pkl")

class DataPreprocessor:
    """Class for handling all data preprocessing operations"""
    
    def __init__(self, scaler_type='standard'):
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_columns = None
        self.target_column = None
        
    def load_and_explore(self, filepath):
        """Load dataset and perform initial exploration"""
        df = pd.read_csv(filepath)
        logger.info(f"Dataset loaded: {filepath}")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        return df
    
    def handle_missing_values(self, df, strategy='mean'):
        """Handle missing values"""
        df_clean = df.copy()
        if strategy == 'drop':
            df_clean = df.dropna()
        else:
            for col in df_clean.columns:
                if df_clean[col].isnull().any():
                    if strategy == 'mean' and pd.api.types.is_numeric_dtype(df_clean[col]):
                        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                    elif strategy == 'median' and pd.api.types.is_numeric_dtype(df_clean[col]):
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    else:
                        df_clean[col].fillna(0, inplace=True)
        return df_clean
    
    def remove_duplicates(self, df):
        """Remove duplicate rows"""
        return df.drop_duplicates()
    
    def handle_outliers(self, df):
        """Handle outliers using IQR method"""
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        return df_clean
    
    def scale_features(self, df, feature_cols, fit=True):
        """Scale numerical features"""
        if fit:
            if self.scaler_type == 'standard':
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler()
            scaled_features = self.scaler.fit_transform(df[feature_cols])
            save_scaler(self.scaler)
        else:
            scaled_features = self.scaler.transform(df[feature_cols])
        
        df_scaled = df.copy()
        df_scaled[feature_cols] = scaled_features
        return df_scaled
    
    def split_data(self, df, target_col):
        """Split data into training and testing sets"""
        feature_cols = get_feature_columns(df, target_col)
        X = df[feature_cols]
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=CONFIG['test_size'], 
            random_state=CONFIG['random_state'], stratify=y
        )
        
        logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        return X_train, X_test, y_train, y_test, feature_cols
    
    def preprocess_pipeline(self, filepath, handle_missing='mean', handle_outliers=True, scale=True):
        """Complete preprocessing pipeline"""
        # Load data
        df = self.load_and_explore(filepath)
        
        # Identify target column
        self.target_column = identify_target_column(df)
        logger.info(f"Target column: {self.target_column}")
        
        # Handle missing values
        df = self.handle_missing_values(df, strategy=handle_missing)
        
        # Remove duplicates
        df = self.remove_duplicates(df)
        
        # Handle outliers
        if handle_outliers:
            df = self.handle_outliers(df)
        
        # Get feature columns
        self.feature_columns = get_feature_columns(df, self.target_column)
        logger.info(f"Feature columns: {len(self.feature_columns)}")
        
        # Scale features
        if scale:
            df = self.scale_features(df, self.feature_columns, fit=True)
        
        # Split data
        X_train, X_test, y_train, y_test, feature_cols = self.split_data(df, self.target_column)
        
        return X_train, X_test, y_train, y_test, feature_cols, self.target_column