"""
feature_selection.py - Feature selection module
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os

def setup_logging():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

logger = setup_logging()

class FeatureSelector:
    """Class for feature selection"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.selected_features = None
        self.feature_scores = None
        
    def correlation_analysis(self, X, y):
        """Analyze correlation between features and target"""
        df_temp = X.copy()
        df_temp['target'] = y
        correlations = df_temp.corr()['target'].drop('target').sort_values(ascending=False)
        
        correlation_df = pd.DataFrame({
            'Feature': correlations.index,
            'Correlation': correlations.values,
            'Abs_Correlation': np.abs(correlations.values)
        }).sort_values('Abs_Correlation', ascending=False)
        
        return correlation_df
    
    def anova_selection(self, X, y, k=10):
        """Select top k features using ANOVA"""
        k = min(k, X.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X, y)
        
        scores = pd.DataFrame({
            'Feature': X.columns,
            'ANOVA_Score': selector.scores_,
            'p_value': selector.pvalues_
        }).sort_values('ANOVA_Score', ascending=False)
        
        selected_features = X.columns[selector.get_support()].tolist()
        return selected_features, scores
    
    def mutual_info_selection(self, X, y, k=10):
        """Select features using Mutual Information"""
        k = min(k, X.shape[1])
        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        
        scores = pd.DataFrame({
            'Feature': X.columns,
            'MI_Score': mi_scores
        }).sort_values('MI_Score', ascending=False)
        
        selected_features = scores.head(k)['Feature'].tolist()
        return selected_features, scores
    
    def random_forest_importance(self, X, y, n_estimators=100):
        """Calculate feature importance using Random Forest"""
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=self.random_state)
        rf.fit(X, y)
        
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def ensemble_selection(self, X, y, n_features=10, methods=['correlation', 'anova', 'mi', 'rf']):
        """Combine multiple feature selection methods"""
        feature_scores = pd.DataFrame({'Feature': X.columns})
        
        if 'correlation' in methods:
            corr_df = self.correlation_analysis(X, y)
            feature_scores['Correlation_Score'] = feature_scores['Feature'].map(
                lambda x: corr_df[corr_df['Feature'] == x]['Abs_Correlation'].values[0] 
                if len(corr_df[corr_df['Feature'] == x]) > 0 else 0
            )
        
        if 'anova' in methods:
            _, anova_df = self.anova_selection(X, y, k=X.shape[1])
            feature_scores['ANOVA_Score'] = feature_scores['Feature'].map(
                lambda x: anova_df[anova_df['Feature'] == x]['ANOVA_Score'].values[0] 
                if len(anova_df[anova_df['Feature'] == x]) > 0 else 0
            )
        
        if 'mi' in methods:
            _, mi_df = self.mutual_info_selection(X, y, k=X.shape[1])
            feature_scores['MI_Score'] = feature_scores['Feature'].map(
                lambda x: mi_df[mi_df['Feature'] == x]['MI_Score'].values[0] 
                if len(mi_df[mi_df['Feature'] == x]) > 0 else 0
            )
        
        if 'rf' in methods:
            rf_df = self.random_forest_importance(X, y)
            feature_scores['RF_Importance'] = feature_scores['Feature'].map(
                lambda x: rf_df[rf_df['Feature'] == x]['Importance'].values[0] 
                if len(rf_df[rf_df['Feature'] == x]) > 0 else 0
            )
        
        # Normalize scores
        for col in feature_scores.columns:
            if col != 'Feature':
                max_val = feature_scores[col].max()
                if max_val > 0:
                    feature_scores[f'{col}_Norm'] = feature_scores[col] / max_val
        
        # Calculate ensemble score
        norm_cols = [col for col in feature_scores.columns if col.endswith('_Norm')]
        if norm_cols:
            feature_scores['Ensemble_Score'] = feature_scores[norm_cols].mean(axis=1)
        else:
            feature_scores['Ensemble_Score'] = 0
        
        feature_scores = feature_scores.sort_values('Ensemble_Score', ascending=False)
        selected_features = feature_scores.head(n_features)['Feature'].tolist()
        
        self.feature_scores = feature_scores
        self.selected_features = selected_features
        
        return selected_features, feature_scores
    
    def plot_feature_importance(self, feature_scores_df, top_n=20, save_path='reports/feature_importance.png'):
        """Plot feature importance"""
        os.makedirs('reports', exist_ok=True)
        plt.figure(figsize=(12, 8))
        top_features = feature_scores_df.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['Ensemble_Score'], color='skyblue')
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Ensemble Importance Score')
        plt.title(f'Top {top_n} Most Important Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Feature importance plot saved to {save_path}")