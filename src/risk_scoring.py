"""
risk_scoring.py - Risk scoring module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def setup_logging():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

logger = setup_logging()

def generate_risk_score(probability):
    """Generate risk score and level"""
    risk_score = probability * 100
    
    if probability < 0.3:
        risk_level = "Low"
    elif probability < 0.7:
        risk_level = "Medium"
    else:
        risk_level = "High"
    
    return {
        'risk_score': round(risk_score, 2),
        'risk_level': risk_level,
        'probability': probability
    }

class RiskAnalyzer:
    """Class for risk analysis"""
    
    def __init__(self, risk_thresholds=None):
        self.risk_thresholds = risk_thresholds or {'low': 0.3, 'medium': 0.7}
    
    def calculate_risk_scores(self, probabilities, module_names=None):
        """Calculate risk scores"""
        if module_names is None:
            module_names = [f"Module_{i}" for i in range(len(probabilities))]
        
        risk_data = []
        for i, prob in enumerate(probabilities):
            risk_info = generate_risk_score(prob)
            risk_data.append({
                'Module': module_names[i],
                'Bug_Probability': prob,
                'Risk_Score': risk_info['risk_score'],
                'Risk_Level': risk_info['risk_level']
            })
        
        risk_df = pd.DataFrame(risk_data)
        risk_df['Risk_Rank'] = risk_df['Risk_Score'].rank(ascending=False).astype(int)
        
        return risk_df
    
    def get_high_risk_modules(self, risk_df, top_n=10):
        """Get top high-risk modules"""
        high_risk = risk_df[risk_df['Risk_Level'] == 'High'].sort_values('Risk_Score', ascending=False)
        if len(high_risk) < top_n:
            additional = top_n - len(high_risk)
            medium_risk = risk_df[risk_df['Risk_Level'] == 'Medium'].sort_values('Risk_Score', ascending=False).head(additional)
            high_risk = pd.concat([high_risk, medium_risk])
        return high_risk.head(top_n)
    
    def generate_risk_report(self, risk_df, save_path='reports/risk_report.txt'):
        """Generate risk report"""
        os.makedirs('reports', exist_ok=True)
        
        report = "="*60 + "\n"
        report += "RISK ANALYSIS REPORT\n"
        report += "="*60 + "\n\n"
        
        risk_dist = risk_df['Risk_Level'].value_counts()
        for level, count in risk_dist.items():
            percentage = (count / len(risk_df)) * 100
            report += f"{level} Risk: {count} modules ({percentage:.1f}%)\n"
        
        report += f"\nTotal Modules: {len(risk_df)}\n"
        report += f"Average Risk Score: {risk_df['Risk_Score'].mean():.2f}\n"
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Risk report saved to {save_path}")
        return report
    
    def plot_risk_distribution(self, risk_df, save_path='reports/risk_distribution.png'):
        """Plot risk distribution"""
        os.makedirs('reports', exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Risk level distribution
        risk_counts = risk_df['Risk_Level'].value_counts()
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        axes[0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
        axes[0].set_title('Risk Level Distribution', fontsize=14, fontweight='bold')
        
        # Risk score histogram
        axes[1].hist(risk_df['Risk_Score'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        axes[1].axvline(risk_df['Risk_Score'].mean(), color='red', linestyle='dashed', 
                       linewidth=2, label=f"Mean: {risk_df['Risk_Score'].mean():.1f}")
        axes[1].set_xlabel('Risk Score (%)')
        axes[1].set_ylabel('Number of Modules')
        axes[1].set_title('Risk Score Distribution', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Risk distribution plot saved to {save_path}")
    
    def export_risk_data(self, risk_df, filepath='reports/risk_data.csv'):
        """Export risk data to CSV"""
        os.makedirs('reports', exist_ok=True)
        risk_df.to_csv(filepath, index=False)
        logger.info(f"Risk data exported to {filepath}")
    
    def generate_risk_metrics(self, risk_df):
        """Generate risk metrics"""
        metrics = {
            'total_modules': len(risk_df),
            'high_risk_modules': len(risk_df[risk_df['Risk_Level'] == 'High']),
            'medium_risk_modules': len(risk_df[risk_df['Risk_Level'] == 'Medium']),
            'low_risk_modules': len(risk_df[risk_df['Risk_Level'] == 'Low']),
            'average_risk_score': risk_df['Risk_Score'].mean(),
            'max_risk_score': risk_df['Risk_Score'].max()
        }
        
        metrics['high_risk_percentage'] = (metrics['high_risk_modules'] / metrics['total_modules']) * 100
        
        return metrics

def analyze_risks(probabilities, module_names=None):
    """Quick risk analysis function"""
    analyzer = RiskAnalyzer()
    risk_df = analyzer.calculate_risk_scores(probabilities, module_names)
    analyzer.plot_risk_distribution(risk_df)
    analyzer.generate_risk_report(risk_df)
    analyzer.export_risk_data(risk_df)
    metrics = analyzer.generate_risk_metrics(risk_df)
    return risk_df, metrics