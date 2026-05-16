"""
evaluate_model.py - Model evaluation module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import os

def setup_logging():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

logger = setup_logging()

class ModelEvaluator:
    """Class for model evaluation"""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate evaluation metrics"""
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics['AUC-ROC'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    def evaluate_model(self, model, model_name, X_test, y_test):
        """Evaluate a single model"""
        y_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = None
        
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, zero_division=0)
        
        self.evaluation_results[model_name] = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.info(f"{model_name} - Acc: {metrics['Accuracy']:.4f}, F1: {metrics['F1-Score']:.4f}")
        return self.evaluation_results[model_name]
    
    def evaluate_all_models(self, trained_models, X_test, y_test):
        """Evaluate all models"""
        for model_name, model in trained_models.items():
            self.evaluate_model(model, model_name, X_test, y_test)
        return self.evaluation_results
    
    def plot_confusion_matrix(self, model_name, save_path=None):
        """Plot confusion matrix"""
        if save_path is None:
            os.makedirs('reports', exist_ok=True)
            save_path = f'reports/confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        
        cm = self.evaluation_results[model_name]['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-Buggy', 'Buggy'],
                   yticklabels=['Non-Buggy', 'Buggy'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix saved to {save_path}")
    
    def plot_roc_curves(self, save_path='reports/roc_curves.png'):
        """Plot ROC curves"""
        os.makedirs('reports', exist_ok=True)
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.evaluation_results.items():
            if results['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(results['predictions'], results['probabilities'])
                auc = results['metrics'].get('AUC-ROC', 0)
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"ROC curves saved to {save_path}")
    
    def plot_model_comparison(self, save_path='reports/model_comparison.png'):
        """Plot model comparison"""
        os.makedirs('reports', exist_ok=True)
        
        metrics_df = pd.DataFrame()
        for model_name, results in self.evaluation_results.items():
            metrics_df[model_name] = pd.Series(results['metrics'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        metrics_df.T.plot(kind='bar', ax=ax, colormap='viridis')
        
        plt.title('Model Performance Comparison')
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.legend(loc='lower right')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim([0, 1.05])
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Model comparison saved to {save_path}")
    
    def generate_evaluation_report(self, save_path='reports/evaluation_report.txt'):
        """Generate evaluation report"""
        os.makedirs('reports', exist_ok=True)
        
        report = "="*70 + "\n"
        report += "MODEL EVALUATION REPORT\n"
        report += "="*70 + "\n\n"
        
        report += "PERFORMANCE SUMMARY\n"
        report += "-"*70 + "\n"
        
        for model_name, results in self.evaluation_results.items():
            metrics = results['metrics']
            report += f"\n{model_name}:\n"
            for metric, value in metrics.items():
                report += f"  {metric}: {value:.4f}\n"
            report += "\n"
        
        with open(save_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation report saved to {save_path}")
        return report
    
    def get_best_model(self, metric='F1-Score'):
        """Get the best performing model"""
        best_model = max(self.evaluation_results.items(), 
                        key=lambda x: x[1]['metrics'].get(metric, 0))
        return best_model

def evaluate_models(trained_models, X_test, y_test, generate_plots=True):
    """Quick evaluation function"""
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all_models(trained_models, X_test, y_test)
    
    if generate_plots:
        for model_name in trained_models.keys():
            evaluator.plot_confusion_matrix(model_name)
        evaluator.plot_roc_curves()
        evaluator.plot_model_comparison()
    
    evaluator.generate_evaluation_report()
    return results, evaluator.get_best_model()