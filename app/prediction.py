"""
Prediction Module for Bug Prediction System
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class BugPredictor:
    """Class for making predictions"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.selected_features = None
        self.load_models()
        
    def load_models(self):
        """Load trained models"""
        try:
            if os.path.exists('models/logistic_regression.pkl'):
                self.models['Logistic Regression'] = joblib.load('models/logistic_regression.pkl')
            if os.path.exists('models/random_forest.pkl'):
                self.models['Random Forest'] = joblib.load('models/random_forest.pkl')
            if os.path.exists('models/xgboost.pkl'):
                self.models['XGBoost'] = joblib.load('models/xgboost.pkl')
            if os.path.exists('models/scaler.pkl'):
                self.scaler = joblib.load('models/scaler.pkl')
            if os.path.exists('models/selected_features.txt'):
                with open('models/selected_features.txt', 'r') as f:
                    self.selected_features = [line.strip() for line in f.readlines()]
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def prepare_features(self, features_dict):
        """Prepare features for prediction"""
        # All features in order
        all_features = ['loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e', 
                       'b', 't', 'lOCode', 'lOComment', 'lOBlank', 'locCodeAndComment', 
                       'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount']
        
        # Create complete feature vector
        feature_vector = []
        for feature in all_features:
            if feature in features_dict:
                feature_vector.append(features_dict[feature])
            else:
                # Provide default values
                if feature in ['loc', 'n', 'v', 'e', 'total_Op', 'total_Opnd']:
                    feature_vector.append(100)
                elif feature in ['v(g)', 'ev(g)', 'iv(g)', 'branchCount', 'uniq_Op', 'uniq_Opnd']:
                    feature_vector.append(10)
                else:
                    feature_vector.append(0)
        
        feature_array = np.array(feature_vector).reshape(1, -1)
        
        # Scale features
        if self.scaler:
            feature_scaled = self.scaler.transform(feature_array)
        else:
            feature_scaled = feature_array
        
        # Select features for model
        if self.selected_features:
            feature_to_idx = {f: i for i, f in enumerate(all_features)}
            selected_indices = [feature_to_idx[f] for f in self.selected_features if f in feature_to_idx]
            model_input = feature_scaled[:, selected_indices]
        else:
            model_input = feature_scaled
        
        return model_input
    
    def predict_single(self, module_name, features):
        """Predict for a single module"""
        try:
            model_input = self.prepare_features(features)
            
            results = {
                'module_name': module_name,
                'timestamp': datetime.now().isoformat(),
                'predictions': {}
            }
            
            probabilities = []
            for model_name, model in self.models.items():
                prob = model.predict_proba(model_input)[0, 1]
                probabilities.append(prob)
                
                risk_score = prob * 100
                if prob >= 0.7:
                    risk_level = "HIGH"
                    recommendation = "Immediate testing required!"
                elif prob >= 0.3:
                    risk_level = "MEDIUM"
                    recommendation = "Schedule for next testing cycle"
                else:
                    risk_level = "LOW"
                    recommendation = "Routine QA process"
                
                results['predictions'][model_name] = {
                    'probability': float(prob),
                    'risk_score': float(risk_score),
                    'risk_level': risk_level,
                    'recommendation': recommendation
                }
            
            # Calculate consensus
            consensus_prob = np.mean(probabilities)
            results['consensus'] = {
                'probability': float(consensus_prob),
                'risk_score': float(consensus_prob * 100),
                'risk_level': "HIGH" if consensus_prob >= 0.7 else "MEDIUM" if consensus_prob >= 0.3 else "LOW"
            }
            
            return results
        except Exception as e:
            return {'error': str(e)}
    
    def predict_batch(self, modules):
        """Predict for multiple modules"""
        results = []
        for module in modules:
            result = self.predict_single(module['name'], module['features'])
            results.append(result)
        return results
    
    def save_prediction_image(self, prediction_result, save_path='reports'):
        """Save prediction result as image"""
        os.makedirs(save_path, exist_ok=True)
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Summary table
        ax1 = axes[0]
        ax1.axis('tight')
        ax1.axis('off')
        
        table_data = [
            ['Module', prediction_result['module_name']],
            ['Timestamp', prediction_result['timestamp']],
            ['Consensus Risk', f"{prediction_result['consensus']['risk_score']:.1f}%"],
            ['Risk Level', prediction_result['consensus']['risk_level']]
        ]
        
        table = ax1.table(cellText=table_data, colLabels=['Metric', 'Value'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.5)
        ax1.set_title('Prediction Summary', fontsize=14, fontweight='bold')
        
        # Model comparison bar chart
        ax2 = axes[1]
        model_names = list(prediction_result['predictions'].keys())
        risk_scores = [prediction_result['predictions'][m]['risk_score'] for m in model_names]
        
        colors = ['#ff6b6b' if score >= 70 else '#ffd93d' if score >= 30 else '#6bcf7f' for score in risk_scores]
        bars = ax2.bar(model_names, risk_scores, color=colors, edgecolor='black')
        ax2.axhline(y=70, color='red', linestyle='--', linewidth=2, label='High Risk Threshold (70%)')
        ax2.axhline(y=30, color='orange', linestyle='--', linewidth=2, label='Medium Risk Threshold (30%)')
        ax2.set_ylabel('Risk Score (%)')
        ax2.set_title('Model Comparison')
        ax2.legend()
        ax2.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, score in zip(bars, risk_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{score:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save figure
        filename = f"prediction_{prediction_result['module_name'].replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def generate_report(self, predictions, save_path='reports'):
        """Generate comprehensive prediction report"""
        os.makedirs(save_path, exist_ok=True)
        
        # Create summary report
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("BUG PREDICTION REPORT")
        report_lines.append("="*60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Models Used: {', '.join(self.models.keys())}")
        report_lines.append("")
        
        for pred in predictions:
            report_lines.append(f"\nModule: {pred['module_name']}")
            report_lines.append("-"*40)
            report_lines.append(f"Consensus Risk: {pred['consensus']['risk_score']:.1f}% ({pred['consensus']['risk_level']})")
            report_lines.append("\nModel Breakdown:")
            for model_name, result in pred['predictions'].items():
                report_lines.append(f"  {model_name}: {result['risk_score']:.1f}% ({result['risk_level']})")
        
        # Save report
        report_path = os.path.join(save_path, f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        return report_path

# Example usage
if __name__ == "__main__":
    predictor = BugPredictor()
    
    if predictor.load_models():
        # Test prediction
        test_features = {
            'loc': 1500,
            'v(g)': 25,
            'branchCount': 30,
            'e': 160000
        }
        
        result = predictor.predict_single("TestModule.java", test_features)
        print(f"Prediction: {result['consensus']['risk_level']} - {result['consensus']['risk_score']:.1f}%")
        
        # Save as image
        image_path = predictor.save_prediction_image(result)
        print(f"Saved image: {image_path}")
