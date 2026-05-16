"""
main.py - Main execution script for the Intelligent Bug Prediction System
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import all modules
from data_preprocessing import DataPreprocessor
from feature_selection import FeatureSelector
from train_model import ModelTrainer, train_models
from evaluate_model import ModelEvaluator, evaluate_models
from risk_scoring import RiskAnalyzer
from utils import setup_logging

def main():
    """Main execution function"""
    
    # Setup logging
    logger = setup_logging()
    print("="*60)
    print("INTELLIGENT BUG PREDICTION SYSTEM")
    print("="*60)
    
    # Configuration
    DATASET_PATH = "dataset/kc1.csv"  # Change to your dataset path
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"\n❌ Dataset not found: {DATASET_PATH}")
        print("Please ensure you have placed kc1.csv or pc1.csv in the 'dataset/' folder")
        print("\nAvailable datasets in dataset folder:")
        if os.path.exists("dataset"):
            for f in os.listdir("dataset"):
                if f.endswith('.csv'):
                    print(f"  - {f}")
        else:
            print("  No dataset folder found!")
            os.makedirs("dataset", exist_ok=True)
            print("  Created 'dataset' folder. Please add your CSV files there.")
        return
    
    try:
        # Step 1: Data Preprocessing
        print("\n[Step 1] Data Preprocessing...")
        preprocessor = DataPreprocessor(scaler_type='standard')
        
        X_train, X_test, y_train, y_test, feature_cols, target_col = preprocessor.preprocess_pipeline(
            DATASET_PATH,
            handle_missing='mean',
            handle_outliers=True,
            scale=True
        )
        
        print(f"✓ Training set: {X_train.shape}")
        print(f"✓ Test set: {X_test.shape}")
        print(f"✓ Features: {len(feature_cols)}")
        print(f"✓ Target column: {target_col}")
        
        # Step 2: Feature Selection
        print("\n[Step 2] Feature Selection...")
        selector = FeatureSelector()
        
        # Use ensemble feature selection
        n_features = min(20, len(feature_cols))
        selected_features, feature_scores = selector.ensemble_selection(
            X_train, y_train, 
            n_features=n_features,
            methods=['correlation', 'anova', 'mi', 'rf']
        )
        
        print(f"✓ Selected {len(selected_features)} important features")
        
        # Reduce datasets to selected features
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        # Plot feature importance
        selector.plot_feature_importance(feature_scores, top_n=15)
        print("✓ Feature importance plot saved")
        
        # Step 3: Model Training
        print("\n[Step 3] Model Training...")
        trainer = ModelTrainer()
        trained_models, training_results = train_models(X_train_selected, y_train, perform_tuning=False)
        print(f"✓ Trained {len(trained_models)} models")
        
        # Step 4: Model Evaluation
        print("\n[Step 4] Model Evaluation...")
        evaluator = ModelEvaluator()
        evaluation_results = evaluator.evaluate_all_models(trained_models, X_test_selected, y_test)
        
        # Generate evaluation visualizations
        for model_name in trained_models.keys():
            evaluator.plot_confusion_matrix(model_name)
        
        evaluator.plot_roc_curves()
        evaluator.plot_model_comparison()
        evaluator.generate_evaluation_report()
        
        # Get best model
        best_model_name, best_model_results = evaluator.get_best_model(metric='F1-Score')
        print(f"\n✓ Best Model: {best_model_name}")
        
        # Step 5: Risk Scoring
        print("\n[Step 5] Risk Scoring...")
        best_model = trained_models[best_model_name]
        
        # Get predictions for test set
        y_pred_proba = best_model.predict_proba(X_test_selected)[:, 1]
        
        # Create module names
        module_names = [f"Module_{i}" for i in range(len(y_pred_proba))]
        
        # Analyze risks
        risk_analyzer = RiskAnalyzer()
        risk_df = risk_analyzer.calculate_risk_scores(y_pred_proba, module_names)
        
        # Generate risk visualizations and reports
        risk_analyzer.plot_risk_distribution(risk_df)
        risk_analyzer.generate_risk_report(risk_df)
        risk_analyzer.export_risk_data(risk_df)
        risk_metrics = risk_analyzer.generate_risk_metrics(risk_df)
        
        # Step 6: Summary
        print("\n" + "="*60)
        print("PROJECT EXECUTION SUMMARY")
        print("="*60)
        
        print(f"\n📊 Dataset Information:")
        print(f"  - Total samples: {len(X_train) + len(X_test)}")
        print(f"  - Features used: {len(selected_features)}")
        print(f"  - Buggy modules: {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.1f}%)")
        
        print(f"\n🎯 Model Performance:")
        for model_name, results in evaluation_results.items():
            metrics = results['metrics']
            print(f"  {model_name}:")
            print(f"    - Accuracy: {metrics.get('Accuracy', 0):.4f}")
            print(f"    - F1-Score: {metrics.get('F1-Score', 0):.4f}")
        
        print(f"\n⚠️ Risk Analysis Summary:")
        print(f"  - High Risk Modules: {risk_metrics['high_risk_modules']} ({risk_metrics['high_risk_percentage']:.1f}%)")
        print(f"  - Medium Risk Modules: {risk_metrics['medium_risk_modules']}")
        print(f"  - Low Risk Modules: {risk_metrics['low_risk_modules']}")
        print(f"  - Average Risk Score: {risk_metrics['average_risk_score']:.2f}")
        
        # Show top high-risk modules
        print(f"\n🔥 Top 5 High-Risk Modules:")
        high_risk = risk_df[risk_df['Risk_Level'] == 'High'].head(5)
        if len(high_risk) > 0:
            for idx, row in high_risk.iterrows():
                print(f"  {row['Risk_Rank']}. {row['Module']} - Score: {row['Risk_Score']:.1f}%")
        else:
            print("  No high-risk modules detected")
        
        print("\n" + "="*60)
        print("✅ EXECUTION COMPLETED SUCCESSFULLY!")
        print("📁 Check the 'reports/' directory for all outputs")
        print("📁 Check the 'models/' directory for saved models")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()