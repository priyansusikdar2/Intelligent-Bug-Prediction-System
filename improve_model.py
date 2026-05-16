"""
Improved model training with hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from src.data_preprocessing import DataPreprocessor
from src.feature_selection import FeatureSelector
from src.evaluate_model import ModelEvaluator

print("="*60)
print("IMPROVED MODEL TRAINING WITH HYPERPARAMETER TUNING")
print("="*60)

# Load and preprocess data
print("\n[1/5] Loading and preprocessing data...")
preprocessor = DataPreprocessor()
X_train, X_test, y_train, y_test, features, target = preprocessor.preprocess_pipeline(
    'dataset/kc1.csv',
    handle_missing='mean',
    handle_outliers=True,
    scale=True
)

print(f"✓ Training set: {X_train.shape}")
print(f"✓ Test set: {X_test.shape}")
print(f"✓ Features: {len(features)}")

# Feature selection
print("\n[2/5] Performing feature selection...")
selector = FeatureSelector()
selected_features, feature_scores = selector.ensemble_selection(
    X_train, y_train,
    n_features=min(15, len(features)),  # Select top 15 features
    methods=['correlation', 'anova', 'mi', 'rf']
)

# Reduce datasets to selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

print(f"✓ Selected {len(selected_features)} important features")
print(f"  Features: {selected_features[:5]}...")

# Define models with parameter grids
print("\n[3/5] Setting up hyperparameter tuning...")

models_to_tune = {
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', None]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss'),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    },
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'class_weight': ['balanced', None],
            'solver': ['lbfgs', 'liblinear']
        }
    }
}

# Train and tune models
print("\n[4/5] Training models with hyperparameter tuning...")

tuned_models = {}
best_models = {}
cv_results = {}

for model_name, model_config in models_to_tune.items():
    print(f"\n{'─'*50}")
    print(f"🎯 Tuning {model_name}...")
    print(f"{'─'*50}")
    
    # Perform Grid Search
    grid_search = GridSearchCV(
        model_config['model'],
        model_config['params'],
        cv=5,  # 5-fold cross-validation
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_selected, y_train)
    
    # Store best model
    best_models[model_name] = grid_search.best_estimator_
    cv_results[model_name] = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }
    
    print(f"✓ Best parameters: {grid_search.best_params_}")
    print(f"✓ Best CV score (AUC-ROC): {grid_search.best_score_:.4f}")
    
    # Save model
    model_filename = f'models/{model_name.lower().replace(" ", "_")}_tuned.pkl'
    joblib.dump(grid_search.best_estimator_, model_filename)
    print(f"✓ Model saved to {model_filename}")

# Evaluate tuned models
print("\n[5/5] Evaluating tuned models on test set...")
print("="*60)

results = {}
for model_name, model in best_models.items():
    # Make predictions
    y_pred = model.predict(X_test_selected)
    y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
    
    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'AUC-ROC': roc_auc_score(y_test, y_pred_proba)
    }
    
    results[model_name] = {
        'metrics': metrics,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"\n📊 {model_name}:")
    print(f"   Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"   Precision: {metrics['Precision']:.4f}")
    print(f"   Recall:    {metrics['Recall']:.4f}")
    print(f"   F1-Score:  {metrics['F1-Score']:.4f}")
    print(f"   AUC-ROC:   {metrics['AUC-ROC']:.4f}")

# Compare with baseline (untuned) models
print("\n" + "="*60)
print("PERFORMANCE COMPARISON: Tuned vs Baseline")
print("="*60)

# Load baseline models if they exist
print("\n📈 Improvement Analysis:")

# Create comparison table
comparison_data = []
for model_name in best_models.keys():
    tuned_f1 = results[model_name]['metrics']['F1-Score']
    tuned_acc = results[model_name]['metrics']['Accuracy']
    
    # Baseline performance (from your earlier run)
    baseline_metrics = {
        'Logistic Regression': {'F1-Score': 0.4295, 'Accuracy': 0.6502},
        'Random Forest': {'F1-Score': 0.2941, 'Accuracy': 0.7037},
        'XGBoost': {'F1-Score': 0.3363, 'Accuracy': 0.6914}
    }
    
    if model_name in baseline_metrics:
        baseline_f1 = baseline_metrics[model_name]['F1-Score']
        baseline_acc = baseline_metrics[model_name]['Accuracy']
        
        f1_improvement = ((tuned_f1 - baseline_f1) / baseline_f1) * 100
        acc_improvement = ((tuned_acc - baseline_acc) / baseline_acc) * 100
        
        comparison_data.append({
            'Model': model_name,
            'Baseline F1': baseline_f1,
            'Tuned F1': tuned_f1,
            'F1 Improvement': f1_improvement,
            'Baseline Acc': baseline_acc,
            'Tuned Acc': tuned_acc,
            'Acc Improvement': acc_improvement
        })
        
        print(f"\n{model_name}:")
        print(f"   F1-Score:  {baseline_f1:.4f} → {tuned_f1:.4f} ({f1_improvement:+.1f}%)")
        print(f"   Accuracy:  {baseline_acc:.4f} → {tuned_acc:.4f} ({acc_improvement:+.1f}%)")

# Find best model
best_model_name = max(results.items(), key=lambda x: x[1]['metrics']['F1-Score'])[0]
best_model_metrics = results[best_model_name]['metrics']

print("\n" + "="*60)
print("🎯 BEST MODEL RECOMMENDATION")
print("="*60)
print(f"\n✅ Best performing model: {best_model_name}")
print(f"\nPerformance metrics:")
print(f"   • Accuracy:  {best_model_metrics['Accuracy']*100:.2f}%")
print(f"   • Precision: {best_model_metrics['Precision']*100:.2f}%")
print(f"   • Recall:    {best_model_metrics['Recall']*100:.2f}%")
print(f"   • F1-Score:  {best_model_metrics['F1-Score']*100:.2f}%")
print(f"   • AUC-ROC:   {best_model_metrics['AUC-ROC']*100:.2f}%")

# Save the best model as the default
best_model = best_models[best_model_name]
joblib.dump(best_model, 'models/best_model.pkl')
print(f"\n💾 Best model saved to 'models/best_model.pkl'")

# Create comparison plot
print("\n📊 Generating performance comparison plot...")
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Prepare data for plotting
    plot_data = []
    for model_name, result in results.items():
        for metric, value in result['metrics'].items():
            plot_data.append({
                'Model': model_name,
                'Metric': metric,
                'Value': value
            })
    
    df_plot = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x='Model', y='Value', hue='Metric')
    plt.title('Model Performance Comparison (Tuned Models)')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('reports/tuned_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Comparison plot saved to 'reports/tuned_model_comparison.png'")
except Exception as e:
    print(f"⚠️ Could not create plot: {e}")

# Save detailed report
print("\n💾 Saving detailed report...")
with open('reports/tuned_models_report.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("IMPROVED MODEL TRAINING REPORT\n")
    f.write("="*70 + "\n\n")
    
    f.write("DATASET INFORMATION\n")
    f.write("-"*40 + "\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Test samples: {len(X_test)}\n")
    f.write(f"Features used: {len(selected_features)}\n")
    f.write(f"Selected features: {', '.join(selected_features)}\n\n")
    
    f.write("MODEL PERFORMANCE\n")
    f.write("-"*40 + "\n")
    for model_name, result in results.items():
        f.write(f"\n{model_name}:\n")
        for metric, value in result['metrics'].items():
            f.write(f"  {metric}: {value:.4f}\n")
    
    f.write("\nHYPERPARAMETERS\n")
    f.write("-"*40 + "\n")
    for model_name, cv_result in cv_results.items():
        f.write(f"\n{model_name}:\n")
        for param, value in cv_result['best_params'].items():
            f.write(f"  {param}: {value}\n")
    
    f.write(f"\nBEST MODEL: {best_model_name}\n")
    f.write(f"Best F1-Score: {best_model_metrics['F1-Score']:.4f}\n")

print("✓ Report saved to 'reports/tuned_models_report.txt'")

print("\n" + "="*60)
print("✅ IMPROVED MODEL TRAINING COMPLETED!")
print("="*60)
print("\nNext steps:")
print("1. Use the best model: joblib.load('models/best_model.pkl')")
print("2. Check the reports/tuned_models_report.txt for details")
print("3. Run final_predict.py to make predictions with the improved model")
print("="*60)