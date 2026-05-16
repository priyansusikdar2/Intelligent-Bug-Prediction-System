"""
Final working prediction script that handles the feature mismatch correctly
"""

import pandas as pd
import joblib
import numpy as np
import os

print("="*60)
print("FINAL BUG PREDICTION SYSTEM")
print("="*60)

# Load model and scaler
model = joblib.load('models/logistic_regression.pkl')
scaler = joblib.load('models/scaler.pkl')

print(f"\n📊 Model expects: {model.n_features_in_} features")
print(f"📊 Scaler expects: {scaler.n_features_in_} features")

# Load the selected features (20 features used by model)
selected_features = []
with open('models/selected_features.txt', 'r') as f:
    selected_features = [line.strip() for line in f.readlines()]

print(f"\n✅ Model features ({len(selected_features)}):")
for i, feat in enumerate(selected_features[:10], 1):
    print(f"   {i}. {feat}")
if len(selected_features) > 10:
    print(f"   ... and {len(selected_features)-10} more")

# The scaler was trained on ALL 21 features (including locCodeAndComment)
# We need to provide all 21 features to the scaler, then select 20 for the model

# All features in original dataset
all_features = ['loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e', 
                'b', 't', 'lOCode', 'lOComment', 'lOBlank', 'locCodeAndComment', 
                'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount']

print(f"\n📋 Scaler expects all {len(all_features)} features")

def predict_module(module_name, metrics):
    """
    Predict bug probability for a module
    
    Parameters:
    module_name: Name of the module/file
    metrics: Dictionary with metric values (can be partial)
    """
    
    # Create complete feature vector with all 21 features
    complete_features = {}
    
    # Fill in provided metrics
    for feature in all_features:
        if feature in metrics:
            complete_features[feature] = metrics[feature]
        else:
            # Provide intelligent defaults based on feature type
            if feature in ['loc', 'n', 'v', 'e', 'total_Op', 'total_Opnd', 'lOCode']:
                complete_features[feature] = 100  # Medium size
            elif feature in ['v(g)', 'ev(g)', 'iv(g)', 'branchCount', 'uniq_Op', 'uniq_Opnd']:
                complete_features[feature] = 10  # Low complexity
            elif feature in ['l', 'd', 'i', 'b', 't']:
                complete_features[feature] = 0.5  # Medium value
            elif feature == 'lOComment':
                complete_features[feature] = 20  # Some comments
            elif feature == 'lOBlank':
                complete_features[feature] = 10  # Some blank lines
            else:
                complete_features[feature] = 0
    
    # Create DataFrame with all features in correct order
    X_full = pd.DataFrame([complete_features])[all_features]
    
    # Scale using the scaler (expects all 21 features)
    X_scaled_full = scaler.transform(X_full)
    
    # Select only the 20 features that the model expects
    # We need to map feature names to indices
    feature_to_idx = {feature: idx for idx, feature in enumerate(all_features)}
    selected_indices = [feature_to_idx[feature] for feature in selected_features]
    X_model_input = X_scaled_full[:, selected_indices]
    
    # Predict
    pred = model.predict(X_model_input)[0]
    prob = model.predict_proba(X_model_input)[0, 1]
    
    return pred, prob

# Modules to test
modules_to_test = [
    {
        'name': 'PaymentService.java',
        'description': 'High complexity payment processing',
        'metrics': {
            'loc': 1500, 'v(g)': 25, 'ev(g)': 15, 'iv(g)': 10,
            'branchCount': 30, 'e': 160000, 'n': 200, 'v': 8000,
            'total_Op': 800, 'total_Opnd': 900, 'uniq_Op': 40, 'uniq_Opnd': 60,
            'lOCode': 1200, 'lOComment': 200, 'lOBlank': 100
        }
    },
    {
        'name': 'UserAuth.java',
        'description': 'Simple authentication module',
        'metrics': {
            'loc': 500, 'v(g)': 8, 'ev(g)': 5, 'iv(g)': 3,
            'branchCount': 10, 'e': 16000, 'n': 80, 'v': 2000,
            'total_Op': 300, 'total_Opnd': 350, 'uniq_Op': 25, 'uniq_Opnd': 30,
            'lOCode': 400, 'lOComment': 80, 'lOBlank': 20
        }
    },
    {
        'name': 'DatabaseHandler.java',
        'description': 'Complex database operations',
        'metrics': {
            'loc': 3000, 'v(g)': 45, 'ev(g)': 30, 'iv(g)': 15,
            'branchCount': 50, 'e': 525000, 'n': 400, 'v': 15000,
            'total_Op': 1500, 'total_Opnd': 1600, 'uniq_Op': 60, 'uniq_Opnd': 90,
            'lOCode': 2500, 'lOComment': 400, 'lOBlank': 100
        }
    },
    {
        'name': 'ReportGenerator.java',
        'description': 'Medium complexity reporting',
        'metrics': {
            'loc': 800, 'v(g)': 12, 'ev(g)': 8, 'iv(g)': 4,
            'branchCount': 15, 'e': 35000, 'n': 120, 'v': 3500,
            'total_Op': 450, 'total_Opnd': 500, 'uniq_Op': 30, 'uniq_Opnd': 45,
            'lOCode': 600, 'lOComment': 150, 'lOBlank': 50
        }
    },
    {
        'name': 'CacheManager.java',
        'description': 'Caching with moderate complexity',
        'metrics': {
            'loc': 2000, 'v(g)': 35, 'ev(g)': 20, 'iv(g)': 12,
            'branchCount': 40, 'e': 280000, 'n': 300, 'v': 10000,
            'total_Op': 1200, 'total_Opnd': 1300, 'uniq_Op': 50, 'uniq_Opnd': 75,
            'lOCode': 1600, 'lOComment': 300, 'lOBlank': 100
        }
    },
    {
        'name': 'SimpleUtil.java',
        'description': 'Simple utility functions',
        'metrics': {
            'loc': 100, 'v(g)': 3, 'ev(g)': 2, 'iv(g)': 1,
            'branchCount': 2, 'e': 1500, 'n': 30, 'v': 500,
            'total_Op': 100, 'total_Opnd': 120, 'uniq_Op': 15, 'uniq_Opnd': 20,
            'lOCode': 80, 'lOComment': 15, 'lOBlank': 5
        }
    }
]

# Make predictions
print("\n" + "="*70)
print("PREDICTION RESULTS")
print("="*70)

results = []
for module in modules_to_test:
    pred, prob = predict_module(module['name'], module['metrics'])
    results.append({
        'name': module['name'],
        'description': module['description'],
        'prediction': pred,
        'probability': prob
    })

# Display detailed results
for i, result in enumerate(results, 1):
    name = result['name']
    desc = result['description']
    prob = result['probability']
    pred = result['prediction']
    
    risk_score = prob * 100
    risk_level = "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.3 else "LOW"
    status = "BUGGY" if pred == 1 else "CLEAN"
    
    # Color coding for risk level
    if risk_level == "HIGH":
        risk_icon = "🔴"
        action = "🚨 IMMEDIATE TESTING REQUIRED!"
        priority = "CRITICAL"
    elif risk_level == "MEDIUM":
        risk_icon = "🟡"
        action = "📋 Schedule for next testing cycle"
        priority = "MEDIUM"
    else:
        risk_icon = "🟢"
        action = "✓ Routine QA process"
        priority = "LOW"
    
    print(f"\n{i}. 📦 {name}")
    print(f"   📝 {desc}")
    print(f"   {'─'*55}")
    print(f"   Prediction: {status} {'🐛' if pred == 1 else '✅'}")
    print(f"   Risk Score: {risk_score:.1f}%")
    print(f"   Risk Level: {risk_icon} {risk_level} (Priority: {priority})")
    print(f"   Action: {action}")
    
    # Add recommendation based on risk level
    if risk_level == "HIGH":
        print(f"   💡 Review: Check cyclomatic complexity and branching logic")
    elif risk_level == "MEDIUM":
        print(f"   💡 Review: Consider code simplification and adding tests")

# Summary statistics
print("\n" + "="*70)
print("SUMMARY REPORT")
print("="*70)

buggy_count = sum(r['prediction'] for r in results)
high_risk = sum(r['probability'] >= 0.7 for r in results)
medium_risk = sum((r['probability'] >= 0.3) & (r['probability'] < 0.7) for r in results)
low_risk = sum(r['probability'] < 0.3 for r in results)

print(f"\n📊 Total Modules Analyzed: {len(results)}")
print(f"🐛 Buggy Modules Detected: {buggy_count}/{len(results)}")
print(f"\nRisk Distribution:")
print(f"   🔴 High Risk: {high_risk} modules ({high_risk/len(results)*100:.1f}%)")
print(f"   🟡 Medium Risk: {medium_risk} modules ({medium_risk/len(results)*100:.1f}%)")
print(f"   🟢 Low Risk: {low_risk} modules ({low_risk/len(results)*100:.1f}%)")

print(f"\n💡 RECOMMENDATIONS:")
if high_risk > 0:
    print(f"   • 🚨 IMMEDIATE ACTION: Test {high_risk} high-risk module(s) first")
    high_risk_modules = [r['name'] for r in results if r['probability'] >= 0.7]
    for module in high_risk_modules:
        print(f"     - {module}")
if medium_risk > 0:
    print(f"   • 📅 Schedule comprehensive testing for {medium_risk} medium-risk module(s)")
if buggy_count > 0:
    print(f"   • 🔍 Perform code review on buggy modules")
    print(f"   • 📈 Track these modules in future sprints")

# Testing priority order
print(f"\n📋 TESTING PRIORITY ORDER:")
sorted_results = sorted(results, key=lambda x: x['probability'], reverse=True)
for i, result in enumerate(sorted_results, 1):
    priority = "🔴 HIGH" if result['probability'] >= 0.7 else "🟡 MEDIUM" if result['probability'] >= 0.3 else "🟢 LOW"
    print(f"   {i}. {result['name']} - Risk: {result['probability']*100:.1f}% [{priority}]")

print("\n" + "="*70)
print("✅ Prediction completed successfully!")
print("💾 Results can be saved to CSV for reporting")
print("="*70)

# Optional: Save results to CSV
save_to_csv = input("\n💾 Save results to CSV? (y/n): ").lower()
if save_to_csv == 'y':
    import csv
    with open('reports/predictions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Module', 'Description', 'Bug Probability', 'Risk Level', 'Prediction'])
        for r in results:
            writer.writerow([
                r['name'], 
                r['description'], 
                f"{r['probability']*100:.1f}%",
                "HIGH" if r['probability'] >= 0.7 else "MEDIUM" if r['probability'] >= 0.3 else "LOW",
                "BUGGY" if r['prediction'] == 1 else "CLEAN"
            ])
    print("✅ Results saved to 'reports/predictions.csv'")