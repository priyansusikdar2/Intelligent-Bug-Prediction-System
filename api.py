"""
Flask API for Intelligent Bug Prediction System
"""

from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import numpy as np
import os
import traceback

app = Flask(__name__)

# HTML template for the home page
HOME_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Bug Prediction API</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .endpoint {
            background: #e9ecef;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
        }
        pre {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }
        .status {
            color: #28a745;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>🤖 Intelligent Bug Prediction System API</h1>
    
    <div class="card">
        <h2>📊 API Status</h2>
        <p>Status: <span class="status">✓ Running</span></p>
        <p>Models loaded: {{ models|join(', ') }}</p>
    </div>
    
    <div class="card">
        <h2>📡 Available Endpoints</h2>
        
        <h3>1. Health Check</h3>
        <div class="endpoint">GET /health</div>
        <p>Check if the API is running.</p>
        
        <h3>2. Make Predictions</h3>
        <div class="endpoint">POST /predict</div>
        <p>Predict bug probability for a software module.</p>
        
        <h4>Example Request:</h4>
        <pre>{
  "module_name": "PaymentService.java",
  "features": {
    "loc": 1500,
    "v(g)": 25,
    "ev(g)": 15,
    "iv(g)": 10,
    "branchCount": 30,
    "e": 160000,
    "n": 200,
    "v": 8000,
    "total_Op": 800,
    "total_Opnd": 900,
    "uniq_Op": 40,
    "uniq_Opnd": 60,
    "lOCode": 1200,
    "lOComment": 200,
    "lOBlank": 100
  }
}</pre>
        
        <h3>3. Batch Predictions</h3>
        <div class="endpoint">POST /predict/batch</div>
        <p>Predict for multiple modules at once.</p>
    </div>
    
    <div class="card">
        <h2>🖥️ Test with cURL</h2>
        <pre># Health check
curl http://localhost:5000/health

# Make a prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "module_name": "TestModule.java",
    "features": {"loc": 1000, "v(g)": 20, "branchCount": 25}
  }'</pre>
    </div>
</body>
</html>
'''

# Load models with error handling
print("="*60)
print("🚀 Starting Bug Prediction API Server")
print("="*60)

# Load all required features
all_features = ['loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e', 
                'b', 't', 'lOCode', 'lOComment', 'lOBlank', 'locCodeAndComment', 
                'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount']

# Load selected features (20 features used by model)
try:
    with open('models/selected_features.txt', 'r') as f:
        selected_features = [line.strip() for line in f.readlines()]
    print(f"✓ Loaded {len(selected_features)} selected features")
except:
    # Fallback: exclude locCodeAndComment which was dropped
    selected_features = [f for f in all_features if f != 'locCodeAndComment']
    print(f"⚠️ Using fallback features: {len(selected_features)}")

print(f"✓ Model features: {len(selected_features)}")
print(f"✓ Scaler features: {len(all_features)}")

# Load models
models = {}
try:
    models['logistic'] = joblib.load('models/logistic_regression.pkl')
    print("✓ Loaded Logistic Regression model")
except Exception as e:
    print(f"⚠️ Could not load Logistic Regression: {e}")

try:
    models['random_forest'] = joblib.load('models/random_forest.pkl')
    print("✓ Loaded Random Forest model")
except Exception as e:
    print(f"⚠️ Could not load Random Forest: {e}")

try:
    models['xgboost'] = joblib.load('models/xgboost.pkl')
    print("✓ Loaded XGBoost model")
except Exception as e:
    print(f"⚠️ Could not load XGBoost: {e}")

# Load scaler
try:
    scaler = joblib.load('models/scaler.pkl')
    print("✓ Loaded scaler")
except Exception as e:
    print(f"❌ Could not load scaler: {e}")
    scaler = None

def prepare_features(features_dict):
    """
    Prepare features for prediction
    Handles both partial feature dicts and missing values
    """
    # Create complete feature vector with all 21 features
    complete_features = {}
    
    for feature in all_features:
        if feature in features_dict:
            complete_features[feature] = features_dict[feature]
        else:
            # Provide intelligent defaults
            if feature in ['loc', 'n', 'v', 'e', 'total_Op', 'total_Opnd', 'lOCode']:
                complete_features[feature] = 100
            elif feature in ['v(g)', 'ev(g)', 'iv(g)', 'branchCount', 'uniq_Op', 'uniq_Opnd']:
                complete_features[feature] = 10
            elif feature in ['l', 'd', 'i', 'b', 't']:
                complete_features[feature] = 0.5
            elif feature == 'lOComment':
                complete_features[feature] = 20
            elif feature == 'lOBlank':
                complete_features[feature] = 10
            else:
                complete_features[feature] = 0
    
    # Create DataFrame
    X_full = pd.DataFrame([complete_features])[all_features]
    
    # Scale
    if scaler:
        X_scaled_full = scaler.transform(X_full)
    else:
        X_scaled_full = X_full.values
    
    # Select only the features the model expects
    feature_to_idx = {feature: idx for idx, feature in enumerate(all_features)}
    selected_indices = [feature_to_idx[feature] for feature in selected_features]
    X_model_input = X_scaled_full[:, selected_indices]
    
    return X_model_input

@app.route('/')
def home():
    """Home page with API documentation"""
    return render_template_string(HOME_TEMPLATE, models=list(models.keys()))

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models': list(models.keys()),
        'features_expected': len(selected_features),
        'scaler_loaded': scaler is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict bug probability for a module"""
    try:
        # Get request data
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        module_name = data.get('module_name', 'Unknown')
        features = data.get('features', {})
        
        if not features:
            return jsonify({'error': 'No features provided'}), 400
        
        # Prepare features
        X_prepared = prepare_features(features)
        
        # Make predictions with each model
        results = {}
        for name, model in models.items():
            try:
                prob = model.predict_proba(X_prepared)[0, 1]
                risk_score = prob * 100
                
                if prob >= 0.7:
                    risk_level = "HIGH"
                    priority = "CRITICAL"
                    action = "Test immediately!"
                elif prob >= 0.3:
                    risk_level = "MEDIUM"
                    priority = "NORMAL"
                    action = "Schedule for testing"
                else:
                    risk_level = "LOW"
                    priority = "LOW"
                    action = "Routine QA"
                
                results[name] = {
                    'bug_probability': float(prob),
                    'risk_score': float(risk_score),
                    'risk_level': risk_level,
                    'priority': priority,
                    'action': action
                }
            except Exception as e:
                results[name] = {'error': str(e)}
        
        # Get consensus (average of all models)
        valid_probs = [r['bug_probability'] for r in results.values() if 'bug_probability' in r]
        if valid_probs:
            consensus_prob = np.mean(valid_probs)
            consensus_risk = "HIGH" if consensus_prob >= 0.7 else "MEDIUM" if consensus_prob >= 0.3 else "LOW"
        else:
            consensus_prob = 0
            consensus_risk = "UNKNOWN"
        
        return jsonify({
            'module_name': module_name,
            'consensus': {
                'bug_probability': float(consensus_prob),
                'risk_score': float(consensus_prob * 100),
                'risk_level': consensus_risk
            },
            'model_predictions': results,
            'features_used': len(selected_features)
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 400

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Predict for multiple modules at once"""
    try:
        data = request.json
        modules = data.get('modules', [])
        
        if not modules:
            return jsonify({'error': 'No modules provided'}), 400
        
        results = []
        for module in modules:
            module_name = module.get('module_name', 'Unknown')
            features = module.get('features', {})
            
            X_prepared = prepare_features(features)
            
            predictions = {}
            for name, model in models.items():
                try:
                    prob = model.predict_proba(X_prepared)[0, 1]
                    predictions[name] = float(prob)
                except:
                    predictions[name] = None
            
            # Calculate consensus
            valid_probs = [p for p in predictions.values() if p is not None]
            consensus = np.mean(valid_probs) if valid_probs else 0
            
            results.append({
                'module_name': module_name,
                'consensus_probability': float(consensus),
                'risk_level': "HIGH" if consensus >= 0.7 else "MEDIUM" if consensus >= 0.3 else "LOW",
                'model_predictions': predictions
            })
        
        return jsonify({
            'total_modules': len(results),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/features', methods=['GET'])
def get_features():
    """Get list of expected features"""
    return jsonify({
        'scaler_features': all_features,
        'model_features': selected_features,
        'total_scaler_features': len(all_features),
        'total_model_features': len(selected_features)
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("✅ API Server Ready!")
    print("="*60)
    print("\n📍 Available URLs:")
    print("   • Home: http://localhost:5000/")
    print("   • Health: http://localhost:5000/health")
    print("   • Predict: http://localhost:5000/predict (POST)")
    print("   • Batch Predict: http://localhost:5000/predict/batch (POST)")
    print("   • Features: http://localhost:5000/features (GET)")
    print("\n💡 Quick Test:")
    print("   Open browser and go to http://localhost:5000/")
    print("\n" + "="*60)
    print("🚀 Starting server...")
    print("Press CTRL+C to stop")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)