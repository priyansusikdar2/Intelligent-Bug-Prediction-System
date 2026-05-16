"""
Test client for the Bug Prediction API
"""

import requests
import json

# API endpoint
API_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

def test_predict():
    """Test prediction endpoint"""
    print("Testing prediction endpoint...")
    
    # Test module
    test_data = {
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
    }
    
    response = requests.post(
        f"{API_URL}/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Module: {result['module_name']}")
        print(f"Consensus Risk: {result['consensus']['risk_level']}")
        print(f"Risk Score: {result['consensus']['risk_score']:.1f}%")
        print(f"\nModel Predictions:")
        for model, pred in result['model_predictions'].items():
            if 'bug_probability' in pred:
                print(f"  {model}: {pred['bug_probability']*100:.1f}% ({pred['risk_level']})")
    else:
        print(f"Error: {response.text}")
    print()

def test_batch_predict():
    """Test batch prediction endpoint"""
    print("Testing batch prediction endpoint...")
    
    test_data = {
        "modules": [
            {
                "module_name": "PaymentService.java",
                "features": {"loc": 1500, "v(g)": 25, "branchCount": 30}
            },
            {
                "module_name": "SimpleUtil.java",
                "features": {"loc": 100, "v(g)": 3, "branchCount": 2}
            },
            {
                "module_name": "DatabaseHandler.java",
                "features": {"loc": 3000, "v(g)": 45, "branchCount": 50}
            }
        ]
    }
    
    response = requests.post(
        f"{API_URL}/predict/batch",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Total modules: {result['total_modules']}")
        for module_result in result['results']:
            print(f"\n  {module_result['module_name']}:")
            print(f"    Risk: {module_result['risk_level']}")
            print(f"    Probability: {module_result['consensus_probability']*100:.1f}%")
    else:
        print(f"Error: {response.text}")
    print()

if __name__ == "__main__":
    print("="*60)
    print("API TEST CLIENT")
    print("="*60)
    print("\nMake sure the API server is running:")
    print("  python api.py")
    print("\n" + "="*60 + "\n")
    
    try:
        test_health()
        test_predict()
        test_batch_predict()
        print("✅ All tests completed!")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server!")
        print("Please start the server first: python api.py")