"""
Advanced API Client for Bug Prediction System
"""

import requests
import json
import pandas as pd
from datetime import datetime

class BugPredictionClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def check_health(self):
        """Check if API is healthy"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def predict_single(self, module_name, features):
        """Predict for a single module"""
        payload = {
            "module_name": module_name,
            "features": features
        }
        response = self.session.post(
            f"{self.base_url}/predict",
            json=payload
        )
        return response.json()
    
    def predict_batch(self, modules):
        """Predict for multiple modules"""
        payload = {"modules": modules}
        response = self.session.post(
            f"{self.base_url}/predict/batch",
            json=payload
        )
        return response.json()
    
    def get_features_info(self):
        """Get information about expected features"""
        response = self.session.get(f"{self.base_url}/features")
        return response.json()

# Example usage
if __name__ == "__main__":
    client = BugPredictionClient()
    
    print("="*60)
    print("BUG PREDICTION API CLIENT")
    print("="*60)
    
    # Check health
    print("\n1. Checking API health...")
    health = client.check_health()
    print(f"   Status: {health['status']}")
    print(f"   Models: {', '.join(health['models'])}")
    
    # Predict multiple modules
    print("\n2. Predicting modules...")
    
    modules = [
        {
            "module_name": "PaymentService.java",
            "features": {
                "loc": 1500, "v(g)": 25, "branchCount": 30,
                "e": 160000, "n": 200, "total_Op": 800
            }
        },
        {
            "module_name": "UserAuth.java",
            "features": {
                "loc": 500, "v(g)": 8, "branchCount": 10,
                "e": 16000, "n": 80, "total_Op": 300
            }
        },
        {
            "module_name": "DatabaseHandler.java",
            "features": {
                "loc": 3000, "v(g)": 45, "branchCount": 50,
                "e": 525000, "n": 400, "total_Op": 1500
            }
        },
        {
            "module_name": "ReportGenerator.java",
            "features": {
                "loc": 800, "v(g)": 12, "branchCount": 15,
                "e": 35000, "n": 120, "total_Op": 450
            }
        },
        {
            "module_name": "CacheManager.java",
            "features": {
                "loc": 2000, "v(g)": 35, "branchCount": 40,
                "e": 280000, "n": 300, "total_Op": 1200
            }
        }
    ]
    
    results = client.predict_batch(modules)
    
    # Create results DataFrame
    results_data = []
    for result in results['results']:
        results_data.append({
            'Module': result['module_name'],
            'Risk Level': result['risk_level'],
            'Probability': f"{result['consensus_probability']*100:.1f}%",
            'Priority': 'HIGH' if result['risk_level'] == 'HIGH' else 'MEDIUM' if result['risk_level'] == 'MEDIUM' else 'LOW'
        })
    
    df_results = pd.DataFrame(results_data)
    df_results = df_results.sort_values('Probability', ascending=False)
    
    print("\n📊 Prediction Results:")
    print(df_results.to_string(index=False))
    
    # Recommendations
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS")
    print("="*60)
    
    high_risk = df_results[df_results['Risk Level'] == 'HIGH']
    medium_risk = df_results[df_results['Risk Level'] == 'MEDIUM']
    
    if len(high_risk) > 0:
        print(f"\n🚨 CRITICAL - Test these modules immediately:")
        for _, row in high_risk.iterrows():
            print(f"   • {row['Module']} ({row['Probability']})")
    
    if len(medium_risk) > 0:
        print(f"\n📋 MEDIUM PRIORITY - Schedule for testing:")
        for _, row in medium_risk.iterrows():
            print(f"   • {row['Module']} ({row['Probability']})")
    
    print("\n✅ Analysis complete!")