"""
Simple website monitor using collected metrics
"""

import json
import pandas as pd
from datetime import datetime
import os

def analyze_metrics():
    """Analyze collected page metrics"""
    
    metrics_file = 'reports/page_metrics.json'
    if not os.path.exists(metrics_file):
        print("❌ No metrics found. Run selenium_testing.py first!")
        return
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    print("="*60)
    print("📊 WEBSITE METRICS ANALYSIS")
    print("="*60)
    
    for page in metrics:
        print(f"\n📄 {page['module_name']}")
        print(f"   URL: {page['url']}")
        print(f"   Title: {page.get('title', 'N/A')}")
        print(f"   Lines of Code: {page['loc']}")
        print(f"   Complexity: {page['v(g)']:.1f}")
        print(f"   JS Errors: {page['js_errors']}")
        print(f"   Load Time: {page['load_time_ms']}ms")
        
        # Risk assessment
        risk_score = (page['v(g)'] * 0.4) + (page['js_errors'] * 10)
        risk_level = "HIGH" if risk_score > 40 else "MEDIUM" if risk_score > 20 else "LOW"
        
        print(f"   Risk Score: {risk_score:.1f}")
        print(f"   Risk Level: {risk_level}")
        
        if risk_level == "HIGH":
            print("   ⚠️  ACTION: Immediate testing recommended!")
        elif risk_level == "MEDIUM":
            print("   📋 ACTION: Schedule for review")
        else:
            print("   ✓ ACTION: Routine monitoring")

if __name__ == "__main__":
    analyze_metrics()