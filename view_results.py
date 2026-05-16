import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

print("="*60)
print("BUG PREDICTION SYSTEM - RESULTS VIEWER")
print("="*60)

# Load risk data
risk_df = pd.read_csv('reports/risk_data.csv')
print(f"\n📊 Total modules analyzed: {len(risk_df)}")
print(f"\nRisk Distribution:")
print(risk_df['Risk_Level'].value_counts())

# Load evaluation report
with open('reports/evaluation_report.txt', 'r') as f:
    print("\n📈 Evaluation Summary:")
    print(f.read()[:500])  # First 500 chars

# Display top 10 highest risk modules
print("\n🔥 TOP 10 HIGHEST RISK MODULES:")
top_10 = risk_df.nlargest(10, 'Risk_Score')
for idx, row in top_10.iterrows():
    print(f"  {row['Risk_Rank']}. {row['Module']} - {row['Risk_Score']:.1f}% ({row['Risk_Level']})")

# Show where to find visualizations
print("\n📁 Visualizations saved in 'reports/' folder:")
for file in os.listdir('reports'):
    if file.endswith('.png'):
        print(f"  - {file}")