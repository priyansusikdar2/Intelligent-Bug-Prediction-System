"""
Standalone Dashboard for Bug Prediction System
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.prediction import BugPredictor

class StandaloneDashboard:
    def __init__(self):
        self.predictor = BugPredictor()
        self.predictor.load_models()
        
    def create_risk_gauge(self, risk_score):
        """Create risk gauge chart"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score", 'font': {'size': 24}},
            delta={'reference': 50, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#6bcf7f'},
                    {'range': [30, 70], 'color': '#ffd93d'},
                    {'range': [70, 100], 'color': '#ff6b6b'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_score
                }
            }
        ))
        fig.update_layout(height=400)
        return fig
    
    def create_model_comparison_chart(self, predictions):
        """Create model comparison chart"""
        model_names = list(predictions.keys())
        risk_scores = [predictions[m]['risk_score'] for m in model_names]
        
        fig = go.Figure(data=[
            go.Bar(name='Risk Score', x=model_names, y=risk_scores,
                  marker_color=['#ff6b6b' if score >= 70 else '#ffd93d' if score >= 30 else '#6bcf7f' for score in risk_scores],
                  text=[f'{score:.1f}%' for score in risk_scores],
                  textposition='auto')
        ])
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     annotation_text="High Risk Threshold")
        fig.add_hline(y=30, line_dash="dash", line_color="orange",
                     annotation_text="Medium Risk Threshold")
        
        fig.update_layout(
            title="Model Risk Score Comparison",
            xaxis_title="Model",
            yaxis_title="Risk Score (%)",
            yaxis_range=[0, 100],
            height=500
        )
        
        return fig
    
    def run_prediction_page(self):
        """Run prediction page"""
        st.title("🔮 Bug Prediction")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Module Information")
            module_name = st.text_input("Module Name", "NewModule.java")
            
            st.subheader("Code Metrics")
            
            loc = st.number_input("Lines of Code (LOC)", min_value=0, value=500, step=100)
            complexity = st.number_input("Cyclomatic Complexity", min_value=1, value=10, step=1)
            branches = st.number_input("Number of Branches", min_value=0, value=15, step=5)
            effort = st.number_input("Halstead Effort", min_value=0, value=50000, step=10000)
            
            predict_button = st.button("🔮 Predict Risk", type="primary", use_container_width=True)
        
        with col2:
            if predict_button:
                with st.spinner("Analyzing code metrics..."):
                    features = {
                        'loc': loc,
                        'v(g)': complexity,
                        'branchCount': branches,
                        'e': effort,
                        'n': loc // 10,
                        'total_Op': branches * 20,
                        'total_Opnd': branches * 25,
                        'uniq_Op': branches * 2,
                        'uniq_Opnd': branches * 3,
                        'lOCode': loc,
                        'lOComment': loc * 0.2,
                        'lOBlank': loc * 0.1,
                        'ev(g)': complexity * 0.6,
                        'iv(g)': complexity * 0.4,
                        'v': loc * complexity,
                        'l': 1/complexity if complexity > 0 else 1,
                        'd': complexity * 2,
                        'i': 1/(loc/complexity) if loc > 0 else 0,
                        'b': loc/3000,
                        't': loc/100
                    }
                    
                    result = self.predictor.predict_single(module_name, features)
                    
                    if 'error' not in result:
                        # Display results
                        consensus = result['consensus']
                        
                        # Risk gauge
                        gauge = self.create_risk_gauge(consensus['risk_score'])
                        st.plotly_chart(gauge, use_container_width=True)
                        
                        # Risk level indicator
                        if consensus['risk_level'] == 'HIGH':
                            st.error(f"🚨 {consensus['risk_level']} RISK - Immediate testing required!")
                        elif consensus['risk_level'] == 'MEDIUM':
                            st.warning(f"⚠️ {consensus['risk_level']} RISK - Schedule testing")
                        else:
                            st.success(f"✅ {consensus['risk_level']} RISK - Routine QA")
                        
                        # Model comparison
                        comparison = self.create_model_comparison_chart(result['predictions'])
                        st.plotly_chart(comparison, use_container_width=True)
                        
                        # Save results
                        self.predictor.save_prediction_image(result)
                        st.info("💾 Prediction result saved to reports folder")
    
    def run_analytics_page(self):
        """Run analytics page"""
        st.title("📊 Analytics Dashboard")
        
        if os.path.exists('reports/risk_data.csv'):
            risk_df = pd.read_csv('reports/risk_data.csv')
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Modules", len(risk_df))
            with col2:
                high_risk = len(risk_df[risk_df['Risk_Level'] == 'High'])
                st.metric("High Risk", high_risk, delta=f"{(high_risk/len(risk_df))*100:.1f}%")
            with col3:
                medium_risk = len(risk_df[risk_df['Risk_Level'] == 'Medium'])
                st.metric("Medium Risk", medium_risk)
            with col4:
                avg_risk = risk_df['Risk_Score'].mean()
                st.metric("Avg Risk Score", f"{avg_risk:.1f}%")
            
            # Risk distribution pie chart
            fig1 = px.pie(risk_df, names='Risk_Level', title='Risk Distribution',
                         color='Risk_Level', color_discrete_map={'High':'#ff6b6b', 'Medium':'#ffd93d', 'Low':'#6bcf7f'})
            st.plotly_chart(fig1, use_container_width=True)
            
            # Risk score histogram
            fig2 = px.histogram(risk_df, x='Risk_Score', nbins=30, title='Risk Score Distribution',
                               color_discrete_sequence=['#667eea'])
            fig2.add_vline(x=70, line_dash="dash", line_color="red", annotation_text="High Risk")
            fig2.add_vline(x=30, line_dash="dash", line_color="orange", annotation_text="Medium Risk")
            st.plotly_chart(fig2, use_container_width=True)
            
            # Top risk modules
            st.subheader("Top High-Risk Modules")
            top_risk = risk_df.nlargest(10, 'Risk_Score')[['Module', 'Risk_Score', 'Risk_Level']]
            st.dataframe(top_risk, use_container_width=True)
    
    def run_reports_page(self):
        """Run reports page"""
        st.title("📁 Reports Gallery")
        
        # List all saved images
        if os.path.exists('reports'):
            image_files = [f for f in os.listdir('reports') if f.endswith('.png')]
            
            if image_files:
                # Sort by creation time
                image_files.sort(key=lambda x: os.path.getctime(os.path.join('reports', x)), reverse=True)
                
                # Display images in grid
                cols = st.columns(3)
                for idx, img_file in enumerate(image_files):
                    with cols[idx % 3]:
                        st.image(os.path.join('reports', img_file), caption=img_file, use_container_width=True)
                        
                        if st.button(f"Delete {img_file}", key=img_file):
                            os.remove(os.path.join('reports', img_file))
                            st.rerun()
            else:
                st.info("No reports generated yet. Run some predictions first!")
    
    def run_about_page(self):
        """Run about page"""
        st.title("ℹ️ About the System")
        
        st.markdown("""
        ## Intelligent Bug Prediction System
        
        This AI-powered system analyzes software metrics to predict which modules are likely to contain bugs.
        
        ### Features
        
        - 🔮 **Real-time Prediction**: Get instant risk assessments for any module
        - 🤖 **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost
        - 📊 **Interactive Visualizations**: Charts and gauges for risk analysis
        - 📁 **Report Generation**: Save predictions as images and reports
        - 🔄 **Batch Processing**: Analyze multiple modules at once
        
        ### How It Works
        
        1. Input code metrics (LOC, complexity, branches, etc.)
        2. System preprocesses and scales the features
        3. Multiple ML models predict bug probability
        4. Consensus risk score is calculated
        5. Results are displayed and saved as reports
        
        ### Models Used
        
        - **Logistic Regression**: Baseline classifier, interpretable
        - **Random Forest**: Ensemble method, handles overfitting
        - **XGBoost**: Gradient boosting, high accuracy
        
        ### Technologies
        
        - Python
        - Scikit-learn
        - XGBoost
        - Streamlit
        - Plotly
        - Matplotlib
        
        ### Reports Location
        
        All generated reports and images are saved in the `reports/` folder.
        """)
    
    def run(self):
        """Main run method"""
        st.set_page_config(
            page_title="Bug Prediction Dashboard",
            page_icon="🐛",
            layout="wide"
        )
        
        st.sidebar.title("🐛 Bug Prediction System")
        st.sidebar.markdown("---")
        
        page = st.sidebar.radio(
            "Navigation",
            ["🔮 Predict", "📊 Analytics", "📁 Reports", "ℹ️ About"]
        )
        
        if page == "🔮 Predict":
            self.run_prediction_page()
        elif page == "📊 Analytics":
            self.run_analytics_page()
        elif page == "📁 Reports":
            self.run_reports_page()
        elif page == "ℹ️ About":
            self.run_about_page()

# Main execution
if __name__ == "__main__":
    dashboard = StandaloneDashboard()
    dashboard.run()