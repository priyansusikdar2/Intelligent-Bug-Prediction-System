"""
Streamlit Dashboard for Intelligent Bug Prediction System
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import local modules
from src.data_preprocessing import DataPreprocessor
from src.feature_selection import FeatureSelector
from src.evaluate_model import ModelEvaluator
from src.risk_scoring import RiskAnalyzer

# Page configuration
st.set_page_config(
    page_title="Intelligent Bug Prediction System",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .risk-high {
        background-color: #ff6b6b;
        padding: 10px;
        border-radius: 5px;
        color: white;
        text-align: center;
    }
    .risk-medium {
        background-color: #ffd93d;
        padding: 10px;
        border-radius: 5px;
        color: #333;
        text-align: center;
    }
    .risk-low {
        background-color: #6bcf7f;
        padding: 10px;
        border-radius: 5px;
        color: white;
        text-align: center;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class StreamlitDashboard:
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
            
            st.success(f"✅ Loaded {len(self.models)} models successfully!")
        except Exception as e:
            st.warning(f"⚠️ Could not load all models: {e}")
    
    def save_plot(self, fig, filename):
        """Save plot to reports folder"""
        os.makedirs('reports', exist_ok=True)
        filepath = os.path.join('reports', filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        return filepath
    
    def show_header(self):
        """Display header"""
        st.markdown("""
        <div class="main-header">
            <h1>🐛 Intelligent Bug Prediction System</h1>
            <p>AI-Powered Software Quality Analysis Dashboard</p>
        </div>
        """, unsafe_allow_html=True)
    
    def show_sidebar(self):
        """Display sidebar with navigation"""
        st.sidebar.image("https://img.icons8.com/color/96/000000/bug.png", width=80)
        st.sidebar.title("Navigation")
        
        page = st.sidebar.radio(
            "Select Page",
            ["📊 Dashboard Overview", "🔮 Predict Module", "📈 Model Performance", "⚠️ Risk Analysis", "📁 Reports"]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.info(
            "**System Info**\n\n"
            f"- Models Loaded: {len(self.models)}\n"
            f"- Features: {len(self.selected_features) if self.selected_features else 'N/A'}\n"
            f"- Status: Active"
        )
        
        return page
    
    def dashboard_overview(self):
        """Display dashboard overview"""
        st.header("📊 Dashboard Overview")
        
        # Load dataset if available
        if os.path.exists('dataset/kc1.csv'):
            df = pd.read_csv('dataset/kc1.csv')
            
            # Metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Modules", len(df))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                if 'defects' in df.columns:
                    bug_count = df['defects'].sum()
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Buggy Modules", bug_count, f"{(bug_count/len(df))*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Features", len(df.columns)-1)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Models Active", len(self.models))
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Distribution plots
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Defect Distribution")
                if 'defects' in df.columns:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    defect_counts = df['defects'].value_counts()
                    colors = ['#6bcf7f', '#ff6b6b']
                    plt.pie(defect_counts.values, labels=['Clean', 'Buggy'], autopct='%1.1f%%', colors=colors, startangle=90)
                    plt.title('Module Defect Distribution')
                    st.pyplot(fig)
                    self.save_plot(fig, 'dashboard_defect_distribution.png')
                    plt.close()
            
            with col2:
                st.subheader("Top Features by Correlation")
                if 'defects' in df.columns:
                    correlations = df.corr()['defects'].drop('defects').abs().sort_values(ascending=False).head(10)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    correlations.plot(kind='barh', color='skyblue', ax=ax)
                    ax.set_xlabel('Correlation with Defects')
                    ax.set_title('Top 10 Feature Correlations')
                    plt.tight_layout()
                    st.pyplot(fig)
                    self.save_plot(fig, 'dashboard_feature_correlations.png')
                    plt.close()
    
    def predict_module(self):
        """Interactive module prediction"""
        st.header("🔮 Predict Module Risk")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Module Information")
            module_name = st.text_input("Module Name", "NewModule.java")
            
            st.subheader("Key Metrics")
            
            loc = st.number_input("Lines of Code (LOC)", min_value=0, value=500)
            complexity = st.number_input("Cyclomatic Complexity (v(g))", min_value=1, value=10)
            branches = st.number_input("Branch Count", min_value=0, value=15)
            effort = st.number_input("Halstead Effort", min_value=0, value=50000)
            
            predict_button = st.button("🔮 Predict Risk", type="primary")
        
        with col2:
            if predict_button:
                with st.spinner("Analyzing module..."):
                    # Prepare features
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
                    
                    # Make predictions
                    results = self.make_predictions(features)
                    
                    if results:
                        # Display results
                        consensus = results['consensus']
                        risk_level = consensus['risk_level']
                        risk_score = consensus['risk_score']
                        
                        if risk_level == 'HIGH':
                            st.markdown('<div class="risk-high">', unsafe_allow_html=True)
                            st.metric("Risk Level", "🔴 HIGH RISK", f"{risk_score:.1f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                        elif risk_level == 'MEDIUM':
                            st.markdown('<div class="risk-medium">', unsafe_allow_html=True)
                            st.metric("Risk Level", "🟡 MEDIUM RISK", f"{risk_score:.1f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="risk-low">', unsafe_allow_html=True)
                            st.metric("Risk Level", "🟢 LOW RISK", f"{risk_score:.1f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Model predictions
                        st.subheader("Model Predictions")
                        model_data = []
                        for model_name, pred in results['model_predictions'].items():
                            model_data.append({
                                'Model': model_name,
                                'Risk Score': f"{pred['risk_score']:.1f}%",
                                'Risk Level': pred['risk_level'],
                                'Action': pred['action']
                            })
                        
                        model_df = pd.DataFrame(model_data)
                        st.dataframe(model_df, use_container_width=True)
                        
                        # Save prediction result as image
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.axis('tight')
                        ax.axis('off')
                        table_data = [['Module', module_name],
                                     ['Risk Score', f"{risk_score:.1f}%"],
                                     ['Risk Level', risk_level]]
                        table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                                        cellLoc='center', loc='center')
                        table.auto_set_font_size(False)
                        table.set_fontsize(12)
                        table.scale(1, 2)
                        plt.title(f'Prediction Result - {module_name}', fontsize=14, pad=20)
                        self.save_plot(fig, f'prediction_{module_name.replace(".", "_")}.png')
                        plt.close()
    
    def make_predictions(self, features):
        """Make predictions using loaded models"""
        results = {'model_predictions': {}}
        
        # Prepare features with all required columns
        all_features = ['loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e', 
                       'b', 't', 'lOCode', 'lOComment', 'lOBlank', 'locCodeAndComment', 
                       'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount']
        
        # Create complete feature vector
        feature_vector = []
        for feature in all_features:
            if feature in features:
                feature_vector.append(features[feature])
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
        
        # Get predictions from each model
        probabilities = []
        for model_name, model in self.models.items():
            try:
                prob = model.predict_proba(model_input)[0, 1]
                probabilities.append(prob)
                
                risk_score = prob * 100
                if prob >= 0.7:
                    risk_level = "HIGH"
                    action = "🚨 Test immediately!"
                elif prob >= 0.3:
                    risk_level = "MEDIUM"
                    action = "📋 Schedule for testing"
                else:
                    risk_level = "LOW"
                    action = "✓ Routine QA"
                
                results['model_predictions'][model_name] = {
                    'bug_probability': prob,
                    'risk_score': risk_score,
                    'risk_level': risk_level,
                    'action': action
                }
            except Exception as e:
                st.warning(f"Error with {model_name}: {e}")
        
        if probabilities:
            consensus_prob = np.mean(probabilities)
            results['consensus'] = {
                'bug_probability': consensus_prob,
                'risk_score': consensus_prob * 100,
                'risk_level': "HIGH" if consensus_prob >= 0.7 else "MEDIUM" if consensus_prob >= 0.3 else "LOW"
            }
            return results
        return None
    
    def model_performance(self):
        """Display model performance comparison"""
        st.header("📈 Model Performance")
        
        # Load evaluation results
        if os.path.exists('reports/evaluation_report.txt'):
            with open('reports/evaluation_report.txt', 'r') as f:
                content = f.read()
                st.text(content)
        
        # Display saved performance plots
        col1, col2 = st.columns(2)
        
        performance_plots = [
            ('reports/model_comparison.png', 'Model Comparison'),
            ('reports/roc_curves.png', 'ROC Curves'),
            ('reports/confusion_matrix_logistic_regression.png', 'Logistic Regression'),
            ('reports/confusion_matrix_random_forest.png', 'Random Forest'),
            ('reports/confusion_matrix_xgboost.png', 'XGBoost')
        ]
        
        plot_idx = 0
        for i in range(0, len(performance_plots), 2):
            col1, col2 = st.columns(2)
            with col1:
                if plot_idx < len(performance_plots):
                    path, title = performance_plots[plot_idx]
                    if os.path.exists(path):
                        st.image(path, caption=title, use_container_width=True)
                    plot_idx += 1
            with col2:
                if plot_idx < len(performance_plots):
                    path, title = performance_plots[plot_idx]
                    if os.path.exists(path):
                        st.image(path, caption=title, use_container_width=True)
                    plot_idx += 1
    
    def risk_analysis(self):
        """Display risk analysis"""
        st.header("⚠️ Risk Analysis")
        
        if os.path.exists('reports/risk_data.csv'):
            risk_df = pd.read_csv('reports/risk_data.csv')
            
            # Risk distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Risk Level Distribution")
                risk_counts = risk_df['Risk_Level'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = {'High': '#ff6b6b', 'Medium': '#ffd93d', 'Low': '#6bcf7f'}
                risk_counts.plot(kind='pie', autopct='%1.1f%%', colors=[colors.get(x, '#ccc') for x in risk_counts.index], ax=ax)
                ax.set_ylabel('')
                ax.set_title('Risk Distribution')
                st.pyplot(fig)
                self.save_plot(fig, 'risk_analysis_distribution.png')
                plt.close()
            
            with col2:
                st.subheader("Risk Score Distribution")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(risk_df['Risk_Score'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
                ax.axvline(risk_df['Risk_Score'].mean(), color='red', linestyle='dashed', linewidth=2, label=f"Mean: {risk_df['Risk_Score'].mean():.1f}")
                ax.set_xlabel('Risk Score (%)')
                ax.set_ylabel('Number of Modules')
                ax.set_title('Risk Score Distribution')
                ax.legend()
                st.pyplot(fig)
                self.save_plot(fig, 'risk_analysis_score_distribution.png')
                plt.close()
            
            # Top high-risk modules
            st.subheader("Top High-Risk Modules")
            high_risk = risk_df[risk_df['Risk_Level'] == 'High'].sort_values('Risk_Score', ascending=False).head(10)
            st.dataframe(high_risk[['Module', 'Risk_Score', 'Risk_Level']], use_container_width=True)
            
            # Risk heatmap
            if len(risk_df) > 10:
                st.subheader("Risk Heatmap")
                fig, ax = plt.subplots(figsize=(12, 8))
                heatmap_data = risk_df.nlargest(20, 'Risk_Score')[['Risk_Score']].values
                sns.heatmap(heatmap_data.T, cmap='RdYlGn_r', cbar_kws={'label': 'Risk Score (%)'},
                           xticklabels=[m[:20] for m in risk_df.nlargest(20, 'Risk_Score')['Module']],
                           yticklabels=['Risk Score'], ax=ax)
                ax.set_title('Top 20 High-Risk Modules Heatmap')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                self.save_plot(fig, 'risk_analysis_heatmap.png')
                plt.close()
    
    def reports_section(self):
        """Display saved reports"""
        st.header("📁 Generated Reports")
        
        report_files = {
            'Model Performance': 'reports/evaluation_report.txt',
            'Risk Report': 'reports/risk_report.txt',
            'Training Report': 'reports/training_report.txt',
            'Feature Selection': 'reports/feature_selection_report.txt'
        }
        
        for name, path in report_files.items():
            if os.path.exists(path):
                with st.expander(f"📄 {name}"):
                    with open(path, 'r') as f:
                        st.code(f.read(), language='text')
        
        # Image gallery
        st.subheader("Visualization Gallery")
        
        image_files = []
        if os.path.exists('reports'):
            for file in os.listdir('reports'):
                if file.endswith('.png'):
                    image_files.append(file)
        
        if image_files:
            cols = st.columns(3)
            for idx, img_file in enumerate(image_files[:9]):
                with cols[idx % 3]:
                    st.image(os.path.join('reports', img_file), caption=img_file, use_container_width=True)
    
    def run(self):
        """Main run method"""
        self.show_header()
        page = self.show_sidebar()
        
        if page == "📊 Dashboard Overview":
            self.dashboard_overview()
        elif page == "🔮 Predict Module":
            self.predict_module()
        elif page == "📈 Model Performance":
            self.model_performance()
        elif page == "⚠️ Risk Analysis":
            self.risk_analysis()
        elif page == "📁 Reports":
            self.reports_section()

# Main execution
if __name__ == "__main__":
    dashboard = StreamlitDashboard()
    dashboard.run()