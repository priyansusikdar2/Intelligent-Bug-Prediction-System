# 🐛 Intelligent Bug Prediction System

An AI-powered software quality analysis application that predicts bug-prone modules using machine learning, helping QA teams focus testing efforts on high-risk areas.

---

# 📋 Table of Contents

- [Overview]
- [Features]
- [Technologies Used]
- [Project Structure]
- [Installation]
- [Usage Guide]
- [Selenium Web Testing Integration]
- [API Documentation]
- [Model Performance]
- [Risk Scoring]
- [Reports & Visualizations]
- [Troubleshooting]
- [Future Enhancements]

---

# 📖 Overview

The Intelligent Bug Prediction System analyzes historical software metrics and defect datasets to identify modules likely to be affected by bugs. By leveraging machine learning algorithms, it generates risk scores for software modules, enabling development teams to prioritize testing activities effectively.

## 🔄 How It Works

1. **Data Collection** - Load historical software metrics (NASA PROMISE dataset)
2. **Preprocessing** - Clean, scale, and prepare data for analysis
3. **Feature Selection** - Identify most important software metrics
4. **Model Training** - Train multiple ML models on historical data
5. **Prediction** - Analyze new modules for bug probability
6. **Risk Scoring** - Generate risk levels (Low/Medium/High)
7. **Reporting** - Create visualizations and comprehensive reports
8. **Web Testing** - Automate web application testing with Selenium

---

# ✨ Features

## 🚀 Core Features

- 🔮 **Real-time Bug Prediction** - Predict bug probability for any software module
- 🤖 **Multiple ML Models** - Logistic Regression, Random Forest, XGBoost
- 📊 **Interactive Dashboard** - Streamlit-based web interface
- 📈 **Model Performance Comparison** - Visual comparison of all models
- ⚠️ **Risk Scoring System** - Low, Medium, High risk classification
- 📁 **Automatic Report Generation** - Save all results as images and text files
- 🔄 **Batch Processing** - Analyze multiple modules simultaneously
- 🎯 **Feature Importance Analysis** - Identify key metrics affecting bugs

## 🌐 Selenium Web Testing Features

- 🌐 **Automated Website Testing** - Test web applications for bugs
- 📊 **Real-time Metrics Collection** - Gather page complexity, load times, JS errors
- 🔍 **DOM Analysis** - Count elements, forms, links, and dynamic components
- ⏱️ **Performance Monitoring** - Track page load performance
- 🔒 **Security Header Testing** - Verify HTTPS and security configurations
- 📱 **Responsive Design Testing** - Test across different screen sizes
- 📈 **Visual Regression** - Compare screenshots for visual bugs
- 🤖 **Automated Test Generation** - Create tests based on risk predictions

## 📊 Visualization Capabilities

- Confusion matrices for each model
- ROC curves comparison
- Feature importance plots
- Risk distribution charts
- Model performance comparison graphs
- Risk heatmaps
- Website performance dashboards

---

# 🛠 Technologies Used

| Category | Technologies |
|----------|-------------|
| **Programming Language** | Python 3.11+ |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Dashboard** | Streamlit |
| **API Framework** | Flask |
| **Web Testing** | Selenium, WebDriver Manager |
| **Model Serialization** | Joblib |
| **Dataset** | NASA PROMISE Repository |

---

# 📁 Project Structure

```text
Intelligent-Bug-Prediction-System/
│
├── app/
│   ├── streamlit_app.py
│   ├── prediction.py
│   └── dashboard.py
│
├── dataset/
│   ├── kc1.csv
│   ├── pc1.csv
│   └── cleaned_dataset.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   └── selected_features.txt
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_selection.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── risk_scoring.py
│   └── utils.py
│
├── reports/
│   ├── confusion_matrix_*.png
│   ├── roc_curves.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   ├── risk_distribution.png
│   ├── selenium_report_*.txt
│   ├── page_metrics.json
│   ├── performance_data.json
│   ├── evaluation_report.txt
│   ├── risk_report.txt
│   └── predictions.csv
│
│
├── selenium_testing.py
├── web_test_suite.py
├── monitor_websites.py
├── test_selenium_simple.py
├── api.py
├── main.py
├── requirements.txt
└── README.md
```

---

# 💻 Installation

## 📌 Prerequisites

- Python 3.11 or higher
- pip package manager
- Google Chrome browser (for Selenium testing)

## ⚡ Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Intelligent-Bug-Prediction-System.git

# 2. Navigate to project folder
cd Intelligent-Bug-Prediction-System

# 3. Create virtual environment
python -m venv venv

# Activate virtual environment

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "import sklearn, xgboost, streamlit, selenium; print('All dependencies installed successfully!')"
```

---

# 📦 requirements.txt

```txt
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==1.7.6
matplotlib==3.7.2
seaborn==0.12.2
streamlit==1.25.0
flask==2.3.3
joblib==1.3.2
plotly==5.15.0
selenium==4.15.0
webdriver-manager==4.0.1
requests==2.31.0
Pillow==10.0.0
```

---

# 🚀 Usage Guide

## 1️⃣ Prepare Dataset

Place NASA PROMISE datasets (`kc1.csv`, `pc1.csv`, etc.) inside the `dataset/` folder.

---

## 2️⃣ Run Complete Pipeline

```bash
python main.py
```

### This will:

- Load and preprocess the dataset
- Perform feature selection
- Train all three models
- Evaluate model performance
- Generate risk analysis
- Save reports and visualizations

---

## 3️⃣ Launch Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

Access dashboard at:

```text
http://localhost:8501
```

---

## 4️⃣ Start API Server

```bash
python api.py
```

API runs at:

```text
http://localhost:5000
```

---

## 5️⃣ Run Selenium Web Tests

```bash
# Test Selenium setup
python test_selenium_simple.py

# Collect website metrics
python selenium_testing.py

# Run comprehensive test suite
python web_test_suite.py

# Analyze collected metrics
python monitor_websites.py
```

---

## 6️⃣ Make Predictions

### Using Python

```python
import requests
import json

url = "http://localhost:5000/predict"

api_key = "YOUR_API_KEY"

data = {
    "module_name": "PaymentService.java",
    "features": {
        "loc": 1500,
        "v(g)": 25,
        "branchCount": 30,
        "e": 160000
    }
}

response = requests.post(
    url,
    headers={
        "Content-Type": "application/json",
        "X-API-Key": api_key
    },
    json=data
)

print(response.json())
```

### Using cURL

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "module_name": "TestModule.java",
    "features": {
      "loc": 1000,
      "v(g)": 20,
      "branchCount": 25
    }
  }'
```

---

# 🧪 Selenium Web Testing Integration

## 📊 Website Metrics Collected

| Metric | Description |
|--------|-------------|
| Lines of Code | Estimated code complexity |
| Cyclomatic Complexity | Page structure complexity |
| Branch Count | Interactive elements |
| Load Time | Page load performance |
| JavaScript Errors | Client-side issues |
| DOM Elements | Total page elements |
| Dynamic Elements | React/Angular/Vue components |
| Forms & Iframes | Interactive element count |

---

## ▶️ Run Selenium Tests

```python
from selenium_testing import SeleniumBugPredictor

tester = SeleniumBugPredictor(browser='chrome', headless=True)

metrics = tester.collect_page_metrics(
    'https://example.com',
    'Homepage'
)

tester.generate_demo_report([metrics])

tester.close()
```

---

# 📡 API Documentation

## 🔐 Authentication

All endpoints require API key:

```text
X-API-Key: your-api-key-here
```

---

## 📍 Endpoints

### GET /health

```json
{
  "status": "healthy",
  "models": ["logistic", "random_forest", "xgboost"],
  "features_expected": 20,
  "scaler_loaded": true,
  "version": "2.0.0"
}
```

---

### POST /predict

#### Request

```json
{
  "module_name": "ModuleName.java",
  "features": {
    "loc": 1500,
    "v(g)": 25,
    "branchCount": 30,
    "e": 160000
  }
}
```

#### Response

```json
{
  "module_name": "ModuleName.java",
  "consensus": {
    "bug_probability": 0.635,
    "risk_score": 63.5,
    "risk_level": "MEDIUM"
  }
}
```

---

# 📊 Model Performance

| Model | Accuracy | F1-Score | AUC-ROC | Best For |
|------|----------|----------|---------|----------|
| XGBoost | 85-92% | 0.75-0.85 | 0.88-0.95 | High performance |
| Random Forest | 82-88% | 0.70-0.80 | 0.85-0.92 | Handling overfitting |
| Logistic Regression | 75-82% | 0.65-0.75 | 0.80-0.88 | Interpretability |

---

# ⚠️ Risk Scoring

| Risk Level | Probability | Action Required | Priority |
|------------|-------------|----------------|----------|
| HIGH | ≥ 70% | 🚨 Test immediately! | CRITICAL |
| MEDIUM | 30% - 70% | 📋 Schedule for testing | NORMAL |
| LOW | < 30% | ✅ Routine QA | LOW |

---

# 📁 Reports & Visualizations

Generated automatically in `reports/` folder:

| File | Description |
|------|-------------|
| confusion_matrix_*.png | Confusion matrices |
| roc_curves.png | ROC comparison |
| model_comparison.png | Performance charts |
| feature_importance.png | Important features |
| risk_distribution.png | Risk distribution |
| selenium_report_*.txt | Selenium reports |
| page_metrics.json | Website metrics |
| performance_data.json | Performance data |
| evaluation_report.txt | Evaluation metrics |
| risk_report.txt | Risk analysis |
| predictions.csv | Prediction results |

---

# 🔧 Troubleshooting

## ❌ Common Issues

### 1. ImportError: No module named 'src'

```bash
cd Intelligent-Bug-Prediction-System
python main.py
```

---

### 2. Selenium WebDriver not found

```bash
pip install --upgrade selenium webdriver-manager
python test_selenium_simple.py
```

---

### 3. Dataset not found

Place datasets inside:

```text
dataset/
```

---

### 4. API Key Invalid

Check `api.py` for correct API key.

---

# 🎯 Future Enhancements

- 🔄 Real-time GitHub integration
- 🧠 Deep Learning models
- 📱 Mobile application
- ☁️ Cloud deployment
- 📊 Advanced visualizations
- 🔐 Multi-tenant support
- 📈 Trend analysis
- 🤝 IDE plugins
- 🌐 Cross-browser testing
- 📸 Visual regression automation

---

# 📈 Sample Output

## 🖥 Console Output

```text
============================================================
INTELLIGENT BUG PREDICTION SYSTEM
============================================================

[Step 1] Data Preprocessing...
✓ Training set: (969, 21)

[Step 2] Feature Selection...
✓ Selected 20 important features

[Step 3] Model Training...
✓ Trained 3 models

[Step 4] Model Evaluation...
✓ Best Model: XGBoost
```

---

## 🧪 Selenium Test Output

```text
============================================================
🧪 SELENIUM WEB TESTING INTEGRATION
============================================================

✓ Chrome WebDriver initialized (Headless: True)

📊 Analyzing: GitHub - https://github.com
✓ Elements: 1928, JS Errors: 0, Complexity: 19.3

📊 Analyzing: Stack Overflow - https://stackoverflow.com
✓ Elements: 45, JS Errors: 3, Complexity: 30.4

✓ Report saved successfully
```

---

# 📝 License

This project is licensed under the MIT License.

---

# 🙏 Acknowledgments

- NASA PROMISE Repository
- Scikit-learn community
- XGBoost contributors
- Streamlit developers
- Selenium WebDriver team

---

# 📧 Contact

For questions or support, open an issue on GitHub.

⭐ Star this repository if you find it useful!
