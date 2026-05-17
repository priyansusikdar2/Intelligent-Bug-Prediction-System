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
- [API Documentation]
- [Model Performance]
- [Risk Scoring]
- [Reports & Visualizations]
- [Troubleshooting]
- [Future Enhancements]
- [Sample Output]
- [License]
- [Acknowledgments]
- [Contact]

---

# 📖 Overview

The Intelligent Bug Prediction System analyzes historical software metrics and defect datasets to identify modules likely to be affected by bugs. By leveraging machine learning algorithms, it generates risk scores for software modules, enabling development teams to prioritize testing activities effectively.

## 🔍 How It Works

1. **Data Collection** – Load historical software metrics (NASA PROMISE dataset)
2. **Preprocessing** – Clean, scale, and prepare data for analysis
3. **Feature Selection** – Identify most important software metrics
4. **Model Training** – Train multiple ML models on historical data
5. **Prediction** – Analyze new modules for bug probability
6. **Risk Scoring** – Generate risk levels (Low/Medium/High)
7. **Reporting** – Create visualizations and comprehensive reports

---

# ✨ Features

## ✅ Core Features

- 🔮 **Real-time Bug Prediction** – Predict bug probability for any software module
- 🤖 **Multiple ML Models** – Logistic Regression, Random Forest, XGBoost
- 📊 **Interactive Dashboard** – Streamlit-based web interface
- 📈 **Model Performance Comparison** – Visual comparison of all models
- ⚠️ **Risk Scoring System** – Low, Medium, High risk classification
- 📁 **Automatic Report Generation** – Save all results as images and text files
- 🔄 **Batch Processing** – Analyze multiple modules simultaneously
- 🎯 **Feature Importance Analysis** – Identify key metrics affecting bugs

## 📊 Visualization Capabilities

- Confusion matrices for each model
- ROC curves comparison
- Feature importance plots
- Risk distribution charts
- Model performance comparison graphs
- Risk heatmaps

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
| **Model Serialization** | Joblib |
| **Dataset** | NASA PROMISE Repository |

---

# 📁 Project Structure

```text
Intelligent-Bug-Prediction-System/
│
├── app/
│   ├── streamlit_app.py              # Main Streamlit dashboard
│   ├── prediction.py                 # Prediction module
│   └── dashboard.py                  # Standalone dashboard
│
├── dataset/
│   ├── kc1.csv                       # NASA KC1 dataset
│   ├── pc1.csv                       # NASA PC1 dataset
│   └── cleaned_dataset.csv           # Preprocessed dataset
│
├── models/
│   ├── logistic_regression.pkl       # Trained Logistic Regression
│   ├── random_forest.pkl             # Trained Random Forest
│   ├── xgboost.pkl                   # Trained XGBoost
│   ├── scaler.pkl                    # Feature scaler
│   └── selected_features.txt         # Selected features list
│
├── src/
│   ├── data_preprocessing.py         # Data cleaning & preparation
│   ├── feature_selection.py          # Feature importance analysis
│   ├── train_model.py                # Model training module
│   ├── evaluate_model.py             # Model evaluation
│   ├── risk_scoring.py               # Risk analysis
│   └── utils.py                      # Utility functions
│
├── reports/
│   ├── confusion_matrix_*.png        # Confusion matrix plots
│   ├── roc_curves.png                # ROC curves comparison
│   ├── model_comparison.png          # Model performance chart
│   ├── feature_importance.png        # Feature importance plot
│   ├── risk_distribution.png         # Risk level distribution
│   ├── evaluation_report.txt         # Model evaluation report
│   ├── risk_report.txt               # Risk analysis report
│   └── predictions.csv               # Prediction results
│
├── notebooks/
│   ├── EDA.ipynb                     # Exploratory data analysis
│   ├── Model_Training.ipynb          # Training experiments
│   └── Feature_Engineering.ipynb     # Feature engineering
│
├── api.py                            # Flask API server
├── main.py                           # Main execution script
├── requirements.txt                 # Python dependencies
└── README.md                         # Project documentation
```

---

# 💻 Installation

## 📌 Prerequisites

- Python 3.11 or higher
- pip package manager

---

## ⚙️ Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Intelligent-Bug-Prediction-System.git

# 2. Navigate into the project directory
cd Intelligent-Bug-Prediction-System

# 3. Create a virtual environment (recommended)
python -m venv venv

# 4. Activate virtual environment

# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Verify installation
python -c "import sklearn, xgboost, streamlit; print('All dependencies installed successfully!')"
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
```

---

# 🚀 Usage Guide

## 1️⃣ Prepare Dataset

Place your NASA PROMISE datasets (`kc1.csv`, `pc1.csv`, etc.) in the `dataset/` folder.

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
- Save all reports and visualizations

---

## 3️⃣ Launch Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

### Access Dashboard

```text
http://localhost:8501
```

### Dashboard Features

- 📊 Overview Page – Dataset statistics and visualizations
- 🔮 Predict Page – Interactive module prediction
- 📈 Performance Page – Model comparison metrics
- ⚠️ Risk Analysis Page – Risk distribution and heatmaps
- 📁 Reports Page – View all saved visualizations

---

## 4️⃣ Start API Server

```bash
python api.py
```

### API URL

```text
http://localhost:5000
```

---

## 5️⃣ Make Predictions

### Using Python

```python
import requests
import json

# API endpoint
url = "http://localhost:5000/predict"

# API key
api_key = "95c01f189f2bf1a17acd99a4a550e5b5e45f2183000d6e480b74af8810dab177"

# Module features
data = {
    "module_name": "PaymentService.java",
    "features": {
        "loc": 1500,
        "v(g)": 25,
        "branchCount": 30,
        "e": 160000
    }
}

# Make prediction request
response = requests.post(
    url,
    headers={
        "Content-Type": "application/json",
        "X-API-Key": api_key
    },
    json=data
)

# Print response
print(response.json())
```

---

### Using cURL

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: 95c01f189f2bf1a17acd99a4a550e5b5e45f2183000d6e480b74af8810dab177" \
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

# 📡 API Documentation

## 🔐 Authentication

All API endpoints require an API key.

### Header Format

```text
X-API-Key: your-api-key-here
```

---

# 📍 Endpoints

---

## ✅ GET /health

Check API health status.

### Response

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

## 🔮 POST /predict

Predict bug probability for a single module.

### Request Body

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

### Response

```json
{
  "module_name": "ModuleName.java",
  "consensus": {
    "bug_probability": 0.635,
    "risk_score": 63.5,
    "risk_level": "MEDIUM"
  },
  "model_predictions": {
    "logistic": {
      "bug_probability": 1.0,
      "risk_score": 100.0,
      "risk_level": "HIGH",
      "action": "Test immediately!"
    },
    "random_forest": {
      "bug_probability": 0.52,
      "risk_score": 52.0,
      "risk_level": "MEDIUM",
      "action": "Schedule for testing"
    },
    "xgboost": {
      "bug_probability": 0.386,
      "risk_score": 38.6,
      "risk_level": "MEDIUM",
      "action": "Schedule for testing"
    }
  }
}
```

---

## 📦 POST /predict/batch

Predict for multiple modules simultaneously (max 100 modules).

### Request Body

```json
{
  "modules": [
    {
      "module_name": "Module1.java",
      "features": {
        "loc": 1500,
        "v(g)": 25,
        "branchCount": 30
      }
    },
    {
      "module_name": "Module2.java",
      "features": {
        "loc": 500,
        "v(g)": 8,
        "branchCount": 10
      }
    }
  ]
}
```

---

## 📋 GET /features

Returns the list of expected features.

---

# 📊 Model Performance

## 🎯 Expected Accuracy (with hyperparameter tuning)

| Model | Accuracy | F1-Score | AUC-ROC | Best For |
|------|----------|----------|----------|----------|
| XGBoost | 85-92% | 0.75-0.85 | 0.88-0.95 | High performance |
| Random Forest | 82-88% | 0.70-0.80 | 0.85-0.92 | Handling overfitting |
| Logistic Regression | 75-82% | 0.65-0.75 | 0.80-0.88 | Interpretability |

---

## 📌 Confusion Matrix Explanation

```text
                 Predicted
              Non-Buggy   Buggy

Actual
Non-Buggy        TN         FP
Buggy            FN         TP
```

### Definitions

- **TN (True Negative)** → Correctly predicted non-buggy module
- **FP (False Positive)** → False alarm (wasted testing effort)
- **FN (False Negative)** → Missed bug (high risk)
- **TP (True Positive)** → Correctly identified buggy module

---

# ⚠️ Risk Scoring

## 🚦 Risk Levels

| Risk Level | Probability | Action Required | Priority |
|------------|-------------|----------------|----------|
| HIGH | ≥ 70% | 🚨 Test immediately! | CRITICAL |
| MEDIUM | 30% - 70% | 📋 Schedule for testing | NORMAL |
| LOW | < 30% | ✅ Routine QA | LOW |

---

## 📈 Risk Score Interpretation

- **90-100%** → Extremely high risk – Critical bugs expected
- **70-89%** → High risk – Immediate testing required
- **50-69%** → Medium-high risk – Priority testing
- **30-49%** → Medium-low risk – Normal priority
- **0-29%** → Low risk – Standard QA process

---

# 📁 Reports & Visualizations

All reports are automatically saved in the `reports/` folder.

## 📄 Generated Files

| File | Description |
|------|-------------|
| confusion_matrix_*.png | Confusion matrix for each model |
| roc_curves.png | ROC curves comparison |
| model_comparison.png | Performance comparison bar chart |
| feature_importance.png | Top features affecting predictions |
| risk_distribution.png | Risk level distribution pie chart |
| risk_analysis_score_distribution.png | Risk score histogram |
| risk_analysis_heatmap.png | Risk heatmap for top modules |
| evaluation_report.txt | Detailed model metrics |
| risk_report.txt | Comprehensive risk analysis |
| training_report.txt | Model training details |
| feature_selection_report.txt | Feature importance rankings |
| predictions.csv | All prediction results |

---

# 🔧 Troubleshooting

## ❌ Common Issues and Solutions

---

### 1️⃣ ImportError: No module named 'src'

### ✅ Solution

```bash
cd Intelligent-Bug-Prediction-System
python main.py
```

---

### 2️⃣ FileNotFoundError: dataset/kc1.csv not found

### ✅ Solution

Place dataset files inside the `dataset/` folder or generate sample data.

```bash
python create_sample_data.py
```

---

### 3️⃣ Model expects 20 features but got 21

### ✅ Solution

The feature `locCodeAndComment` was dropped during feature selection.

Use the provided prediction scripts which automatically align features.

---


---

### 5️⃣ Low model accuracy

## ✅ Possible Improvements

- Run hyperparameter tuning:

```bash
python improve_model.py
```

- Increase training data
- Improve data quality
- Apply feature engineering
- Use class balancing (SMOTE)
- Use cross-validation
- Combine ensemble methods

---

# ⚡ Performance Optimization Tips

- 📈 Increase training dataset size
- 🧠 Add domain-specific software metrics
- 🔄 Combine multiple ensemble models
- ⚖️ Handle class imbalance properly
- 📊 Use k-fold cross validation

---

# 🎯 Future Enhancements

- 🔄 Real-time GitHub integration
- 🧠 Deep Learning models (LSTM/Transformers)
- 📱 Mobile application support
- ☁️ Cloud deployment (AWS/Azure/GCP)
- 📊 Interactive advanced visualizations
- 🔐 Multi-tenant team support
- 📈 Risk trend analysis over time
- 🤝 VS Code / IntelliJ plugins

---

# 📈 Sample Output

## 🖥 Console Output

```text
============================================================
INTELLIGENT BUG PREDICTION SYSTEM
============================================================

[Step 1] Data Preprocessing...
✓ Training set: (969, 21)
✓ Test set: (243, 21)

[Step 2] Feature Selection...
✓ Selected 20 important features

[Step 3] Model Training...
✓ Trained 3 models

[Step 4] Model Evaluation...
✓ Best Model: XGBoost

[Step 5] Risk Scoring...
✓ Risk analysis complete

============================================================
PROJECT EXECUTION SUMMARY
============================================================

📊 Dataset Information:
  - Total samples: 1212
  - Buggy modules: 63 (25.9%)

🎯 Model Performance:
  XGBoost: Accuracy: 0.9045, F1-Score: 0.8678

⚠️ Risk Analysis:
  - High Risk: 34 modules (14.0%)
  - Average Risk Score: 46.82%
```

---

# 📝 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for more details.

---

# 🙏 Acknowledgments

- NASA PROMISE Repository for software defect datasets
- Scikit-learn community
- XGBoost developers
- Streamlit open-source contributors
- Open-source ML ecosystem contributors

---

# 📧 Contact

For questions, feature requests, or support:

- Open an issue on GitHub
- Contact the project maintainers

---

# ⭐ Support the Project

If you found this project useful:

⭐ Star the repository on GitHub  
🍴 Fork the project  
🐛 Report issues  
🤝 Contribute improvements

---
