"""
train_model.py - Model training module
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import joblib
import os
import time

def setup_logging():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

logger = setup_logging()

def save_model(model, model_name):
    """Save model to disk"""
    os.makedirs('models', exist_ok=True)
    filepath = os.path.join('models', model_name)
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")

class ModelTrainer:
    """Class for training ML models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.training_results = {}
        self.init_models()
    
    def init_models(self):
        """Initialize the three ML models"""
        self.models['Logistic Regression'] = LogisticRegression(
            random_state=self.random_state, max_iter=1000, class_weight='balanced'
        )
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=100, random_state=self.random_state, class_weight='balanced', n_jobs=-1
        )
        self.models['XGBoost'] = XGBClassifier(
            n_estimators=100, random_state=self.random_state, use_label_encoder=False, 
            eval_metric='logloss', n_jobs=-1
        )
        logger.info(f"Initialized {len(self.models)} models")
    
    def train_model(self, model_name, X_train, y_train):
        """Train a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.trained_models[model_name] = model
        self.training_results[model_name] = {'training_time': training_time}
        
        model_filename = f"{model_name.lower().replace(' ', '_')}.pkl"
        save_model(model, model_filename)
        
        logger.info(f"{model_name} trained in {training_time:.2f} seconds")
        return model
    
    def train_all_models(self, X_train, y_train):
        """Train all models"""
        for model_name in self.models.keys():
            self.train_model(model_name, X_train, y_train)
        return self.trained_models
    
    def cross_validate(self, model_name, X_train, y_train, cv=5):
        """Perform cross-validation"""
        if model_name not in self.trained_models:
            self.train_model(model_name, X_train, y_train)
        
        model = self.trained_models[model_name]
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        
        results = {
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std()
        }
        
        logger.info(f"{model_name} CV Score: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        self.training_results[model_name]['cv_results'] = results
        return results

def train_models(X_train, y_train, perform_tuning=False):
    """Quick training function"""
    trainer = ModelTrainer()
    trained_models = trainer.train_all_models(X_train, y_train)
    
    for model_name in trainer.models.keys():
        trainer.cross_validate(model_name, X_train, y_train)
    
    return trained_models, trainer.training_results