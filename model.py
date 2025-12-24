"""
model.py - Core machine learning operations for customer churn prediction
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, 
                           confusion_matrix, classification_report)
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

class ChurnModel:
    def __init__(self):
        self.pipeline = None
        self.feature_names = None
        self.target_name = 'Churn'
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and preprocess data"""
        df = pd.read_csv(filepath)
        
        # Clean data
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        # Convert target
        df[self.target_name] = df[self.target_name].map({'Yes': 1, 'No': 0})
        return df.dropna()

    def train(self, df: pd.DataFrame, model_type: str = 'rf', **params) -> dict:
        """Train model with automatic feature processing"""
        X = df.drop(self.target_name, axis=1)
        y = df[self.target_name]
        
        # Identify feature types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # Select model
        models = {
            'lr': LogisticRegression(max_iter=1000),
            'rf': RandomForestClassifier(),
            'xgb': XGBClassifier()
        }
        
        # Update model params
        model = models[model_type]
        model.set_params(**params)
        
        # Create full pipeline
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train and evaluate
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.pipeline.fit(X_train, y_train)
        self.feature_names = self._get_feature_names(X)
        
        return self.evaluate(X_test, y_test)
    
    def _get_feature_names(self, X: pd.DataFrame) -> list:
        """Get feature names after preprocessing"""
        preprocessor = self.pipeline.named_steps['preprocessor']
        num_features = X.select_dtypes(include=['number']).columns
        cat_features = X.select_dtypes(include=['object']).columns
        
        # Get numeric feature names
        num_names = list(num_features)
        
        # Get categorical feature names
        if hasattr(preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
            cat_names = list(preprocessor.named_transformers_['cat']
                              .get_feature_names_out(cat_features))
        else:
            cat_names = []
        
        return num_names + cat_names
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluate model performance"""
        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)[:, 1]
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'features': self.feature_names
        }
    
    def predict(self, input_data: dict) -> dict:
        """Predict churn for a single customer"""
        input_df = pd.DataFrame([input_data])
        proba = self.pipeline.predict_proba(input_df)[0][1]
        pred = self.pipeline.predict(input_df)[0]
        return {'probability': proba, 'prediction': pred}
    
    def save(self, filepath: str):
        """Save trained model"""
        dump({'pipeline': self.pipeline, 'features': self.feature_names}, filepath)
    
    @classmethod
    def load(cls, filepath: str):
        """Load trained model"""
        data = load(filepath)
        model = cls()
        model.pipeline = data['pipeline']
        model.feature_names = data['features']
        return model
