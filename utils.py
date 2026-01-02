"""
utils.py - Visualization and utility functions
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict

def plot_metrics(metrics: Dict) -> plt.Figure:
    """Create visualization of model metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Confusion Matrix
    sns.heatmap(metrics['confusion_matrix'], 
                annot=True, fmt='d', cmap='Blues', 
                ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(metrics['y_true'], metrics['y_proba'])
    axes[1].plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.2f}")
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    
    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names: list) -> plt.Figure:
    """Plot feature importance if available"""
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        importances = model.named_steps['model'].feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=importances[indices], 
                    y=np.array(feature_names)[indices], 
                    ax=ax)
        ax.set_title('Top 15 Important Features')
        return fig
    return None

def get_sample_input(df: pd.DataFrame) -> dict:
    """Generate sample input with correct data types"""
    sample = df.drop('Churn', axis=1).iloc[0].to_dict()
    for col, val in sample.items():
        if pd.api.types.is_numeric_dtype(df[col]):
            sample[col] = float(val)
    return sample
