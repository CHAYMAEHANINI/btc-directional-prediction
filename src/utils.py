"""
Utility functions for BTC prediction project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import os

def load_and_prepare_data(filepath, skiprows=3):
    """
    Load and prepare raw BTC data
    """
    df = pd.read_csv(filepath, skiprows=skiprows, 
                     names=['Price', 'Close', 'High', 'Low', 'Open', 'Volume'])
    df = df.rename(columns={'Price': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d %H:%M:%S%z', utc=True)
    df = df.sort_values('Datetime').reset_index(drop=True)
    
    # Convert numeric columns
    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def clean_data(df):
    """
    Clean the data: handle missing values, create returns
    """
    # Handle zero volumes
    df['Volume'] = df['Volume'].replace(0, np.nan)
    df['Volume'] = df['Volume'].fillna(df['Volume'].median())
    
    # Calculate returns
    df['returns'] = df['Close'].pct_change()
    
    # Drop NaN returns
    df = df.dropna(subset=['returns'])
    
    # Handle zero returns (optional - keep as is or replace)
    df.loc[:, 'returns'] = df['returns'].replace(0, np.nan)
    df.loc[:, 'returns'] = df['returns'].fillna(0)

    
    return df

def create_target(df, horizon=4, threshold=0.002):
    """
    Create binary target variable
    """
    df['return_future'] = df['Close'].pct_change(horizon).shift(-horizon)
    df['target'] = (df['return_future'] > threshold).astype(int)
    df = df.dropna(subset=['target', 'return_future'])
    return df

def calculate_metrics(y_true, y_pred, y_proba):
    """
    Calculate classification metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba)
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.show()

def plot_cumulative_returns(df, save_path=None):
    """
    Plot cumulative returns
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['Datetime'], df['cum_return'], label='Strategy', linewidth=2)
    plt.plot(df['Datetime'], (1 + df['returns']).cumprod(), label='Buy & Hold', linewidth=2, alpha=0.7)
    plt.title('Cumulative Returns: Strategy vs Buy & Hold')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'cumulative_returns.png'))
    plt.show()

def plot_feature_importance(model, feature_names, save_path=None):
    """
    Plot feature importance for tree-based models
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'feature_importance.png'))
        plt.show()

def save_model(model, path):
    """
    Save trained model
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path):
    """
    Load trained model
    """
    return joblib.load(path)

def calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
    """
    Calculate Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()
    return sharpe

def calculate_max_drawdown(cumulative_returns):
    """
    Calculate maximum drawdown
    """
    cum_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / cum_max - 1
    return drawdown.min()

def calculate_hit_ratio(trades_returns):
    """
    Calculate hit ratio (percentage of winning trades)
    """
    if len(trades_returns) == 0:
        return 0
    return (trades_returns > 0).mean()