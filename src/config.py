"""
Configuration file for BTC prediction project
"""

# Data paths
RAW_DATA_PATH = "data/raw/btc_1h.csv"
CLEAN_DATA_PATH = "data/processed/btc_clean.csv"
FEATURES_DATA_PATH = "data/processed/btc_features.csv"
FINAL_DATA_PATH = "data/processed/btc_final.csv"

# Model paths
LR_MODEL_PATH = "models/logistic_regression.pkl"
RF_MODEL_PATH = "models/random_forest.pkl"
XGB_MODEL_PATH = "models/xgboost.pkl"

# Reports paths
FIGURES_PATH = "reports/figures/"
METRICS_PATH = "reports/metrics/results.csv"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
HORIZON = 4  # Prediction horizon in hours
THRESHOLD = 0.002  # 0.2% threshold for target
SIGNAL_THRESHOLD = 0.55  # Probability threshold for trading signal

# Trading parameters
TRANSACTION_COST = 0.001  # 0.1% per transaction
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate

# Features list
FEATURES = [
    'return_lag_1', 'return_lag_2', 'return_lag_4', 'return_lag_8', 'return_lag_24',
    'volatility_24h', 'rsi', 'macd', 'macd_signal',
    'bb_high', 'bb_low', 'bb_position', 'volume_ratio', 'range'
]

# Target column
TARGET = 'target'