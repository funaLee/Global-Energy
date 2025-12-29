import pandas as pd
import numpy as np
import sys
import os
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

TARGET = 'Value_co2_emissions_kt_by_country'

# Load Data
df_common = load_data('data/processed/common_preprocessed.csv')
df_lr = load_data('data/processed/lr_final_prep.csv')

# Align
common_idx = df_common.index.intersection(df_lr.index)
df_lr = df_lr.loc[common_idx]
unscaled_raw = df_common.loc[common_idx]

lag_col_name = next(c for c in df_lr.columns if f'{TARGET}_lag1' in c)
print(f"Lag Column: {lag_col_name}")

# Get Series
raw_lag = unscaled_raw[lag_col_name]
lr_lag = df_lr[lag_col_name]

# 1. Linear Correlation
corr_lin = np.corrcoef(raw_lag, lr_lag)[0,1]
print(f"Linear Correlation (Raw vs ModelInput): {corr_lin:.4f}")

# 2. Log Correlation
log_raw_lag = np.log1p(raw_lag)
corr_log = np.corrcoef(log_raw_lag, lr_lag)[0,1]
print(f"Log Correlation (Log(Raw) vs ModelInput): {corr_log:.4f}")

# Identify correct scaler stats
if corr_log > corr_lin:
    print("MATCH: Features are Log-Transformed!")
    # Calculate Mean/Std of Log
    true_mean = log_raw_lag.mean()
    true_std = log_raw_lag.std()
    print(f"Log Mean: {true_mean:.4f}, Log Std: {true_std:.4f}")
    
    # Verify reconstruction
    reconstructed = (log_raw_lag - true_mean) / true_std
    corr_recon = np.corrcoef(reconstructed, lr_lag)[0,1]
    print(f"Reconstructed (Log->Scale) Correlation: {corr_recon:.4f}")
else:
    print("Features are Likely Just Scaled (but mismatch found earlier).")
    true_mean = raw_lag.mean()
    true_std = raw_lag.std()
    
# Check Target Distribution
y_model = df_lr[TARGET]
print(f"Model Target Mean: {y_model.mean():.2f} (If small < 20, likely Log)")
