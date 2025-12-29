import pandas as pd
import numpy as np
import sys
import os
from sklearn.linear_model import LinearRegression

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

TARGET = 'Value_co2_emissions_kt_by_country'

# Load Data
df_common = load_data('data/processed/common_preprocessed.csv')
df_lr = load_data('data/processed/lr_final_prep.csv')

# Align indices like in the notebook
common_idx = df_common.index.intersection(df_lr.index)
df_lr = df_lr.loc[common_idx]

# IMPORTANT: Re-attach Entity and Year from df_common
# Assuming df_common has them as columns or we can extract from index/columns
# If load_data sets them as index, we reset.
if 'Entity' not in df_common.columns:
    df_common = df_common.reset_index()

df_lr['Year'] = df_common.loc[common_idx, 'Year']
df_lr['Entity'] = df_common.loc[common_idx, 'Entity']

lag_col_name = next(c for c in df_lr.columns if f'{TARGET}_lag1' in c)

print(f"Target: {TARGET}")
print(f"Lag Feature: {lag_col_name}")

# Sort to perform shift
df_lr = df_lr.sort_values(['Entity', 'Year'])

# Create "Theoretical Raw Lag" from Target
df_lr['Raw_Lag_Calculated'] = df_lr.groupby('Entity')[TARGET].shift(1)

# Drop NaNs
valid = df_lr.dropna(subset=['Raw_Lag_Calculated', lag_col_name])

# Check Correlation
corr_val = np.corrcoef(valid['Raw_Lag_Calculated'], valid[lag_col_name])[0,1]
print(f"Correlation (Raw_Lat_Calc vs Scaled_Lag_Feat): {corr_val:.4f}")

# Learn Mapping
if abs(corr_val) > 0.5:
    lr = LinearRegression()
    X = valid[['Raw_Lag_Calculated']]
    y = valid[lag_col_name]
    lr.fit(X, y)
    print(f"Mapping Found: Scaled = {lr.coef_[0]:.6e} * Raw + {lr.intercept_:.4f}")
    
    # Check Log Mapping too
    valid['Log_Raw_Lag'] = np.log1p(valid['Raw_Lag_Calculated'])
    corr_log = np.corrcoef(valid['Log_Raw_Lag'], valid[lag_col_name])[0,1]
    print(f"Log Correlation: {corr_log:.4f}")
    
    if abs(corr_log) > abs(corr_val):
        print("Log mapping is better.")
else:
    print("No obvious mapping found. Something is wrong with the data file or column identification.")
