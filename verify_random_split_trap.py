import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import r2_score
import sys
import os

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data, encode_features

print("--- Verifying 'Interpolation Trap' (Random vs Time-Series Split) ---")

# Load Common Data
df_raw = load_data('data/processed/common_preprocessed.csv')
target = 'Value_co2_emissions_kt_by_country'

# We need a quick preprocessing compatible with all 3 models for this experiment.
# LR/SVR need One-Hot + Scaling.
# XGBoost can handle Ordinal.
# To make it comparable, let's use One-Hot for all, or specific per model.
# Let's use specific datasets if available, or just quick prep.
# Using 'lr_final_prep.csv' is best for LR.
# For SVR/XGB, we can assume standard prep.

results = []

def evaluate_model(name, model, X, y, years):
    # 1. Random Split (The Trap)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train_r, y_train_r)
    r2_random = model.score(X_test_r, y_test_r)

    # 2. Time Series Split (The Truth)
    # Train: < 2015, Test: >= 2015
    mask_train = years < 2015
    mask_test = years >= 2015
    
    X_train_t = X[mask_train]
    y_train_t = y[mask_train]
    X_test_t = X[mask_test]
    y_test_t = y[mask_test]
    
    model.fit(X_train_t, y_train_t)
    r2_time = model.score(X_test_t, y_test_t)
    
    drop = (r2_time - r2_random) / r2_random * 100
    return r2_random, r2_time, drop

# --- Linear Regression ---
# Use lr_final_prep.csv (Has One-Hot, Scaled, No Outliers)
print("Evaluating Linear Regression...")
df_lr = pd.read_csv('data/processed/lr_final_prep.csv')
# Need Years to split. `lr_final_prep.csv` usually doesn't have Year?
# Phase 1 script reconstructed Year from map.
# Let's reconstruct it quickly.
map_df = pd.read_csv('data/processed/recovered_index_map.csv')
df_common = load_data('data/processed/common_preprocessed.csv')
# Ensure indices align
df_lr_subset = df_lr.iloc[map_df['LR_Index']].reset_index(drop=True)
original_indices = map_df['Original_Index'].values
years_lr = df_common.loc[original_indices, 'Year'].values

cols_lr = [c for c in df_lr.columns if c != target and c != 'Year'] # target is in csv?
X_lr = df_lr[cols_lr]
y_lr = df_lr[target]

r2_rand, r2_time, drop = evaluate_model("Linear Regression", Ridge(), X_lr, y_lr, years_lr)
results.append(["Linear Regression", "Panel", r2_rand, r2_time, f"{drop:.1f}%"])


# --- XGBoost (Gradient Boosting) ---
# Trees fail at extrapolation. Use Ordinal encoded data for fair comparison.
print("Evaluating XGBoost (GradientBoostingRegressor)...")
# Quick prep: Ordinal Encode Entity
df_xgb = df_common.copy()
enc = OrdinalEncoder()
df_xgb['Entity_Ord'] = enc.fit_transform(df_xgb[['Entity']])
# Drop non-numeric
cols_xgb = ['Entity_Ord', 'Year', 'gdp_per_capita', 'Primary energy consumption per capita (kWh/person)']
# Add lags if available in common? Yes, common has lags.
cols_xgb = [c for c in df_xgb.columns if df_xgb[c].dtype in [float, int] and c != target]

X_xgb = df_xgb[cols_xgb].fillna(0)
y_xgb = df_xgb[target]
years_xgb = df_xgb['Year'].values

model_xgb = GradientBoostingRegressor(n_estimators=100, random_state=42)
r2_rand, r2_time, drop = evaluate_model("XGBoost", model_xgb, X_xgb, y_xgb, years_xgb)
results.append(["XGBoost", "Panel", r2_rand, r2_time, f"{drop:.1f}%"])


# --- SVR ---
# SVR is heavy. Limit iterations or data size if needed.
print("Evaluating SVR...")
# SVR needs scaling. Use LR data but maybe subset?
# SVR on 2000 rows is fine.
model_svr = SVR(kernel='rbf') # RBF is typical for interpolation but fails extrapolation
r2_rand, r2_time, drop = evaluate_model("SVR", model_svr, X_lr, y_lr, years_lr) # Use LR One-Hot data
results.append(["SVR", "Panel", r2_rand, r2_time, f"{drop:.1f}%"])


# Print Table
print("\n--- Updated Table Results ---")
print(f"{'Algorithm':<20} | {'Approach':<10} | {'Random R2':<10} | {'Time R2':<10} | {'Drop'}")
for row in results:
    print(f"{row[0]:<20} | {row[1]:<10} | {row[2]:.4f}     | {row[3]:.4f}     | {row[4]}")
