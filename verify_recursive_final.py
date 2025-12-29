import pandas as pd
import numpy as np
import sys
import os
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

print("--- Final Verification of Recursive Forecasting ---")

# 1. Load Data
# We rely on the aligned LR data which has whitelisting applied
df_lr_raw = load_data('data/processed/lr_final_prep.csv') 
df_common = load_data('data/processed/common_preprocessed.csv')
map_df = pd.read_csv('data/processed/recovered_index_map.csv')

# Align
df_lr = df_lr_raw.iloc[map_df['LR_Index']].copy().reset_index(drop=True)
original_indices = map_df['Original_Index'].values
df_lr['Year'] = df_common.loc[original_indices, 'Year'].values
df_lr['Entity'] = df_common.loc[original_indices, 'Entity'].values

# Define Split
SPLIT_YEAR = 2015
TARGET = 'Value_co2_emissions_kt_by_country'

# 2. Prepare Train/Test
train_df = df_lr[df_lr['Year'] < SPLIT_YEAR].copy()
test_df = df_lr[df_lr['Year'] >= SPLIT_YEAR].copy().sort_values(['Year', 'Entity'])

# Features
drop_cols = [TARGET, 'Year', 'Entity']
# Filter out non-numeric cols just in case
feature_cols = [c for c in df_lr.columns if c not in drop_cols and df_lr[c].dtype in [np.float64, np.int64]]

X_train = train_df[feature_cols]
y_train = train_df[TARGET]

# 3. Train Model
print(f"Training Global Model on {len(X_train)} samples...")
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# 4. Determine Scaler for Prediction -> Lag Update
# The Lag feature in X is StandardScaled. 
# The Prediction y is Clean CO2 (Original Scale? Wait)
# CHECK PREPROCESSING LOGIC:
# Target 'Value_co2_emissions_kt_by_country' is NOT Scaled in lr_final_prep.csv (usually).
# But the FEATURE 'Value_co2_emissions_kt_by_country_lag1' WAS Scaled in preprocessing.
# Logic:
# df_lr has Scaled Features.
# df_lr target column: The preprocessing script REMOVES target from numeric columns before scaling?
# Let's check the code or infer.
# In notebooks/2_...: 
# numeric_cols_lr = ...
# feature_cols_lr = [c for c ... if c != target]
# df_lr[feature_cols_lr] = scaler.fit_transform(...)
# SO: Target is RAW (Unscaled). Features are SCALED.

# We need to find the mean/std used for scaling the Lag Feature.
# Typically Lag1 is derived from Target. So Lag1 Mean/Std ~= Target Mean/Std (shifted).
# Let's calculate Mean/Std of TARGET in TRAINING SET.
target_mean = y_train.mean()
target_std = y_train.std()
print(f"Target Stats (Train): Mean={target_mean:.2f}, Std={target_std:.2f}")

# For Lag1 feature name, we verify if it exists.
# VIF removal might have dropped it!
lag_col = 'Value_co2_emissions_kt_by_country_lag1'
if lag_col not in X_train.columns:
    print(f"WARNING: '{lag_col}' NOT found in features! Recursive loop might be irrelevant if Lag1 was dropped.")
    # Check if we have lag vars
    lags = [c for c in X_train.columns if 'lag' in c.lower()]
    print(f"Found Lag Features: {lags}")
    
    if not lags:
        print("NO Lag features found. Recursive forecasting is identical to Teacher Forcing.")
        # Just run standard predict
        X_test = test_df[feature_cols]
        preds = model.predict(X_test)
        r2 = r2_score(test_df[TARGET], preds)
        print(f"R2 Score: {r2:.4f}")
        sys.exit(0)

print(f"Lag Feature to Update: {lag_col} (Assuming this is the main autoregressive term)")

# 5. Recursive Loop
print("Starting Recursive Loop...")
test_years = sorted(test_df['Year'].unique())
recursive_preds = []
actuals = []

# We need a working copy of test_df that we can mutate (update lags)
test_df_dynamic = test_df.copy()

for year in test_years:
    # Get current year data
    current_year_mask = test_df_dynamic['Year'] == year
    X_current = test_df_dynamic.loc[current_year_mask, feature_cols]
    
    if X_current.empty: continue
    
    # Predict
    preds = model.predict(X_current)
    
    # Save predictions
    test_df_dynamic.loc[current_year_mask, 'Prediction'] = preds
    
    # Update NEXT Year's Lag
    next_year = year + 1
    if next_year <= max(test_years):
        # We need to map current entities to next year's rows
        # For each entity in current year, calculate scaled lag
        
        # Scaling logic: (Pred - Mean) / Std
        scaled_preds = (preds - target_mean) / target_std
        
        # In this simplified loop, assuming DataFrame is sorted or we join
        # Vectorized update is safer
        current_entities = test_df_dynamic.loc[current_year_mask, 'Entity'].values
        
        # Find next year rows for these entities
        next_year_mask = (test_df_dynamic['Year'] == next_year) & (test_df_dynamic['Entity'].isin(current_entities))
        
        # This is tricky without strict alignment.
        # Let's iterate entities to be safe (slow but sure)
        for entity, pred_val in zip(current_entities, preds):
            # Scale
            val_scaled = (pred_val - target_mean) / target_std
            
            # Update
            mask_next = (test_df_dynamic['Year'] == next_year) & (test_df_dynamic['Entity'] == entity)
            if mask_next.any():
                test_df_dynamic.loc[mask_next, lag_col] = val_scaled

# Evaluate
y_true = test_df_dynamic[TARGET]
y_pred = test_df_dynamic['Prediction']
r2_rec = r2_score(y_true, y_pred)
rmse_rec = np.sqrt(mean_squared_error(y_true, y_pred))

print("-" * 30)
print(f"Recursive R2 Score: {r2_rec:.4f}")
print(f"Recursive RMSE: {rmse_rec:,.2f}")
print("-" * 30)

# Compare with Teacher Forcing (Standard)
preds_tf = model.predict(test_df[feature_cols])
r2_tf = r2_score(test_df[TARGET], preds_tf)
print(f"Teacher Forcing R2: {r2_tf:.4f}")
print(f"Difference: {(r2_tf - r2_rec):.4f}")
