"""
Save Hybrid Model (LR + XGBoost on Residuals)
==============================================
Train and save both components of the best-performing Hybrid Model.

Components saved:
1. models/hybrid_lr_model.pkl - Ridge LR (α=10.0)
2. models/hybrid_xgb_residual_model.pkl - XGBoost on residuals
3. models/hybrid_model_metadata.json - Metadata for inference
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import joblib
import json
import sys
import os

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

print("=" * 70)
print("SAVING HYBRID MODEL ARTIFACTS")
print("=" * 70)

# ===========================
# 1. LOAD DATA
# ===========================
SPLIT_YEAR = 2015
TARGET = 'Value_co2_emissions_kt_by_country'

df_lr = pd.read_csv('data/processed/lr_final_prep.csv')
df_common = load_data('data/processed/common_preprocessed.csv')
map_df = pd.read_csv('data/processed/recovered_index_map.csv')

# Add Year and Entity back
original_indices = map_df['Original_Index'].values
df_lr['Year'] = df_common.loc[original_indices, 'Year'].values
df_lr['Entity'] = df_common.loc[original_indices, 'Entity'].values

print(f"Data shape: {df_lr.shape}")
print(f"Years: {df_lr['Year'].min()} - {df_lr['Year'].max()}")

# ===========================
# 2. PREPARE FEATURES
# ===========================
feature_cols = [c for c in df_lr.columns if c not in [TARGET, 'Year', 'Entity'] 
                and df_lr[c].dtype in [np.float64, np.int64]]

train_mask = df_lr['Year'] < SPLIT_YEAR
X_train = df_lr.loc[train_mask, feature_cols]
y_train = df_lr.loc[train_mask, TARGET]

print(f"\nTraining samples: {len(X_train)}")
print(f"Feature count: {len(feature_cols)}")

# ===========================
# 3. TRAIN HYBRID MODEL
# ===========================
# Hyperparameters from notebooks/12_Hybrid_Model.py (best performing)
LR_ALPHA = 10.0
XGB_PARAMS = {
    'n_estimators': 500,
    'max_depth': 3,
    'learning_rate': 0.1,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'random_state': 42,
    'n_jobs': -1
}

# 3.1 Train Ridge LR
print("\n[1/2] Training Ridge LR...")
lr_model = Ridge(alpha=LR_ALPHA)
lr_model.fit(X_train, y_train)
lr_preds_train = lr_model.predict(X_train)
print(f"  LR Train R²: {lr_model.score(X_train, y_train):.4f}")

# 3.2 Calculate residuals
residuals_train = y_train - lr_preds_train
print(f"  Residuals - Mean: {residuals_train.mean():.2f}, Std: {residuals_train.std():.2f}")

# 3.3 Train XGBoost on residuals
print("\n[2/2] Training XGBoost on Residuals...")
xgb_residual_model = XGBRegressor(**XGB_PARAMS)
xgb_residual_model.fit(X_train, residuals_train)
print("  XGBoost trained successfully")

# ===========================
# 4. SAVE ARTIFACTS
# ===========================
print("\n" + "=" * 70)
print("SAVING ARTIFACTS")
print("=" * 70)

os.makedirs('models', exist_ok=True)

# Save Ridge LR
lr_path = 'models/hybrid_lr_model.pkl'
joblib.dump(lr_model, lr_path)
print(f"✅ Ridge LR saved to: {lr_path}")

# Save XGBoost
xgb_path = 'models/hybrid_xgb_residual_model.pkl'
joblib.dump(xgb_residual_model, xgb_path)
print(f"✅ XGBoost Residual saved to: {xgb_path}")

# Save Metadata
metadata = {
    'model_type': 'Hybrid (LR + XGBoost on Residuals)',
    'lr_alpha': LR_ALPHA,
    'xgb_params': XGB_PARAMS,
    'feature_names': feature_cols,
    'target': TARGET,
    'training_years': f"2001-{SPLIT_YEAR-1}",
    'training_samples': len(X_train),
    'inference_formula': 'prediction = lr_model.predict(X) + xgb_model.predict(X)'
}

meta_path = 'models/hybrid_model_metadata.json'
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Metadata saved to: {meta_path}")

# ===========================
# 5. VERIFICATION (Load & Predict)
# ===========================
print("\n" + "=" * 70)
print("VERIFICATION: Load and Predict")
print("=" * 70)

# Reload models
lr_loaded = joblib.load(lr_path)
xgb_loaded = joblib.load(xgb_path)

# Test prediction on first sample
X_sample = X_train.iloc[[0]]
y_sample_actual = y_train.iloc[0]

lr_pred = lr_loaded.predict(X_sample)[0]
xgb_pred = xgb_loaded.predict(X_sample)[0]
hybrid_pred = lr_pred + xgb_pred

print(f"\nSample Prediction Test:")
print(f"  Actual CO2: {y_sample_actual:,.0f} kt")
print(f"  LR Pred: {lr_pred:,.0f} kt")
print(f"  XGB Residual Pred: {xgb_pred:,.0f} kt")
print(f"  Hybrid Pred (LR + XGB): {hybrid_pred:,.0f} kt")
print(f"  Error: {abs(y_sample_actual - hybrid_pred):,.0f} kt ({abs(y_sample_actual - hybrid_pred)/y_sample_actual*100:.2f}%)")

print("\n" + "=" * 70)
print("✅ HYBRID MODEL SAVED SUCCESSFULLY")
print("=" * 70)
