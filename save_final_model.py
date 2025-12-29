import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib
import json
import sys
import os

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

print("--- Finalizing and Saving Model Artifacts ---")

# 1. Load Clean Data (2000-2019)
# Assuming 'lr_final_prep.csv' has been cleaned of 2020 data by clean_2020_safe.py
df_lr_raw = load_data('data/processed/lr_final_prep.csv') 
df_common = load_data('data/processed/common_preprocessed.csv')
map_df = pd.read_csv('data/processed/recovered_index_map.csv')

# Align
df_lr = df_lr_raw.iloc[map_df['LR_Index']].copy().reset_index(drop=True)
original_indices = map_df['Original_Index'].values
df_lr['Year'] = df_common.loc[original_indices, 'Year'].values
df_lr['Entity'] = df_common.loc[original_indices, 'Entity'].values

TARGET = 'Value_co2_emissions_kt_by_country'
drop_cols = [TARGET, 'Year', 'Entity']
feature_cols = [c for c in df_lr.columns if c not in drop_cols and df_lr[c].dtype in [np.float64, np.int64]]

print(f"Training Data Shape: {df_lr.shape}")
print(f"Years Included: {df_lr['Year'].min()} - {df_lr['Year'].max()}")

# 2. Train Model
X_train = df_lr[feature_cols]
y_train = df_lr[TARGET]

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
print(f"Model Trained. Coefficients: {len(model.coef_)}")

# 3. Save Model
model_path = 'models/global_ridge_model.pkl'
joblib.dump(model, model_path)
print(f"✅ Model saved to: {model_path}")

# 4. Save Scaler / Preprocessing Stats
# Since 'lr_final_prep.csv' is ALREADY scaled, we can't fit a scaler on it perfectly to reverse engineer.
# However, for INFERENCE (e.g. Validation script), we utilized the Mean/Std of the 'common_preprocessed' data
# corresponding to the training rows.
# We should save THESE stats so the inference pipeline is reproducible.

train_common = df_common.iloc[original_indices].reset_index(drop=True)
means = train_common.mean(numeric_only=True)
stds = train_common.std(numeric_only=True)
# Convert to dict
scaler_stats = {
    'means': means.to_dict(),
    'stds': stds.to_dict(),
    'target_mean': train_common[TARGET].mean(),
    'target_std': train_common[TARGET].std(),
    'feature_names': feature_cols
}

scaler_path = 'models/scaler_stats.json'
with open(scaler_path, 'w') as f:
    json.dump(scaler_stats, f, indent=4)
print(f"✅ Scaler Stats saved to: {scaler_path}")

print("--- Pipeline Finalization Complete ---")
