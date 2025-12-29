import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import sys
import os

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

print("--- FINAL AUDIT: Internal Metrics Check ---")

# 1. Load Data
df_lr_raw = load_data('data/processed/lr_final_prep.csv') 
df_common = load_data('data/processed/common_preprocessed.csv')
map_df = pd.read_csv('data/processed/recovered_index_map.csv')

df_lr = df_lr_raw.iloc[map_df['LR_Index']].copy().reset_index(drop=True)
original_indices = map_df['Original_Index'].values
df_lr['Year'] = df_common.loc[original_indices, 'Year'].values
df_lr['Entity'] = df_common.loc[original_indices, 'Entity'].values

print(f"Data Shape: {df_lr.shape}")
print(f"Years: {sorted(df_lr['Year'].unique())}")

# 2. Split
SPLIT_YEAR = 2015
train_mask = df_lr['Year'] < SPLIT_YEAR
test_mask = (df_lr['Year'] >= SPLIT_YEAR) & (df_lr['Year'] <= 2019) # Explicitly cap at 2019

# Verify 2020 is gone
if 2020 in df_lr['Year'].values:
    print("CRITICAL VALIDATION FAILED: 2020 Data Found in Dataset!")
    sys.exit(1)
    
X_train = df_lr.loc[train_mask].drop(columns=['Value_co2_emissions_kt_by_country', 'Year', 'Entity'])
# Ensure only numeric
numeric_cols = X_train.select_dtypes(include=[np.number]).columns
X_train = X_train[numeric_cols]
y_train = df_lr.loc[train_mask, 'Value_co2_emissions_kt_by_country']

X_test = df_lr.loc[test_mask, numeric_cols]
y_test = df_lr.loc[test_mask, 'Value_co2_emissions_kt_by_country']

print(f"Train Size: {len(X_train)} (2000-2014)")
print(f"Test Size: {len(X_test)} (2015-2019)")

# 3. Train Global Model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# 4. Evaluate
preds = model.predict(X_test)
r2 = r2_score(y_test, preds)

print("-" * 30)
print(f"AUDITED One-Step R2 (2015-2019): {r2:.6f}")
print("-" * 30)

# Check Clustering Logic (Optional: Just to see if it matches 0.88)
# We won't re-run full clustering here, just the Global number is key.
