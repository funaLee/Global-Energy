import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data, encode_features, remove_outliers, remove_high_vif
from sklearn.preprocessing import StandardScaler

print("--- CRITICAL FIX: Expanding Whitelist to Rescue Major Economies ---")

# 1. Pipeline Simulation
df = load_data('data/processed/common_preprocessed.csv')

# CRITICAL: Remove 2020 data (Known data quality issue)
if 'Year' in df.columns:
    original_len = len(df)
    df = df[df['Year'] != 2020]
    print(f"Removed {original_len - len(df)} rows of 2020 data.")

# EXPANDED WHITELIST (G20 + Major Regional Powers)
# Original: ['China', 'United States', 'India', 'Russian Federation', 'Japan', 'Germany', 'Brazil', 'Canada']
# New Additions based on 'dropped_entities_report': 
# United Kingdom, France, Italy, Australia, Saudi Arabia, South Korea, Turkey, Spain, Indonesia, Mexico, South Africa, Thailand, etc.
# We will Whitelist ANYONE who is an OUTLIER but is a VALID COUNTRY.
# Actually, a better whitelist strategy: "Don't remove outliers based on CO2/GDP Magnitude if they are valid countries".
# But to stick to the existing logic, we list them explicitly.

EXPANDED_WHITELIST = [
    'China', 'United States', 'India', 'Russian Federation', 'Japan', 'Germany', 'Brazil', 'Canada',
    'United Kingdom', 'France', 'Italy', 'Australia', 'South Korea', 'Saudi Arabia', 'Turkey', 'Indonesia', 
    'Spain', 'Mexico', 'South Africa', 'Thailand', 'Poland', 'Iran', 'Egypt', 'Pakistan', 'Viet Nam', 'Vietnam',
    'Argentina', 'Netherlands', 'Philippines', 'Malaysia', 'Belgium', 'Sweden', 'Poland', 'Ukraine', 'Kazakhstan',
    'United Arab Emirates', 'Algeria', 'Singapore', 'Nigeria'
]

print(f"Expanded Whitelist Size: {len(EXPANDED_WHITELIST)}")

# Log Transform (Standard Step)
skewed_cols = ['Financial flows to developing countries (US $)', 'Renewables (% equivalent primary energy)']
for col in skewed_cols:
    if col in df.columns:
        df[col] = np.log1p(df[col])

df_lr = encode_features(df, method='onehot')

# Outlier Logic - Protected
whitelist_mask = pd.Series(False, index=df_lr.index)
for country in EXPANDED_WHITELIST:
    col_name = f'Entity_{country}'
    if col_name in df_lr.columns:
        whitelist_mask = whitelist_mask | (df_lr[col_name] == 1)

df_protected = df_lr[whitelist_mask].copy()
df_to_clean = df_lr[~whitelist_mask].copy()

print(f"Protected Rows: {len(df_protected)}")
print(f"Rows subject to cleaning: {len(df_to_clean)}")

# Remove outliers from remaining small nations
df_cleaned_subset = remove_outliers(df_to_clean, method='iqr', threshold=3.0)

# Merge
df_lr_final = pd.concat([df_protected, df_cleaned_subset], axis=0).sort_index()

# Save index map for later recovery
original_indices = df_lr_final.index
map_df = pd.DataFrame({'LR_Index': range(len(df_lr_final)), 'Original_Index': original_indices})
map_df.to_csv('data/processed/recovered_index_map.csv', index=False)
print("Saved map validation.")

# VIF & Scaling
target = 'Value_co2_emissions_kt_by_country'
# CRITICAL: Protect Lag Features from VIF removal
exclude_from_vif = ['Financial flows to developing countries (US $)', 'Value_co2_emissions_kt_by_country_lag1']
df_lr_final = remove_high_vif(df_lr_final, target, threshold=10, exclude_cols=exclude_from_vif)

scaler_lr = StandardScaler()
numeric_cols = df_lr_final.select_dtypes(include=['float64', 'int64']).columns
feature_cols = [c for c in numeric_cols if c != target and not c.startswith('Entity_')]
df_lr_final[feature_cols] = scaler_lr.fit_transform(df_lr_final[feature_cols])

# Save
df_lr_final.to_csv('data/processed/lr_final_prep.csv', index=False)
print(f"Saved FIXED LR data: {df_lr_final.shape}")
print(f"Restored Major Economies. Total Rows: {len(df_lr_final)} (Prev was ~2129)")
