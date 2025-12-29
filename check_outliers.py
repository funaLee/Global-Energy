import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

print("--- Checking for Major Economies (USA, China) in Datasets ---")

# 1. Load Common Data (Source of Truth)
df_common = load_data('data/processed/common_preprocessed.csv')
print(f"Common Data Shape: {df_common.shape}")

countries_to_check = ['United States', 'China', 'India', 'Germany']

print("\n[Common Data]")
for c in countries_to_check:
    count = df_common[df_common['Entity'] == c].shape[0]
    print(f"  {c}: {count} records")

# 2. Load LR Final Data (Outliers Removed)
# Note: LR data might not have 'Entity' column directly if it was dropped,
# but we recovered the mapping in 'recovered_index_map.csv'
df_lr_raw = load_data('data/processed/lr_final_prep.csv') 
map_df = pd.read_csv('data/processed/recovered_index_map.csv')

# Align LR data to get Entities
df_lr_aligned = df_lr_raw.iloc[map_df['LR_Index']].copy().reset_index(drop=True)
original_indices = map_df['Original_Index'].values
df_lr_aligned['Entity'] = df_common.loc[original_indices, 'Entity'].values

print(f"\n[LR Final Data (Outliers Removed)] Shape: {df_lr_aligned.shape}")
for c in countries_to_check:
    count = df_lr_aligned[df_lr_aligned['Entity'] == c].shape[0]
    print(f"  {c}: {count} records")
    
# 3. Check Percentage Retained
print("\n[Retention Rate]")
for c in countries_to_check:
    original = df_common[df_common['Entity'] == c].shape[0]
    retained = df_lr_aligned[df_lr_aligned['Entity'] == c].shape[0]
    if original > 0:
        pct = (retained / original) * 100
        print(f"  {c}: {retained}/{original} ({pct:.1f}%)")
