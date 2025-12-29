import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

print("--- Checking 'Noise' Entities Status ---")

# 1. Load Data
df_common = load_data('data/processed/common_preprocessed.csv')
df_lr_raw = load_data('data/processed/lr_final_prep.csv') 
map_df = pd.read_csv('data/processed/recovered_index_map.csv')

# Align LR
df_lr = df_lr_raw.iloc[map_df['LR_Index']].copy().reset_index(drop=True)
original_indices = map_df['Original_Index'].values
df_lr['Entity'] = df_common.loc[original_indices, 'Entity'].values

# 2. Check Specific Countries
targets = ['Bermuda', 'Qatar', 'Luxembourg', 'Iceland', 'United States', 'China']

print(f"{'Country':<20} | {'Original':<10} | {'Final (LR)':<10} | {'Status'}")
print("-" * 55)

for c in targets:
    orig_count = df_common[df_common['Entity'] == c].shape[0]
    final_count = df_lr[df_lr['Entity'] == c].shape[0]
    
    status = "REMOVED" if final_count == 0 else "KEPT"
    print(f"{c:<20} | {orig_count:<10} | {final_count:<10} | {status}")
