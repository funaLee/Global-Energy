import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

print("--- Analysis of Dropped Countries ---")

# 1. Raw Data
df_raw = load_data('data/raw/global-data-on-sustainable-energy.csv')
raw_entities = set(df_raw['Entity'].unique())
print(f"Original Entities: {len(raw_entities)}")

# 2. Common Preprocessed (After Imputation & Lags)
df_common = load_data('data/processed/common_preprocessed.csv')
common_entities = set(df_common['Entity'].unique())
print(f"After Basic Cleaning & Lags: {len(common_entities)}")
missing_after_common = raw_entities - common_entities
print(f"Dropped Step 1 (Lags/Basic): {len(missing_after_common)}") 
if len(missing_after_common) > 0:
    print(f"Examples: {list(missing_after_common)[:5]}")

# 3. Final LR Data (After IQR Outlier Removal)
# Need to reconstruct map to know who is who because Entity is one-hot or dropped
map_df = pd.read_csv('data/processed/recovered_index_map.csv')
# The map maps the Final LR rows back to Common rows.
final_indices = map_df['Original_Index'].unique()
df_final_survivors = df_common.loc[final_indices]
final_entities = set(df_final_survivors['Entity'].unique())

print(f"Final Model Entities: {len(final_entities)}")
dropped_final = common_entities - final_entities
print(f"Dropped Step 2 (IQR Outlier Cleaning): {len(dropped_final)}")
print(f"Countries Removed by Outlier Filter: {sorted(list(dropped_final))}")

# 4. Check overlap with Real World List (103)
# The user mentioned 103. Is that the Final Internal list or External list?
# If Final Internal is > 103, then 103 is just the External Validation Set intersection.

print(f"\nSummary:")
print(f"  - Started with: {len(raw_entities)}")
print(f"  - Valid for Modeling (Internal): {len(final_entities)}")
print(f"  - Drop Count: {len(raw_entities) - len(final_entities)}")
