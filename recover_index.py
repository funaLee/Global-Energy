import pandas as pd
import numpy as np
import sys
import os

df_common = pd.read_csv('data/processed/common_preprocessed.csv')
df_lr = pd.read_csv('data/processed/lr_final_prep.csv')

TARGET = 'Value_co2_emissions_kt_by_country'
LAG = f'{TARGET}_lag1'



# Try very loose matching
df_common['key_target'] = df_common[TARGET].round(2)
#df_common['key_lag'] = df_common[LAG].round(4)

df_lr['key_target'] = df_lr[TARGET].round(2)
#df_lr['key_lag'] = df_lr[LAG].round(4)

print("Analyzing duplication (Target Only, Round 2)...")
# Check uniqueness in common
common_keys = df_common[['key_target']].copy()
common_keys['Original_Index'] = common_keys.index

# Check uniqueness in LR
lr_keys = df_lr[['key_target']].copy()
lr_keys['LR_Index'] = lr_keys.index

# Join
merged = pd.merge(lr_keys, common_keys, on=['key_target'], how='inner')



print(f"LR Rows: {len(df_lr)}")
print(f"Merged Rows: {len(merged)}")
print(f"Unique Matches: {merged['LR_Index'].nunique()}")

if len(merged) < len(df_lr):
    print("Warning: Some LR rows could not be matched uniquely.")
    
# Check for duplicates (multiple Common rows matching one LR row)
dupes = merged.duplicated(subset=['LR_Index'], keep=False)
if dupes.any():
    print(f"Ambiguous Matches found: {dupes.sum()} rows.")
    # For ambiguous, maybe try to match on more columns?
    # Let's inspect columns intersection
    cols_intersect = list(set(df_lr.columns) & set(df_common.columns))
    print(f"Intersecting Feature Columns: {len(cols_intersect)}")
    if len(cols_intersect) > 2:
        print("Retrying with all intersecting columns...")
        merged = pd.merge(df_lr[cols_intersect].reset_index(), 
                          df_common[cols_intersect].reset_index(), 
                          on=cols_intersect, 
                          suffixes=('_LR', '_Common'))
        print(f"Refined Merged Rows: {len(merged)}")
        

# Save the recovery map
# We want: LR_Index -> Matching Common_Index
# Just saving the successful matches
final_map = merged[['index_LR', 'index_Common']].rename(columns={'index_LR': 'LR_Index', 'index_Common': 'Original_Index'})
final_map.to_csv('data/processed/recovered_index_map.csv', index=False)
print("Saved recovery map to data/processed/recovered_index_map.csv")
