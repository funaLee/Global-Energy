import pandas as pd
import sys
import os

df_common = pd.read_csv('data/processed/common_preprocessed.csv')
df_lr = pd.read_csv('data/processed/lr_final_prep.csv')

print("Common Head Index:", df_common.index.tolist()[:5])
print("LR Head Index:", df_lr.index.tolist()[:5])

print(f"Common Shape: {df_common.shape}")
print(f"LR Shape: {df_lr.shape}")

# Check content match?
# lr_final_prep usually has scaled features.
# common_preprocessed has raw.
# If index is lost, we can't match.

# Check if 'Unnamed: 0' exists (saved index)
print("LR Columns:", df_lr.columns.tolist()[:5])
