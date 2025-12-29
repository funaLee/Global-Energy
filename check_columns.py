import pandas as pd
import sys
import os

df_common = pd.read_csv('data/processed/common_preprocessed.csv')
df_lr = pd.read_csv('data/processed/lr_final_prep.csv')

TARGET = 'Value_co2_emissions_kt_by_country'
LAG = f'{TARGET}_lag1'

print("Common Columns:", [c for c in df_common.columns if 'co2' in c.lower()])
print("LR Columns:", [c for c in df_lr.columns if 'co2' in c.lower()])

if TARGET in df_lr.columns:
    print("\nLR Target Sample:")
    print(df_lr[TARGET].head().values)

if TARGET in df_common.columns:
    print("\nCommon Target Sample:")
    print(df_common[TARGET].head().values)
    
# Check Min/Max to see if scaled
print(f"\nLR range: {df_lr[TARGET].min()} - {df_lr[TARGET].max()}")
print(f"Common range: {df_common[TARGET].min()} - {df_common[TARGET].max()}")
