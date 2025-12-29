import pandas as pd
import sys
import os
sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

print("--- Checking 2020 Data Integrity ---")

# Load Data
df_lr_raw = load_data('data/processed/lr_final_prep.csv') 
df_common = load_data('data/processed/common_preprocessed.csv')
map_df = pd.read_csv('data/processed/recovered_index_map.csv')

# Align
df_lr = df_lr_raw.iloc[map_df['LR_Index']].copy().reset_index(drop=True)
original_indices = map_df['Original_Index'].values
df_lr['Year'] = df_common.loc[original_indices, 'Year'].values
df_lr['Entity'] = df_common.loc[original_indices, 'Entity'].values

TARGET = 'Value_co2_emissions_kt_by_country'

# 1. Compare 2019 vs 2020
for year in [2019, 2020]:
    df_y = df_lr[df_lr['Year'] == year]
    print(f"\n--- {year} Stats ---")
    print(f"Record Count: {len(df_y)}")
    total_co2 = df_y[TARGET].sum()
    print(f"Total CO2 Sum: {total_co2:,.0f}")
    
    # Check Major Countries
    majors = ['United States', 'China', 'India']
    for c in majors:
        row = df_y[df_y['Entity'] == c]
        if not row.empty:
            co2_val = row[TARGET].values[0]
            print(f"  {c}: {co2_val:,.0f}")
        else:
            print(f"  {c}: MISSING")

# 2. Check Zeros
zeros_2020 = df_lr[(df_lr['Year'] == 2020) & (df_lr[TARGET] == 0)]
if not zeros_2020.empty:
    print(f"\nCountries with 0 CO2 in 2020: {len(zeros_2020)}")
    print(zeros_2020['Entity'].head(5).tolist())
    
print("\nConclusion on 2020 Data Quality:")
if len(df_lr[df_lr['Year'] == 2020]) < 10:
    print("CRITICAL: Almost no data for 2020.")
elif df_lr[df_lr['Year'] == 2020][TARGET].sum() < 0.5 * df_lr[df_lr['Year'] == 2019][TARGET].sum():
    print("CRITICAL: 2020 Sum is less than half of 2019. Data likely incomplete.")
else:
    print("Data seems structurally present. R2=0 might be real divergence.")
