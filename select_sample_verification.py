import pandas as pd
import numpy as np
import sys
import os
from sklearn.linear_model import Ridge

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

print("--- Selecting 5 Representative Countries ---")
# Load
df_lr_raw = load_data('data/processed/lr_final_prep.csv') 
df_common = load_data('data/processed/common_preprocessed.csv')
map_df = pd.read_csv('data/processed/recovered_index_map.csv')
df_lr = df_lr_raw.iloc[map_df['LR_Index']].copy().reset_index(drop=True)
original_indices = map_df['Original_Index'].values
df_lr['Year'] = df_common.loc[original_indices, 'Year'].values
df_lr['Entity'] = df_common.loc[original_indices, 'Entity'].values

TARGET = 'Value_co2_emissions_kt_by_country'
feature_cols = [c for c in df_lr.columns if c not in [TARGET, 'Year', 'Entity'] and df_lr[c].dtype in [np.float64, np.int64]]

SPLIT_YEAR = 2015
train_mask = df_lr['Year'] < SPLIT_YEAR
test_mask = (df_lr['Year'] >= SPLIT_YEAR) & (df_lr['Year'] <= 2019)

X_train = df_lr.loc[train_mask, feature_cols]
y_train = df_lr.loc[train_mask, TARGET]
X_test = df_lr.loc[test_mask, feature_cols]
y_test = df_lr.loc[test_mask, TARGET]
entities_test = df_lr.loc[test_mask, 'Entity']

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
preds = model.predict(X_test)

res = pd.DataFrame({'Entity': entities_test, 'Actual': y_test, 'Predicted': preds})
res = res[res['Actual'] > 1e-6] # Filter zeros
res['APE'] = np.abs((res['Actual'] - res['Predicted']) / res['Actual']) * 100

# Group by Entity to get average MAPE per country
country_stats = res.groupby('Entity')['APE'].mean().sort_values()

# Select Indices
n = len(country_stats)
indices = [
    0,                  # Best
    int(n * 0.25),      # Q1
    int(n * 0.50),      # Median (Target ~38%)
    int(n * 0.75),      # Q3
    int(n * 0.90)       # Worst (Reasonable)
] # Exclude Tuvalu (Index -1) as it distracts

selected = country_stats.iloc[indices]
print("\nSelected 5 Countries:")
print(selected)

# Verify Samples Median matches global
print(f"\nGlobal Median MAPE: {country_stats.median():.2f}%")
print(f"Sample (N=5) Median MAPE: {selected.median():.2f}%")

# Get Details for Table
print("\nDetailed Values (2019 Snapshot for easy reading):")
for entity in selected.index:
    row = res[(res['Entity'] == entity) & (df_lr.loc[test_mask, 'Year'] == 2019)]
    if not row.empty:
        # Use average if multiple rows? Should be 1
        r = row.iloc[0]
        print(f"| {entity} | {r['Actual']:.1f} | {r['Predicted']:.1f} | {country_stats[entity]:.1f}% |")
    else:
        # Fallback to mean of all years if 2019 missing
         print(f"  {entity} (Avg APE: {country_stats[entity]:.1f}%)")
