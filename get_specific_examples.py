import pandas as pd
import numpy as np
import sys
import os
from sklearn.linear_model import Ridge

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

print("--- Fetching Specific Examples for Manual Calculation ---")
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
test_mask = (df_lr['Year'] == 2019) # Get 2019 specifically for the example

X_train = df_lr.loc[train_mask, feature_cols]
y_train = df_lr.loc[train_mask, TARGET]
X_test = df_lr.loc[test_mask, feature_cols]
y_test = df_lr.loc[test_mask, TARGET]
entities_test = df_lr.loc[test_mask, 'Entity']

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
preds = model.predict(X_test)

res = pd.DataFrame({'Entity': entities_test, 'Actual': y_test, 'Predicted': preds})

targets = ['China', 'United States', 'Vietnam', 'Tuvalu', 'Nauru']
print(f"\n--- Specific Values for {targets} in 2019 ---")
for t in targets:
    row = res[res['Entity'] == t]
    if not row.empty:
        act = row['Actual'].values[0]
        pred = row['Predicted'].values[0]
        ape = abs(act - pred)/act * 100
        print(f"Entity: {t}")
        print(f"  Actual: {act:.2f}")
        print(f"  Predicted: {pred:.2f}")
        print(f"  Diff: {act - pred:.2f}")
        print(f"  APE: {ape:.2f}%")
    else:
        print(f"Entity: {t} not found in 2019 data")
