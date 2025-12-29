import pandas as pd
import numpy as np
import sys
import os
from sklearn.linear_model import Ridge

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

print("--- Debugging High MAPE ---")
# Load (Simplified for brevity, assuming same load logic)
df_lr_raw = load_data('data/processed/lr_final_prep.csv') 
df_common = load_data('data/processed/common_preprocessed.csv')
map_df = pd.read_csv('data/processed/recovered_index_map.csv')
df_lr = df_lr_raw.iloc[map_df['LR_Index']].copy().reset_index(drop=True)
original_indices = map_df['Original_Index'].values
df_lr['Year'] = df_common.loc[original_indices, 'Year'].values
df_lr['Entity'] = df_common.loc[original_indices, 'Entity'].values
SPLIT_YEAR = 2015
train_mask = df_lr['Year'] < SPLIT_YEAR
test_mask = (df_lr['Year'] >= SPLIT_YEAR) & (df_lr['Year'] <= 2019)
TARGET = 'Value_co2_emissions_kt_by_country'
feature_cols = [c for c in df_lr.columns if c not in [TARGET, 'Year', 'Entity'] and df_lr[c].dtype in [np.float64, np.int64]]
X_train = df_lr.loc[train_mask, feature_cols]
y_train = df_lr.loc[train_mask, TARGET]
X_test = df_lr.loc[test_mask, feature_cols]
y_test = df_lr.loc[test_mask, TARGET]
entities_test = df_lr.loc[test_mask, 'Entity']

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
preds = model.predict(X_test)

res = pd.DataFrame({'Entity': entities_test, 'Actual': y_test, 'Predicted': preds})
res['APE'] = np.abs((res['Actual'] - res['Predicted']) / res['Actual']) * 100
entity_mapes = res.groupby('Entity')['APE'].mean().sort_values(ascending=False)

print("\nTop 10 Worst MAPEs:")
print(entity_mapes.head(10))

print("\nTop 10 Best MAPEs:")
print(entity_mapes.tail(10))
