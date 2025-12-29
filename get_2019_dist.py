import pandas as pd
import numpy as np
import sys
import os
from sklearn.linear_model import Ridge

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

print("--- 2019 Distribution for Proof Table ---")
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

# Train: < 2015, Test: == 2019
X_train = df_lr.loc[df_lr['Year'] < 2015, feature_cols]
y_train = df_lr.loc[df_lr['Year'] < 2015, TARGET]
X_test = df_lr.loc[df_lr['Year'] == 2019, feature_cols]
y_test = df_lr.loc[df_lr['Year'] == 2019, TARGET]
entities = df_lr.loc[df_lr['Year'] == 2019, 'Entity']

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
preds = model.predict(X_test)

res = pd.DataFrame({'Entity': entities, 'Actual': y_test, 'Predicted': preds})
res['APE'] = np.abs((res['Actual'] - res['Predicted']) / res['Actual']) * 100
res = res.sort_values('APE')

# Percentiles
n = len(res)
indices = [0, int(n*0.25), int(n*0.50), int(n*0.75), int(n*0.90)]
selection = res.iloc[indices]

print(selection)
print(f"Global 2019 Median APE: {res['APE'].median():.2f}%")
