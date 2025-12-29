import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import sys
import os

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

print("--- Verifying One-Step R2 for 2015-2019 (Internal Test Set) ---")

# 1. Load Clean Data
df_lr_raw = load_data('data/processed/lr_final_prep.csv') 
df_common = load_data('data/processed/common_preprocessed.csv')
map_df = pd.read_csv('data/processed/recovered_index_map.csv')

df_lr = df_lr_raw.iloc[map_df['LR_Index']].copy().reset_index(drop=True)
original_indices = map_df['Original_Index'].values
df_lr['Year'] = df_common.loc[original_indices, 'Year'].values
df_lr['Entity'] = df_common.loc[original_indices, 'Entity'].values
df_lr['Cluster'] = df_common.loc[original_indices, 'Cluster'].values

TARGET = 'Value_co2_emissions_kt_by_country'
feature_cols = [c for c in df_lr.columns if c not in [TARGET, 'Year', 'Entity', 'Cluster'] and df_lr[c].dtype in [np.float64, np.int64]]

# 2. Split Train/Test
SPLIT_YEAR = 2015
train_mask = df_lr['Year'] < SPLIT_YEAR
test_mask = df_lr['Year'] >= SPLIT_YEAR

X_train = df_lr.loc[train_mask, feature_cols]
y_train = df_lr.loc[train_mask, TARGET]
X_test = df_lr.loc[test_mask, feature_cols]
y_test = df_lr.loc[test_mask, TARGET]

print(f"Test Set Usage: {df_lr.loc[test_mask, 'Year'].unique()}")
print(f"Test Set Size: {len(X_test)}")

# 3. Train & Predict (Global)
model_global = Ridge(alpha=1.0)
model_global.fit(X_train, y_train)
y_pred_global = model_global.predict(X_test)
r2_global = r2_score(y_test, y_pred_global)
print(f"Global Model (One-Step) R2: {r2_global:.4f}")

# 4. Train & Predict (Cluster)
# The report refers to Cluster-Based LR usually.
y_pred_cluster = np.zeros(len(y_test))
for c in df_lr['Cluster'].unique():
    c_train_mask = train_mask & (df_lr['Cluster'] == c)
    c_test_mask = test_mask & (df_lr['Cluster'] == c)
    
    if not c_train_mask.any() or not c_test_mask.any(): continue
    
    X_c_train = df_lr.loc[c_train_mask, feature_cols]
    y_c_train = df_lr.loc[c_train_mask, TARGET]
    X_c_test = df_lr.loc[c_test_mask, feature_cols]
    
    model_c = Ridge(alpha=1.0)
    model_c.fit(X_c_train, y_c_train)
    y_pred_cluster[y_test.index.isin(df_lr[c_test_mask].index)] = model_c.predict(X_c_test) # Tricky indexing
    
    # Simpler loop
    # Re-index
    
preds_cluster = []
actuals = []

for c in sorted(df_lr['Cluster'].unique()):
    c_train_df = df_lr[(df_lr['Cluster'] == c) & (df_lr['Year'] < SPLIT_YEAR)]
    c_test_df = df_lr[(df_lr['Cluster'] == c) & (df_lr['Year'] >= SPLIT_YEAR)]
    
    if c_train_df.empty or c_test_df.empty: continue
    
    m = Ridge(alpha=1.0)
    m.fit(c_train_df[feature_cols], c_train_df[TARGET])
    p = m.predict(c_test_df[feature_cols])
    
    preds_cluster.extend(p)
    actuals.extend(c_test_df[TARGET].values)

r2_cluster = r2_score(actuals, preds_cluster)
print(f"Cluster Model (One-Step) R2: {r2_cluster:.4f}")
