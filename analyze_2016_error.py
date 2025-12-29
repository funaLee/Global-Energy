import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import sys
import os

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

TARGET = 'Value_co2_emissions_kt_by_country'
LAG_COL = f'{TARGET}_lag1'

# Empirical Scaler (from previous fix)
EMPIRICAL_SLOPE = 2.829519e-05
EMPIRICAL_INTERCEPT = -0.5886

def scale_value(val):
    return (val * EMPIRICAL_SLOPE) + EMPIRICAL_INTERCEPT

# Load Data
df_common = load_data('data/processed/common_preprocessed.csv')
df_lr = load_data('data/processed/lr_final_prep.csv')

# Align & Restore columns
common_idx = df_common.index.intersection(df_lr.index)
df_lr = df_lr.loc[common_idx]
if 'Entity' not in df_common.columns: df_common = df_common.reset_index()
df_lr['Year'] = df_common.loc[common_idx, 'Year']
df_lr['Entity'] = df_common.loc[common_idx, 'Entity']
df_lr = df_lr.reset_index(drop=True)

# Load Params
with open('data/results/best_hyperparameters.json', 'r') as f:
    best_params = json.load(f)
params = best_params.get('Linear Regression', {})

# Train Models (Cluster-based)
# Re-create clusters (simplified for speed/reproducibility using saved csv column if possible, 
# but notebook re-runs clustering. We'll trust df_lr has the right structure if we follow notebook logic.
# Wait, df_lr doesn't have 'Cluster' column saved in it? 
# The notebook calculated it. We need to recalculate or load.
# To be safe/fast, let's just train one Global model first to check lag sensitivity, 
# or try to replicate the clustering quickly.
# Replicating Clustering:
cluster_cols = ['gdp_per_capita', 'Access to electricity (% of population)', 
                'Renewable energy share in the total final energy consumption (%)', 
                'Primary energy consumption per capita (kWh/person)']
df_profile = df_common[df_common['Year'] < 2015].groupby('Entity')[cluster_cols].mean().dropna()

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df_profile['Cluster'] = kmeans.fit_predict(StandardScaler().fit_transform(df_profile))
df_lr['Cluster'] = df_lr['Entity'].map(df_profile['Cluster']).fillna(-1).astype(int)
df_lr = df_lr[df_lr['Cluster'] != -1]

models = {}
train_df = df_lr[df_lr['Year'] < 2015]
drop_cols = [TARGET, 'Year', 'Cluster', 'Entity']
feat_cols = [c for c in df_lr.columns if c not in drop_cols]

print("Coefficients for Lag Feature per Cluster:")
for c in sorted(train_df['Cluster'].unique()):
    m = Ridge(**params)
    d = train_df[train_df['Cluster'] == c]
    m.fit(d[feat_cols], d[TARGET])
    models[c] = m
    
    # Find coefficient for lag
    if LAG_COL in feat_cols:
        idx = feat_cols.index(LAG_COL)
        coef = m.coef_[idx]
        print(f"Cluster {c}: Lag Coefficient = {coef:.4f}")

# Analyze 2016
print("\n--- Top Error Contributors in 2016 ---")
df_2015 = df_lr[df_lr['Year'] == 2015].copy()
df_2016 = df_lr[df_lr['Year'] == 2016].copy()

errors = []

for entity in df_2016['Entity'].unique():
    row_16 = df_2016[df_2016['Entity'] == entity]
    row_15 = df_2015[df_2015['Entity'] == entity]
    
    if row_16.empty or row_15.empty: continue
    
    row_16 = row_16.iloc[0]
    row_15 = row_15.iloc[0]
    c = row_16['Cluster']
    
    # 1. Predict 2015 (to get input for 2016)
    feat_15 = row_15[feat_cols].values.reshape(1, -1)
    pred_15 = models[c].predict(feat_15)[0]
    
    # 2. Construct Recursive Input for 2016
    scaled_pred_15 = scale_value(pred_15)
    
    # 3. Predict 2016 (Recursive)
    feat_16_rec = row_16[feat_cols].copy().values # This has ACTUAL lag
    # Replace lag with recursive
    lag_idx = feat_cols.index(LAG_COL)
    feat_16_rec[lag_idx] = scaled_pred_15
    pred_16_rec = models[c].predict(feat_16_rec.reshape(1, -1))[0]
    
    # 4. Predict 2016 (Teacher Forcing - for ref)
    pred_16_tf = models[c].predict(row_16[feat_cols].values.reshape(1, -1))[0]
    
    actual_16 = row_16[TARGET]
    
    err_rec = abs(pred_16_rec - actual_16)
    err_tf = abs(pred_16_tf - actual_16)
    
    # Input Error (Did we predict 2015 wrong?)
    actual_15 = row_15[TARGET]
    err_15 = pred_15 - actual_15 # Raw error
    
    errors.append({
        'Entity': entity,
        'Cluster': c,
        'Actual_2016': actual_16,
        'Pred_2016_Rec': pred_16_rec,
        'Error_Rec': err_rec,
        'Error_TF': err_tf,
        'Input_Error_fr_2015': err_15,
        'Lag_Coef': models[c].coef_[lag_idx]
    })

res_df = pd.DataFrame(errors)
res_df = res_df.sort_values('Error_Rec', ascending=False)



# Deep Dive Analysis for Malta (or top error)
target_entity = 'Malta'
print(f"\n--- Deep Dive: {target_entity} ---")
row_16 = df_2016[df_2016['Entity'] == target_entity].iloc[0]
row_15 = df_2015[df_2015['Entity'] == target_entity].iloc[0]
c = row_16['Cluster']
model = models[c]

# 1. Teacher Forcing Details
feat_tf = row_16[feat_cols].values.reshape(1, -1)
pred_tf = model.predict(feat_tf)[0]
lag_idx = feat_cols.index(LAG_COL)
actual_lag_val = feat_tf[0][lag_idx]
lag_contribution_tf = actual_lag_val * model.coef_[lag_idx]
print(f"TF Pred: {pred_tf:.2f} (Actual: {row_16[TARGET]:.2f})")
print(f"  TF Lag Value (Scaled Actual): {actual_lag_val:.4f}")
print(f"  TF Lag Contribution: {lag_contribution_tf:.2f}")

# 2. Recursive Details
feat_15 = row_15[feat_cols].values.reshape(1, -1)
pred_15 = model.predict(feat_15)[0]
print(f"Pred 2015: {pred_15:.2f} (Actual 15: {row_15[TARGET]:.2f})")

scaled_pred_15 = scale_value(pred_15)
lag_contribution_rec = scaled_pred_15 * model.coef_[lag_idx]

feat_rec = feat_tf.copy()
feat_rec[0][lag_idx] = scaled_pred_15
pred_rec = model.predict(feat_rec)[0]

print(f"Rec Pred: {pred_rec:.2f}")
print(f"  Rec Lag Value (Scaled Pred): {scaled_pred_15:.4f}")
print(f"  Rec Lag Contribution: {lag_contribution_rec:.2f}")
print(f"  Delta Contribution: {lag_contribution_rec - lag_contribution_tf:.2f}")

print("\n--- Coefficients check ---")
print(f"Lag Coef: {model.coef_[lag_idx]:.2f}")

