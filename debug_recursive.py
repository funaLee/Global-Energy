import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sys
import os

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

SPLIT_YEAR = 2015
TARGET = 'Value_co2_emissions_kt_by_country'

# Load Best Params
with open('data/results/best_hyperparameters.json', 'r') as f:
    best_params = json.load(f)

# Load Data
df_common = load_data('data/processed/common_preprocessed.csv')
df_lr = load_data('data/processed/lr_final_prep.csv')

# Align indices
common_idx = df_common.index.intersection(df_lr.index)
df_lr = df_lr.loc[common_idx]
df_lr['Year'] = df_common.loc[common_idx, 'Year']
df_lr['Entity'] = df_common.loc[common_idx, 'Entity']

# Lag Feature Logic
lag_cols = [c for c in df_lr.columns if 'lag' in c]
target_lag_col = next((c for c in lag_cols if TARGET in c), None)

# Scaling stats
unscaled_lag = df_common.loc[df_lr.index, target_lag_col]
lag_mean = unscaled_lag.mean()
lag_std = unscaled_lag.std()

def scale_value(val):
    return (val - lag_mean) / lag_std

# Cluster
cluster_cols = ['gdp_per_capita', 'Access to electricity (% of population)', 
                'Renewable energy share in the total final energy consumption (%)', 
                'Primary energy consumption per capita (kWh/person)']
df_profile = df_common[df_common['Year'] < SPLIT_YEAR].groupby('Entity')[cluster_cols].mean().dropna()
scaler_cluster = StandardScaler()
X_cluster = scaler_cluster.fit_transform(df_profile)
kmeans = KMeans(n_clusters=3, random_state=42)
df_profile['Cluster'] = kmeans.fit_predict(X_cluster)
df_lr['Cluster'] = df_lr['Entity'].map(df_profile['Cluster'])
df_lr.dropna(subset=['Cluster'], inplace=True)
df_lr['Cluster'] = df_lr['Cluster'].astype(int)

# Train
models = {}
params_lr = best_params.get('Linear Regression', {})
train_full = df_lr[df_lr['Year'] < SPLIT_YEAR]
drop_cols = [TARGET, 'Year', 'Cluster', 'Entity']
feature_cols = [c for c in df_lr.columns if c not in drop_cols]

for c in sorted(df_lr['Cluster'].unique()):
    data_c = train_full[train_full['Cluster'] == c]
    model = Ridge(**params_lr)
    model.fit(data_c[feature_cols], data_c[TARGET])
    models[c] = model

# Debug 2015 -> 2016 Transition
print("--- Debugging 2015 -> 2016 Transition ---")
test_df = df_lr[df_lr['Year'] >= 2015].copy()
y_2015_stats = {'actual': [], 'pred': []}

subset_2015 = test_df[test_df['Year'] == 2015]

# Predict 2015
for idx, row in subset_2015.iterrows():
    c = row['Cluster']
    feat = row[feature_cols].values.reshape(1, -1)
    pred = models[c].predict(feat)[0]
    y_2015_stats['actual'].append(row[TARGET])
    y_2015_stats['pred'].append(pred)

print(f"2015 Mean Actual: {np.mean(y_2015_stats['actual']):.2f}")
print(f"2015 Mean Pred: {np.mean(y_2015_stats['pred']):.2f}")
print(f"Ratio Pred/Actual: {np.mean(y_2015_stats['pred']) / np.mean(y_2015_stats['actual']):.2f}")

# Check Scaling for 2016 Input
print("\n--- Scaling Check ---")
print(f"Lag Mean: {lag_mean:.2f}, Lag Std: {lag_std:.2f}")

# Sample Country Debug
sample_entity = subset_2015.iloc[0]['Entity']
sample_row_2015 = subset_2015.iloc[0]
sample_pred_2015 = models[sample_row_2015['Cluster']].predict(sample_row_2015[feature_cols].values.reshape(1, -1))[0]
scaled_sample_pred = scale_value(sample_pred_2015)

print(f"\nSample Country: {sample_entity}")
print(f"2015 Actual: {sample_row_2015[TARGET]:.2f}")
print(f"2015 Pred: {sample_pred_2015:.2f}")
print(f"Scaled 2015 Pred (Input for 2016): {scaled_sample_pred:.4f}")

# Verify 2016 Actual Lag
rows_2016 = test_df[(test_df['Year'] == 2016) & (test_df['Entity'] == sample_entity)]
if not rows_2016.empty:
    actual_lag_2016 = rows_2016.iloc[0][target_lag_col]
    print(f"Actual Lag in 2016 Data (scaled): {actual_lag_2016:.4f}")
    
    # Reverse Scale Actual Lag
    unscaled_actual_lag = actual_lag_2016 * lag_std + lag_mean
    print(f"Unscaled Actual Lag (should match 2015 Actual): {unscaled_actual_lag:.2f}")
    
    diff = scaled_sample_pred - actual_lag_2016
    print(f"Difference in Lag Input (Pred - Actual): {diff:.4f}")
else:
    print("No 2016 data for this sample.")
