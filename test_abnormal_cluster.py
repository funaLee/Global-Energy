import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

SPLIT_YEAR = 2015
TARGET = 'Value_co2_emissions_kt_by_country'
WHITELIST = ['China', 'United States', 'India', 'Japan', 'Russian Federation', 'Germany', 'Brazil', 'Canada']

print("--- Testing 'Abnormal Cluster' Strategy ---")

# 1. Load Data
df_common = load_data('data/processed/common_preprocessed.csv')
df_lr_raw = load_data('data/processed/lr_final_prep.csv') 
map_df = pd.read_csv('data/processed/recovered_index_map.csv')

# Align
df_lr = df_lr_raw.iloc[map_df['LR_Index']].copy().reset_index(drop=True)
original_indices = map_df['Original_Index'].values
df_lr['Year'] = df_common.loc[original_indices, 'Year'].values
df_lr['Entity'] = df_common.loc[original_indices, 'Entity'].values

# Features
drop_cols = [TARGET, 'Year', 'Entity', 'Cluster'] 
feat_cols = [c for c in df_lr.columns if c not in drop_cols]

# 2. Hybrid Clustering
# Step A: Identify Abnormal Giants
print(f"Assigning 'Abnormal' Cluster 99 to: {WHITELIST}")
df_lr['Cluster'] = -1 # Default

# Assign Giants
mask_giants = df_lr['Entity'].isin(WHITELIST)
df_lr.loc[mask_giants, 'Cluster'] = 99

# Step B: Cluster the Rest
# Get profiles for Normal countries
normal_mask = ~df_common['Entity'].isin(WHITELIST)
train_common_normal = df_common[(df_common['Year'] < SPLIT_YEAR) & normal_mask]

cluster_cols = ['gdp_per_capita', 'Access to electricity (% of population)', 
                'Renewable energy share in the total final energy consumption (%)', 
                'Primary energy consumption per capita (kWh/person)']

df_profile = train_common_normal.groupby('Entity')[cluster_cols].mean().dropna()

# Cluster into 3 groups
scaler_cluster = StandardScaler()
X_cluster = scaler_cluster.fit_transform(df_profile)

kmeans = KMeans(n_clusters=3, random_state=42)
df_profile['Cluster'] = kmeans.fit_predict(X_cluster)

# Map back (Only for those with Cluster == -1)
# We map normal entities to their cluster
normal_cluster_map = df_profile['Cluster'].to_dict()
df_lr.loc[~mask_giants, 'Cluster'] = df_lr.loc[~mask_giants, 'Entity'].map(normal_cluster_map)

# Drop any that failed clustering (should be none if data exists)
df_lr_clustered = df_lr.dropna(subset=['Cluster']).copy()
df_lr_clustered['Cluster'] = df_lr_clustered['Cluster'].astype(int)

# 3. Evaluate High-Level Stats
print("\n[Cluster Counts]")
print(df_lr_clustered['Cluster'].value_counts())

# 4. Train Models per Cluster
train_c = df_lr_clustered[df_lr_clustered['Year'] < SPLIT_YEAR]
test_c = df_lr_clustered[df_lr_clustered['Year'] >= SPLIT_YEAR]

weighted_r2 = 0
total_samples = 0
results = []

print("\n[Performance per Cluster]")
for c in sorted(train_c['Cluster'].unique()):
    tr = train_c[train_c['Cluster'] == c]
    te = test_c[test_c['Cluster'] == c]
    
    if len(te) == 0: continue
    
    X_tr = tr[feat_cols]
    y_tr = tr[TARGET]
    X_te = te[feat_cols]
    y_te = te[TARGET]
    
    # Ridge Regression
    model = Ridge(alpha=10.0) 
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    
    r2 = r2_score(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    
    n = len(te)
    weighted_r2 += r2 * n
    total_samples += n
    
    # Description
    desc = "Abnormal Giants" if c == 99 else "Normal Group"
    print(f"Cluster {c} ({desc}): R2={r2:.4f} (RMSE={rmse:,.0f}, N={n})")
    results.append({'Cluster': c, 'R2': r2, 'N': n})

final_r2 = weighted_r2 / total_samples if total_samples > 0 else 0
print(f"\nWeighted R2 (Hybrid Strategy): {final_r2:.4f}")

# Compare with Global
global_r2_baseline = 0.7817 
print(f"Global Baseline: {global_r2_baseline:.4f}")
print(f"Improvement: {(final_r2 - global_r2_baseline)*100:.2f}%")
