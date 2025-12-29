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

print("--- Recalculating Metrics with Correct Alignment ---")

# 1. Load Data & Map
df_common = load_data('data/processed/common_preprocessed.csv')
df_lr_raw = load_data('data/processed/lr_final_prep.csv') # Has RangeIndex
map_df = pd.read_csv('data/processed/recovered_index_map.csv')

print(f"Loaded LR Raw: {df_lr_raw.shape}")
print(f"Loaded Map: {map_df.shape}")

# 2. Align df_lr
# We only keep rows that were successfully mapped
df_lr = df_lr_raw.iloc[map_df['LR_Index']].copy().reset_index(drop=True)
original_indices = map_df['Original_Index'].values

# Assign proper metadata from Common
df_lr['Year'] = df_common.loc[original_indices, 'Year'].values
df_lr['Entity'] = df_common.loc[original_indices, 'Entity'].values

print(f"Aligned LR Shape: {df_lr.shape}")

# Verify Whitelist Presence
major_countries = ['United States', 'China', 'India', 'Germany']
print("\n[Verifying Major Economies Presence in Final Set]")
for c in major_countries:
    count = df_lr[df_lr['Entity'] == c].shape[0]
    print(f"  {c}: {count} records")

print(f"Year Sample: {df_lr['Year'].head().tolist()}")

# 3. GLOBAL RIDGE REGRESSION
print("\n[1] Global Linear Regression (Phase 1 Correction)")
train_mask = df_lr['Year'] < SPLIT_YEAR
test_mask = df_lr['Year'] >= SPLIT_YEAR

train_df = df_lr[train_mask]
test_df = df_lr[test_mask]

# Features: Drop non-features
drop_cols = [TARGET, 'Year', 'Entity', 'Cluster'] # Cluster might not exist yet
feat_cols = [c for c in df_lr.columns if c not in drop_cols]

X_train = train_df[feat_cols]
y_train = train_df[TARGET]
X_test = test_df[feat_cols]
y_test = test_df[TARGET]

print(f"Train Size: {len(X_train)}, Test Size: {len(X_test)}")

global_model = Ridge(alpha=1.0) # Using Phase 1 param
global_model.fit(X_train, y_train)
global_preds = global_model.predict(X_test)
global_r2 = r2_score(y_test, global_preds)
global_rmse = np.sqrt(mean_squared_error(y_test, global_preds))

print(f"Global LR R2: {global_r2:.4f}")
print(f"Global LR RMSE: {global_rmse:.4f}")


# 4. CLUSTER RIDGE REGRESSION (Phase 3 Correction)
print("\n[2] Cluster-Based Linear Regression (Phase 3 Correction)")

# Re-generate clusters on TRAINING data only
cluster_cols = ['gdp_per_capita', 'Access to electricity (% of population)', 
                'Renewable energy share in the total final energy consumption (%)', 
                'Primary energy consumption per capita (kWh/person)']

# Get profiles from COMMON (all entities available in training period)
# We use df_common for clustering to define the space, then map to df_lr entities
train_common = df_common[df_common['Year'] < SPLIT_YEAR]
df_profile = train_common.groupby('Entity')[cluster_cols].mean().dropna()

scaler_cluster = StandardScaler()
X_cluster = scaler_cluster.fit_transform(df_profile)

kmeans = KMeans(n_clusters=3, random_state=42)
df_profile['Cluster'] = kmeans.fit_predict(X_cluster)

# Map clusters to df_lr
df_lr['Cluster'] = df_lr['Entity'].map(df_profile['Cluster'])
df_lr_clustered = df_lr.dropna(subset=['Cluster']).copy()
df_lr_clustered['Cluster'] = df_lr_clustered['Cluster'].astype(int)

# Split Train/Test again (subset might strictly be smaller if entities dropped)
train_c = df_lr_clustered[df_lr_clustered['Year'] < SPLIT_YEAR]
test_c = df_lr_clustered[df_lr_clustered['Year'] >= SPLIT_YEAR]

weighted_r2 = 0
total_samples = 0
cluster_results = []

for c in sorted(train_c['Cluster'].unique()):
    tr = train_c[train_c['Cluster'] == c]
    te = test_c[test_c['Cluster'] == c]
    
    if len(te) == 0: continue
    
    X_tr = tr[feat_cols]
    y_tr = tr[TARGET]
    X_te = te[feat_cols]
    y_te = te[TARGET]
    
    model = Ridge(alpha=10.0) # Phase 3 Tuned Param
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    
    r2 = r2_score(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    
    n = len(te)
    weighted_r2 += r2 * n
    total_samples += n
    
    cluster_results.append({'Cluster': c, 'R2': r2, 'N': n})
    print(f"Cluster {c}: R2={r2:.4f} (N={n})")

final_cluster_r2 = weighted_r2 / total_samples if total_samples > 0 else 0
print(f"Weighted Cluster LR R2: {final_cluster_r2:.4f}")

# Save these results for the report
with open('data/results/corrected_metrics.txt', 'w') as f:
    f.write(f"Global_R2={global_r2:.4f}\n")
    f.write(f"Cluster_R2={final_cluster_r2:.4f}\n")
