import pandas as pd
import numpy as np
import sys
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

SPLIT_YEAR = 2015
TARGET = 'Value_co2_emissions_kt_by_country'

print("--- Debugging Clustering Strategy ---")

# 1. Load Data
df_common = load_data('data/processed/common_preprocessed.csv')
df_lr_raw = load_data('data/processed/lr_final_prep.csv') 
map_df = pd.read_csv('data/processed/recovered_index_map.csv')

# Align
df_lr = df_lr_raw.iloc[map_df['LR_Index']].copy().reset_index(drop=True)
original_indices = map_df['Original_Index'].values
df_lr['Year'] = df_common.loc[original_indices, 'Year'].values
df_lr['Entity'] = df_common.loc[original_indices, 'Entity'].values

# 2. Perform Clustering (Logic from Recalc Script)
cluster_cols = ['gdp_per_capita', 'Access to electricity (% of population)', 
                'Renewable energy share in the total final energy consumption (%)', 
                'Primary energy consumption per capita (kWh/person)']

train_common = df_common[df_common['Year'] < SPLIT_YEAR]
df_profile = train_common.groupby('Entity')[cluster_cols].mean().dropna()

scaler_cluster = StandardScaler()
X_cluster = scaler_cluster.fit_transform(df_profile)

# Try n=3
kmeans = KMeans(n_clusters=3, random_state=42)
df_profile['Cluster'] = kmeans.fit_predict(X_cluster)

# Map back to DF
df_lr['Cluster'] = df_lr['Entity'].map(df_profile['Cluster'])
df_lr_clustered = df_lr.dropna(subset=['Cluster']).copy()
df_lr_clustered['Cluster'] = df_lr_clustered['Cluster'].astype(int)

# 3. Analyze Clusters
whitelist = ['China', 'United States', 'India', 'Germany', 'Japan', 'Russian Federation', 'Brazil', 'Canada']

print("\n[Cluster Breakdown]")
for c in sorted(df_lr_clustered['Cluster'].unique()):
    subset = df_lr_clustered[df_lr_clustered['Cluster'] == c]
    entities = subset['Entity'].unique()
    n_train = subset[subset['Year'] < SPLIT_YEAR].shape[0]
    n_test = subset[subset['Year'] >= SPLIT_YEAR].shape[0]
    
    print(f"\nCluster {c}:")
    print(f"  Countries: {len(entities)}")
    print(f"  Train Samples: {n_train}")
    print(f"  Test Samples: {n_test}")
    
    # Check Whitelist
    special_members = [e for e in entities if e in whitelist]
    print(f"  Giants included: {special_members}")
    
    # Check stats
    if 'gdp_per_capita' in subset.columns:
        avg_gdp = subset['gdp_per_capita'].mean()
        print(f"  Avg GDP (Scaled): {avg_gdp:.2f}")
    else:
        print("  (GDP column not in subset)")

print("\n[Hypothesis Check]")
# Are the giants isolated?
# If USA is alone in a cluster with small nations, maybe it overpowers them?
# Or if USA/China are separated, maybe that's good?
