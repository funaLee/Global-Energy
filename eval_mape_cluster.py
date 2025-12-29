import pandas as pd
import numpy as np
import sys
import os
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

print("--- Calculating Phase 3 (Cluster) Metrics ---")

# 1. Load Data
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

# 2. Re-Cluster (Training Data Only)
cluster_cols = ['gdp_per_capita', 'Access to electricity (% of population)', 
                'Renewable energy share in the total final energy consumption (%)', 
                'Primary energy consumption per capita (kWh/person)']

train_common = df_common[df_common['Year'] < SPLIT_YEAR]
# Ensure we only use entities present in LR dataset?
# Actually broadly using all common valid data for profiling is safer
df_profile = train_common.groupby('Entity')[cluster_cols].mean().dropna()

scaler_c = StandardScaler()
X_c = scaler_c.fit_transform(df_profile)
kmeans = KMeans(n_clusters=3, random_state=42)
df_profile['Cluster'] = kmeans.fit_predict(X_c)

# Map back
df_lr['Cluster'] = df_lr['Entity'].map(df_profile['Cluster'])
# Drop unclustered (if any)
df_lr = df_lr.dropna(subset=['Cluster'])
df_lr['Cluster'] = df_lr['Cluster'].astype(int)

# 3. Cluster Ridge Training
models = {}
for c in range(3):
    c_mask = train_mask & (df_lr['Cluster'] == c)
    X_tr = df_lr.loc[c_mask, feature_cols]
    y_tr = df_lr.loc[c_mask, TARGET]
    m = Ridge(alpha=10.0) # Phase 3 param
    m.fit(X_tr, y_tr)
    models[c] = m

# 4. Evaluate on Test (Overall)
preds = np.zeros(test_mask.sum())
y_true = df_lr.loc[test_mask, TARGET]
unique_test_indices = df_lr.loc[test_mask].index

# We need to fill preds in correct order
# Easier: Iterate df_test rows
df_test = df_lr[test_mask].copy()
pred_list = []

for idx, row in df_test.iterrows():
    c = int(row['Cluster'])
    if c in models:
        # Extract features
        # Vectorize is faster but loop is safer for explicit logic
        feat_vec = row[feature_cols].values.reshape(1, -1)
        p = models[c].predict(feat_vec)[0]
        pred_list.append(p)
    else:
        pred_list.append(0) # Fallback

df_test['Prediction'] = pred_list

# Metrics
y_pred = df_test['Prediction']
y_true = df_test[TARGET]

r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# MAPE
valid = y_true > 1e-6
ape = np.abs((y_true[valid] - y_pred[valid]) / y_true[valid]) * 100
median_mape = np.median(ape)
mean_mape = np.mean(ape)

print(f"Cluster Model (Phase 3):")
print(f"  R2: {r2:.4f}")
print(f"  RMSE: {rmse:.2f}")
print(f"  Median MAPE: {median_mape:.2f}%")
print(f"  Mean MAPE: {mean_mape:.2f}%")
