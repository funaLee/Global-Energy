# Notebook 12: Hybrid Model (LR + XGBoost on Residuals)
# "Công thức bí mật": Dự báo = Linear Regression (Xu hướng) + XGBoost (Phần dư)

"""
HYBRID MODEL IMPLEMENTATION
============================
- Train LR to capture main trend (CO2 ~ GDP, Energy, Lag...)
- Train XGBoost on LR's residuals to capture non-linear patterns
- Final prediction = LR prediction + XGBoost residual prediction

VARIANTS:
1. Global Hybrid (no clustering)
2. Global Hybrid + Hyperparameter Tuning  
3. Cluster-Based Hybrid (K-Means)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

print("=" * 70)
print("HYBRID MODEL: Linear Regression + XGBoost Residuals")
print("=" * 70)

# ===========================
# 1. LOAD DATA
# ===========================
SPLIT_YEAR = 2015
TARGET = 'Value_co2_emissions_kt_by_country'

df_lr = pd.read_csv('data/processed/lr_final_prep.csv')
df_common = load_data('data/processed/common_preprocessed.csv')
map_df = pd.read_csv('data/processed/recovered_index_map.csv')

# Add Year and Entity back
original_indices = map_df['Original_Index'].values
df_lr['Year'] = df_common.loc[original_indices, 'Year'].values
df_lr['Entity'] = df_common.loc[original_indices, 'Entity'].values

print(f"Data shape: {df_lr.shape}")
print(f"Years: {df_lr['Year'].min()} - {df_lr['Year'].max()}")
print(f"Unique Entities: {df_lr['Entity'].nunique()}")

# ===========================
# 2. TRAIN/TEST SPLIT
# ===========================
feature_cols = [c for c in df_lr.columns if c not in [TARGET, 'Year', 'Entity']]

train_mask = df_lr['Year'] < SPLIT_YEAR
test_mask = (df_lr['Year'] >= SPLIT_YEAR) & (df_lr['Year'] <= 2019)

X_train = df_lr.loc[train_mask, feature_cols]
y_train = df_lr.loc[train_mask, TARGET]
X_test = df_lr.loc[test_mask, feature_cols]
y_test = df_lr.loc[test_mask, TARGET]
test_entities = df_lr.loc[test_mask, 'Entity']

print(f"\nTrain: {len(X_train)} samples")
print(f"Test: {len(X_test)} samples")

# ===========================
# 3. HELPER FUNCTIONS
# ===========================
def calculate_metrics(y_true, y_pred, entities):
    """Calculate R2 and Median MAPE"""
    r2 = r2_score(y_true, y_pred)
    
    # Per-entity MAPE
    df_temp = pd.DataFrame({
        'Entity': entities.values,
        'Actual': y_true.values,
        'Pred': y_pred
    })
    df_temp['APE'] = np.abs(df_temp['Actual'] - df_temp['Pred']) / np.abs(df_temp['Actual'].replace(0, np.nan)) * 100
    entity_mape = df_temp.groupby('Entity')['APE'].mean()
    median_mape = entity_mape.median()
    
    return r2, median_mape

# ===========================
# 4. BASELINE MODELS
# ===========================
print("\n" + "=" * 70)
print("BASELINE MODELS")
print("=" * 70)

# 4.1 Standalone Ridge
lr_model = Ridge(alpha=10.0)
lr_model.fit(X_train, y_train)
lr_preds_test = lr_model.predict(X_test)
r2_lr, mape_lr = calculate_metrics(y_test, lr_preds_test, test_entities)
print(f"\n[Standalone Ridge LR]")
print(f"  R² = {r2_lr:.4f}, Median MAPE = {mape_lr:.2f}%")

# 4.2 Standalone XGBoost
xgb_model = XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.1, 
                          subsample=0.7, colsample_bytree=0.7, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
xgb_preds_test = xgb_model.predict(X_test)
r2_xgb, mape_xgb = calculate_metrics(y_test, xgb_preds_test, test_entities)
print(f"\n[Standalone XGBoost]")
print(f"  R² = {r2_xgb:.4f}, Median MAPE = {mape_xgb:.2f}%")

# ===========================
# 5. HYBRID MODEL (GLOBAL)
# ===========================
print("\n" + "=" * 70)
print("HYBRID MODEL: LR + XGBoost on Residuals")
print("=" * 70)

# 5.1 Train LR and get residuals on training set
lr_preds_train = lr_model.predict(X_train)
residuals_train = y_train - lr_preds_train

print(f"\nTrain Residuals Stats:")
print(f"  Mean: {residuals_train.mean():.2f}")
print(f"  Std: {residuals_train.std():.2f}")
print(f"  Min: {residuals_train.min():.2f}, Max: {residuals_train.max():.2f}")

# 5.2 Train XGBoost on residuals
xgb_residual_model = XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.1,
                                   subsample=0.7, colsample_bytree=0.7, random_state=42, n_jobs=-1)
xgb_residual_model.fit(X_train, residuals_train)

# 5.3 Predict: Hybrid = LR + XGBoost(residual)
lr_preds_test = lr_model.predict(X_test)
residual_preds_test = xgb_residual_model.predict(X_test)
hybrid_preds_test = lr_preds_test + residual_preds_test

r2_hybrid, mape_hybrid = calculate_metrics(y_test, hybrid_preds_test, test_entities)
print(f"\n[Hybrid Global (LR + XGB Residuals)]")
print(f"  R² = {r2_hybrid:.4f}, Median MAPE = {mape_hybrid:.2f}%")

# ===========================
# 6. HYBRID MODEL + HYPERPARAMETER TUNING
# ===========================
print("\n" + "=" * 70)
print("HYBRID MODEL + HYPERPARAMETER TUNING")
print("=" * 70)

# Sort by Year for TimeSeriesSplit
train_data = df_lr[train_mask].sort_values('Year')
X_train_sorted = train_data[feature_cols]
y_train_sorted = train_data[TARGET]

# 6.1 Tune LR alpha
print("\nTuning Ridge alpha...")
tscv = TimeSeriesSplit(n_splits=3)
lr_params = {'alpha': [0.1, 1.0, 10.0, 50.0, 100.0]}
lr_search = GridSearchCV(Ridge(), lr_params, cv=tscv, scoring='r2', n_jobs=-1)
lr_search.fit(X_train_sorted, y_train_sorted)
best_lr_alpha = lr_search.best_params_['alpha']
print(f"  Best LR alpha: {best_lr_alpha}")

# 6.2 Re-train LR with best alpha
lr_tuned = Ridge(alpha=best_lr_alpha)
lr_tuned.fit(X_train, y_train)
lr_preds_train_tuned = lr_tuned.predict(X_train)
residuals_train_tuned = y_train - lr_preds_train_tuned

# 6.3 Tune XGBoost for residuals
print("\nTuning XGBoost for residuals...")
xgb_params = {
    'n_estimators': [100, 300, 500],
    'max_depth': [2, 3, 5],
    'learning_rate': [0.01, 0.05, 0.1]
}
xgb_search = GridSearchCV(XGBRegressor(subsample=0.7, colsample_bytree=0.7, random_state=42, n_jobs=-1),
                           xgb_params, cv=tscv, scoring='r2', n_jobs=-1)
xgb_search.fit(X_train_sorted, residuals_train_tuned.loc[X_train_sorted.index])
best_xgb_params = xgb_search.best_params_
print(f"  Best XGB params: {best_xgb_params}")

# 6.4 Train tuned XGBoost on residuals
xgb_residual_tuned = XGBRegressor(**best_xgb_params, subsample=0.7, colsample_bytree=0.7, 
                                   random_state=42, n_jobs=-1)
xgb_residual_tuned.fit(X_train, residuals_train_tuned)

# 6.5 Predict with tuned hybrid
lr_preds_test_tuned = lr_tuned.predict(X_test)
residual_preds_test_tuned = xgb_residual_tuned.predict(X_test)
hybrid_tuned_preds_test = lr_preds_test_tuned + residual_preds_test_tuned

r2_hybrid_tuned, mape_hybrid_tuned = calculate_metrics(y_test, hybrid_tuned_preds_test, test_entities)
print(f"\n[Hybrid Tuned (LR α={best_lr_alpha} + XGB {best_xgb_params})]")
print(f"  R² = {r2_hybrid_tuned:.4f}, Median MAPE = {mape_hybrid_tuned:.2f}%")

# ===========================
# 7. HYBRID MODEL + K-MEANS CLUSTERING
# ===========================
print("\n" + "=" * 70)
print("HYBRID MODEL + K-MEANS CLUSTERING")
print("=" * 70)

# 7.1 Create country profiles for clustering (train data only)
cluster_cols = [
    'gdp_per_capita',
    'Access to electricity (% of population)',
    'Renewable energy share in the total final energy consumption (%)',
    'Primary energy consumption per capita (kWh/person)'
]

# Check if columns exist in df_common
available_cluster_cols = [c for c in cluster_cols if c in df_common.columns]
print(f"Using cluster columns: {available_cluster_cols}")

df_profile = df_common[df_common['Year'] < SPLIT_YEAR].groupby('Entity')[available_cluster_cols].mean().dropna()

# Scale and cluster
scaler_cluster = StandardScaler()
X_cluster = scaler_cluster.fit_transform(df_profile)
kmeans = KMeans(n_clusters=3, random_state=42)
df_profile['Cluster'] = kmeans.fit_predict(X_cluster)

# Map clusters to main data
df_lr['Cluster'] = df_lr['Entity'].map(df_profile['Cluster'])
df_lr_clustered = df_lr.dropna(subset=['Cluster']).copy()
df_lr_clustered['Cluster'] = df_lr_clustered['Cluster'].astype(int)

print(f"\nClustered data shape: {df_lr_clustered.shape}")
print(f"Cluster distribution:\n{df_lr_clustered.groupby('Cluster').size()}")

# 7.2 Train cluster-specific hybrid models
hybrid_cluster_models = {}
all_cluster_predictions = []

for c in sorted(df_lr_clustered['Cluster'].unique()):
    cluster_data = df_lr_clustered[df_lr_clustered['Cluster'] == c]
    
    train_c = cluster_data[cluster_data['Year'] < SPLIT_YEAR]
    test_c = cluster_data[(cluster_data['Year'] >= SPLIT_YEAR) & (cluster_data['Year'] <= 2019)]
    
    if len(train_c) < 20 or len(test_c) < 5:
        print(f"  Cluster {c}: Not enough data, skipping")
        continue
    
    X_train_c = train_c[feature_cols]
    y_train_c = train_c[TARGET]
    X_test_c = test_c[feature_cols]
    y_test_c = test_c[TARGET]
    test_entities_c = test_c['Entity']
    
    # Train LR
    lr_c = Ridge(alpha=best_lr_alpha)
    lr_c.fit(X_train_c, y_train_c)
    
    # Get residuals and train XGBoost
    residuals_c = y_train_c - lr_c.predict(X_train_c)
    xgb_c = XGBRegressor(**best_xgb_params, subsample=0.7, colsample_bytree=0.7, 
                          random_state=42, n_jobs=-1)
    xgb_c.fit(X_train_c, residuals_c)
    
    # Predict
    lr_pred_c = lr_c.predict(X_test_c)
    xgb_pred_c = xgb_c.predict(X_test_c)
    hybrid_pred_c = lr_pred_c + xgb_pred_c
    
    r2_c, mape_c = calculate_metrics(y_test_c, hybrid_pred_c, test_entities_c)
    print(f"  Cluster {c}: R² = {r2_c:.4f}, Median MAPE = {mape_c:.2f}%, N = {len(test_c)}")
    
    # Store predictions for overall calculation
    all_cluster_predictions.append(pd.DataFrame({
        'Entity': test_entities_c.values,
        'Actual': y_test_c.values,
        'Pred': hybrid_pred_c
    }))

# 7.3 Calculate overall cluster-based metrics
if all_cluster_predictions:
    all_preds_df = pd.concat(all_cluster_predictions, ignore_index=True)
    r2_cluster_hybrid = r2_score(all_preds_df['Actual'], all_preds_df['Pred'])
    
    all_preds_df['APE'] = np.abs(all_preds_df['Actual'] - all_preds_df['Pred']) / np.abs(all_preds_df['Actual'].replace(0, np.nan)) * 100
    entity_mape = all_preds_df.groupby('Entity')['APE'].mean()
    mape_cluster_hybrid = entity_mape.median()
    
    print(f"\n[Hybrid + K-Means (Cluster-Specific)]")
    print(f"  R² = {r2_cluster_hybrid:.4f}, Median MAPE = {mape_cluster_hybrid:.2f}%")

# ===========================
# 8. SUMMARY COMPARISON
# ===========================
print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

results = pd.DataFrame([
    {'Model': 'Standalone Ridge LR', 'R²': r2_lr, 'Median MAPE (%)': mape_lr},
    {'Model': 'Standalone XGBoost', 'R²': r2_xgb, 'Median MAPE (%)': mape_xgb},
    {'Model': 'Hybrid Global (LR + XGB)', 'R²': r2_hybrid, 'Median MAPE (%)': mape_hybrid},
    {'Model': 'Hybrid Tuned', 'R²': r2_hybrid_tuned, 'Median MAPE (%)': mape_hybrid_tuned},
])

if all_cluster_predictions:
    results = pd.concat([results, pd.DataFrame([
        {'Model': 'Hybrid + K-Means', 'R²': r2_cluster_hybrid, 'Median MAPE (%)': mape_cluster_hybrid}
    ])], ignore_index=True)

results = results.sort_values('R²', ascending=False)
print("\n", results.to_string(index=False))

# Save results
results.to_csv('data/results/hybrid_model_comparison.csv', index=False)
print("\n✅ Results saved to data/results/hybrid_model_comparison.csv")

# ===========================
# 9. VISUALIZATION
# ===========================
print("\n" + "=" * 70)
print("GENERATING COMPARISON PLOT")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 9.1 R² Comparison
ax1 = axes[0]
colors = ['#3498db', '#e74c3c', '#27ae60', '#9b59b6', '#f39c12']
bars = ax1.barh(results['Model'], results['R²'], color=colors[:len(results)])
ax1.set_xlabel('R² Score')
ax1.set_title('Model Comparison: R² Score (Higher is Better)')
ax1.set_xlim(0, 1.05)
for bar, val in zip(bars, results['R²']):
    ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center')

# 9.2 MAPE Comparison
ax2 = axes[1]
bars2 = ax2.barh(results['Model'], results['Median MAPE (%)'], color=colors[:len(results)])
ax2.set_xlabel('Median MAPE (%)')
ax2.set_title('Model Comparison: Median MAPE (Lower is Better)')
for bar, val in zip(bars2, results['Median MAPE (%)']):
    ax2.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', va='center')

plt.tight_layout()
plt.savefig('reports/figures/hybrid_model_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Plot saved to reports/figures/hybrid_model_comparison.png")
plt.show()

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
