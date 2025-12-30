
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path if needed, but we might just load data directly
sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

print('=' * 80)
print('PHASE 0: RANDOM SPLIT vs TIME-SERIES SPLIT (WITH MAPE)')
print('=' * 80)

# Load Common Data for Entity/Year alignment
df_common = load_data('data/processed/common_preprocessed.csv')
TARGET = 'Value_co2_emissions_kt_by_country'

def calculate_metrics(y_true, y_pred, entities):
    r2 = r2_score(y_true, y_pred)
    
    # Median MAPE Calculation
    df_temp = pd.DataFrame({
        'Entity': entities.values if hasattr(entities, 'values') else entities,
        'Actual': y_true.values if hasattr(y_true, 'values') else y_true,
        'Pred': y_pred
    })
    
    # Avoid division by zero
    epsilon = 1e-10
    df_temp['APE'] = np.abs(df_temp['Actual'] - df_temp['Pred']) / np.abs(df_temp['Actual'].replace(0, epsilon)) * 100
    
    # Mean MAPE per Entity, then Median of those Means
    entity_mape = df_temp.groupby('Entity')['APE'].mean()
    median_mape = entity_mape.median()
    
    return r2, median_mape

def run_experiment(name, df, model, feature_cols, target_col='Value_co2_emissions_kt_by_country'):
    print(f'\nRunning {name}...')
    
    # Ensure Year and Entity are available and correct (overwrite if alignment likely)
    if len(df) == len(df_common):
        # Force overwrite to ensure unscaled Year and correct Entity for splitting/grouping
        df = df.copy()
        df['Year'] = df_common['Year'].values
        df['Entity'] = df_common['Entity'].values
    elif 'Year' not in df.columns or 'Entity' not in df.columns:
         # Try to map using index map if available (for LR)
            try:
                map_df = pd.read_csv('data/processed/recovered_index_map.csv')
                # Check if this df matches common or filtered
                if len(df) < len(df_common):
                     # Likely LR filtered data
                    original_indices = map_df['Original_Index'].values
                    df = df.copy()
                    # map_df might map 1-to-1 with df rows if df was saved after mapping?
                    # Let's check logic: map_df['LR_Index'] are indices in df_common? No. 
                    # Usually filtering creates a subset. 
                    # Let's assume for now we can rely on passed df having Year/Entity or being aligned.
                    pass
            except:
                pass

    # Random Split
    X = df[feature_cols]
    y = df[target_col]
    entities = df['Entity']
    
    X_train_rnd, X_test_rnd, y_train_rnd, y_test_rnd, ent_train_rnd, ent_test_rnd = train_test_split(
        X, y, entities, test_size=0.2, random_state=42, shuffle=True
    )
    
    model.fit(X_train_rnd, y_train_rnd)
    preds_rnd = model.predict(X_test_rnd)
    r2_rnd, mape_rnd = calculate_metrics(y_test_rnd, preds_rnd, ent_test_rnd)
    
    # Time-Series Split (< 2015 vs >= 2015)
    train_mask = df['Year'] < 2015
    test_mask = df['Year'] >= 2015
    
    X_train_ts = df.loc[train_mask, feature_cols]
    y_train_ts = df.loc[train_mask, target_col]
    
    X_test_ts = df.loc[test_mask, feature_cols]
    y_test_ts = df.loc[test_mask, target_col]
    ent_test_ts = df.loc[test_mask, 'Entity']
    
    model.fit(X_train_ts, y_train_ts)
    preds_ts = model.predict(X_test_ts)
    r2_ts, mape_ts = calculate_metrics(y_test_ts, preds_ts, ent_test_ts)
    
    return {
        'Model': name,
        'Random R2': r2_rnd,
        'Random MAPE': mape_rnd,
        'Time-Series R2': r2_ts,
        'Time-Series MAPE': mape_ts
    }

results = []

# --- 1. Linear Regression ---
print('Loading LR data...')
df_lr_raw = load_data('data/processed/lr_final_prep.csv')
map_df = pd.read_csv('data/processed/recovered_index_map.csv')
# Reconstruct LR dataframe with Year/Entity
df_lr = df_lr_raw.iloc[map_df['LR_Index']].copy().reset_index(drop=True)
original_indices = map_df['Original_Index'].values
df_lr['Year'] = df_common.loc[original_indices, 'Year'].values
df_lr['Entity'] = df_common.loc[original_indices, 'Entity'].values

feature_cols_lr = [c for c in df_lr_raw.columns if c not in [TARGET]]
# Note: df_lr_raw features might be scaled. 'Year', 'Entity' added are for splitting.

lr_res = run_experiment('Linear Regression', df_lr, Ridge(alpha=10.0), feature_cols_lr)
results.append(lr_res)


# --- 2. XGBoost ---
print('Loading XGBoost data...')
df_xgb = load_data('data/processed/xgb_final_prep.csv')
# Align with common if needed, but header says it has Year/Entity?
# Let's inspect columns from previous tool output: Entity, Year are present.
# Entity might be encoded? "Value_co2..." is target.
if df_xgb['Entity'].dtype in [np.int64, np.float64]:
    # It is ordinal encoded or something. We need original Entity names for MAPE grouping?
    # Actually, grouping by Encoded Entity is fine for Median MAPE calculation.
    pass

feature_cols_xgb = [c for c in df_xgb.columns if c not in [TARGET, 'Entity', 'Year', 'Entity_Encoded']]

xgb_res = run_experiment('XGBoost', df_xgb, 
                         XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, n_jobs=-1, random_state=42), 
                         feature_cols_xgb)
results.append(xgb_res)


# --- 3. SVR ---
print('Loading SVR data...')
df_svr = load_data('data/processed/svr_final_prep.csv')
# SVR prep usually has Year.
# Check alignment.
if len(df_svr) == len(df_common):
    df_svr['Entity'] = df_common['Entity'].values
    # Ensure Year is correct (it has Year column)

feature_cols_svr = [c for c in df_svr.columns if c not in [TARGET, 'Entity', 'Year']]

# SVR can be slow, use subset or limit it? Or parameters?
# Using default SVR or maybe LinearSVR for speed if dataset is large? 
# Dataset ~3000 rows is fine for SVR.
svr_res = run_experiment('SVR', df_svr, SVR(kernel='rbf'), feature_cols_svr)
results.append(svr_res)


# --- Output ---
results_df = pd.DataFrame(results)
print('\n' + '='*80)
print('FINAL RESULTS WITH MAPE')
print('='*80)
print(results_df.to_string(index=False))

# Calculate Drops
results_df['R2 Drop'] = (results_df['Random R2'] - results_df['Time-Series R2']) / results_df['Random R2'] * 100
results_df['MAPE Increase'] = (results_df['Time-Series MAPE'] - results_df['Random MAPE']) / results_df['Random MAPE'] * 100

print('\nDetailed Changes:')
for _, row in results_df.iterrows():
    print(f"\n{row['Model']}:")
    print(f"  R2: {row['Random R2']:.4f} -> {row['Time-Series R2']:.4f} (Drop: {row['R2 Drop']:.1f}%)")
    print(f"  MAPE: {row['Random MAPE']:.2f}% -> {row['Time-Series MAPE']:.2f}% (Increase: {row['MAPE Increase']:.1f}%)")

