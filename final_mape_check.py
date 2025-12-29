import pandas as pd
import numpy as np
import sys
import os
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.linear_model import Ridge

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

def calculate_macro_mape(y_true, y_pred, entities):
    """
    Calculates Macro-Averaged MAPE.
    1. Compute MAPE for each entity (averaged over its time points).
    2. Compute Mean of Entity MAPEs.
    """
    df = pd.DataFrame({'Entity': entities, 'Actual': y_true, 'Predicted': y_pred})
    # Avoid division by zero
    df = df[df['Actual'] > 1e-6] 
    
    df['APE'] = np.abs((df['Actual'] - df['Predicted']) / df['Actual']) * 100
    
    # MAPE per Entity
    entity_mapes = df.groupby('Entity')['APE'].mean()
    
    # Macro Average
    macro_mape = entity_mapes.mean()
    median_mape = entity_mapes.median()
    return macro_mape, median_mape

print("--- Calculating Macro-MAPE for Internal Models (2015-2019) ---")

# Load Clean Data
df_lr_raw = load_data('data/processed/lr_final_prep.csv') 
df_common = load_data('data/processed/common_preprocessed.csv')
map_df = pd.read_csv('data/processed/recovered_index_map.csv')

df_lr = df_lr_raw.iloc[map_df['LR_Index']].copy().reset_index(drop=True)
original_indices = map_df['Original_Index'].values
df_lr['Year'] = df_common.loc[original_indices, 'Year'].values
df_lr['Entity'] = df_common.loc[original_indices, 'Entity'].values

SPLIT_YEAR = 2015
train_mask = df_lr['Year'] < SPLIT_YEAR
test_mask = (df_lr['Year'] >= SPLIT_YEAR) & (df_lr['Year'] <= 2019)

TARGET = 'Value_co2_emissions_kt_by_country'
feature_cols = [c for c in df_lr.columns if c not in [TARGET, 'Year', 'Entity'] and df_lr[c].dtype in [np.float64, np.int64]]

X_train = df_lr.loc[train_mask, feature_cols]
y_train = df_lr.loc[train_mask, TARGET]
X_test = df_lr.loc[test_mask, feature_cols]
y_test = df_lr.loc[test_mask, TARGET]
entities_test = df_lr.loc[test_mask, 'Entity']

# 1. Global Ridge
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
preds = model.predict(X_test)

macro_mape, median_mape = calculate_macro_mape(y_test, preds, entities_test)
print(f"Global Linear Regression:")
print(f"  Macro-MAPE: {macro_mape:.2f}%")
print(f"  Median-MAPE: {median_mape:.2f}%")
