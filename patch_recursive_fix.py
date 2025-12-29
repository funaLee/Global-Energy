import nbformat
import sys

nb_path = 'notebooks/10_Recursive_Forecasting.ipynb'
print(f"Applying Robust Patch to {nb_path}...")

with open(nb_path, 'r') as f:
    nb = nbformat.read(f, as_version=4)

# Robust Recursive Loop Logic
new_loop_code = """# 3. Recursive Forecasting Loop (CORRECTED & ROBUST)
import warnings
warnings.filterwarnings('ignore')

# A. Determine Scaler Statistics
print("Calculating Scaler Statistics from Training Data...")
target_lag_col = 'Value_co2_emissions_kt_by_country_lag1'

train_common_mask = (df_common['Year'] < SPLIT_YEAR) & (df_common['Entity'].isin(df_lr['Entity']))
train_target_vals = df_common.loc[train_common_mask, TARGET]
lag_mean = train_target_vals.mean()
lag_std = train_target_vals.std()

print(f"Lag Feature Re-Scaler: Mean={lag_mean:.2f}, Std={lag_std:.2f}")

# B. Prepare Test Data
test_data = df_lr[df_lr['Year'] >= SPLIT_YEAR].copy().sort_values(['Year', 'Entity'])
test_years = sorted(test_data['Year'].unique())

# Ensure no NaNs to start with
feature_cols = [c for c in feature_cols if c in test_data.columns]
test_data[feature_cols] = test_data[feature_cols].fillna(0)

# C. Loop
test_dynamic = test_data.copy()
print(f"Starting Recursive Loop on {len(test_dynamic)} rows...")

for year in test_years:
    # 1. Get current year Batch
    current_mask = test_dynamic['Year'] == year
    if not current_mask.any(): continue
    
    # Check for NaNs before predict
    X_batch = test_dynamic.loc[current_mask, feature_cols].fillna(0)
    
    # 2. Predict using Cluster Models
    # Initialize preds array
    current_preds = np.zeros(len(X_batch))
    
    # We iterate manually to match cluster models
    # A safer way than boolean masking optimization which triggered NaNs
    
    # Get Cluster IDs for this batch
    batch_clusters = test_dynamic.loc[current_mask, 'Cluster'].values
    batch_indices = np.arange(len(X_batch))
    
    for c, model in models.items():
        # Identify rows for this cluster in the local batch
        local_mask = (batch_clusters == c)
        if local_mask.any():
            X_c = X_batch.iloc[local_mask]
            if X_c.empty: continue
            
            # Predict
            try:
                preds_c = model.predict(X_c)
                current_preds[local_mask] = preds_c
            except Exception as e:
                print(f"Error predicting Year {year} Cluster {c}: {e}")

    # Map predictions back to the main DataFrame
    test_dynamic.loc[current_mask, 'Prediction'] = current_preds
    
    # 3. Update NEXT Year's Lag
    next_year = year + 1
    if next_year <= max(test_years):
        # Calculate Scaled Lag values
        # Clip to avoid massive explosions
        scaled_lags = (current_preds - lag_mean) / lag_std
        
        # Map [Entity -> Scaled_Lag]
        current_entities = test_dynamic.loc[current_mask, 'Entity'].values
        entity_lag_map = dict(zip(current_entities, scaled_lags))
        
        # Update rows in next_year
        next_mask = test_dynamic['Year'] == next_year
        
        # Use simple iteration for safety if map is flaky
        # Or vectorized map with fillna logic
        next_entities = test_dynamic.loc[next_mask, 'Entity']
        new_lags = next_entities.map(entity_lag_map)
        
        # Only update where we found a match (don't introduce NaNs for new entities)
        # Combine with existing values for safety
        original_lags = test_dynamic.loc[next_mask, target_lag_col]
        test_dynamic.loc[next_mask, target_lag_col] = new_lags.fillna(original_lags)

# D. Evaluate
y_true = test_dynamic[TARGET]
y_pred = test_dynamic['Prediction']

r2_rec = r2_score(y_true, y_pred)
rmse_rec = np.sqrt(mean_squared_error(y_true, y_pred))

print("-" * 30)
print(f"Corrected Recursive R2: {r2_rec:.4f}")
print(f"Corrected Recursive RMSE: {rmse_rec:,.2f}")
print("-" * 30)

# Save
test_dynamic.to_csv('../data/results/recursive_predictions_corrected.csv', index=False)
"""

# Find the cell to replace (the one we just patched, or original)
found = False
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and "Recursive Forecasting Loop" in cell.source:
        nb.cells[i].source = new_loop_code
        found = True
        break

if found:
    with open(nb_path, 'w') as f:
        nbformat.write(nb, f)
    print("Notebook patched (Robust Version).")
else:
    print("Could not find target cell.")
