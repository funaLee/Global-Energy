import json
import os

notebook_path = 'notebooks/10_Recursive_Forecasting.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Alignment Cell Replacement
alignment_source = [
    "# Load Data\n",
    "df_common = load_data('../data/processed/common_preprocessed.csv')\n",
    "df_lr = load_data('../data/processed/lr_final_prep.csv')\n",
    "\n",
    "# Load Alignment Map (Recovered via recover_sequence.py)\n",
    "try:\n",
    "    recovery_map = pd.read_csv('../data/processed/recovered_index_map.csv')\n",
    "    print(\"Loaded Alignment Map.\")\n",
    "except FileNotFoundError:\n",
    "    print(\"Error: Map not found. Please run recover_sequence.py\")\n",
    "\n",
    "# Ensure df_lr has RangeIndex\n",
    "df_lr = df_lr.reset_index(drop=True)\n",
    "\n",
    "if len(df_lr) == len(recovery_map):\n",
    "    # Extract indices\n",
    "    original_indices = recovery_map['Original_Index'].values\n",
    "    \n",
    "    # Assign Metadata from Common\n",
    "    df_lr['Year'] = df_common.loc[original_indices, 'Year'].values\n",
    "    df_lr['Entity'] = df_common.loc[original_indices, 'Entity'].values\n",
    "    print(\"Metadata (Entity/Year) assigned via Map.\")\n",
    "    \n",
    "    # Also align df_common support (for scaler reconstruction)\n",
    "    # We need the 'unscaled' values corresponding to these rows\n",
    "    unscaled_subset = df_common.loc[original_indices].reset_index(drop=True)\n",
    "    \n",
    "else:\n",
    "    print(\"Length mismatch! Alignment failed.\")\n",
    "\n",
    "# Restore Variable Definitions (Deleted by patch)\n",
    "lag_cols = [c for c in df_lr.columns if 'lag' in c]\n",
    "print(\"Lag Features:\", lag_cols)\n",
    "TARGET = 'Value_co2_emissions_kt_by_country'\n",
    "target_lag_col = next((c for c in lag_cols if TARGET in c), None)\n",
    "print(\"Target Lag Column:\", target_lag_col)\n",
    "SPLIT_YEAR = 2015\n"
]


# Find the Alignment Cell (Look for old code OR bad-patch code)
found_alignment = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        if "common_idx = df_common.index.intersection" in src or "recovery_map = pd.read_csv" in src:
            cell['source'] = alignment_source
            cell['outputs'] = []
            found_alignment = True
            break
            
if found_alignment:
    print("Alignment Logic Patched.")
else:
    print("Warning: Alignment Cell NOT FOUND.")


# 2. Scaler Reconstruction Replacement (Revert hack, use clean logic)
scaler_source = [
    "# Reconstruct Scaler from Aligned Data\n",
    "# Now that rows align, we can calculate true Mean/Std\n",
    "\n",
    "unscaled_vals = unscaled_subset[target_lag_col]\n",
    "scaled_vals = df_lr[target_lag_col]\n",
    "\n",
    "lag_mean = unscaled_vals.mean()\n",
    "lag_std = unscaled_vals.std()\n",
    "print(f\"Reconstructed Scaler: Mean={lag_mean:.4f}, Std={lag_std:.4f}\")\n",
    "\n",
    "# Verify Correlation\n",
    "corr = np.corrcoef(unscaled_vals, scaled_vals)[0,1]\n",
    "print(f\"Correlation (Unscaled vs Scaled): {corr:.4f}\")\n",
    "\n",
    "def scale_value(val):\n",
    "    return (val - lag_mean) / lag_std\n",
    "\n",
    "# Define Test Set\n",
    "test_df = df_lr[df_lr['Year'] >= SPLIT_YEAR]\n",
    "test_years = sorted(test_df['Year'].unique())\n",
    "\n",
    "# --- Recursive Loop ---\n",
    "recursive_preds = []\n",
    "teacher_forcing_preds = []\n",
    "actuals = []\n",
    "years = []\n",
    "entities_list = []\n",
    "\n",
    "# We must iterate year by year. \n",
    "# Modifications to df_lr (test set) must persist to the next iteration.\n",
    "test_df_recursive = test_df.copy()\n",
    "\n",
    "print(\"Starting Recursive Forecasting...\")\n",
    "\n",
    "for year in test_years:\n",
    "    print(f\"Processing Year: {year}\")\n",
    "    # Get data for this year\n",
    "    current_year_df = test_df_recursive[test_df_recursive['Year'] == year]\n",
    "    \n",
    "    if current_year_df.empty: continue\n",
    "        \n",
    "    for idx, row in current_year_df.iterrows():\n",
    "        cluster = int(row['Cluster'])\n",
    "        entity = row['Entity']\n",
    "        \n",
    "        # Predict using current features (which contain previous year's updated lags)\n",
    "        features = row[feature_cols].values.reshape(1, -1)\n",
    "        pred = models[cluster].predict(features)[0]\n",
    "        \n",
    "        # Store Results\n",
    "        recursive_preds.append(pred)\n",
    "        actuals.append(row[TARGET])\n",
    "        years.append(year)\n",
    "        entities_list.append(entity)\n",
    "        \n",
    "        # Teacher Forcing Prediction (for comparison)\n",
    "        # We use the original test_df which has the 'true' lags\n",
    "        tf_row = test_df[(test_df['Year'] == year) & (test_df['Entity'] == entity)]\n",
    "        if not tf_row.empty:\n",
    "            tf_features = tf_row[feature_cols].values.reshape(1, -1)\n",
    "            tf_pred = models[cluster].predict(tf_features)[0]\n",
    "            teacher_forcing_preds.append(tf_pred)\n",
    "        \n",
    "        # UPDATE NEXT YEAR'S LAG \n",
    "        # The predicted 'pred' is for Year T. \n",
    "        # This becomes the Lag for Year T+1.\n",
    "        next_year = year + 1\n",
    "        if next_year <= max(test_years):\n",
    "            # Find the row for this entity in next year\n",
    "            mask = (test_df_recursive['Year'] == next_year) & (test_df_recursive['Entity'] == entity)\n",
    "            if mask.any():\n",
    "                # Scale the prediction before inserting as feature\n",
    "                scaled_pred = scale_value(pred)\n",
    "                test_df_recursive.loc[mask, target_lag_col] = scaled_pred\n"
]

# Find the Scaler Cell (Look for "scale_value" or "Recursive Loop")
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        if "Recursive Loop" in src and "scale_value" in src:
            cell['source'] = scaler_source
            cell['outputs'] = []
            break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook patched with Correct Alignment and Clean Scaler.")
