import json
import os

notebook_path = 'notebooks/10_Recursive_Forecasting.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Improved Code Block
new_source = [
    "# --- Improved Scaling Logic ---\n",
    "# The previous attempt to reconstruct the scaler from global mean/std failed because \n",
    "# the model training data (lr_final_prep.csv) likely excluded outliers or used a different subset.\n",
    "# We empirically found the mapping between Raw Target (shifted) and Scaled Lag Feature in the data.\n",
    "# Mapping: Scaled = (Raw * Slope) + Intercept\n",
    "# Found via debug_mapping.py: Slope=2.829519e-05, Intercept=-0.5886\n",
    "\n",
    "EMPIRICAL_SLOPE = 2.829519e-05\n",
    "EMPIRICAL_INTERCEPT = -0.5886\n",
    "\n",
    "def scale_value(val):\n",
    "    # Apply the empirical linear transformation\n",
    "    return (val * EMPIRICAL_SLOPE) + EMPIRICAL_INTERCEPT\n",
    "\n",
    "print(f\"Using Empirical Scaler: Slope={EMPIRICAL_SLOPE}, Intercept={EMPIRICAL_INTERCEPT}\")\n",
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
    "for year in sorted(test_df['Year'].unique()):\n",
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
    "        if next_year <= 2020: # Limit to test range\n",
    "            # Find the row for this entity in next year\n",
    "            mask = (test_df_recursive['Year'] == next_year) & (test_df_recursive['Entity'] == entity)\n",
    "            if mask.any():\n",
    "                # Scale the prediction before inserting as feature\n",
    "                scaled_pred = scale_value(pred)\n",
    "                test_df_recursive.loc[mask, target_lag_col] = scaled_pred"
]

# Find the cell to replace
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_text = "".join(cell['source'])
        if "def scale_value(val):" in source_text and "recursive_preds = []" in source_text:
            cell['source'] = new_source
            cell['outputs'] = [] # Clear outputs
            found = True
            break

if found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully (Scaler Fixed).")
else:
    print("Target cell not found. Please check identifiers.")
