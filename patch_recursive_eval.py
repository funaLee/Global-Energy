import nbformat
import sys

nb_path = 'notebooks/10_Recursive_Forecasting.ipynb'
print(f"Patching Evaluation Cell in {nb_path}...")

with open(nb_path, 'r') as f:
    nb = nbformat.read(f, as_version=4)

# New Evaluation Code compatible with test_dynamic
new_eval_code = """# 4. Evaluation & Comparison
# Updated to work with test_dynamic dataframe from previous cell

# We need Teacher Forcing Predictions for comparison
print("Generating Teacher Forcing Predictions...")
# Re-predict using original test_data (where lags are TRUE actuals)
test_data_tf = df_lr[df_lr['Year'] >= SPLIT_YEAR].copy().sort_values(['Year', 'Entity'])
X_tf = test_data_tf[feature_cols].fillna(0)

# Predict Cluster-wise for Teacher Forcing to be fair
tf_preds = np.zeros(len(X_tf))
batch_clusters = test_data_tf['Cluster'].values

for c, model in models.items():
    local_mask = (batch_clusters == c)
    if local_mask.any():
        X_c = X_tf.iloc[local_mask]
        tf_preds[local_mask] = model.predict(X_c)

test_data_tf['Teacher_Forcing_Pred'] = tf_preds

# Merge Recursive Preds
# test_dynamic has 'Prediction' column which is Recursive
results_df = test_dynamic[['Entity', 'Year', TARGET, 'Prediction']].rename(columns={'Prediction': 'Recursive_Pred', TARGET: 'Actual'})
results_df['Teacher_Forcing_Pred'] = test_data_tf['Teacher_Forcing_Pred'].values

# Calculate Global Metrics
r2_rec = r2_score(results_df['Actual'], results_df['Recursive_Pred'])
r2_tf = r2_score(results_df['Actual'], results_df['Teacher_Forcing_Pred'])

rmse_rec = np.sqrt(mean_squared_error(results_df['Actual'], results_df['Recursive_Pred']))
rmse_tf = np.sqrt(mean_squared_error(results_df['Actual'], results_df['Teacher_Forcing_Pred']))

print("-" * 40)
print(f"Teacher Forcing (One-Step) R2: {r2_tf:.4f}")
print(f"Recursive Forecasting   R2: {r2_rec:.4f}")
print(f"Drop in Performance: {(r2_tf - r2_rec) / r2_tf * 100:.2f}%")
print("-" * 40)

# Create Comparison Table
comparison = pd.DataFrame({
    'Method': ['One-Step (Teacher Forcing)', 'Recursive (Multi-Step)'],
    'R2 Score': [r2_tf, r2_rec],
    'RMSE': [rmse_tf, rmse_rec]
})

# Save Results
comparison.to_csv('../data/results/recursive_comparison.csv', index=False)
print(comparison)
"""

# Find the evaluation cell
found = False
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and "Calculate Global Metrics" in cell.source:
        nb.cells[i].source = new_eval_code
        found = True
        break
    if cell.cell_type == 'code' and "results_df = pd.DataFrame" in cell.source:
         nb.cells[i].source = new_eval_code
         found = True
         break

if found:
    with open(nb_path, 'w') as f:
        nbformat.write(nb, f)
    print("Evaluation logic patched.")
else:
    print("Could not find evaluation cell.")
