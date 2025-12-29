import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

print("--- Updating Recursive vs One-Step Comparison Plot ---")

# 1. Load Data
df_lr_raw = load_data('data/processed/lr_final_prep.csv') 
df_common = load_data('data/processed/common_preprocessed.csv')
map_df = pd.read_csv('data/processed/recovered_index_map.csv')

# Align
df_lr = df_lr_raw.iloc[map_df['LR_Index']].copy().reset_index(drop=True)
original_indices = map_df['Original_Index'].values
df_lr['Year'] = df_common.loc[original_indices, 'Year'].values
df_lr['Entity'] = df_common.loc[original_indices, 'Entity'].values

TARGET = 'Value_co2_emissions_kt_by_country'
SPLIT_YEAR = 2015

# 2. Train Global Model (for One-Step)
feature_cols = [c for c in df_lr.columns if c not in [TARGET, 'Year', 'Entity'] and df_lr[c].dtype in [np.float64, np.int64]]
train_df = df_lr[df_lr['Year'] < SPLIT_YEAR]
test_df = df_lr[df_lr['Year'] >= SPLIT_YEAR].copy()

model = Ridge(alpha=1.0)
model.fit(train_df[feature_cols], train_df[TARGET])

# 3. Generate One-Step Predictions
# Uses ACTUAL Lags (Teacher Forcing) present in test_df
test_df['One_Step_Pred'] = model.predict(test_df[feature_cols])

# 4. Load Recursive Predictions
rec_df = pd.read_csv('data/results/recursive_predictions_corrected.csv')
# rec_df has 'Prediction' column. Need to merge on Entity/Year or assume sort.
# Ideally merge.
rec_df = rec_df[['Entity', 'Year', 'Prediction']].rename(columns={'Prediction': 'Recursive_Pred'})

# Merge
merged = pd.merge(test_df, rec_df, on=['Entity', 'Year'], how='left')
merged['Actual'] = merged[TARGET]

# 5. Plotting
# Aggregate Global Sums
global_trends = merged.groupby('Year')[['Actual', 'One_Step_Pred', 'Recursive_Pred']].sum().reset_index()

# Add Historical Data for Context (last 5 years of training)
history_trends = train_df[train_df['Year'] >= 2010].groupby('Year')[TARGET].sum().reset_index().rename(columns={TARGET: 'Actual'})
# Attach to global trends for continuous line?
# Easier to just plot separate lines

plt.figure(figsize=(10, 6))

# Plot History
sns.lineplot(data=history_trends, x='Year', y='Actual', color='black', alpha=0.5)

# Plot Test Period
sns.lineplot(data=global_trends, x='Year', y='Actual', label='Actual Data', color='black', linewidth=2.5)
sns.lineplot(data=global_trends, x='Year', y='One_Step_Pred', label='One-Step Ahead (Teacher Forcing)', color='blue', linestyle='-.', linewidth=2)
sns.lineplot(data=global_trends, x='Year', y='Recursive_Pred', label='Recursive Forecast (multi-step)', color='red', linestyle='--', linewidth=2)

# Styling
plt.title('Global CO2 Forecast: One-Step vs Recursive', fontsize=14)
plt.ylabel('Total CO2 Emissions (kt)', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline(2015, color='gray', linestyle=':', label='Forecast Start')

# Save
save_path = 'reports/figures/recursive_comparison_plot.png'
plt.savefig(save_path, dpi=300)
print(f"Plot saved to {save_path}")
