import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

print("--- Analyzing Recursive Forecast Year-by-Year ---")
df = pd.read_csv('data/results/recursive_predictions_corrected.csv')
TARGET = 'Value_co2_emissions_kt_by_country' # Used as 'True' column potentially? 
# In the patch code: y_true = test_dynamic[TARGET], y_pred = test_dynamic['Prediction']
# So the CSV has columns: [TARGET (Actual), 'Prediction', 'Year', 'Entity', ...]

print(f"Columns: {df.columns.tolist()}")

years = sorted(df['Year'].unique())
print(f"{'Year':<6} | {'R2 Score':<10} | {'RMSE':<15} | {'N Samples':<10}")
print("-" * 55)

for year in years:
    mask = df['Year'] == year
    y_true = df.loc[mask, TARGET]
    y_pred = df.loc[mask, 'Prediction']
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    n = len(y_true)
    
    print(f"{year:<6} | {r2:<10.4f} | {rmse:<15,.0f} | {n:<10}")

print("-" * 55)
print("Analysis complete.")
