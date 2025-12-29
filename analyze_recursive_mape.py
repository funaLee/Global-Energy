import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import sys
import os

print("--- Analyzing Recursive Forecast Year-by-Year (MAPE Added) ---")
try:
    df = pd.read_csv('data/results/recursive_predictions_corrected.csv')
except FileNotFoundError:
    print("Error: Run the notebook to generate predictions first.")
    sys.exit(1)

# Metric functions
def get_metrics(g):
    y_true = g['Value_co2_emissions_kt_by_country']
    y_pred = g['Prediction']
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Robust MAPE
    # Filter tiny values to avoid infinity
    valid = y_true > 1e-6
    if valid.sum() == 0:
        mape_med = np.nan
        mape_mean = np.nan
    else:
        ape = np.abs((y_true[valid] - y_pred[valid]) / y_true[valid]) * 100
        mape_med = np.median(ape)
        mape_mean = np.mean(ape)
        
    return pd.Series({'R2': r2, 'RMSE': rmse, 'Median_MAPE': mape_med, 'Mean_MAPE': mape_mean, 'N': len(g)})

stats = df.groupby('Year').apply(get_metrics)
print(stats)
