from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

def evaluate_model(y_true, y_pred, model_name="Model"):
    """Calculates RMSE, MAE, R2 for a model."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"--- {model_name} Performance ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")
    
    return {"Model": model_name, "RMSE": rmse, "MAE": mae, "R2": r2}

def compare_models(results_list):
    """Creates a DataFrame to compare model performances."""
    return pd.DataFrame(results_list)
