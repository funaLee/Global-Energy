import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OrdinalEncoder, OneHotEncoder

def load_data(path):
    """Loads dataset from CSV."""
    try:
        df = pd.read_csv(path)
        print(f"Loaded data from {path}: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def basic_cleaning(df):
    """Performs basic cleaning like format adjustment (removing commas)."""
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object' and col != 'Entity':
            try:
                df_clean[col] = df_clean[col].str.replace(',', '').astype(float)
            except:
                pass
    return df_clean

def handle_missing_values(df, strategy='median'):
    """Imputes missing values."""
    df_imputed = df.copy()
    numeric_columns = df_imputed.select_dtypes(include=[np.number]).columns
    
    if strategy == 'median':
        for col in numeric_columns:
            if df_imputed[col].isnull().any():
                median_val = df_imputed[col].median()
                df_imputed[col].fillna(median_val, inplace=True)
    
    # Check for remaining NaNs
    remaining = df_imputed[numeric_columns].isnull().sum().sum()
    print(f"Missing values after imputation: {remaining}")
    return df_imputed

def create_lag_features(df, target_col, lag_cols, shifts=[1]):
    """Generates lag features for time-series/panel analysis."""
    df_lagged = df.copy()
    df_lagged.sort_values(['Entity', 'Year'], inplace=True)
    
    for col in lag_cols:
        for shift in shifts:
            df_lagged[f'{col}_lag{shift}'] = df_lagged.groupby('Entity')[col].shift(shift)
            
    # Drop rows with NaNs caused by shifting
    original_len = len(df_lagged)
    df_lagged.dropna(inplace=True)
    print(f"Dropped {original_len - len(df_lagged)} rows due to lags.")
    return df_lagged

def encode_features(df, method='onehot'):
    """Encodes categorical features (Entity)."""
    if method == 'onehot':
        return pd.get_dummies(df, columns=['Entity'], drop_first=True)
    elif method == 'ordinal':
        df_encoded = df.copy()
        encoder = OrdinalEncoder()
        df_encoded['Entity'] = encoder.fit_transform(df_encoded[['Entity']])
        return df_encoded
    else:
        return df

def remove_outliers(df, method='iqr', threshold=1.5, exclude_cols=None):
    """Removes outliers from numerical columns using IQR, excluding specified columns."""
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    # Exclude columns (e.g., One-Hot encoded)
    if exclude_cols:
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols and not c.startswith('Entity_')]
    else:
        # Auto-exclude likely One-Hot columns
        numeric_cols = [c for c in numeric_cols if not c.startswith('Entity_')]

    original_rows = len(df_clean)
    
    if method == 'iqr':
        Q1 = df_clean[numeric_cols].quantile(0.25)
        Q3 = df_clean[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        
        # Identify valid columns (IQR > 0)
        valid_cols = IQR[IQR > 0].index
        if len(valid_cols) < len(numeric_cols):
            print(f"Skipping outlier removal for {len(numeric_cols) - len(valid_cols)} columns with IQR=0 (likely imputed): {list(set(numeric_cols) - set(valid_cols))}")
        
        # Filter: Keep rows that are NOT outliers in ANY of the valid numeric columns
        if len(valid_cols) > 0:
            condition = ~((df_clean[valid_cols] < (Q1[valid_cols] - threshold * IQR[valid_cols])) | (df_clean[valid_cols] > (Q3[valid_cols] + threshold * IQR[valid_cols]))).any(axis=1)
            df_clean = df_clean[condition]
        
    print(f"Removed {original_rows - len(df_clean)} outlier rows (threshold={threshold}).")
    return df_clean

def remove_high_vif(df, target_col, threshold=10, exclude_cols=None):
    """Iteratively removes features with VIF > threshold, excluding specified columns."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    df_vif = df.copy()
    # Select numeric features only
    features = list(df_vif.select_dtypes(include=[np.number]).columns)
    
    # Exclude target and One-Hot columns from VIF calculation
    if target_col in features:
        features.remove(target_col)
        
    # Exclude specified columns or One-Hot columns
    if exclude_cols:
        features = [f for f in features if f not in exclude_cols and not f.startswith('Entity_')]
    else:
        features = [f for f in features if not f.startswith('Entity_')]

    dropped_features = []
    
    while True:
        # VIF requires no NaNs and no infinite values
        X = df_vif[features].dropna()
        
        # Determine VIF
        try:
            vif_data = pd.DataFrame()
            vif_data["feature"] = features
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(features))]
            
            max_vif = vif_data["VIF"].max()
            if max_vif > threshold:
                feature_to_drop = vif_data.sort_values("VIF", ascending=False)["feature"].iloc[0]
                features.remove(feature_to_drop)
                dropped_features.append(feature_to_drop)
            else:
                break
        except Exception as e:
            print(f"Error calculating VIF: {e}")
            break
            
    print(f"Dropped features due to VIF > {threshold}: {dropped_features}")
    # Return df with dropped features removed
    # Keep target, kept features, and any other columns (like One-Hot or Entity)
    keep_cols = features + [target_col] + [c for c in df.columns if c not in features and c != target_col and c not in dropped_features]
    return df_vif[keep_cols]
