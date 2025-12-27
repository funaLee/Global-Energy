
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, RobustScaler, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- Custom Transformers ---

class VIFSelector(BaseEstimator, TransformerMixin):
    """
    Removes features with Variance Inflation Factor (VIF) > threshold.
    ADDRESSES: Multicollinearity for Linear Regression (Statistical Rigor).
    """
    def __init__(self, threshold=10.0):
        self.threshold = threshold
        self.feature_names_ = None

    def fit(self, X, y=None):
        # Assumes X is a DataFrame for easier VIF calculation with column names
        # If X is array, we'd need to handle it. Pipelines usually pass arrays unless configured.
        # We will cast to DF if needed, assuming numeric input from previous steps.
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Iteratively remove features with high VIF
        X_temp = X.copy()
        while True:
            # Drop constant columns (VIF is inf) or NaN columns
            X_temp = X_temp.dropna(axis=1) # Simple handle, strictly imputation should happen before
            
            # VIF needs numeric data
            numeric_cols = X_temp.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                break
                
            vifs = pd.Series(
                [variance_inflation_factor(X_temp[numeric_cols].values, i) for i in range(len(numeric_cols))],
                index=numeric_cols
            )
            
            max_vif = vifs.max()
            if max_vif > self.threshold:
                drop_col = vifs.idxmax()
                X_temp = X_temp.drop(columns=[drop_col])
                # print(f"DEBUG: Dropped {drop_col} with VIF {max_vif:.2f}")
            else:
                break
        
        self.feature_names_ = X_temp.columns.tolist()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.feature_names_] 
    
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """
    Keeps features with Pearson correlation > threshold with the target.
    ADDRESSES: Feature Selection for SVR to reduce noise (Distance-Based Optimization).
    """
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.selected_indices_ = []

    def fit(self, X, y):
        if y is None:
            raise ValueError("Target y is required for CorrelationSelector")
        
        # Calculate correlations
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X)
        else:
            X_df = X
            
        correlations = X_df.corrwith(pd.Series(y, index=X_df.index))
        high_corr_features = correlations[abs(correlations) > self.threshold].index
        
        # Get integer indices for transforming if likely to receive array later, 
        # but sticking to column names is safer if pipeline preserves them.
        self.feature_names_ = high_corr_features.tolist()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.feature_names_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


class LagFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Creates lag features (t-1) for specified columns grouped by Entity.
    ADDRESSES: Temporal dependency for XGBoost (Panel Data Approach).
    """
    def __init__(self, group_col='Entity', time_col='Year', lag_cols=['Value_co2_emissions_kt_by_country', 'gdp_growth']):
        self.group_col = group_col
        self.time_col = time_col
        self.lag_cols = lag_cols

    def fit(self, X, y=None):
        # Lag generation is stateless in terms of fitting parameters, 
        # but requires historical context which implies it acts on the dataset structure.
        return self

    def transform(self, X):
        # X must contain Entity and Year to perform lagging correctly.
        # We assume X is the full DataFrame passed at the start of the pipeline.
        X_out = X.copy()
        
        # Sort to ensure lags are correct
        X_out = X_out.sort_values(by=[self.group_col, self.time_col])
        
        for col in self.lag_cols:
            if col in X_out.columns:
                X_out[f'{col}_lag1'] = X_out.groupby(self.group_col)[col].shift(1)
            else:
                print(f"Warning: Column {col} not found for lagging.")
        
        # Lags introduce NaNs for the first year of each group.
        # Strategy: Impute or drop. For XGBoost, it handles NaNs, but filling with 0 or specialized imputation is safer.
        # Here we leave as NaNs to be handled by the subsequent Imputer step.
        return X_out


# --- Pipeline Construction Functions ---

def create_linear_regression_pipeline(numerical_cols, categorical_cols=['Entity']):
    """
    Pipeline 1: Linear Regression (Statistical Rigor)
    - Log transforms skewed features.
    - One-Hot Encodes Country.
    - VIF Selection to remove multicollinear features.
    """
    
    # 1. Feature Engineering: Numerical Branch
    # Log transform -> Impute -> VIF Select
    numerical_transformer = Pipeline([
        ('log_trans', FunctionTransformer(np.log1p, validate=False)),
        ('imputer', SimpleImputer(strategy='median')),
        ('vif_selection', VIFSelector(threshold=10.0))
    ])

    categorical_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_trans', numerical_transformer, numerical_cols), 
            ('cat_trans', categorical_transformer, categorical_cols) # Fixed Effects
        ],
        verbose_feature_names_out=False
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor)
        # Imputer and VIF are now handled within the numerical branch
    ])
    
    return pipeline


def create_svr_pipeline(numerical_cols):
    """
    Pipeline 2: SVR (Distance-Based Optimization)
    - Robust Scaling to handle outliers.
    - Correlation Filter to select relevant features.
    """
    # Note: SVR pipeline doesn't use OneHotEncoded Entity usually due to dimensionality explosion,
    # relying instead on feature scaling and general indicators.
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()), # Handles variance between small/large nations
        ('selector', CorrelationSelector(threshold=0.1)) # Filtering noise
    ])
    
    return pipeline


def create_xgboost_pipeline(numerical_cols, categorical_cols=['Entity']):
    """
    Pipeline 3: XGBoost (Feature Engineering Focus)
    - Lag Features.
    - Ordinal Encoding for high-cardinality categorical variables.
    - Median Imputation.
    """
    
    # Note: LagFeatureGenerator works on the full DataFrame structure before splitting into num/cat strictly.
    # So it usually goes first if we are passing a cohesive DF. 
    # However, sklearn pipelines usually get X as array after the first step if composed.
    # To use LagFeatureGenerator inside, we assume the input to this pipeline is the Dataframe.
    
    # We will assume 'numerical_cols' here *includes* the newly created lag columns 
    # or we simply select all numeric types after lagging.
    
    feature_generation = LagFeatureGenerator(lag_cols=['Value_co2_emissions_kt_by_country', 'gdp_growth'])
    
    preprocessor = ColumnTransformer(
        transformers=[
            # Tree-based models handle Ordinal (Integer) encoding well, easier splittable than OHE
            ('cat_trans', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols),
            ('num_trans', SimpleImputer(strategy='median'), numerical_cols) # Impute numeric (including lags)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    
    pipeline = Pipeline([
        ('feature_generation', feature_generation), # Create 'memory' of past emissions
        ('preprocessor', preprocessor)
    ])
    
    return pipeline

# --- Demo Execution (Verification) ---
if __name__ == "__main__":
    # Simulate loading data
    print("Loading data for verification...")
    try:
        df = pd.read_csv('data/processed/global-data-imputed.csv')
        
        # Targets and Feature Definitions
        target = 'Value_co2_emissions_kt_by_country'
        y = df[target]
        
        # Drop non-feature columns for X (Keep Entity/Year for Lags!)
        X = df.drop(columns=[target], errors='ignore') 
        
        # Define columns groups
        cat_cols = ['Entity']
        # Identify numeric columns, excluding Year if treated strictly as time index, but keeping for now if useful
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c != 'Year'] # Year usually not scaled/logged directly in this context

        print(f"Original shape: {X.shape}")

        # 1. Linear Regression Pipeline
        print("\n--- Testing Linear Regression Pipeline ---")
        lr_pipe = create_linear_regression_pipeline(num_cols, cat_cols)
        # Note: fit_transform might fail if VIFSelector receives numpy array but expects DF to check cols.
        # Our VIFSelector handles it by casting to DF, but column names are lost if previous step returns array.
        # ColumnTransformer returns array by default. 
        # FIX: We enable pandas output for sklearn >= 1.2 or wrap.
        lr_pipe.set_output(transform="pandas") 
        X_lr = lr_pipe.fit_transform(X, y)
        print(f"LR Output Shape: {X_lr.shape} (Reduced from OHE expansion by VIF)")

        # 2. SVR Pipeline
        print("\n--- Testing SVR Pipeline ---")
        svr_pipe = create_svr_pipeline(num_cols)
        svr_pipe.set_output(transform="pandas")
        # SVR pipeline only uses numeric columns in this design
        X_svr = svr_pipe.fit_transform(X[num_cols], y)
        print(f"SVR Output Shape: {X_svr.shape} (Filtered by correlation)")

        # 3. XGBoost Pipeline
        print("\n--- Testing XGBoost Pipeline ---")
        xgb_pipe = create_xgboost_pipeline(num_cols, cat_cols)
        xgb_pipe.set_output(transform="pandas")
        X_xgb = xgb_pipe.fit_transform(X, y)
        print(f"XGBoost Output Shape: {X_xgb.shape} (Includes Lags)")
        
        print("\nVerification Complete: All pipelines instantiated and ran on sample data.")

    except Exception as e:
        print(f"\nVerification Failed: {e}")
        import traceback
        traceback.print_exc()
