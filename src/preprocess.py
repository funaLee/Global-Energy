
import pandas as pd
import numpy as np
import os

def preprocess_data(input_path, output_path):
    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return

    # Check for missing values before imputation
    print("\nMissing values before imputation:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # Median Imputation for numeric columns
    print("\nPerforming Median Imputation...")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Imputed {col} with median: {median_val}")

    # Verify no missing values in numeric columns
    print("\nMissing values after imputation (numeric):")
    print(df[numeric_columns].isnull().sum()[df[numeric_columns].isnull().sum() > 0])
    
    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to {output_path}")

if __name__ == "__main__":
    INPUT_PATH = 'data/raw/global-data-on-sustainable-energy.csv'
    OUTPUT_PATH = 'data/processed/global-data-imputed.csv'
    preprocess_data(INPUT_PATH, OUTPUT_PATH)
