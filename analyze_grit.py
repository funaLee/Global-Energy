import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('src'))
from preprocessing import load_data, remove_outliers, encode_features

print("--- Analyzing 'Noisy' Outliers ---")

# 1. Load Data
df = load_data('data/processed/common_preprocessed.csv')
print(f"Original Data: {df.shape}")

# 2. Simulate Outlier Removal (Threshold 3.0) on Key Features
# We focus on the numeric columns used in the model
numeric_cols = df.select_dtypes(include=[np.number]).columns
# Exclude One-Hot/Year/Target from detection if we want to be strict, 
# but let's use the exact logic from preprocessing.py (mimicked)
# Preprocessing One-Hots first, then removes.
# But simply checking distribution of key vars is enough to find the "grit".

key_vars = ['Value_co2_emissions_kt_by_country', 
            'Access to electricity (% of population)',
            'Renewables (% equivalent primary energy)',
            'gdp_per_capita',
            'Primary energy consumption per capita (kWh/person)']

print(f"\nScanning for outliers in: {key_vars}")

# Calculate IQR limits
outlier_records = []
whitelist = ['China', 'United States', 'India', 'Germany', 'Japan', 'Russian Federation', 'Brazil', 'Canada']

for col in key_vars:
    if col not in df.columns: continue
    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3.0 * IQR
    upper = Q3 + 3.0 * IQR
    
    # Find outliers
    mask = (df[col] < lower) | (df[col] > upper)
    outliers = df[mask].copy()
    
    # Filter out Whitelist (Good Outliers)
    # effective_outliers are the ones we call "Grit"
    grit = outliers[~outliers['Entity'].isin(whitelist)]
    
    if len(grit) > 0:
        print(f"\nVariable: {col}")
        print(f"  Thresholds: < {lower:.2f} or > {upper:.2f}")
        print(f"  Total Outliers: {len(outliers)}")
        print(f"  'Good' Outliers (Whitelist): {len(outliers) - len(grit)}")
        print(f"  'Grit' Outliers (Noise): {len(grit)}")
        
        # Top contributing countries to Grit
        top_grit = grit['Entity'].value_counts().head(5)
        print(f"  Top 'Grit' Countries: {top_grit.to_dict()}")
        
        # Show specific examples
        for country, count in top_grit.items():
            example_val = grit[grit['Entity'] == country][col].iloc[0]
            print(f"    - {country}: {example_val:.2f}")

# 3. Overall Dropped Rows Analysis
# Let's apply the full multi-variate filter
mask_all = pd.Series(False, index=df.index)
for col in key_vars:
    if col not in df.columns: continue
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    mask_all = mask_all | (df[col] < (Q1 - 3.0*IQR)) | (df[col] > (Q3 + 3.0*IQR))

grit_all = df[mask_all & ~df['Entity'].isin(whitelist)]
print(f"\n--- Summary of 'Grit' ---")
print(f"Total Unique Rows identified as Noise: {len(grit_all)}")
print("Top 10 Countries with most 'Noisy' rows:")
print(grit_all['Entity'].value_counts().head(10))
