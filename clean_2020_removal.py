import pandas as pd
import sys
import os

print("--- Phase 0 Refactor: Removing Corrupt 2020 Data ---")

files = [
    'data/processed/lr_final_prep.csv',
    'data/processed/common_preprocessed.csv'
]

for fp in files:
    if os.path.exists(fp):
        df = pd.read_csv(fp)
        count_before = len(df)
        
        # Filter
        df_clean = df[df['Year'] != 2020].copy()
        count_after = len(df_clean)
        
        diff = count_before - count_after
        print(f"File: {fp}")
        print(f"  Rows Before: {count_before}")
        print(f"  Rows After:  {count_after}")
        print(f"  Dropped:     {diff} (Year=2020)")
        
        if diff > 0:
            df_clean.to_csv(fp, index=False)
            print("  ✅ Saved Cleaned File.")
        else:
            print("  ⚠️ No 2020 rows found or already cleaned.")
    else:
        print(f"File not found: {fp}")

print("Phase 0 Complete.")
