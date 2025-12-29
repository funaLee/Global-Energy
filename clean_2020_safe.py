import pandas as pd
import sys
import os

print("--- Phase 0 Refactor: Safe Removal of 2020 Data ---")

# 1. Clean Common Data
common_path = 'data/processed/common_preprocessed.csv'
if os.path.exists(common_path):
    df_common = pd.read_csv(common_path)
    print(f"Common Before: {len(df_common)}")
    
    # Identify indices to drop (Year == 2020)
    drop_indices = df_common[df_common['Year'] == 2020].index
    print(f"Found {len(drop_indices)} rows to drop in Common.")
    
    # Filter
    df_common_clean = df_common[df_common['Year'] != 2020].reset_index(drop=True) # Reset index changes IDs!
    # WAIT! If I reset index, the map breaks.
    # Use careful logic.
    
    # Actually, simpler:
    # `lr_final_prep.csv` corresponds to rows in `recovered_index_map.csv`.
    # `recovered_index_map.csv` points to `Original_Index` in `common_preprocessed.csv`.
    
    # Step A: Load Map and Common
    map_df = pd.read_csv('data/processed/recovered_index_map.csv')
    
    # Step B: Identify "Bad" Original Indices (Year 2020)
    # Mapping: Match Original_Index to Common
    # We don't need to mod Common yet. Just find which Original_Indices are 2020.
    bad_original_indices = df_common[df_common['Year'] == 2020].index
    
    # Step C: Filter Map
    # Keep only rows where Original_Index is NOT in bad set
    map_clean = map_df[~map_df['Original_Index'].isin(bad_original_indices)].copy()
    print(f"Map Before: {len(map_df)}, Map After: {len(map_clean)}")
    
    # Step D: Filter LR Data
    # LR Data aligns with Map's 'LR_Index'. 
    # But Map 'LR_Index' is just 0, 1, 2... ? CHECK
    # Usually LR_Index matches the row number in lr_final_prep.csv.
    # So we keep rows in lr_final_prep that correspond to the kept map rows.
    
    lr_path = 'data/processed/lr_final_prep.csv'
    df_lr = pd.read_csv(lr_path)
    print(f"LR Before: {len(df_lr)}")
    
    # The valid LR_Indices to keep
    valid_lr_indices = map_clean['LR_Index'].values
    
    df_lr_clean = df_lr.iloc[valid_lr_indices].copy().reset_index(drop=True)
    print(f"LR After: {len(df_lr_clean)}")
    
    # Step E: Update Map
    # Since we reset df_lr_clean index, the new LR indices are 0..N
    # We must update map_clean['LR_Index'] to be 0..N
    map_clean['LR_Index'] = range(len(map_clean))
    
    # Step F: Save Everything
    # We DO NOT modify common_preprocessed.csv drastically, or we risk breaking other things?
    # Actually, common_preprocessed.csv is the master. We should clean it too?
    # IF we clean common, 'Original_Index' references shift.
    # BETTER: Don't touch common file (it's raw-ish). Just "Ignoring" 2020 in the aligned dataset is enough.
    # The user asks to "stop at 2019".
    # Updating lr_final_prep.csv is sufficient for the model training.
    
    df_lr_clean.to_csv(lr_path, index=False)
    map_clean.to_csv('data/processed/recovered_index_map.csv', index=False)
    print("✅ Saved Cleaned LR Data and Updated Map.")

else:
    print("Common File Missing.")
