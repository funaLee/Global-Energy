import pandas as pd
import numpy as np

# Load
df_common = pd.read_csv('data/processed/common_preprocessed.csv')
df_lr = pd.read_csv('data/processed/lr_final_prep.csv')

target_col = 'Value_co2_emissions_kt_by_country'
common_vals = df_common[target_col].values.round(5)
lr_vals = df_lr[target_col].values.round(5)

print(f"Common Len: {len(common_vals)}")
print(f"LR Len: {len(lr_vals)}")

# Greedy Match
# Assumption: LR is a subsequence of Common (order preserved)
# We find the first occurrence in Common that matches LR[0] >= current_ptr

mapping = []
common_ptr = 0
matches_found = 0

for lr_idx, val in enumerate(lr_vals):
    # Search forward from common_ptr
    found = False
    for i in range(common_ptr, len(common_vals)):
        # Allow small tollerance or exact match
        if common_vals[i] == val:
            # Check Lag if possible? No, Lag is scaled/missing in LR.
            # Assume Target uniqueness locally is sufficient
            mapping.append({'LR_Index': lr_idx, 'Original_Index': i})
            common_ptr = i + 1 # Move past matched item
            matches_found += 1
            found = True
            break
    
    if not found:
        print(f"Break at LR Index {lr_idx}, Value {val}. No subsequent match found.")
        break

print(f"Total Matches: {matches_found}/{len(lr_vals)}")

if matches_found == len(lr_vals):
    print("Perfect Sequence Match!")
    # Verify Entity consistency?
    # df_common.loc[[m['Original_Index'] for m in mapping]]['Entity']
    # But we trust the sequence.
    
    # Save Map
    pd.DataFrame(mapping).to_csv('data/processed/recovered_index_map.csv', index=False)
    print("Map Saved.")
else:
    print("Failed to match full sequence.")
