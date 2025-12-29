import nbformat

NB_PATH = 'notebooks/2_Model_Specific_Preprocessing.ipynb'

print(f"Reading {NB_PATH}...")
with open(NB_PATH, 'r') as f:
    nb = nbformat.read(f, as_version=4)

# Define the Whitelist Code (Method A)
whitelist_code = """# 1.1 Outliers Removal (IQR) - WITH WHITELIST PROTECTION
# Goals: Remove statistical outliers but KEEP critical economies (USA, China, etc.)
# which are outliers by definition but vital for the model.

# List of Major Economies to Protect (Global Top Emitters/GDP)
WHITELIST = ['China', 'United States', 'India', 'Russian Federation', 'Japan', 'Germany', 'Brazil', 'Canada']
print(f"Whitelisted Entities (protected from outlier removal): {WHITELIST}")

# Strategy:
# 1. Split df into Whitelisted vs Non-Whitelisted
# 2. Apply Outlier Removal ONLY to Non-Whitelisted rows
# 3. Concatenate back together

# Identify index of whitelisted rows
whitelist_mask = pd.Series(False, index=df_lr.index)
for country in WHITELIST:
    col_name = f'Entity_{country}'
    if col_name in df_lr.columns:
        whitelist_mask = whitelist_mask | (df_lr[col_name] == 1)

df_protected = df_lr[whitelist_mask].copy()
df_to_clean = df_lr[~whitelist_mask].copy()

print(f"Protected Rows: {len(df_protected)}, Rows to Clean: {len(df_to_clean)}")

# Remove outliers from the 'To Clean' subset using original threshold
# Note: remove_outliers function in src/preprocessing.py handles the basic logic
df_cleaned_subset = remove_outliers(df_to_clean, method='iqr', threshold=3.0)

# Merge back and sort
df_lr = pd.concat([df_protected, df_cleaned_subset], axis=0).sort_index()
print(f"Final LR Data Shape after Whitelist Protection: {df_lr.shape}")

# 1.2 VIF Removal (Multicollinearity)
print("Running VIF Removal...")
df_lr = remove_high_vif(df_lr, target, threshold=10, exclude_cols=['Financial flows to developing countries (US $)'])

# 1.3 Standard Scaling
print("Running Standard Scaling...")
scaler_lr = StandardScaler()
numeric_cols_lr = df_lr.select_dtypes(include=['float64', 'int64']).columns
feature_cols_lr = [c for c in numeric_cols_lr if c != target and not c.startswith('Entity_')]\n
df_lr[feature_cols_lr] = scaler_lr.fit_transform(df_lr[feature_cols_lr])

df_lr.to_csv('../data/processed/lr_final_prep.csv', index=False)
print(f"Saved LR data: {df_lr.shape}")"""

# Find and Replace
found = False
for cell in nb.cells:
    if cell.cell_type == 'code':
        if "Outliers Removal" in cell.source:
            # Replace the whole block starting from 1.1 down to save
            parts = cell.source.split('# 1.1 Outliers Removal')
            if len(parts) > 1:
                pre_block = parts[0]
                
                svr_split = cell.source.split('# --- 2. SVR ---')
                if len(svr_split) > 1:
                    post_block = "\n\n# --- 2. SVR ---" + svr_split[1]
                else:
                    post_block = ""
                
                cell.source = pre_block + whitelist_code + post_block
                found = True
                break

if found:
    with open(NB_PATH, 'w') as f:
        nbformat.write(nb, f)
    print("Notebook Patched Successfully (Restored Whitelist).")
else:
    print("Could not find target cell.")
