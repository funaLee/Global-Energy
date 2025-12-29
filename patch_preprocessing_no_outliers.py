import nbformat
import json

NB_PATH = 'notebooks/2_Model_Specific_Preprocessing.ipynb'

print(f"Reading {NB_PATH}...")
with open(NB_PATH, 'r') as f:
    nb = nbformat.read(f, as_version=4)

# Define the new code for the outlier removal cell - SKIPPING IT
new_code = """# 1.1 Outliers Removal (IQR) - SKIPPED
# Experiment: Keep ALL outliers to see impact on major economies and overall R2.
print("SKIPPING Outlier Removal for Linear Regression per User Request.")
# df_lr = remove_outliers(df_lr, method='iqr', threshold=3.0)
print(f"LR Data Shape (No Outliers Removed): {df_lr.shape}")

# 1.2 VIF Removal (Multicollinearity)
print("Running VIF Removal...")
df_lr = remove_high_vif(df_lr, target, threshold=10, exclude_cols=['Financial flows to developing countries (US $)'])

# 1.3 Standard Scaling
print("Running Standard Scaling...")
scaler_lr = StandardScaler()
numeric_cols_lr = df_lr.select_dtypes(include=['float64', 'int64']).columns
feature_cols_lr = [c for c in numeric_cols_lr if c != target and not c.startswith('Entity_')]
df_lr[feature_cols_lr] = scaler_lr.fit_transform(df_lr[feature_cols_lr])

df_lr.to_csv('../data/processed/lr_final_prep.csv', index=False)
print(f"Saved LR data: {df_lr.shape}")"""

# Find the cell to replace (Cell containing "1.1 Outliers Removal")
found = False
for cell in nb.cells:
    if cell.cell_type == 'code':
        if "1.1 Outliers Removal" in cell.source:
            print("Found target cell. Replacing...")
            
            # Split existing source to preserve imports/header if any
            # But simpler to just replace the block we know we modified last time
            # Last time we added "WITH WHITELIST PROTECTION"
            # Or the original "1.1 Outliers Removal (IQR)"
            
            # Let's handle the specific cell structure:
            # It starts with "# --- 1. Linear Regression ---" and ends with "df_lr.to_csv..."
            # But in the previous patch we replaced a middle chunk.
            # Let's look for the marker we inserted or the original one.
            
            parts = cell.source.split('# 1.1 Outliers Removal')
            if len(parts) > 1:
                # Keep the part before 1.1
                pre_block = parts[0]
                
                # Check for SVR block end
                svr_split = cell.source.split('# --- 2. SVR ---')
                if len(svr_split) > 1:
                    post_block = "\n\n# --- 2. SVR ---" + svr_split[1]
                else:
                    post_block = ""
                
                cell.source = pre_block + new_code + post_block
                found = True
                break

if found:
    with open(NB_PATH, 'w') as f:
        nbformat.write(nb, f)
    print("Notebook Patched Successfully (No Outliers).")
else:
    print("Could not find target cell pattern.")
