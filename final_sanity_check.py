import pandas as pd
import numpy as np
import sys
import os
import nbformat
import subprocess

print("==================================================")
print("       FINAL PIPELINE SANITY CHECK PROTOCOL       ")
print("==================================================\n")

all_passed = True

# --- CHECK 1: CODE LOGIC (NOTEBOOK) ---
print("[1] Verifying Preprocessing Logic (Notebook 2)...")
nb_path = 'notebooks/2_Model_Specific_Preprocessing.ipynb'
with open(nb_path, 'r') as f:
    nb_content = f.read()

if "WHITELIST" in nb_content and "Method A" not in nb_content: # Just checking key terms
    print("✅ OK: Notebook contains 'WHITELIST' logic.")
else:
    print("❌ FAIL: Notebook does not appear to have Whitelist logic active.")
    all_passed = False

if "SKIPPING Outlier Removal" in nb_content:
    print("❌ FAIL: Notebook is set to SKIP outlier removal entirely!")
    all_passed = False
else:
    print("✅ OK: Notebook is NOT skipping outlier removal (good).")


# --- CHECK 2: DATA INTEGRITY (CSV) ---
print("\n[2] Verifying Data File (lr_final_prep.csv)...")
try:
    # We need to map it back to check Entities
    df_lr = pd.read_csv('data/processed/lr_final_prep.csv')
    df_common = pd.read_csv('data/processed/common_preprocessed.csv')
    map_df = pd.read_csv('data/processed/recovered_index_map.csv')
    
    aligned_indices = map_df['Original_Index'].values
    entities = df_common.loc[aligned_indices, 'Entity'].values
    
    # Check Giants
    giants = ['United States', 'China', 'India']
    missing_giants = [g for g in giants if g not in entities]
    
    if not missing_giants:
         print(f"✅ OK: Giants found: {giants}")
    else:
         print(f"❌ FAIL: Missing Giants: {missing_giants}")
         all_passed = False

    # Check Noise
    noise = ['Bermuda', 'Qatar']
    found_noise = [n for n in noise if n in entities]
    
    if not found_noise:
        print(f"✅ OK: Noise removed: {noise}")
    else:
        print(f"❌ FAIL: Noise still present: {found_noise}")
        all_passed = False
        
    # Check Count
    count = len(df_lr)
    print(f"   Record Count: {count}")
    if 2100 < count < 2300: # Expected ~2232
        print("✅ OK: Record count is within expected range for Whitelist strategy.")
    else:
        print("warning: Record count seems off (Expected ~2232)")

except Exception as e:
    print(f"❌ CRITICAL FAIL: Could not load/verify data. {e}")
    all_passed = False


# --- CHECK 3: METRIC CONSISTENCY ---
print("\n[3] Verifying Metrics Consistency...")
# Run recalc
try:
    result = subprocess.run(['/home/funalee/UIT/DS102/Global-Energy/.venv/bin/python3', 'recalc_metrics.py'], capture_output=True, text=True)
    output = result.stdout
    
    if "0.78" in output:
        print("✅ OK: Calc Script confirms R2 ~ 0.78")
    else:
        print("❌ FAIL: Calc Script output mismatch (Expected 0.78)")
        print(f"Output snippet: {output[-200:]}")
        all_passed = False
except Exception as e:
    print(f"❌ CRITICAL FAIL: Could not run metric script. {e}")
    all_passed = False

# --- CHECK 4: REPORT CONSISTENCY ---
print("\n[4] Verifying Report (Markdown)...")
with open('reports/Final_Project_Report.md', 'r') as f:
    report = f.read()

if "0.782" in report and "0.533" in report:
    print("✅ OK: Report mentions correct updated metrics (0.782 / 0.533).")
else:
    print("❌ FAIL: Report metrics do not match latest validation.")
    all_passed = False

if "Panel (Fixed Effects)" in report:
    print("✅ OK: Report correctly labels model as Panel Fixed Effects.")
else:
    print("warning: Report might miss the 'Fixed Effects' terminology.")


print("\n==================================================")
if all_passed:
    print("             ALL CHECKS PASSED ✅                 ")
    print("   Pipeline is Consistent, Valid, and Ready.      ")
else:
    print("             SOME CHECKS FAILED ❌                ")
    print("           Review errors above.                   ")
print("==================================================")
