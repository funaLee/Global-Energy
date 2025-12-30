"""
AUDIT: Data Visualization Decisions vs Actual Implementation
==============================================================
This script verifies if the decisions made in Section 2 (Data Visualization)
are correctly implemented in the preprocessing code.
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

print("=" * 70)
print("AUDIT: DATA VIZ DECISIONS vs ACTUAL IMPLEMENTATION")
print("=" * 70)

# Load all datasets
df_raw = load_data('data/raw/global-data-on-sustainable-energy.csv')
df_common = load_data('data/processed/common_preprocessed.csv')
df_lr = pd.read_csv('data/processed/lr_final_prep.csv')

# Also check SVR and XGB if they exist
svr_exists = os.path.exists('data/processed/svr_final_prep.csv')
xgb_exists = os.path.exists('data/processed/xgb_final_prep.csv')

if svr_exists:
    df_svr = pd.read_csv('data/processed/svr_final_prep.csv')
if xgb_exists:
    df_xgb = pd.read_csv('data/processed/xgb_final_prep.csv')

results = []

# ==================== CHECK 1: COUNTRY FILTERING ====================
print("\n" + "=" * 70)
print("CHECK 1: COUNTRY FILTERING (Decision: >= 15 years)")
print("=" * 70)

# Count years per country in raw
years_per_country = df_raw.groupby('Entity')['Year'].count()
countries_with_15_plus = (years_per_country >= 15).sum()
countries_dropped_by_years = (years_per_country < 15).sum()

# What's actually in final LR?
map_df = pd.read_csv('data/processed/recovered_index_map.csv')
final_indices = map_df['Original_Index'].values
final_entities = df_common.loc[final_indices, 'Entity'].unique()

print(f"Raw Data: {df_raw['Entity'].nunique()} countries")
print(f"Countries with >= 15 years: {countries_with_15_plus}")
print(f"Countries with < 15 years: {countries_dropped_by_years}")
print(f"Final LR Dataset: {len(final_entities)} countries")

# Check if any country with <15 years made it through
low_year_countries = years_per_country[years_per_country < 15].index.tolist()
sneaky_countries = [c for c in low_year_countries if c in final_entities]
if sneaky_countries:
    print(f"⚠️ WARNING: Countries with <15 years in FINAL data: {sneaky_countries}")
    results.append(("Country Filtering", "PARTIAL", "Some low-data countries still in final"))
else:
    print(f"✅ PASS: No country with <15 years in final dataset")
    results.append(("Country Filtering", "PASS", "All low-data countries removed"))

# ==================== CHECK 2: LOG TRANSFORM ====================
print("\n" + "=" * 70)
print("CHECK 2: LOG TRANSFORM (Decision: Financial flows, Renewables)")
print("=" * 70)

# Check if Financial flows is log-transformed in common/lr
fin_col = 'Financial flows to developing countries (US $)'
ren_col = 'Renewables (% equivalent primary energy)'

if fin_col in df_common.columns and fin_col in df_raw.columns:
    # Log transform should make max value MUCH smaller
    raw_max = df_raw[fin_col].max()
    common_max = df_common[fin_col].max()
    # If log1p was applied, common_max should be ~log(raw_max)
    expected_log = np.log1p(raw_max)
    
    if abs(common_max - expected_log) < expected_log * 0.1:  # Within 10%
        print(f"✅ PASS: Financial flows appears log-transformed")
        print(f"   Raw max: {raw_max:,.0f} → Common max: {common_max:.2f} (expected ~{expected_log:.2f})")
        results.append(("Log Transform Financial", "PASS", "Log applied"))
    else:
        print(f"⚠️ CHECK: Financial flows may not be log-transformed")
        print(f"   Raw max: {raw_max:,.0f} → Common max: {common_max:,.0f}")
        results.append(("Log Transform Financial", "CHECK", "May not be applied"))

# ==================== CHECK 3: OUTLIER HANDLING ====================
print("\n" + "=" * 70)
print("CHECK 3: OUTLIER HANDLING (Decision: NO IQR for CO2)")
print("=" * 70)

# Check if major emitters are in final dataset
major_emitters = ['China', 'United States', 'India', 'Germany', 'Japan']
major_in_final = [c for c in major_emitters if c in final_entities]
major_missing = [c for c in major_emitters if c not in final_entities]

print(f"Major emitters in final: {major_in_final}")
if major_missing:
    print(f"⚠️ WARNING: Major emitters MISSING: {major_missing}")
    results.append(("Outlier Handling", "FAIL", f"Missing: {major_missing}"))
else:
    print(f"✅ PASS: All major emitters preserved (IQR not blindly applied)")
    results.append(("Outlier Handling", "PASS", "Major emitters preserved"))

# ==================== CHECK 4: VIF REMOVAL ====================
print("\n" + "=" * 70)
print("CHECK 4: VIF REMOVAL (Decision: Remove GDP, Access cols; Keep CO2_lag1)")
print("=" * 70)

# Check LR columns
lr_cols = df_lr.columns.tolist()

# These should be REMOVED
should_remove = ['gdp_per_capita', 'Access to electricity (% of population)', 
                 'Access to clean fuels for cooking']
# This should be KEPT
should_keep = ['Value_co2_emissions_kt_by_country_lag1']

removed_correctly = []
removed_incorrectly = []
kept_correctly = []
kept_incorrectly = []

for col in should_remove:
    if col in lr_cols:
        removed_incorrectly.append(col)
    else:
        removed_correctly.append(col)

for col in should_keep:
    if col in lr_cols:
        kept_correctly.append(col)
    else:
        kept_incorrectly.append(col)

print(f"Correctly REMOVED: {removed_correctly}")
print(f"Incorrectly STILL IN DATA: {removed_incorrectly}")
print(f"Correctly KEPT: {kept_correctly}")
print(f"Incorrectly REMOVED: {kept_incorrectly}")

if removed_incorrectly:
    results.append(("VIF Removal", "PARTIAL", f"Still has: {removed_incorrectly}"))
elif kept_incorrectly:
    results.append(("VIF Removal", "FAIL", f"Wrongly removed: {kept_incorrectly}"))
else:
    results.append(("VIF Removal", "PASS", "VIF applied correctly"))
    print(f"✅ PASS: VIF applied correctly")

# ==================== CHECK 5: 2020 DATA REMOVAL ====================
print("\n" + "=" * 70)
print("CHECK 5: 2020 DATA REMOVAL")
print("=" * 70)

# Check if 2020 is in any processed dataset
years_in_common = df_common['Year'].unique()
has_2020_common = 2020 in years_in_common

# Check LR via map
lr_years = df_common.loc[final_indices, 'Year'].unique()
has_2020_lr = 2020 in lr_years

print(f"2020 in Common Preprocessed: {has_2020_common}")
print(f"2020 in Final LR: {has_2020_lr}")

if has_2020_lr:
    print(f"⚠️ WARNING: 2020 data still in LR dataset!")
    results.append(("2020 Removal", "FAIL", "2020 still present in LR"))
else:
    print(f"✅ PASS: 2020 removed from LR dataset")
    results.append(("2020 Removal", "PASS", "2020 removed"))

# ==================== CHECK 6: SVR/XGB DIFFERENCES ====================
print("\n" + "=" * 70)
print("CHECK 6: ALGORITHM-SPECIFIC PREPROCESSING")
print("=" * 70)

if svr_exists:
    svr_cols = pd.read_csv('data/processed/svr_final_prep.csv', nrows=1).columns.tolist()
    # SVR should have One-Hot, NO outlier removal (more rows?)
    print(f"SVR columns: {len(svr_cols)}")
    print(f"SVR has Entity One-Hot: {'Entity_' in str(svr_cols)}")
else:
    print("⚠️ SVR preprocessed file not found")

if xgb_exists:
    xgb_df = pd.read_csv('data/processed/xgb_final_prep.csv', nrows=5)
    xgb_cols = xgb_df.columns.tolist()
    # XGB should have Ordinal encoding (numeric Entity column or original)
    has_entity_col = 'Entity' in xgb_cols
    has_entity_onehot = any('Entity_' in c for c in xgb_cols)
    print(f"XGB columns: {len(xgb_cols)}")
    print(f"XGB uses Ordinal (Entity as string/int): {has_entity_col and not has_entity_onehot}")
else:
    print("⚠️ XGB preprocessed file not found")

# ==================== FINAL SUMMARY ====================
print("\n" + "=" * 70)
print("FINAL AUDIT SUMMARY")
print("=" * 70)

print("\n| Check | Status | Notes |")
print("|---|---|---|")
for check, status, notes in results:
    emoji = "✅" if status == "PASS" else "⚠️" if status == "PARTIAL" else "❌"
    print(f"| {check} | {emoji} {status} | {notes} |")

# Overall verdict
pass_count = sum(1 for _, s, _ in results if s == "PASS")
total_count = len(results)
print(f"\nOVERALL: {pass_count}/{total_count} checks passed")

if pass_count == total_count:
    print("🎉 ALL DECISIONS FROM DATA VISUALIZATION ARE CORRECTLY IMPLEMENTED!")
else:
    print("⚠️ SOME DECISIONS MAY NEED REVIEW - See details above")
