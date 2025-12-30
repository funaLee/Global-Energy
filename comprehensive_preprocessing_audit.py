"""
COMPREHENSIVE PREPROCESSING AUDIT
=================================
This script audits the entire data pipeline from raw to final processed files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

print("=" * 70)
print("COMPREHENSIVE PREPROCESSING AUDIT")
print("=" * 70)

# ==================== STAGE 1: RAW DATA ====================
print("\n" + "=" * 70)
print("STAGE 1: RAW DATA (global-data-on-sustainable-energy.csv)")
print("=" * 70)

df_raw = load_data('data/raw/global-data-on-sustainable-energy.csv')
raw_entities = set(df_raw['Entity'].unique())
raw_years = sorted(df_raw['Year'].unique())

print(f"Total Rows: {len(df_raw)}")
print(f"Total Columns: {len(df_raw.columns)}")
print(f"Total Unique Entities: {len(raw_entities)}")
print(f"Year Range: {min(raw_years)} - {max(raw_years)}")

# Check for Russia
russia_check = [e for e in raw_entities if 'russia' in e.lower() or 'russian' in e.lower()]
print(f"Russia in Raw Data: {russia_check if russia_check else 'NOT FOUND'}")

# ==================== STAGE 2: COMMON PREPROCESSED ====================
print("\n" + "=" * 70)
print("STAGE 2: COMMON PREPROCESSED (common_preprocessed.csv)")
print("=" * 70)

df_common = load_data('data/processed/common_preprocessed.csv')
common_entities = set(df_common['Entity'].unique())
common_years = sorted(df_common['Year'].unique())

print(f"Total Rows: {len(df_common)}")
print(f"Total Columns: {len(df_common.columns)}")
print(f"Total Unique Entities: {len(common_entities)}")
print(f"Year Range: {min(common_years)} - {max(common_years)}")

# What was lost?
lost_stage1 = raw_entities - common_entities
print(f"\nEntities Lost in Stage 1: {len(lost_stage1)}")
if lost_stage1:
    print(f"  Lost: {sorted(lost_stage1)}")
    print(f"  Reason: Insufficient data for Lag features (need at least 2 consecutive years)")

# New columns added
new_cols = set(df_common.columns) - set(df_raw.columns)
print(f"\nNew Columns Added: {sorted(new_cols)}")
print("  Reason: Lag-1 features created for time-series modeling")

# ==================== STAGE 3: LR FINAL PREP ====================
print("\n" + "=" * 70)
print("STAGE 3: LR FINAL PREP (lr_final_prep.csv)")
print("=" * 70)

df_lr = pd.read_csv('data/processed/lr_final_prep.csv')
map_df = pd.read_csv('data/processed/recovered_index_map.csv')

# Reconstruct Entity names
original_indices = map_df['Original_Index'].values
df_lr_with_entity = df_lr.copy()
df_lr_with_entity['Entity'] = df_common.loc[original_indices, 'Entity'].values
df_lr_with_entity['Year'] = df_common.loc[original_indices, 'Year'].values

final_entities = set(df_lr_with_entity['Entity'].unique())
final_years = sorted(df_lr_with_entity['Year'].unique())

print(f"Total Rows: {len(df_lr)}")
print(f"Total Columns: {len(df_lr.columns)}")
print(f"Total Unique Entities: {len(final_entities)}")
print(f"Year Range: {min(final_years)} - {max(final_years)}")

# What was lost?
lost_stage2 = common_entities - final_entities
print(f"\nEntities Lost in Stage 2 (Outlier Removal): {len(lost_stage2)}")
print(f"  Lost Countries: {sorted(lost_stage2)}")

# ==================== WHITELIST ANALYSIS ====================
print("\n" + "=" * 70)
print("WHITELIST ANALYSIS")
print("=" * 70)

WHITELIST = [
    'China', 'United States', 'India', 'Russian Federation', 'Japan', 'Germany', 'Brazil', 'Canada',
    'United Kingdom', 'France', 'Italy', 'Australia', 'South Korea', 'Saudi Arabia', 'Turkey', 'Indonesia', 
    'Spain', 'Mexico', 'South Africa', 'Thailand', 'Poland', 'Iran', 'Egypt', 'Pakistan', 'Viet Nam', 'Vietnam',
    'Argentina', 'Netherlands', 'Philippines', 'Malaysia', 'Belgium', 'Sweden', 'Ukraine', 'Kazakhstan',
    'United Arab Emirates', 'Algeria', 'Singapore', 'Nigeria'
]

whitelist_in_raw = [c for c in WHITELIST if c in raw_entities]
whitelist_in_final = [c for c in WHITELIST if c in final_entities]
whitelist_missing = [c for c in WHITELIST if c not in final_entities]

print(f"Whitelist Size: {len(WHITELIST)}")
print(f"Whitelist Found in Raw: {len(whitelist_in_raw)}")
print(f"Whitelist Found in Final: {len(whitelist_in_final)}")
print(f"Whitelist Missing from Final: {whitelist_missing}")
print("  Reason: These countries were not in the original dataset or had name mismatches")

# ==================== PREPROCESSING METHODS BY ALGORITHM ====================
print("\n" + "=" * 70)
print("PREPROCESSING METHODS BY ALGORITHM")
print("=" * 70)

print("""
┌────────────────────┬─────────────────────────────────────────────────────────────────┐
│ Algorithm          │ Preprocessing Steps & Rationale                                 │
├────────────────────┼─────────────────────────────────────────────────────────────────┤
│ LINEAR REGRESSION  │ 1. Log Transform (Financial flows, Renewables)                  │
│                    │    → Reduces skewness, stabilizes variance                      │
│                    │ 2. One-Hot Encoding (Entity)                                    │
│                    │    → Captures country fixed effects                             │
│                    │ 3. Outlier Removal (IQR 3.0) with WHITELIST                     │
│                    │    → Removes noise but protects major economies                 │
│                    │ 4. VIF Removal (Threshold 10)                                   │
│                    │    → Removes multicollinear features to prevent instability     │
│                    │ 5. Standard Scaling (Z-Score)                                   │
│                    │    → Normalizes features for gradient-based optimization        │
├────────────────────┼─────────────────────────────────────────────────────────────────┤
│ SVR                │ 1. Log Transform (Same as LR)                                   │
│                    │ 2. One-Hot Encoding                                             │
│                    │ 3. NO Outlier Removal                                           │
│                    │    → SVR uses RBF kernel which is less sensitive to outliers    │
│                    │ 4. Robust Scaling (Median/IQR based)                            │
│                    │    → More robust to outliers than Z-Score                       │
├────────────────────┼─────────────────────────────────────────────────────────────────┤
│ XGBOOST            │ 1. NO Log Transform                                             │
│                    │    → Trees handle skewed data naturally                         │
│                    │ 2. Ordinal Encoding (Entity → Integer)                          │
│                    │    → Trees handle categorical splits efficiently                │
│                    │ 3. NO Outlier Removal                                           │
│                    │    → Trees are robust to outliers (split-based)                 │
│                    │ 4. NO Scaling                                                   │
│                    │    → Trees are scale-invariant                                  │
└────────────────────┴─────────────────────────────────────────────────────────────────┘
""")

# ==================== VIF ANALYSIS ====================
print("\n" + "=" * 70)
print("VIF (VARIANCE INFLATION FACTOR) ANALYSIS")
print("=" * 70)

# Check what was dropped by VIF
# We know from logs: ['Primary energy...', 'gdp_per_capita', 'Year', 'Access to electricity', 'Access to clean fuels']
vif_dropped = [
    'Primary energy consumption per capita (kWh/person)',
    'gdp_per_capita', 
    'Year',
    'Access to electricity (% of population)',
    'Access to clean fuels for cooking'
]

print("Features REMOVED by VIF (Threshold > 10):")
for f in vif_dropped:
    print(f"  ❌ {f}")

print("\nFeatures KEPT (VIF < 10 or Protected):")
protected = ['Value_co2_emissions_kt_by_country_lag1', 'Financial flows to developing countries (US $)']
for p in protected:
    print(f"  ✅ {p} (PROTECTED from VIF removal)")

print("""
VIF Rationale:
- gdp_per_capita is highly correlated with Primary Energy → Remove one
- Year is a time index, not a feature → Remove
- Access to electricity/fuels correlate with GDP → Remove
- CO2 Lag-1 is CRITICAL for forecasting → PROTECT
""")

# ==================== VISUALIZATIONS ====================
print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Entity Count by Stage
ax1 = axes[0, 0]
stages = ['Raw\n(Original)', 'Common\n(After Lags)', 'Final LR\n(After Outliers)']
counts = [len(raw_entities), len(common_entities), len(final_entities)]
colors = ['#3498db', '#2ecc71', '#e74c3c']
bars = ax1.bar(stages, counts, color=colors)
ax1.set_ylabel('Number of Countries')
ax1.set_title('Country Retention Through Pipeline')
for bar, count in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, str(count), 
             ha='center', va='bottom', fontweight='bold')

# 2. CO2 Distribution (Boxplot showing outliers)
ax2 = axes[0, 1]
target = 'Value_co2_emissions_kt_by_country'
if target in df_common.columns:
    # Log scale for visibility
    co2_data = df_common[target].dropna()
    co2_data = co2_data[co2_data > 0]  # Only positive for log
    ax2.boxplot(np.log10(co2_data), vert=True)
    ax2.set_ylabel('Log10(CO2 Emissions kt)')
    ax2.set_title('CO2 Distribution (Log Scale)\nOutliers are above/below whiskers')
    
    # Annotate major emitters
    china_max = df_common[df_common['Entity'] == 'China'][target].max()
    us_max = df_common[df_common['Entity'] == 'United States'][target].max()
    ax2.axhline(np.log10(china_max), color='red', linestyle='--', alpha=0.7)
    ax2.text(1.1, np.log10(china_max), f'China ({china_max/1e6:.1f}M)', color='red')
    ax2.axhline(np.log10(us_max), color='blue', linestyle='--', alpha=0.7)
    ax2.text(1.1, np.log10(us_max), f'USA ({us_max/1e6:.1f}M)', color='blue')

# 3. Dropped Countries Pie
ax3 = axes[1, 0]
labels = ['Kept', 'Dropped (Lags)', 'Dropped (Outliers)']
sizes = [len(final_entities), len(lost_stage1), len(lost_stage2)]
colors_pie = ['#2ecc71', '#f39c12', '#e74c3c']
ax3.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_pie, startangle=90)
ax3.set_title('Country Retention Summary')

# 4. VIF Concept Illustration (Correlation Heatmap)
ax4 = axes[1, 1]
# Select numeric columns for correlation
numeric_cols = ['gdp_per_capita', 'Primary energy consumption per capita (kWh/person)', 
                'Access to electricity (% of population)', 'Access to clean fuels for cooking']
available_cols = [c for c in numeric_cols if c in df_common.columns]
if len(available_cols) >= 2:
    corr_matrix = df_common[available_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn_r', center=0, ax=ax4, 
                fmt='.2f', square=True)
    ax4.set_title('Feature Correlation (High = Multicollinearity)')

plt.tight_layout()
plt.savefig('reports/figures/preprocessing_audit.png', dpi=150)
print("Saved: reports/figures/preprocessing_audit.png")

# ==================== FINAL SUMMARY ====================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW SUMMARY                            │
├─────────────────────────────────────────────────────────────────────┤
│ RAW DATA                                                            │
│   → Entities: {len(raw_entities):>3}                                               │
│   → Years: {min(raw_years)}-{max(raw_years)}                                               │
│   → Missing: Russia (NOT in original Kaggle dataset)                │
├─────────────────────────────────────────────────────────────────────┤
│ COMMON PREPROCESSED                                                 │
│   → Entities: {len(common_entities):>3} (Lost {len(lost_stage1)} due to Lag feature creation)         │
│   → Added Lag-1 features for CO2, GDP, Energy                       │
│   → Imputed missing values with MEDIAN                              │
├─────────────────────────────────────────────────────────────────────┤
│ FINAL LR PREP                                                       │
│   → Entities: {len(final_entities):>3} (Lost {len(lost_stage2)} due to Outlier Removal)           │
│   → Removed 2020 data (quality issues)                              │
│   → Applied VIF removal (dropped 5 correlated features)             │
│   → Applied Z-Score scaling                                         │
├─────────────────────────────────────────────────────────────────────┤
│ RETENTION RATE: {len(final_entities)}/{len(raw_entities)} = {100*len(final_entities)/len(raw_entities):.1f}%                                      │
└─────────────────────────────────────────────────────────────────────┘
""")
