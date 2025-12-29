import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

print("--- Visualizing Outliers & Whitelist Justification ---")
# Load Common Data (Pre-Outlier Removal)
df = load_data('data/processed/common_preprocessed.csv')

# Features to check
target = 'Value_co2_emissions_kt_by_country'
cols = [target, 'gdp_per_capita', 'Primary energy consumption per capita (kWh/person)']

plt.figure(figsize=(15, 6))

# 1. Boxplot of CO2
plt.subplot(1, 1, 1)
sns.boxplot(x=df[target])
plt.title('Distribution of CO2 Emissions (Pre-Cleaning)')
plt.xlabel('CO2 (kt)')

# Calculate IQR Thresholds
Q1 = df[target].quantile(0.25)
Q3 = df[target].quantile(0.75)
IQR = Q3 - Q1
Upper_Bound = Q3 + 1.5 * IQR

plt.axvline(Upper_Bound, color='r', linestyle='--', label=f'IQR Threshold ({Upper_Bound:,.0f})')

# Annotate Whitelisted Countries (Max values)
whitelist = ['China', 'United States', 'India', 'Germany']
for country in whitelist:
    # Get max value for that country
    val = df[df['Entity'] == country][target].max()
    plt.text(val, 0, f" {country}\n ({val:,.0f})", verticalalignment='bottom', color='blue', fontweight='bold')
    plt.plot(val, 0, 'bo')

plt.legend()
plt.tight_layout()
plt.savefig('reports/figures/outlier_justification_plot.png')
print(f"Plot saved to reports/figures/outlier_justification_plot.png")
print(f"IQR Upper Bound: {Upper_Bound:,.0f}")
print(f"China Max: {df[df['Entity']=='China'][target].max():,.0f}")
print(f"US Max: {df[df['Entity']=='United States'][target].max():,.0f}")
