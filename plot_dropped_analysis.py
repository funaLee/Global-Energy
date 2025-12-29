import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

print("--- Visualizing Dropped Countries Analysis ---")
# 1. Re-run analysis logic
df_raw = load_data('data/raw/global-data-on-sustainable-energy.csv')
raw_entities = set(df_raw['Entity'].unique())

map_df = pd.read_csv('data/processed/recovered_index_map.csv')
df_common = load_data('data/processed/common_preprocessed.csv')
final_indices = map_df['Original_Index'].unique()
df_final = df_common.loc[final_indices]
final_entities = set(df_final['Entity'].unique())

dropped = raw_entities - final_entities
kept = final_entities

# Categorize Dropped
# 1. Whitelisted (Should be Kept)
whitelist = ['China', 'United States', 'India', 'Russian Federation', 'Japan', 'Germany', 'Brazil', 'Canada']
# Verify they are kept
missing_whitelist = set(whitelist) - kept
print(f"Missing Whitelisted (Should be 0): {len(missing_whitelist)} {missing_whitelist}")

# 2. Plot
labels = ['Kept (Training)', 'Dropped (Outliers/Noise)']
sizes = [len(kept), len(dropped)]
colors = ['#4CAF50', '#F44336']

plt.figure(figsize=(10, 6))
# Pie Chart
plt.subplot(1, 2, 1)
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title(f'Data Retention (Total: {len(raw_entities)} Countries)')

# Text List of Key Dropped Nations
plt.subplot(1, 2, 2)
plt.axis('off')
dropped_list = sorted(list(dropped))
# Show top 20 alphabetically or maybe manually pick significant ones
# Let's show a random sample or "Tier 2 Economies"
tier2 = ['United Kingdom', 'France', 'Italy', 'Australia', 'Saudi Arabia', 'South Korea', 'Turkey', 'Spain', 'Indonesia']
actually_dropped_tier2 = [c for c in tier2 if c in dropped]

text = "Status of Major Economies:\n\n"
text += f"✅ Whitelisted (Kept): {len(whitelist)}\n"
text += f"   {', '.join(whitelist[:4])}...\n\n"
text += f"❌ Filtered as Outliers (Dropped): {len(dropped)}\n"
text += "   (Too large/unique vs Global Median)\n"
text += "   Significant Examples:\n"
for c in actually_dropped_tier2:
    text += f"   - {c}\n"
text += f"   ...and {len(dropped)-len(actually_dropped_tier2)} others."

plt.text(0.1, 0.5, text, fontsize=12, verticalalignment='center')

plt.tight_layout()
plt.savefig('reports/figures/dropped_entities_report.png')
print("Saved reports/figures/dropped_entities_report.png")
