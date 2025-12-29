import json
import os

notebook_path = 'notebooks/10_Recursive_Forecasting.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# New Plotting Code
new_source = [
    "# Plot Global Trend Comparison (3 Lines)\n",
    "# Aggregating by Year to show the global divergence\n",
    "global_trends = results_df.groupby('Year')[['Actual', 'Teacher_Forcing_Pred', 'Recursive_Pred']].sum()\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.plot(global_trends.index, global_trends['Actual'], marker='o', label='Actual (Ground Truth)', color='black', linewidth=2)\n",
    "plt.plot(global_trends.index, global_trends['Teacher_Forcing_Pred'], marker='s', linestyle='--', label='One-Step Ahead (Teacher Forcing)', color='green')\n",
    "plt.plot(global_trends.index, global_trends['Recursive_Pred'], marker='x', linestyle='-.', label='Recursive Forecasting', color='red')\n",
    "\n",
    "plt.title('Global CO2 Emissions Forecast: Reality vs Models')\n",
    "plt.ylabel('Total CO2 Emissions (kt)')\n",
    "plt.xlabel('Year')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.savefig('../reports/figures/recursive_comparison_plot.png')\n",
    "print(\"Plot saved to ../reports/figures/recursive_comparison_plot.png\")"
]

# Find the cell to replace
# Looking for the cell that calculates 'Error_TF' and plots 'Error Propagation'
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_text = "".join(cell['source'])
        if "Error_TF" in source_text and "plt.plot" in source_text:
            cell['source'] = new_source
            # Clear output to ensure clean execution state (optional but good practice)
            cell['outputs'] = []
            found = True
            break

if found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully.")
else:
    print("Target cell not found.")
