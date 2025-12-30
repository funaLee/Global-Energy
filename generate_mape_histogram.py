"""
Generate MAPE Distribution Histogram for Section 4.2
Visualizes the skewness in MAPE distribution caused by micro-states
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

def calculate_entity_mape(y_true, y_pred, entities):
    """Calculate MAPE per entity (country)."""
    df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred,
        'entity': entities
    })
    
    entity_mape = df.groupby('entity').apply(
        lambda g: np.mean(np.abs(g['actual'] - g['predicted']) / np.abs(g['actual'])) * 100
        if (g['actual'] != 0).all() else np.nan
    ).dropna()
    
    return entity_mape

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/processed/common_preprocessed.csv')
    
    # Column name mapping (actual names from dataset)
    target_col = 'Value_co2_emissions_kt_by_country'
    lag1_col = 'Value_co2_emissions_kt_by_country_lag1'
    
    # Prepare features
    print("Preparing features...")
    print(f"Columns available: {list(df.columns)[:10]}...")
    
    feature_columns = [
        'Year', 'gdp_per_capita', 'gdp_growth',
        'Primary energy consumption per capita (kWh/person)', 
        'Electricity from fossil fuels (TWh)',
        'Electricity from nuclear (TWh)', 'Electricity from renewables (TWh)',
        'Renewable energy share in the total final energy consumption (%)', 
        'Access to electricity (% of population)',
        'Low-carbon electricity (% electricity)', 'Latitude', 'Longitude',
        'Land Area(Km2)', 'Density\\n(P/Km2)', 
        lag1_col, 'gdp_growth_lag1', 'gdp_per_capita_lag1',
        'Primary energy consumption per capita (kWh/person)_lag1'
    ]
    
    available_features = [f for f in feature_columns if f in df.columns]
    print(f"Using {len(available_features)} features")
    
    # Train-test split
    train_df = df[df['Year'] < 2015]
    test_df = df[df['Year'] >= 2015]
    
    X_train = train_df[available_features].fillna(0)
    y_train = train_df[target_col]
    X_test = test_df[available_features].fillna(0)
    y_test = test_df[target_col]
    entities_test = test_df['Entity']
    
    # Train Ridge Regression
    print("Training Ridge Regression...")
    model = Ridge(alpha=10.0)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate per-entity MAPE
    print("Calculating per-entity MAPE...")
    entity_mape = calculate_entity_mape(y_test.values, y_pred, entities_test.values)
    
    print(f"\nStatistics:")
    print(f"  - Number of countries: {len(entity_mape)}")
    print(f"  - Median MAPE: {entity_mape.median():.2f}%")
    print(f"  - Mean MAPE: {entity_mape.mean():.2f}%")
    print(f"  - Min MAPE: {entity_mape.min():.2f}% ({entity_mape.idxmin()})")
    print(f"  - Max MAPE: {entity_mape.max():.2f}% ({entity_mape.idxmax()})")
    
    # Create histogram
    print("\nGenerating histogram...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Plot 1: Full distribution (log scale) ---
    ax1 = axes[0]
    
    # Define bins for log scale
    bins_log = [0, 5, 10, 25, 50, 100, 250, 500, 1000, 5000, 100000]
    
    # Count countries in each bin
    counts, _ = np.histogram(entity_mape.values, bins=bins_log)
    
    # Create bar positions
    bar_positions = range(len(counts))
    bar_labels = ['0-5%', '5-10%', '10-25%', '25-50%', '50-100%', 
                  '100-250%', '250-500%', '500-1K%', '1K-5K%', '>5K%']
    
    # Color bars by category
    colors = ['#2ecc71', '#2ecc71', '#3498db', '#3498db', '#f1c40f', 
              '#e67e22', '#e74c3c', '#c0392b', '#8e44ad', '#8e44ad']
    
    bars = ax1.bar(bar_positions, counts, color=colors, edgecolor='black', alpha=0.8)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(int(count)), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(bar_labels, rotation=45, ha='right')
    ax1.set_xlabel('MAPE Range')
    ax1.set_ylabel('Number of Countries')
    ax1.set_title('MAPE Distribution (All Countries)\n(Log-scale bins)', fontweight='bold')
    
    # Add median and mean lines
    median_val = entity_mape.median()
    mean_val = entity_mape.mean()
    
    # Find bin index for median
    for i, (low, high) in enumerate(zip(bins_log[:-1], bins_log[1:])):
        if low <= median_val < high:
            ax1.axvline(i, color='green', linestyle='--', linewidth=2, label=f'Median = {median_val:.1f}%')
            break
    
    # Legend for colors
    legend_patches = [
        mpatches.Patch(color='#2ecc71', label='Low MAPE (0-10%): Major Economies'),
        mpatches.Patch(color='#3498db', label='Medium (10-50%): Developing'),
        mpatches.Patch(color='#f1c40f', label='High (50-100%): Small Economies'),
        mpatches.Patch(color='#c0392b', label='Very High (>100%): Micro-states')
    ]
    ax1.legend(handles=legend_patches, loc='upper right', fontsize=9)
    
    # --- Plot 2: Focus on main distribution (exclude outliers) ---
    ax2 = axes[1]
    
    # Filter out extreme outliers (MAPE < 200%)
    mape_filtered = entity_mape[entity_mape < 200]
    
    # Create histogram
    n, bins_hist, patches = ax2.hist(mape_filtered, bins=20, color='#3498db', 
                                      edgecolor='black', alpha=0.7)
    
    # Color bars by value
    for i, (patch, bin_center) in enumerate(zip(patches, bins_hist[:-1])):
        if bin_center < 10:
            patch.set_facecolor('#2ecc71')
        elif bin_center < 50:
            patch.set_facecolor('#3498db')
        elif bin_center < 100:
            patch.set_facecolor('#f1c40f')
        else:
            patch.set_facecolor('#e67e22')
    
    # Add vertical lines for median and mean
    ax2.axvline(median_val, color='green', linestyle='--', linewidth=2.5, 
                label=f'Median = {median_val:.1f}%')
    ax2.axvline(mape_filtered.mean(), color='blue', linestyle=':', linewidth=2, 
                label=f'Mean (filtered) = {mape_filtered.mean():.1f}%')
    
    ax2.set_xlabel('MAPE (%)')
    ax2.set_ylabel('Number of Countries')
    ax2.set_title(f'MAPE Distribution (Excluding Outliers > 200%)\nN = {len(mape_filtered)} countries', 
                  fontweight='bold')
    ax2.legend(loc='upper right')
    
    # Add text annotation
    ax2.annotate(f'Full Mean = {mean_val:.0f}%\n(inflated by micro-states)',
                xy=(0.95, 0.7), xycoords='axes fraction',
                fontsize=10, ha='right',
                bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'reports/figures/mape_distribution_histogram.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nHistogram saved to: {output_path}")
    
    # Also create a summary table
    print("\n" + "="*60)
    print("MAPE DISTRIBUTION BY CATEGORY")
    print("="*60)
    
    categories = {
        'Top Emitters (0-5%)': (entity_mape < 5).sum(),
        'Excellent (5-10%)': ((entity_mape >= 5) & (entity_mape < 10)).sum(),
        'Good (10-25%)': ((entity_mape >= 10) & (entity_mape < 25)).sum(),
        'Fair (25-50%)': ((entity_mape >= 25) & (entity_mape < 50)).sum(),
        'Poor (50-100%)': ((entity_mape >= 50) & (entity_mape < 100)).sum(),
        'Micro-states (>100%)': (entity_mape >= 100).sum()
    }
    
    for cat, count in categories.items():
        print(f"  {cat}: {count} countries ({count/len(entity_mape)*100:.1f}%)")
    
    print("="*60)
    
    # Show some examples
    print("\nTop 5 Best Predictions:")
    for country, mape in entity_mape.nsmallest(5).items():
        print(f"  - {country}: {mape:.2f}%")
    
    print("\nTop 5 Worst Predictions (Micro-states):")
    for country, mape in entity_mape.nlargest(5).items():
        print(f"  - {country}: {mape:.0f}%")
    
    plt.show()
    
    return entity_mape

if __name__ == "__main__":
    main()
