import pandas as pd
import numpy as np
from pandas_datareader import wb
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import sys
import os

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

print("--- Real-World Validation [2020-2023] ---")

# 1. Load Training Data
df_lr_raw = load_data('data/processed/lr_final_prep.csv') 
df_common = load_data('data/processed/common_preprocessed.csv')
map_df = pd.read_csv('data/processed/recovered_index_map.csv')

df_lr = df_lr_raw.iloc[map_df['LR_Index']].copy().reset_index(drop=True)
original_indices = map_df['Original_Index'].values
df_lr['Year'] = df_common.loc[original_indices, 'Year'].values
df_lr['Entity'] = df_common.loc[original_indices, 'Entity'].values

TARGET = 'Value_co2_emissions_kt_by_country'
feature_cols = [c for c in df_lr.columns if c not in [TARGET, 'Year', 'Entity'] and df_lr[c].dtype in [np.float64, np.int64]]

# Train (2000-2019)
X_train = df_lr[feature_cols]
y_train = df_lr[TARGET]
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# 2. Reconstruct Scalers
train_common = df_common.iloc[original_indices].reset_index(drop=True) # All years available < 2020
means = train_common.mean(numeric_only=True)
stds = train_common.std(numeric_only=True)
target_mean = train_common[TARGET].mean()
target_std = train_common[TARGET].std()

# 3. Validation Years
validation_years = [2020, 2021, 2022, 2023]
wb_features = {
    'NY.GDP.PCAP.CD': 'gdp_per_capita',
    'EG.USE.PCAP.KG.OE': 'Primary energy consumption per capita (kWh/person)',
}
wb_iso2 = ['US', 'CN', 'IN', 'JP', 'DE', 'GB', 'FR', 'BR', 'CA', 'IT']
owid_iso3 = ['USA', 'CHN', 'IND', 'JPN', 'DEU', 'GBR', 'FRA', 'BRA', 'CAN', 'ITA']
wb_map = dict(zip(['United States', 'China', 'India', 'Japan', 'Germany', 'United Kingdom', 'France', 'Brazil', 'Canada', 'Italy'], owid_iso3))

# Fetch Inputs
print(f"Fetching WB Data for {validation_years}...")
try:
    df_wb = wb.download(indicator=wb_features.keys(), country=wb_iso2, start=2020, end=2023)
    df_wb = df_wb.rename(columns=wb_features)
    df_wb = df_wb.sort_index()
except Exception as e:
    print(f"WB Fetch Failed: {e}")
    sys.exit(1)

df_wb = df_wb.reset_index()
df_wb['Year'] = df_wb['year'].astype(int)
if 'Primary energy consumption per capita (kWh/person)' in df_wb.columns:
    df_wb['Primary energy consumption per capita (kWh/person)'] *= 11.63 

# Fetch Ground Truth
print("Fetching OWID Ground Truth...")
url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
df_owid = pd.read_csv(url)
df_all_lags = df_owid[df_owid['iso_code'].isin(owid_iso3)][['iso_code', 'year', 'co2']]

metrics = []

for year in validation_years:
    print(f"\n--- Validating Year {year} ---")
    results = []
    
    # Lags (Prev Year)
    prev_year = year - 1
    lags = df_all_lags[df_all_lags['year'] == prev_year]
    lag_map = dict(zip(lags['iso_code'], lags['co2'] * 1000))
    
    # Truth (Current Year)
    truth = df_all_lags[df_all_lags['year'] == year]
    truth_map = dict(zip(truth['iso_code'], truth['co2'] * 1000))
    
    df_wb_year = df_wb[df_wb['Year'] == year].copy()
    
    # Handle Missing Energy in 2023 (Forward Fill from 2022)
    # Simple strategy: If Energy is NaN, try to look up 2022 energy for same country
    
    for row in df_wb_year.itertuples():
        country_name = row.country
        # Map ISO
        if country_name == 'United States': iso3 = 'USA'
        elif country_name == 'China': iso3 = 'CHN'
        elif country_name == 'India': iso3 = 'IND'
        elif country_name == 'Japan': iso3 = 'JPN'
        elif country_name == 'Germany': iso3 = 'DEU'
        elif country_name in wb_map: iso3 = wb_map[country_name]
        else: continue
            
        if iso3 not in truth_map: continue
        actual = truth_map[iso3]
        
        input_vector = pd.Series(0.0, index=feature_cols)
        
        # GDP
        if hasattr(row, 'gdp_per_capita') and not pd.isna(row.gdp_per_capita):
            raw = row.gdp_per_capita
            if 'gdp_per_capita' in means:
                input_vector['gdp_per_capita'] = (raw - means['gdp_per_capita']) / stds['gdp_per_capita']
        
        # Energy (Check NaN)
        feat_energy = 'Primary energy consumption per capita (kWh/person)'
        energy_val = getattr(row, '_3', np.nan) # _3 index might vary based on columns order? 
        # Safer: use dict access or named tuple field if known. 
        # Pandas namedtuple fields replace spaces.
        # Let's use direct access via index or column name logic
        
        if feat_energy in df_wb_year.columns:
            energy_val = df_wb_year.loc[row.Index, feat_energy] # Careful with Index
            
            # IMPUTATION: If NaN, check 2022
            if pd.isna(energy_val) and year == 2023:
                 # Find 2022 row for this country
                 val_2022 = df_wb[(df_wb['country'] == country_name) & (df_wb['Year'] == 2022)][feat_energy].values
                 if len(val_2022) > 0:
                     energy_val = val_2022[0]
            
            if not pd.isna(energy_val) and feat_energy in means:
                input_vector[feat_energy] = (energy_val - means[feat_energy]) / stds[feat_energy]

        # Lag
        if iso3 in lag_map:
            raw_lag = lag_map[iso3]
            feat = 'Value_co2_emissions_kt_by_country_lag1'
            input_vector[feat] = (raw_lag - target_mean) / target_std
            
        # Entity
        entity_col = f"Entity_{country_name}"
        if entity_col in input_vector.index:
             input_vector[entity_col] = 1.0

        pred = model.predict([input_vector.values])[0]
        results.append({'ISO': iso3, 'Actual': actual, 'Predicted': pred})

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        r2 = r2_score(res_df['Actual'], res_df['Predicted'])
        print(f"Year {year} R2: {r2:.4f}")
        metrics.append({'Year': year, 'R2': r2})

print(pd.DataFrame(metrics))
