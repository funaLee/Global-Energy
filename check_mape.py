import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

print("--- Checking MAPE for Real World Validation ---")

# Load the logic from validate_full_clean_list outputs or run snippet
# Easier: Re-run a simplified validation for 2023 and compute MAPE
# Or just load 'data/results/owid_validation_2023.csv' IS NOT SAVED?
# Ah, validate_full_clean_list didn't save CSVs? It just printed.
# I need to modify it or re-run the calculation part.

# Re-running 2023 part quickly
from pandas_datareader import wb
from sklearn.linear_model import Ridge
import sys
import os
sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

# Load & Train
df_lr_raw = load_data('data/processed/lr_final_prep.csv') 
df_common = load_data('data/processed/common_preprocessed.csv')
map_df = pd.read_csv('data/processed/recovered_index_map.csv')
df_lr = df_lr_raw.iloc[map_df['LR_Index']].copy().reset_index(drop=True)
original_indices = map_df['Original_Index'].values
df_lr['Year'] = df_common.loc[original_indices, 'Year'].values
df_lr['Entity'] = df_common.loc[original_indices, 'Entity'].values
TARGET = 'Value_co2_emissions_kt_by_country'
feature_cols = [c for c in df_lr.columns if c not in [TARGET, 'Year', 'Entity'] and df_lr[c].dtype in [np.float64, np.int64]]
X_train = df_lr[feature_cols]
y_train = df_lr[TARGET]
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

train_common = df_common.iloc[original_indices].reset_index(drop=True)
means = train_common.mean(numeric_only=True)
stds = train_common.std(numeric_only=True)
target_mean = train_common[TARGET].mean()
target_std = train_common[TARGET].std()
clean_entities = df_lr['Entity'].unique()

# Fetch 2023
wb_features = {'NY.GDP.PCAP.CD': 'gdp_per_capita', 'EG.USE.PCAP.KG.OE': 'Primary energy consumption per capita (kWh/person)'}
import pycountry
wb_iso2_map = {}
owid_iso3_map = {}
for entity in clean_entities:
    try:
        matches = pycountry.countries.search_fuzzy(entity)
        if matches:
            owid_iso3_map[entity] = matches[0].alpha_3
            wb_iso2_map[entity] = matches[0].alpha_2
    except: pass
    # Manuals
    if entity == 'Vietnam': wb_iso2_map[entity]='VN'; owid_iso3_map[entity]='VNM'
    if entity == 'South Korea': wb_iso2_map[entity]='KR'; owid_iso3_map[entity]='KOR'
    if entity == 'Russia': wb_iso2_map[entity]='RU'; owid_iso3_map[entity]='RUS'

iso2_list = list(set(wb_iso2_map.values()))
print(f"Fetching data for {len(iso2_list)} countries...")
try:
    df_wb = wb.download(indicator=wb_features.keys(), country=iso2_list, start=2023, end=2023)
    df_wb = df_wb.rename(columns=wb_features).reset_index()
except:
    print("Fetch failed")
    sys.exit(1)

url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
df_owid = pd.read_csv(url)
truth_map = dict(zip(df_owid[df_owid['year']==2023]['iso_code'], df_owid[df_owid['year']==2023]['co2']*1000))
lag_map = dict(zip(df_owid[df_owid['year']==2022]['iso_code'], df_owid[df_owid['year']==2022]['co2']*1000))

results = []
for entity in clean_entities:
    if entity not in wb_iso2_map: continue
    iso3 = owid_iso3_map.get(entity)
    if not iso3 or iso3 not in truth_map: continue
    
    wb_search = df_wb[df_wb['country'] == entity] # Fuzzy issues again?
    # Assume 1st hit for now or match iso? 
    # wb.download result doesn't have ISO.
    # We skip strict matching for quick check.
    if wb_search.empty: continue
    row = wb_search.iloc[0]
    
    actual = truth_map[iso3]
    input_vector = pd.Series(0.0, index=feature_cols)
    
    if hasattr(row, 'gdp_per_capita') and not pd.isna(row.gdp_per_capita):
        input_vector['gdp_per_capita'] = (row.gdp_per_capita - means['gdp_per_capita']) / stds['gdp_per_capita']
        
    if iso3 in lag_map:
        input_vector['Value_co2_emissions_kt_by_country_lag1'] = (lag_map[iso3] - target_mean) / target_std
        
    entity_col = f"Entity_{entity}"
    if entity_col in input_vector.index: input_vector[entity_col] = 1.0
    
    pred = model.predict([input_vector.values])[0]
    results.append({'Entity': entity, 'Actual': actual, 'Predicted': pred})

res = pd.DataFrame(results)
res['AbsError'] = abs(res['Actual'] - res['Predicted'])
res['APE'] = res['AbsError'] / res['Actual'] * 100

print(f"\nAnalyzed {len(res)} countries.")
print(f"R2: {r2_score(res['Actual'], res['Predicted']):.4f}")
print(f"MAE: {mean_absolute_error(res['Actual'], res['Predicted']):.2f}")
print(f"MAPE (Mean): {res['APE'].mean():.2f}%")
print(f"MAPE (Median): {res['APE'].median():.2f}%")

print("\nTop 5 Worst Errors (MAPE):")
print(res.sort_values('APE', ascending=False).head(5)[['Entity', 'Actual', 'Predicted', 'APE']])
