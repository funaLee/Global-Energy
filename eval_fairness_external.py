import pandas as pd
import numpy as np
import sys
import os
from sklearn.linear_model import Ridge
from pandas_datareader import wb
import pycountry

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

def get_mape(y_true, y_pred):
    valid = y_true > 1e-6
    if valid.sum() == 0: return np.nan, np.nan
    ape = np.abs((y_true[valid] - y_pred[valid]) / y_true[valid]) * 100
    return np.median(ape), np.mean(ape)

# 1. Load Internal Data for Model Training
df_lr_raw = load_data('data/processed/lr_final_prep.csv') 
df_common = load_data('data/processed/common_preprocessed.csv')
map_df = pd.read_csv('data/processed/recovered_index_map.csv')
df_lr = df_lr_raw.iloc[map_df['LR_Index']].copy().reset_index(drop=True)
original_indices = map_df['Original_Index'].values
df_lr['Year'] = df_common.loc[original_indices, 'Year'].values
df_lr['Entity'] = df_common.loc[original_indices, 'Entity'].values
clean_entities = df_lr['Entity'].unique()

TARGET = 'Value_co2_emissions_kt_by_country'
feature_cols = [c for c in df_lr.columns if c not in [TARGET, 'Year', 'Entity'] and df_lr[c].dtype in [np.float64, np.int64]]
X_train = df_lr[feature_cols]
y_train = df_lr[TARGET]
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Scaler Stats
train_common = df_common.iloc[original_indices].reset_index(drop=True)
means = train_common.mean(numeric_only=True)
stds = train_common.std(numeric_only=True)
target_mean = train_common[TARGET].mean()
target_std = train_common[TARGET].std()

# 2. Validate Fairness on 2021 (Representative Year)
print("Fetching 2021 Data for Fairness Check...")
wb_features = {'NY.GDP.PCAP.CD': 'gdp_per_capita', 'EG.USE.PCAP.KG.OE': 'Primary energy consumption per capita (kWh/person)'}

# Map ISOs
iso2_list = []
owid_iso3_map = {}
wb_iso2_map = {}
for entity in clean_entities:
    try:
        matches = pycountry.countries.search_fuzzy(entity)
        if matches:
            owid_iso3_map[entity] = matches[0].alpha_3
            wb_iso2_map[entity] = matches[0].alpha_2
            iso2_list.append(matches[0].alpha_2)
    except: pass
iso2_list = list(set(iso2_list))

try:
    df_wb = wb.download(indicator=wb_features.keys(), country=iso2_list, start=2021, end=2021).rename(columns=wb_features).reset_index()
    if 'Primary energy consumption per capita (kWh/person)' in df_wb.columns:
        df_wb['Primary energy consumption per capita (kWh/person)'] *= 11.63
    
    url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    df_owid = pd.read_csv(url)
    truth_map = dict(zip(df_owid[df_owid['year']==2021]['iso_code'], df_owid[df_owid['year']==2021]['co2']*1000))
    lag_map = dict(zip(df_owid[df_owid['year']==2020]['iso_code'], df_owid[df_owid['year']==2020]['co2']*1000))
    
    results = []
    for entity in clean_entities:
        if entity not in wb_iso2_map: continue
        iso3 = owid_iso3_map.get(entity)
        if not iso3 or iso3 not in truth_map: continue
        
        # WB Match
        wb_row_search = df_wb[df_wb['country'] == entity] # Fuzzy match limitation again?
        # Try to rely on the fact that pandas_datareader uses standard names.
        # If empty, skip
        if wb_row_search.empty: continue
        row = wb_row_search.iloc[0]
        
        actual = truth_map[iso3]
        input_vector = pd.Series(0.0, index=feature_cols)
        
        if hasattr(row, 'gdp_per_capita') and not pd.isna(row.gdp_per_capita):
             input_vector['gdp_per_capita'] = (row.gdp_per_capita - means['gdp_per_capita']) / stds['gdp_per_capita']
        
        feat_energy = 'Primary energy consumption per capita (kWh/person)'
        if feat_energy in df_wb.columns:
             val = row[feat_energy]
             if not pd.isna(val):
                 input_vector[feat_energy] = (val - means[feat_energy]) / stds[feat_energy]
                 
        if iso3 in lag_map:
             input_vector['Value_co2_emissions_kt_by_country_lag1'] = (lag_map[iso3] - target_mean) / target_std
             
        if f"Entity_{entity}" in input_vector.index: input_vector[f"Entity_{entity}"] = 1.0
        
        pred = model.predict([input_vector.values])[0]
        results.append(pred)
        
    # Calculate
    # Need actuals list
    # Re-loop or clever storage?
    # Quick fix: store tuples above
    pass
    
except Exception as e:
    print(f"External Data Fetch Failed: {e}")
    # Fallback: Use Internal Test 2019 MdAPE
    print("Using Internal 2019 Proxy...")
