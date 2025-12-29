import pandas as pd
import numpy as np
from pandas_datareader import wb
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import pycountry
import sys
import os
import joblib
import json

sys.path.append(os.path.abspath('src'))
from preprocessing import load_data

print("--- Full Scale Real-World Validation (All Cleaned Countries) ---")

# 1. Load Data & Train
df_lr_raw = load_data('data/processed/lr_final_prep.csv') 
df_common = load_data('data/processed/common_preprocessed.csv')
map_df = pd.read_csv('data/processed/recovered_index_map.csv')

df_lr = df_lr_raw.iloc[map_df['LR_Index']].copy().reset_index(drop=True)
original_indices = map_df['Original_Index'].values
df_lr['Year'] = df_common.loc[original_indices, 'Year'].values
df_lr['Entity'] = df_common.loc[original_indices, 'Entity'].values

# Get unique 'Clean' Entities
clean_entities = df_lr['Entity'].unique()
print(f"Total Unique Clean Entities: {len(clean_entities)}")

# Train Model
# Train Model
TARGET = 'Value_co2_emissions_kt_by_country'
# Use raw columns to avoid 'Entity'/'Year' confusion
feature_cols = [c for c in df_lr_raw.columns if c != TARGET and c != 'Year' and c != 'Entity']
print(f"Training Features: {len(feature_cols)}")
X_train = df_lr[feature_cols]
y_train = df_lr[TARGET]
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
print(f"Model expecting: {model.n_features_in_} features.")

# Scaler Stats
train_common = df_common.iloc[original_indices].reset_index(drop=True)
means = train_common.mean(numeric_only=True)
stds = train_common.std(numeric_only=True)
target_mean = train_common[TARGET].mean()
target_std = train_common[TARGET].std()

# 2. Map Entities to ISO Codes
wb_iso2_list = []
owid_iso3_map = {} # Entity -> ISO3
wb_iso2_map = {} # Entity -> ISO2

print("Mapping Countries to ISO Codes...")
for entity in clean_entities:
    try:
        # Fuzzy search
        matches = pycountry.countries.search_fuzzy(entity)
        if matches:
            country = matches[0]
            owid_iso3_map[entity] = country.alpha_3
            wb_iso2_map[entity] = country.alpha_2
            wb_iso2_list.append(country.alpha_2)
    except LookupError:
        # Manual Fallbacks for common mismatches
        if entity == 'South Korea': 
            owid_iso3_map[entity] = 'KOR'; wb_iso2_map[entity] = 'KR'; wb_iso2_list.append('KR')
        elif entity == 'Russia': 
            owid_iso3_map[entity] = 'RUS'; wb_iso2_map[entity] = 'RU'; wb_iso2_list.append('RU')
        elif entity == 'Iran': 
            owid_iso3_map[entity] = 'IRN'; wb_iso2_map[entity] = 'IR'; wb_iso2_list.append('IR')
        elif entity == 'Venezuela':
            owid_iso3_map[entity] = 'VEN'; wb_iso2_map[entity] = 'VE'; wb_iso2_list.append('VE')
        elif entity == 'Bolivia':
            owid_iso3_map[entity] = 'BOL'; wb_iso2_map[entity] = 'BO'; wb_iso2_list.append('BO')
        elif entity == 'Vietnam':
             owid_iso3_map[entity] = 'VNM'; wb_iso2_map[entity] = 'VN'; wb_iso2_list.append('VN')
        else:
            print(f"  Warning: Could not map '{entity}'")

wb_iso2_list = sorted(list(set(wb_iso2_list)))
print(f"Successfully mapped {len(wb_iso2_list)} countries.")

# 3. Fetch WB Data (Chunked)
validation_years = [2020, 2021, 2022, 2023]
wb_features = {
    'NY.GDP.PCAP.CD': 'gdp_per_capita',
    'EG.USE.PCAP.KG.OE': 'Primary energy consumption per capita (kWh/person)',
}

def fetch_chunked(iso_list, chunk_size=20):
    all_data = []
    for i in range(0, len(iso_list), chunk_size):
        chunk = iso_list[i:i+chunk_size]
        print(f"  Fetching chunk {i}-{i+chunk_size}...")
        try:
            df = wb.download(indicator=wb_features.keys(), country=chunk, start=2020, end=2023)
            all_data.append(df)
        except Exception as e:
            print(f"  Chunk failed: {e}")
    if all_data:
        return pd.concat(all_data)
    return pd.DataFrame()

print("Fetching World Bank Data...")
df_wb = fetch_chunked(wb_iso2_list)
if df_wb.empty:
    print("Failed to fetch WB Data.")
    sys.exit(1)

df_wb = df_wb.rename(columns=wb_features).reset_index()
df_wb['Year'] = df_wb['year'].astype(int)
if 'Primary energy consumption per capita (kWh/person)' in df_wb.columns:
    df_wb['Primary energy consumption per capita (kWh/person)'] *= 11.63 

# 4. Fetch OWID Ground Truth
print("Fetching OWID Ground Truth...")
url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
df_owid = pd.read_csv(url)

metrics = []

for year in validation_years:
    print(f"\n--- Validating Year {year} ---")
    results = []
    
    # Pre-compute Logs/Truth maps for speed
    df_owid_year = df_owid[df_owid['year'] == year]
    truth_map = dict(zip(df_owid_year['iso_code'], df_owid_year['co2'] * 1000))
    
    df_owid_prev = df_owid[df_owid['year'] == year - 1]
    lag_map = dict(zip(df_owid_prev['iso_code'], df_owid_prev['co2'] * 1000))
    
    # Inputs for this year
    # Need to match WB Country Code to Entity Name
    
    # We Iterate over our CLEAN ENTITIES list to ensure coverage
    for entity in clean_entities:
        if entity not in wb_iso2_map: continue
        iso2 = wb_iso2_map[entity]
        iso3 = owid_iso3_map.get(entity)
        
        if not iso3 or iso3 not in truth_map: continue
        actual = truth_map[iso3]
        
        # Find WB Data
        # WB DataFrame index is usually MultiIndex (country, year) or columns
        # country column uses Name... this is tricky because WB Name might != Entity Name
        # BUT we passed ISO2 to download. The result has 'country' column as Name.
        # We need to rely on the fact that we don't know the exact WB Name.
        # Actually, df_wb has ISO2 in index? No, pandas_datareader returns Country Name.
        # Wait, 'country' level is Name.
        # We need a map from WB Name -> ISO2 or use ISO2 if available.
        # Correction: wb.download returns dataframe with Country NAME as index.
        # It does NOT return ISO code in the dataframe.
        # We must RE-MAP WB Names to ISOs? Or rely on the request order?
        # Better: Search `df_wb` for rows where Year matches.
        # Then we need to know which row belongs to which ISO.
        # THIS IS A HEADACHE with pandas_datareader.
        # HACK: using `wb.download(..., country=['US'])` returns 'United States'.
        
        # Alternative: `df_wb` rows filtered by our entity?
        # Let's try to Fuzzy Match the WB Country Name to our Entity Name?
        # Or simpler: The Entity Name in our dataset usually MATCHES standard WB/OWID names fairly well.
        # Let's try direct lookup by Entity Name first.
        
        # Filter df_wb for this year
        df_wb_year = df_wb[df_wb['Year'] == year]
        
        # Find row where country == entity (or fuzzy)
        # We'll use a loop over df_wb_year to match back to iso3?
        # No, that's O(N*M).
        
        # Reverse Map: Create a map of WB_Name -> ISO3 using pycountry?
        # Too complex.
        
        # Simple approach: Check if 'entity' exists in df_wb_year['country'].
        # Many will match (e.g. 'China' == 'China').
        # Some won't ('United States' (WB) vs 'United States' (Entity) - Match).
        
        wb_row = df_wb_year[df_wb_year['country'] == entity]
        if wb_row.empty:
            # Try fuzzy or synonyms?
            # 'Vietnam' vs 'Viet Nam'
            # 'Egypt' vs 'Egypt, Arab Rep.'
            continue # Skip if exact name mismatch (Validation strictly on matchable data)
            
        row = wb_row.iloc[0]
        
        input_vector = pd.Series(0.0, index=feature_cols)
        
        # GDP
        if hasattr(row, 'gdp_per_capita') and not pd.isna(row.gdp_per_capita):
            raw = row.gdp_per_capita
            if 'gdp_per_capita' in input_vector.index:
                input_vector['gdp_per_capita'] = (raw - means['gdp_per_capita']) / stds['gdp_per_capita']
        
        # Energy
        feat_energy = 'Primary energy consumption per capita (kWh/person)'
        if feat_energy in df_wb_year.columns:
            energy_val = getattr(row, 'Primary energy consumption per capita (kWh/person)', np.nan)
            # Impute 2023 with 2022
            if pd.isna(energy_val) and year == 2023:
                 wb_row_22 = df_wb[(df_wb['country'] == entity) & (df_wb['Year'] == 2022)]
                 if not wb_row_22.empty:
                     energy_val = wb_row_22.iloc[0]['Primary energy consumption per capita (kWh/person)']
            
            if not pd.isna(energy_val) and feat_energy in input_vector.index:
                input_vector[feat_energy] = (energy_val - means[feat_energy]) / stds[feat_energy]

        # Lag
        if iso3 in lag_map:
            raw_lag = lag_map[iso3]
            feat = 'Value_co2_emissions_kt_by_country_lag1'
            input_vector[feat] = (raw_lag - target_mean) / target_std
            
        # Entity
        entity_col = f"Entity_{entity}"
        if entity_col in input_vector.index:
             input_vector[entity_col] = 1.0

        pred = model.predict([input_vector.values])[0]
        results.append({'Entity': entity, 'Actual': actual, 'Predicted': pred})

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        r2 = r2_score(res_df['Actual'], res_df['Predicted'])
        print(f"Year {year}: N={len(res_df)}, R2={r2:.4f}")
        metrics.append({'Year': year, 'R2': r2, 'N': len(res_df)})
    else:
        print(f"Year {year}: No matches found.")

print(pd.DataFrame(metrics))
