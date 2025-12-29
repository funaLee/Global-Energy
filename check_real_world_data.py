from pandas_datareader import wb
import pandas as pd

print("--- Checking World Bank Data Availability (Corrected) ---")

indicators = {
    'EN.ATM.CO2E.KT': 'CO2',
    'NY.GDP.PCAP.CD': 'GDP_per_capita',
    'EG.USE.PCAP.KG.OE': 'Energy_per_capita'
}

countries = ['US', 'CN', 'VN', 'DE', 'IN'] # ISO2 codes for WB

try:
    df = wb.download(indicator=indicators.keys(), country=countries, start=2020, end=2023)
    df = df.rename(columns=indicators)
    print("\nSample Data (2020-2023):")
    print(df)
    
    # Check completeness
    print("\nMissing Values Count:")
    print(df.isna().sum())
    
except Exception as e:
    print(f"Error fetching data: {e}")
