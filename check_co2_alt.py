from pandas_datareader import wb
import pandas as pd

print("--- Checking Alternative CO2 Indicators ---")

# Try per capita or GHG
indicators = {
    'EN.ATM.CO2E.PC': 'CO2_per_capita',
    'EN.ATM.GHGT.KT.CE': 'Total_GHG'
}

countries = ['US', 'CN', 'VN'] 

try:
    df = wb.download(indicator=indicators.keys(), country=countries, start=2020, end=2022)
    print(df)
except Exception as e:
    print(e)
