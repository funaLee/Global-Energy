import pandas as pd

url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
print("Checking OWID Years...")
try:
    df = pd.read_csv(url)
    max_year = df['year'].max()
    print(f"Max Year in OWID Data: {max_year}")
    
    recent_years = df[df['year'] >= 2020]['year'].value_counts()
    print("\nRecord Counts per Year:")
    print(recent_years)
    
    # Check specific countries for 2022/2023
    countries = ['USA', 'CHN', 'VNM']
    print("\nData for USA/CHN/VNM (2022-2023):")
    print(df[df['iso_code'].isin(countries) & (df['year'] >= 2022)][['iso_code', 'year', 'co2']])
    
except Exception as e:
    print(e)
