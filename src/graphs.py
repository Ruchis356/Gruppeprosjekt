import sys, os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', 'src'))) # This construct was reworked with the assistance of AI (DeepSeek) 


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Filstier til CSV-er
file_path_air = 'data/refined_air_qualty_data.csv'
file_path_weather = 'data/refined_weather_data.csv'

# Funksjon for å importere CSV
def import_for_analysis(file_path):
    try:
        df = pd.read_csv(file_path, sep=',', encoding='utf-8')
        return df
    except FileNotFoundError:
        print(f"Error: Filen '{file_path}' ble ikke funnet.")
        return None

# Importer data
df_air = import_for_analysis(file_path_air)
df_weather = import_for_analysis(file_path_weather)

# Konverter dato-kolonner
if df_air is not None:
    df_air['Date'] = pd.to_datetime(df_air['Date'])

if df_weather is not None:
    df_weather['Date'] = pd.to_datetime(df_weather['Date'])

# ----------- LUFTFORURENSNING SOM PUNKTER -------------
if df_air is not None:
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df_air, x='Date', y='NO', label='NO')
    sns.scatterplot(data=df_air, x='Date', y='NO2', label='NO2')
    sns.scatterplot(data=df_air, x='Date', y='NOx', label='NOx')
    sns.scatterplot(data=df_air, x='Date', y='PM10', label='PM10')
    sns.scatterplot(data=df_air, x='Date', y='PM2.5', label='PM2.5')
    plt.title('Luftforurensning over tid (punktvis)')
    plt.xlabel('Dato')
    plt.ylabel('Konsentrasjon (µg/m³)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ----------- VÆRDATA SOM PUNKTER -------------
if df_weather is not None:
    plt.figure(figsize=(12, 6))
    if 'temperature (C)' in df_weather.columns:
        sns.scatterplot(data=df_weather, x='Date', y='temperature (C)', label='Temperatur (°C)')
    if 'precipitation (mm)' in df_weather.columns:
        sns.scatterplot(data=df_weather, x='Date', y='precipitation (mm)', label='Nedbør (mm)')
    if 'wind_speed (m/s)' in df_weather.columns:
        sns.scatterplot(data=df_weather, x='Date', y='wind_speed (m/s)', label='Vindhastighet (m/s)')
    plt.title('Værdata over tid (punktvis)')
    plt.xlabel('Dato')
    plt.ylabel('Målinger')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()









