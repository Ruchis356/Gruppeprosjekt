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

# Graf 1:----------- LUFTFORURENSNING SOM PUNKTER -------------
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

# Graf 2:----------- VÆRDATA SOM PUNKTER -------------
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




import pandas as pd
import matplotlib.pyplot as plt

# Importere data
file_path_air = 'data/refined_air_qualty_data.csv'
file_path_weather = 'data/refined_weather_data.csv'

def import_for_analysis(file_path):
    try:
        df = pd.read_csv(file_path, sep=',', encoding='utf-8')
        return df
    except FileNotFoundError:
        print(f"Error: Filen '{file_path}' ble ikke funnet.")
        return None

df_air = import_for_analysis(file_path_air)
df_weather = import_for_analysis(file_path_weather)

# Konvertere dato-kolonner til datetime
if df_air is not None:
    df_air['Date'] = pd.to_datetime(df_air['Date'])

if df_weather is not None:
    df_weather['Date'] = pd.to_datetime(df_weather['Date'])

# ----------- Graf 3:: NO, NO2, NOx, PM2.5 og PM10 med Temperatur -------------
if df_air is not None and df_weather is not None:
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Venstre y-akse for drivhusgasser (NO, NO2, NOx)
    ax1.plot(df_air['Date'], df_air['NO'], label='NO', color='purple', lw=2)
    ax1.plot(df_air['Date'], df_air['NO2'], label='NO2', color='green', lw=2)
    ax1.plot(df_air['Date'], df_air['NOx'], label='NOx', color='red', lw=2)
    ax1.plot(df_air['Date'], df_air['PM2.5'], label='PM2.5', color='yellow', lw=2)
    ax1.plot(df_air['Date'], df_air['PM10'], label='PM10', color='orange', lw=2)

    ax1.set_xlabel('Dato')
    ax1.set_ylabel('Luftforurensning (µg/m³)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    ax1.legend(loc='upper left')

    # Sekundær y-akse for temperatur
    ax2 = ax1.twinx()
    ax2.plot(df_weather['Date'], df_weather['temperature (C)'], label='Temperatur (°C)', color='blue', lw=2)
    
    # Sett y-lim for temperatur
    ax2.set_ylim(-30, 40)  

    ax2.set_ylabel('Temperatur (°C)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    ax2.legend(loc='upper right')

    plt.title('Luftforurensning (NO, NO2, NOx,PM2.5,PM10) og Temperatur')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# -----------  GRAF 4: PM10, PM2.5, NO, NOx, NO2 med Vindhastighet -------------
if df_air is not None and df_weather is not None:
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Venstre y-akse for drivhusgasser (PM10, PM2.5)
    ax1.plot(df_air['Date'], df_air['PM10'], label='PM10', color='purple', lw=2)
    ax1.plot(df_air['Date'], df_air['PM2.5'], label='PM2.5', color='orange', lw=2)
    ax1.plot(df_air['Date'], df_air['NO'], label='NO', color='blue', lw=2)
    ax1.plot(df_air['Date'], df_air['NOx'], label='NOx', color='yellow', lw=2)
    ax1.plot(df_air['Date'], df_air['NO2'], label='NO2', color='red', lw=2)

    ax1.set_xlabel('Dato')
    ax1.set_ylabel('Luftforurensning (µg/m³)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    ax1.legend(loc='upper left')

    # Sekundær y-akse for vindhastighet
    ax2 = ax1.twinx()
    ax2.plot(df_weather['Date'], df_weather['wind_speed (m/s)'], label='Vindhastighet (m/s)', color='green', lw=2)
    
    # Sett y-lim for vindhastighet
    ax2.set_ylim(0, 10)  

    ax2.set_ylabel('Vindhastighet (m/s)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    ax2.legend(loc='upper right')

    plt.title('Luftforurensning (PM10, PM2.5) og Vindhastighet')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ----------- GRAF 5 NO, NO2, NOx, PM2.5,PM10 med Nedbør -------------
if df_air is not None and df_weather is not None:
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Venstre y-akse for alle drivhusgasser
    ax1.plot(df_air['Date'], df_air['NO'], label='NO', color='blue', lw=2)
    ax1.plot(df_air['Date'], df_air['NO2'], label='NO2', color='green', lw=2)
    ax1.plot(df_air['Date'], df_air['NOx'], label='NOx', color='red', lw=2)
    ax1.plot(df_air['Date'], df_air['PM10'], label='PM10', color='purple', lw=2)
    ax1.plot(df_air['Date'], df_air['PM2.5'], label='PM2.5', color='orange', lw=2)

    ax1.set_xlabel('Dato')
    ax1.set_ylabel('Luftforurensning (µg/m³)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    ax1.legend(loc='upper left')

    # Sekundær y-akse for nedbør
    ax2 = ax1.twinx()
    ax2.plot(df_weather['Date'], df_weather['precipitation (mm)'], label='Nedbør (mm)', color='black', lw=2)

    # Sett y-lim for nedbør
    ax2.set_ylim(0, 100)  

    ax2.set_ylabel('Nedbør (mm)', color='black')
    ax2.tick_params(axis='y', labelcolor='red')

    ax2.legend(loc='upper right')

    plt.title('Alle drivhusgasser og Nedbør')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
