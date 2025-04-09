import sys, os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', 'src'))) # This construct was reworked with the assistance of AI (DeepSeek) 



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importere data
file_path_air = 'data/refined_air_qualty_data.csv'
file_path_weather = 'data/refined_weather_data.csv'

def import_for_analysis(file_path):
    try:
        df = pd.read_csv(file_path, sep=',', encoding='utf-8')
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None

df_air = import_for_analysis(file_path_air)
df_weather = import_for_analysis(file_path_weather)

# Konverter tid-kolonner til datetime
if df_air is not None:
    df_air.rename(columns={'Tid': 'Date'}, inplace=True)
    df_air['Date'] = pd.to_datetime(df_air['Date'])

if df_weather is not None:
    df_weather.rename(columns={'referenceTime': 'Date'}, inplace=True)
    df_weather['Date'] = pd.to_datetime(df_weather['Date'])
    print(df_weather.info())  # Sjekk om Date er datetime
    print(df_weather.head())  # Sjekk kolonnenavn og verdier

# Plot luftforurensning over tid
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_air, x='Date', y='Elgeseter NO2 µg/m³ Day', label='NO2')
sns.scatterplot(data=df_air, x='Date', y='Elgeseter PM10 µg/m³ Day', label='PM10')
sns.scatterplot(data=df_air, x='Date', y='Elgeseter PM2.5 µg/m³ Day', label='PM2.5')
plt.title('Luftforurensning over tid')
plt.xlabel('Dato')
plt.ylabel('Konsentrasjon (µg/m³)')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Filtrer data for temperatur, nedbør og vindhastighet
weather_temp = df_weather[df_weather['unit'] == 'degC']
weather_precip = df_weather[df_weather['unit'] == 'mm']
weather_wind = df_weather[df_weather['unit'] == 'm/s']

# Plot temperatur, nedbør og vindhastighet over tid
plt.figure(figsize=(12, 6))
sns.scatterplot(data=weather_temp, x='Date', y='value', label='Temperatur (°C)')
sns.scatterplot(data=weather_precip, x='Date', y='value', label='Nedbør (mm)')
sns.scatterplot(data=weather_wind, x='Date', y='value', label='Vindhastighet (m/s)')
plt.title('Værdata over tid')
plt.xlabel('Dato')
plt.ylabel('Målinger')
plt.legend()
plt.xticks(rotation=45)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importere data
file_path_air = 'data/refined_air_qualty_data.csv'
file_path_weather = 'data/refined_weather_data.csv'

def import_for_analysis(file_path):
    try:
        df = pd.read_csv(file_path, sep=',', encoding='utf-8')
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None

df_air = import_for_analysis(file_path_air)
df_weather = import_for_analysis(file_path_weather)

# Konverter tid-kolonner til datetime
if df_air is not None:
    df_air.rename(columns={'Tid': 'Date'}, inplace=True)
    df_air['Date'] = pd.to_datetime(df_air['Date'])

if df_weather is not None:
    df_weather.rename(columns={'referenceTime': 'Date'}, inplace=True)
    df_weather['Date'] = pd.to_datetime(df_weather['Date'])
    print(df_weather.info())  # Sjekk om Date er datetime
    print(df_weather.head())  # Sjekk kolonnenavn og verdier

# Plot luftforurensning over tid
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_air, x='Date', y='Elgeseter NO2 µg/m³ Day', label='NO2')
sns.scatterplot(data=df_air, x='Date', y='Elgeseter PM10 µg/m³ Day', label='PM10')
sns.scatterplot(data=df_air, x='Date', y='Elgeseter PM2.5 µg/m³ Day', label='PM2.5')
plt.title('Luftforurensning over tid')
plt.xlabel('Dato')
plt.ylabel('Konsentrasjon (µg/m³)')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Filtrer data for temperatur, nedbør og vindhastighet
weather_temp = df_weather[df_weather['unit'] == 'degC']
weather_precip = df_weather[df_weather['unit'] == 'mm']
weather_wind = df_weather[df_weather['unit'] == 'm/s']

# Plot temperatur, nedbør og vindhastighet over tid
plt.figure(figsize=(12, 6))
sns.scatterplot(data=weather_temp, x='Date', y='value', label='Temperatur (°C)')
sns.scatterplot(data=weather_precip, x='Date', y='value', label='Nedbør (mm)')
sns.scatterplot(data=weather_wind, x='Date', y='value', label='Vindhastighet (m/s)')
plt.title('Værdata over tid')
plt.xlabel('Dato')
plt.ylabel('Målinger')
plt.legend()
plt.xticks(rotation=45)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importere data
file_path_air = 'data/refined_air_qualty_data.csv'
file_path_weather = 'data/refined_weather_data.csv'

def import_for_analysis(file_path):
    try:
        df = pd.read_csv(file_path, sep=',', encoding='utf-8')
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None

df_air = import_for_analysis(file_path_air)
df_weather = import_for_analysis(file_path_weather)

# Konverter tid-kolonner til datetime
if df_air is not None:
    df_air.rename(columns={'Tid': 'Date'}, inplace=True)
    df_air['Date'] = pd.to_datetime(df_air['Date'])

if df_weather is not None:
    df_weather.rename(columns={'referenceTime': 'Date'}, inplace=True)
    df_weather['Date'] = pd.to_datetime(df_weather['Date'])
    print(df_weather.info())  # Sjekk om Date er datetime
    print(df_weather.head())  # Sjekk kolonnenavn og verdier

# Filtrer data for temperatur, nedbør og vindhastighet
weather_temp = df_weather[df_weather['unit'] == 'degC']
weather_precip = df_weather[df_weather['unit'] == 'mm']
weather_wind = df_weather[df_weather['unit'] == 'm/s']

# Plot temperatur, nedbør og vindhastighet over tid
plt.figure(figsize=(12, 6))
sns.scatterplot(data=weather_temp, x='Date', y='value', label='Temperatur (°C)')
sns.scatterplot(data=weather_precip, x='Date', y='value', label='Nedbør (mm)')
sns.scatterplot(data=weather_wind, x='Date', y='value', label='Vindhastighet (m/s)')
plt.title('Værdata over tid')
plt.xlabel('Dato')
plt.ylabel('Målinger')
plt.legend()
plt.xticks(rotation=45)
plt.show()









