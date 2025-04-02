import sys, os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', 'src'))) # This construct was reworked with the assistance of AI (DeepSeek) 

def import_for_analysis(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            df = pd.read_csv(
                file,
                skiprows=0,  
                sep=',',    
            )
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Check the file path.")
        return None

# Importing air quality data
file_path = 'data/refined_air_qualty_data.csv'
df_air = import_for_analysis(file_path)

# Importing weather data
file_path = 'data/refined_weather_data.csv'
df_weather = import_for_analysis(file_path)

#Print head (just to check that it's working)
print(df_air.info())
print(df_weather.info())

import pandas as pd

# Function to read CSV file
def import_for_analysis(file_path):
    try:
        df = pd.read_csv(file_path)  # Load file
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None

# Function to calculate the average of "Elgeseter NO µg/m³ Day" column for every 7-row group
def calculate_avg_every_7th_row(df):
    # Check if 'Elgeseter NO µg/m³ Day' column exists
    if 'Elgeseter NO µg/m³ Day' not in df.columns:
        print("Error: 'Elgeseter NO µg/m³ Day' column not found in DataFrame.")
        return None
    
    # List to store weekly averages
    weekly_averages = []

    # Loop through the DataFrame in chunks of 7 rows
    for start in range(0, len(df), 7):
        df_week = df['Elgeseter NO µg/m³ Day'].iloc[start:start+7]  # Select "Elgeseter NO µg/m³ Day" column from 7 rows
        df_clean = df_week.dropna()  # Drop NaN Elgeseter NO µg/m³ Day
        avg_value = df_clean.mean()  # Compute the average
        weekly_averages.append(round(avg_value, 2))

    return weekly_averages  # Returns a list of weekly averages

# Load air quality data
df_air = import_for_analysis('data/refined_air_qualty_data.csv')

# Load weather data
df_weather = import_for_analysis('data/refined_weather_data.csv')

# Calculate and print results
print("Weekly averages for air quality data (from 'Elgeseter NO µg/m³ Day' column ):")
if df_air is not None:
    print(calculate_avg_every_7th_row(df_air))
