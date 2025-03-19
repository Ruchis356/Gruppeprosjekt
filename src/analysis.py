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





