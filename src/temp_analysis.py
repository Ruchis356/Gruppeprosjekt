import sys, os
import pandas as pd
#Import the class VisualTable
from utils import VisualTable
pretty_table = VisualTable()

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



#Code to see if one can print the average of all pollutants


# Function to calculate the average of a given column for every 7-row group
def calculate_avg_every_7th_row(df, column_name):
    if column_name not in df.columns:
        print(f"Error: '{column_name}' column not found in DataFrame.")
        return []

    weekly_averages = []

    for start in range(0, len(df), 7):
        df_week = df[column_name].iloc[start:start+7]
        df_clean = df_week.dropna()
        avg_value = df_clean.mean()
        weekly_averages.append(round(avg_value, 2))

    return weekly_averages

# Function to build a DataFrame table of weekly averages for multiple pollutants
def build_weekly_avg_table(df, columns):
    weekly_data = {}

    for pollutant in columns:
        weekly_data[pollutant] = calculate_avg_every_7th_row(df, pollutant)

    # Determine the number of weeks from the longest list
    max_weeks = max(len(v) for v in weekly_data.values())

    # Create row numbers
    weekly_data['Week'] = list(range(1, max_weeks + 1))

    # Convert to DataFrame and reorder columns
    result_df = pd.DataFrame(weekly_data)
    cols = ['Week'] + columns
    return result_df[cols]



# Define pollutants to include
pollutants = ['NO', 'NO2', 'NOx', 'PM10']

# Generate and print the weekly average table for air
if df_air is not None:
    weekly_avg_table = build_weekly_avg_table(df_air, pollutants)
    print("Weekly averages for air quality data:")
    print(weekly_avg_table)

# Define parameters to include
pollutants = ['temperature (C)', 'wind_speed (m/s)', 'precipitation (mm)']

# Generate and print the weekly average table for weather 
if df_weather is not None:
    weekly_avg_table = build_weekly_avg_table(df_weather, pollutants)
    print("Weekly averages for weather quality data:")
    print(weekly_avg_table)


print("heihei!!<3")
#Code for standard deviation from all days of data for (1) air data:


# List of pollutants
pollutants = ['NO', 'NO2', 'NOx', 'PM10']

# Calculate standard deviation for each pollutant
for col in pollutants:
    if col in df_air.columns:
        std_val = df_air[col].std()
        print(f"Standard deviation for {col}: {round(std_val, 2)}")
    else:
        print(f"Column '{col}' not found in the DataFrame.")



# List of pollutants
parameters = ['temperature (C)', 'wind_speed (m/s)', 'precipitation (mm)']

# Calculate standard deviation for each pollutant
for col in parameters:
    if col in df_weather.columns:
        std_val = df_weather[col].std()
        print(f"Standard deviation for {col}: {round(std_val, 2)}")
    else:
        print(f"Column '{col}' not found in the DataFrame.")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


#-----------------------------------

sns.boxplot(df_air)

# Create boxplots for each column
sns.boxplot(data=df_air, orient='v')  # or orient='h' for horizontal boxplots

plt.xlabel("Air pollutant")  # X-axis label
plt.ylabel("Values")       # Y-axis label

plt.title("Boxplot of air quality data")  # Optional title
plt.show()


# Table of outlier data for air quality data


import pandas as pd
import numpy as np

# Select the columns to check
cols_to_check = df_air.columns[1:5]

# Function to identify outliers using IQR method
def find_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series[(series < lower_bound) | (series > upper_bound)]

# Create outlier summary table
outlier_table = {}
for col in cols_to_check:
    outliers = find_outliers(df_air[col])
    outlier_table[col] = outliers.values.tolist()

outlier_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in outlier_table.items()]))
print("Outlier Table:")
print(outlier_df)

# Create a copy of the DataFrame and replace outliers with NaN
df_air_cleaned = df_air.copy()

for col in cols_to_check:
    outliers = find_outliers(df_air[col])
    df_air_cleaned.loc[outliers.index, col] = np.nan

print("\nModified DataFrame (outliers replaced with NaN):")
print(df_air_cleaned[cols_to_check])





#-----


cols_to_plot = df_weather.columns[1:4]  # Columns at positions 1, 2, 3


n_cols = len(cols_to_plot)

# Create subplots — one row per selected column
fig, axes = plt.subplots(nrows=n_cols, ncols=1, figsize=(6, 4 * n_cols))

# Loop through selected columns and axes
for ax, col in zip(axes, cols_to_plot):
    sns.boxplot(y=df_weather[col], ax=ax)
    ax.set_title(f'Boxplot of {col}')
    ax.set_ylabel(f'{col} Value')
    ax.set_xlabel('')

plt.tight_layout()
plt.show()

#Outlier data for WEATHER data

import pandas as pd
import numpy as np

# Select the columns to check
cols_to_check = df_weather.columns[1:4]

# Function to identify outliers using IQR method
def find_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series[(series < lower_bound) | (series > upper_bound)]

# Create outlier summary table
outlier_table = {}
for col in cols_to_check:
    outliers = find_outliers(df_weather[col])
    outlier_table[col] = outliers.values.tolist()

outlier_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in outlier_table.items()]))
print("Outlier Table:")
print(outlier_df)

# Create a copy of the DataFrame and replace outliers with NaN
df_weather_cleaned = df_weather.copy()

for col in cols_to_check:
    outliers = find_outliers(df_weather[col])
    df_weather_cleaned.loc[outliers.index, col] = np.nan

print("\nModified DataFrame (outliers replaced with NaN):")
print(df_weather_cleaned[cols_to_check])





