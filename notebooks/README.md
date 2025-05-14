Describes the notebooks directory

# IMPORT AND SETUP

This script sets up the environment for data processing and visualization:

Imports necessary modules and custom classes for handling raw data, cleaning, analysis, visualization, and graphing.

Defining configuration variables for weather data retrieval, file paths, and analysis parameters like outlier thresholds and plotting settings.

# Weather Data - Importing and Proccessing 

Fetches weather data using the FROST API, handles any missing values, and prepares the data for analysis.

Retrieves weather data via the RawData class using provided API settings.

Checks for and reports missing values, then fills them if needed.

Displays the cleaned data in a readable table format 

# AIR QUALITY DATA - IMPORT AND PROCESSING

Does the same as above but for air quality data. 

## DATA AND PATTERN ANALYSIS

Performs statistical analysis on weather and air pollution data:

Weekly Averages: Calculates and displays weekly averages for both weather and air quality variables.

Standard Deviations: Computes and displays variability for each data set.

Total Averages: Shows the overall mean values for weather and air pollution data.

Outlier Detection: Identifies and removes outliers based on standard deviation thresholds, then displays both the outliers and the cleaned data.

## DATA VISUALISATION AND GRAPHING

 Generates various visualizations to explore air quality and weather data:

Scatter plots of daily and weekly air pollution levels.

Scatter plots of daily and weekly weather variables (e.g., temperature, wind, precipitation).

Comparative graphs showing relationships between air pollutants and individual weather variables, with customized y-axis limits and alignment.

# INTERACTIVE DATA VISUALIZATION

adds interactive visualizations using Plotly:

Enables toggling between different pollutants in daily and weekly scatter plots.

Includes a comparative graph with a dropdown to explore the relationship between air pollutants and a selected weather variable.

Uses InteractiveGraphs, a custom class for dynamic, user-friendly plots inside Jupyter notebooks.

# PREDICTIVE ANALYSIS



