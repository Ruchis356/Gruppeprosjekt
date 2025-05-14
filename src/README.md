## analysis.py
🔹 averages(df, column_names)
Calculates weekly averages for specified columns, skipping missing values.
Returns a DataFrame with one row per week, including the week's starting date and averages. Warns on missing/invalid columns.

🔹 total_average(df, column_names)
Computes overall means for valid weather and air quality columns, ignoring invalid or missing ones.

🔹 standard_deviation(df, column_names)
Returns standard deviation for each valid column in the DataFrame.

🔹 outliers(df, column_names, standard_deviation, average, sd_modifier)
Identifies outliers using avg ± (std_dev × modifier).
Returns:

A DataFrame of detected outliers with dates

A copy of the data with outliers replaced by NaN

## data_handling.py

Handles missing data with the following modes:

'report': Returns locations of missing values

'drop': Removes rows with missing values

'fill': Replaces missing values with fill_value

Returns a modified DataFrame or None if no missing values found.

## data_import.py

🔹 Class: RawData
Handles weather and air quality data import and cleanup.

get_met(...):
Fetches weather data from Frost API, cleans and pivots it, returns daily values. Robust error handling included.

get_nilu(threshold, file_path):
Reads and cleans NILU CSV data, filters by coverage threshold, renames columns, and handles parsing errors.

## graphs.py

Creates time-series plots with Matplotlib & Seaborn:

dot_graph(...) – Scatter plots of multiple variables over time

comparative_graph(...) – Dual-axis plot of pollutants vs. weather

box_plot(...) – Boxplots including all outlier data

## predictive_analysis.py

Builds seasonal weather models and compares them to API forecasts.

load_and_prepare_data(): Cleans historical temp/precip data

create_model(): Trains polynomial regression (default degree 4)

get_daily_api_forecast(): Aggregates MET Norway forecast

predict_future(): Uses models to predict upcoming weather

plot_full_overview(): Visualizes historical vs. predicted vs. API

plot_week_comparison(): Compares 7-day model vs. API forecast

## utils.py
Defines VisualTable for rendering large DataFrames.
In Jupyter: scrollable HTML; otherwise: standard printout.