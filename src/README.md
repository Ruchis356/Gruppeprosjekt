# Source Code Documentation
Note: This file was formatted by AI (DeepSeek)

## analysis.py
**Statistical analysis of environmental data**

### Key Functions:
- `averages(df, column_names)`  
  Calculates weekly averages for specified columns. Handles missing values intelligently. Returns DataFrame with weekly dates and averages.

- `total_average(df, column_names)`  
  Computes overall means for valid columns.

- `standard_deviation(df, column_names)`  
  Calculates standard deviations.

- `outliers(df, column_names, std_dev, avg, sd_modifier=2)`  
  Identifies outliers using `avg ± (std_dev × modifier)`.  
  Returns tuple: (DataFrame of outliers, cleaned data with outliers as NaN)

- `box_plots(df, columns, ...)`  
  Creates boxplots with outlier visualization.

---

## data_handling.py
**Missing data processing**

### Key Function:
- `missing_data(df, strategy='report', fill_value=None)`  
  Processes missing values with three strategies:  
  • `report` - Returns missing value locations  
  • `drop` - Removes rows with missing values  
  • `fill` - Replaces with specified `fill_value`

---

## data_import.py
**Class: RawData**  
Fetches and cleans environmental data from multiple sources.

### Methods:
- `get_met(station_id, elements, time_range, resolution)`  
  Fetches weather data from Frost API. Automatically pivots to wide format. Includes robust error handling.

- `get_nilu(threshold, file_path)`  
  Processes air quality CSVs from NILU. Filters by coverage threshold (0-100). Standardizes column names.

- `get_forecast(station_id=None, lat=63.419, lon=10.395)`  
  Fetches **7-day weather forecast** from MET Norway API.  
  - Uses station coordinates if provided  
  - Returns daily aggregates (temp, wind, precip)  

---

## graphs.py
**Class: Graphs**
Static data visualization.

### Key Functions:
- `dot_graph(df, columns, title, x_axis, y_axis)`  
  Creates scatter plots for time-series data.

- `comparative_graph(df, columns, df_predictor, predictor, ...)`  
  Generates dual-axis plots for pollutants vs weather.

**Class: PredictiveGraphs**  
Advanced visualization for model outputs.

### Key Methods:
- `plot_full_overview()`  
  Compares historical vs predicted vs API forecast data  
- `plot_week_comparison()`  
  Detailed 7-day forecast comparison  
- `plot_results()`  
  Shows actual vs predicted pollution levels  
- `plot_comparison()`  
  Side-by-side train/test set performance  
- `plot_pollutant_forecasts()`  
  Multi-pollutant forecast visualization  
- `model_information()`  
  Displays feature importance and diagnostic plots  

---

## predictive_analysis.py
**Weather forecasting and pollution prediction**

### Core Features:
- `load_and_merge_data()` - Cleans and merges historical data
- `create_model()` - Trains polynomial regression models
- `train_model()` - Configures Random Forest models
- `forecast_pollutants_with_lags()` - Recursive forecasting
- `predict_future(model, last_date, days_to_predict)` - Generates future predictions using trained models

---

## utils.py
**Data display utilities**

### Class: VisualTable
- `pretty_data(df)`  
  Displays DataFrames as:  
  • Scrollable HTML tables (Jupyter)  
  • Formatted text (terminal)  
  Auto-formats dates and numbers.

---

## graph_test.py
*Experimental Plotly visualizations*  
⚠️ Note: Not fully integrated into main workflow