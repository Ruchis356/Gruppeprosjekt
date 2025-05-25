import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ----------------------------- #
# Create a model for temperature and precipitation
# ----------------------------- #

def load_and_prepare_data(filepath):
    """
    Load and prepare historical weather data from CSV file
    Args:
        filepath: Path to the CSV file containing historical data
    Returns:
        DataFrame with processed weather data
    """
    df = pd.read_csv(filepath, parse_dates=['Date'])

    # Convert temperature, precipitation and wind to numeric, handle errors
    df['temperature (C)'] = pd.to_numeric(df['temperature (C)'], errors='coerce')
    df['precipitation (mm)'] = pd.to_numeric(df['precipitation (mm)'], errors='coerce')
    df['wind_speed (m/s)'] = pd.to_numeric(df['wind_speed (m/s)'], errors='coerce')
    # Remove rows with missing dates or weather data
    df = df.dropna(subset=['Date', 'temperature (C)', 'precipitation (mm)', 'wind_speed (m/s)'])
    # Add day of year column for seasonal modeling
    df['DayOfYear'] = df['Date'].dt.dayofyear
    return df

def create_model(degree=4):
    """Polinomial regression"""
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )

def get_daily_api_forecast():
    """
    Fetch daily weather forecast from MET Norway API
    Returns:
        DataFrame with daily average temperature and total precipitation
    """
    LAT, LON = 63.419, 10.395
    url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={LAT}&lon={LON}"
    headers = {"User-Agent": "myweatherapp/1.0 your_email@example.com"}
    response = requests.get(url, headers=headers)
    data = response.json()

    timeseries = data["properties"]["timeseries"]
    daily_data = {}


    # Aggregate hourly data into daily values
    for entry in timeseries:
        #AI-suggested improvement for robust datetime handling (DeepSeek)
        time = pd.to_datetime(entry["time"]).tz_localize(None)
        date = time.date()
        details = entry["data"]["instant"]["details"]
        temp = details.get("air_temperature")
        wind = details.get("wind_speed")
        precip = entry.get("data", {}).get("next_1_hours", {}).get("details", {}).get("precipitation_amount", 0)

        if date not in daily_data:
            daily_data[date] = {"temps": [], "precips": [], "winds": []}
        daily_data[date]["temps"].append(temp)
        daily_data[date]["precips"].append(precip)
        daily_data[date]["winds"].append(wind)

 # Calculate daily averages/sums
 # The following list comprehension was optimized by AI (DeepSeek)
    result = []
    for date, values in daily_data.items():
        result.append({
            "Date": pd.to_datetime(date),
            "API_Temp": np.mean(values["temps"]),
            "API_Precip": np.sum(values["precips"]),
            "API_Wind": np.mean(values["winds"]),
        })

    return pd.DataFrame(result)

def predict_future(model, last_date, days_to_predict):
    """
    Generate future predictions using trained model
    Args:
        model: Trained regression model
        last_date: Last date of historical data
        days_to_predict: Number of days to predict
    Returns:
        DataFrame with future dates and predictions
    """
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
    day_of_year = [d.dayofyear for d in future_dates]
    future_df = pd.DataFrame({'Date': future_dates, 'DayOfYear': day_of_year})
    future_df['Prediction'] = model.predict(future_df[['DayOfYear']])
    return future_df



def plot_full_overview(historical_df, forecast_temp, forecast_precip, forecast_wind, api_df):
    """
    Plot comprehensive overview of historical data, predictions and API forecast
    Args:
        historical_df: DataFrame with historical weather data
        forecast_temp: DataFrame with temperature predictions
        forecast_precip: DataFrame with precipitation predictions
        api_df: DataFrame with API forecast data
    """

    fig, ax = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    # Temperature plot
    ax[0].scatter(historical_df['Date'], historical_df['temperature (C)'], color='blue', alpha=0.3, label='Historisk temp')
    ax[0].plot(forecast_temp['Date'], forecast_temp['Prediction'], 'r-', label='Prognose temp')
    ax[0].scatter(api_df['Date'], api_df['API_Temp'], color='green', s=80, label='API temp')
    ax[0].set_ylabel('Temperatur (°C)')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # precipitation plot
    ax[1].scatter(historical_df['Date'], historical_df['precipitation (mm)'], color='blue', alpha=0.3, label='Historisk nedbør')
    ax[1].plot(forecast_precip['Date'], forecast_precip['Prediction'], 'r-', label='Prognose nedbør')
    ax[1].scatter(api_df['Date'], api_df['API_Precip'], color='green', s=80, label='API nedbør')
    ax[1].set_ylabel('Nedbør (mm)')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    # Wind plot
    ax[2].scatter(historical_df['Date'], historical_df['wind_speed (m/s)'], color='blue', alpha=0.3, label='Historisk vind')
    ax[2].plot(forecast_wind['Date'], forecast_wind['Prediction'], 'r-', label='Prognose vind')
    ax[2].scatter(api_df['Date'], api_df['API_Wind'], color='green', s=80, label='API vind')
    ax[2].set_ylabel('Vind (m/s)')
    ax[2].legend()
    ax[2].grid(True, alpha=0.3)

    plt.suptitle('Historikk, Prognose og API-data: Temperatur, Nedbør og Vind')
    plt.tight_layout()
    plt.show()

def plot_week_comparison(forecast_temp, forecast_precip, forecast_wind, api_df):
    """
    Plot 7 day comparison between model predictions and API forecast
    Args:
        forecast_temp: Temperature predictions DataFrame
        forecast_precip: Precipitation predictions DataFrame
        api_df: API forecast DataFrame
    """
    today = pd.to_datetime(datetime.now().date())
    end = today + timedelta(days=7)

    week_temp = forecast_temp[(forecast_temp['Date'] >= today) & (forecast_temp['Date'] <= end)]
    week_precip = forecast_precip[(forecast_precip['Date'] >= today) & (forecast_precip['Date'] <= end)]
    week_wind = forecast_wind[(forecast_wind['Date'] >= today) & (forecast_wind['Date'] <= end)]
    api_week = api_df[(api_df['Date'] >= today) & (api_df['Date'] <= end)]

    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Temperature comparison plot
    ax[0].plot(week_temp['Date'], week_temp['Prediction'], 'r-', label='Prognose temp')
    ax[0].scatter(api_week['Date'], api_week['API_Temp'], color='green', s=100, label='API temp')
    ax[0].set_ylabel('Temperatur (°C)')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Precipitation comparison plot
    ax[1].plot(week_precip['Date'], week_precip['Prediction'], 'r-', label='Prognose nedbør')
    ax[1].scatter(api_week['Date'], api_week['API_Precip'], color='green', s=100, label='API nedbør')
    ax[1].set_ylabel('Nedbør (mm)')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    # Wind comparison plot
    ax[2].plot(week_wind['Date'], week_wind['Prediction'], 'r-', label='Prognose vind')
    ax[2].scatter(api_week['Date'], api_week['API_Wind'], color='green', s=100, label='API vind')
    ax[2].set_ylabel('Vind (m/s)')
    ax[2].legend()
    ax[2].grid(True, alpha=0.3)

    plt.xticks(api_week['Date'], [d.strftime('%a\n%d.%m') for d in api_week['Date']])
    plt.xlabel('Dato')
    plt.suptitle('7-dagers sammenligning: Temperatur, Nedbør og Vind')
    plt.tight_layout()
    plt.show()

# ----------------------- #
# Main program
# ----------------------- #


def main():
    data_dir = 'data'
    filename = 'refined_weather_data.csv'
    filepath = os.path.join(data_dir, filename)

    try:
        historical_df = load_and_prepare_data(filepath)
        api_df = get_daily_api_forecast()

        model_temp = create_model()
        model_precip = create_model()
        model_wind = create_model()

        model_temp.fit(historical_df[['DayOfYear']], historical_df['temperature (C)'])
        model_precip.fit(historical_df[['DayOfYear']], historical_df['precipitation (mm)'])
        model_wind.fit(historical_df[['DayOfYear']], historical_df['wind_speed (m/s)'])

        last_hist_date = historical_df['Date'].max()
        last_api_date = api_df['Date'].max()
        days_to_predict = (last_api_date - last_hist_date).days
        
        # Generate temperature, precipitation and wind forecasts
        forecast_temp = predict_future(model_temp, last_hist_date, days_to_predict)
        forecast_precip = predict_future(model_precip, last_hist_date, days_to_predict)
        forecast_wind = predict_future(model_wind, last_hist_date, days_to_predict)
        #Plotting
        plot_full_overview(historical_df, forecast_temp, forecast_precip, forecast_wind, api_df)
        plot_week_comparison(forecast_temp, forecast_precip, forecast_wind, api_df)

    except Exception as e:
        print(f"FEIL: {e}")

if __name__ == "__main__":
    main()



def load_and_merge_data(weather_path, airquality_path):
    """
    Load and merge weather and air quality data from CSV files.
    
    Args:
        weather_path (str): Path to weather data CSV file
        airquality_path (str): Path to air quality data CSV file
    """

    # Load data with automatic date parsing
    weather = pd.read_csv(weather_path, parse_dates=['Date'])
    airquality = pd.read_csv(airquality_path, parse_dates=['Date'])
    

    merged = pd.merge(weather, airquality, on='Date', how='inner')
    merged = merged.dropna(subset=['temperature (C)', 'precipitation (mm)', 'wind_speed (m/s)', 'PM10', 'NO2'])
    
    merged['DayOfYear'] = merged['Date'].dt.dayofyear
    return merged

# Use of Train Random Forest model was suggested by AI for a better prediction (Deepseek)
def train_and_predict(merged_data, target, features, split_year=2018):
    """
    Train Random Forest model and generate predictions.
    
    Args:
        merged_data (pd.DataFrame): Combined weather and air quality data
        target (str): Target variable name ('PM10' or 'NO2')
        features (list): List of feature column names
        split_year (int): Year to split training/test data (default=2018)
    """
    # Create explicit copies with .copy() to avoid SettingWithCopyWarning suggested AI (Deepseek)
    train = merged_data.loc[merged_data['Date'].dt.year < split_year].copy()
    test = merged_data.loc[merged_data['Date'].dt.year >= split_year].copy()
    
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train[features], train[target])
    
    # Generate predictions
    predictions = model.predict(test[features])
    test = test.assign(Prediction=predictions) 
    
    mse = mean_squared_error(test[target], test['Prediction'])
    print(f"MSE for {target}: {mse:.2f}")
    
    return test

# ----------------------------- #
# Visualization Functions
# ----------------------------- #

def plot_results(test_data, target):
    """
    Plot comparison between actual and predicted pollution levels.
    
    Args:
        test_data (pd.DataFrame): Test dataset with predictions
        target (str): Pollution metric being plotted ('PM10' or 'NO2')
    """
    plt.figure(figsize=(12, 5))
    plt.plot(test_data['Date'], test_data[target], 'b-', label='Faktisk')
    plt.plot(test_data['Date'], test_data['Prediction'], 'r--', label='Predikert')
    plt.title(f'Forurensningsprognose for {target} (2018)')
    plt.ylabel(f'{target} (µg/m³)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()



if __name__ == "__main__":
    
    merged = load_and_merge_data(
        weather_path='data/refined_weather_data.csv',
        airquality_path='data/refined_air_qualty_data.csv'
    )
    
    # NO2 Prediction
    no2_test = train_and_predict(
        merged_data=merged,
        target='NO2',
        features=['temperature (C)', 'wind_speed (m/s)', 'DayOfYear']
    )
    
    # PM10 Prediction
    pm10_test = train_and_predict(
        merged_data=merged,
        target='PM10',
        features=['precipitation (mm)', 'wind_speed (m/s)', 'DayOfYear']
    )
    
    # Plott resultater
    plot_results(pm10_test, 'PM10')
    plot_results(no2_test, 'NO2')



