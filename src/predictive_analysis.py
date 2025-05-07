


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import requests

# ---------------------- #
# Felles funksjoner      #
# ---------------------- #

def load_and_prepare_data(filepath):
    """Laster og forbereder historiske data fra CSV"""
    try:
        df = pd.read_csv(filepath, parse_dates=['Date'])
        df['temperature (C)'] = pd.to_numeric(df['temperature (C)'], errors='coerce')
        df = df.dropna(subset=['Date', 'temperature (C)'])
        
        if df.empty:
            raise ValueError("Ingen gyldige data etter rensing")
            
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['Year'] = df['Date'].dt.year
        return df
    
    except Exception as e:
        print(f"Feil under datalasting: {str(e)}")
        return None

def create_model(degree=3):
    """Oppretter polynomisk regresjonsmodell"""
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )

def get_daily_api_forecast():
    """Henter dagsgjennomsnitt fra MET API (compact versjon)"""
    LAT, LON = 63.419, 10.395  # Trondheim
    url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={LAT}&lon={LON}"
    headers = {"User-Agent": "myweatherapp/1.0 your_email@example.com"}
    
    response = requests.get(url, headers=headers)
    data = response.json()
    timeseries = data["properties"]["timeseries"]
    
    # Lag en dictionary for å gruppere etter dato
    daily_data = {}
    
    for entry in timeseries:
        time = pd.to_datetime(entry["time"]).tz_localize(None)
        date = time.date()
        temp = entry["data"]["instant"]["details"]["air_temperature"]
        
        if date not in daily_data:
            daily_data[date] = []
        daily_data[date].append(temp)
    
    # Beregn dagsgjennomsnitt
    api_forecast = []
    for date, temps in daily_data.items():
        api_forecast.append({
            "Date": pd.to_datetime(date),
            "API_Temp": np.mean(temps)
        })
    
    return pd.DataFrame(api_forecast)

# ---------------------- #
# Prognosefunksjoner     #
# ---------------------- #

def predict_future_temperatures(model, last_date, days_to_predict):
    """Predikerer temperaturer for angitt antall dager (én verdi per dag)"""
    future_dates = [last_date + timedelta(days=x) for x in range(1, days_to_predict+1)]
    
    future_df = pd.DataFrame({
        'Date': future_dates,
        'DayOfYear': [x.dayofyear for x in future_dates]
    })
    
    future_df['Predicted_Temp'] = model.predict(future_df[['DayOfYear']])
    return future_df

# ---------------------- #
# Plotting-funksjoner    #
# ---------------------- #

def plot_complete_forecast(historical_df, forecast_df, api_df):
    """Plotter full prognose med dagsgjennomsnitt"""
    plt.figure(figsize=(14, 7))
    
    # Historiske data
    plt.scatter(historical_df['Date'], historical_df['temperature (C)'], 
               color='blue', alpha=0.3, label='Historiske data')
    
    # Predikert temperatur
    plt.plot(forecast_df['Date'], forecast_df['Predicted_Temp'], 
            'r-', linewidth=2, label='Vår prognose')
    
    # API-data (dagsgjennomsnitt)
    api_mask = (api_df['Date'] >= forecast_df['Date'].min()) & (api_df['Date'] <= forecast_df['Date'].max())
    plt.scatter(api_df[api_mask]['Date'], api_df[api_mask]['API_Temp'], 
               color='green', s=100, label='API-prognose', zorder=3)
    
    plt.xlabel('Dato')
    plt.ylabel('Temperatur (°C)')
    plt.title('Komplett temperaturprognose (dagsgjennomsnitt)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_weekly_comparison(forecast_df, api_df):
    """Plotter detaljert ukesammenligning med dagsgjennomsnitt"""
    plt.figure(figsize=(14, 7))
    
    # Finn ukesperiode
    today = pd.to_datetime(datetime.now().date())
    week_end = today + timedelta(days=7)
    
    # Filtrer data
    forecast_week = forecast_df[(forecast_df['Date'] >= today) & 
                              (forecast_df['Date'] <= week_end)]
    api_week = api_df[(api_df['Date'] >= today) & 
                     (api_df['Date'] <= week_end)]
    
    # Plot vår prognose
    plt.plot(forecast_week['Date'], forecast_week['Predicted_Temp'],
            'r-', linewidth=2, label='Vår prognose')
    
    # Plot API-data
    plt.scatter(api_week['Date'], api_week['API_Temp'],
               color='green', s=100, label='API-prognose', zorder=3)
    
    # Legg til datoetiketter for bedre lesbarhet
    plt.xticks(forecast_week['Date'], [d.strftime('%a\n%d.%m') for d in forecast_week['Date']])
    
    plt.xlabel('Dato')
    plt.ylabel('Temperatur (°C)')
    plt.title('Ukesammenligning: Vår modell vs MET API (dagsgjennomsnitt)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ---------------------- #
# Hovedprogram           #
# ---------------------- #

def main():
    # Konfigurasjon
    data_dir = 'data'
    filename = 'refined_weather_data.csv'
    filepath = os.path.join(data_dir, filename)
    
    try:
        # 1. Last historiske data
        historical_df = load_and_prepare_data(filepath)
        if historical_df is None:
            return
            
        print(f"Historiske data lastet: {len(historical_df)} rader")
        print(f"Periode: {historical_df['Date'].min().date()} til {historical_df['Date'].max().date()}")

        # 2. Hent API-data (dagsgjennomsnitt)
        api_df = get_daily_api_forecast()
        last_api_date = api_df['Date'].max()
        print(f"\nAPI-prognose hentet for {api_df['Date'].min().date()} til {last_api_date.date()}")
        
        # 3. Beregn prognoseperiode
        days_to_predict = (last_api_date - historical_df['Date'].max()).days
        print(f"Antall dager å predikere: {days_to_predict}")

        # 4. Tren modell og prediker
        model = create_model(degree=4)
        model.fit(historical_df[['DayOfYear']], historical_df['temperature (C)'])
        
        forecast_df = predict_future_temperatures(
            model,
            historical_df['Date'].max(),
            days_to_predict
        )

        # 5. Visualiser resultater
        plot_complete_forecast(historical_df, forecast_df, api_df)
        plot_weekly_comparison(forecast_df, api_df)

        # 6. Lagre resultater
        forecast_df.to_csv(os.path.join(data_dir, 'temperature_forecast.csv'), index=False)
        api_df.to_csv(os.path.join(data_dir, 'api_forecast.csv'), index=False)
        print("\nResultater lagret i 'data/' mappen")

    except Exception as e:
        print(f"\nFEIL: {str(e)}")

if __name__ == "__main__":
    main()