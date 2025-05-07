


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

def get_api_last_date():
    """Henter siste dato fra MET API (7 dager frem i tid)"""
    LAT, LON = 63.419, 10.395  # Trondheim
    url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={LAT}&lon={LON}"
    headers = {"User-Agent": "myweatherapp/1.0 your_email@example.com"}
    
    response = requests.get(url, headers=headers)
    data = response.json()
    timeseries = data["properties"]["timeseries"]
    last_date = datetime.fromisoformat(timeseries[-1]["time"]).date()
    return last_date

def predict_until_api_date(model, last_historical_date, last_api_date):
    """Predikerer temperaturer fra historisk sluttdato til API-sistedato"""
    total_days = (last_api_date - last_historical_date.date()).days
    future_dates = [last_historical_date + timedelta(days=x) for x in range(1, total_days + 1)]
    
    future_df = pd.DataFrame({
        'Date': future_dates,
        'DayOfYear': [x.dayofyear for x in future_dates]
    })
    
    future_df['Predicted_Temp'] = model.predict(future_df[['DayOfYear']])
    return future_df

# ---------------------- #
# Hovedprogram           #
# ---------------------- #

def main():
    # Konfigurasjon
    data_dir = 'data'
    filename = 'refined_weather_data.csv'  # Bytt til din fil
    filepath = os.path.join(data_dir, filename)
    
    try:
        # 1. Last historiske data
        df = load_and_prepare_data(filepath)
        if df is None:
            return
            
        print(f"Historiske data lastet: {len(df)} rader")
        print(f"Periode: {df['Date'].min().date()} til {df['Date'].max().date()}")

        # 2. Hent siste API-dato (7 dager frem)
        last_api_date = get_api_last_date()
        print(f"\nPrognoseperiode: {df['Date'].max().date()} til {last_api_date}")

        # 3. Tren modell
        forecast_model = create_model(degree=4)
        forecast_model.fit(df[['DayOfYear']], df['temperature (C)'])

        # 4. Prediker frem til API-sistedato
        future_df = predict_until_api_date(
            forecast_model, 
            df['Date'].max(), 
            last_api_date
        )

        # 5. Plot resultater
        plt.figure(figsize=(14, 7))
        plt.scatter(df['Date'], df['temperature (C)'], color='blue', alpha=0.3, label='Historiske data')
        plt.plot(future_df['Date'], future_df['Predicted_Temp'], 'r-', linewidth=2, label='Predikert temperatur')
        
        # Marker API-perioden
        api_start_date = datetime.now().date()
        plt.axvspan(api_start_date, last_api_date, color='green', alpha=0.1, label='API-dekket periode')
        
        plt.xlabel('Dato')
        plt.ylabel('Temperatur (°C)')
        plt.title(f'Temperaturprognose frem til {last_api_date}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 6. Lagre prognose
        future_df.to_csv(os.path.join(data_dir, 'prognose.csv'), index=False)
        print(f"\nPrognose lagret til {data_dir}/prognose.csv")

    except Exception as e:
        print(f"\nFEIL: {str(e)}")

if __name__ == "__main__":
    main()

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
# Hjelpefunksjoner       #
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

def get_api_forecast():
    """Henter temperaturprognose for de neste 7 dagene fra MET API"""
    LAT, LON = 63.419, 10.395  # Trondheim
    url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={LAT}&lon={LON}"
    headers = {"User-Agent": "myweatherapp/1.0 your_email@example.com"}
    
    response = requests.get(url, headers=headers)
    data = response.json()
    timeseries = data["properties"]["timeseries"]
    
    api_forecast = []
    for entry in timeseries[:24*7]:  # Begrens til 7 dager
        time = pd.to_datetime(entry["time"]).tz_localize(None)  # Fjern tidssone
        temp = entry["data"]["instant"]["details"]["air_temperature"]
        api_forecast.append({"Date": time, "API_Temp": temp})
    
    return pd.DataFrame(api_forecast)

def predict_until_api_date(model, last_historical_date, last_api_date):
    """Predikerer temperaturer fra historisk sluttdato til API-sistedato"""
    total_days = (last_api_date - last_historical_date.date()).days
    future_dates = [last_historical_date + timedelta(days=x) for x in range(1, total_days + 1)]
    
    future_df = pd.DataFrame({
        'Date': future_dates,
        'DayOfYear': [x.dayofyear for x in future_dates]
    })
    
    future_df['Predicted_Temp'] = model.predict(future_df[['DayOfYear']])
    return future_df

# ---------------------- #
# Plotting-funksjoner    #
# ---------------------- #

def plot_full_forecast(df, future_df, api_df):
    """Plotter full prognose med API-perioden markert"""
    plt.figure(figsize=(14, 7))
    
    # Historiske data
    plt.scatter(df['Date'], df['temperature (C)'], color='blue', alpha=0.3, label='Historiske data')
    
    # Predikert temperatur
    plt.plot(future_df['Date'], future_df['Predicted_Temp'], 'r-', linewidth=2, label='Predikert temperatur')
    
    # API-data (kun de neste 7 dagene)
    api_mask = (api_df['Date'] >= future_df['Date'].min()) & (api_df['Date'] <= future_df['Date'].max())
    plt.scatter(api_df[api_mask]['Date'], api_df[api_mask]['API_Temp'], color='green', s=100, label='API-prognose', zorder=3)
    
    plt.xlabel('Dato')
    plt.ylabel('Temperatur (°C)')
    plt.title('Full temperaturprognose')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_weekly_forecast(future_df, api_df):
    """Plotter kun ukesprognosen (neste 7 dager)"""
    plt.figure(figsize=(14, 7))
    
    # Konverter til naive datetime for sammenligning
    today = pd.to_datetime(datetime.now().date())
    next_week = today + timedelta(days=7)
    
    # Filter data
    future_mask = (future_df['Date'] >= today) & (future_df['Date'] <= next_week)
    api_mask = (api_df['Date'] >= today) & (api_df['Date'] <= next_week)
    
    # Plot
    plt.plot(future_df[future_mask]['Date'], 
             future_df[future_mask]['Predicted_Temp'], 
             'r-', linewidth=2, label='Vår prognose')
    
    plt.scatter(api_df[api_mask]['Date'], 
                api_df[api_mask]['API_Temp'], 
                color='green', s=100, label='API-prognose', zorder=3)
    
    plt.xlabel('Dato')
    plt.ylabel('Temperatur (°C)')
    plt.title('Ukesprognose: Vår modell vs. MET API')
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
        df = load_and_prepare_data(filepath)
        if df is None:
            return
            
        print(f"Historiske data lastet: {len(df)} rader")
        print(f"Periode: {df['Date'].min().date()} til {df['Date'].max().date()}")

        # 2. Hent API-prognose
        api_df = get_api_forecast()
        last_api_date = api_df['Date'].max().date()
        print(f"\nAPI-prognose hentet for {api_df['Date'].min().date()} til {last_api_date}")

        # 3. Tren modell
        forecast_model = create_model(degree=4)
        forecast_model.fit(df[['DayOfYear']], df['temperature (C)'])

        # 4. Prediker frem til API-sistedato
        future_df = predict_until_api_date(
            forecast_model, 
            df['Date'].max(), 
            last_api_date
        )

        # 5. Plott resultater
        plot_full_forecast(df, future_df, api_df)       # Full prognose
        plot_weekly_forecast(future_df, api_df)         # Fokus på neste uke

        # 6. Lagre data
        future_df.to_csv(os.path.join(data_dir, 'prognose.csv'), index=False)
        api_df.to_csv(os.path.join(data_dir, 'api_prognose.csv'), index=False)
        print(f"\nData lagret i {data_dir}/")

    except Exception as e:
        print(f"\nFEIL: {str(e)}")

if __name__ == "__main__":
    main()




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

def get_detailed_api_forecast():
    """Henter detaljert API-prognose med time-for-time data"""
    LAT, LON = 63.419, 10.395  # Trondheim
    url = f"https://api.met.no/weatherapi/locationforecast/2.0/complete?lat={LAT}&lon={LON}"
    headers = {"User-Agent": "myweatherapp/1.0 your_email@example.com"}
    
    response = requests.get(url, headers=headers)
    data = response.json()
    timeseries = data["properties"]["timeseries"]
    
    forecast_data = []
    for entry in timeseries:
        time = pd.to_datetime(entry["time"]).tz_localize(None)
        temp = entry["data"]["instant"]["details"]["air_temperature"]
        forecast_data.append({"Date": time, "API_Temp": temp})
    
    return pd.DataFrame(forecast_data)

# ---------------------- #
# Prognosefunksjoner     #
# ---------------------- #

def predict_future_temperatures(model, last_date, days_to_predict):
    """Predikerer temperaturer for angitt antall dager"""
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
    """Plotter full prognose fra historisk data til API-slutt"""
    plt.figure(figsize=(14, 7))
    
    # Historiske data
    plt.scatter(historical_df['Date'], historical_df['temperature (C)'], 
               color='blue', alpha=0.3, label='Historiske data')
    
    # Predikert temperatur
    plt.plot(forecast_df['Date'], forecast_df['Predicted_Temp'], 
            'r-', linewidth=2, label='Vår prognose')
    
    # API-data
    plt.scatter(api_df['Date'], api_df['API_Temp'], 
               color='green', s=30, label='API-prognose', zorder=3)
    
    plt.xlabel('Dato')
    plt.ylabel('Temperatur (°C)')
    plt.title('Komplett temperaturprognose')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_weekly_comparison(forecast_df, api_df):
    """Plotter detaljert ukesammenligning"""
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
    
    plt.xlabel('Dato')
    plt.ylabel('Temperatur (°C)')
    plt.title('Detaljert ukesammenligning: Vår modell vs MET API')
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

        # 2. Hent API-data
        api_df = get_detailed_api_forecast()
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