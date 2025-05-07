
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# ---------------------- #
# Felles funksjoner      #
# ---------------------- #

def load_and_prepare_data(filepath):
    """Laster og forbereder data for begge analyser"""
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

# ---------------------- #
# Analyse 1: Historisk   #
# ---------------------- #

def plot_historical_trends(df, model):
    """Visualiser historiske data med trendlinje"""
    plt.figure(figsize=(14, 8))
    
    # Fargekoding for år
    years = df['Year'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(years)))
    
    for i, year in enumerate(years):
        year_data = df[df['Year'] == year]
        plt.scatter(year_data['DayOfYear'], year_data['temperature (C)'],
                   color=colors[i], alpha=0.6, label=str(year), s=10)
    
    # Trendlinje
    days = np.linspace(1, 366, 366)
    plt.plot(days, model.predict(days.reshape(-1, 1)), 
             'r-', linewidth=3, label='Sesongtrend')
    
    plt.xlabel('Dag i året', fontsize=12)
    plt.ylabel('Temperatur (°C)', fontsize=12)
    plt.title(f'Temperaturanalyse ({len(years)} års data)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

# ---------------------- #
# Analyse 2: Prognose    #
# ---------------------- #

def predict_future_temperatures(model, last_date, years=1):
    """Predikerer fremtidige temperaturer"""
    future_dates = [last_date + timedelta(days=x) for x in range(1, 365*years + 1)]
    
    future_df = pd.DataFrame({
        'Date': future_dates,
        'DayOfYear': [x.dayofyear for x in future_dates]
    })
    
    future_df['Predicted_Temp'] = model.predict(future_df[['DayOfYear']])
    return future_df

def plot_forecast(df, future_df):
    """Visualiser historiske data og prognose"""
    plt.figure(figsize=(14, 7))
    
    plt.scatter(df['Date'], df['temperature (C)'], 
                color='blue', alpha=0.3, label='Historiske data')
    plt.plot(future_df['Date'], future_df['Predicted_Temp'], 
             'r-', linewidth=2, label='Predikert temperatur')
    
    plt.xlabel('Dato')
    plt.ylabel('Temperatur (°C)')
    plt.title('Temperaturprognose basert på historiske data')
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
        # Del 1: Last og forbered data
        df = load_and_prepare_data(filepath)
        if df is None:
            return
            
        print(f"\nData statistikk:")
        print(f"Antall datapunkter: {len(df):,}")
        print(f"Periode: {df['Date'].min().date()} til {df['Date'].max().date()}")
        print(f"Antall år: {len(df['Year'].unique())}")
        
        # Del 2: Historisk analyse
        print("\nKjører historisk analyse...")
        hist_model = create_model(degree=3)
        hist_model.fit(df[['DayOfYear']], df['temperature (C)'])
        plot_historical_trends(df, hist_model)
        
        # Del 3: Prognoseanalyse
        print("\nGenererer temperaturprognose...")
        forecast_model = create_model(degree=4)
        forecast_model.fit(df[['DayOfYear']], df['temperature (C)'])
        
        future_df = predict_future_temperatures(
            forecast_model, 
            df['Date'].max(), 
            years=5
        )
        
        plot_forecast(df, future_df)
        
        # Lagre prognose
        future_df[['Date', 'Predicted_Temp']].to_csv(
            os.path.join(data_dir, 'temperature_forecast.csv'), 
            index=False
        )
        print("\nPrognose lagret til 'data/temperature_forecast.csv'")
        
    except Exception as e:
        print(f"\nFEIL: {str(e)}")
        if 'df' in locals():
            print("\nDatadiagnostikk:")
            print(f"NaN-verdier: {df['temperature (C)'].isna().sum()}")
            print("Eksempeldata:\n", df.head())

if __name__ == "__main__":
    main()