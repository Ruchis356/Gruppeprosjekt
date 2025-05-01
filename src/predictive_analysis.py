
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import os

# Sett opp filbane
data_dir = 'data'
filename = 'refined_weather_data.csv'
filepath = os.path.join(data_dir, filename)

def load_and_clean_data(filepath):
    """Laster og renser data med grundig NaN-håndtering"""
    try:
        df = pd.read_csv(filepath)
        
        # Konverter kolonner - mer robust håndtering
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['temperature (C)'] = pd.to_numeric(df['temperature (C)'], errors='coerce')
        
        # Fjern ugyldige rader
        df = df.dropna(subset=['Date', 'temperature (C)'])
        
        # Sjekk for tomme datasett
        if df.empty:
            raise ValueError("Ingen gyldige data etter rensing")
            
        # Lag tidsfunksjoner
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['Year'] = df['Date'].dt.year
        
        return df
    
    except Exception as e:
        print(f"Feil under datalasting: {str(e)}")
        return None

# Hovedanalyse
try:
    # 1. Datainnlasting
    df = load_and_clean_data(filepath)
    if df is None:
        raise SystemExit("Kunne ikke laste data")
    
    print(f"Antall datapunkter: {len(df)}")
    print(f"Periode: {df['Date'].min().date()} til {df['Date'].max().date()}")
    
    # 2. Forbered data - ekstra NaN-sjekk
    X = df[['DayOfYear']].values
    y = df['temperature (C)'].values
    
    if np.isnan(y).any():
        raise ValueError("NaN-verdier oppdaget i temperaturdata etter rensing")
    
    # 3. Modellering
    model = make_pipeline(
        PolynomialFeatures(degree=3, include_bias=False),
        LinearRegression()
    )
    model.fit(X, y)
    
    # 4. Visualisering
    plt.figure(figsize=(14, 8))
    
    # Plott hvert år med egen farge
    years = df['Year'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(years)))
    
    for i, year in enumerate(years):
        year_data = df[df['Year'] == year]
        plt.scatter(year_data['DayOfYear'], year_data['temperature (C)'],
                   color=colors[i], alpha=0.6, label=str(year), s=10)
    
    # Plott trendlinje
    days = np.linspace(1, 366, 366)
    y_pred = model.predict(days.reshape(-1, 1))
    plt.plot(days, y_pred, 'r-', linewidth=3, label='Sesongtrend')
    
    # Formatering
    plt.xlabel('Dag i året', fontsize=12)
    plt.ylabel('Temperatur (°C)', fontsize=12)
    plt.title(f'Temperaturanalyse ({len(years)} års data)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"\nFEIL: {str(e)}")
    
    # Diagnostikk hvis feil oppstår
    if 'df' in locals():
        print("\nDatadiagnostikk:")
        print(f"Antall NaN i temperatur: {df['temperature (C)'].isna().sum()}")
        print("Eksempel på data:")
        print(df.head())
        print("\nDatoer med NaN temperatur:")
        print(df[df['temperature (C)'].isna()]['Date'].head())
    
    print("\nVanlige årsaker til denne feilen:")
    print("1. Manglende eller ugyldige temperaturverdier i CSV-filen")
    print("2. Problemer med desimalseparator (f.eks. komma istedenfor punktum)")
    print("3. Ugyldige datoformat som forhindrer korrekt parsing")