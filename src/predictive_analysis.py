import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error

# Oppsett
data_dir = 'data'
filename = 'refined_weather_data.csv'
filepath = os.path.join(data_dir, filename)

def find_optimal_degree(X, y, max_degree=5):
    """Finner optimal polynomgrad ved kryssvalidering"""
    degrees = range(1, max_degree+1)
    mse_scores = []
    
    for degree in degrees:
        model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            LinearRegression()
        )
        scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        mse_scores.append(np.mean(scores))
    
    optimal_degree = degrees[np.argmin(mse_scores)]
    return optimal_degree, mse_scores

try:
    # Laster data
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfYear'] = df['Date'].dt.dayofyear
    
    # Forbereder data
    X = df[['DayOfYear']]  # Bruker kun dagnummer for enklere visualisering
    y = df['temperature (C)']
    
    # Finner optimal grad
    optimal_degree, mse_scores = find_optimal_degree(X, y, max_degree=6)
    print(f"Optimal polynomgrad funnet: {optimal_degree}")
    
    # Trener modell med optimal grad
    model = make_pipeline(
        PolynomialFeatures(degree=optimal_degree, include_bias=False),
        LinearRegression()
    )
    model.fit(X, y)
    
    # Genererer glatt kurve for plotting
    days = np.linspace(1, 366, 500).reshape(-1, 1)
    y_vis = model.predict(days)
    
    # Plotting
    plt.figure(figsize=(14, 7))
    
    # Faktiske data
    plt.scatter(df['DayOfYear'], df['temperature (C)'], 
                color='#2ecc71', alpha=0.6, label='Faktiske målinger', s=30)
    
    # Modellkurve
    plt.plot(days, y_vis, color='#e67e22', linewidth=3, 
             label=f'Polynomisk tilpasning (grad {optimal_degree})')
    
    # Formatering
    plt.xlabel('Dag i året', fontsize=13)
    plt.ylabel('Temperatur (°C)', fontsize=13)
    plt.title(f'Temperaturutvikling gjennom året\n(Best passende polynomgrad: {optimal_degree})', fontsize=15)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.2)
    
    # Månedsmerker
    month_pos = [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'Mai', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Des']
    plt.xticks(month_pos, month_names, fontsize=11)
    
    # Legg til RMSE i plottet
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    plt.text(0.02, 0.05, f'RMSE: {rmse:.2f}°C', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Kunne ikke finne filen: {filepath}")
except Exception as e:
    print(f"Feil under kjøring: {str(e)}")
    