import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Last inn datasett
df_air = pd.read_csv('data/refined_air_qualty_data.csv', parse_dates=['Date'])
df_weather = pd.read_csv('data/refined_weather_data.csv', parse_dates=['Date'])

# Merge datasett på felles kolonne 'Date'
df = pd.merge(df_air, df_weather, on='Date')

# Sjekk for manglende verdier
missing = df.isna().sum()
print("Manglende verdier:\n", missing)

# Enkel imputasjon: fyll ut med gjennomsnitt
df.fillna(df.mean(numeric_only=True), inplace=True)



plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='Date', y='NOx', color='tomato')
plt.title('Utvikling av NOx-nivå over tid')
plt.xlabel('Dato')
plt.ylabel('NOx (µg/m³)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#visualisering søylediagram
avg_pollution = df[['NO', 'NO2', 'NOx', 'PM10', 'PM2.5']].mean()
avg_pollution.plot(kind='bar', color=sns.color_palette("Set2"))
plt.title('Gjennomsnittlige verdier av luftforurensning')
plt.ylabel('µg/m³')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



#scatteplot med regresjonslinje
plt.figure(figsize=(8, 6))
sns.regplot(x='temperature (C)', y='NOx', data=df, scatter_kws={'alpha':0.5})
plt.title('Sammenheng mellom temperatur og NOx')
plt.xlabel('Temperatur (°C)')
plt.ylabel('NOx (µg/m³)')
plt.tight_layout()
plt.show()



import missingno as msno

# Visualisering av manglende verdier????
msno.matrix(df_air)
plt.title('Manglende verdier i luftdata')
plt.show()



#evaluering for alle drivhusgassene
# Funksjoner (X) er de samme for alle modeller
X = df[['temperature (C)', 'wind_speed (m/s)', 'precipitation (mm)']]

# Liste over målvariabler
pollutants = ['NO', 'NO2', 'NOx', 'PM10', 'PM2.5']

print("Evaluering av lineær regresjonsmodell for hver luftforurensningskomponent:\n")

for pollutant in pollutants:
    y = df[pollutant]
    
    # Splitte data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Trene modell
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediksjon
    y_pred = model.predict(X_test)

    # Evaluering
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"{pollutant} ➤ R²: {r2:.3f} | RMSE: {rmse:.3f}")





#fretidige verdier for alle drivhusgasser (en måned)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generer datoer for én måned fremover
start_date = datetime(2025, 5, 1)
future_dates = [start_date + timedelta(days=i) for i in range(30)]

# Simulér værdata (du kan bruke gjennomsnitt eller tilfeldig variasjon)
np.random.seed(42)  # for reproduserbarhet

future_weather = pd.DataFrame({
    'Date': future_dates,
    'temperature (C)': np.random.normal(loc=10, scale=5, size=30),  # gj.snitt 10°C
    'wind_speed (m/s)': np.random.normal(loc=3, scale=1, size=30),  # gj.snitt 3 m/s
    'precipitation (mm)': np.random.exponential(scale=1.0, size=30)  # regn er mer uforutsigbart
})
# Lagre alle modeller og målvariabler
pollutants = ['NO', 'NO2', 'NOx', 'PM10', 'PM2.5']
predictions = {}

for pollutant in pollutants:
    # Målvariabel
    y = df[pollutant]

    # Treningsdata
    X = df[['temperature (C)', 'wind_speed (m/s)', 'precipitation (mm)']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelltrening
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediksjoner for fremtidig vær
    pred = model.predict(future_weather[['temperature (C)', 'wind_speed (m/s)', 'precipitation (mm)']])
    predictions[pollutant] = pred




#lage graf for alle drivhusgassene over tid (30 dager)
import matplotlib.pyplot as plt

# Legg til prediksjoner i DataFrame
for pollutant in pollutants:
    future_weather[pollutant] = predictions[pollutant]

# Plot alle drivhusgassene over tid
plt.figure(figsize=(14, 7))

for pollutant in pollutants:
    plt.plot(future_weather['Date'], future_weather[pollutant], label=pollutant)

plt.title('Predikterte nivåer av luftforurensning i løpet av én måned')
plt.xlabel('Dato')
plt.ylabel('Konsentrasjon (µg/m³)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


