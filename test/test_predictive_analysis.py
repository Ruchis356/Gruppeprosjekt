import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys, os

# Legg til stien til modulen som skal testes
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from predictive_analysis import WeatherAnalyser 

class TestLoadAndMergeData(unittest.TestCase):
    def setUp(self):
        """Setter opp testdata som brukes i alle tester"""
        self.analyser = TestWeatherAnalyser()  # Antar at Analyser er klassen som inneholder load_and_merge_data
        
        # Oppretter testdata for vær
        dates = pd.date_range(start="2020-01-01", periods=30)
        self.df_weather = pd.DataFrame({
            'Date': dates,
            'temperature (C)': np.random.uniform(-5, 25, 30),
            'precipitation (mm)': np.random.uniform(0, 10, 30),
            'wind speed (m/s)': np.random.uniform(0, 15, 30)
        })
        
        # Oppretter testdata for luftkvalitet
        self.df_quality = pd.DataFrame({
            'Date': dates,
            'PM10': np.random.uniform(0, 50, 30),
            'NO2': np.random.uniform(0, 100, 30),
            'O3': np.random.uniform(0, 120, 30)
        })
        
        # Definerer variabler som brukes i testen
        self.weather_vars = ['temperature (C)', 'precipitation (mm)', 'wind speed (m/s)']
        self.pollutant_vars = ['PM10', 'NO2', 'O3']

    def test_load_and_merge_data_basic(self):
        """Test basic merging functionality"""
        merged = self.analyser.load_and_merge_data(
            self.df_weather, 
            self.df_quality,
            self.weather_vars,
            self.pollutant_vars
        )
        
        # Sjekk grunnleggende form og kolonner
        self.assertEqual(merged.shape[0], 30)
        self.assertIn('PM10', merged.columns)
        self.assertIn('temperature (C)', merged.columns)
        self.assertIn('DayOfYear', merged.columns)
        
        # Sjekk avledete funksjoner
        self.assertIn('DayOfYear_sin', merged.columns)
        self.assertIn('DayOfYear_cos', merged.columns)
        self.assertIn('Weekend', merged.columns)
        self.assertIn('Season', merged.columns)
        
        # Sjekk interaksjonsledd
        self.assertIn('temp_wind_interaction', merged.columns)
        self.assertIn('temp_precip_interaction', merged.columns)
        
        # Sjekk sortering
        self.assertTrue(merged['Date'].is_monotonic_increasing)

    def test_load_and_merge_data_missing_columns(self):
        """Test that function handles missing columns correctly"""
        # Fjern en kolonne som forventes å være der
        df_quality_missing = self.df_quality.drop(columns=['PM10'])
        
        with self.assertRaises(KeyError):
            self.analyser.load_and_merge_data(
                self.df_weather,
                df_quality_missing,
                self.weather_vars,
                self.pollutant_vars
            )

    def test_load_and_merge_data_empty_input(self):
        """Test handling of empty DataFrames"""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(ValueError):
            self.analyser.load_and_merge_data(
                empty_df,
                self.df_quality,
                self.weather_vars,
                self.pollutant_vars
            )
        
        with self.assertRaises(ValueError):
            self.analyser.load_and_merge_data(
                self.df_weather,
                empty_df,
                self.weather_vars,
                self.pollutant_vars
            )

if __name__ == '__main__':
    unittest.main()




