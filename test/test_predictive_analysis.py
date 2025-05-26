
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from predictive_analysis import WeatherAnalyser

class TestWeatherAnalyser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Oppsett av testdata som gjelder for alle tester"""
        cls.analyser = WeatherAnalyser()
        
        # Create weather test data with linear temperature and wind speed values
        dates = pd.date_range(start='2023-01-01', end='2023-01-10')
        cls.df_weather = pd.DataFrame({
            'Date': dates,
            'temperature (C)': np.linspace(0, 10, 10),
            'wind_speed (m/s)': np.linspace(1, 5, 10),
            'precipitation (mm)': np.zeros(10)
        })
        
        # Create air quality test data with linear pollutant values
        cls.df_quality = pd.DataFrame({
            'Date': dates,
            'PM10': np.linspace(5, 15, 10),
            'NO2': np.linspace(10, 20, 10)
        })
        
        # Test data for prediction methods
        cls.last_date = datetime(2023, 1, 10)
        cls.days_to_predict = 5
        
        # Create merged test data for evaluation
        cls.test_data = cls.analyser.load_and_merge_data(
            cls.df_weather, 
            cls.df_quality,
            ['temperature (C)', 'wind_speed (m/s)', 'precipitation (mm)'],
            ['PM10', 'NO2']
        )
        cls.train_data = cls.test_data.copy()
        # The following RandomForest setup was simplified for testing purposes with use of AI (Deepseek)
        # In production, more estimators and careful parameter tuning would be needed
        cls.model = RandomForestRegressor(n_estimators=10, random_state=42)
        cls.model.fit(
            cls.train_data[['temperature (C)', 'wind_speed (m/s)']], 
            cls.train_data['PM10']
        )

    def test_load_and_merge_data(self):
        """Test at data lastes og merges korrekt"""
        weather_vars = ['temperature (C)', 'wind_speed (m/s)', 'precipitation (mm)']
        pollutant_vars = ['PM10', 'NO2']
        

        merged = self.analyser.load_and_merge_data(
            self.df_weather, 
            self.df_quality,
            weather_vars,
            pollutant_vars
        )
        
        self.assertIsInstance(merged, pd.DataFrame)
        self.assertGreater(len(merged), 0)
        self.assertIn('DayOfYear', merged.columns)
        self.assertIn('DayOfYear_sin', merged.columns)
        self.assertIn('DayOfYear_cos', merged.columns)
        
        # Test handling of missing required columns
        with self.assertRaises(ValueError):
            self.analyser.load_and_merge_data(
                self.df_weather.drop(columns=['Date']),
                self.df_quality,
                weather_vars,
                pollutant_vars
            )
            

        test_quality = self.df_quality.copy()
        test_quality.columns = [col + '_test' for col in test_quality.columns]
        merged_test = self.analyser.load_and_merge_data(
            self.df_weather,
            test_quality,
            weather_vars,
            pollutant_vars,
            mode='test'
        )
        self.assertIn('PM10', merged_test.columns)

    def test_safe_fit(self):
        """Test at modell kan trenes selv med NaN-verdier"""
        X = self.test_data[['temperature (C)', 'wind_speed (m/s)']]
        y = self.test_data['PM10'].copy()
        
        # Introducer noen NaN-verdier
        y.iloc[2:4] = np.nan
        
        # AI Declaration: The safe_fit method's NaN handling was suggested by AI (Deepseek)
        # to make training more robust against missing target values
        model = LinearRegression()
        fitted_model = self.analyser.safe_fit(model, X, y)
        
        self.assertTrue(hasattr(fitted_model, 'coef_'))
        self.assertEqual(len(fitted_model.coef_), 2)

    def test_create_model(self):
        """Test opprettelse av modellpipeline"""
        model = self.analyser.create_model(degree=2)
        
        self.assertIsInstance(model, Pipeline)
        self.assertEqual(len(model.steps), 3)
        self.assertEqual(model.steps[0][0], 'polynomialfeatures')
        self.assertEqual(model.steps[2][0], 'linearregression')

    def test_train_model(self):
        """Test trening av modell"""

        model_with_features = self.analyser.train_model(
            self.test_data,
            'PM10',
            features=['temperature (C)', 'wind_speed (m/s)']
        )
        self.assertIsInstance(model_with_features, RandomForestRegressor)
        
        # Test with automatic feature selection
        # The mutual_info_regression feature selection was AI-suggested
        # to automatically find relevant features
        model_auto_features = self.analyser.train_model(
            self.test_data,
            'PM10'
        )
        self.assertIsInstance(model_auto_features, RandomForestRegressor)

        with self.assertRaises(ValueError):
            self.analyser.train_model(self.test_data, 'NON_EXISTENT')

    def test_predict_future(self):
        """Test fremtidsprediksjoner"""
        # Create a simple model for testing predictions
        model = LinearRegression()
        X = self.test_data[['DayOfYear']]
        y = self.test_data['PM10']
        model.fit(X, y)
        
        predictions = self.analyser.predict_future(
            model,
            self.last_date,
            self.days_to_predict
        )
        
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertEqual(len(predictions), self.days_to_predict)
        self.assertIn('Prediction', predictions.columns)
        
        # Test invalid input handling
        with self.assertRaises(ValueError):
            self.analyser.predict_future(model, 'invalid_date', 5)
            
        with self.assertRaises(ValueError):
            self.analyser.predict_future(model, self.last_date, -1)

    def test_evaluate_model(self):
        """Test evaluering av modell"""
        predictions, valid_data, mse, r2 = self.analyser.evaluate_model(
            self.model,
            self.test_data,
            'PM10',
            features=['temperature (C)', 'wind_speed (m/s)'],
            train_data=self.train_data
        )
        
        self.assertIsNotNone(predictions)
        self.assertIsInstance(valid_data, pd.DataFrame)
        self.assertGreaterEqual(len(valid_data), 0)
        self.assertIsInstance(mse, float)
        self.assertIsInstance(r2, float)
        
         # Test handling of missing features
        with self.assertRaises(ValueError):
            self.analyser.evaluate_model(
                self.model,
                self.test_data,
                'PM10',
                features=['NON_EXISTENT_FEATURE']
            )
            
        # Test handling of invalid target column
        with self.assertRaises(ValueError):
            self.analyser.evaluate_model(
                self.model,
                self.test_data,
                'NON_EXISTENT_TARGET'
            )

if __name__ == '__main__':
    unittest.main()



