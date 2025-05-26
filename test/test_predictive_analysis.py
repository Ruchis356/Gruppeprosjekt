
# This code was created and modified using the assistance of AI (ChatGPT) 
# See detailed use of AI file for further explanation
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import sys
import os
from unittest.mock import MagicMock

# Mock tqdm before importing predictive_analysis
sys.modules['tqdm'] = MagicMock()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from predictive_analysis import WeatherAnalyser

class TestWeatherAnalyser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Test data setup that works with existing predictive_analysis.py"""
        cls.analyser = WeatherAnalyser()
        
        # Create 15 days of data to ensure lag features work
        dates = pd.date_range(start='2023-01-01', periods=15)
        cls.df_weather = pd.DataFrame({
            'Date': dates,
            'temperature (C)': np.linspace(0, 14, 15),
            'wind_speed (m/s)': np.linspace(1, 7.5, 15),
            'precipitation (mm)': np.zeros(15)
        })
        
        cls.df_quality = pd.DataFrame({
            'Date': dates,
            'PM10': np.linspace(5, 20, 15),  # Ensure values > 20 exist
            'NO2': np.linspace(10, 25, 15)
        })
        
        cls.last_date = dates[-1]
        cls.days_to_predict = 5
        
        # Create test data with show_info=False
        cls.test_data = cls.analyser.load_and_merge_data(
            cls.df_weather, 
            cls.df_quality,
            ['temperature (C)', 'wind_speed (m/s)', 'precipitation (mm)'],
            ['PM10', 'NO2'],
            show_info=False
        )
        
        # Manually ensure lag features exist
        if 'PM10' in cls.test_data.columns:
            cls.test_data['PM10_lag_1'] = cls.test_data['PM10'].shift(1)
        
        # The following RandomForest setup was simplified for testing purposes with use of AI (Deepseek)
        # In production, more estimators and careful parameter tuning would be needed

        cls.model = RandomForestRegressor(n_estimators=10, random_state=42)
        cls.model.fit(
            cls.test_data[['temperature (C)', 'wind_speed (m/s)']].dropna(),
            cls.test_data['PM10'].dropna()
        )

    def test_load_and_merge_data(self):
        """Test data loading and merging"""

        # Test normal mode (used for training, should include lag features)
        normal_df = self.analyser.load_and_merge_data(
            self.df_weather.iloc[:10],
            self.df_quality.iloc[:10],
            ['temperature (C)', 'wind_speed (m/s)', 'precipitation (mm)'],
            ['PM10', 'NO2'],
            mode='train',
            show_info=False
        )

        self.assertIsInstance(normal_df, pd.DataFrame)
        self.assertGreater(len(normal_df), 0)
        
        # These should exist in training mode
        self.assertIn('PM10_lag_1', normal_df.columns)
        self.assertIn('spike_indicator', normal_df.columns)

        # --- Test test-mode merge without expecting lag/spike ---
        # Simulate test-data with renamed columns
        test_quality = self.df_quality.iloc[:10].copy()
        test_quality.columns = [col + '_test' if col != 'Date' else col for col in test_quality.columns]

        merged_test = self.analyser.load_and_merge_data(
            self.df_weather.iloc[:10],
            test_quality,
            ['temperature (C)', 'wind_speed (m/s)', 'precipitation (mm)'],
            ['PM10', 'NO2'],
            mode='test',
            show_info=False
        )

        self.assertIsInstance(merged_test, pd.DataFrame)
        self.assertGreater(len(merged_test), 0)
        
        # We do NOT expect lag or spike in test mode
        self.assertNotIn('PM10_lag_1', merged_test.columns)
        self.assertNotIn('spike_indicator', merged_test.columns)

        # But we do expect target columns to be renamed back to normal
        self.assertIn('PM10', merged_test.columns)



    def test_safe_fit(self):
        """Test model training with NaN values"""
        X = self.test_data[['temperature (C)', 'wind_speed (m/s)']]
        y = self.test_data['PM10'].copy()
        y.iloc[2:4] = np.nan
            

        # AI Declaration: The safe_fit method's NaN handling was suggested by AI (Deepseek)
        # to make training more robust against missing target values

        model = LinearRegression()
        fitted_model = self.analyser.safe_fit(model, X, y)
        self.assertTrue(hasattr(fitted_model, 'coef_'))
        self.assertEqual(len(fitted_model.coef_), 2)

    def test_create_model(self):
        """Test model pipeline creation"""
        model = self.analyser.create_model(degree=2)
        self.assertIsInstance(model, Pipeline)
        self.assertEqual(len(model.steps), 3)
        self.assertEqual(model.steps[0][0], 'polynomialfeatures')
        self.assertEqual(model.steps[2][0], 'linearregression')

    def test_train_model(self):
        """Test model training functionality"""
        # Skip if PM10 column is missing
        if 'PM10' not in self.test_data.columns:
            self.skipTest("PM10 column not available for testing")
            
        # Test with explicit features
        # The mutual_info_regression feature selection was AI-suggested
        # to automatically find relevant features

        model = self.analyser.train_model(
            self.test_data,
            'PM10',
            features=['temperature (C)', 'wind_speed (m/s)']
        )
        self.assertIsInstance(model, RandomForestRegressor)
        
        # Test with invalid target
    # Test with invalid target
        with self.assertRaises(KeyError):  # or ValueError if you modify the code
            self.analyser.train_model(self.test_data, 'NON_EXISTENT')

    def test_predict_future(self):
        """Test future predictions"""
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
        
        # Test invalid date handling
        with self.assertRaises(ValueError):
            self.analyser.predict_future(model, 'invalid_date', 5)
        
        # Test invalid days_to_predict
        with self.assertRaises(ValueError):
            self.analyser.predict_future(model, self.last_date, -1)

    def test_evaluate_model(self):
        """Test model evaluation"""
        # Skip if required columns are missing
        if 'PM10' not in self.test_data.columns:
            self.skipTest("PM10 column not available for testing")
            
        predictions, valid_data, mse, r2 = self.analyser.evaluate_model(
            self.model,
            self.test_data,
            'PM10',
            features=['temperature (C)', 'wind_speed (m/s)'],
            train_data=self.test_data
        )
        
        self.assertIsNotNone(predictions)
        self.assertIsInstance(valid_data, pd.DataFrame)
        self.assertGreaterEqual(len(valid_data), 0)
        self.assertIsInstance(mse, float)
        self.assertIsInstance(r2, float)
        
        # Test missing features
        with self.assertRaises(ValueError):
            self.analyser.evaluate_model(
                self.model,
                self.test_data,
                'PM10',
                features=['NON_EXISTENT_FEATURE']
            )
        
        # Test invalid target
        with self.assertRaises(ValueError):
            self.analyser.evaluate_model(
                self.model,
                self.test_data,
                'NON_EXISTENT_TARGET'
            )

if __name__ == '__main__':
    # Add verbosity=2 to see all test names and results
    unittest.main(verbosity=2)