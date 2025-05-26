import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from predictive_analysis import WeatherAnalyser

class TestWeatherAnalyser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create test data that will be used across multiple tests"""
        cls.analyser = WeatherAnalyser()
        
        # Create sample weather data
        dates = pd.date_range(start='2023-01-01', periods=30)
        cls.df_weather = pd.DataFrame({
            'Date': dates,
            'temperature (C)': np.linspace(0, 15, 30),
            'wind_speed (m/s)': np.random.uniform(1, 5, 30),
            'precipitation (mm)': np.random.uniform(0, 10, 30)
        })
        
        # Create sample air quality data
        cls.df_quality = pd.DataFrame({
            'Date': dates,
            'PM10': np.random.uniform(5, 50, 30),
            'NO2': np.random.uniform(10, 60, 30),
            'NO': np.random.uniform(2, 30, 30)
        })
        
        # Create test data with missing values
        cls.df_quality_missing = cls.df_quality.copy()
        cls.df_quality_missing.loc[::5, 'PM10'] = np.nan
        
        # Weather variables for testing
        cls.weather_vars = ['temperature (C)', 'wind_speed (m/s)', 'precipitation (mm)']
        cls.pollutant_vars = ['PM10', 'NO2', 'NO']

    def test_load_and_merge_data_basic(self):
        """Test basic merging functionality"""
        merged = self.analyser.load_and_merge_data(
            self.df_weather, 
            self.df_quality,
            self.weather_vars,
            self.pollutant_vars
        )
        
        # Check basic shape and columns
        self.assertEqual(merged.shape[0], 30)
        self.assertIn('PM10', merged.columns)
        self.assertIn('temperature (C)', merged.columns)
        self.assertIn('DayOfYear', merged.columns)
        
        # Check derived features
        self.assertIn('DayOfYear_sin', merged.columns)
        self.assertIn('DayOfYear_cos', merged.columns)
        self.assertIn('Weekend', merged.columns)
        self.assertIn('Season', merged.columns)
        
        # Check interaction terms
        self.assertIn('temp_wind_interaction', merged.columns)
        self.assertIn('temp_precip_interaction', merged.columns)
        
        # Check sorting
        self.assertTrue(merged['Date'].is_monotonic_increasing)

    def test_load_and_merge_data_missing_columns(self):
        """Test handling of missing columns"""
        # Remove one column from each dataframe
        df_weather_missing = self.df_weather.drop(columns=['precipitation (mm)'])
        df_quality_missing = self.df_quality.drop(columns=['NO'])
        
        merged = self.analyser.load_and_merge_data(
            df_weather_missing, 
            df_quality_missing,
            self.weather_vars,
            self.pollutant_vars
        )
        
        # Should still merge successfully with available columns
        self.assertIn('temperature (C)', merged.columns)
        self.assertIn('PM10', merged.columns)
        self.assertNotIn('precipitation (mm)', merged.columns)
        self.assertNotIn('NO', merged.columns)

    def test_load_and_merge_data_missing_date_column(self):
        """Test error handling for missing date column"""
        df_weather_no_date = self.df_weather.drop(columns=['Date'])
        
        with self.assertRaises(ValueError):
            self.analyser.load_and_merge_data(
                df_weather_no_date,
                self.df_quality,
                self.weather_vars,
                self.pollutant_vars
            )

    def test_load_and_merge_data_test_mode(self):
        """Test handling of test data with suffix"""
        df_quality_test = self.df_quality.copy()
        df_quality_test.columns = [f"{col}_test" if col != 'Date' else col for col in df_quality_test.columns]
        
        merged = self.analyser.load_and_merge_data(
            self.df_weather,
            df_quality_test,
            self.weather_vars,
            [f"{col}_test" for col in self.pollutant_vars],
            mode='test'
        )
        
        # Should have original column names after processing
        self.assertIn('PM10', merged.columns)
        self.assertNotIn('PM10_test', merged.columns)

    def test_safe_fit(self):
        """Test safe_fit with NaN values in target"""
        X = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
        y = pd.Series([10, 20, np.nan, 40, 50])
        
        model = RandomForestRegressor(random_state=42)
        fitted_model = self.analyser.safe_fit(model, X, y)
        
        # Should fit on 4 rows (excluding the NaN)
        self.assertEqual(len(fitted_model.predict(X)), 5)

    def test_create_model(self):
        """Test model creation pipeline"""
        model = self.analyser.create_model(degree=3)
        
        self.assertIsInstance(model, Pipeline)
        self.assertEqual(model.steps[0][1].degree, 3)
        self.assertIsInstance(model.steps[1][1], SimpleImputer)
        self.assertIsInstance(model.steps[2][1], LinearRegression)

    def test_train_model_default_features(self):
        """Test training with default feature selection"""
        merged = self.analyser.load_and_merge_data(
            self.df_weather,
            self.df_quality,
            self.weather_vars,
            self.pollutant_vars
        )
        
        model = self.analyser.train_model(merged, 'PM10')
        
        self.assertIsInstance(model, RandomForestRegressor)
        self.assertTrue(hasattr(model, 'feature_names_in_'))
        self.assertLessEqual(len(model.feature_names_in_), 5)  # Should select top 5 features

    def test_train_model_custom_features(self):
        """Test training with specified features"""
        merged = self.analyser.load_and_merge_data(
            self.df_weather,
            self.df_quality,
            self.weather_vars,
            self.pollutant_vars
        )
        
        features = ['temperature (C)', 'wind_speed (m/s)', 'DayOfYear_sin']
        model = self.analyser.train_model(merged, 'PM10', features=features)
        
        self.assertIsInstance(model, RandomForestRegressor)
        self.assertEqual(list(model.feature_names_in_), features)

    def test_train_model_missing_target(self):
        """Test error handling for missing target column"""
        merged = self.analyser.load_and_merge_data(
            self.df_weather,
            self.df_quality,
            self.weather_vars,
            self.pollutant_vars
        )
        
        with self.assertRaises(ValueError):
            self.analyser.train_model(merged, 'NonExistentColumn')

    def test_predict_future(self):
        """Test future prediction generation"""
        merged = self.analyser.load_and_merge_data(
            self.df_weather,
            self.df_quality,
            self.weather_vars,
            self.pollutant_vars
        )
        
        model = self.analyser.train_model(merged, 'PM10', features=['DayOfYear'])
        future = self.analyser.predict_future(model, merged['Date'].iloc[-1], 7)
        
        self.assertEqual(len(future), 7)
        self.assertIn('Prediction', future.columns)
        self.assertTrue((future['Date'] > merged['Date'].iloc[-1]).all())

    def test_predict_future_invalid_input(self):
        """Test error handling for invalid future prediction inputs"""
        merged = self.analyser.load_and_merge_data(
            self.df_weather,
            self.df_quality,
            self.weather_vars,
            self.pollutant_vars
        )
        
        model = self.analyser.train_model(merged, 'PM10')
        
        with self.assertRaises(ValueError):
            # Invalid date format
            self.analyser.predict_future(model, 'not-a-date', 7)
            
        with self.assertRaises(ValueError):
            # Invalid days_to_predict
            self.analyser.predict_future(model, merged['Date'].iloc[-1], -1)

    def test_evaluate_model(self):
        """Test model evaluation"""
        merged = self.analyser.load_and_merge_data(
            self.df_weather,
            self.df_quality,
            self.weather_vars,
            self.pollutant_vars
        )
        
        # Split into train/test (simple split for testing)
        train = merged.iloc[:20]
        test = merged.iloc[20:]
        
        model = self.analyser.train_model(train, 'PM10')
        predictions, valid_data, mse, r2 = self.analyser.evaluate_model(
            model, test, 'PM10', train_data=train
        )
        
        self.assertIsNotNone(predictions)
        self.assertIsNotNone(valid_data)
        self.assertIsInstance(mse, float)
        self.assertIsInstance(r2, float)
        self.assertEqual(len(predictions), len(valid_data))

    def test_evaluate_model_missing_data(self):
        """Test evaluation with missing target values"""
        merged = self.analyser.load_and_merge_data(
            self.df_weather,
            self.df_quality_missing,
            self.weather_vars,
            self.pollutant_vars
        )
        
        # Split into train/test
        train = merged.iloc[:20]
        test = merged.iloc[20:]
        
        model = self.analyser.train_model(train, 'PM10')
        predictions, valid_data, mse, r2 = self.analyser.evaluate_model(
            model, test, 'PM10', train_data=train
        )
        
        # Should still work, just with fewer rows
        self.assertLess(len(valid_data), len(test))

    def test_evaluate_model_empty_test(self):
        """Test handling of empty test data"""
        merged = self.analyser.load_and_merge_data(
            self.df_weather,
            self.df_quality,
            self.weather_vars,
            self.pollutant_vars
        )
        
        model = self.analyser.train_model(merged, 'PM10')
        
        with self.assertRaises(ValueError):
            self.analyser.evaluate_model(model, pd.DataFrame(), 'PM10')

if __name__ == '__main__':
    unittest.main()