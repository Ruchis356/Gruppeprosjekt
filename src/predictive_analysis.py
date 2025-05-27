
__all__ = ['WeatherAnalyser']  

import pandas as pd
import numpy as np
from tqdm import tqdm  
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression
import logging # The use of logging was suggested by AI (DeepSeek)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


class WeatherAnalyser:
    """Analyzes and predicts weather/pollution using machine learning.
    
    Key Functionality:
    - Data loading and merging
    - Feature engineering
    - Model training (Random Forest)
    - Forecasting with lag features
    - Model evaluation
    
    Note: Uses scikit-learn models with consistent logging throughout.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_weather_data(self, df_weather, weather_vars, show_info=False):
        """Load only weather data without merging with pollutants"""
        
        # Input validation
        if 'Date' not in df_weather.columns:
            raise ValueError("df_weather must contain a 'Date' column")
        
        # Filter available columns
        available_weather = [col for col in weather_vars if col in df_weather.columns]
        df = df_weather[['Date'] + available_weather].copy()
        
        # Add temporal features
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear']/365)
        df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear']/365)
        
        return df

    def load_and_merge_data(self, df_weather, df_quality, weather_vars, pollutant_vars, show_info, mode='train'):
        """
        Merges weather and air quality dataframes with robust column handling.
        
        Args:
            df_weather: DataFrame with weather data (must contain 'Date' column)
            df_quality: DataFrame with air quality data (must contain 'Date' column)
            weather_vars: List of weather columns to include
            pollutant_vars: List of pollutant columns to include
            show_info: Whether to log processing information
            mode: 'train' or 'test' (affects column name handling)
            
        Returns:
            Merged DataFrame with engineered features
        """
        # Input validation
        if 'Date' not in df_weather.columns:
            raise ValueError("df_weather must contain a 'Date' column")
        if 'Date' not in df_quality.columns:
            raise ValueError("df_quality must contain a 'Date' column")

        # Handle test data column renaming
        if mode == 'test':
            # Create mapping for columns that match requested pollutants when _test is removed
            rename_map = {
                col: col.replace('_test', '')
                for col in df_quality.columns
                if '_test' in col and col.replace('_test', '') in pollutant_vars
            }
            df_quality = df_quality.rename(columns=rename_map)
            
            # Log any columns that couldn't be renamed
            if show_info and rename_map:
                self.logger.info(f"Renamed test columns: {rename_map}")

        # Find available columns in both datasets
        available_weather = [col for col in weather_vars if col in df_weather.columns]
        available_pollutants = [col for col in pollutant_vars if col in df_quality.columns]
        
        # Log missing columns if requested
        if show_info:
            missing_weather = set(weather_vars) - set(available_weather)
            missing_pollutants = set(pollutant_vars) - set(available_pollutants)
            
            if missing_weather:
                self.logger.info(f"Missing weather columns: {missing_weather}")
            if missing_pollutants:
                self.logger.warning(f"Missing pollutant columns: {missing_pollutants}")

        # Merge the datasets
        try:
            merged = pd.merge(
                df_weather[['Date'] + available_weather],
                df_quality[['Date'] + available_pollutants],
                on='Date',
                how='inner'
            )
        except KeyError as e:
            raise ValueError(f"Merge failed due to missing columns: {str(e)}")

        # Early return if no data after merge
        if merged.empty:
            self.logger.warning("Empty DataFrame after merge - check input data")
            return merged

        # --- Basic temporal features ---
        merged['Date'] = pd.to_datetime(merged['Date'])
        merged['DayOfYear'] = merged['Date'].dt.dayofyear
        merged['DayOfYear_sin'] = np.sin(2 * np.pi * merged['DayOfYear']/365)
        merged['DayOfYear_cos'] = np.cos(2 * np.pi * merged['DayOfYear']/365)
        merged['Weekend'] = merged['Date'].dt.weekday >= 5
        merged['Season'] = merged['Date'].dt.month % 12 // 3 + 1
        merged['weekend_effect'] = merged['Weekend'] * merged['DayOfYear_sin']
        merged = merged.sort_values('Date')

        # --- Interaction features ---
        weather_cols_present = [col for col in weather_vars if col in merged.columns]
        for i, var1 in enumerate(weather_cols_present):
            for var2 in weather_cols_present[i+1:]:
                base_name1 = var1.split()[0]
                base_name2 = var2.split()[0]
                merged[f'{base_name1}_{base_name2}_interaction'] = merged[var1] * merged[var2]

        # --- Rolling features ---
        rolling_windows = [('temperature (C)', 7), ('wind_speed (m/s)', 7), ('precipitation (mm)', 7)]
        for var, window in rolling_windows:
            if var in merged.columns:
                merged[f'rolling_{var.split()[0]}_{window}'] = (
                    merged[var].rolling(window, min_periods=1).mean()
                )

        # --- Lag features ---
        for lag in [1, 2, 3]:
            for var in weather_cols_present + available_pollutants:
                if var in merged.columns:
                    merged[f'{var}_lag_{lag}'] = merged[var].shift(lag)

        # --- Additional pollutant-specific features ---
        for pollutant in available_pollutants:
            if pollutant in merged.columns:
                # 14-day lag
                merged[f'{pollutant}_lag_14'] = merged[pollutant].shift(14)
                # 30-day rolling average
                merged[f'rolling_{pollutant}_30'] = (
                    merged[pollutant].rolling(30, min_periods=1).mean()
                )

        # --- Cleanup ---
        # Drop rows where essential weather vars are missing
        merged = merged.dropna(subset=available_weather)
        
        if show_info:
            self.logger.info(
                f"Merged data shape: {merged.shape}\n"
                f"Available weather vars: {available_weather}\n"
                f"Available pollutant vars: {available_pollutants}"
            )
        
        return merged
    
    def safe_fit(self, model, X, y):
        """
        Safely fits a model by dropping rows where the target (y) has NaN values.
        
        Args:
            model: A scikit-learn compatible model with a `fit` method.
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
            
        Returns:
            The fitted model.

        Example:
            >>> model = LinearRegression()
            >>> safe_fit(model, X_train, y_train)  # Drops rows where y_train is NaN
        """

        # The following block of code was generated by AI
            # Purpose: Dealin with NaN values that were causing trouble in a more efficient manner
            # AI Tool: DeepSeek
    
        valid_mask = y.notna()
        return model.fit(X[valid_mask], y[valid_mask])

    def create_model(self, degree=4):

        """
        Creates a polynomial regression pipeline with imputation.
    
        Args:
            degree (int): Degree of polynomial features (default=4).
            
        Returns:
            sklearn.pipeline.Pipeline: A pipeline with:
                - PolynomialFeatures
                - SimpleImputer (mean strategy)
                - LinearRegression
        """

        return make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            SimpleImputer(strategy='mean'),
            LinearRegression()
        )
    
    def train_model(self, data, target, features=None):

        """
        Trains a model with the given data using the randomforestregressor

        Args:
            data(pd.DataFrame): Combined weather and air quality data
            target (str): Target variable name 
            features (list): List of feature column names (default=None utilises a default list)

        Returns:
            sklearn.ensemble.RandomForestRegressor: Trained model.
        """

        # The following 2 blocks of code was developed with assistance from AI
            # Purpose: Creating a functional method of dealing with features if none are provided
            # AI Tool: DeepSeek

        # Set default features if None
        potential_features = [
            'temperature (C)', 'wind_speed (m/s)', 
            'precipitation (mm)', 'DayOfYear_sin',
            'DayOfYear_cos', 'Weekend', 'Season'
        ]

        # Only calculate mutual info if features is None
        if features is None:
            self.logger.info("No features provided. Using mutual information for selection.")

            # Filter to only available features
            available_features = [f for f in potential_features if f in data.columns]
            
            # Calculate mutual information
            mi = mutual_info_regression(
                data[available_features].fillna(data[available_features].median()),
                data[target].fillna(data[target].median())
            )
            # Get top 5 features
            features = [f for _, f in sorted(zip(mi, available_features), reverse=True)][:5]
            self.logger.info(f"Selected features for {target}: {features}")

        # Input validation
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        valid_data = data[data[target].notna()]
        if len(valid_data) == 0:
            raise ValueError(f"No valid rows remaining for target '{target}'")

        # Use of Train Random Forest model was suggested by AI for a better prediction (Deepseek)

        # Model configuration
        if target in ('NO2', 'NO'):
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=3,
                max_features=0.5,
                random_state=42,
                n_jobs=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=2,
                max_features=0.8,
                random_state=42,
                n_jobs=-1
            )

        model.fit(valid_data[features], valid_data[target])

        return model

    def predict_future(self, model, last_date, days_to_predict):

        """
        Generate future predictions using a simplistic prediction method with a trained model

        Args:
            model: Trained regression model
            last_date: Last date of historical data
            days_to_predict: Number of days to predict

        Returns: 
            pd.Dataframe: DataFrame with columns:
                - 'Date': Future dates.
                - 'DayOfYear': Day of the year (1-365).
                - 'Prediction': Model predictions.
        """

        # Input validation    
        try:
            last_date = pd.to_datetime(last_date) 
        except ValueError:
            raise ValueError("last_date must be a parsable date")
        
        if not isinstance(days_to_predict, int) or days_to_predict <= 0:
            raise ValueError("days_to_predict must be a positive integer")
        
        # Check model is fitted
        if not hasattr(model, 'predict'):
            raise ValueError("Provided model must have a predict method")

        # Create continuous date range starting day after last_date
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days_to_predict,
            freq='D'
        )
        
        # Create DataFrame with DayOfYear
        future_df = pd.DataFrame({
            'Date': future_dates,
            'DayOfYear': future_dates.dayofyear
        })

        # Model validation and prediction
        if not hasattr(model, 'predict'):
            raise ValueError("Model must have predict method")
        
        # Maintain your existing feature warning
        if hasattr(model, 'feature_names_in_') and 'DayOfYear' not in model.feature_names_in_:
            self.logger.warning(
                "Model expects features: %s. Predictions may be unreliable "
                "as only 'DayOfYear' is provided.", 
                model.feature_names_in_
            )
        
        future_df['Prediction'] = model.predict(future_df[['DayOfYear']])

        return future_df

    def evaluate_model(self, model, test_data, target, features=None, train_data=None):

        """
        Evaluaes the model against the test data
        
        Args:
            model: Trained model
            test_data (pd.DataFrame): Combined weather and air quality data
            target (str): Target variable name 
            features (list): List of feature column names
            train_data: Training data used for median imputation 

        Returns:
            Tuple: (predictions, valid_data, mse, r2)  
                - predictions (pd.DataFrame): Array of model predictions
                - valid_data (pd.DataFrame): Filtered(NaN) DataFrame used for evaluation 
                - mse (float):  Mean squared error for the model
                - r2 (float): R² score for the model
            Returns tuple(None, None, None, None) if no valid rows remain after NaN filtering

        Note: The returns were generated by AI and edited by developers
        """

        # The following block of code was developed with assistance from AI
            # Purpose: Deciding which edge cases to run through input validation
            # AI Tool: DeepSeek

        # Input validation
        if test_data.empty:
            raise ValueError("test_data is empty")
        if features is None:
            features = getattr(model, 'feature_names_in_', None)
            if features is None:
                raise ValueError("No features provided and model has no feature_names_in_ attribute")
        if target not in test_data.columns:
            raise ValueError(f"Target column '{target}' not found in data")
        missing_features = [f for f in features if f not in test_data.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Create explicit copies with .copy() to avoid SettingWithCopyWarning suggested AI (Deepseek)

        # Handle missing features by imputing with training median
        missing_features = set(features) - set(test_data.columns)
        if missing_features:
            self.logger.warning(f"Imputing missing features with training median: {missing_features}")
            if train_data is None:
                raise ValueError(f"Missing features {missing_features} and no train_data provided for imputation")
            test_data = test_data.copy()
            for f in missing_features:
                test_data[f] = train_data[f].median()

        valid_mask = test_data[target].notna() & test_data[features].notna().all(axis=1)
        valid_data = test_data.loc[valid_mask].copy()

        if len(valid_data) == 0:
            self.logger.warning(f"No valid test rows for target '{target}'")
            return None, None, None, None

        predictions = model.predict(valid_data[features])
        mse = mean_squared_error(valid_data[target], predictions)
        r2 = r2_score(valid_data[target], predictions)

        return predictions, valid_data, mse, r2
    
    def forecast_pollutants_with_lags(self, weather_forecast, models_dict, features_dict, 
                                    historical_data, n_days=7):
        """
        Forecast pollutant levels with proper handling of lag features
        
        Args:
            weather_forecast: DataFrame with future weather predictions
            models_dict: Dictionary of trained models
            features_dict: Dictionary of feature lists
            historical_data: DataFrame with recent historical data (for initial lags)
            n_days: Number of days to forecast
            
        Returns:
            Dictionary of DataFrames with forecasts for each pollutant
        """

        # The following block of code was generated by AI
            # Purpose: Including parameter validation
            # AI Tool: DeepSeek
        
        if not isinstance(weather_forecast, pd.DataFrame):
            raise ValueError("weather_forecast must be a DataFrame")
        if 'Date' not in weather_forecast.columns:
            raise ValueError("weather_forecast must contain 'Date' column")

        forecasts = {}
        required_weather = ['temperature (C)', 'precipitation (mm)', 'wind_speed (m/s)']
        
        # Prepare working copy with historical data for lags
        full_df = pd.concat([historical_data, weather_forecast], axis=0).reset_index(drop=True)
        full_df = full_df.sort_values('Date').reset_index(drop=True)
        
        # For each pollutant, generate recursive forecasts
        for pollutant, model in tqdm(models_dict.items(), desc="Forecasting pollutants"):
            features = features_dict[pollutant]
            result = pd.DataFrame()
            
            # Initialize with historical data
            working_df = full_df.copy()
            
            # Get the max lag we need (e.g., lag_14 means we need 14 days history)
            max_lag = max([int(f.split('_')[-1]) for f in features 
                        if f.startswith(f"{pollutant}_lag_") and f.split('_')[-1].isdigit()], 
                        default=0)
            
            # Only forecast days where we have weather data
            forecast_start_idx = len(historical_data)
            forecast_end_idx = min(forecast_start_idx + n_days, len(working_df))
            
            # Recursive prediction day by day
            for i in range(forecast_start_idx, forecast_end_idx):
                # Prepare features for this day
                current_features = {}
                
                for feature in features:
                    # Handle lag features
                    if feature.startswith(f"{pollutant}_lag_"):
                        lag_days = int(feature.split('_')[-1])
                        if i - lag_days >= 0:
                            current_features[feature] = working_df.at[i - lag_days, pollutant]
                    
                    # Handle rolling features
                    elif feature.startswith(f"rolling_{pollutant}_"):
                        window = int(feature.split('_')[-1])
                        lookback = working_df.iloc[max(0, i-window):i][pollutant]
                        if len(lookback) > 0:
                            current_features[feature] = lookback.mean()
                    
                    # Handle weather features
                    elif feature in required_weather:
                        current_features[feature] = working_df.at[i, feature]
                    
                    # Handle other features (interactions, etc.)
                    else:
                        try:
                            current_features[feature] = working_df.at[i, feature]
                        except KeyError:
                            pass  # Will be handled by model
                
                # Convert to DataFrame for prediction
                feature_df = pd.DataFrame([current_features])
                
                # Make prediction (handle missing features)
                available_features = [f for f in features if f in feature_df.columns]
                if not available_features:
                    raise ValueError(f"No available features for {pollutant} on day {i}")
                
                prediction = model.predict(feature_df[available_features])[0]
                
                # Store prediction and update working DataFrame
                working_df.at[i, pollutant] = prediction
                date = working_df.at[i, 'Date']
                
                # Store results
                result = pd.concat([result, pd.DataFrame({
                    'Date': [date],
                    f'{pollutant}_forecast': [prediction]
                })])
            
            forecasts[pollutant] = result.sort_values('Date')
        
        return forecasts