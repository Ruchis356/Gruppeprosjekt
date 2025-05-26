Explains the test structure and how to run tests

# Data_handling:

This unit test suite verifies the missing_data method of the RefinedData class using Python’s unittest framework. It tests how the method handles missing values in a DataFrame with different strategies:

report: Returns a DataFrame showing the location of missing values.

drop: Removes rows with missing values.

fill: Replaces missing values with a specified value.

No missing values: Returns None.

This project provides a simple Python class `RefinedData` for handling missing values in Pandas DataFrames. It includes a unit test suite using Python's `unittest` framework.

- `datahandling.py`: Contains the `RefinedData` class and `missing_data()` method.
- `test_datahandling.py`: Unit tests to validate all method strategies.

The setUp method uses self to create reusable instance variables: a RefinedData processor and a sample DataFrame with missing values. Using self allows these to be accessed across all test methods, ensuring consistency and reducing code repetition.


# TESTING STRATEGY NOTE:
# - `VisualTable.pretty_data()` is intentionally untested because:
#   1. It is a thin presentation-layer wrapper around pandas/IPython (already stable).
#   2. Input validity is enforced by earlier pipeline stages (tested elsewhere).
#   3. Manual verification confirms display behavior in Jupyter/non-Jupyter contexts.

# Test_analysis:

    Averages: Tests weekly averages are correctly computed.
        Handles missing columns with a warning (NAN)
        Returns None and logs an error if no valid columns exist.
        Fails if the date column is invalid.
        Weeks with fewer than 3 valid values return NaN.
        Logs exception if grouping fails.

    Total Average: 
        Verifies correct computation of column-wise averages.
        Raises ValueError for invalid inputs or column names.
        Logs a warning if a calculation error occurs.

    Standard Deviation:
        Validates standard deviation calculation for columns.
        Handles missing columns with warnings and NaN values.
        Returns None and logs error if no valid columns.
        Raises ValueError for invalid DataFrame or column names.
        Catches and logs exceptions during calculation.

    Outliers:
        Detects and returns outliers based on average and std deviation.
        Replaces outlier values with NaN in the original DataFrame.
        Returns None if no outliers are found.
        Logs warnings for missing columns.
        Raises ValueError for invalid standard_deviation or average inputs.

# predictive_analysis

This unit test suite verifies the WeatherAnalyser class using Python's unittest framework. It tests core functionality for weather and pollution analysis:

load_and_merge_data: Combines weather and air quality data with feature engineering.

safe_fit: Handles training with missing values.

train_model: Builds models with automatic feature selection.

predict_future: Generates forecasts from trained models.

evaluate_model: Calculates performance metrics (MSE, R²).

The setUpClass method creates shared test data: weather measurements, pollution levels, and a pre-trained model. This ensures consistent test conditions across all methods while minimizing repetition




