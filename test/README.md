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