# ------------------------------------------
# WEATHER DATA - PROCESSING
# ------------------------------------------

def missing_weather(df):

    # Check for missing values
    missing_values = df.isna()

    # If there are missing values, print their locations
    if missing_values.any().any():
        print("⚠️ Missing values found at these locations:")
    
        # Iterate over the DataFrame to find exact locations
        for row, col in zip(*missing_values.to_numpy().nonzero()):
            print(f"Missing value at Row {row}, Column '{df.columns[col]}'")
    else:
        print("No missing values found in the weather data! \n")






# ------------------------------------------
# WAIR QUALITY DATA - PROCESSING
# ------------------------------------------

import pandas as pd

def show_zeroes(df):

    # Specifying the coverage columns to check
    columns_to_check = ['Dekning', 'Dekning.1', 'Dekning.2', 'Dekning.3', 'Dekning.4']

    # Find all data points with a value of 0 in those columns
    zero_values = df[columns_to_check][df[columns_to_check] == 0]

    # Show the index numbers of the rows that contain a datapoint 0 in a coverage column
    rows_with_zero = zero_values.dropna(how='all')

    if not rows_with_zero.empty:
        return rows_with_zero.index

    else:
        # If there are no rows, return an empty index
        print("No coverage was poor enough to be excluded.")
        return pd.Index([])
