import pandas as pd

class RefinedData:

    # ------------------------------------------
    # PROCESSING MISSING DATA
    # ------------------------------------------

    # Check for missing data points and return a list
    def missing_data(self, df):

        """
        Args:
            df (pd.DataFrame): The DataFrame to check for missing values.
        """

        # Check for missing values
        missing_values = df.isna()

        # Initialize an empty list to store missing value locations
        missing_locations = []

        # If there are missing values, print their locations
        if missing_values.any().any():
            print("These are the locations of the missing data:\n")
            # Iterate over the DataFrame to find exact locations
            for row, col in zip(*missing_values.to_numpy().nonzero()):
                missing_locations.append((row, df.columns[col]))
                return missing_locations
        else:
            print("No missing values found in the data set! \n")
            return None


    # ------------------------------------------
    # WAIR QUALITY DATA - PROCESSING
    # ------------------------------------------

    # Check for datapoints = 0
    def show_zeroes(self, df):

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
