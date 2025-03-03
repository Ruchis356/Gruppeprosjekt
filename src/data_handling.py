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
            # Iterate over the DataFrame to find exact locations
            for row, col in zip(*missing_values.to_numpy().nonzero()):
                missing_locations.append({'index': row, 'column': df.columns[col]})
            missing_df = pd.DataFrame(missing_locations) 
            return missing_df
        else:
            print("No missing values found in the data set! \n")
            return None


    # ------------------------------------------
    # WAIR QUALITY DATA - PROCESSING
    # ------------------------------------------

    # Check for datapoints = 0
    def show_zeroes(self, df):
        """
        Find all data points with a value of 0 in the specified columns and return their locations as a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to check for zero values.

        Returns:
            pd.DataFrame: A DataFrame with columns 'index' and 'column' indicating the locations of zero values.
                        Returns an empty DataFrame if no zero values are found.
        """

        # Specifying the coverage columns to check
        columns_to_check = ['Dekning', 'Dekning.1', 'Dekning.2', 'Dekning.3', 'Dekning.4']

        # Initialize an empty list to store zero value locations
        zero_locations = []

        # Iterate over the specified columns to find zero values
        for col in columns_to_check:

            # Find rows where the value in the current column is 0
            zero_indices = df.index[df[col] == 0].tolist()
            
            # Append the (index, column) pairs to the list
            for index in zero_indices:
                zero_locations.append({'index': index, 'column': col})

        # Convert the list of dictionaries to a DataFrame
        zero_df = pd.DataFrame(zero_locations)

        # If no zero values are found, print a message and return an empty DataFrame
        if zero_df.empty:
            print("No coverage was poor enough to be excluded.")
            return None

        else:
            return zero_df
