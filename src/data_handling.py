import pandas as pd

class RefinedData:

    # ------------------------------------------
    # PROCESSING MISSING DATA
    # ------------------------------------------

    # Check for missing data points and return a list
    def missing_data(self, df, strategy='report', fill_value=None):

        """
        Handle missing values in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to check for missing values.
            strategy (str): How to handle missing values ('report', 'drop', or 'fill').
            fill_value: The value to use when strategy is 'fill'.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """

        # Check for missing values
        missing_values = df.isna()

        # Initialize an empty list to store missing value locations
        missing_locations = []

        if missing_values.any().any():
            # Iterate over the DataFrame to find exact locations
            for row, col in zip(*missing_values.to_numpy().nonzero()):
                missing_locations.append({'index': row, 'column': df.columns[col]})
            missing_df = pd.DataFrame(missing_locations) 

            # Handle missing values based on the strategy
            if strategy == 'report':
                print("Missing values found in the data set! \n")
                return missing_df
            elif strategy == 'drop':
                print("Dropping rows with missing values...")
                return df.dropna()
            elif strategy == 'fill':
                print(f"Filling missing values with {fill_value}...")
                return df.fillna(fill_value)
        else:
            print("No missing values found in the data set! \n")
            return df  # Return the original DataFrame if no missing values are found

        # If there are missing values, print their locations
        if missing_df.empty:
            print("No missing values found in the data set! \n")
            return None
        else:
            return missing_df



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
            Returns an None if no zero values are found.
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
