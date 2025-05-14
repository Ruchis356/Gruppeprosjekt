
import pandas as pd
import numpy as np

class AnalysedData:

    def __init__(self):
        self.df = None

    # ------------------------------------------
    # CALCULATING RUNNING AVERAGES
    # ------------------------------------------

    """
    Calculates weekly averages for specified columns and returns a DataFrame.
    
    Args:
        df: Input DataFrame
        column_names: List of column names to analyze
            
    Returns:
        pd.DataFrame: Date colum + all valid input columns
        None: If no valid columns exist
    """

    def averages(self, df, column_names):

        weekly_data = {}

        # Checking for missing or invalid columns to exclude them from the calculations

        # This construct has been refined by the use of AI
        # Purpose: Providing a more efficient way of printing weekly averages for weeks with missing data, by using commands such as "if missing_cols := set(column_names) - set(valid_columns):"
        # Tool: ChatGPT

        valid_columns = [col for col in column_names if col in df.columns]
        if missing_cols := set(column_names) - set(valid_columns):
            print(f"Warning: Column(s) '{missing_cols}' not found in DataFrame. Skipping...")

        if not valid_columns:
            print("Error: No valid columns to process.")
            return None
        
        # Collect and store the first date of each week
        date_column = df.columns[0]
        weekly_dates = []

        # Calculating weekly averages for each column
        for column in valid_columns:
            weekly_averages = []
            for start in range(0, len(df), 7):
                df_week = df.iloc[start:start+7]

                # Creating column for datetime
                if column == valid_columns[0]: 
                    weekly_dates.append(
                        df_week[date_column].iloc[0] 
                        if not df_week.empty 
                        else None
                    )

                # Computing averages for the given column
                avg_value = df_week[column].mean(skipna=True)
                weekly_averages.append(
                    round(avg_value, 2) 
                    if not pd.isna(avg_value) 
                    else None
                )

            weekly_data[column] = weekly_averages               

        # Returns the dataframe with Date and all valid columns
        # This construct was refined with the assistance of AI 
        # Use of AI was implemented to suggest how adaptions are made to dataframes by new code 
        # Source: ChatGPT

        weekly_data[date_column] = weekly_dates 
        return pd.DataFrame(weekly_data)[[date_column] + valid_columns]
    
    # ------------------------------------------
    # CALCULATING TOTAL AVERAGES
    # ------------------------------------------

    """
    Calculates averages for specified columns and returns a DataFrame.
    
    Args:
        df: Input DataFrame
        column_names: List of column names to analyze
            
    Returns:
        pd.DataFrame: Two columns ["Metric", "Average"]
            Rows correspond to input columns.
    """
# This construct was refined with the assistance of AI 
        # Use of AI was implemented to pinpoint an explain error messages that were received while running the code. 
        # Source: ChatGPT

    def total_average(self, df, column_names):

        averages = []

        # Checking for missing or invalid columns to exclude them from the calculations
        valid_columns = [col for col in column_names if col in df.columns]
        if missing_cols := set(column_names) - set(valid_columns):
            print(f"Warning: Column(s) '{missing_cols}' not found in DataFrame. Skipping...")

        if not valid_columns:
            print("Error: No valid columns to process.")
            return None

        # Calculate average for each column and returning a list
        for col in column_names:
            if col in df.columns:
                average = df[col].mean()
                averages.append({"Metric": col, "Average": average})
            else:
                print(f"Column '{col}' not found in the DataFrame. Skipping...")
                averages.append({"Metric": col, "Average": None})
        
        return pd.DataFrame(averages)


    # ------------------------------------------
    # CALCULATING STANDARD DEVIATION
    # ------------------------------------------

    """
    Calculates standard deviations for specified columns and returns a DataFrame.
    
    Args:
        df: Input DataFrame
        column_names: List of column names to analyze
            
    Returns:
        pd.DataFrame: Two columns ["Metric", "Standard Deviation"]
            Rows correspond to input columns.
    """

        # This construct was refined with the assistance of AI 
        # Use of AI was implemented to pinpoint an explain error messages that were received while running the code. 
        # Source: ChatGPT

    def standard_deviation(self, df, column_names):

        standard_deviations = []

        # Calculate standard deviation for each column and returning a list
        for col in column_names:
            if col in df.columns:
                std_val = df[col].std()
                standard_deviations.append({"Metric": col, "Standard Deviation": std_val})
            else:
                print(f"Column '{col}' not found in the DataFrame. Skipping...")
                standard_deviations.append({"Metric": col, "Standard Deviation": None})
        
        return pd.DataFrame(standard_deviations)
    
    # ------------------------------------------
    # IDENTIFYING OUTLIERS
    # ------------------------------------------

    """
    Identifies outliers in the given dataset with the given standard deviation
    
    Args:
        df: Input DataFrame
        column_name: The column of data to be analysed
        standard_devition: The standard deviation belonging to the column of data
            
    Returns:
        pd.DataFrame: Date column + datapoints that are outliers
        None: If no outliers are found, or invalid data is input
    """

     # This construct was refined with the assistance of AI 
        # Use of AI was implemented to suggest shorter code lines for calling standard deviation and average values from data frames. 
        # Use of AI was implemented to suggest and explain error codes received while running the code. 
        # Source: ChatGPT
    def outliers(self, df, column_names, standard_deviation, average, sd_modifier):    

        date_column = df.columns[0]
        numeric_columns = [col for col in column_names if col != date_column]

        outlier_table = {}
        df_x_outliers = df.copy()

        for col in numeric_columns:

            # Get the standard deviation for the current column
            std_row = standard_deviation[standard_deviation.iloc[:, 0] == col]
            col_std = std_row.iloc[0, 1] if not std_row.empty else 0  # Default to 0 if not found
            
            # Get the average for the current column
            avg_row = average[average.iloc[:, 0] == col]
            col_avg = avg_row.iloc[0, 1] if not avg_row.empty else 0  # Default to 0 if not found

            # Calculate bounds and find outliers
            lower_bound = col_avg - col_std*sd_modifier
            upper_bound = col_avg + col_std*sd_modifier
            column_outliers = df[col][(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_table[col] = column_outliers.values.tolist()

            # Mark outliers as NaN
            df_x_outliers.loc[column_outliers.index, col] = np.nan
        
#        outliers_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in outlier_table.items()]))

        # Convert to DataFrame and add dates
        outliers_df = pd.DataFrame(outlier_table)
        outliers_df[date_column] = df.loc[outliers_df.index, date_column]
        
        # Reorder to put date column first
        cols = [date_column] + numeric_columns
        outliers_df = outliers_df[cols]

        return outliers_df, df_x_outliers


'''

    def outliers(self, df, column_names, standard_deviation):    

        outlier_table = {}
        df_x_outliers = ()

        for col in column_names:
            Q1 = df[col].quantile(0.25, 0, True, 'interpolation')
            Q3 = df[col].quantile(0.75, 0, True, 'Interpolation')
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            column_outliers = df[col][(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_table[col] = column_outliers.values.tolist()
        
        outliers_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in outlier_table.items()]))

        for col in column_names:
            df_x_outliers.loc[outlier_table.index, col] = np.nan

        return outliers_df, df_x_outliers

'''