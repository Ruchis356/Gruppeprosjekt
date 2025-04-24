
import pandas as pd

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
        weekly_data[date_column] = weekly_dates 
        return pd.DataFrame(weekly_data)[[date_column] + valid_columns]


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


