
__all__ = ['AnalysedData']  

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging # The use of logging was suggested by AI (DeepSeek)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class AnalysedData:
    """A class for statistical analysis of environmental data.
    
    Key Functionality:
    - Temporal aggregation (weekly averages)
    - Descriptive statistics (mean, standard deviation)
    - Outlier detection
    - Visualization (box plots)
    
    Note: All methods use consistent logging and error handling patterns.
    """

    def __init__(self):
        self.df = None # Stores the most recently processed dataset

    # ------------------------------------------
    # CALCULATING RUNNING AVERAGES
    # ------------------------------------------

    # The following function has been refined with the assistance of AI
        # Purpose: Providing a more efficient way of printing weekly averages for weeks with missing data, 
            # - eg. by using commands such as "if missing_cols := set(column_names) - set(valid_columns):"
            # Suggesting ways of building weekly average tables for multiple pollutants using shorter codelines
        # AI Tool: ChatGPT

    def averages(self, df, column_names):

        """
        Calculates weekly averages for specified columns and returns a DataFrame.
        
        Args:
            df: Input DataFrame
            column_names: List of column names to analyze
                
        Returns:
            pd.DataFrame: Weekly averages with date column first
            None: If no valid columns exist
        """

        #  Validate input columns
        valid_columns = [col for col in column_names if col in df.columns]
        if missing_cols := set(column_names) - set(valid_columns):
            logger.warning("Column(s) %s not found in DataFrame. Skipping...", missing_cols)

        if not valid_columns:
            logger.error("No valid columns to process")
            return None
        
        date_column = df.columns[0]
        
        # Ensure date column is datetime type
        try:
            df_date = pd.to_datetime(df[date_column], format='%Y-%m-%d')
        except Exception as e:
            logger.error("Could not parse date column: %s", str(e))
            return None

        # The following block of code was refined with the assistance of AI 
            # Purpose: Using certain construct, eg. ".dropna()", for calcualting weekly averages with missing weekly data.
            # AI Tool: ChatGPT   

        # Calculating weekly averages for each column
        try:
            df = df.copy()
            df['days_since_start'] = (df_date - df_date.min()).dt.days
            df['week_group'] = df['days_since_start'] // 7  
            
            weekly_avg = (
                df[valid_columns]
                .groupby(df['week_group'])
                .agg(lambda x: x.dropna().mean() if len(x.dropna()) >= 3 else None)
                .reset_index()
            )
            
            # Get first date of each week
            weekly_dates = (
                df[[date_column, 'week_group']]
                .groupby('week_group')
                .first()
                .reset_index()[date_column]
            )
            
            # The following block of code was refined with the assistance of AI 
                # AI helped optimize how we construct and return the final DataFrame by suggesting this specific pandas pattern
                # AI Tool: ChatGPT

            # Returns the dataframe with Date and all valid columns
            weekly_avg[date_column] = weekly_dates
            return weekly_avg[[date_column] + valid_columns]
        
        except Exception as e:
            logger.error("Weekly average calculation failed: %s", str(e))
            return None
    
    # ------------------------------------------
    # CALCULATING TOTAL AVERAGES (MEAN)
    # ------------------------------------------

    def total_average(self, df, column_names):

        """
        Calculates averages for specified columns and returns a DataFrame.
        
        Args:
            df: Input DataFrame
            column_names: List of column names to analyze
                
        Returns:
            pd.DataFrame: Two columns ["Metric", "Average"]
            None: If no valid columns exist
        """

        # Input validation
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if not isinstance(column_names, (list, tuple)):
            raise ValueError("column_names must be a list or tuple")

        results = []
        has_valid_results = False

        # The following block of code was improved with assistance from AI
            # Purpose: vectorization for performance
            # AI Tool: DeepSeek

        # Calculating the average (mean) for each column
        for col in column_names:
            try:
                if col in df.columns:
                    avg = df[col].mean(skipna=True) 
                    results.append({"Metric": col, "Average": avg})
                    has_valid_results = True
                else:
                    logger.warning(f"Column not found: {col}")
                    results.append({"Metric": col, "Average": None})
                    
            except Exception as e:
                logger.warning(f"Error calculating {col}: {str(e)}")
                results.append({"Metric": col, "Average": None})

        if not has_valid_results:
            logger.error("No processable columns found")
            return None

        # Returns a dataframe with the averages for each metric    
        return pd.DataFrame(results)

    # ------------------------------------------
    # CALCULATING STANDARD DEVIATION
    # ------------------------------------------

    def standard_deviation(self, df, column_names):

        """
        Calculates standard deviations for specified columns and returns a DataFrame.
        
        Args:
            df: Input DataFrame
            column_names: List of column names to analyze
                
        Returns:
            pd.DataFrame: Two columns ["Metric", "Standard Deviation"]
            None: If no valid columns exist
        """

        # Input validation
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if not isinstance(column_names, (list, tuple)):
            raise ValueError("column_names must be a list or tuple")

        results = []
        has_valid_columns = False  

        # The following block of code was improved with assistance from AI
            # Purpose: vectorization for performance
            # AI Tool: DeepSeek

        # Calculating the standard deviation for each column
        for col in column_names:
            try:
                if col in df.columns:
                    std = df[col].std(ddof=0, skipna=True)
                    results.append({"Metric": col, "Standard Deviation": std})
                    has_valid_columns = True
                else:
                    logger.warning(f"Column not found: {col}")
                    results.append({"Metric": col, "Standard Deviation": None})
                    
            except Exception as e:
                logger.warning(f"Error calculating {col}: {str(e)}")
                results.append({"Metric": col, "Standard Deviation": None})

        if not has_valid_columns:
            logger.error("No valid columns to process")
            return None
            
        # Returns a dataframe with the standard deviation for each metric    
        return pd.DataFrame(results)
    
    # ------------------------------------------
    # IDENTIFYING OUTLIERS
    # ------------------------------------------

     # This function was refined with the assistance of AI 
        # Purpose: Suggesting shorter code lines for calling standard deviation and average values from data frames. 
        # AI Tool: ChatGPT

    def outliers(self, df, column_names, standard_deviation, average, sd_modifier=2):  

        """
        Identifies outliers beyond specified standard deviations from mean in the given dataset.
        
        Args:
            df: Input DataFrame
            column_names: Columns to analyze
            standard_deviation: The standard deviation belonging to the columns of data
            average: The averages calculated for each parameter
            sd_modifier: Number of standard deviations for bounds (default = 2)
                
        Returns:
            tuple: (outliers_df, df_x_outliers)
            tuple: (None, df_x_outliers) If no outliers are found
        """  

        # The following block of code was generated by AI and edited by developers
            # Purpose: Input validation
            # Tool: DeepSeek

        # Input validation
        if not isinstance(standard_deviation, pd.DataFrame) or not isinstance(average, pd.DataFrame):
            raise ValueError("standard_deviation and average must be DataFrames")
        if not all(col in df.columns for col in column_names if col != df.columns[0]):
            logger.warning("Some columns not found in input DataFrame")

        date_column = df.columns[0]
        numeric_columns = [col for col in column_names if col != date_column]

        outlier_table = {}
        df_x_outliers = df.copy()

        for col in numeric_columns:
            try:

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

            # The following block of code was generated by AI
                # Purpose: Including error handling in a simple way
                # Tool: DeepSeek

            except (IndexError, KeyError) as e:
                logger.warning("Skipping %s - missing statistics: %s", col, str(e))
                outlier_table[col] = []
        
        # Format output
        outliers_df = pd.DataFrame({k: pd.Series(v) for k, v in outlier_table.items()})
        outliers_df[date_column] = df.loc[outliers_df.index, date_column]
        
        # Reorder to put date column first
        cols = [date_column] + numeric_columns
        outliers_df = outliers_df[cols]

        if outliers_df.drop(date_column, axis=1).isna().all().all():
            logger.info("No outliers found in any column")
            return None, df_x_outliers
        return outliers_df, df_x_outliers
    
    # ------------------------------------------
    # PLOTTING BOX PLOTS
    # ------------------------------------------


    # This function was refined with the assistance of AI 
        # AI suggested the code line "sns.boxplot" to make boxplots
        # AI suggested ways of enhancing the boxplot diagrams to be more viewer friendly
        # AI Tool: ChatGPT


    def box_plots (self, df, columns, color='skyblue', figsize=(8, 6), title=None, x_label=None):

        """
        Creates a box plot based on the given dataframe
        
        Args:
            df: Input DataFrame
            column_names: List of columns to include
            color: Boxplot color (default: 'skyblue')
            figsize: Figure size (default: (8, 6))
            title: Plot title (default: column name)            
        """

        # The following block of code was generated by AI
            #Purpose: Data validation
            # AI Tool: DeepSeek

        # Parameter validation
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if not columns or all(col not in df.columns for col in ([columns] if isinstance(columns, str) else columns)):
            raise ValueError("No valid columns provided")

        try:
            # Convert single string to list for consistency
            if isinstance(columns, str):
                columns = [columns]
            plt.figure(figsize=figsize)  # Now using figsize directly for both cases
                
            # SINGLE COLUMN CASE
            if len(columns) == 1:
                col = columns[0]
                sns.boxplot(x=df[col], color=color)
                plt.title(title if title else f"Distribution of {col}")
                plt.xlabel(x_label)
                plt.ylabel("")
                
            # MULTIPLE COLUMNS CASE
            else:
                # Convert color to list if needed
                if isinstance(color, str):
                    color = [color] * len(columns)
                
                df_melted = df[columns].melt(var_name='Variable', value_name='Value')
                sns.boxplot(
                    data=df_melted,
                    y='Variable',
                    x='Value',
                    hue='Variable',
                    palette=color,
                    orient='h',
                    legend=False
                )
                plt.title(title if title else "Boxplot Analysis")
                plt.xlabel(x_label)
                plt.ylabel("")
                
                # Adjust spacing based on number of variables
                plt.gca().margins(y=0.1)  # Add 10% padding
                    
            plt.tight_layout()
            plt.show()

        except KeyError as e:
            logger.error(f"Column not found: {str(e)}")
        except Exception as e:
            logger.error(f"Boxplot generation failed: {str(e)}")