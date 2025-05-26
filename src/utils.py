




# Add a way to remove the 00:00:00 from the dates before printing








import pandas as pd

# Importing the display-function for a jupyter file in a conditional manner
try:
    from IPython.display import display, HTML
    IN_JUPYTER = True
except ImportError:
    IN_JUPYTER = False


class VisualTable:
    """Handles DataFrame display with scrollable tables in Jupyter."""

    def __init__(self):
        self.df = None

    # ------------------------------------------
    # FORMATTING OF DATAFRAME VISUALISATION
    # ------------------------------------------

    def pretty_data(self, df):

        """
        Display a DataFrame in a readable, scrollable format in Jupyter Notebooks,
        or print normally in other environments.

        Args:
            df (pd.DataFrame): The DataFrame to be displayed.

        Note:
            This function was reworked with the assistance of AI (DeepSeek) to
            improve readability of large DataFrames in Jupyter Notebooks, by
            making tables scrollable.
        """

        # Set pandas display options for maximum visibility
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)

        # Formatting numbers to clean up decimals
        def clean_number(x):
            if pd.isna(x):
                return x
            if isinstance(x, (int, float)):
                if x == int(x):
                    return int(x) 
                
                rounded = round(x, 2)
                if rounded == round(x, 1):
                    return round(x, 1)  
                return rounded
            return x 

        # Apply formatting
        formatted_df = df.copy()
        numeric_cols = df.select_dtypes(include=['int', 'float']).columns
        for col in numeric_cols:
            formatted_df[col] = df[col].apply(clean_number)

        if IN_JUPYTER:
            # Jupyter display 
            styler = formatted_df.style.format(
                na_rep="NaN",
                subset=numeric_cols,  # Only format numeric cols
                formatter=lambda x: (
                    f"{int(x)}" if isinstance(x, float) and x.is_integer()
                    else f"{x:.2f}".rstrip('0').rstrip('.') 
                    if f"{x:.2f}".endswith('0') 
                    else f"{x:.2f}"
                )
            )
            display(HTML(
                f'<div style="max-height:400px; overflow:auto">'
                f'{styler.to_html()}'
                f'</div>'
            ))

        else:
            # Fall back to standard printing outside jupyter
            with pd.option_context(
                'display.float_format', 
                lambda x: (
                    f"{int(x)}" if x == int(x)
                    else f"{x:.1f}" if f"{x:.2f}"[-1] == '0'
                    else f"{x:.2f}"
                )
            ):
                print(formatted_df.to_string(na_rep='NaN'))