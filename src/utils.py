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

        if IN_JUPYTER:
            # Convert DataFrame to HTML and wrap in a scrollable div
            df_html = df.to_html()

            scrollable_table = f"""
            <div style="
                overflow-x: auto; 
                overflow-y: auto; 
                max-height: 400px; 
                max-width: 100%;
            ">
                {df_html}
            </div>
            """

            # Display the scrollable table
            display(HTML(scrollable_table))

        # Fall back to standard printing outside Jupyter
        else:
            print(df)