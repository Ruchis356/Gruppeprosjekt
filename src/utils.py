import pandas as pd



# Importing the display-funtion for a jupyter file in a conditional manner
try:
    from IPython.display import display, HTML
    IN_JUPYTER = True
except ImportError:
    IN_JUPYTER = False

# A class to import and handle environmental data (weather and air quality)
class VisualTable:

    # Initialize the EnvironmentalData class
    def __init__(self):
        self.df = None

    # ------------------------------------------
    # FORMATTING OF DATAFRAME VISUALISATION
    # ------------------------------------------

    # A function to make pandas show an entire dataframe in a readable manner
    def pretty_data(self, df):
        """
        This function was reworked with the assistance of AI:
        - Purpose: To improve the readability of large DataFrames in Jupyter Notebooks 
          by making the displayed tables scrollable
        - AI Tool: DeepSeek
        """
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)

        # If the function is called in a jupyter notebook, make the dataframe HTML and scrollable
        if IN_JUPYTER:
            df_html = df.to_html()

            # Wrap the HTML table in a scrollable div
            scrollable_table = f"""
            <div style="overflow-x: auto; overflow-y: auto; max-height: 400px; max-width: 100%;">
                {df_html}
            </div>
            """

            # Display the scrollable table
            display(HTML(scrollable_table))

        # Outside of jupyter notebooks, the table is printed as normal if needed
        else:
            print(df)

