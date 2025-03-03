import pandas as pd

#A class to import and handle environmental data (weather and air quality)
class VisualData:

    #Initialize the EnvironmentalData class
    def __init__(self):
        self.df = None

    # ------------------------------------------
    # FORMATTING OF DATAFRAME VISUALISATION
    # ------------------------------------------

    # A function to make pandas show an entire dataframe in a readable manner
    def pretty_data(self, df):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        display(df)