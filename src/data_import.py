import requests
import pandas as pd

#A class to import and handle environmental data (weather and air quality)
class RawData:

    #Initialize the EnvironmentalData class
    def __init__(self):
        self.df = None

    # ------------------------------------------
    # WEATHER DATA - IMPORT
    # ------------------------------------------

    #Fetch weather data from the Frost API 
    def get_met(self, weather_station, weather_elements, weather_time, weather_resolution):

        """
        Args:
            weather_station (str): The ID of the weather station.
            weather_elements (str): The measurements to include.
            weather_time (str): The time range for the data.
            weather_resolution (str): The granularity of the data.

        Returns:
            pd.DataFrame: A DataFrame containing the weather data.
        """        

        try:
            # Client ID to access data from Frost API
            client_id = 'd933f861-70f3-4d0f-adc6-b1edb5978a9e'

            # Define endpoints and parameters
            endpoint = 'https://frost.met.no/observations/v0.jsonld'
            parameters = {
                'sources': weather_station,  # Station ID for Voll weather station
                'elements': weather_elements,  # Requestion various types of eather data
                'referencetime': weather_time,  # Limiting the time frame for the data request
                'timeresolutions': weather_resolution,  # Set the resolution(granularity) of the data
            }

            # Send an HTTP GET-request
            response = requests.get(endpoint, params=parameters, auth=(client_id, ''))

            # Extract JSON-data
            json_data = response.json()

            # Check if the request was succesfull, and exit if not
            if response.status_code == 200:
                data = json_data['data']
                print('Data collected from frost.met.no!')
            else:
                print('Error! Statuscode:', response.status_code)
                print('Message:', json_data['error']['message'])
                print('Cause:', json_data['error']['reason'])
                return None

            # Create and set up the dataframe
            df = pd.DataFrame()
            for obs in data:
                row = pd.DataFrame(obs['observations'])
                row['referenceTime'] = obs['referenceTime']
                df = pd.concat([df, row], ignore_index=True)


            # Remove uneeded collumns  

            '''add timeoffset if we decide not to use it: , "timeOffset"'''

            columns_to_drop = ["level", "timeResolution", "timeSeriesId", "elementId", "performanceCategory", "exposureCategory", "qualityCode"]
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

            # Remove time portion from 'referenceTime' (keep only 'YYYY-MM-DD')
            df["referenceTime"] = df["referenceTime"].str.split("T").str[0]

            print('There are ', df.shape[0], 'lines of data in this dataframe.\n')
            self.df = df

            #Returns dataframe upon request
            return(df)
        
        # Return an error code if fetching the weather data fails
        except requests.exceptions.ConnectionError:
            print("Error: Failed to connect to the Frost API. Check your internet connection.")
            return None
        except requests.exceptions.Timeout:
            print("Error: The request to the Frost API timed out. Please try again later.")
            return None
        except requests.exceptions.HTTPError as e:
            print(f"Error: The Frost API returned an HTTP error: {e}")
            return None
        except ValueError as e:
            print(f"Error: The Frost API response could not be parsed as JSON: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error has occured: {e}")
            return None


    # ------------------------------------------
    # WAIR QUALITY DATA - IMPORT
    # ------------------------------------------

    # Fetch air quality data by Nilu from a CSV file
    def get_nilu(self, threshold, file_path): 

        """
        Args:
            file_path (str): Path to the CSV file.
            threshold (float): Threshold for coverage values.

        Returns:
            pd.DataFrame: A DataFrame containing the air quality data.
        """

        try:
            df = pd.read_csv(
                file_path,
                skiprows=3,  
                sep=';',  
                encoding='utf-8',  
                on_bad_lines='skip'  
            )
        
            # Convert the 'Tid' column to a date-time format
            df['Tid'] = pd.to_datetime(df['Tid'], format='%d.%m.%Y %H:%M')
            time_column = 'Tid' 

            # Replace empty values with NaN
            df.replace('', pd.NA, inplace=True)      
    
            # Replacing any commas with periods, for the right number format
            for col in df.columns.difference([time_column]):
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False).str.strip()

            # Converting all collumns except for the 'Tid' column to numerical values
            df[df.columns.difference([time_column])] = df[
                df.columns.difference([time_column])
            ].apply(pd.to_numeric, errors='coerce')            

            # Replace the coverage(uptime) values that fall below the treshold with 0, to exclude the data from analysis
            columns_coverage = ['Dekning', 'Dekning.1', 'Dekning.2', 'Dekning.3', 'Dekning.4']
            for col in columns_coverage:
                df.loc[df[col] < threshold, col] = 0

            # Print some basic information about the dataframe
            print('Data collected from nilu.com!')
            print('There are ', df.shape[0], 'lines of data in this dataframe.\n')
            self.df = df

            #Returns dataframe upon request        
            return(df)

        # Return an error code if reading the csv file failed
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found. Check the file path.")
        except pd.errors.ParserError:
            print("Error: Could not read the csv file. Check the formatting and encoding.")
        except Exception as e:
            print(f"An unexpected error has occured: {e}")

