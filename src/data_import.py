
__all__ = ['RawData'] 

import requests
import pandas as pd
import numpy as np
import logging # The use of logging was suggested by AI (DeepSeek)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class RawData:
    """A class to fetch and preprocess environmental data (weather and air quality).
    
    Attributes:
        df: Stores the most recently imported dataset
        
    Methods:
        get_met(): Fetches weather data from Frost API
        get_nilu(): Fetches air quality data from CSV
        get_forecast(): Fetches weather forecast data
        
    Note:
        Most methods accept a `show_info` parameter (bool) to control logging verbosity
    """

    def __init__(self):
        self.df = None # Stores the most recently imported dataset

    # ------------------------------------------
    # WEATHER DATA - IMPORT
    # ------------------------------------------

    #Fetch weather data from the Frost API 
    def get_met(self, station_id, elements, time_range, resolution, show_info=False):

        """
        Fetch weather data from the Frost API.

        Args:
            station_id (str): The ID of the weather station.
            elements (str): The measurements to include.
            time_range (str): Date range for the data.
            resolution (str): Time resolution.
            show_info (bool): Decides if all info is logged or printed.

        Returns:
            pd.DataFrame: A DataFrame containing the weather data.
                Returns None if an error occurs.
        """        

        try:

            # The following block of code (if statements) was mainly generated by AI, lightly reworked by developers
                # Purpose: Raise an error if the given arguments are invalid
                # AI tool: DeepSeek

            # Validate the arguments given to the function
            if not station_id or not elements or not time_range or not resolution:
                raise ValueError("All input parameters must be provided.")

            if not isinstance(station_id, str):
                raise ValueError("station_id must be a string.")

            if not isinstance(elements, str):
                raise ValueError("elements must be a string.")

            if not isinstance(time_range, str):
                raise ValueError("time_range must be a string.")

            if not isinstance(resolution, str):
                raise ValueError("resolution must be a string.")

            # If the arguments are valid, proceed with the API request

            # Client ID to access data from Frost API
            client_id = 'd933f861-70f3-4d0f-adc6-b1edb5978a9e'

            # The following block of code (the four variable lines) was generated with the assistance of AI
                # Purpose: Correctly call upon the four variables in this function
                # AI tool: DeepSeek

            # Define endpoints and parameters
            endpoint = 'https://frost.met.no/observations/v0.jsonld'
            parameters = {
                'sources': station_id,  # Station ID 
                'elements': elements,  # Requestion various types of weather data
                'referencetime': time_range,  # Limiting the time frame for the data request
                'timeresolutions': resolution,  # Set the resolution(granularity) of the data
                'levels': 'default',
                'timeoffsets': 'default', # Selects the best timeiffset value available, first PT6H then PT0H
                'qualities': '0,1,2,3,4' # Only import data of a high enough quality as explained here: https://frost.met.no/dataclarifications.html#quality-code
            }

            # Send an HTTP GET-request
            response = requests.get(endpoint, params=parameters, auth=(client_id, ''))

            # Extract JSON-data
            json_data = response.json()

            # Check if the request was successfull, and exit if not
            if response.status_code == 200:
                data = json_data['data']
                unique_times = len({obs['referenceTime'] for obs in data})

                if show_info:
                    logger.info(
                        '\nSuccessfully collected %s raw observations (%s unique timestamps) from Frost API\n',
                        len(data), 
                        unique_times
                    )
            else:
                logger.error(
                    'API Error %s: %s (Reason: %s)',
                    response.status_code,
                    json_data['error']['message'],
                    json_data['error']['reason']
                )
                return None

            # Create and set up pd.DataFrame
            data_list = []
            for obs in data:
                if isinstance(obs['observations'], list):
                    for observation in obs['observations']:
                        row = observation.copy()
                        row['referenceTime'] = obs['referenceTime']
                        data_list.append(row)
                else:
                    row = obs['observations'].copy()
                    row['referenceTime'] = obs['referenceTime']
                    data_list.append(row)
            df = pd.DataFrame(data_list)

            # Remove unneeded collumns  
            columns_to_drop = ["level", "timeResolution", "timeSeriesId", "elementId", "performanceCategory", "exposureCategory", "qualityCode"]
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

            # Remove time portion from 'referenceTime' (keep only 'YYYY-MM-DD')
            df["referenceTime"] = df["referenceTime"].str.split("T").str[0]

            self.df = df

            # The following block (if) was generated with the assistance of AI
                # Purpose: AI generated the df.drop_duplicates line and suggested how to use df.pivot
                # AI tool: DeepSeek            

            if not df.empty:
                # Check rows for duplicate/multiple values and only keeping one
                cols = ['referenceTime']
                if 'unit' in df.columns:
                    cols.append('unit')
                df = df.drop_duplicates(subset=cols, keep='first')
                
                # Pivot to wide format
                pivoted_df = df.pivot(
                    index='referenceTime',
                    columns='unit',
                    values='value'
                ).reset_index()

                try:
                    pivoted_df['referenceTime'] = pd.to_datetime(pivoted_df['referenceTime'])
                except ValueError as e:
                    logger.error("Date conversion failed: %s", e)
                    return None
                
                # Clean up column names
                pivoted_df.columns.name = None
                df = pivoted_df.rename(columns={
                    'degC': 'temperature (C)',
                    'mm': 'precipitation (mm)',
                    'm/s': 'wind_speed (m/s)',
                    'referenceTime': 'Date'
                })

                df['Date'] = pd.to_datetime(df['Date'])

            if show_info:
                logger.info(
                    '\nProcessed DataFrame: %s rows x %s parameters (%.1f%% non-empty values)\n',
                    df.shape[0],
                    df.shape[1] - 1, #Ignoring date column
                    100 * df.iloc[:, 1:].notna().mean().mean() 
                )

            # Returns pd.DataFrame upon request
            return df 
        
        # The following block was generated with the assistance of AI
            # Purpose: Including more specific errors; Connection, timeout, and HTTP
            # AI tool: DeepSeek

        # Return an error code if fetching the weather data fails
        except ValueError as e:
            logger.error("Invalid input: %s", e)
            return None
        except requests.exceptions.ConnectionError:
            logger.error("Connection failed - check internet")
            return None
        except requests.exceptions.Timeout:
            logger.warning("API timeout - try again later")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error("API HTTP error: %s", e)
            return None
        except Exception as e:
            logger.exception("Unexpected error during processing")
            return None

    # ------------------------------------------
    # AIR QUALITY DATA - IMPORT
    # ------------------------------------------

    # Fetch air quality data by Nilu from a CSV file
    def get_nilu(self, threshold, file_path, show_info=False): 



        """
        Fetch air quality data from a CSV file in the data directory.

        Args:
            file_path (str): Path to NILU CSV file (expected format: semicolon-delimited)
            threshold (float): Minimum coverage percentage (0-100) - measurements with lower coverage will be set to NaN
            show_info (bool): Decides if all info is logged or printed.
            
        Returns:
            pd.DataFrame: A DataFrame containing the air quality data.
        """

        # The following block was optimised for lower resource use with the assistance of AI
            # Purpose: reduce runtime and improve of the code when reading the csv file
            # AI tool: DeepSeek

        try:
            # DEBUG: Print raw file contents
            print("\nDEBUG - Raw file contents:")
            with open(file_path, 'r', encoding='utf-8') as f:
                print(f.read())
            
            # DEBUG: Print detected columns
            headers = pd.read_csv(file_path, nrows=1, skiprows=3, sep=';').columns
            print("\nDEBUG - Detected columns:", headers.tolist())

            # parsing csv file
            df = pd.read_csv(
                file_path,
                skiprows=3,
                sep=';',
                on_bad_lines='skip',
                encoding='utf-8',
                parse_dates=[0],
                date_format='%d.%m.%Y %H:%M',
                na_values='',  # Replace empty strings with NaN
                decimal=',',   # Handle comma decimals correctly
                dtype={col: 'float64' for col in pd.read_csv(file_path, nrows=1, skiprows=3, sep=';').columns 
                    if col != 'Tid'}  # Pre-specify dtypes
            )

            if show_info:
                logger.info(
                    '\nSuccessfully collected and processed %s rows of data from \n%s\n',
                    df.shape[0], 
                    file_path
                )
      
            # The following two blocks are an improvement made based on a suggestion from AI
                # Purpose: Entirely replace a function in 'data_handling' that dealt with coverage by utilising and removing the overage columns within this function
                # AI tool: DeepSeek
            
            # Process coverage columns
            coverage_cols = df.columns[df.columns.str.contains('Dekning', case=False)].tolist()            
            for cov_col in coverage_cols:
                meas_col = df.columns[df.columns.get_loc(cov_col) - 1] # Get corresponding measurement column
                df.loc[df[cov_col] < threshold, meas_col] = np.nan # Set measurements to NaN where coverage is below threshold
            df = df.drop(columns=coverage_cols) # Remove all coverage columns after processing
            
            # Create column names
            new_cols = {col: col.split()[1] if col != 'Tid' else 'Date' for col in df.columns}            
            df = df.rename(columns=new_cols)

            self.df = df
            
            if show_info:
                logger.info(
                    '\nProcessed DataFrame: %s rows x %s parameters (%.1f%% non-empty values)\n',
                    df.shape[0],
                    df.shape[1] - 1, #Ignoring date column
                    100 * df.iloc[:, 1:].notna().mean().mean() 
                )
            
            # Returns pd.DataFrame
            return df
        
        # The following block was generated with the assistance of AI
            # Purpose: Including more specific errors; FileNotFound, ParserError
            # AI tool: DeepSeek

        # Return an error code if reading the csv file failed
        except FileNotFoundError:
            logger.error("File not found: %s", file_path)
            return None
        except pd.errors.ParserError:
            logger.error("CSV parsing failed - check file encoding")
            return None
        except Exception as e:
            logger.exception("An unexpected error has occured")
            return None

    # ------------------------------------------
    # WEATHER FORECAST - IMPORT
    # ------------------------------------------

    def get_forecast(self, station_id=None, lat=63.419, lon=10.395):

        """
        Fetches daily weather forecast from MET Norway's Locationforecast API.
        
        Args:
            station_id (str): Optional – if provided, uses its coordinates.
            lat (float): Latitude (default: Trondheim)
            lon (float): Longitude (default: Trondheim)
        
        Returns:
            pd.DataFrame | None: Forecast data or None if failed
        """

        url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat}&lon={lon}"
        headers = {"User-Agent": "myweatherapp/1.0 your_email@example.com"}
        
        try:
            # If station_id is given, look up its coordinates
            if station_id:
                client_id = 'd933f861-70f3-4d0f-adc6-b1edb5978a9e'
                source_url = f'https://frost.met.no/sources/v0.jsonld?ids={station_id}'
                response = requests.get(source_url, auth=(client_id, ''))
                response.raise_for_status()
                source_data = response.json()
                lat = source_data['data'][0]['geometry']['coordinates'][1]
                lon = source_data['data'][0]['geometry']['coordinates'][0]

            # Fetch forecast data (hourly resolution)
            url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat}&lon={lon}"
            headers = {"User-Agent": "myweatherapp/1.0 your_email@example.com"}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            # Process into daily aggregates
            daily_data = {}
            last_precip_time = None

            for entry in data["properties"]["timeseries"]:
                #AI-suggested improvement for robust datetime handling (DeepSeek)
                time = pd.to_datetime(entry["time"])
                date_str = time.strftime("%Y-%m-%d")  # Store as string early
                details = entry["data"]["instant"]["details"]

                if date_str not in daily_data:
                    daily_data[date_str] = {"temps": [], "precips": [], "winds": []}

                # Always add temperature/wind
                daily_data[date_str]["temps"].append(details.get("air_temperature"))
                daily_data[date_str]["winds"].append(details.get("wind_speed"))

                # Handle precipitation (skip overlaps)
                precip = 0
                if "next_6_hours" in entry["data"]:
                    precip_end = time + pd.Timedelta(hours=6)
                    
                    # Only add precipitation if this is a new period
                    if last_precip_time is None or time >= last_precip_time:
                        precip = entry["data"]["next_6_hours"]["details"].get("precipitation_amount", 0)
                        last_precip_time = precip_end  # Update tracking
                elif "next_1_hours" in entry["data"]:
                    precip = entry["data"]["next_1_hours"]["details"].get("precipitation_amount", 0)
                    
                daily_data[date_str]["precips"].append(precip)

            # Convert to daily averages/sums
            forecast_days = []
            for date_str, values in daily_data.items():
                forecast_days.append({
                    "Date": date_str,  # Already in YYYY-MM-DD format
                    "temperature (C)": np.nanmean(values["temps"]),
                    "wind_speed (m/s)": np.nanmean(values["winds"]),
                    "precipitation (mm)": np.nansum(values["precips"]),
                })

            df = pd.DataFrame(forecast_days)
            desired_order = ["Date", "temperature (C)", "wind_speed (m/s)", "precipitation (mm)"]
            df = df.reindex(columns=[col for col in desired_order if col in df.columns])

            # Only keep future forecasts
            today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
            if not df.empty:
                df['Date'] = pd.to_datetime(df['Date'])
                today = pd.Timestamp.now().normalize()
                return df[df['Date'] >= today].copy()

        except Exception as e:
            logger.error(f"Failed to fetch forecast: {e}")
            return None