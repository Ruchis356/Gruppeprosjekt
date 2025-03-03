
import requests
import pandas as pd

# ------------------------------------------
# WEATHER DATA - IMPORT
# ------------------------------------------

    # Global variables needed to run the function independently
#weather_time = '2024-02-19/2024-03-19'
#weather_station = 'SN68860'
#weather_elements = 'mean(air_temperature P1D),sum(precipitation_amount P1D),mean(wind_speed P1D)'
#weather_resolution = 'P1D' 


class EnvironmentaleData:

    def __init__(self, data):
        self.df = []

    def get_met():

        # Client ID to access data from 
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
        r = requests.get(endpoint, params=parameters, auth=(client_id, ''))

        # Extract JSON-data
        json_data = r.json()

        # Check if the request was succesfull, and exit if not
        if r.status_code == 200:
            data = json_data['data']
            print('Data collected from frost.met.no!')
        else:
            print('Error! Statuscode:', r.status_code)
            print('Message:', json_data['error']['message'])
            print('Cause:', json_data['error']['reason'])
            exit()  

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

        #Returns dataframe upon request
        return(df)




    # *** EVERYTHING below this line is purely for local testing purposes of this specific file ***
    '''Fjern # foran de to linjene under som kjører funksjonen og så printer den, for å se om det virker :)
    Husk og de kommenterte variablene på toppen!'''

        # Run the function to get the dataframe with the weather data
    #df=get_met()

        # Show the first few rows of the dataframe
    #print(df.head())












    # ------------------------------------------
    # WAIR QUALITY DATA - IMPORT
    # ------------------------------------------



    # Global variables needed to run the function independently
    threshold = 95 
    file_path = '../Data/luftkvalitet_trondheim_dag.csv'

    def get_nilu(): 

        # Trying to read the data from the csv file
        try:
            df = pd.read_csv(
                file_path,
                skiprows=3,  
                sep=';',  
                encoding='utf-8',  
                on_bad_lines='skip'  
            )
        
            '''# Print av data gjøres i main, så denne delen kan slettes.
    #        print("Første rader av datasettet:")
    #        print(luftkvalitet_df.head())
    #        print("\nAntall rader og kolonner i datasettet:")
    #        print(luftkvalitet_df.shape)'''


            # Convert the 'Tid' column to a date-time format
            df['Tid'] = pd.to_datetime(df['Tid'], format='%d.%m.%Y %H:%M')
            time_column = 'Tid' 


            '''#Vi vil ikke ha tid som indeks. Slett?
            # Sett 'Tid' som indeks
        #    luftkvalitet_df.set_index('Tid', inplace=True)'''


            # Replace empty values with NaN
            df.replace('', pd.NA, inplace=True)      
    
            # Replacing any commas with periods, for the right number format
            for col in df.columns.difference([time_column]):
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False).str.strip()

            # Converting all collumns except for the 'Tid' column to numerical values
            df[df.columns.difference([time_column])] = df[
                df.columns.difference([time_column])
            ].apply(pd.to_numeric, errors='coerce')


            ''' Den gamle koden for å gjøre om til nummer. Jeg har skrevet den litt mer robust, og tatt hensyn til "Tid" kolonnen
                    #    df = df.apply(pd.to_numeric, errors='ignore')'''
            

            # Replace the coverage(uptime) values that fall below the treshold with 0, to exclude the data from analysis
            columns_coverage = ['Dekning', 'Dekning.1', 'Dekning.2', 'Dekning.3', 'Dekning.4']
            for col in columns_coverage:
                df.loc[df[col] < threshold, col] = 0


            '''Likeså har jeg utvidet funksjonen som bytter ut "ikke god nok" dekning med 0, bl. annet tar med alle 5 deknings-kolonnene, ikke bare de tre første.'''


            ''' Er det tenkt å bruke denne delen til noe spesifikt? Hvis ikke kan vi slette den og jobbe med statistikken senere (i en annen fil).
            # Vis statistikk for datasettet
            #print("\nStatistikk for datasettet:")
            #print(df.describe())'''


            # Print some basic information about the dataframe
            print('Data collected from nilu.com!')
            print('There are ', df.shape[0], 'lines of data in this dataframe.\n')

            #Returns dataframe upon request        
            return(df)


            '''#renger vi å lagre datene til en ny csv-fil? Eller skal vi bare la det være en dataframe? Slett?
            # Lagrer de behandlede dataene til en ny CSV-fil
        #    output_file_path = 'behandlet_luftkvalitet_trondheim.csv'
        #    luftkvalitet_df.to_csv(output_file_path)
        #    print(f"\nBehandlede data lagret til {output_file_path}")'''


        # Return an error code if reading the csv file failed
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found. Check the file path.")
        except pd.errors.ParserError:
            print("Error: Could not read the csv file. Check the formatting and encoding.")
        except Exception as e:
            print(f"An unexpected error has occured: {e}")



    '''#Denne kan fjernes. Den ligger i main. Slett?
        # Changes pandas to show the whole dataframe 
        #pd.set_option('display.max_rows', None)
        #pd.set_option('display.max_columns', None)
        #pd.set_option('display.width', None)
        # Shows the whole dataframe
        #display(luftkvalitet_df)'''



    # *** EVERYTHING below this line is purely for local testing purposes of this specific file ***
    '''Fjern # foran de to linjene under som kjører funksjonen og så printer den, for å se om det virker :)
    Husk og de kommenterte variablene på toppen!'''

        # Run the function to get the dataframe with the air quality data
    df=get_nilu()

        # Show the first few rows of the dataframe
    print(df)    
