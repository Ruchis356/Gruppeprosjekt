# Gruppeprosjekt: notat for del 1

Begge datasettene er planlagt å utvides til 10 år med daglige gjennomsnitt, men er foreløbig begrenset til en kortere periode for å minske prosesseringstiden mens vi jobber med selve koden. 


# Miljødataanalyseapplikasjon

## Project Setup
1. Clone the repository: `git clone <https://github.com/Ruchis356/Gruppeprosjekt/tree/main>`
2. Install dependencies: `pip install pandas requests`
3. Open the project in VSCode and run `main.ipynb`.

## Project Structure
- `main.ipynb`: Main script for importing and processing data.
- `data_import.py`: Contains the `RawData` class for importing weather and air quality data.
- `data_handling.py`: Contains the `RefinedData` class for processing data.
- `utils.py`: Contains a collection of simple tools for data handling and visualisation. 

## Expected irregularities from environmental data sources
- Missing values
- Missing columns or rows
- Inconsistent date formats

## Evaluation of data sources
- Frost API (meterologisk institutt): High-quality data from the Norwegian Meteorological Institute, reliable and well-documented.
- Nilu: Provides detailed air quality data but requires preprocessing.

The main objective of our program is to compare data from two sources: Meteorologisk institutt and Norsk institutt for luftforskning.
Meteorologisk institutt is a well-known source that provides meteorological data to users by providing Norwegian weather data to websites such as yr.no. The data published by Meteorologisk institutt is available through the Frost API data format, which offers a user-friendly method for data extraction.
Norsk institutt for luftforskning (NILU) is a nonprofit organization that collects environmental data, focusing specifically on air quality in different regions of Norway. The organization was first established in 1969 and has gained credibility as an air research institute due to its extensive historical data. This data has been extracted using a csv file. 
By collecting weather data (from Meteorologisk institutt) and air pollution data (from NILU)  from the same time period, we aim to explore patterns in the correlation between these datasets. We intend to examine data from the period 2010 to 2020. This data will be used to make predictions and compare them with actual trends in weather data and air quality from 2020 to 2025.
For the first part of the project, we have written a primary script titled “main”, which has been placed in the notebooks folder. The output of this script depends on two supporting scripts, “data_handling” and “data_import”, both located in the src folder. The data_import script provides access to imported data from NILU and Meteorologisk institutt, while the code within data_import checks for the exact row placement holding any invalid or inaccessible data within the selected time frame. By calling these codes within the main function, we obtain an overview of the refined dataset, filtering out anomalous data from the selected time period.
Note that for Part 1, only an excerpt of data from 2024 has been extracted and refined. This is for testing purposes and to avoid handling large datasets at once, which will be done directly in Part 2.
When using this data for future predictions, we have chosen to replace invalid or missing data points (columns or rows) with empty spaces (treating all values in the corresponding rows as empty slots). As a result, these will not be used to assess the future trendline. This approach appears to be the most realistic for the analysis ahead.






*** DELETE BELOW BEFORE HANDING IN ***

# Hvordan takle de forventede iregularitetene?
- Inconsistent date formats: Use pd.to_datetime with a specified format.
- Missing columns: Check for required columns and raise an error if they’re missing.


# kildeautoritet, data kvalitet, tigljengelighet, brukervennlighet, osv.
  - Meterologisk institutt. Kildeautoritet er det viktigste kriteriet. Dette er et statlig institutt som står for meterologisk utforskning og suppler meterologisk data for folkeinformasjon via blant annet yr.no. Tilgjengeligheten er også god, da MI legger ut all meterologisk data via API. Fordi de bruker API er det også relativt brukervennlig. Frost API. Med Frost API har de veiledning og tutorials på hvordan vi kan importere og bruke data fra MI i python spesifikt. 
  
  - Nilu (Tidligere norsk institutt for luftforskning) for miljødata. Nonprofit og uavhengig. Forskningsinstitutt. Brukervennlig. De har eksistert i flere tiår og utviklet seg over tid. 

  Hensikten med å velge disse kildene er å sammenligne dataene med hverandre. Ved å samle inn værdata fra Meteorologisk institutt, samt data om luftforurensning fra samme tidsperiode, ønsker vi å utforske et mønster i korrelasjonen mellom disse datasettene. Vi ønsker å se på data fra perioden 2010 til 2020. Disse dataene vil vi bruke til å lage prediksjoner og sammenligne dem med den faktiske trenden for værdata og luftkvalitet fra 2020 til 2025.


