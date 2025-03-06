# Gruppeprosjekt: notat for del 1

Begge datasettene er planlagt å utvides til 10 år med daglige gjennomsnitt, men er foreløbig begrenset til en kortere periode for å minske prosesseringstiden mens vi jobber med selve koden. Enhetstesting, errorhandling, og hvordan behandle uregelmessigheter vil bli utforsket dypere i del 2, i tilegg til analyse og prediksjon. 


## Main objective
To gather data from two sources: Meteorologisk institutt and Norsk institutt for luftforskning, analyse the data, and use the analysis to predict air polution based on past and current weather measurements. 

## Application summary
By collecting weather data (from Meteorologisk institutt) and air pollution data (from NILU) from the same time period, we aim to explore patterns in the correlation between these datasets. We intend to examine data from the period 2010 to 2020. This data will be used to make predictions for future air polution, and we will compare them with actual trends in weather data and air quality from 2020 to 2025.

Meteorologisk institutt is a well-known source that provides meteorological data to users through websites such as yr.no. The data published by Meteorologisk institutt is available through the Frost API data format, which offers a user-friendly method for data extraction. NILU, previously Norsk institutt for luftforskning, is a nonprofit organization that collects environmental data, focusing specifically on air quality and pollution in different regions of Norway. The organization was first established in 1969 and has gained credibility as an air research institute due to its extensive historical data. This data has been extracted to a csv file for import. 

For the first part of the project, we have written a primary script titled “main”, which has been placed in the notebooks folder. The output of this script depends on two supporting scripts, “data_handling” and “data_import”, both located in the src folder. The data_import script provides access to imported data from NILU and Meteorologisk institutt, while the code within data_handling checks for the exact location of any invalid or missing data within the selected dataframe. By calling these codes within the main function, we obtain an overview of the refined dataset, filtering out anomalous data for furter handling and anlysis.

When using this data for analysis and future predictions, we have chosen to replace invalid, poor quality, or missing data points with NaN, excluding them from the dataframe. As a result, these will not be used to assess the future trendline. This approach appears to be the most realistic for the analysis ahead.

## Project Setup (Codeblock supplied by Deepseek)
1. Clone the repository: `git clone <https://github.com/Ruchis356/Gruppeprosjekt/tree/main>`
2. Install dependencies: `pip install pandas requests`
3. Open the project in VSCode and run `main.ipynb`.

## Project Structure
- `main.ipynb`: Main script for importing and processing data. All variables can be set in the code of 'main'. 
- `data_import.py`: Contains the `RawData` class for importing weather and air quality data.
- `data_handling.py`: Contains the `RefinedData` class for processing data.
- `utils.py`: Contains a collection of simple tools for data handling and visualisation. 

## Expected irregularities from environmental data sources and possible solutions
- Missing values: Look up missing or faulty errors and replace with NaN. 
- Missing columns or rows: Check for required columns and raise an error if they’re missing.
- Inconsistent date formats: Use pd.to_datetime with a specified format.

## Evaluation of data sources
- The most important criteria was source authority, followed by accecability and ease of use, 
- Frost API (meterologisk institutt): Data from the Norwegian Meteorological Institute, reliable and well-documented. 
- Nilu: Provides detailed air quality data, but requires preprocessing.