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





*** DELETE BELOW BEFORE HANDING IN ***

# Hvordan takle de forventede iregularitetene?
- Inconsistent date formats: Use pd.to_datetime with a specified format.
- Missing columns: Check for required columns and raise an error if they’re missing.


# kildeautoritet, data kvalitet, tigljengelighet, brukervennlighet, osv.
  - Meterologisk institutt. Kildeautoritet er det viktigste kriteriet. Dette er et statlig institutt som står for meterologisk utforskning og suppler meterologisk data for folkeinformasjon via blant annet yr.no. Tilgjengeligheten er også god, da MI legger ut all meterologisk data via API. Fordi de bruker API er det også relativt brukervennlig. Frost API. Med Frost API har de veiledning og tutorials på hvordan vi kan importere og bruke data fra MI i python spesifikt. 
  
  - Nilu (Tidligere norsk institutt for luftforskning) for miljødata. Nonprofit og uavhengig. Forskningsinstitutt. Brukervennlig. De har eksistert i flere tiår og utviklet seg over tid. 

  Hensikten med å velge disse kildene er å sammenligne dataene med hverandre. Ved å samle inn værdata fra Meteorologisk institutt, samt data om luftforurensning fra samme tidsperiode, ønsker vi å utforske et mønster i korrelasjonen mellom disse datasettene. Vi ønsker å se på data fra perioden 2010 til 2020. Disse dataene vil vi bruke til å lage prediksjoner og sammenligne dem med den faktiske trenden for værdata og luftkvalitet fra 2020 til 2025.

