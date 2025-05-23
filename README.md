## Main objective
To gather data from two sources: Meteorologisk institutt and NILU, analyse the data, and use the analysis to predict air polution based on the current weather forecast. 

## Application summary
By collecting weather data (from Meteorologisk institutt) and air pollution data (from NILU) from the same time period at nearly the same location, we explore the patterns in, and correlation between, these two datasets. We use data from the period 2006 to 2018 to create models for predictive analysis, and data from 2020 to 2025 for the testing of these models. The models are then applied to the current weatherforecast to present a forecast for pollution. 

Meteorologisk institutt is a well-known source that provides meteorological data to users through websites such as yr.no. The data published by Meteorologisk institutt is available through the Frost API data format, which offers a user-friendly method for data extraction. NILU, previously Norsk institutt for luftforskning, is a nonprofit organization that collects environmental data, focusing specifically on air quality and pollution in different regions of Norway. The organization was first established in 1969 and has gained credibility as an air research institute due to its extensive historical data. The data used in this project has been exported to a csv file for import. 

For this project we have written a primary script titled “main”, which has been placed in the (jupyter) notebooks folder. This notebook file utilises the python files in the src folder to handle the data from the two sources to generate the output. The functions within these python files are used to: 
- extract data from the API and the csv file
- cleaning up the resulting data and handling missing/faulty data
- mathematicaly analysing the data and preparing for analysis
- analysing data for patterns and trends
- creating interactive graphs for various versions of the datasets
- predictive analysis of weather and pollution
- creating a prediction of pollution levels based on the weather forecast

When using this data for analysis and future predictions, we have chosen to replace invalid, poor quality, or missing data points with NaN, excluding them from the dataframe. As a result, these will not be used to assess the future trendline. We have not created datapoints to replace these values, as we have utilised more than ten years of daily data which is a sufficient amount to create the mathematical models we are using. 

## Project Setup  
   # This block was generated by AI (DeepSeek) and reworked and edited by the developers
1. Clone the repository: `git clone <https://github.com/Ruchis356/Gruppeprosjekt/tree/main>`
2. Create and activate a virtual environment:
   - On macOS/Linux: `python3 -m venv venv && source venv/bin/activate`
   - On Windows: `python -m venv venv && venv\Scripts\activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Open the project in VSCode and run `main.ipynb`

## Project Structure
> data
   - : `FILLFILLFILL`
   - : `FILLFILLFILL`
   - README.md
> docs
   > AI declaration
   -
      - AI declaration.pdf: `The declaration of AI usage in this project, signed by the developers.`
      - detailed AI use.md:  `A line for line/block for block breakdown of AI usage in this project.` 
   - bibliography.md: `A list of external resources used in this project.`
   - README.md
> notebooks
   - main.ipynb: `Main script for importing and processing data. All variables can be set in the code of main.`
   - README.md
> resources
   - README.md
> src
   > __psycache__: ``
   - __init.py__: ``
   - analysis.py: `Contains the AnalysedData class for numerical analysis eg. average(mean) and standard deviation.`
   - data_handling.py: `Contains the RefinedData class for processing data eg. dealing with missing values.`
   - data_import.py: `Contains the RawData class for importing weather and air quality data.`
   - graphs.py: `Contains the Graphs class for creating graphs based on the data to visualise patterns and comparing factors.`
   - predictive_analysis.py: `FILLFILLFILL`
   - README.md
   - utils.py: `Contains a collection of simple tools for data handling and visualisation.`
> test
   - : `FILLFILLFILL`
   - : `FILLFILLFILL`
- .gitignore: `This file specifies which files and directories are ignored when commits are made to the remote repository by the developers.`
- README.md: `This file -contains the necesarry instructions for understanding the project, the folder structure, and how each part functions. Each sub-folder also contains a README file for its specific content, with each file described in more detail.`
- reuqirements.txt: `Describes the requirement for running the complete project on your computer.`

## Expected irregularities from environmental data sources and possible solutions
- Missing values: Look up missing or faulty data and replace with NaN. 
- Missing columns or rows: Check for required columns and raise an error if they’re missing.
- Inconsistent date formats: Use pd.to_datetime with a specified format.

## Evaluation of data sources
- The most important criteria was source authority, followed by accecability and ease of use. 
- Frost API (meterologisk institutt): Data from the Norwegian Meteorological Institute. This is a reputable institution with detailed and high quality data at many locations nationally and world wide. The API is reliable and well-documented, giving a high level of ease of use.
- Nilu: Previously known as the national institute for air research. It is one of the leading laboratories in Europe researching issues around air pollution and climate change. The data provided is detailed, structured, and high quality, but the files requires some amount of pre-processing.

## Use of AI in this project
AI tools were used for the following purposes:
- Understanding error codes and suggesting possible solutions.
- Suggestions for specific constructs (e.g., exception handling constructs).
- Smaller blocks of code that were reworked and integrated into the overall code.
For more details see the bibliography and AI declaration in the docs folder.