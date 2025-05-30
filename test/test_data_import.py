
import sys, os
import pandas as pd
import unittest
import requests
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_import import RawData


class TestRawData(unittest.TestCase):

    ### TESTING GET_MET FUNCTION###

    # Test that the function correctly fetches data and returns a DataFrame with the expected parameters
    @patch('requests.get')
    def test_get_met_success(self, mock_get):

        # The following codeblock was generated with the assistance of AI (reused for later code blocks)
        # - Purpose: Creating a functioning mock API response
        # - AI-tool: DeepSeek

        # Creating a mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [
                {
                    'referenceTime': '2023-10-01T00:00:00Z',
                    'observations': [
                        {'elementId': 'temperature', 'value': 15.0, 'unit': 'degC'},
                        {'elementId': 'precipitation', 'value': 60.0, 'unit': 'mm'}
                    ]
                },
                {
                    'referenceTime': '2023-10-02T00:00:00Z',
                    'observations': [
                        {'elementId': 'temperature', 'value': 16.0, 'unit': 'degC'},
                        {'elementId': 'precipitation', 'value': 0.0, 'unit': 'mm'}
                    ]
                }
            ]
        }
        mock_get.return_value = mock_response

        # Call the function
        raw_data = RawData()
        result = raw_data.get_met('SN12345', 'temperature,precipitation', '2023-10-01/2023-10-02', 'PT1H')

        # Assertions:   
        self.assertIsInstance(result, pd.DataFrame)     # Is it a dataframe?
        self.assertEqual(result.shape[0], 1)            # Is it the expected shape?
        self.assertIn('referenceTime', result.columns)  # Did the function place the values in the right columns?
    
    # Test that the function handles invalid inputs gracefully
    @patch('requests.get')
    def test_get_met_invalid_input(self, mock_get):

        # Creating a mock API response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            'error': {
                'message': 'Invalid parameter: elements',
                'reason': 'The requested elements are not valid.'
            }     
        }
        mock_get.return_value = mock_response

        # Call the function with invalid inputs
        raw_data = RawData()
        result = raw_data.get_met('SN12345', 'temperature,precipitation,pinkness', '2023-10-01/2023-10-02', 'PT1K')

        #Assertions: Does the function return "None" when given invalid inputs?
        self.assertIsNone(result) 

    # Check if the function handles an API error gracefully
    @patch('requests.get')
    def test_get_met_api_error(self, mock_get):

        # Create a mock API error response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {
            'error': {
                'message': 'Not Found',
                'reason': 'Resource not found'
            }
        }
        mock_get.return_value = mock_response

        # Call the function
        raw_data = RawData()
        result = raw_data.get_met('SN12345', 'temperature,precipitation', '2023-10-01/2023-10-02', 'PT1H')

        # Assertions: Does the function return the expected "None" with an API error?
        self.assertIsNone(result)

    # Check if the function handles a connection error gracefully
    @patch('requests.get')
    def test_get_met_connection_error(self, mock_get):

        # Create a mock connection error
        mock_get.side_effect = requests.exceptions.ConnectionError("Failed to connect")

        # Call the function
        raw_data = RawData()
        result = raw_data.get_met('SN12345', 'temperature,precipitation', '2023-10-01/2023-10-02', 'PT1H')

        # Assertions: Does the function return the expected "None" with a connection error?
        self.assertIsNone(result)


        # The following codeblock was generated with the assistance of AI 
        # - Purpose: Creating a functioning mock CSV file
        # - AI-tool: DeepSeek



    ### TESTING GET_NILU FUNCTION###

    # Test that the function is able to read a csv file, and creates the expected dataframe
    def test_get_nilu_success(self):
        # Create a temporary directory in the data folder
        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(exist_ok=True)  # Ensure directory exists
        
        # Create a temporary file in the data directory
        csv_content = """Header 1
        Header 2
        Header 3
        Tid;Measurement1;Dekning;Measurement2;Dekning.1
        01.10.2023 00:00;1.5;100;2.3;90
        02.10.2023 00:00;1.2;50;2.1;40"""
        
        try:
            # Use a proper temporary file that will auto-delete
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.csv',
                dir=data_dir,
                encoding='utf-8',
                delete=False  # We'll handle deletion manually
            ) as temp_file:
                temp_file.write(csv_content)
                file_path = temp_file.name
            
            print(f"Temporary file created at: {file_path}")
            
            # Call the function
            raw_data = RawData()
            result = raw_data.get_nilu(threshold=70, file_path=file_path)
            
            # Debugging: Print the result if test fails
            print("Resulting DataFrame:")
            print(result)
            
            # Assertions  
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(result.shape[0], 2)
            self.assertIn('Date', result.columns)
            
            # Check column names - the error suggests 'Dekning' might not exist
            print("Columns in result:", result.columns.tolist())
            
            # Verify the column exists before accessing it
            self.assertIn('Dekning', result.columns)
            self.assertEqual(result['Dekning'].iloc[0], 100)
            self.assertEqual(result['Dekning'].iloc[1], 0)
            
        finally:
            # Ensure cleanup happens even if test fails
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Temporary file removed: {file_path}")
            except Exception as e:
                print(f"Error removing temporary file: {e}")

    # Check if the function handles a non-existent file gracefully
    def test_get_nilu_file_not_found(self):

        # Call the method with a non-existent file
        raw_data = RawData()
        result = raw_data.get_nilu(threshold=70, file_path='non_existent_file.csv')

        # Assertions: Does the 
        self.assertIsNone(result)

    # Check if the function handles a malformed CSV file gracefully
    def test_get_nilu_parsing_error(self):

        # Create a malformed CSV file
        with open('malformed.csv', 'w', encoding='utf-8') as f:
            f.write("Tid;Dekning\n01.10.2023 00:00;100\ninvalid_line") # This line was generated by AI (DeepSeek)

        # Call the method
        raw_data = RawData()
        result = raw_data.get_nilu(threshold=70, file_path='malformed.csv')

        # Assertions
        self.assertIsNone(result)

        # Remove the temporary file after use
        os.remove('malformed.csv')


    ### TESTING GET_FORECAST FUNCTION###

    # The code for TestGetForecast code was created entirely by the use of ChatGPT
    # Developer learned the cocnept of mock_response and continued use of self.assertIsInstance and self.assertEqual fucntions. 

    @patch("data_import.requests.get")
    @patch("data_import.pd.Timestamp")
    def test_get_forecast_basic(self, mock_timestamp, mock_requests_get):
        # Prepare mock timestamp
        mock_now = pd.Timestamp("2024-05-01")
        mock_timestamp.now.return_value = mock_now

        # Mock response from MET API (locationforecast)
        met_response = {
            "properties": {
                "timeseries": [
                    {
                        "time": "2024-05-01T00:00:00Z",
                        "data": {
                            "instant": {
                                "details": {
                                    "air_temperature": 10,
                                    "wind_speed": 5
                                }
                            },
                            "next_6_hours": {
                                "details": {
                                    "precipitation_amount": 1
                                }
                            }
                        }
                    },
                    {
                        "time": "2024-05-01T06:00:00Z",
                        "data": {
                            "instant": {
                                "details": {
                                    "air_temperature": 12,
                                    "wind_speed": 6
                                }
                            },
                            "next_6_hours": {
                                "details": {
                                    "precipitation_amount": 2
                                }
                            }
                        }
                    }
                ]
            }
        }

        # Configure mock
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = met_response
        mock_requests_get.return_value = mock_response

        # Run method
        fetcher = RawData()
        df = fetcher.get_forecast()

        # Assert DataFrame contents
        self.assertIsNotNone(df)
        self.assertIn("Date", df.columns)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["Date"], "2024-05-01")
        self.assertAlmostEqual(df.iloc[0]["temperature (C)"], 11)
        self.assertAlmostEqual(df.iloc[0]["wind_speed (m/s)"], 5.5)
        self.assertAlmostEqual(df.iloc[0]["precipitation (mm)"], 3)

if __name__ == "__main__":
    unittest.main()