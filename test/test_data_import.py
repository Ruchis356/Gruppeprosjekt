import sys, os
import pandas as pd
import unittest
import requests
from unittest.mock import patch, Mock

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
                        {'elementId': 'temperature', 'value': 15.0},
                        {'elementId': 'precipitation', 'value': 60.0}
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
        self.assertEqual(result.shape[0], 2)            # Is it the expected shape?
        self.assertIn('referenceTime', result.columns)  # Did the function place the values in the right columns?
    
    # Thest that the function handles invalid inputs gracefully
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

    # Check if the function handles an API error graciously
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

    # Check if the function handles a connection error graciously
    @patch('requests.get')
    def test_get_met_connection_error(self, mock_get):

        # Create a mock connection error
        mock_get.side_effect = requests.exceptions.ConnectionError("Failed to connect")

        # Call the function
        raw_data = RawData()
        result = raw_data.get_met('SN12345', 'temperature,precipitation', '2023-10-01/2023-10-02', 'PT1H')

        # Assertions: Does the function return the expected "None" with a connection error?
        self.assertIsNone(result)







    ### TESTING GET_NILU FUNCTION###


    # Test that the function is able to read a csv file, and creates the expected dataframe
    def test_get_nilu_success(self):

        # The following codeblock was generated with the assistance of AI 
        # - Purpose: Creating a functioning mock CSV file
        # - AI-tool: DeepSeek

        # Create a temporary (mock) CSV file
        csv_content = """Header 1
Header 2
Header 3
Tid;Dekning;Dekning.1;Dekning.2;Dekning.3;Dekning.4
01.10.2023 00:00;100;90;80;70;60
02.10.2023 00:00;50;40;30;20;10"""
        file_path = 'test_nilu.csv'
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(csv_content)
        print(f"Temporary file created at: {os.path.abspath(file_path)}")

        # Call the funtion
        raw_data = RawData()
        result = raw_data.get_nilu(threshold=70, file_path=file_path)

        # Assertions:  
        self.assertIsInstance(result, pd.DataFrame)         # Is it a dataframe? 
        self.assertEqual(result.shape[0], 2)                # Is it the expected shape?        
        self.assertIn('Date', result.columns)                # Did the function create the date column correctly? 
        self.assertEqual(result['Dekning'].iloc[0], 100)
        self.assertEqual(result['Dekning'].iloc[1], 0)      # Does the function remove the poor quality data as expected?

        # Remove the temporary file after use
        os.remove('test_nilu.csv') # This construct was generated by AI (DeepSeek)

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



if __name__ == '__main__':
    unittest.main()


