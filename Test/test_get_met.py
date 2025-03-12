import sys, os
import pandas as pd
import unittest
import requests
from unittest.mock import patch, Mock

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_import import RawData

class TestRawData(unittest.TestCase):

    # Test that the function correctly fetches data and returns a DataFrame with the expected parameters
    @patch('requests.get')
    def test_get_met_success(self, mock_get):

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

if __name__ == '__main__':
    unittest.main()

