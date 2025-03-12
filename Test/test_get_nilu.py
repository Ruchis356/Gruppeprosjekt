import unittest
import sys, os
import pandas as pd

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_import import RawData

# Successful CSV read: Test that the function correctly reads a CSV file and returns a dataframe.
# Error handling: Test how the function handles some errors (file not found, parsing, other exceptions)
# Dataframe structure: Test that the returned DataFrame has the expected structure
# Threshold: Test that the function correctlt applies the threshold method to the coverage columns

class TestRawData(unittest.TestCase):

    # Test that the function is able to read a csv file, and creates the expected dataframe
    def test_get_nilu_success(self):

        # Create a temporary (mock) CSV file
        csv_content = """Tid;Dekning;Dekning.1;Dekning.2;Dekning.3;Dekning.4
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
        self.assertIn('Tid', result.columns)                # Did the function create the date column correctly? 
        self.assertEqual(result['Dekning'].iloc[0], 100)
        self.assertEqual(result['Dekning'].iloc[1], 0)      # Does the function remove the poor quality data as expected?

        # Remove the temporary file after use
        os.remove('test_nilu.csv')

    # Check if the function handles a non-existent file gracefully
    def test_get_nilu_file_not_found(self):

        # Call the method with a non-existent file
        raw_data = RawData()
        result = raw_data.get_nilu(threshold=70, file_path='non_existent_file.csv')

        # Assertions: Does the 
        self.assertIsNone(result)

    # Check if the function handles a bad CSV file gracefully
    def test_get_nilu_parsing_error(self):

        # Create a malformed CSV file
        with open('malformed.csv', 'w', encoding='utf-8') as f:
            f.write("Tid;Dekning\n01.10.2023 00:00;100\ninvalid_line")

        # Call the method
        raw_data = RawData()
        result = raw_data.get_nilu(threshold=70, file_path='malformed.csv')

        # Assertions
        self.assertIsNone(result)

        # Remove the temporary file after use
        os.remove('malformed.csv')

if __name__ == '__main__':
    unittest.main()