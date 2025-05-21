# This code was modified using the assistance of AI (ChatGPT) 
# See detailed use of AI file for further explanation

import unittest
import pandas as pd
from pandas.testing import assert_frame_equal


class TestRefinedData(unittest.TestCase):

    def setUp(self):
        # Create an instance of RefinedData for reuse
        self.processor = RefinedData()

        # Sample DataFrame with missing values
        self.df = pd.DataFrame({
            'A': [1, None, 3],
            'B': [4, 5, None]
        })

    def test_missing_data_report(self):
        # Test the 'report' strategy
        result = self.processor.missing_data(self.df, strategy='report')

        # Expected output: a DataFrame with locations of missing values
        expected = pd.DataFrame([
            {'index': 1, 'column': 'A'},
            {'index': 2, 'column': 'B'}
        ])

        # Reset index for comparison
        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

    def test_missing_data_drop(self):
        # Test the 'drop' strategy
        result = self.processor.missing_data(self.df, strategy='drop')

        # Only the first row has no missing values
        expected = pd.DataFrame({
            'A': [1.0],
            'B': [4.0]
        })

        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

    def test_missing_data_fill(self):
        # Test the 'fill' strategy
        result = self.processor.missing_data(self.df, strategy='fill', fill_value=0)

        # All missing values should be replaced with 0
        expected = pd.DataFrame({
            'A': [1.0, 0.0, 3.0],
            'B': [4.0, 5.0, 0.0]
        })

        assert_frame_equal(result, expected)

    def test_missing_data_none(self):
        # Test a DataFrame with no missing values
        clean_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        result = self.processor.missing_data(clean_df)

        # The method should return None and print a message
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
