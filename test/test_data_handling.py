# This code was modified using the assistance of AI (ChatGPT) 
# See detailed use of AI file for further explanation

import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

# Add the parent directory to the sys.path
import sys
import os
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..',"src")))

from data_handling import RefinedData


class TestRefinedData(unittest.TestCase):
    """Unit tests for the RefinedData class from data_handling module."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data that will be used across all test methods."""
        cls.processor = RefinedData()
        cls.test_df = pd.DataFrame({
            'A': [1, None, 3],
            'B': [4, 5, None]
        })
        cls.clean_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    
    def test_missing_data_report(self):
        """Test the 'report' strategy for missing data."""
        result = self.processor.missing_data(self.test_df, strategy='report')
        expected = pd.DataFrame([
            {'index': 1, 'column': 'A'},
            {'index': 2, 'column': 'B'}
        ])
        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))
    
    def test_missing_data_drop(self):
        """Test the 'drop' strategy for missing data."""
        result = self.processor.missing_data(self.test_df, strategy='drop')
        expected = pd.DataFrame({'A': [1.0], 'B': [4.0]}, index=[0])
        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))
    
    def test_missing_data_fill(self):
        """Test the 'fill' strategy for missing data."""
        result = self.processor.missing_data(self.test_df, strategy='fill', fill_value=0)
        expected = pd.DataFrame({
            'A': [1.0, 0.0, 3.0],
            'B': [4.0, 5.0, 0.0]
        })
        assert_frame_equal(result, expected)
    
    def test_no_missing_data(self):
        """Test behavior when no missing data exists."""
        result = self.processor.missing_data(self.clean_df)
        self.assertIsNone(result)
    
    def test_invalid_input_type(self):
        """Test that non-DataFrame input raises ValueError."""
        with self.assertRaises(ValueError):
            self.processor.missing_data("not a dataframe")
    
    def test_invalid_strategy(self):
        """Test that invalid strategy raises ValueError."""
        with self.assertRaises(ValueError):
            self.processor.missing_data(self.test_df, strategy='invalid')

if __name__ == '__main__':
    unittest.main()