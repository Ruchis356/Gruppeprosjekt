# This code was modified using the assistance of AI (ChatGPT) 
# See detailed use of AI file for further explanation

import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

class RefinedData:
    def missing_data(self, df, strategy='report', fill_value=None):
        if df.isnull().sum().sum() == 0:
            print("No missing data found.")
            return None

        if strategy == 'report':
            return pd.DataFrame([
                {'index': i, 'column': col}
                for col in df.columns
                for i in df[df[col].isnull()].index
            ])

        elif strategy == 'drop':
            return df.dropna()

        elif strategy == 'fill':
            return df.fillna(fill_value)

        else:
            raise ValueError(f"Invalid strategy '{strategy}'. Choose from 'report', 'drop', or 'fill'.")

class TestRefinedData(unittest.TestCase):

    def setUp(self):
        self.processor = RefinedData()
        self.df = pd.DataFrame({
            'A': [1, None, 3],
            'B': [4, 5, None]
        })

    def test_missing_data_report(self):
        result = self.processor.missing_data(self.df, strategy='report')
        expected = pd.DataFrame([
            {'index': 1, 'column': 'A'},
            {'index': 2, 'column': 'B'}
        ])
        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

    def test_missing_data_drop(self):
        result = self.processor.missing_data(self.df, strategy='drop')
        expected = pd.DataFrame({'A': [1.0], 'B': [4.0]})
        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

    def test_missing_data_fill(self):
        result = self.processor.missing_data(self.df, strategy='fill', fill_value=0)
        expected = pd.DataFrame({
            'A': [1.0, 0.0, 3.0],
            'B': [4.0, 5.0, 0.0]
        })
        assert_frame_equal(result, expected)

    def test_missing_data_none(self):
        clean_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        result = self.processor.missing_data(clean_df)
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
