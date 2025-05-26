
# This code was created and modified using the assistance of AI (ChatGPT) 
# See detailed use of AI file for further explanation

# - averages

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
# Add the parent directory to the sys.path
import sys
import os
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..',"src")))
from analysis import AnalysedData 

class TestAverages(unittest.TestCase):
    def setUp(self):
        self.processor = AnalysedData()

    def test_valid_data_returns_weekly_averages(self):
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=14, freq='D'),
            'A': [1, 2, 3, 4, 5, 6, 7, np.nan, 9, 10, 11, 12, 13]
        })
        result = self.processor.averages(df, ['A'])

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)  # Expecting 2 weeks of data
        self.assertIn('A', result.columns)
        self.assertIn('date', result.columns)
        self.assertAlmostEqual(result['A'].iloc[0], np.mean([1,2,3,4,5,6,7]), places=2)

    @patch('analysis.logger') 
    def test_missing_column_logged_and_skipped(self, mock_logger):
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'A': [1, 2, 3, 4, 5]
        })
        result = self.processor.averages(df, ['A', 'B'])  # 'B' missing
        self.assertIsNotNone(result)
        self.assertIn('A', result.columns)
        mock_logger.warning.assert_called()

    @patch('analysis.logger') # AI suggested the us of @patch and mock_logger to verify logging behavior without producing actual log output
    def test_all_invalid_columns_returns_none(self, mock_logger):
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'A': [1, 2, 3, 4, 5]
        })
        result = self.processor.averages(df, ['B'])  # 'B' is not present
        self.assertIsNone(result)
        mock_logger.error.assert_called_with("No valid columns to process")

    @patch('analysis.logger')
    def test_invalid_date_column_returns_none(self, mock_logger):
        df = pd.DataFrame({
            'not_a_date': ['x', 'y', 'z', 'a', 'b'],
            'A': [1, 2, 3, 4, 5]
        })
        result = self.processor.averages(df, ['A'])
        self.assertIsNone(result)
        self.assertTrue(any("Could not parse date column" in call.args[0] for call in mock_logger.error.call_args_list))

    def test_week_with_less_than_three_values_is_none(self):
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=7),
            'A': [np.nan, 1, np.nan, 2, np.nan, np.nan, 3]  # Only 3 values in week
        })
        result = self.processor.averages(df, ['A'])

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertTrue(np.isnan(result['A'].iloc[0]))  # mean should be None → NaN in DataFrame

    @patch('analysis.logger')
    def test_exception_during_grouping_logged(self, mock_logger):
        # Force an exception by giving invalid data (e.g., object in numeric column)
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'A': ['a', 'b', 'c', 'd', 'e']
        })
        result = self.processor.averages(df, ['A'])
        self.assertIsNone(result)
        self.assertTrue(any("Weekly average calculation failed" in call.args[0] for call in mock_logger.error.call_args_list))


if __name__ == '__main__':
    unittest.main()


# - total_average


from unittest.mock import patch, MagicMock

import AnalysedData
from analysis import AnalysedData  


class TestTotalAverage(unittest.TestCase):
    def setUp(self):
        self.processor = AnalysedData()

    def test_valid_columns_return_averages(self):
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, np.nan],
            'B': [10, 20, 30, 40, 50],
        })
        result = self.processor.total_average(df, ['A', 'B'])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(set(result.columns), {'Metric', 'Average'})
        self.assertEqual(len(result), 2)

        # Check correct average calculations
        avg_a = df['A'].mean()
        avg_b = df['B'].mean()
        self.assertAlmostEqual(result.loc[result['Metric'] == 'A', 'Average'].values[0], avg_a)
        self.assertAlmostEqual(result.loc[result['Metric'] == 'B', 'Average'].values[0], avg_b)

    @patch('analysis.logger')
    def test_missing_column_logs_warning_and_sets_none(self, mock_logger):
        df = pd.DataFrame({'A': [1, 2, 3]})
        result = self.processor.total_average(df, ['A', 'B'])  # 'B' missing

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertTrue(np.isnan(result.loc[result['Metric'] == 'B', 'Average']).values[0] or
                        result.loc[result['Metric'] == 'B', 'Average'].values[0] is None)

        mock_logger.warning.assert_any_call("Column not found: B")

    @patch('analysis.logger')
    def test_no_valid_columns_returns_none_and_logs_error(self, mock_logger):
        df = pd.DataFrame({'A': [1, 2, 3]})
        result = self.processor.total_average(df, ['X', 'Y'])  # No valid columns
        self.assertIsNone(result)
        mock_logger.error.assert_called_with("No processable columns found")

    def test_invalid_df_input_raises_value_error(self):
        with self.assertRaises(ValueError) as context:
            self.processor.total_average("not a df", ['A'])
        self.assertIn("Input must be a pandas DataFrame", str(context.exception))

    def test_invalid_column_names_type_raises_value_error(self):
        df = pd.DataFrame({'A': [1, 2, 3]})
        with self.assertRaises(ValueError) as context:
            self.processor.total_average(df, "A")  # should be list or tuple
        self.assertIn("column_names must be a list or tuple", str(context.exception))

    @patch('analysis.logger')
    def test_exception_in_column_calculation_logged(self, mock_logger):
        df = pd.DataFrame({'A': [1, 2, 3]})

        # Monkeypatch df[col].mean to throw error for testing
        original_mean = pd.Series.mean

        def raise_error(*args, **kwargs):
            raise Exception("Forced error")

        pd.Series.mean = raise_error

        try:
            result = self.processor.total_average(df, ['A'])
            self.assertIsInstance(result, pd.DataFrame)
            self.assertTrue(np.isnan(result.loc[result['Metric'] == 'A', 'Average']).values[0] or
                            result.loc[result['Metric'] == 'A', 'Average'].values[0] is None)
            mock_logger.warning.assert_any_call("Error calculating A: Forced error")
        finally:
            pd.Series.mean = original_mean  # Restore original method


if __name__ == '__main__':
    unittest.main()


# - standard_deviation

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch #The following blocks of code were generated by AI, source: ChatGPT

class TestStandardDeviation(unittest.TestCase):
    def setUp(self):
        self.analysis = AnalysedData()

    def test_valid_columns_return_std(self):
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, np.nan],
            'B': [10, 20, 30, 40, 50],
        })
        result = self.analysis.standard_deviation(df, ['A', 'B'])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(set(result.columns), {'Metric', 'Standard Deviation'})
        self.assertEqual(len(result), 2)

        std_a = df['A'].std(ddof=0)
        std_b = df['B'].std(ddof=0)
        self.assertAlmostEqual(result.loc[result['Metric'] == 'A', 'Standard Deviation'].values[0], std_a)
        self.assertAlmostEqual(result.loc[result['Metric'] == 'B', 'Standard Deviation'].values[0], std_b)

    @patch('analysis.logger')
    def test_missing_column_logs_warning_and_sets_none(self, mock_logger):
        df = pd.DataFrame({'A': [1, 2, 3]})
        result = self.analysis.standard_deviation(df, ['A', 'C'])  # 'C' missing

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertTrue(np.isnan(result.loc[result['Metric'] == 'C', 'Standard Deviation']).values[0] or
                        result.loc[result['Metric'] == 'C', 'Standard Deviation'].values[0] is None)

        mock_logger.warning.assert_any_call("Column not found: C")

    @patch('analysis.logger')
    def test_no_valid_columns_returns_none_and_logs_error(self, mock_logger):
        df = pd.DataFrame({'A': [1, 2, 3]})
        result = self.analysis.standard_deviation(df, ['X', 'Y'])  # no valid cols
        self.assertIsNone(result)
        mock_logger.error.assert_called_with("No valid columns to process")

    def test_invalid_df_raises_value_error(self):
        with self.assertRaises(ValueError) as cm:
            self.analysis.standard_deviation("not_a_df", ['A'])
        self.assertIn("Input must be a pandas DataFrame", str(cm.exception))

    def test_invalid_column_names_raises_value_error(self):
        df = pd.DataFrame({'A': [1, 2, 3]})
        with self.assertRaises(ValueError) as cm:
            self.analysis.standard_deviation(df, "A")  # Should be list or tuple
        self.assertIn("column_names must be a list or tuple", str(cm.exception))

    @patch('analysis.logger') 
    def test_exception_in_std_calculation_logs_warning(self, mock_logger):
        df = pd.DataFrame({'A': [1, 2, 3]})

        original_std = pd.Series.std
        def raise_error(*args, **kwargs):
            raise Exception("Forced error")

        pd.Series.std = raise_error

        try:
            result = self.analysis.standard_deviation(df, ['A'])
            self.assertIsInstance(result, pd.DataFrame)
            self.assertTrue(np.isnan(result.loc[result['Metric'] == 'A', 'Standard Deviation']).values[0] or
                            result.loc[result['Metric'] == 'A', 'Standard Deviation'].values[0] is None)
            mock_logger.warning.assert_any_call("Error calculating A: Forced error")
        finally:
            pd.Series.std = original_std


if __name__ == '__main__':
    unittest.main()


# - outliers


class TestOutliersMethod(unittest.TestCase):
    def setUp(self):
        self.processor = AnalysedData()

        self.df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'A': [10, 15, 10, 100, 12],  # 100 is an outlier
            'B': [1, 2, 3, 4, 5]
        })

        self.std_df = pd.DataFrame({
            'Metric': ['A', 'B'],
            'Standard Deviation': [5, 1.4]
        })

        self.avg_df = pd.DataFrame({
            'Metric': ['A', 'B'],
            'Average': [15, 3]
        })

    def test_outliers_found(self):
        outliers_df, df_x_outliers = self.processor.outliers(
            self.df, ['date', 'A', 'B'], self.std_df, self.avg_df, sd_modifier=2
        )

        self.assertIsInstance(outliers_df, pd.DataFrame)
        self.assertIn('A', outliers_df.columns)
        self.assertIn(100, outliers_df['A'].values)
        self.assertTrue(np.isnan(df_x_outliers.loc[self.df['A'] == 100, 'A']).all())

    def test_no_outliers(self):
        df_no_outliers = self.df.copy()
        df_no_outliers['A'] = [12, 13, 14, 15, 16]

        outliers_df, df_x_outliers = self.processor.outliers(
            df_no_outliers, ['date', 'A', 'B'], self.std_df, self.avg_df, sd_modifier=2
        )

        self.assertIsNone(outliers_df)
        pd.testing.assert_frame_equal(df_no_outliers, df_x_outliers)

    @patch('analysis.logger')
    def test_missing_columns_logs_warning(self, mock_logger):
        outliers_df, df_x_outliers = self.processor.outliers(
            self.df, ['date', 'A', 'C'], self.std_df, self.avg_df, sd_modifier=2
        )

        mock_logger.warning.assert_called()
        self.assertIn('Some columns not found', mock_logger.warning.call_args[0][0])

    def test_invalid_standard_deviation_type_raises(self):
        with self.assertRaises(ValueError):
            self.processor.outliers(self.df, ['date', 'A'], standard_deviation='not a df', average=self.avg_df)

    def test_invalid_average_type_raises(self):
        with self.assertRaises(ValueError):
            self.processor.outliers(self.df, ['date', 'A'], standard_deviation=self.std_df, average='not a df')


if __name__ == '__main__':
    unittest.main()


