import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Assuming your class is named WeatherFetcher and is imported properly
# from your_module import WeatherFetcher

class TestGetForecast(unittest.TestCase):

    @patch("your_module.requests.get")
    @patch("your_module.pd.Timestamp")
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
        fetcher = WeatherFetcher()
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
