import unittest

import pandas as pd

from person1_preprocessing import preprocess_data


class PreprocessingTests(unittest.TestCase):
    def test_historical_features_exclude_current_target(self):
        rows = []
        for bin_id, values in (("BIN_001", [10, 20, 30]), ("BIN_002", [70, 80, 90])):
            for hour, fill in enumerate(values):
                rows.append(
                    {
                        "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=hour),
                        "bin_id": bin_id,
                        "latitude": 28.5,
                        "longitude": 77.3,
                        "capacity_liters": 200,
                        "criticality": "Medium",
                        "fill_level_pct": fill,
                        "temperature_c": 25,
                    }
                )

        result = preprocess_data(pd.DataFrame(rows))

        first_bin = result[result["bin_id"] == "BIN_001"].reset_index(drop=True)
        self.assertEqual(first_bin.loc[1, "fill_lag_1h"], 10)
        self.assertEqual(first_bin.loc[1, "fill_lag_3h"], 10)
        self.assertEqual(first_bin.loc[1, "rolling_mean_6h"], 10)
        self.assertEqual(first_bin.loc[2, "fill_lag_1h"], 20)
        self.assertEqual(first_bin.loc[2, "rolling_mean_6h"], 15)

    def test_features_do_not_cross_bin_boundaries(self):
        rows = []
        for bin_id, fill in (("BIN_001", 10), ("BIN_002", 90)):
            rows.append(
                {
                    "timestamp": pd.Timestamp("2026-01-01"),
                    "bin_id": bin_id,
                    "latitude": 28.5,
                    "longitude": 77.3,
                    "capacity_liters": 200,
                    "criticality": "Medium",
                    "fill_level_pct": fill,
                    "temperature_c": 25,
                }
            )

        result = preprocess_data(pd.DataFrame(rows))
        self.assertEqual(result.loc[result["bin_id"] == "BIN_001", "fill_lag_1h"].iloc[0], 10)
        self.assertEqual(result.loc[result["bin_id"] == "BIN_002", "fill_lag_1h"].iloc[0], 90)


if __name__ == "__main__":
    unittest.main()
