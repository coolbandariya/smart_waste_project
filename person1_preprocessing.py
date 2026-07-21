import numpy as np
import pandas as pd


def generate_synthetic_bin_data(num_bins=20, days=30, seed=42):
    """Generates synthetic IoT waste bin data for demonstration."""
    np.random.seed(seed)
    date_range = pd.date_range(end=pd.Timestamp.now(), periods=days * 24, freq="h")

    data = []
    # Base coordinates (around a central city area)
    base_lat, base_lon = 28.5355, 77.3910

    for bin_id in range(1, num_bins + 1):
        # Assign random location and capacity
        lat = base_lat + np.random.uniform(-0.05, 0.05)
        lon = base_lon + np.random.uniform(-0.05, 0.05)
        capacity_liters = np.random.choice([100, 200, 500])
        criticality = np.random.choice(["High", "Medium", "Low"], p=[0.2, 0.5, 0.3])

        current_fill = np.random.uniform(5, 20)

        for timestamp in date_range:
            # Waste generation rate varies by hour (peak generation during daytime)
            hour = timestamp.hour
            hourly_rate = (
                np.random.uniform(3, 8)
                if 8 <= hour <= 20
                else np.random.uniform(0.5, 2)
            )

            # Accumulate fill level
            current_fill += hourly_rate

            # Simulated empty events (when bin gets collected/cleared)
            if current_fill >= 90 or (
                current_fill > 70 and np.random.rand() < 0.15
            ):
                current_fill = np.random.uniform(0, 5)

            data.append(
                {
                    "timestamp": timestamp,
                    "bin_id": f"BIN_{bin_id:03d}",
                    "latitude": lat,
                    "longitude": lon,
                    "capacity_liters": capacity_liters,
                    "criticality": criticality,
                    "fill_level_pct": round(min(current_fill, 100.0), 2),
                    "temperature_c": round(np.random.uniform(20, 38), 1),
                }
            )

    return pd.DataFrame(data)


def preprocess_data(df):
    """Cleans raw IoT sensor data and performs feature engineering."""
    df = df.copy()

    # Ensure correct datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["bin_id", "timestamp"]).reset_index(drop=True)

    # 1. Handle Missing Values / Outliers
    df["fill_level_pct"] = df["fill_level_pct"].clip(0, 100)

    # 2. Time-Series Feature Engineering
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # 3. Lag & Rolling Features per Bin
    df["fill_lag_1h"] = df.groupby("bin_id")["fill_level_pct"].shift(1)
    df["fill_lag_3h"] = df.groupby("bin_id")["fill_level_pct"].shift(3)
    df["rolling_mean_6h"] = df.groupby("bin_id")["fill_level_pct"].transform(
        lambda x: x.rolling(6, min_periods=1).mean()
    )

    # Fill NaNs created by lagging
    df = df.bfill().ffill()

    return df


if __name__ == "__main__":
    print("Generating raw waste bin dataset...")
    raw_df = generate_synthetic_bin_data()
    print(f"Generated {len(raw_df)} raw records.")

    print("Preprocessing data and engineering features...")
    clean_df = preprocess_data(raw_df)

    # Save to CSV for Person 2 (ML Model)
    output_filename = "cleaned_waste_data.csv"
    clean_df.to_csv(output_filename, index=False)
    print(f"✅ Successfully saved preprocessed data to '{output_filename}'!")