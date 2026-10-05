import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


FEATURES = [
    "capacity_liters",
    "temperature_c",
    "hour",
    "day_of_week",
    "is_weekend",
    "fill_lag_1h",
    "fill_lag_3h",
    "rolling_mean_6h",
]
TARGET = "next_fill_level_pct"
MODEL_FILENAME = "waste_model.pkl"


def train_ml_model(csv_path="cleaned_waste_data.csv"):
    """Train a leakage-safe one-step-ahead fill-level predictor."""
    print(f"Loading preprocessed dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError("Dataset must contain timestamp")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Dataset contains invalid timestamps")

    missing = set(FEATURES).difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing features: {sorted(missing)}")

    # Predict the next observation for the same bin. This is created after
    # preprocessing so the feature set remains strictly historical.
    df[TARGET] = df.groupby("bin_id")["fill_level_pct"].shift(-1)
    model_df = df.dropna(subset=FEATURES + [TARGET]).copy()
    if len(model_df) < 20:
        raise ValueError("Not enough rows to train the waste prediction model")

    # A time-ordered holdout avoids leaking future observations into training.
    cutoff = model_df["timestamp"].quantile(0.8)
    X_train = model_df.loc[model_df["timestamp"] < cutoff, FEATURES]
    y_train = model_df.loc[model_df["timestamp"] < cutoff, TARGET]
    X_test = model_df.loc[model_df["timestamp"] >= cutoff, FEATURES]
    y_test = model_df.loc[model_df["timestamp"] >= cutoff, TARGET]
    if X_train.empty or X_test.empty:
        raise ValueError("Unable to create chronological train/test split")

    print("Training Random Forest Regressor model...")
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)

    print("Model Evaluation Results:")
    print(f"   - Mean Absolute Error (MAE): {mae:.2f}%")
    print(f"   - Root Mean Squared Error (RMSE): {rmse:.2f}%")

    joblib.dump(model, MODEL_FILENAME)
    print(f"Successfully saved trained model to '{MODEL_FILENAME}'.")

    return model


if __name__ == "__main__":
    train_ml_model()
