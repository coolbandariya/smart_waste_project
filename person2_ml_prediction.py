import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def train_ml_model(csv_path="cleaned_waste_data.csv"):
    print(f"Loading preprocessed dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Features and Target selection
    features = [
        "capacity_liters",
        "temperature_c",
        "hour",
        "day_of_week",
        "is_weekend",
        "fill_lag_1h",
        "fill_lag_3h",
        "rolling_mean_6h",
    ]
    target = "fill_level_pct"

    X = df[features]
    y = df[target]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training Random Forest Regressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions & Evaluation
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)

    print(f" Model Evaluation Results:")
    print(f"   - Mean Absolute Error (MAE): {mae:.2f}%")
    print(f"   - Root Mean Squared Error (RMSE): {rmse:.2f}%")

    # Save the model
    model_filename = "waste_model.pkl"
    joblib.dump(model, model_filename)
    print(f" Successfully saved trained model to '{model_filename}'!")

    return model


if __name__ == "__main__":
    train_ml_model()