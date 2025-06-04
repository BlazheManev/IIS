import os
import joblib
import random
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from preprocess import DatePreprocessor, SlidingWindowTransformer

# Load training params
params = yaml.safe_load(open("params.yaml"))["train"]
test_size = params["test_size"]
window_size = params["window_size"]
target_col = params["target_col"]
random_state = params["random_state"]
model_dir = params["model_path"]

os.makedirs("models", exist_ok=True)

# Set seeds
os.environ["PYTHONHASHSEED"] = str(random_state)
random.seed(random_state)
np.random.seed(random_state)
tf.random.set_seed(random_state)

# Define model structure
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# Scan all station CSVs
data_dir = "data/preprocessed/air"
for file_name in os.listdir(data_dir):
    if not file_name.endswith(".csv"):
        continue

    station = file_name.replace(".csv", "")
    print(f"\n--- Training for station {station} ---")

    df = pd.read_csv(os.path.join(data_dir, file_name))
    if target_col not in df.columns or "date_to" not in df.columns:
        print(f"Skipping {station}: required columns missing.")
        continue

    df = df[["date_to", target_col]]
    print(f"Original data shape: {df.shape}")

    # Preprocess dates
    date_preprocessor = DatePreprocessor("date_to")
    df = date_preprocessor.fit_transform(df)
    df = df.drop(columns=["date_to"])

    # Split data
    if len(df) <= test_size + window_size:
        print(f"Skipping {station}: not enough data.")
        continue

    df_test = df.iloc[-test_size:]
    df_train = df.iloc[:-test_size]

    numeric_transformer = Pipeline([
        ("fillna", SimpleImputer(strategy="mean")),
        ("normalize", MinMaxScaler())
    ])

    preprocess = ColumnTransformer([
        ("numeric_transformer", numeric_transformer, [target_col])
    ])

    sliding_window_transformer = SlidingWindowTransformer(window_size)

    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("sliding_window_transformer", sliding_window_transformer)
    ])

    # Transform data
    try:
        X_train, y_train = pipeline.fit_transform(df_train)
        X_test, y_test = pipeline.transform(df_test)
    except Exception as e:
        print(f"Failed on station {station}: {e}")
        continue

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)

    # Train
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{station} - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {np.sqrt(mse):.4f}")

    # Save
    model.save(f"models/model_{station}.keras")
    joblib.dump(pipeline, f"models/pipeline_{station}.pkl")
