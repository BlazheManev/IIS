import os
import joblib
import random
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
import tf2onnx
import mlflow
import mlflow.tensorflow

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from preprocess import DatePreprocessor, SlidingWindowTransformer

# Load training config
params = yaml.safe_load(open("params.yaml"))["train"]
test_size = params["test_size"]
window_size = params["window_size"]
target_col = params["target_col"]
random_state = params["random_state"]
model_dir = params["model_path"]

os.makedirs(model_dir, exist_ok=True)

# Reproducibility
os.environ["PYTHONHASHSEED"] = str(random_state)
random.seed(random_state)
np.random.seed(random_state)
tf.random.set_seed(random_state)

# DagsHub MLflow integration
mlflow.set_experiment("iis_training")

# Loop through stations
data_dir = "data/preprocessed/air"
for file_name in os.listdir(data_dir):
    if not file_name.endswith(".csv"):
        continue

    station = file_name.replace(".csv", "")
    print(f"\n🔧 Training for station: {station}")

    df = pd.read_csv(os.path.join(data_dir, file_name))
    if target_col not in df.columns or "date_to" not in df.columns:
        print(f"⚠️ Skipping {station}: required columns missing.")
        continue

    df = df[["date_to", target_col]]
    date_preprocessor = DatePreprocessor("date_to")
    df = date_preprocessor.fit_transform(df).drop(columns=["date_to"])

    if len(df) <= test_size + window_size:
        print(f"⚠️ Skipping {station}: not enough data.")
        continue

    df_train = df.iloc[:-test_size]
    df_test = df.iloc[-test_size:]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler())
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_transformer, [target_col])
    ])

    sliding = SlidingWindowTransformer(window_size)
    pipeline = Pipeline([
        ("pre", preprocess),
        ("window", sliding)
    ])

    try:
        X_train, y_train = pipeline.fit_transform(df_train)
        X_test, y_test = pipeline.transform(df_test)
    except Exception as e:
        print(f"❌ Failed for {station}: {e}")
        continue

    input_shape = (X_train.shape[1], X_train.shape[2])

    def build_model(shape):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    with mlflow.start_run(run_name=f"train_{station}"):
        mlflow.log_params({
            "station": station,
            "test_size": test_size,
            "window_size": window_size,
            "target_col": target_col,
            "random_state": random_state
        })

        mlflow.tensorflow.autolog()

        model = build_model(input_shape)
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        print(f"🚀 Training model for {station}...")
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=1)

        print(f"📊 Evaluating model for {station}...")
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        mlflow.log_metrics({
            "mae": mae,
            "mse": mse,
            "rmse": rmse
        })

        print(f"✅ {station} - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

        # Save models
        keras_path = f"{model_dir}/model_{station}.keras"
        model.save(keras_path)
        mlflow.log_artifact(keras_path)

        onnx_path = f"{model_dir}/model_{station}.onnx"
        onnx_model, _ = tf2onnx.convert.from_keras(model)
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        mlflow.log_artifact(onnx_path)

        # Save pipeline
        pipeline_path = f"{model_dir}/pipeline_{station}.pkl"
        joblib.dump(pipeline, pipeline_path)
        mlflow.log_artifact(pipeline_path)

print("\n🏁 All stations processed.")
