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

# Load training params
params = yaml.safe_load(open("params.yaml"))["train"]
test_size = params["test_size"]
window_size = params["window_size"]
target_col = params["target_col"]
random_state = params["random_state"]
model_dir = params["model_path"]

os.makedirs(model_dir, exist_ok=True)

# Set seeds
os.environ["PYTHONHASHSEED"] = str(random_state)
random.seed(random_state)
np.random.seed(random_state)
tf.random.set_seed(random_state)

# MLflow tracking URI (DagsHub)
mlflow.set_tracking_uri("https://dagshub.com/BlazheManev/IIS.mlflow")

# Ensure experiment exists
experiment_name = "iis_training"
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# Epoch logger callback
class EpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch+1:02d} — loss: {logs.get('loss'):.4f} — val_loss: {logs.get('val_loss'):.4f}")

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
    df = date_preprocessor.fit_transform(df)
    df = df.drop(columns=["date_to"])

    if len(df) <= test_size + window_size:
        print(f"⚠️ Skipping {station}: not enough data.")
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

    try:
        X_train, y_train = pipeline.fit_transform(df_train)
        X_test, y_test = pipeline.transform(df_test)
    except Exception as e:
        print(f"❌ Failed on station {station}: {e}")
        continue

    input_shape = (X_train.shape[1], X_train.shape[2])

    def build_model(input_shape):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    with mlflow.start_run(run_name=f"train_{station}"):

        # Log parameters
        mlflow.log_param("station", station)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("window_size", window_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("target_col", target_col)

        mlflow.tensorflow.autolog()

        model = build_model(input_shape)
        early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        print("🚀 Starting model training...")
        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, EpochLogger()],
            verbose=0
        )
        print("✅ Training completed.")

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        print(f"📊 {station} — MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)

        # Save Keras model
        keras_path = f"{model_dir}/model_{station}.keras"
        model.save(keras_path)
        mlflow.log_artifact(keras_path)

        # Save ONNX model
        onnx_path = f"{model_dir}/model_{station}.onnx"
        onnx_model, _ = tf2onnx.convert.from_keras(model)
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        mlflow.log_artifact(onnx_path)

        # Save pipeline
        pipeline_path = f"{model_dir}/pipeline_{station}.pkl"
        joblib.dump(pipeline, pipeline_path)
        mlflow.log_artifact(pipeline_path)

print("\n🏁 All stations trained successfully.")
