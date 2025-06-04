import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DatePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Convert date_to to datetime
        X[self.col] = pd.to_datetime(X[self.col])
        # Sort by date_to
        X = X.sort_values(by=self.col)
        # Check for missing hourly data
        date_range = pd.date_range(start=X[self.col].min(), end=X[self.col].max(), freq="h")
        new_df = pd.DataFrame(date_range, columns=[self.col])
        # Merge with original data
        X = pd.merge(new_df, X, on=self.col, how="left")
        return X


class SlidingWindowTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, window_size):
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed, y_transformed = self.create_sliding_windows(X, self.window_size)
        return X_transformed, y_transformed

    @staticmethod
    def create_sliding_windows(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)