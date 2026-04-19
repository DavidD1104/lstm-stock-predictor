"""
LSTM Stock Price Predictor — Core Module
Adapted from prediccion_acciones_lstm.py for web integration.

Architecture: 2-layer LSTM → Dense → Output
Uncertainty:  Monte Carlo Dropout (inference-time dropout sampling)
Data source:  Yahoo Finance via yfinance
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class LSTMPredictor:
    """
    LSTM-based stock price predictor.

    Parameters
    ----------
    ticker : str        Stock symbol (e.g. 'AAPL')
    window : int        Look-back window in trading days (default 30)
    epochs : int        Maximum training epochs (default 25)
    units  : int        Units in the first LSTM layer (default 100)
    """

    def __init__(self, ticker: str, window: int = 30, epochs: int = 25, units: int = 100):
        self.ticker = ticker
        self.window = window
        self.epochs = epochs
        self.units  = units

        self.scaler: MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
        self.model: Sequential | None = None
        self.prices: np.ndarray | None = None
        self.dates: list[str] | None = None

    # ── Data ─────────────────────────────────────────────────────────────────

    def fetch_data(self, start_date: str, end_date: str) -> tuple[np.ndarray, list[str]]:
        """Download adjusted closing prices from Yahoo Finance."""
        df = yf.download(
            self.ticker, start=start_date, end=end_date,
            progress=False, auto_adjust=True
        )
        if df.empty:
            raise ValueError(
                f"No se encontraron datos para {self.ticker} "
                f"en el rango {start_date} → {end_date}."
            )
        # yfinance ≥0.2 may return MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        self.prices = df["Close"].values.flatten().astype(np.float64)
        self.dates  = df.index.strftime("%Y-%m-%d").tolist()
        return self.prices, self.dates

    # ── Preprocessing ────────────────────────────────────────────────────────

    def _make_sequences(self, scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(self.window, len(scaled)):
            X.append(scaled[i - self.window : i, 0])
            y.append(scaled[i, 0])
        X = np.array(X).reshape(-1, self.window, 1)
        y = np.array(y)
        return X, y

    def prepare_data(self) -> tuple:
        """Scale prices and create train/test sequences."""
        scaled = self.scaler.fit_transform(self.prices.reshape(-1, 1))
        X, y   = self._make_sequences(scaled)
        split  = int(len(X) * 0.8)
        return X[:split], X[split:], y[:split], y[split:], split

    # ── Model ────────────────────────────────────────────────────────────────

    def build_model(self):
        """Build a 2-layer stacked LSTM model."""
        self.model = Sequential([
            LSTM(self.units,
                 input_shape=(self.window, 1),
                 return_sequences=True,
                 dropout=0.2),
            LSTM(50,
                 dropout=0.2),
            Dense(25, activation="relu"),
            Dropout(0.15),
            Dense(1, activation="linear"),
        ])
        self.model.compile(
            optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
            loss="mse",
            metrics=["mae"],
        )

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              progress_callback=None) -> dict:
        """
        Train the model.

        Parameters
        ----------
        progress_callback : callable(epoch: int, logs: dict) | None
            Called at the end of each epoch with epoch number (1-based) and
            Keras logs dict containing 'loss' and 'val_loss'.
        """

        class _EpochCB(tf.keras.callbacks.Callback):
            def on_epoch_end(self_cb, epoch: int, logs: dict = None):
                if progress_callback and logs:
                    progress_callback(epoch + 1, logs)

        callbacks = [
            _EpochCB(),
            EarlyStopping(monitor="val_loss", patience=10,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=5, min_lr=1e-6, verbose=0),
        ]

        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=32,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=0,
            shuffle=False,
        )
        return history.history

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Run model on test set and compute performance metrics."""
        pred_scaled = self.model.predict(X_test, verbose=0)
        pred   = self.scaler.inverse_transform(pred_scaled).flatten()
        actual = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        mae  = float(mean_absolute_error(actual, pred))
        rmse = float(np.sqrt(mean_squared_error(actual, pred)))
        mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
        ss_res = np.sum((actual - pred) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "mae":         mae,
            "rmse":        rmse,
            "mape":        mape,
            "r2":          r2,
            "predictions": pred.tolist(),
            "actual":      actual.tolist(),
            "errors":      (pred - actual).tolist(),
        }

    # ── Forecast ─────────────────────────────────────────────────────────────

    def forecast(self, n_days: int = 10, n_simulations: int = 40) -> dict:
        """
        Multi-step forecast with Monte Carlo dropout for uncertainty bands.

        Uses training=True at inference time to keep dropout active, producing
        a distribution of predictions that estimates model uncertainty.
        """
        last_window = self.scaler.transform(
            self.prices[-self.window :].reshape(-1, 1)
        )

        all_sims: list[list[float]] = []
        for _ in range(n_simulations):
            window = last_window.copy()
            sim: list[float] = []
            for _ in range(n_days):
                x    = window.reshape(1, self.window, 1)
                pred = float(self.model(x, training=True).numpy()[0, 0])
                pred = float(np.clip(pred, 0.0, 1.0))
                sim.append(pred)
                window = np.vstack([window[1:], [[pred]]])
            all_sims.append(sim)

        sims     = np.array(all_sims)          # (n_simulations, n_days)
        mean_p   = np.mean(sims, axis=0)
        std_p    = np.std(sims, axis=0)

        def inv(arr: np.ndarray) -> list[float]:
            clipped = np.clip(arr, 0.0, 1.0)
            return self.scaler.inverse_transform(clipped.reshape(-1, 1)).flatten().tolist()

        last_date    = pd.Timestamp(self.dates[-1])
        future_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1), periods=n_days
        )

        return {
            "dates":      future_dates.strftime("%Y-%m-%d").tolist(),
            "mean":       inv(mean_p),
            "upper_95":   inv(mean_p + 2 * std_p),
            "lower_95":   inv(mean_p - 2 * std_p),
            "upper_68":   inv(mean_p + std_p),
            "lower_68":   inv(mean_p - std_p),
        }

    # ── Full Pipeline ────────────────────────────────────────────────────────

    def run(
        self,
        start_date: str,
        end_date: str,
        progress_callback=None,
        phase_callback=None,
        forecast_days: int = 10,
        include_forecast: bool = True,
    ) -> dict:
        """
        End-to-end pipeline: fetch → prepare → build → train → evaluate → forecast.

        Returns a dict ready to be serialized as JSON for the frontend.

        Parameters
        ----------
        phase_callback : callable(phase: str, progress: int) | None
            Called when the pipeline transitions between phases.
        include_forecast : bool
            If False, skip Monte Carlo forecast (faster).
        """
        # 1. Data
        if phase_callback:
            phase_callback("download", 2)
        prices, dates = self.fetch_data(start_date, end_date)
        if len(prices) < self.window + 20:
            raise ValueError(
                f"Datos insuficientes ({len(prices)} días). "
                "Amplía el rango de fechas."
            )

        # 2. Preprocessing
        if phase_callback:
            phase_callback("preparing", 5)
        X_train, X_test, y_train, y_test, split = self.prepare_data()

        # 3. Build
        if phase_callback:
            phase_callback("building", 8)
        self.build_model()

        # 4. Train
        if phase_callback:
            phase_callback("training", 10)
        history = self.train(X_train, y_train, progress_callback)

        # 5. Evaluate
        if phase_callback:
            phase_callback("evaluating", 80 if include_forecast else 90)
        eval_res = self.evaluate(X_test, y_test)

        # 6. Forecast (optional)
        forecast = None
        if include_forecast and forecast_days > 0:
            if phase_callback:
                phase_callback("forecasting", 90)
            forecast = self.forecast(n_days=forecast_days)

        # Align test predictions with dates
        test_start = split + self.window
        test_dates = dates[test_start : test_start + len(eval_res["predictions"])]

        epochs_run = len(history.get("loss", []))

        return {
            "ticker": self.ticker,
            "historical": {
                "dates":  dates,
                "prices": prices.tolist(),
            },
            "test": {
                "dates":       test_dates,
                "predictions": eval_res["predictions"],
                "actual":      eval_res["actual"],
                "errors":      eval_res["errors"],
            },
            "forecast": forecast,
            "metrics": {
                "mae":        eval_res["mae"],
                "rmse":       eval_res["rmse"],
                "mape":       eval_res["mape"],
                "r2":         eval_res["r2"],
                "train_size": len(X_train),
                "test_size":  len(X_test),
            },
            "training_history": {
                "loss":       [float(x) for x in history.get("loss", [])],
                "val_loss":   [float(x) for x in history.get("val_loss", [])],
                "epochs_run": epochs_run,
            },
        }
