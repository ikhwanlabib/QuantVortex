"""
Machine-learning alpha generation strategy for QuantVortex.

Combines XGBoost and PyTorch LSTM models via ensemble prediction, uses
SHAP for feature importance, and performs walk-forward validation with
TimeSeriesSplit.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import torch
import torch.nn as nn
import shap

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PyTorch LSTM module
# ---------------------------------------------------------------------------

class _LSTMNet(nn.Module):
    """Lightweight LSTM regressor.

    Parameters
    ----------
    input_size : int
        Number of input features per time step.
    hidden_size : int
        Number of LSTM hidden units.
    num_layers : int, optional
        Number of stacked LSTM layers. Default is 2.
    dropout : float, optional
        Dropout probability between LSTM layers. Default is 0.2.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, seq_len, input_size)``.

        Returns
        -------
        torch.Tensor
            Predictions of shape ``(batch, 1)``.
        """
        out, _ = self.lstm(x)
        # Use the last time-step output
        return self.fc(out[:, -1, :])


# ---------------------------------------------------------------------------
# MLAlpha strategy
# ---------------------------------------------------------------------------

class MLAlpha(BaseStrategy):
    """ML-based alpha generation using XGBoost + LSTM ensemble.

    Parameters
    ----------
    name : str, optional
        Strategy name. Default is ``"MLAlpha"``.
    allocation : float, optional
        Capital allocation fraction. Default is 1.0.
    xgb_weight : float, optional
        Weight of XGBoost predictions in the ensemble. Default is 0.6.
    lstm_weight : float, optional
        Weight of LSTM predictions in the ensemble. Default is 0.4.
    lstm_hidden_size : int, optional
        LSTM hidden dimension. Default is 64.
    lstm_seq_len : int, optional
        Sequence length (look-back) fed to LSTM. Default is 20.
    lstm_epochs : int, optional
        Training epochs for LSTM. Default is 30.
    lstm_lr : float, optional
        Learning rate for LSTM Adam optimiser. Default is 1e-3.
    long_n : int, optional
        Number of top assets in the long book. Default is 5.
    short_n : int, optional
        Number of bottom assets in the short book. Default is 5.

    Attributes
    ----------
    xgb_model : xgb.XGBRegressor or None
        Fitted XGBoost model.
    lstm_model : _LSTMNet or None
        Fitted LSTM model.
    scaler : StandardScaler
        Feature scaler (fitted during training).
    feature_columns : list of str
        Names of engineered features.
    """

    def __init__(
        self,
        name: str = "MLAlpha",
        allocation: float = 1.0,
        xgb_weight: float = 0.6,
        lstm_weight: float = 0.4,
        lstm_hidden_size: int = 64,
        lstm_seq_len: int = 20,
        lstm_epochs: int = 30,
        lstm_lr: float = 1e-3,
        long_n: int = 5,
        short_n: int = 5,
    ) -> None:
        super().__init__(name=name, allocation=allocation)
        _WEIGHT_SUM_TOLERANCE = 1e-6
        if abs(xgb_weight + lstm_weight - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError("xgb_weight + lstm_weight must equal 1.0.")
        self.xgb_weight = xgb_weight
        self.lstm_weight = lstm_weight
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_seq_len = lstm_seq_len
        self.lstm_epochs = lstm_epochs
        self.lstm_lr = lstm_lr
        self.long_n = long_n
        self.short_n = short_n
        self.xgb_model: Optional[xgb.XGBRegressor] = None
        self.lstm_model: Optional[_LSTMNet] = None
        self.scaler: StandardScaler = StandardScaler()
        self.feature_columns: List[str] = []
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info("MLAlpha will use device: %s", self._device)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Construct a rich feature matrix from price/volume data.

        Features created:
        - Returns: 1-, 5-, 21-, 63-day simple returns.
        - Realised volatility: 21- and 63-day rolling std of 1-day returns.
        - RSI (14-day).
        - Momentum: 126-day minus 21-day cumulative return.
        - Volume ratio: 5-day average volume / 21-day average volume
          (requires a ``"volume"`` column; silently omitted if absent).

        Parameters
        ----------
        data : pd.DataFrame
            Price (and optionally volume) DataFrame with DatetimeIndex and
            ticker/feature columns.  If ``data`` contains a single column
            called ``"close"`` (or is single-column), it is treated as a
            price series.

        Returns
        -------
        pd.DataFrame
            Feature matrix with DatetimeIndex, NaN rows dropped.
        """
        if data.empty:
            raise ValueError("engineer_features: data is empty.")

        # Support both wide (multi-ticker) and single-price-column formats
        price_col = "close" if "close" in data.columns else data.columns[0]
        prices = data[price_col].astype(float)

        has_volume = "volume" in data.columns
        features: Dict[str, pd.Series] = {}

        ret = prices.pct_change()
        for window in [1, 5, 21, 63]:
            features[f"ret_{window}d"] = prices.pct_change(window)

        for window in [21, 63]:
            features[f"vol_{window}d"] = ret.rolling(window).std()

        # RSI (14-day)
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        features["rsi_14"] = 100 - 100 / (1 + rs)

        # Momentum (skip-1-month)
        features["momentum_105d"] = (
            prices.pct_change(126) - prices.pct_change(21)
        )

        if has_volume:
            vol_series = data["volume"].astype(float)
            vol5 = vol_series.rolling(5).mean()
            vol21 = vol_series.rolling(21).mean()
            features["volume_ratio"] = vol5 / vol21.replace(0, np.nan)

        feat_df = pd.DataFrame(features, index=data.index).dropna()
        self.feature_columns = feat_df.columns.tolist()
        self.logger.info(
            "engineer_features: %d rows × %d features",
            len(feat_df),
            len(self.feature_columns),
        )
        return feat_df

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------

    def train_xgboost(
        self, X: pd.DataFrame, y: pd.Series
    ) -> xgb.XGBRegressor:
        """Train an XGBoost regressor for return prediction.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (rows = samples, columns = features).
        y : pd.Series
            Target return series aligned to ``X``.

        Returns
        -------
        xgb.XGBRegressor
            Fitted model.

        Raises
        ------
        ValueError
            If ``X`` and ``y`` have different lengths.
        """
        if len(X) != len(y):
            raise ValueError(f"X ({len(X)}) and y ({len(y)}) length mismatch.")

        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )
        model.fit(X, y)
        self.xgb_model = model
        self.logger.info(
            "XGBoost trained: %d estimators, train MSE=%.6f",
            model.n_estimators,
            float(mean_squared_error(y, model.predict(X))),
        )
        return model

    def train_lstm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hidden_size: int = 64,
    ) -> _LSTMNet:
        """Train a PyTorch LSTM on sequential feature windows.

        Parameters
        ----------
        X : np.ndarray
            Feature array of shape ``(N, seq_len, n_features)`` or
            ``(N, n_features)`` (will be reshaped to seq_len windows).
        y : np.ndarray
            Target returns of shape ``(N,)``.
        hidden_size : int, optional
            Override ``self.lstm_hidden_size``. Default is 64.

        Returns
        -------
        _LSTMNet
            Trained LSTM model in evaluation mode.

        Raises
        ------
        ValueError
            If ``X`` and ``y`` are incompatible in length.
        """
        hs = hidden_size if hidden_size else self.lstm_hidden_size

        if X.ndim == 2:
            # Build (N - seq_len + 1) windows of length seq_len
            seq_len = self.lstm_seq_len
            n_features = X.shape[1]
            n_samples = X.shape[0] - seq_len + 1
            if n_samples <= 0:
                raise ValueError(
                    f"X has too few rows ({X.shape[0]}) for seq_len={seq_len}."
                )
            X_seq = np.stack(
                [X[i : i + seq_len] for i in range(n_samples)], axis=0
            )
            y_seq = y[seq_len - 1 :]
        elif X.ndim == 3:
            X_seq = X
            y_seq = y
            n_features = X.shape[2]
        else:
            raise ValueError("X must be 2-D or 3-D.")

        if len(X_seq) != len(y_seq):
            raise ValueError(
                f"Windowed X ({len(X_seq)}) and y ({len(y_seq)}) mismatch."
            )

        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self._device)
        y_t = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1).to(self._device)

        model = _LSTMNet(
            input_size=n_features, hidden_size=hs, num_layers=2, dropout=0.2
        ).to(self._device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.lstm_lr)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(self.lstm_epochs):
            optimiser.zero_grad()
            preds = model(X_t)
            loss = criterion(preds, y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            if (epoch + 1) % 10 == 0:
                self.logger.debug(
                    "LSTM epoch %d/%d, loss=%.6f", epoch + 1, self.lstm_epochs, loss.item()
                )

        model.eval()
        self.lstm_model = model
        self.logger.info(
            "LSTM trained: %d epochs, final loss=%.6f",
            self.lstm_epochs,
            float(loss.item()),
        )
        return model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def ensemble_predict(self, X: pd.DataFrame) -> pd.Series:
        """Weighted average of XGBoost and LSTM predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with the same columns as training data.

        Returns
        -------
        pd.Series
            Ensemble predictions indexed by ``X``'s index.

        Raises
        ------
        RuntimeError
            If neither XGBoost nor LSTM has been trained.
        """
        if self.xgb_model is None and self.lstm_model is None:
            raise RuntimeError("No trained models available; call train_xgboost or train_lstm first.")

        X_scaled = self.scaler.transform(X.values)

        preds_xgb = np.zeros(len(X))
        preds_lstm = np.zeros(len(X))

        if self.xgb_model is not None:
            preds_xgb = self.xgb_model.predict(X_scaled)

        if self.lstm_model is not None:
            seq_len = self.lstm_seq_len
            if len(X_scaled) < seq_len:
                self.logger.warning(
                    "Fewer rows (%d) than LSTM seq_len (%d); padding with zeros.",
                    len(X_scaled),
                    seq_len,
                )
                pad = np.zeros((seq_len - len(X_scaled), X_scaled.shape[1]))
                X_padded = np.vstack([pad, X_scaled])
            else:
                X_padded = X_scaled

            n = len(X_scaled)
            preds_list: List[float] = []
            self.lstm_model.eval()
            with torch.no_grad():
                for i in range(len(X_padded) - seq_len + 1):
                    window = torch.tensor(
                        X_padded[i : i + seq_len][np.newaxis, :, :],
                        dtype=torch.float32,
                    ).to(self._device)
                    pred = self.lstm_model(window).item()
                    preds_list.append(pred)
            # Align to the last n predictions
            preds_lstm_raw = np.array(preds_list)
            if len(preds_lstm_raw) >= n:
                preds_lstm = preds_lstm_raw[-n:]
            else:
                preds_lstm = np.zeros(n)
                preds_lstm[n - len(preds_lstm_raw) :] = preds_lstm_raw

        ensemble = self.xgb_weight * preds_xgb + self.lstm_weight * preds_lstm
        return pd.Series(ensemble, index=X.index, name="ensemble_pred")

    # ------------------------------------------------------------------
    # Walk-forward validation
    # ------------------------------------------------------------------

    def walk_forward_validate(
        self,
        data: pd.DataFrame,
        n_splits: int = 5,
    ) -> pd.DataFrame:
        """Evaluate alpha quality via time-series cross-validation.

        For each fold: trains XGBoost on the in-sample window and predicts
        on the out-of-sample window.  Returns per-fold MSE and IC
        (information coefficient = Spearman rank correlation).

        Parameters
        ----------
        data : pd.DataFrame
            Price (and optional volume) DataFrame.
        n_splits : int, optional
            Number of TimeSeriesSplit folds. Default is 5.

        Returns
        -------
        pd.DataFrame
            Per-fold metrics with columns ``["fold", "mse", "ic", "n_test"]``.
        """
        self._validate_price_df(data, min_rows=100)
        feat_df = self.engineer_features(data)

        price_col = "close" if "close" in data.columns else data.columns[0]
        target = data[price_col].pct_change(1).shift(-1).reindex(feat_df.index).dropna()
        feat_df = feat_df.reindex(target.index)

        X = feat_df.values
        y = target.values

        tscv = TimeSeriesSplit(n_splits=n_splits)
        results: List[Dict[str, Any]] = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler_fold = StandardScaler()
            X_train_s = scaler_fold.fit_transform(X_train)
            X_test_s = scaler_fold.transform(X_test)

            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                verbosity=0,
                n_jobs=-1,
            )
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)

            mse = float(mean_squared_error(y_test, preds))
            # Spearman IC
            ic = float(
                pd.Series(preds).corr(pd.Series(y_test), method="spearman")
            )
            results.append(
                {"fold": fold + 1, "mse": mse, "ic": ic, "n_test": len(test_idx)}
            )
            self.logger.info("WFV fold %d: MSE=%.6f, IC=%.4f", fold + 1, mse, ic)

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Feature importance via SHAP
    # ------------------------------------------------------------------

    def get_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute SHAP-based feature importance from the XGBoost model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (same columns as used during training).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``["feature", "mean_abs_shap"]``, sorted
            by importance descending.

        Raises
        ------
        RuntimeError
            If :meth:`train_xgboost` has not been called.
        """
        if self.xgb_model is None:
            raise RuntimeError("XGBoost model not trained; call train_xgboost first.")

        X_scaled = pd.DataFrame(
            self.scaler.transform(X.values),
            index=X.index,
            columns=X.columns,
        )
        explainer = shap.TreeExplainer(self.xgb_model)
        shap_values = explainer.shap_values(X_scaled)

        mean_abs = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame(
            {"feature": X.columns.tolist(), "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        self.logger.info(
            "Top feature: %s (SHAP=%.4f)",
            importance_df.iloc[0]["feature"],
            importance_df.iloc[0]["mean_abs_shap"],
        )
        return importance_df

    # ------------------------------------------------------------------
    # Strategy pipeline
    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Train the ensemble and produce forward return predictions.

        If models are not yet trained, performs a single in-sample fit
        on the full ``data`` window (walk-forward validation should be
        used for rigorous out-of-sample evaluation).

        Parameters
        ----------
        data : pd.DataFrame
            Price (and optional volume) DataFrame.

        Returns
        -------
        pd.DataFrame
            Single-column DataFrame ``["pred_return"]`` of ensemble
            predictions per date.
        """
        self._validate_price_df(data, min_rows=100)
        feat_df = self.engineer_features(data)

        price_col = "close" if "close" in data.columns else data.columns[0]
        target = (
            data[price_col]
            .pct_change(1)
            .shift(-1)
            .reindex(feat_df.index)
            .dropna()
        )
        feat_aligned = feat_df.reindex(target.index)

        X_raw = feat_aligned.values
        y_raw = target.values

        # Fit scaler and models if not already trained
        X_scaled = self.scaler.fit_transform(X_raw)

        if self.xgb_model is None:
            self.train_xgboost(
                pd.DataFrame(X_scaled, index=feat_aligned.index, columns=feat_aligned.columns),
                pd.Series(y_raw, index=feat_aligned.index),
            )

        if self.lstm_model is None:
            self.train_lstm(X_scaled, y_raw)

        preds = self.ensemble_predict(feat_aligned)
        signals = preds.to_frame(name="pred_return")
        self.logger.info(
            "generate_signals: %d predictions, mean=%.5f, std=%.5f",
            len(signals),
            float(signals["pred_return"].mean()),
            float(signals["pred_return"].std()),
        )
        return signals

    def compute_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Construct equal-weight long/short book from ML predictions.

        For each date, goes long the top ``long_n`` and short the bottom
        ``short_n`` assets by predicted return.  Expects signals to
        contain one column per asset (multi-asset mode) or a single
        ``"pred_return"`` column (single-asset mode).

        Parameters
        ----------
        signals : pd.DataFrame
            Output of :meth:`generate_signals`.

        Returns
        -------
        pd.DataFrame
            Position weights in ``[-1, +1]``.
        """
        if signals.empty:
            self.logger.warning("compute_positions: empty signals.")
            return pd.DataFrame(index=signals.index)

        # Single-asset case: binary long/short on sign of prediction
        if signals.shape[1] == 1:
            col = signals.columns[0]
            positions = pd.DataFrame(
                np.sign(signals[col].values),
                index=signals.index,
                columns=[col],
            )
            return positions

        # Multi-asset case: rank cross-section
        positions = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
        for date, row in signals.iterrows():
            valid = row.dropna()
            if len(valid) < self.long_n + self.short_n:
                continue
            top = valid.nlargest(self.long_n).index
            bottom = valid.nsmallest(self.short_n).index
            positions.loc[date, top] = 1.0 / self.long_n
            positions.loc[date, bottom] = -1.0 / self.short_n

        self.logger.info(
            "compute_positions: active rows=%d",
            int((positions != 0).any(axis=1).sum()),
        )
        return positions
