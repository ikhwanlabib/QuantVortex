"""
Deep reinforcement learning trading agent for QuantVortex.

Implements a custom Gymnasium environment with differential Sharpe
ratio rewards and trains a PPO agent via stable-baselines3.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gymnasium trading environment
# ---------------------------------------------------------------------------

class TradingEnv(gym.Env):
    """Multi-asset trading environment for deep RL.

    State
    -----
    Concatenation of:
    - OHLCV-based features (normalised) for each asset.
    - Technical indicators: 10-day and 30-day moving averages, 14-day RSI,
      20-day Bollinger Band width, and 14-day ATR.
    - Portfolio value (normalised by initial capital).
    - Current position vector (one scalar per asset).

    Action Space
    ------------
    ``Box(-1, 1, shape=(n_assets,))`` — continuous target weights.

    Reward
    ------
    Differential Sharpe ratio (DSR):

    .. math::

        R_t = \\frac{A_{t-1} \\Delta r_t - 0.5 B_{t-1} \\Delta r_t^2}
                    {\\left(B_{t-1} - A_{t-1}^2\\right)^{3/2}}

    where :math:`A_t` and :math:`B_t` are exponential moving averages of
    portfolio returns and squared portfolio returns.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide-format OHLCV DataFrame.  Must have columns:
        ``["open", "close", "high", "low", "volume"]`` or a flat
        MultiIndex ``(field, ticker)``.
    n_assets : int
        Number of tradeable assets.
    initial_capital : float, optional
        Starting portfolio value. Default is 1_000_000.
    transaction_cost : float, optional
        One-way transaction cost per unit of notional traded.
        Default is 0.001.
    dsr_eta : float, optional
        Adaptation rate for the DSR exponential moving averages.
        Default is 0.01.
    max_steps : int or None, optional
        Maximum episode length (steps). If ``None``, the episode runs to
        the end of the price data. Default is ``None``.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices: pd.DataFrame,
        n_assets: int,
        initial_capital: float = 1_000_000.0,
        transaction_cost: float = 0.001,
        dsr_eta: float = 0.01,
        max_steps: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.prices = prices.copy()
        self.n_assets = n_assets
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.dsr_eta = dsr_eta
        self.max_steps = max_steps if max_steps is not None else len(prices) - 1

        # Pre-compute features matrix
        self._features = self._precompute_features()
        self.obs_dim = self._features.shape[1] + 1 + n_assets  # features + pv + positions

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_assets,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self._current_step: int = 0
        self._portfolio_value: float = initial_capital
        self._positions: np.ndarray = np.zeros(n_assets, dtype=np.float32)
        self._A: float = 0.0   # DSR first moment
        self._B: float = 0.0   # DSR second moment

    # ------------------------------------------------------------------
    # Feature pre-computation
    # ------------------------------------------------------------------

    def _precompute_features(self) -> np.ndarray:
        """Compute a normalised feature matrix for the entire price series.

        Returns
        -------
        np.ndarray
            Feature array of shape ``(len(prices), n_features)``.
        """
        close_cols = [c for c in self.prices.columns if "close" in str(c).lower()]
        if not close_cols:
            close_cols = self.prices.columns.tolist()[: self.n_assets]
        close = self.prices[close_cols[: self.n_assets]].astype(float)

        feats: List[pd.Series] = []

        for col in close.columns:
            px = close[col]
            ret = px.pct_change().fillna(0.0)
            feats.append(ret.rename(f"{col}_ret"))

            # MA10 / MA30 normalised
            ma10 = px.rolling(10, min_periods=1).mean()
            ma30 = px.rolling(30, min_periods=1).mean()
            feats.append(((px - ma10) / ma10.replace(0, np.nan)).fillna(0.0).rename(f"{col}_ma10_dev"))
            feats.append(((px - ma30) / ma30.replace(0, np.nan)).fillna(0.0).rename(f"{col}_ma30_dev"))

            # RSI 14
            delta = px.diff()
            gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
            loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = (100 - 100 / (1 + rs)).fillna(50.0) / 100.0
            feats.append(rsi.rename(f"{col}_rsi"))

            # Bollinger band width
            ma20 = px.rolling(20, min_periods=1).mean()
            std20 = px.rolling(20, min_periods=1).std().fillna(0.0)
            bb_width = (2 * std20 / ma20.replace(0, np.nan)).fillna(0.0)
            feats.append(bb_width.rename(f"{col}_bb_width"))

            # ATR 14 (approximated with close only: |ret|)
            atr = ret.abs().rolling(14, min_periods=1).mean()
            feats.append(atr.rename(f"{col}_atr"))

        feat_df = pd.concat(feats, axis=1).fillna(0.0)
        arr = feat_df.values.astype(np.float32)
        # Clip extreme values
        arr = np.clip(arr, -5.0, 5.0)
        return arr

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to the initial state.

        Parameters
        ----------
        seed : int or None, optional
            Random seed (passed to super). Default is ``None``.
        options : dict or None, optional
            Reserved for future use. Default is ``None``.

        Returns
        -------
        observation : np.ndarray
            Initial observation vector.
        info : dict
            Empty info dictionary.
        """
        super().reset(seed=seed)
        self._current_step = 0
        self._portfolio_value = self.initial_capital
        self._positions = np.zeros(self.n_assets, dtype=np.float32)
        self._A = 0.0
        self._B = 0.0
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one step.

        Parameters
        ----------
        action : np.ndarray
            Desired portfolio weights, shape ``(n_assets,)``.  Clipped and
            normalised to sum to at most 1 in absolute value.

        Returns
        -------
        observation : np.ndarray
            Next observation.
        reward : float
            Differential Sharpe ratio reward.
        terminated : bool
            ``True`` if the episode has reached the end of the data or
            portfolio value has fallen to zero.
        truncated : bool
            ``True`` if ``max_steps`` is reached.
        info : dict
            Diagnostic information (portfolio value, step).
        """
        action = np.clip(action, -1.0, 1.0)
        # Normalise so sum of |weights| ≤ 1
        abs_sum = np.abs(action).sum()
        if abs_sum > 1.0:
            action = action / abs_sum

        # Compute transaction costs
        trade_sizes = np.abs(action - self._positions)
        cost = self.transaction_cost * trade_sizes.sum()

        self._positions = action.astype(np.float32)
        self._current_step += 1

        # Get asset returns at this step
        feat_row = self._features[min(self._current_step, len(self._features) - 1)]
        # ret features are the first n_assets columns of the feature matrix
        asset_rets = feat_row[: self.n_assets]

        portfolio_ret = float(np.dot(self._positions, asset_rets)) - cost
        self._portfolio_value *= 1.0 + portfolio_ret

        # Differential Sharpe Ratio
        reward = self._dsr_reward(portfolio_ret)

        done = self._portfolio_value <= 0.0
        truncated = self._current_step >= self.max_steps

        info = {
            "portfolio_value": self._portfolio_value,
            "step": self._current_step,
            "portfolio_return": portfolio_ret,
        }
        return self._get_obs(), reward, done, truncated, info

    def render(self) -> None:
        """Render current portfolio state to stdout."""
        print(
            f"Step={self._current_step:4d}  "
            f"PV={self._portfolio_value:,.2f}  "
            f"Positions={np.round(self._positions, 3)}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Construct the observation vector for the current step."""
        step_idx = min(self._current_step, len(self._features) - 1)
        feat = self._features[step_idx]
        pv_norm = np.array(
            [self._portfolio_value / self.initial_capital], dtype=np.float32
        )
        obs = np.concatenate([feat, pv_norm, self._positions]).astype(np.float32)
        return obs

    def _dsr_reward(self, r: float) -> float:
        """Compute the differential Sharpe ratio reward.

        Parameters
        ----------
        r : float
            Portfolio return at this step.

        Returns
        -------
        float
            DSR reward value.
        """
        eta = self.dsr_eta
        delta_A = r - self._A
        delta_B = r ** 2 - self._B

        denom = (self._B - self._A ** 2)
        if denom > 1e-12:
            dsr = (self._B * delta_A - 0.5 * self._A * delta_B) / (denom ** 1.5)
        else:
            dsr = 0.0

        # Update exponential moving averages
        self._A += eta * delta_A
        self._B += eta * delta_B

        return float(np.clip(dsr, -1.0, 1.0))


# ---------------------------------------------------------------------------
# RL trader strategy
# ---------------------------------------------------------------------------

class RLTrader(BaseStrategy):
    """Deep RL trading strategy using stable-baselines3 PPO.

    Parameters
    ----------
    name : str, optional
        Strategy name. Default is ``"RLTrader"``.
    allocation : float, optional
        Capital allocation fraction. Default is 1.0.
    n_assets : int, optional
        Number of tradeable assets. Default is 1.
    config : dict or None, optional
        Additional configuration overrides for :class:`TradingEnv` and
        PPO.  Supported keys:

        - ``"initial_capital"`` (float)
        - ``"transaction_cost"`` (float)
        - ``"dsr_eta"`` (float)
        - ``"ppo_learning_rate"`` (float)
        - ``"ppo_n_steps"`` (int)
        - ``"ppo_batch_size"`` (int)
        - ``"ppo_n_epochs"`` (int)
        - ``"ppo_gamma"`` (float)
        - ``"ppo_gae_lambda"`` (float)

    Attributes
    ----------
    model : stable_baselines3.PPO or None
        Trained PPO agent.
    env : TradingEnv or None
        Trading environment instance.
    """

    def __init__(
        self,
        name: str = "RLTrader",
        allocation: float = 1.0,
        n_assets: int = 1,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name=name, allocation=allocation)
        self.n_assets = n_assets
        self.config: Dict[str, Any] = config or {}
        self.model: Optional[PPO] = None
        self.env: Optional[TradingEnv] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        data: pd.DataFrame,
        n_timesteps: int = 10_000,
    ) -> None:
        """Train the PPO agent on historical market data.

        Parameters
        ----------
        data : pd.DataFrame
            Price DataFrame passed to :class:`TradingEnv`.
        n_timesteps : int, optional
            Total environment interaction steps. Default is 10 000.

        Raises
        ------
        ValueError
            If ``data`` is empty or has insufficient rows.
        """
        self._validate_price_df(data, min_rows=50)

        env_kwargs = {
            "prices": data,
            "n_assets": self.n_assets,
            "initial_capital": self.config.get("initial_capital", 1_000_000.0),
            "transaction_cost": self.config.get("transaction_cost", 0.001),
            "dsr_eta": self.config.get("dsr_eta", 0.01),
        }
        self.env = TradingEnv(**env_kwargs)

        def make_env() -> TradingEnv:
            return TradingEnv(**env_kwargs)

        vec_env = DummyVecEnv([make_env])

        ppo_kwargs: Dict[str, Any] = {
            "policy": "MlpPolicy",
            "env": vec_env,
            "learning_rate": self.config.get("ppo_learning_rate", 3e-4),
            "n_steps": self.config.get("ppo_n_steps", 2048),
            "batch_size": self.config.get("ppo_batch_size", 64),
            "n_epochs": self.config.get("ppo_n_epochs", 10),
            "gamma": self.config.get("ppo_gamma", 0.99),
            "gae_lambda": self.config.get("ppo_gae_lambda", 0.95),
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "verbose": 0,
        }

        self.model = PPO(**ppo_kwargs)
        self.logger.info(
            "Training PPO for %d timesteps on %d assets.", n_timesteps, self.n_assets
        )
        self.model.learn(total_timesteps=n_timesteps)
        self.logger.info("PPO training complete.")

    # ------------------------------------------------------------------
    # Strategy pipeline
    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the trained PPO agent on ``data`` to produce position signals.

        If the model has not been trained, trains it automatically using
        ``data`` as the historical episode.

        Parameters
        ----------
        data : pd.DataFrame
            Price DataFrame (same format as used for training).

        Returns
        -------
        pd.DataFrame
            Position weight signals with shape
            ``(len(data), n_assets)``, columns named ``"asset_0"``, …
        """
        self._validate_price_df(data, min_rows=50)

        if self.model is None:
            self.logger.info("Model not trained; running train() automatically.")
            self.train(data)

        env = TradingEnv(
            prices=data,
            n_assets=self.n_assets,
            initial_capital=self.config.get("initial_capital", 1_000_000.0),
            transaction_cost=self.config.get("transaction_cost", 0.001),
            dsr_eta=self.config.get("dsr_eta", 0.01),
        )

        obs, _ = env.reset()
        all_actions: List[np.ndarray] = []
        dates: List[Any] = []

        for t, date in enumerate(data.index):
            action, _ = self.model.predict(obs, deterministic=True)
            all_actions.append(action.copy())
            dates.append(date)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        col_names = [f"asset_{i}" for i in range(self.n_assets)]
        actions_arr = np.stack(all_actions, axis=0)
        signals = pd.DataFrame(
            actions_arr, index=dates[: len(all_actions)], columns=col_names
        )
        self.logger.info(
            "generate_signals: produced %d rows for %d assets.",
            len(signals),
            self.n_assets,
        )
        return signals

    def compute_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Pass PPO action signals through as final positions.

        The PPO agent already outputs normalised continuous weights in
        ``[-1, +1]``, so this method simply enforces the leverage budget.

        Parameters
        ----------
        signals : pd.DataFrame
            Output of :meth:`generate_signals`.

        Returns
        -------
        pd.DataFrame
            Position weights (clipped and L1-normalised per row).
        """
        if signals.empty:
            self.logger.warning("compute_positions: empty signals.")
            return pd.DataFrame(index=signals.index)

        positions = signals.clip(-1.0, 1.0).astype(float)
        row_l1 = positions.abs().sum(axis=1).replace(0, 1.0)
        positions = positions.div(row_l1.where(row_l1 > 1.0, 1.0), axis=0)

        self.logger.info(
            "compute_positions: mean gross exposure=%.4f",
            float(positions.abs().sum(axis=1).mean()),
        )
        return positions
