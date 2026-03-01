"""Deep hedging with neural networks using PyTorch.

Implements the deep hedging framework of Buehler et al. (2019) for learning
dynamic hedging strategies from simulated or historical price paths. Minimises
the CVaR of the residual hedging P&L under proportional transaction costs.
"""

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PyTorch imports with graceful error
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. DeepHedger requires torch to be installed.")


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for DeepHedger: pip install torch")


# ---------------------------------------------------------------------------
# Black-Scholes helpers (no external dependency)
# ---------------------------------------------------------------------------

def _bs_delta(S: np.ndarray, K: float, T_remaining: np.ndarray, r: float, sigma: float) -> np.ndarray:
    """Vectorised Black-Scholes delta for call options."""
    T_remaining = np.maximum(T_remaining, 1e-9)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T_remaining) / (
        sigma * np.sqrt(T_remaining)
    )
    from scipy.stats import norm
    return norm.cdf(d1)


def _bs_price_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    from scipy.stats import norm
    T = max(T, 1e-9)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return float(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))


def _simulate_gbm(S0: float, T: float, r: float, sigma: float,
                  n_steps: int, n_paths: int, seed: Optional[int] = None) -> np.ndarray:
    """Simulate geometric Brownian motion paths."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_returns = (r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
    S = np.full((n_paths, n_steps + 1), S0, dtype=float)
    S[:, 1:] = S0 * np.exp(np.cumsum(log_returns, axis=1))
    return S


# ---------------------------------------------------------------------------
# Neural network module
# ---------------------------------------------------------------------------

class HedgingNetwork(nn.Module):
    """Recurrence-free feedforward hedging network.

    At each time step, takes a feature vector of
    ``[normalised price, time to maturity, current portfolio value]``
    concatenated with the previous hedge ratio and outputs the new hedge ratio
    for each asset.

    Parameters
    ----------
    n_assets : int
        Number of assets to hedge.
    n_features : int
        Dimension of the input feature vector per step.
    n_hidden : int
        Number of hidden units per layer.
    n_layers : int
        Number of hidden layers.
    """

    def __init__(self, n_assets: int, n_features: int, n_hidden: int, n_layers: int) -> None:
        _require_torch()
        super().__init__()
        self.n_assets = n_assets

        layers: List[nn.Module] = [nn.Linear(n_features + n_assets, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers.append(nn.Linear(n_hidden, n_assets))

        self.network = nn.Sequential(*layers)

        # Weight initialisation (Xavier uniform)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: "torch.Tensor", prev_delta: "torch.Tensor") -> "torch.Tensor":
        """Forward pass.

        Parameters
        ----------
        features : torch.Tensor
            Shape ``(batch, n_features)``.
        prev_delta : torch.Tensor
            Previous hedge ratio, shape ``(batch, n_assets)``.

        Returns
        -------
        torch.Tensor
            New hedge ratios, shape ``(batch, n_assets)``.
        """
        x = torch.cat([features, prev_delta], dim=-1)
        return torch.sigmoid(self.network(x))  # hedge ratios in [0, 1]


# ---------------------------------------------------------------------------
# Deep hedger
# ---------------------------------------------------------------------------

class DeepHedger:
    """Deep hedging agent that learns dynamic hedging strategies.

    Trains a neural network to minimise the CVaR of the residual P&L
    from hedging a contingent claim under proportional transaction costs,
    following Buehler et al. (2019).

    Parameters
    ----------
    n_assets : int
        Number of risky assets.
    n_hidden : int, optional
        Hidden units per layer, by default 64.
    n_layers : int, optional
        Number of hidden layers, by default 3.

    Examples
    --------
    >>> import numpy as np
    >>> dh = DeepHedger(n_assets=1, n_hidden=64, n_layers=3)
    >>> paths = np.random.lognormal(0, 0.01, (1000, 53))  # 1000 paths, 52 steps
    >>> payoffs = np.maximum(paths[:, -1] - 100.0, 0.0)
    >>> losses = dh.train(paths, payoffs, epochs=5)
    """

    def __init__(
        self,
        n_assets: int = 1,
        n_hidden: int = 64,
        n_layers: int = 3,
    ) -> None:
        _require_torch()
        if n_assets < 1:
            raise ValueError(f"n_assets must be >= 1, got {n_assets}")
        self.n_assets = n_assets
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        # n_features = n_assets (prices) + 1 (time to maturity) + 1 (portfolio value)
        n_features = n_assets + 2
        self._network = HedgingNetwork(n_assets, n_features, n_hidden, n_layers)
        self._is_trained: bool = False
        logger.debug(
            "DeepHedger initialised: n_assets=%d, n_hidden=%d, n_layers=%d",
            n_assets, n_hidden, n_layers,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        paths: np.ndarray,
        payoffs: np.ndarray,
        transaction_cost: float = 0.001,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        cvar_alpha: float = 0.95,
        seed: Optional[int] = None,
    ) -> List[float]:
        """Train the hedging network to minimise CVaR of P&L.

        The loss function is the Conditional Value-at-Risk (CVaR / Expected
        Shortfall) at confidence level ``cvar_alpha`` of the terminal
        hedging P&L, including proportional transaction costs.

        Parameters
        ----------
        paths : np.ndarray
            Asset price paths of shape ``(n_paths, n_steps + 1)``.
            For ``n_assets > 1``, shape should be
            ``(n_paths, n_steps + 1, n_assets)``.
        payoffs : np.ndarray
            Terminal payoff for each path, shape ``(n_paths,)``.
        transaction_cost : float, optional
            Proportional transaction cost rate, by default 0.001.
        epochs : int, optional
            Training epochs, by default 100.
        batch_size : int, optional
            Mini-batch size, by default 256.
        learning_rate : float, optional
            Adam learning rate, by default 1e-3.
        cvar_alpha : float, optional
            CVaR confidence level in (0, 1), by default 0.95.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        List[float]
            Per-epoch training loss values.

        Raises
        ------
        ValueError
            If ``paths`` or ``payoffs`` have incompatible shapes.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        paths = np.asarray(paths, dtype=float)
        payoffs = np.asarray(payoffs, dtype=float)

        # Handle single-asset paths with shape (n_paths, n_steps+1)
        if paths.ndim == 2:
            paths = paths[:, :, np.newaxis]  # (n_paths, n_steps+1, 1)

        n_paths, n_steps_plus1, n_assets_in = paths.shape
        n_steps = n_steps_plus1 - 1

        if n_assets_in != self.n_assets:
            raise ValueError(
                f"paths has {n_assets_in} asset dimensions, expected {self.n_assets}"
            )
        if len(payoffs) != n_paths:
            raise ValueError(
                f"payoffs length {len(payoffs)} does not match n_paths {n_paths}"
            )

        # Normalise prices by initial price
        S0 = paths[:, 0, :]  # (n_paths, n_assets)
        paths_norm = paths / (S0[:, np.newaxis, :] + 1e-10)

        # Convert to tensors
        paths_t = torch.tensor(paths_norm, dtype=torch.float32)
        payoffs_t = torch.tensor(payoffs, dtype=torch.float32)

        dataset = TensorDataset(paths_t, payoffs_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self._network.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        self._n_steps = n_steps
        epoch_losses: List[float] = []

        logger.info(
            "DeepHedger training: %d paths, %d steps, %d epochs",
            n_paths, n_steps, epochs,
        )

        self._network.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch_paths, batch_payoffs in loader:
                batch_size_actual = batch_paths.shape[0]
                prev_delta = torch.zeros(batch_size_actual, self.n_assets)
                portfolio_value = torch.zeros(batch_size_actual)
                tc_cost = torch.zeros(batch_size_actual)

                for t in range(n_steps):
                    S_t = batch_paths[:, t, :]  # (batch, n_assets)
                    time_remaining = torch.full((batch_size_actual, 1), (n_steps - t) / n_steps)
                    features = torch.cat([S_t, time_remaining, portfolio_value.unsqueeze(-1)], dim=-1)

                    delta = self._network(features, prev_delta)  # (batch, n_assets)

                    # Transaction costs
                    delta_change = torch.abs(delta - prev_delta)
                    tc_cost = tc_cost + torch.sum(
                        delta_change * S_t * transaction_cost, dim=-1
                    )

                    # Portfolio P&L from rebalancing
                    S_next = batch_paths[:, t + 1, :]
                    pnl_step = torch.sum(delta * (S_next - S_t), dim=-1)
                    portfolio_value = portfolio_value + pnl_step

                    prev_delta = delta

                # Terminal P&L: hedge P&L - payoff - transaction costs
                pnl = portfolio_value - batch_payoffs - tc_cost

                # CVaR loss
                loss = self._cvar_loss(pnl, alpha=cvar_alpha)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            epoch_losses.append(avg_loss)

            if (epoch + 1) % max(1, epochs // 10) == 0:
                logger.info("Epoch %d/%d — CVaR loss: %.6f", epoch + 1, epochs, avg_loss)

        self._is_trained = True
        logger.info("DeepHedger training complete")
        return epoch_losses

    @staticmethod
    def _cvar_loss(pnl: "torch.Tensor", alpha: float = 0.95) -> "torch.Tensor":
        """CVaR (Expected Shortfall) loss: mean of worst (1-alpha) fraction of losses."""
        losses = -pnl  # negate so higher = worse
        n = losses.shape[0]
        k = max(1, int(math.ceil((1.0 - alpha) * n)))
        top_k, _ = torch.topk(losses, k)
        return top_k.mean()

    # ------------------------------------------------------------------
    # Hedging
    # ------------------------------------------------------------------

    def hedge(
        self,
        paths: np.ndarray,
    ) -> np.ndarray:
        """Compute hedge ratios at each time step for given price paths.

        Parameters
        ----------
        paths : np.ndarray
            Asset price paths of shape ``(n_paths, n_steps + 1)`` or
            ``(n_paths, n_steps + 1, n_assets)``.

        Returns
        -------
        np.ndarray
            Hedge ratios of shape ``(n_paths, n_steps, n_assets)``.

        Raises
        ------
        RuntimeError
            If the model has not been trained.
        """
        if not self._is_trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

        paths = np.asarray(paths, dtype=float)
        if paths.ndim == 2:
            paths = paths[:, :, np.newaxis]

        n_paths, n_steps_plus1, n_assets = paths.shape
        n_steps = n_steps_plus1 - 1

        S0 = paths[:, 0, :]
        paths_norm = paths / (S0[:, np.newaxis, :] + 1e-10)
        paths_t = torch.tensor(paths_norm, dtype=torch.float32)

        self._network.eval()
        all_deltas = np.zeros((n_paths, n_steps, n_assets), dtype=float)

        with torch.no_grad():
            prev_delta = torch.zeros(n_paths, self.n_assets)
            portfolio_value = torch.zeros(n_paths)

            for t in range(n_steps):
                S_t = paths_t[:, t, :]
                time_remaining = torch.full((n_paths, 1), (n_steps - t) / n_steps)
                features = torch.cat([S_t, time_remaining, portfolio_value.unsqueeze(-1)], dim=-1)

                delta = self._network(features, prev_delta)
                all_deltas[:, t, :] = delta.numpy()

                S_next = paths_t[:, t + 1, :]
                pnl_step = torch.sum(delta * (S_next - S_t), dim=-1)
                portfolio_value = portfolio_value + pnl_step
                prev_delta = delta

        return all_deltas

    # ------------------------------------------------------------------
    # Comparison with Black-Scholes delta hedging
    # ------------------------------------------------------------------

    def compare_with_bs(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_steps: int = 52,
        n_paths: int = 1000,
        transaction_cost: float = 0.001,
        seed: Optional[int] = None,
    ) -> dict:
        """Compare deep hedging P&L versus Black-Scholes delta hedging.

        Simulates GBM paths, computes the P&L distributions for both the
        deep hedger and a Black-Scholes delta hedge (with the same
        transaction costs), and returns summary statistics.

        Parameters
        ----------
        S0 : float
            Initial spot price.
        K : float
            Option strike price.
        T : float
            Option maturity in years.
        r : float
            Continuously compounded risk-free rate.
        sigma : float
            GBM volatility.
        n_steps : int, optional
            Hedging rebalancing steps, by default 52.
        n_paths : int, optional
            Number of Monte Carlo paths, by default 1000.
        transaction_cost : float, optional
            Proportional transaction cost rate, by default 0.001.
        seed : int, optional
            Random seed.

        Returns
        -------
        dict
            Keys: ``deep_pnl_mean``, ``deep_pnl_std``, ``deep_cvar_95``,
            ``bs_pnl_mean``, ``bs_pnl_std``, ``bs_cvar_95``,
            ``option_price``, ``improvement``.
        """
        paths = _simulate_gbm(S0, T, r, sigma, n_steps, n_paths, seed=seed)
        payoffs = np.maximum(paths[:, -1] - K, 0.0)

        # --- Deep hedging P&L ---
        if self._is_trained:
            dh_deltas = self.hedge(paths)[:, :, 0]  # (n_paths, n_steps)
        else:
            logger.warning("DeepHedger not trained; using BS deltas for deep hedge arm.")
            dt = T / n_steps
            dh_deltas = np.zeros((n_paths, n_steps))
            for t in range(n_steps):
                T_rem = T - t * dt
                dh_deltas[:, t] = _bs_delta(paths[:, t], K, np.full(n_paths, T_rem), r, sigma)

        deep_pnl = self._compute_pnl(paths, payoffs, dh_deltas, transaction_cost)

        # --- BS delta hedging P&L ---
        dt = T / n_steps
        bs_deltas = np.zeros((n_paths, n_steps))
        for t in range(n_steps):
            T_rem = max(T - t * dt, 1e-9)
            bs_deltas[:, t] = _bs_delta(paths[:, t], K, np.full(n_paths, T_rem), r, sigma)

        bs_pnl = self._compute_pnl(paths, payoffs, bs_deltas, transaction_cost)

        option_price = _bs_price_call(S0, K, T, r, sigma)

        def cvar95(x: np.ndarray) -> float:
            k = max(1, int(math.ceil(0.05 * len(x))))
            return float(np.mean(np.sort(x)[:k]))

        result = {
            "deep_pnl_mean": float(np.mean(deep_pnl)),
            "deep_pnl_std": float(np.std(deep_pnl)),
            "deep_cvar_95": cvar95(deep_pnl),
            "bs_pnl_mean": float(np.mean(bs_pnl)),
            "bs_pnl_std": float(np.std(bs_pnl)),
            "bs_cvar_95": cvar95(bs_pnl),
            "option_price": option_price,
            "improvement": float(cvar95(bs_pnl) - cvar95(deep_pnl)),
        }
        logger.info(
            "Comparison: DeepHedge CVaR95=%.4f, BS CVaR95=%.4f, improvement=%.4f",
            result["deep_cvar_95"], result["bs_cvar_95"], result["improvement"],
        )
        return result

    @staticmethod
    def _compute_pnl(
        paths: np.ndarray,
        payoffs: np.ndarray,
        deltas: np.ndarray,
        transaction_cost: float,
    ) -> np.ndarray:
        """Compute terminal hedging P&L for given paths and deltas."""
        n_paths, n_steps = deltas.shape
        pnl = np.zeros(n_paths)
        tc = np.zeros(n_paths)
        prev_delta = np.zeros(n_paths)

        for t in range(n_steps):
            delta = deltas[:, t]
            S_t = paths[:, t]
            S_next = paths[:, t + 1]
            tc += np.abs(delta - prev_delta) * S_t * transaction_cost
            pnl += delta * (S_next - S_t)
            prev_delta = delta

        return pnl - payoffs - tc
