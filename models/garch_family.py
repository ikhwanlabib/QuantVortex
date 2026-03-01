"""GARCH family volatility models: GARCH, EGARCH, and GJR-GARCH.

Implements GARCH(p,q), EGARCH(p,q), and GJR-GARCH(p,q) via maximum likelihood
estimation using scipy.optimize, along with rolling volatility forecasting and
AIC/BIC model comparison.
"""

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

logger = logging.getLogger(__name__)


class GARCHFamily:
    """GARCH family volatility models: GARCH, EGARCH, and GJR-GARCH.

    Fits GARCH(p,q), EGARCH(p,q) and GJR-GARCH(p,q) models to a series of
    returns using Gaussian quasi-maximum likelihood estimation, provides
    rolling out-of-sample volatility forecasts, and produces an AIC/BIC
    comparison table across all three model variants.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> returns = rng.normal(0, 0.01, 1000)
    >>> gf = GARCHFamily()
    >>> res = gf.fit_garch(returns)
    >>> print(res["aic"])
    """

    # ------------------------------------------------------------------
    # Internal shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_returns(returns: np.ndarray) -> np.ndarray:
        r = np.asarray(returns, dtype=float)
        if r.ndim != 1:
            raise ValueError("returns must be a 1-D array")
        if len(r) < 10:
            raise ValueError("returns array is too short (< 10 observations)")
        if np.any(np.isnan(r)):
            raise ValueError("returns array contains NaN values")
        return r

    @staticmethod
    def _aic_bic(log_likelihood: float, n_params: int, n_obs: int) -> tuple[float, float]:
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * math.log(n_obs) - 2 * log_likelihood
        return aic, bic

    # ------------------------------------------------------------------
    # GARCH(p, q)
    # ------------------------------------------------------------------

    def fit_garch(
        self,
        returns: np.ndarray,
        p: int = 1,
        q: int = 1,
    ) -> dict:
        """Fit a GARCH(p,q) model by Gaussian quasi-maximum likelihood.

        The conditional variance equation is:

        .. math::
            h_t = \\omega
                  + \\sum_{i=1}^{q} \\alpha_i \\varepsilon_{t-i}^2
                  + \\sum_{j=1}^{p} \\beta_j h_{t-j}

        Parameters
        ----------
        returns : np.ndarray
            1-D array of (demeaned) returns.
        p : int, optional
            GARCH lag order, by default 1.
        q : int, optional
            ARCH lag order, by default 1.

        Returns
        -------
        dict
            Keys: ``omega``, ``alpha`` (array of length q), ``beta``
            (array of length p), ``log_likelihood``, ``aic``, ``bic``,
            ``conditional_variances``, ``model``.

        Raises
        ------
        ValueError
            If ``returns`` fails validation.
        RuntimeError
            If optimisation fails to converge.
        """
        returns = self._validate_returns(returns)
        n = len(returns)
        n_params = 1 + q + p  # omega, alpha_1..q, beta_1..p

        def neg_ll(params: np.ndarray) -> float:
            omega = params[0]
            alphas = params[1: 1 + q]
            betas = params[1 + q:]
            if omega <= 0 or np.any(alphas < 0) or np.any(betas < 0):
                return 1e12
            if np.sum(alphas) + np.sum(betas) >= 1.0:
                return 1e12  # stationarity constraint

            h = np.full(n, np.var(returns))
            ll = 0.0
            max_lag = max(p, q)
            for t in range(max_lag, n):
                arch_term = sum(alphas[i] * returns[t - 1 - i] ** 2 for i in range(q))
                garch_term = sum(betas[j] * h[t - 1 - j] for j in range(p))
                h[t] = omega + arch_term + garch_term
                h[t] = max(h[t], 1e-12)
                ll += -0.5 * (math.log(2 * math.pi) + math.log(h[t]) + returns[t] ** 2 / h[t])
            return -ll

        # Initial parameter guesses
        var0 = np.var(returns)
        x0 = np.array([var0 * 0.05] + [0.1] * q + [0.8 / p] * p)
        bounds = [(1e-8, None)] + [(0.0, 1.0)] * (q + p)

        result = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 2000, "ftol": 1e-12})
        if not result.success:
            logger.warning("GARCH(%d,%d) optimisation: %s", p, q, result.message)

        omega = float(result.x[0])
        alphas = result.x[1: 1 + q]
        betas = result.x[1 + q:]
        ll = float(-result.fun)
        aic, bic = self._aic_bic(ll, n_params, n)

        # Recover conditional variances
        h = np.full(n, np.var(returns))
        max_lag = max(p, q)
        for t in range(max_lag, n):
            arch_term = sum(alphas[i] * returns[t - 1 - i] ** 2 for i in range(q))
            garch_term = sum(betas[j] * h[t - 1 - j] for j in range(p))
            h[t] = max(omega + arch_term + garch_term, 1e-12)

        logger.info(
            "GARCH(%d,%d) fitted: omega=%.6f, alpha=%s, beta=%s, AIC=%.4f",
            p, q, omega, alphas.tolist(), betas.tolist(), aic,
        )
        return {
            "model": f"GARCH({p},{q})",
            "omega": omega,
            "alpha": alphas.tolist(),
            "beta": betas.tolist(),
            "log_likelihood": ll,
            "aic": aic,
            "bic": bic,
            "conditional_variances": h,
        }

    # ------------------------------------------------------------------
    # EGARCH(p, q)
    # ------------------------------------------------------------------

    def fit_egarch(
        self,
        returns: np.ndarray,
        p: int = 1,
        q: int = 1,
    ) -> dict:
        """Fit an EGARCH(p,q) model using Nelson's (1991) formulation.

        The log-conditional-variance equation is:

        .. math::
            \\ln h_t = \\omega
                       + \\sum_{i=1}^{q} \\alpha_i g(z_{t-i})
                       + \\sum_{j=1}^{p} \\beta_j \\ln h_{t-j}

        where :math:`g(z) = \\theta z + \\gamma (|z| - E|z|)` and
        :math:`E|z| = \\sqrt{2/\\pi}` for the standard normal.

        Parameters
        ----------
        returns : np.ndarray
            1-D array of returns.
        p : int, optional
            GARCH lag order, by default 1.
        q : int, optional
            ARCH lag order, by default 1.

        Returns
        -------
        dict
            Keys: ``omega``, ``alpha``, ``gamma`` (leverage, length q),
            ``beta`` (length p), ``log_likelihood``, ``aic``, ``bic``,
            ``conditional_variances``, ``model``.
        """
        returns = self._validate_returns(returns)
        n = len(returns)
        # params: omega, alpha_1..q, gamma_1..q, beta_1..p
        n_params = 1 + 2 * q + p
        E_abs_z = math.sqrt(2.0 / math.pi)

        def neg_ll(params: np.ndarray) -> float:
            omega = params[0]
            alphas = params[1: 1 + q]
            gammas = params[1 + q: 1 + 2 * q]
            betas = params[1 + 2 * q:]
            # No positivity constraints needed (log spec)
            log_h = np.full(n, math.log(max(np.var(returns), 1e-12)))
            ll = 0.0
            max_lag = max(p, q)
            for t in range(max_lag, n):
                arch_sum = 0.0
                for i in range(q):
                    h_prev = math.exp(log_h[t - 1 - i])
                    z = returns[t - 1 - i] / max(math.sqrt(h_prev), 1e-10)
                    arch_sum += alphas[i] * (gammas[i] * z + (abs(z) - E_abs_z))
                garch_sum = sum(betas[j] * log_h[t - 1 - j] for j in range(p))
                log_h[t] = max(min(omega + arch_sum + garch_sum, 100.0), -100.0)
                h_t = math.exp(log_h[t])
                h_t = max(h_t, 1e-12)
                ll += -0.5 * (math.log(2 * math.pi) + log_h[t] + returns[t] ** 2 / h_t)
            return -ll

        x0 = np.array([math.log(max(np.var(returns), 1e-12)) * 0.05]
                      + [0.1] * q
                      + [-0.1] * q
                      + [0.9 / p] * p)

        result = minimize(neg_ll, x0, method="L-BFGS-B",
                          options={"maxiter": 2000, "ftol": 1e-12})
        if not result.success:
            logger.warning("EGARCH(%d,%d) optimisation: %s", p, q, result.message)

        omega = float(result.x[0])
        alphas = result.x[1: 1 + q]
        gammas = result.x[1 + q: 1 + 2 * q]
        betas = result.x[1 + 2 * q:]
        ll = float(-result.fun)
        aic, bic = self._aic_bic(ll, n_params, n)

        # Recover conditional variances
        log_h = np.full(n, math.log(max(np.var(returns), 1e-12)))
        max_lag = max(p, q)
        for t in range(max_lag, n):
            arch_sum = 0.0
            for i in range(q):
                h_prev = math.exp(log_h[t - 1 - i])
                z = returns[t - 1 - i] / max(math.sqrt(h_prev), 1e-10)
                arch_sum += alphas[i] * (gammas[i] * z + (abs(z) - E_abs_z))
            log_h[t] = omega + arch_sum + sum(betas[j] * log_h[t - 1 - j] for j in range(p))
        h = np.exp(log_h)

        logger.info(
            "EGARCH(%d,%d) fitted: omega=%.6f, AIC=%.4f",
            p, q, omega, aic,
        )
        return {
            "model": f"EGARCH({p},{q})",
            "omega": omega,
            "alpha": alphas.tolist(),
            "gamma": gammas.tolist(),
            "beta": betas.tolist(),
            "log_likelihood": ll,
            "aic": aic,
            "bic": bic,
            "conditional_variances": h,
        }

    # ------------------------------------------------------------------
    # GJR-GARCH(p, q)
    # ------------------------------------------------------------------

    def fit_gjr_garch(
        self,
        returns: np.ndarray,
        p: int = 1,
        q: int = 1,
    ) -> dict:
        """Fit a GJR-GARCH(p,q) model (Glosten, Jagannathan & Runkle, 1993).

        The conditional variance includes an asymmetric leverage term:

        .. math::
            h_t = \\omega
                  + \\sum_{i=1}^{q} (\\alpha_i + \\gamma_i \\mathbf{1}_{\\varepsilon_{t-i} < 0})
                    \\varepsilon_{t-i}^2
                  + \\sum_{j=1}^{p} \\beta_j h_{t-j}

        Parameters
        ----------
        returns : np.ndarray
            1-D array of returns.
        p : int, optional
            GARCH lag order, by default 1.
        q : int, optional
            ARCH/leverage lag order, by default 1.

        Returns
        -------
        dict
            Keys: ``omega``, ``alpha``, ``gamma`` (leverage, length q),
            ``beta``, ``log_likelihood``, ``aic``, ``bic``,
            ``conditional_variances``, ``model``.
        """
        returns = self._validate_returns(returns)
        n = len(returns)
        n_params = 1 + 2 * q + p

        def neg_ll(params: np.ndarray) -> float:
            omega = params[0]
            alphas = params[1: 1 + q]
            gammas = params[1 + q: 1 + 2 * q]
            betas = params[1 + 2 * q:]
            if omega <= 0 or np.any(alphas < 0) or np.any(betas < 0):
                return 1e12
            # Stationarity: alpha + 0.5*gamma + beta < 1
            if np.sum(alphas) + 0.5 * np.sum(np.maximum(gammas, 0)) + np.sum(betas) >= 1.0:
                return 1e12

            h = np.full(n, np.var(returns))
            ll = 0.0
            max_lag = max(p, q)
            for t in range(max_lag, n):
                arch_sum = 0.0
                for i in range(q):
                    e_sq = returns[t - 1 - i] ** 2
                    indicator = 1.0 if returns[t - 1 - i] < 0 else 0.0
                    arch_sum += (alphas[i] + gammas[i] * indicator) * e_sq
                garch_sum = sum(betas[j] * h[t - 1 - j] for j in range(p))
                h[t] = max(omega + arch_sum + garch_sum, 1e-12)
                ll += -0.5 * (math.log(2 * math.pi) + math.log(h[t]) + returns[t] ** 2 / h[t])
            return -ll

        var0 = np.var(returns)
        x0 = np.array([var0 * 0.05] + [0.05] * q + [0.1] * q + [0.8 / p] * p)
        bounds = [(1e-8, None)] + [(0.0, 1.0)] * (2 * q) + [(0.0, 1.0)] * p

        result = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 2000, "ftol": 1e-12})
        if not result.success:
            logger.warning("GJR-GARCH(%d,%d) optimisation: %s", p, q, result.message)

        omega = float(result.x[0])
        alphas = result.x[1: 1 + q]
        gammas = result.x[1 + q: 1 + 2 * q]
        betas = result.x[1 + 2 * q:]
        ll = float(-result.fun)
        aic, bic = self._aic_bic(ll, n_params, n)

        h = np.full(n, np.var(returns))
        max_lag = max(p, q)
        for t in range(max_lag, n):
            arch_sum = 0.0
            for i in range(q):
                e_sq = returns[t - 1 - i] ** 2
                indicator = 1.0 if returns[t - 1 - i] < 0 else 0.0
                arch_sum += (alphas[i] + gammas[i] * indicator) * e_sq
            h[t] = max(omega + arch_sum + sum(betas[j] * h[t - 1 - j] for j in range(p)), 1e-12)

        logger.info(
            "GJR-GARCH(%d,%d) fitted: omega=%.6f, AIC=%.4f",
            p, q, omega, aic,
        )
        return {
            "model": f"GJR-GARCH({p},{q})",
            "omega": omega,
            "alpha": alphas.tolist(),
            "gamma": gammas.tolist(),
            "beta": betas.tolist(),
            "log_likelihood": ll,
            "aic": aic,
            "bic": bic,
            "conditional_variances": h,
        }

    # ------------------------------------------------------------------
    # Volatility forecasting
    # ------------------------------------------------------------------

    def forecast_volatility(
        self,
        params: dict,
        returns: np.ndarray,
        n_ahead: int = 1,
    ) -> np.ndarray:
        """Generate rolling h-step-ahead annualised volatility forecasts.

        Supports GARCH, EGARCH, and GJR-GARCH parameter dicts as returned
        by the corresponding ``fit_*`` methods.

        Parameters
        ----------
        params : dict
            Fitted model parameters (output of ``fit_garch``,
            ``fit_egarch``, or ``fit_gjr_garch``).
        returns : np.ndarray
            Return series used for initialising the variance filter.
        n_ahead : int, optional
            Forecast horizon in steps, by default 1.

        Returns
        -------
        np.ndarray
            Array of annualised volatility forecasts of length
            ``n_ahead``, continuing from the last observation.

        Raises
        ------
        ValueError
            If ``params`` does not contain a recognised ``model`` key.
        """
        returns = self._validate_returns(returns)
        model = params.get("model", "")
        omega = params["omega"]
        alphas = np.asarray(params["alpha"])
        betas = np.asarray(params["beta"])
        n = len(returns)
        q = len(alphas)
        p = len(betas)

        if "EGARCH" in model:
            gammas = np.asarray(params.get("gamma", np.zeros(q)))
            E_abs_z = math.sqrt(2.0 / math.pi)
            log_h = np.full(n, math.log(max(np.var(returns), 1e-12)))
            max_lag = max(p, q)
            for t in range(max_lag, n):
                arch_sum = 0.0
                for i in range(q):
                    h_prev = math.exp(log_h[t - 1 - i])
                    z = returns[t - 1 - i] / max(math.sqrt(h_prev), 1e-10)
                    arch_sum += alphas[i] * (gammas[i] * z + (abs(z) - E_abs_z))
                log_h[t] = max(min(omega + arch_sum + sum(betas[j] * log_h[t - 1 - j] for j in range(p)), 100.0), -100.0)
            h_last = np.exp(log_h[-p:]) if p > 0 else np.array([np.exp(log_h[-1])])
            res_last = returns[-q:]

            forecasts = []
            log_h_ext = list(log_h)
            ret_ext = list(returns)
            for _ in range(n_ahead):
                t_cur = len(log_h_ext)
                arch_sum = 0.0
                for i in range(q):
                    h_prev = math.exp(log_h_ext[-1 - i])
                    z = ret_ext[-1 - i] / max(math.sqrt(h_prev), 1e-10)
                    arch_sum += alphas[i] * (gammas[i] * z + (abs(z) - E_abs_z))
                log_h_new = omega + arch_sum + sum(betas[j] * log_h_ext[-1 - j] for j in range(p))
                log_h_ext.append(log_h_new)
                ret_ext.append(0.0)  # expected residual = 0
                forecasts.append(math.sqrt(math.exp(log_h_new) * 252))

        elif "GJR" in model:
            gammas = np.asarray(params.get("gamma", np.zeros(q)))
            h = np.full(n, np.var(returns))
            max_lag = max(p, q)
            for t in range(max_lag, n):
                arch_sum = 0.0
                for i in range(q):
                    e_sq = returns[t - 1 - i] ** 2
                    indicator = 1.0 if returns[t - 1 - i] < 0 else 0.0
                    arch_sum += (alphas[i] + gammas[i] * indicator) * e_sq
                h[t] = max(omega + arch_sum + sum(betas[j] * h[t - 1 - j] for j in range(p)), 1e-12)

            h_ext = list(h)
            ret_ext = list(returns)
            forecasts = []
            for _ in range(n_ahead):
                # For multi-step GARCH forecasts, use expected value: E[e^2] = h
                arch_sum = 0.0
                for i in range(q):
                    if i == 0:
                        e_sq = h_ext[-1]
                        # Expected leverage term: P(e<0) = 0.5 for symmetric innovations
                        arch_sum += (alphas[i] + 0.5 * gammas[i]) * e_sq
                    else:
                        e_sq = ret_ext[-i] ** 2 if i < len(ret_ext) else h_ext[-i]
                        indicator = 1.0 if (i < len(ret_ext) and ret_ext[-i] < 0) else 0.5
                        arch_sum += (alphas[i] + gammas[i] * indicator) * e_sq
                h_new = max(omega + arch_sum + sum(betas[j] * h_ext[-1 - j] for j in range(p)), 1e-12)
                h_ext.append(h_new)
                ret_ext.append(0.0)
                forecasts.append(math.sqrt(h_new * 252))

        else:
            # Standard GARCH
            h = np.full(n, np.var(returns))
            max_lag = max(p, q)
            for t in range(max_lag, n):
                arch_term = sum(alphas[i] * returns[t - 1 - i] ** 2 for i in range(q))
                garch_term = sum(betas[j] * h[t - 1 - j] for j in range(p))
                h[t] = max(omega + arch_term + garch_term, 1e-12)

            # Multi-step forecast: iterate using E[e_t^2] = h_t
            h_ext = list(h)
            forecasts = []
            for _ in range(n_ahead):
                arch_term = sum(alphas[i] * h_ext[-1 - i] for i in range(q))
                garch_term = sum(betas[j] * h_ext[-1 - j] for j in range(p))
                h_new = max(omega + arch_term + garch_term, 1e-12)
                h_ext.append(h_new)
                forecasts.append(math.sqrt(h_new * 252))

        return np.array(forecasts)

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    def compare_models(
        self,
        returns: np.ndarray,
        p: int = 1,
        q: int = 1,
    ) -> pd.DataFrame:
        """Fit GARCH, EGARCH, and GJR-GARCH and return AIC/BIC comparison.

        Parameters
        ----------
        returns : np.ndarray
            1-D array of returns.
        p : int, optional
            GARCH order for all models, by default 1.
        q : int, optional
            ARCH order for all models, by default 1.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``model``, ``log_likelihood``, ``aic``,
            ``bic``, sorted by AIC ascending (best first).
        """
        returns = self._validate_returns(returns)
        logger.info("Comparing GARCH family models on %d observations", len(returns))

        results = []
        for fit_fn, label in [
            (self.fit_garch, "GARCH"),
            (self.fit_egarch, "EGARCH"),
            (self.fit_gjr_garch, "GJR-GARCH"),
        ]:
            try:
                res = fit_fn(returns, p=p, q=q)
                results.append({
                    "model": res["model"],
                    "log_likelihood": res["log_likelihood"],
                    "aic": res["aic"],
                    "bic": res["bic"],
                })
            except Exception as exc:
                logger.warning("Failed to fit %s: %s", label, exc)

        df = pd.DataFrame(results).sort_values("aic").reset_index(drop=True)
        logger.info("Model comparison complete:\n%s", df.to_string())
        return df
