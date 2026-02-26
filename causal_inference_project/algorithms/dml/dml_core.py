"""
Double/Debiased Machine Learning (DML) Core Implementation

Provides single-split DML, K-fold cross-fitted DML with inference
(standard errors, confidence intervals), and multi-valued treatment support.

References:
    Chernozhukov, V., et al. (2018). "Double/debiased machine learning for
    treatment and structural parameters." The Econometrics Journal, 21(1).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from sklearn.model_selection import KFold
from copy import deepcopy
import scipy.stats as stats


@dataclass
class DMLResult:
    """Container for DML estimation results with inference."""
    theta: float
    se: float
    ci_lower: float
    ci_upper: float
    t_stat: float
    p_value: float
    n_samples: int
    n_folds: int
    y_residuals: Optional[np.ndarray] = field(default=None, repr=False)
    t_residuals: Optional[np.ndarray] = field(default=None, repr=False)

    def summary(self) -> str:
        sig = "*" if self.p_value < 0.05 else ""
        if self.p_value < 0.01:
            sig = "**"
        if self.p_value < 0.001:
            sig = "***"
        return (
            f"DML Estimate\n"
            f"  θ       = {self.theta:.6f} {sig}\n"
            f"  SE      = {self.se:.6f}\n"
            f"  t-stat  = {self.t_stat:.4f}\n"
            f"  p-value = {self.p_value:.4f}\n"
            f"  95% CI  = [{self.ci_lower:.6f}, {self.ci_upper:.6f}]\n"
            f"  N       = {self.n_samples}, K-folds = {self.n_folds}"
        )


@dataclass
class DMLMultiResult:
    """Container for multi-treatment DML results."""
    effects: dict
    reference_level: int

    def summary(self) -> str:
        lines = [f"DML Multi-Treatment (reference = {self.reference_level})"]
        for level, res in self.effects.items():
            lines.append(
                f"  T={level}: θ={res.theta:.4f}  SE={res.se:.4f}  "
                f"95%CI=[{res.ci_lower:.4f}, {res.ci_upper:.4f}]  p={res.p_value:.4f}"
            )
        return "\n".join(lines)


def _predict_treatment(model, X, treatment_is_binary):
    """Get E[T|X] predictions, using predict_proba for binary classifiers."""
    if treatment_is_binary and hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def _ols_coef(t_res, y_res):
    """OLS coefficient from regressing y_res on t_res (no intercept)."""
    return np.dot(t_res, y_res) / np.dot(t_res, t_res)


def _neyman_orthogonal_se(theta, y_res, t_res):
    """Standard error via the Neyman-orthogonal influence function.

    ψ_i = (y_res_i − θ · t_res_i) · t_res_i
    Var(θ) ≈ (1/n) · E[ψ²] / (E[t_res²])²
    """
    n = len(y_res)
    psi = (y_res - theta * t_res) * t_res
    var_psi = np.mean(psi ** 2)
    mean_t_res_sq = np.mean(t_res ** 2)
    if mean_t_res_sq == 0:
        return np.inf
    return np.sqrt(var_psi / (mean_t_res_sq ** 2) / n)


def _build_result(theta, y_res, t_res, n_folds, alpha=0.05):
    """Construct a DMLResult from the estimated theta and residuals."""
    n = len(y_res)
    se = _neyman_orthogonal_se(theta, y_res, t_res)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = theta - z_crit * se
    ci_upper = theta + z_crit * se
    t_stat = theta / se if se > 0 else np.inf
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    return DMLResult(
        theta=float(theta), se=float(se),
        ci_lower=float(ci_lower), ci_upper=float(ci_upper),
        t_stat=float(t_stat), p_value=float(p_value),
        n_samples=n, n_folds=n_folds,
        y_residuals=y_res, t_residuals=t_res,
    )


# ---------------------------------------------------------------------------
# Original single-split DML (backward-compatible)
# ---------------------------------------------------------------------------

def double_ml(X, y, T, ml_model_y, ml_model_t, treatment_is_binary):
    """Single-split DML (backward-compatible). Returns a scalar theta."""
    X, y, T = np.asarray(X), np.asarray(y), np.asarray(T)

    ml_model_y.fit(X, y)
    y_res = y - ml_model_y.predict(X)

    ml_model_t.fit(X, T)
    t_hat = _predict_treatment(ml_model_t, X, treatment_is_binary)
    t_res = T - t_hat

    t_res_reshaped = t_res.reshape(-1, 1)
    theta, _, _, _ = np.linalg.lstsq(t_res_reshaped, y_res, rcond=None)
    return theta[0]


# ---------------------------------------------------------------------------
# P0: K-fold cross-fitted DML with inference
# ---------------------------------------------------------------------------

def double_ml_crossfit(
    X, y, T, ml_model_y, ml_model_t, treatment_is_binary,
    n_folds=5, random_state=42, alpha=0.05,
):
    """K-fold cross-fitted DML with Neyman-orthogonal inference.

    Splits data into K folds.  For each fold k the nuisance models are
    trained on the complement and residuals are predicted out-of-sample on
    fold k.  The final θ is estimated from the full vector of cross-fitted
    residuals together with an asymptotically valid standard error.

    Args:
        X: Feature matrix (n, p).
        y: Outcome vector (n,).
        T: Treatment vector (n,).
        ml_model_y: Outcome nuisance model (sklearn-compatible).
        ml_model_t: Treatment nuisance model (sklearn-compatible).
        treatment_is_binary: Whether T is binary.
        n_folds: Number of cross-fitting folds (default 5).
        random_state: Random seed for fold splitting.
        alpha: Significance level for confidence interval (default 0.05).

    Returns:
        DMLResult with theta, se, confidence interval, p-value, residuals.
    """
    X, y, T = np.asarray(X), np.asarray(y), np.asarray(T)
    n = len(y)

    y_res = np.zeros(n)
    t_res = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]

        model_y_fold = deepcopy(ml_model_y)
        model_t_fold = deepcopy(ml_model_t)

        model_y_fold.fit(X_train, y_train)
        y_res[test_idx] = y_test - model_y_fold.predict(X_test)

        model_t_fold.fit(X_train, T_train)
        t_hat_test = _predict_treatment(model_t_fold, X_test, treatment_is_binary)
        t_res[test_idx] = T_test - t_hat_test

    theta = _ols_coef(t_res, y_res)
    return _build_result(theta, y_res, t_res, n_folds, alpha)


# ---------------------------------------------------------------------------
# P1: Multi-valued treatment DML
# ---------------------------------------------------------------------------

def double_ml_multi_treatment(
    X, y, T, ml_model_y, ml_model_t_factory,
    n_folds=5, random_state=42, alpha=0.05,
    reference_level=0,
):
    """DML for multi-valued (categorical) treatments.

    Estimates the causal effect of each treatment level relative to
    a reference level by converting each contrast into a binary DML problem
    via one-vs-reference residualisation.

    Args:
        X: Feature matrix (n, p).
        y: Outcome vector (n,).
        T: Treatment vector with integer levels (n,).
        ml_model_y: Outcome nuisance model (sklearn-compatible).
        ml_model_t_factory: Callable that returns a *new* treatment model
            instance for each binary sub-problem (e.g.
            ``lambda: LogisticRegression()``).
        n_folds: Number of cross-fitting folds.
        random_state: Random seed.
        alpha: Significance level.
        reference_level: Treatment level to use as the control/reference.

    Returns:
        DMLMultiResult containing a DMLResult for each non-reference level.
    """
    X, y, T = np.asarray(X), np.asarray(y), np.asarray(T, dtype=int)
    levels = sorted(set(T))
    if reference_level not in levels:
        raise ValueError(f"reference_level={reference_level} not found in T")

    effects = {}
    for level in levels:
        if level == reference_level:
            continue
        mask = (T == reference_level) | (T == level)
        X_sub = X[mask]
        y_sub = y[mask]
        T_binary = (T[mask] == level).astype(float)

        result = double_ml_crossfit(
            X_sub, y_sub, T_binary,
            deepcopy(ml_model_y), ml_model_t_factory(),
            treatment_is_binary=True,
            n_folds=n_folds, random_state=random_state, alpha=alpha,
        )
        effects[level] = result

    return DMLMultiResult(effects=effects, reference_level=reference_level)
