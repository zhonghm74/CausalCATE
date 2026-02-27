"""
Instrumental Variable DML (IV-DML) for Local Average Treatment Effect.

Extends DML to settings where unconfoundedness fails but valid
instruments are available.  Also includes a Difference-in-Differences
DML variant for panel / repeated cross-section designs.

References:
    Chernozhukov, V., et al. (2018). "Double/debiased machine learning
    for treatment and structural parameters."
"""

import numpy as np
from copy import deepcopy
from sklearn.model_selection import KFold
from dataclasses import dataclass, field
from typing import Optional

from .dml_core import DMLResult, _neyman_orthogonal_se, _build_result


# ---------------------------------------------------------------------------
# IV-DML (two-stage style with instruments)
# ---------------------------------------------------------------------------

@dataclass
class IVDMLResult:
    """Result container for instrumental-variable DML."""
    theta: float
    se: float
    ci_lower: float
    ci_upper: float
    t_stat: float
    p_value: float
    n_samples: int
    n_folds: int
    first_stage_f_stat: float
    y_residuals: Optional[np.ndarray] = field(default=None, repr=False)
    t_residuals: Optional[np.ndarray] = field(default=None, repr=False)
    z_residuals: Optional[np.ndarray] = field(default=None, repr=False)

    def summary(self) -> str:
        sig = ""
        if self.p_value < 0.05: sig = "*"
        if self.p_value < 0.01: sig = "**"
        if self.p_value < 0.001: sig = "***"
        return (
            f"IV-DML (LATE) Estimate\n"
            f"  θ       = {self.theta:.6f} {sig}\n"
            f"  SE      = {self.se:.6f}\n"
            f"  t-stat  = {self.t_stat:.4f}\n"
            f"  p-value = {self.p_value:.4f}\n"
            f"  95% CI  = [{self.ci_lower:.6f}, {self.ci_upper:.6f}]\n"
            f"  1st-stage F = {self.first_stage_f_stat:.2f}\n"
            f"  N       = {self.n_samples}, K-folds = {self.n_folds}"
        )


def _predict_instrument(model, X, instrument_is_binary):
    if instrument_is_binary and hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def iv_dml(
    X, y, T, Z,
    ml_model_y, ml_model_t, ml_model_z,
    treatment_is_binary=True,
    instrument_is_binary=True,
    n_folds=5, random_state=42, alpha=0.05,
):
    """Cross-fitted IV-DML for estimating the LATE.

    Partials out confounders from Y, T, and Z, then uses IV regression:
        θ = Σ(Z_res · Y_res) / Σ(Z_res · T_res)

    Args:
        X: Feature matrix of confounders (n, p).
        y: Outcome vector (n,).
        T: (Endogenous) treatment vector (n,).
        Z: Instrument vector (n,). Must be relevant for T.
        ml_model_y: Model for E[Y|X].
        ml_model_t: Model for E[T|X].
        ml_model_z: Model for E[Z|X].
        treatment_is_binary: Whether T is binary.
        instrument_is_binary: Whether Z is binary.
        n_folds: Number of cross-fitting folds.
        random_state: Random seed.
        alpha: Significance level.

    Returns:
        IVDMLResult with LATE estimate, SE, CI, and first-stage F-stat.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    T = np.asarray(T, dtype=float)
    Z = np.asarray(Z, dtype=float)
    n = len(y)

    y_res = np.zeros(n)
    t_res = np.zeros(n)
    z_res = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]

        my = deepcopy(ml_model_y)
        my.fit(X_tr, y[train_idx])
        y_res[test_idx] = y[test_idx] - my.predict(X_te)

        mt = deepcopy(ml_model_t)
        mt.fit(X_tr, T[train_idx])
        t_hat = _predict_instrument(mt, X_te, treatment_is_binary)
        t_res[test_idx] = T[test_idx] - t_hat

        mz = deepcopy(ml_model_z)
        mz.fit(X_tr, Z[train_idx])
        z_hat = _predict_instrument(mz, X_te, instrument_is_binary)
        z_res[test_idx] = Z[test_idx] - z_hat

    denom = np.dot(z_res, t_res)
    if abs(denom) < 1e-12:
        raise ValueError("Weak instrument: Z_res has near-zero covariance with T_res")

    theta = np.dot(z_res, y_res) / denom

    # SE via influence function for IV moment:
    # ψ_i = (y_res_i - θ · t_res_i) · z_res_i / E[z_res · t_res]
    psi = (y_res - theta * t_res) * z_res
    J = np.mean(z_res * t_res)
    var_theta = np.mean(psi ** 2) / (J ** 2) / n
    se = np.sqrt(max(var_theta, 0))

    import scipy.stats as st
    z_crit = st.norm.ppf(1 - alpha / 2)
    ci_lower = theta - z_crit * se
    ci_upper = theta + z_crit * se
    t_stat = theta / se if se > 0 else np.inf
    p_value = 2 * (1 - st.norm.cdf(abs(t_stat)))

    # First-stage F-stat: regression of T_res on Z_res
    ss_reg = np.sum((z_res * np.dot(z_res, t_res) / np.dot(z_res, z_res)) ** 2)
    ss_res = np.sum((t_res - z_res * np.dot(z_res, t_res) / np.dot(z_res, z_res)) ** 2)
    f_stat = (ss_reg / 1) / (ss_res / max(n - 2, 1)) if ss_res > 0 else np.inf

    return IVDMLResult(
        theta=float(theta), se=float(se),
        ci_lower=float(ci_lower), ci_upper=float(ci_upper),
        t_stat=float(t_stat), p_value=float(p_value),
        n_samples=n, n_folds=n_folds,
        first_stage_f_stat=float(f_stat),
        y_residuals=y_res, t_residuals=t_res, z_residuals=z_res,
    )


# ---------------------------------------------------------------------------
# Difference-in-Differences DML
# ---------------------------------------------------------------------------

@dataclass
class DiDDMLResult:
    """Result for Difference-in-Differences DML."""
    theta: float
    se: float
    ci_lower: float
    ci_upper: float
    t_stat: float
    p_value: float
    n_samples: int
    n_folds: int

    def summary(self) -> str:
        sig = ""
        if self.p_value < 0.05: sig = "*"
        if self.p_value < 0.01: sig = "**"
        if self.p_value < 0.001: sig = "***"
        return (
            f"DiD-DML Estimate\n"
            f"  ATT     = {self.theta:.6f} {sig}\n"
            f"  SE      = {self.se:.6f}\n"
            f"  t-stat  = {self.t_stat:.4f}\n"
            f"  p-value = {self.p_value:.4f}\n"
            f"  95% CI  = [{self.ci_lower:.6f}, {self.ci_upper:.6f}]\n"
            f"  N       = {self.n_samples}, K-folds = {self.n_folds}"
        )


def did_dml(
    X, y_pre, y_post, T,
    ml_model_dy, ml_model_t,
    n_folds=5, random_state=42, alpha=0.05,
):
    """Difference-in-Differences DML for panel data.

    Uses the change ΔY = Y_post − Y_pre as the outcome and applies
    cross-fitted DML, controlling for pre-treatment covariates.
    Estimates the ATT under a conditional parallel-trends assumption.

    Args:
        X: Pre-treatment covariates (n, p).
        y_pre: Pre-treatment outcome (n,).
        y_post: Post-treatment outcome (n,).
        T: Binary treatment indicator (n,).
        ml_model_dy: Model for E[ΔY | X].
        ml_model_t: Model for P(T=1 | X).
        n_folds: Cross-fitting folds.
        random_state: Random seed.
        alpha: Significance level.

    Returns:
        DiDDMLResult with ATT estimate, SE, and CI.
    """
    X = np.asarray(X, dtype=float)
    y_pre = np.asarray(y_pre, dtype=float)
    y_post = np.asarray(y_post, dtype=float)
    T = np.asarray(T, dtype=float)

    delta_y = y_post - y_pre
    n = len(delta_y)

    dy_res = np.zeros(n)
    t_res = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]

        mdy = deepcopy(ml_model_dy)
        mdy.fit(X_tr, delta_y[train_idx])
        dy_res[test_idx] = delta_y[test_idx] - mdy.predict(X_te)

        mt = deepcopy(ml_model_t)
        mt.fit(X_tr, T[train_idx])
        if hasattr(mt, 'predict_proba'):
            t_hat = mt.predict_proba(X_te)[:, 1]
        else:
            t_hat = mt.predict(X_te)
        t_res[test_idx] = T[test_idx] - t_hat

    denom = np.dot(t_res, t_res)
    if abs(denom) < 1e-12:
        raise ValueError("Treatment residuals have near-zero variance")

    theta = np.dot(t_res, dy_res) / denom
    se = _neyman_orthogonal_se(theta, dy_res, t_res)

    import scipy.stats as st
    z_crit = st.norm.ppf(1 - alpha / 2)
    ci_lower = theta - z_crit * se
    ci_upper = theta + z_crit * se
    t_stat = theta / se if se > 0 else np.inf
    p_value = 2 * (1 - st.norm.cdf(abs(t_stat)))

    return DiDDMLResult(
        theta=float(theta), se=float(se),
        ci_lower=float(ci_lower), ci_upper=float(ci_upper),
        t_stat=float(t_stat), p_value=float(p_value),
        n_samples=n, n_folds=n_folds,
    )
