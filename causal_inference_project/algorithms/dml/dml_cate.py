"""
Conditional Average Treatment Effect (CATE) estimators built on DML.

Implements the R-Learner and DR-Learner for heterogeneous treatment effect
estimation τ(x) = E[Y(1) − Y(0) | X = x].

References:
    Nie, X. & Wager, S. (2021). "Quasi-oracle estimation of heterogeneous
    treatment effects." Biometrika, 108(2).
    Kennedy, E. H. (2023). "Towards optimal doubly robust estimation of
    heterogeneous causal effects." Electronic Journal of Statistics.
"""

import numpy as np
from copy import deepcopy
from sklearn.model_selection import KFold
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CATEResult:
    """Container for CATE estimation results."""
    method: str
    tau_hat: np.ndarray = field(repr=False)
    ate: float
    ate_se: float
    cate_model: object = field(default=None, repr=False)
    y_residuals: Optional[np.ndarray] = field(default=None, repr=False)
    t_residuals: Optional[np.ndarray] = field(default=None, repr=False)
    pseudo_outcomes: Optional[np.ndarray] = field(default=None, repr=False)

    def predict(self, X_new):
        """Predict τ(x) for new observations."""
        if self.cate_model is None:
            raise RuntimeError("No fitted CATE model available for prediction")
        return self.cate_model.predict(np.asarray(X_new))

    def summary(self) -> str:
        return (
            f"{self.method} CATE Estimation\n"
            f"  ATE     = {self.ate:.6f} ± {self.ate_se:.6f}\n"
            f"  τ(x) range = [{self.tau_hat.min():.4f}, {self.tau_hat.max():.4f}]\n"
            f"  τ(x) std   = {self.tau_hat.std():.4f}\n"
            f"  N          = {len(self.tau_hat)}"
        )


def _predict_treatment(model, X, treatment_is_binary):
    if treatment_is_binary and hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def _crossfit_residuals(X, y, T, ml_model_y, ml_model_t,
                        treatment_is_binary, n_folds, random_state):
    """Compute out-of-fold residuals for both outcome and treatment."""
    n = len(y)
    y_res = np.zeros(n)
    t_res = np.zeros(n)
    y_hat_full = np.zeros(n)
    t_hat_full = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(X):
        my = deepcopy(ml_model_y)
        mt = deepcopy(ml_model_t)

        my.fit(X[train_idx], y[train_idx])
        y_hat_full[test_idx] = my.predict(X[test_idx])
        y_res[test_idx] = y[test_idx] - y_hat_full[test_idx]

        mt.fit(X[train_idx], T[train_idx])
        t_hat_full[test_idx] = _predict_treatment(mt, X[test_idx], treatment_is_binary)
        t_res[test_idx] = T[test_idx] - t_hat_full[test_idx]

    return y_res, t_res, y_hat_full, t_hat_full


# ---------------------------------------------------------------------------
# R-Learner
# ---------------------------------------------------------------------------

def r_learner(
    X, y, T, ml_model_y, ml_model_t, cate_model,
    treatment_is_binary=True, n_folds=5, random_state=42,
):
    """R-Learner for CATE estimation (Nie & Wager, 2021).

    Minimises the R-loss:  Σ_i [ (Y_res_i − τ(X_i) · T_res_i)² ]
    which is equivalent to a weighted least-squares regression of
    Y_res / T_res on X with weights T_res².

    Args:
        X: Feature matrix (n, p).
        y: Outcome vector (n,).
        T: Treatment vector (n,). Binary recommended.
        ml_model_y: Outcome nuisance model.
        ml_model_t: Treatment nuisance model.
        cate_model: sklearn-compatible model for τ(x). Must support
            ``fit(X, y, sample_weight=...)`` for weighted regression.
            Examples: ``Ridge()``, ``GradientBoostingRegressor()``.
        treatment_is_binary: Whether T is binary.
        n_folds: Number of cross-fitting folds.
        random_state: Random seed.

    Returns:
        CATEResult with per-sample τ̂(x), ATE, and fitted CATE model.
    """
    X, y, T = np.asarray(X, dtype=float), np.asarray(y, dtype=float), np.asarray(T, dtype=float)

    y_res, t_res, _, _ = _crossfit_residuals(
        X, y, T, ml_model_y, ml_model_t,
        treatment_is_binary, n_folds, random_state,
    )

    eps = 1e-6
    weights = t_res ** 2
    pseudo_target = np.where(np.abs(t_res) > eps, y_res / t_res, 0.0)

    valid = np.abs(t_res) > eps
    cate_model_fit = deepcopy(cate_model)
    if hasattr(cate_model_fit, 'fit') and 'sample_weight' in cate_model_fit.fit.__code__.co_varnames:
        cate_model_fit.fit(X[valid], pseudo_target[valid],
                           sample_weight=weights[valid])
    else:
        cate_model_fit.fit(X[valid], pseudo_target[valid])

    tau_hat = cate_model_fit.predict(X)
    ate = float(tau_hat.mean())
    ate_se = float(tau_hat.std() / np.sqrt(len(tau_hat)))

    return CATEResult(
        method="R-Learner", tau_hat=tau_hat,
        ate=ate, ate_se=ate_se, cate_model=cate_model_fit,
        y_residuals=y_res, t_residuals=t_res,
    )


# ---------------------------------------------------------------------------
# DR-Learner
# ---------------------------------------------------------------------------

def dr_learner(
    X, y, T, ml_model_y0, ml_model_y1, ml_model_t, cate_model,
    n_folds=5, random_state=42,
):
    """Doubly-Robust Learner for CATE estimation (Kennedy, 2023).

    Constructs doubly-robust pseudo-outcomes:
        Γ_i = μ̂₁(Xᵢ) − μ̂₀(Xᵢ)
             + Tᵢ·(Yᵢ − μ̂₁(Xᵢ)) / ê(Xᵢ)
             − (1−Tᵢ)·(Yᵢ − μ̂₀(Xᵢ)) / (1 − ê(Xᵢ))
    and regresses Γ on X.

    Args:
        X: Feature matrix (n, p).
        y: Outcome vector (n,).
        T: Binary treatment vector (n,).
        ml_model_y0: Model for E[Y|X, T=0].
        ml_model_y1: Model for E[Y|X, T=1].
        ml_model_t: Propensity model for P(T=1|X).
        cate_model: Final-stage model for τ(x).
        n_folds: Number of cross-fitting folds.
        random_state: Random seed.

    Returns:
        CATEResult with per-sample τ̂(x), ATE, pseudo-outcomes, and model.
    """
    X, y, T = np.asarray(X, dtype=float), np.asarray(y, dtype=float), np.asarray(T, dtype=float)
    n = len(y)

    mu0_hat = np.zeros(n)
    mu1_hat = np.zeros(n)
    e_hat = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, T_tr = y[train_idx], T[train_idx]

        mask0 = T_tr == 0
        mask1 = T_tr == 1

        m0 = deepcopy(ml_model_y0)
        m1 = deepcopy(ml_model_y1)
        mt = deepcopy(ml_model_t)

        if mask0.sum() > 0:
            m0.fit(X_tr[mask0], y_tr[mask0])
            mu0_hat[test_idx] = m0.predict(X_te)
        if mask1.sum() > 0:
            m1.fit(X_tr[mask1], y_tr[mask1])
            mu1_hat[test_idx] = m1.predict(X_te)

        mt.fit(X_tr, T_tr)
        e_hat[test_idx] = _predict_treatment(mt, X_te, treatment_is_binary=True)

    clip_eps = 0.01
    e_clipped = np.clip(e_hat, clip_eps, 1 - clip_eps)

    pseudo = (
        (mu1_hat - mu0_hat)
        + T * (y - mu1_hat) / e_clipped
        - (1 - T) * (y - mu0_hat) / (1 - e_clipped)
    )

    cate_fit = deepcopy(cate_model)
    cate_fit.fit(X, pseudo)
    tau_hat = cate_fit.predict(X)

    ate = float(pseudo.mean())
    ate_se = float(pseudo.std() / np.sqrt(n))

    return CATEResult(
        method="DR-Learner", tau_hat=tau_hat,
        ate=ate, ate_se=ate_se, cate_model=cate_fit,
        pseudo_outcomes=pseudo,
    )
