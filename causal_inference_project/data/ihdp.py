"""
IHDP-style semi-synthetic data generator.

Generates data following the Infant Health and Development Program (IHDP)
data-generating process described in Hill (2011).  The surface-B response
surface is used by default (nonlinear, heterogeneous effects).

This generator produces realistic causal inference benchmark data with
known ground-truth potential outcomes, allowing rigorous evaluation of
treatment effect estimators.

Reference:
    Hill, J.L. (2011). "Bayesian nonparametric modeling for causal
    inference." JCGS, 20(1).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IHDPDataset:
    """Container for an IHDP-style benchmark dataset."""
    X: np.ndarray
    T: np.ndarray
    Y: np.ndarray
    Y0: np.ndarray
    Y1: np.ndarray
    ITE: np.ndarray
    ATE: float
    ATT: float
    n_samples: int
    n_features: int
    description: str = "IHDP-style semi-synthetic data (surface-B)"

    def train_test_split(self, test_size=0.2, random_state=42):
        """Split into train/test sets, returning two IHDPDataset objects."""
        rng = np.random.RandomState(random_state)
        n = self.n_samples
        idx = rng.permutation(n)
        n_test = int(n * test_size)
        test_idx, train_idx = idx[:n_test], idx[n_test:]

        def _subset(indices):
            return IHDPDataset(
                X=self.X[indices], T=self.T[indices], Y=self.Y[indices],
                Y0=self.Y0[indices], Y1=self.Y1[indices],
                ITE=self.ITE[indices],
                ATE=float((self.Y1[indices] - self.Y0[indices]).mean()),
                ATT=float((self.Y1[indices][self.T[indices] == 1] -
                           self.Y0[indices][self.T[indices] == 1]).mean())
                    if (self.T[indices] == 1).any() else 0.0,
                n_samples=len(indices), n_features=self.n_features,
                description=self.description,
            )
        return _subset(train_idx), _subset(test_idx)


def generate_ihdp(n_samples=747, n_features=25, seed=42, confounding_strength=0.5):
    """Generate an IHDP-style semi-synthetic dataset.

    Args:
        n_samples: Number of observations (default 747 to match real IHDP).
        n_features: Number of covariates (default 25 to match real IHDP).
        seed: Random seed.
        confounding_strength: Controls how strongly X affects treatment
            assignment (higher = more confounding).

    Returns:
        IHDPDataset with X, T, Y, Y0, Y1, ITE, ATE, ATT.
    """
    rng = np.random.RandomState(seed)

    n_cont = n_features // 2
    n_bin = n_features - n_cont

    X_cont = rng.randn(n_samples, n_cont)
    X_bin = rng.binomial(1, 0.5, (n_samples, n_bin)).astype(float)
    X = np.hstack([X_cont, X_bin])

    # Treatment assignment (confounded)
    logits = confounding_strength * (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2])
    logits += confounding_strength * (0.4 * X[:, n_cont] - 0.2 * X[:, n_cont + 1])
    logits += rng.normal(0, 0.1, n_samples)
    propensity = 1 / (1 + np.exp(-logits))
    T = rng.binomial(1, propensity, n_samples).astype(float)

    # Surface-B response (nonlinear, heterogeneous)
    beta = rng.randn(n_features) * 0.5

    mu_0 = np.exp((X[:, :3] + 0.5).sum(axis=1)) + np.sin(X[:, 3] * np.pi)
    mu_0 += X @ beta * 0.3

    treatment_effect = (
        0.5 * np.abs(X[:, 0]) +
        0.3 * X[:, 1] ** 2 +
        0.8 * X[:, n_cont].astype(float) +
        rng.normal(0, 0.1, n_samples)
    )

    Y0 = mu_0 + rng.normal(0, 0.2, n_samples)
    Y1 = mu_0 + treatment_effect + rng.normal(0, 0.2, n_samples)
    Y = T * Y1 + (1 - T) * Y0
    ITE = Y1 - Y0

    ATE = float(ITE.mean())
    treated_mask = T == 1
    ATT = float(ITE[treated_mask].mean()) if treated_mask.any() else ATE

    return IHDPDataset(
        X=X, T=T, Y=Y, Y0=Y0, Y1=Y1, ITE=ITE,
        ATE=ATE, ATT=ATT,
        n_samples=n_samples, n_features=n_features,
    )
