"""
Unified synthetic data generators for all three algorithms.

Each generator returns a dict with consistent keys so downstream code
(evaluation, UI) can use them interchangeably.
"""

import numpy as np


def generate_binary_treatment(n_samples=2000, n_features=5, true_effect=2.0, seed=42):
    """Binary-treatment DGP with linear confounding."""
    rng = np.random.RandomState(seed)
    X = rng.rand(n_samples, n_features)
    prop = 1 / (1 + np.exp(-X[:, 0] - X[:, 1]))
    T = rng.binomial(1, prop, n_samples).astype(float)
    Y0 = X[:, 2] + X[:, 3] + rng.normal(0, 0.1, n_samples)
    Y1 = Y0 + true_effect
    Y = T * Y1 + (1 - T) * Y0
    return dict(X=X, T=T, Y=Y, Y0=Y0, Y1=Y1, true_effect=true_effect)


def generate_continuous_treatment(n_samples=2000, n_features=5, true_effect=-1.5, seed=123):
    """Continuous-treatment DGP."""
    rng = np.random.RandomState(seed)
    X = rng.rand(n_samples, n_features)
    T = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.1, n_samples)
    conf = X[:, 2] * 0.5 + X[:, 3] * 0.3
    Y = conf + true_effect * T + rng.normal(0, 0.1, n_samples)
    return dict(X=X, T=T, Y=Y, true_effect=true_effect)


def generate_multi_treatment(n_samples=3000, n_features=5, effects=None, seed=42):
    """Multi-valued treatment DGP (3 levels by default)."""
    if effects is None:
        effects = {0: 0.0, 1: 1.5, 2: 3.0}
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    probs = np.column_stack([np.exp(0.5 * X[:, i]) for i in range(len(effects))])
    probs /= probs.sum(axis=1, keepdims=True)
    T = np.array([rng.choice(list(effects.keys()), p=p) for p in probs])
    Y = X[:, -1] + np.array([effects[t] for t in T]) + rng.normal(0, 0.2, n_samples)
    return dict(X=X, T=T, Y=Y, effects=effects)
