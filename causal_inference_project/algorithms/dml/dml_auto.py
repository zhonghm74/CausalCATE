"""
Automatic model selection wrapper for DML.

Tries a catalogue of ML models for the outcome and treatment nuisance
functions, selects the best by cross-validation, and runs cross-fitted DML
with the chosen pair.
"""

import numpy as np
from copy import deepcopy
from sklearn.model_selection import cross_val_score
from dataclasses import dataclass, field
from typing import List, Optional, Callable

from .dml_core import double_ml_crossfit, DMLResult


@dataclass
class AutoDMLResult:
    """Result from automatic DML including model selection metadata."""
    dml_result: DMLResult
    best_model_y_name: str
    best_model_t_name: str
    model_y_scores: dict = field(repr=False)
    model_t_scores: dict = field(repr=False)

    def summary(self) -> str:
        lines = [
            "AutoDML Result",
            f"  Best outcome model:   {self.best_model_y_name} "
            f"(CV R² = {self.model_y_scores[self.best_model_y_name]:.4f})",
            f"  Best treatment model: {self.best_model_t_name} "
            f"(CV score = {self.model_t_scores[self.best_model_t_name]:.4f})",
            "",
            self.dml_result.summary(),
        ]
        return "\n".join(lines)


def _default_outcome_candidates():
    from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    return {
        "LinearRegression": LinearRegression(),
        "Lasso(α=0.1)": Lasso(alpha=0.1, max_iter=5000),
        "Ridge(α=1)": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=42),
    }


def _default_binary_treatment_candidates():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    return {
        "LogisticRegression": LogisticRegression(
            solver='liblinear', random_state=42, max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42),
    }


def _default_continuous_treatment_candidates():
    from sklearn.linear_model import LinearRegression, Lasso, Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    return {
        "LinearRegression": LinearRegression(),
        "Lasso(α=0.1)": Lasso(alpha=0.1, max_iter=5000),
        "Ridge(α=1)": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=42),
    }


def auto_dml(
    X, y, T,
    treatment_is_binary=True,
    outcome_candidates=None,
    treatment_candidates=None,
    cv_folds=3,
    dml_folds=5,
    random_state=42,
    alpha=0.05,
    scoring_y='r2',
    scoring_t=None,
):
    """Run DML with automatic model selection.

    For each candidate model, cross-validation is used to estimate
    predictive performance.  The best outcome model and best treatment
    model are then used in cross-fitted DML.

    Args:
        X: Feature matrix (n, p).
        y: Outcome vector (n,).
        T: Treatment vector (n,).
        treatment_is_binary: Whether T is binary.
        outcome_candidates: Dict of {name: model} for outcome. If None,
            uses a default catalogue.
        treatment_candidates: Dict of {name: model} for treatment. If
            None, uses a default catalogue.
        cv_folds: Number of CV folds for model selection.
        dml_folds: Number of folds for DML cross-fitting.
        random_state: Random seed.
        alpha: Significance level for CI.
        scoring_y: Scoring metric for outcome model selection.
        scoring_t: Scoring metric for treatment model. Defaults to
            'accuracy' for binary, 'r2' for continuous.

    Returns:
        AutoDMLResult with DML estimates and model selection metadata.
    """
    X, y, T = np.asarray(X), np.asarray(y), np.asarray(T)

    if outcome_candidates is None:
        outcome_candidates = _default_outcome_candidates()
    if treatment_candidates is None:
        if treatment_is_binary:
            treatment_candidates = _default_binary_treatment_candidates()
        else:
            treatment_candidates = _default_continuous_treatment_candidates()
    if scoring_t is None:
        scoring_t = 'accuracy' if treatment_is_binary else 'r2'

    model_y_scores = {}
    for name, model in outcome_candidates.items():
        scores = cross_val_score(
            deepcopy(model), X, y, cv=cv_folds, scoring=scoring_y,
            error_score='raise',
        )
        model_y_scores[name] = float(scores.mean())

    model_t_scores = {}
    for name, model in treatment_candidates.items():
        scores = cross_val_score(
            deepcopy(model), X, T, cv=cv_folds, scoring=scoring_t,
            error_score='raise',
        )
        model_t_scores[name] = float(scores.mean())

    best_y_name = max(model_y_scores, key=model_y_scores.get)
    best_t_name = max(model_t_scores, key=model_t_scores.get)

    best_model_y = deepcopy(outcome_candidates[best_y_name])
    best_model_t = deepcopy(treatment_candidates[best_t_name])

    dml_result = double_ml_crossfit(
        X, y, T, best_model_y, best_model_t,
        treatment_is_binary=treatment_is_binary,
        n_folds=dml_folds, random_state=random_state, alpha=alpha,
    )

    return AutoDMLResult(
        dml_result=dml_result,
        best_model_y_name=best_y_name,
        best_model_t_name=best_t_name,
        model_y_scores=model_y_scores,
        model_t_scores=model_t_scores,
    )
