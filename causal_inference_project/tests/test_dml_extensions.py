"""Tests for DML extensions: cross-fitting, CATE, auto, IV, DiD."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from algorithms.dml.dml_core import (
    double_ml, double_ml_crossfit, double_ml_multi_treatment,
    DMLResult, DMLMultiResult,
)
from algorithms.dml.dml_cate import r_learner, dr_learner, CATEResult
from algorithms.dml.dml_auto import auto_dml, AutoDMLResult
from algorithms.dml.dml_iv import iv_dml, did_dml, IVDMLResult, DiDDMLResult


# ---------------------------------------------------------------------------
# Shared data generators
# ---------------------------------------------------------------------------

def _binary_data(n=2000, true_effect=2.0, seed=42):
    np.random.seed(seed)
    X = np.random.rand(n, 5)
    prop = 1 / (1 + np.exp(-X[:, 0] - X[:, 1]))
    T = np.random.binomial(1, prop, n)
    y0 = X[:, 2] + X[:, 3] + np.random.normal(0, 0.1, n)
    y1 = y0 + true_effect
    y = T * y1 + (1 - T) * y0
    return X, y, T, true_effect


def _continuous_data(n=2000, true_effect=-1.5, seed=123):
    np.random.seed(seed)
    X = np.random.rand(n, 5)
    T = X[:, 0] + 0.5 * X[:, 1] + np.random.normal(0, 0.1, n)
    conf = X[:, 2] * 0.5 + X[:, 3] * 0.3
    y = conf + true_effect * T + np.random.normal(0, 0.1, n)
    return X, y, T, true_effect


def _multi_treatment_data(n=3000, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, 5)
    probs = np.column_stack([
        np.exp(0.5 * X[:, 0]),
        np.exp(0.3 * X[:, 1]),
        np.exp(0.2 * X[:, 2]),
    ])
    probs /= probs.sum(axis=1, keepdims=True)
    T = np.array([np.random.choice([0, 1, 2], p=p) for p in probs])
    effects = {0: 0.0, 1: 1.5, 2: 3.0}
    y = X[:, 3] + np.array([effects[t] for t in T]) + np.random.normal(0, 0.2, n)
    return X, y, T, effects


def _iv_data(n=2000, true_late=3.0, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, 3)
    U = np.random.randn(n)
    Z = (np.random.rand(n) > 0.5).astype(float)
    T = (0.5 * X[:, 0] + 1.5 * Z + 0.5 * U + np.random.normal(0, 0.3, n) > 1).astype(float)
    y = X[:, 1] + true_late * T + U + np.random.normal(0, 0.3, n)
    return X, y, T, Z, true_late


def _did_data(n=2000, true_att=2.0, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, 3)
    T = (X[:, 0] + np.random.normal(0, 0.5, n) > 0).astype(float)
    y_pre = X[:, 1] + np.random.normal(0, 0.2, n)
    trend = 0.5
    y_post = y_pre + trend + true_att * T + np.random.normal(0, 0.2, n)
    return X, y_pre, y_post, T, true_att


# ===================================================================
# P0: Cross-fitting + inference
# ===================================================================

class TestCrossFitDML:
    def test_backward_compat(self):
        X, y, T, te = _binary_data()
        est = double_ml(X, y, T, LinearRegression(),
                        LogisticRegression(solver='liblinear', random_state=42),
                        treatment_is_binary=True)
        assert isinstance(est, (float, np.floating))

    def test_crossfit_returns_result(self):
        X, y, T, te = _binary_data()
        res = double_ml_crossfit(
            X, y, T, LinearRegression(),
            LogisticRegression(solver='liblinear', random_state=42),
            treatment_is_binary=True, n_folds=5,
        )
        assert isinstance(res, DMLResult)
        assert res.n_folds == 5
        assert res.n_samples == len(y)

    def test_crossfit_binary_accuracy(self):
        X, y, T, te = _binary_data(n=3000)
        res = double_ml_crossfit(
            X, y, T, LinearRegression(),
            LogisticRegression(solver='liblinear', random_state=42),
            treatment_is_binary=True, n_folds=5,
        )
        assert res.theta == pytest.approx(te, abs=0.3)
        assert res.ci_lower < te < res.ci_upper

    def test_crossfit_continuous(self):
        X, y, T, te = _continuous_data(n=3000)
        res = double_ml_crossfit(
            X, y, T, LinearRegression(), LinearRegression(),
            treatment_is_binary=False, n_folds=5,
        )
        assert res.theta == pytest.approx(te, abs=0.3)

    def test_se_and_pvalue(self):
        X, y, T, te = _binary_data(n=3000)
        res = double_ml_crossfit(
            X, y, T, LinearRegression(),
            LogisticRegression(solver='liblinear', random_state=42),
            treatment_is_binary=True,
        )
        assert res.se > 0
        assert 0 <= res.p_value <= 1
        assert res.ci_lower < res.theta < res.ci_upper

    def test_summary_string(self):
        X, y, T, te = _binary_data(n=500)
        res = double_ml_crossfit(
            X, y, T, LinearRegression(),
            LogisticRegression(solver='liblinear', random_state=42),
            treatment_is_binary=True,
        )
        s = res.summary()
        assert "DML Estimate" in s
        assert "SE" in s

    def test_residuals_stored(self):
        X, y, T, _ = _binary_data(n=500)
        res = double_ml_crossfit(
            X, y, T, LinearRegression(),
            LogisticRegression(solver='liblinear', random_state=42),
            treatment_is_binary=True,
        )
        assert res.y_residuals is not None
        assert res.t_residuals is not None
        assert len(res.y_residuals) == len(y)


# ===================================================================
# P1: Multi-treatment
# ===================================================================

class TestMultiTreatment:
    def test_multi_returns_result(self):
        X, y, T, effects = _multi_treatment_data()
        res = double_ml_multi_treatment(
            X, y, T,
            ml_model_y=LinearRegression(),
            ml_model_t_factory=lambda: LogisticRegression(solver='liblinear', random_state=42),
            n_folds=3, reference_level=0,
        )
        assert isinstance(res, DMLMultiResult)
        assert res.reference_level == 0
        assert 1 in res.effects
        assert 2 in res.effects

    def test_multi_accuracy(self):
        X, y, T, effects = _multi_treatment_data(n=4000)
        res = double_ml_multi_treatment(
            X, y, T,
            ml_model_y=RandomForestRegressor(n_estimators=50, random_state=42),
            ml_model_t_factory=lambda: LogisticRegression(solver='liblinear', random_state=42),
            n_folds=3, reference_level=0,
        )
        assert res.effects[1].theta == pytest.approx(effects[1], abs=0.5)
        assert res.effects[2].theta == pytest.approx(effects[2], abs=0.5)

    def test_multi_summary(self):
        X, y, T, _ = _multi_treatment_data(n=500)
        res = double_ml_multi_treatment(
            X, y, T, LinearRegression(),
            lambda: LogisticRegression(solver='liblinear', random_state=42),
            n_folds=3,
        )
        s = res.summary()
        assert "Multi-Treatment" in s

    def test_invalid_reference(self):
        X, y, T, _ = _multi_treatment_data(n=500)
        with pytest.raises(ValueError, match="reference_level"):
            double_ml_multi_treatment(
                X, y, T, LinearRegression(),
                lambda: LogisticRegression(solver='liblinear', random_state=42),
                reference_level=99,
            )


# ===================================================================
# P1: CATE (R-Learner, DR-Learner)
# ===================================================================

class TestCATE:
    def test_r_learner_returns_result(self):
        X, y, T, _ = _binary_data(n=1000)
        res = r_learner(
            X, y, T,
            ml_model_y=LinearRegression(),
            ml_model_t=LogisticRegression(solver='liblinear', random_state=42),
            cate_model=Ridge(alpha=1.0),
            n_folds=3,
        )
        assert isinstance(res, CATEResult)
        assert res.method == "R-Learner"
        assert len(res.tau_hat) == len(y)

    def test_r_learner_constant_effect(self):
        X, y, T, te = _binary_data(n=3000)
        res = r_learner(
            X, y, T,
            ml_model_y=RandomForestRegressor(n_estimators=50, random_state=42),
            ml_model_t=RandomForestClassifier(n_estimators=50, random_state=42),
            cate_model=Ridge(alpha=1.0),
            n_folds=3,
        )
        assert res.ate == pytest.approx(te, abs=0.5)
        assert res.tau_hat.std() < 1.0

    def test_r_learner_predict(self):
        X, y, T, _ = _binary_data(n=1000)
        res = r_learner(
            X, y, T,
            LinearRegression(),
            LogisticRegression(solver='liblinear', random_state=42),
            Ridge(), n_folds=3,
        )
        X_new = np.random.rand(10, 5)
        tau_new = res.predict(X_new)
        assert tau_new.shape == (10,)

    def test_dr_learner_returns_result(self):
        X, y, T, _ = _binary_data(n=1000)
        res = dr_learner(
            X, y, T,
            ml_model_y0=LinearRegression(),
            ml_model_y1=LinearRegression(),
            ml_model_t=LogisticRegression(solver='liblinear', random_state=42),
            cate_model=Ridge(),
            n_folds=3,
        )
        assert isinstance(res, CATEResult)
        assert res.method == "DR-Learner"
        assert res.pseudo_outcomes is not None

    def test_dr_learner_constant_effect(self):
        X, y, T, te = _binary_data(n=3000)
        res = dr_learner(
            X, y, T,
            ml_model_y0=RandomForestRegressor(n_estimators=50, random_state=42),
            ml_model_y1=RandomForestRegressor(n_estimators=50, random_state=42),
            ml_model_t=RandomForestClassifier(n_estimators=50, random_state=42),
            cate_model=Ridge(),
            n_folds=3,
        )
        assert res.ate == pytest.approx(te, abs=0.5)

    def test_cate_summary(self):
        X, y, T, _ = _binary_data(n=500)
        res = r_learner(X, y, T, LinearRegression(),
                        LogisticRegression(solver='liblinear', random_state=42),
                        Ridge(), n_folds=3)
        s = res.summary()
        assert "R-Learner" in s
        assert "ATE" in s


# ===================================================================
# P2: Auto DML
# ===================================================================

class TestAutoDML:
    def test_auto_returns_result(self):
        X, y, T, _ = _binary_data(n=500)
        res = auto_dml(X, y, T, treatment_is_binary=True,
                       cv_folds=2, dml_folds=3)
        assert isinstance(res, AutoDMLResult)
        assert isinstance(res.dml_result, DMLResult)
        assert res.best_model_y_name in res.model_y_scores
        assert res.best_model_t_name in res.model_t_scores

    def test_auto_accuracy(self):
        X, y, T, te = _binary_data(n=2000)
        res = auto_dml(X, y, T, treatment_is_binary=True,
                       cv_folds=2, dml_folds=5)
        assert res.dml_result.theta == pytest.approx(te, abs=0.5)

    def test_auto_continuous(self):
        X, y, T, te = _continuous_data(n=2000)
        res = auto_dml(X, y, T, treatment_is_binary=False,
                       cv_folds=2, dml_folds=3)
        assert res.dml_result.theta == pytest.approx(te, abs=0.5)

    def test_auto_summary(self):
        X, y, T, _ = _binary_data(n=500)
        res = auto_dml(X, y, T, cv_folds=2, dml_folds=3)
        s = res.summary()
        assert "AutoDML" in s
        assert "Best outcome model" in s

    def test_auto_custom_candidates(self):
        X, y, T, te = _binary_data(n=1000)
        res = auto_dml(
            X, y, T,
            treatment_is_binary=True,
            outcome_candidates={"LR": LinearRegression()},
            treatment_candidates={
                "Logistic": LogisticRegression(solver='liblinear', random_state=42)
            },
            cv_folds=2, dml_folds=3,
        )
        assert res.best_model_y_name == "LR"
        assert res.best_model_t_name == "Logistic"


# ===================================================================
# P2: IV-DML
# ===================================================================

class TestIVDML:
    def test_iv_returns_result(self):
        X, y, T, Z, _ = _iv_data(n=1000)
        res = iv_dml(
            X, y, T, Z,
            LinearRegression(), LogisticRegression(solver='liblinear', random_state=42),
            LogisticRegression(solver='liblinear', random_state=42),
            n_folds=3,
        )
        assert isinstance(res, IVDMLResult)
        assert res.first_stage_f_stat > 0

    def test_iv_accuracy(self):
        X, y, T, Z, late = _iv_data(n=5000)
        res = iv_dml(
            X, y, T, Z,
            LinearRegression(),
            LogisticRegression(solver='liblinear', random_state=42),
            LogisticRegression(solver='liblinear', random_state=42),
            n_folds=5,
        )
        assert res.theta == pytest.approx(late, abs=1.5)

    def test_iv_summary(self):
        X, y, T, Z, _ = _iv_data(n=500)
        res = iv_dml(
            X, y, T, Z,
            LinearRegression(),
            LogisticRegression(solver='liblinear', random_state=42),
            LogisticRegression(solver='liblinear', random_state=42),
            n_folds=3,
        )
        s = res.summary()
        assert "IV-DML" in s
        assert "1st-stage F" in s


# ===================================================================
# P2: DiD-DML
# ===================================================================

class TestDiDDML:
    def test_did_returns_result(self):
        X, y_pre, y_post, T, _ = _did_data(n=1000)
        res = did_dml(
            X, y_pre, y_post, T,
            LinearRegression(),
            LogisticRegression(solver='liblinear', random_state=42),
            n_folds=3,
        )
        assert isinstance(res, DiDDMLResult)

    def test_did_accuracy(self):
        X, y_pre, y_post, T, att = _did_data(n=3000)
        res = did_dml(
            X, y_pre, y_post, T,
            LinearRegression(),
            LogisticRegression(solver='liblinear', random_state=42),
            n_folds=5,
        )
        assert res.theta == pytest.approx(att, abs=0.5)
        assert res.ci_lower < att < res.ci_upper

    def test_did_summary(self):
        X, y_pre, y_post, T, _ = _did_data(n=500)
        res = did_dml(
            X, y_pre, y_post, T,
            LinearRegression(),
            LogisticRegression(solver='liblinear', random_state=42),
            n_folds=3,
        )
        s = res.summary()
        assert "DiD-DML" in s
        assert "ATT" in s
