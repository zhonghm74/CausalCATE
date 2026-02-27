"""
Standard causal inference evaluation metrics.

All functions accept numpy arrays.  Predicted values may come from any of
the three algorithms (DML, DRCFR, SRCVAE) as long as they provide the
required quantities.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class EvalReport:
    """Structured evaluation report."""
    ate_true: float
    ate_pred: float
    ate_abs_error: float
    pehe: float
    att_true: float = None
    att_pred: float = None
    att_abs_error: float = None
    policy_risk: float = None

    def summary(self) -> str:
        lines = [
            "Causal Inference Evaluation Report",
            f"  ATE  true = {self.ate_true:.4f}",
            f"  ATE  pred = {self.ate_pred:.4f}",
            f"  ATE  |err|= {self.ate_abs_error:.4f}",
            f"  PEHE      = {self.pehe:.4f}",
        ]
        if self.att_true is not None:
            lines.append(f"  ATT  true = {self.att_true:.4f}")
            lines.append(f"  ATT  pred = {self.att_pred:.4f}")
            lines.append(f"  ATT  |err|= {self.att_abs_error:.4f}")
        if self.policy_risk is not None:
            lines.append(f"  Policy risk = {self.policy_risk:.4f}")
        return "\n".join(lines)


def ate_error(ite_true, ite_pred):
    """Absolute error in Average Treatment Effect.

    Args:
        ite_true: Ground-truth ITE array (n,).
        ite_pred: Predicted ITE array (n,).

    Returns:
        float: |ATE_true − ATE_pred|.
    """
    return float(abs(np.mean(ite_true) - np.mean(ite_pred)))


def pehe(ite_true, ite_pred):
    """Precision in Estimation of Heterogeneous Effects.

    PEHE = sqrt( mean( (ITE_true − ITE_pred)² ) )

    Args:
        ite_true: Ground-truth ITE array (n,) or (n,1).
        ite_pred: Predicted ITE array (n,) or (n,1).

    Returns:
        float: PEHE score (lower is better).
    """
    ite_true = np.asarray(ite_true).flatten()
    ite_pred = np.asarray(ite_pred).flatten()
    return float(np.sqrt(np.mean((ite_true - ite_pred) ** 2)))


def att_error(ite_true, ite_pred, treatment):
    """Absolute error in Average Treatment effect on the Treated.

    Args:
        ite_true: Ground-truth ITE array (n,).
        ite_pred: Predicted ITE array (n,).
        treatment: Binary treatment vector (n,).

    Returns:
        float: |ATT_true − ATT_pred|.
    """
    ite_true = np.asarray(ite_true).flatten()
    ite_pred = np.asarray(ite_pred).flatten()
    treatment = np.asarray(treatment).flatten()
    mask = treatment == 1
    if mask.sum() == 0:
        return 0.0
    return float(abs(np.mean(ite_true[mask]) - np.mean(ite_pred[mask])))


def policy_risk(ite_true, ite_pred):
    """Policy risk: expected loss from treating according to the predicted
    ITE sign instead of the true ITE sign.

    Risk = 1 − E[ Y(1)·1{τ̂>0} + Y(0)·1{τ̂≤0} ] / E[ Y(1)·1{τ>0} + Y(0)·1{τ≤0} ]

    Simplified form: fraction of individuals where the sign of predicted
    ITE disagrees with the true ITE, weighted by the absolute true ITE.

    Returns:
        float: Weighted disagreement rate (0 = perfect, 1 = worst).
    """
    ite_true = np.asarray(ite_true).flatten()
    ite_pred = np.asarray(ite_pred).flatten()
    sign_true = (ite_true > 0).astype(float)
    sign_pred = (ite_pred > 0).astype(float)
    weights = np.abs(ite_true)
    total_weight = weights.sum()
    if total_weight == 0:
        return 0.0
    wrong = np.abs(sign_true - sign_pred)
    return float((wrong * weights).sum() / total_weight)


def evaluation_report(ite_true, ite_pred, treatment=None):
    """Compute a full evaluation report.

    Args:
        ite_true: Ground-truth ITE (n,).
        ite_pred: Predicted ITE (n,).
        treatment: Binary treatment vector (n,), optional.

    Returns:
        EvalReport dataclass with all metrics.
    """
    ite_true = np.asarray(ite_true).flatten()
    ite_pred = np.asarray(ite_pred).flatten()

    _ate_err = ate_error(ite_true, ite_pred)
    _pehe = pehe(ite_true, ite_pred)
    _policy = policy_risk(ite_true, ite_pred)

    report = EvalReport(
        ate_true=float(np.mean(ite_true)),
        ate_pred=float(np.mean(ite_pred)),
        ate_abs_error=_ate_err,
        pehe=_pehe,
        policy_risk=_policy,
    )

    if treatment is not None:
        treatment = np.asarray(treatment).flatten()
        mask = treatment == 1
        if mask.any():
            report.att_true = float(np.mean(ite_true[mask]))
            report.att_pred = float(np.mean(ite_pred[mask]))
            report.att_abs_error = att_error(ite_true, ite_pred, treatment)

    return report
