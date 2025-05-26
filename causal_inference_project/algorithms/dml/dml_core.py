"""
Double/Debiased Machine Learning (DML) Core Implementation

This module provides the core implementation of the Double/Debiased Machine
Learning (DML) algorithm. DML is a method for estimating causal effects in
the presence of confounding variables. It uses machine learning models to
partial out the effects of confounders from both the treatment and outcome
variables, leading to a more robust estimate of the causal effect.
"""
import numpy as np

def double_ml(X, y, T, ml_model_y, ml_model_t, treatment_is_binary):
    """
    Implements the Double/Debiased Machine Learning (DML) algorithm.

    DML estimates the causal effect of a treatment (T) on an outcome (y)
    conditional on a set of features (X). It does so by "partialling out"
    the effect of X from both y and T using two separate machine learning models.
    The causal effect is then estimated from the residuals.

    Args:
        X (numpy.ndarray or pandas.DataFrame): Feature matrix of shape (n_samples, n_features).
            These are the confounding variables.
        y (numpy.ndarray or pandas.Series): Outcome vector of shape (n_samples,).
        T (numpy.ndarray or pandas.Series): Treatment vector of shape (n_samples,).
            Can be binary (0/1) or continuous.
        ml_model_y (object): A machine learning model instance for the outcome model (E[Y|X]).
            Must implement `fit(X, y)` and `predict(X)` methods.
            Example: `sklearn.linear_model.LinearRegression()`, `sklearn.ensemble.RandomForestRegressor()`.
        ml_model_t (object): A machine learning model instance for the treatment model (E[T|X]).
            Must implement `fit(X, y)` and `predict(X)` methods.
            Example: `sklearn.linear_model.LinearRegression()` or `sklearn.linear_model.LogisticRegression()`.
        treatment_is_binary (bool): Boolean indicating if the treatment variable T is binary.
            If True, it's assumed T takes values {0, 1}. This parameter is currently
            used for clarity but not strictly enforced in the residualization logic
            if `ml_model_t` correctly handles binary/continuous predictions.

    Returns:
        float: The estimated causal effect (theta) of T on y, after controlling for X.
               This is the coefficient from the final regression of y_res on T_res.

    Methodology:
    1. Fit the outcome model: `ml_model_y.fit(X, y)` to estimate E[Y|X].
       Calculate outcome residuals: `y_res = y - ml_model_y.predict(X)`.
    2. Fit the treatment model: `ml_model_t.fit(X, T)` to estimate E[T|X].
       Calculate treatment residuals: `T_res = T - ml_model_t.predict(X)`.
    3. Estimate the causal effect (theta) by performing a linear regression of
       `y_res` on `T_res`. The coefficient of `T_res` is the DML estimate.
    """

    # Step 1: Estimate outcome model E[Y|X] and get residuals Y_res = Y - E[Y|X]
    ml_model_y.fit(X, y)
    y_hat = ml_model_y.predict(X)
    y_res = y - y_hat

    # Step 2: Estimate treatment model E[T|X] and get residuals T_res = T - E[T|X]
    ml_model_t.fit(X, T)
    t_hat = ml_model_t.predict(X)
    t_res = T - t_hat

    # Step 3: Estimate the causal effect by regressing y_res on t_res
    # Reshape t_res to be a 2D array for lstsq, as it's a single regressor
    t_res_reshaped = t_res.reshape(-1, 1)
    
    # Using numpy.linalg.lstsq for the final regression
    # lstsq returns: coefficients, residuals, rank, singular_values
    # We are interested in the first element of the coefficients array (theta)
    theta, _, _, _ = np.linalg.lstsq(t_res_reshaped, y_res, rcond=None)

    return theta[0]
