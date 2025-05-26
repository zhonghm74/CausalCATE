# Causal Inference Algorithms Project

This project implements various causal inference algorithms in Python.

## Implemented Algorithms

### Double/Debiased Machine Learning (DML)

Double/Debiased Machine Learning (DML) is a method for estimating causal effects in the presence of confounding variables. It utilizes machine learning models to partial out the effects of these confounders from both the treatment and outcome variables, providing a more robust estimate of the causal effect.

The core implementation of the DML algorithm can be found in `algorithms/dml/dml_core.py`.

For a usage example, see `examples/dml_example.py`. This script demonstrates how to apply the DML algorithm to synthetic data for both binary and continuous treatment variables. It also shows how to integrate different machine learning models from `scikit-learn` for the nuisance function estimation.

### MIM-DRCFR (Learning Disentangled Representations for Counterfactual Regression)

- **Description:** An algorithm that learns disentangled representations of features to improve counterfactual predictions. It separates features into those influencing the outcome (`z_y`) and those influencing treatment assignment (`z_s`), minimizing mutual information between `z_s` and treatment `t` to reduce confounding.
- **Core Implementation:** `algorithms/drcfr/`
- **Example Usage:** `examples/drcfr_example.py`
