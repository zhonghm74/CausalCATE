# Causal Inference Algorithms Project

A comprehensive Python library implementing **6 causal inference algorithms** for estimating treatment effects from observational data, with an interactive Streamlit UI, benchmark evaluation framework, and 134 automated tests.

## Quick Start

### 1. Install Dependencies

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn pytest pandas streamlit matplotlib plotly
```

### 2. Run Tests

```bash
python -m pytest causal_inference_project/tests/ -v
```

### 3. Launch the Interactive UI

```bash
streamlit run causal_inference_project/app.py --server.port 8501 --server.headless true
```

Open http://localhost:8501 in your browser. The UI provides:
- Interactive pages for each of the 6 algorithms with configurable parameters
- Expandable **📖 Theory** sections with mathematical formulations and architecture diagrams
- **IHDP Benchmark** page for side-by-side algorithm comparison

### 4. Run an Example Script

```bash
python causal_inference_project/examples/dml_example.py
python causal_inference_project/examples/drcfr_example.py
```

---

## Implemented Algorithms

| Algorithm | Type | Key Idea | Module |
|-----------|------|----------|--------|
| **DML** | Statistical | Partials out confounders via two ML models + cross-fitting | `algorithms/dml/` |
| **MIM-DRCFR** | Deep Learning | Disentangled representations (z_y, z_s) with MMD/Sinkhorn regularisation | `algorithms/drcfr/` |
| **SRCVAE** | Deep Learning | Variational autoencoder with latent confounders (u) and instruments (v) | `algorithms/srcvae/` |
| **TARNet** | Deep Learning | Shared representation → two treatment-specific outcome heads | `algorithms/tarnet/` |
| **DragonNet** | Deep Learning | TARNet + propensity head + targeted regularization (TMLE-style) | `algorithms/dragonnet/` |
| **GANITE** | Deep Learning | Two-stage GAN: counterfactual generator + ITE predictor | `algorithms/ganite/` |

---

## DML — Double/Debiased Machine Learning

Cross-fitted DML with Neyman-orthogonal inference (standard errors, confidence intervals, p-values).

```python
from algorithms.dml import double_ml_crossfit

result = double_ml_crossfit(X, y, T, model_y, model_t, treatment_is_binary=True, n_folds=5)
print(result.summary())
# θ = 2.4940 ***,  SE = 0.0049,  95% CI = [2.4843, 2.5036]
```

**Extensions:**

| Feature | Function | Description |
|---------|----------|-------------|
| Cross-fitting | `double_ml_crossfit()` | K-fold sample splitting with inference |
| CATE | `r_learner()`, `dr_learner()` | Heterogeneous treatment effects τ(x) |
| Multi-treatment | `double_ml_multi_treatment()` | Categorical T ∈ {0,1,...,K} |
| Auto selection | `auto_dml()` | CV-based model selection from catalogue |
| IV-DML | `iv_dml()` | Instrumental variable → LATE |
| DiD-DML | `did_dml()` | Difference-in-Differences → ATT |

---

## MIM-DRCFR — Disentangled Representation for Counterfactual Regression

Learns disentangled latent representations with IPM balancing and mutual information minimisation.

```python
from algorithms.drcfr import DRCFRModel

model = DRCFRModel(
    input_dim=25, hidden_dims_phi=[128, 64],
    latent_dim_zy=20, latent_dim_zs=20, hidden_dims_h=[64, 32],
    alpha=1.0, beta=0.1, dropout=0.1, use_batchnorm=True,
    ipm_method='sinkhorn',  # or 'mmd'
    encoder_type='attention',  # or 'mlp'
)
history = model.fit(x_train, y_train, t_train, num_epochs=100, batch_size=128,
                    patience=10, lr_scheduler='plateau')
ite = model.predict_ite(x_test)
ite_mean, ite_std = model.predict_ite_with_uncertainty(x_test, n_mc=50)  # MC Dropout
```

**Features:** Dropout/BatchNorm, Early Stopping, LR Scheduling, Sinkhorn IPM, Attention Encoder, MC Dropout Uncertainty.

---

## SRCVAE — Causal Effect Variational Autoencoder

Deep generative model that explicitly models unobserved confounders (u) and instrumental variables (v).

```python
from algorithms.srcvae import SRCVAEModel

model = SRCVAEModel(
    x_dim=25, t_dim=1, y_dim=1, u_dim=5, v_dim=5,
    hidden_dims_encoder_u=[64, 32], hidden_dims_encoder_v=[64, 32],
    hidden_dims_decoder_x=[32, 64], hidden_dims_decoder_t=[32, 64],
    hidden_dims_decoder_y=[32, 64],
    hidden_dims_aux_qtx=[32], hidden_dims_aux_qyxt=[32],
    alpha_x=1.0, alpha_t=1.0, alpha_y=1.0,
    beta_u=0.1, beta_v=0.1, gamma_t=1.0, gamma_y=1.0,
    heteroscedastic=True, prior='mog', n_mog_components=5,
    conditional_prior_u=True,
)
history = model.fit(x, t, y, num_epochs=100, batch_size=128,
                    kl_warmup_epochs=20, patience=10)
ite, y0, y1 = model.estimate_causal_effect(x_test, n_samples_u=100)
```

**Features:** Heteroscedastic Decoders, KL Annealing, IWAE, MoG Prior, Conditional Prior p(u|x), Semi-supervised Training.

---

## TARNet — Treatment-Agnostic Representation Network

Baseline architecture: shared representation → two outcome heads. Simple, fast, no distributional balancing.

```python
from algorithms.tarnet import TARNetModel

model = TARNetModel(input_dim=25, hidden_dims_shared=[128, 64], hidden_dims_head=[32], dropout=0.1)
model.fit(x_train, y_train, t_train, num_epochs=100, batch_size=128, patience=10)
ite = model.predict_ite(x_test)
```

---

## DragonNet — Adapted Neural Network for Treatment Effects

TARNet + propensity score head + targeted regularization for doubly-robust bias correction.

```python
from algorithms.dragonnet import DragonNetModel

model = DragonNetModel(
    input_dim=25, hidden_dims_shared=[128, 64], hidden_dims_head=[32],
    alpha=1.0, beta=1.0, dropout=0.1,
)
model.fit(x_train, y_train, t_train, num_epochs=100, batch_size=128, patience=10)
ite = model.predict_ite(x_test)
propensity = model.predict_propensity(x_test)
```

---

## GANITE — GAN for Individualised Treatment Effects

Two-stage GAN: Stage 1 trains counterfactual generator + discriminator, Stage 2 trains ITE predictor.

```python
from algorithms.ganite import GANITEModel

model = GANITEModel(input_dim=25, hidden_dims=[64, 32], noise_dim=8)
model.fit(x_train, y_train, t_train, num_epochs=100, batch_size=128)
ite = model.predict_ite(x_test)
```

---

## Data & Evaluation

### IHDP Benchmark Dataset

```python
from data import generate_ihdp

dataset = generate_ihdp(n_samples=747, n_features=25, seed=42)
train, test = dataset.train_test_split(test_size=0.2)
print(f"ATE = {dataset.ATE:.4f}, ATT = {dataset.ATT:.4f}")
```

### Evaluation Metrics

```python
from evaluation import evaluation_report

report = evaluation_report(true_ite, predicted_ite, treatment)
print(report.summary())
# ATE error, PEHE, ATT error, Policy Risk
```

### Synthetic Data Generators

```python
from data import generate_binary_treatment, generate_continuous_treatment, generate_multi_treatment

data = generate_binary_treatment(n_samples=2000, true_effect=2.0)
```

---

## Project Structure

```
causal_inference_project/
├── algorithms/
│   ├── dml/                    # DML + cross-fitting + CATE + IV + DiD + auto
│   │   ├── dml_core.py         # double_ml, double_ml_crossfit, multi-treatment
│   │   ├── dml_cate.py         # R-Learner, DR-Learner
│   │   ├── dml_auto.py         # Automatic model selection
│   │   └── dml_iv.py           # IV-DML (LATE), DiD-DML (ATT)
│   ├── drcfr/                  # MIM-DRCFR + Sinkhorn + Attention + MC Dropout
│   │   ├── networks.py         # Encoder, PredictionHead, AttentionEncoder
│   │   ├── losses.py           # MSE, MMD, Sinkhorn, MI losses
│   │   └── drcfr_model.py      # DRCFRModel
│   ├── srcvae/                 # SRCVAE + heteroscedastic + IWAE + MoG + conditional prior
│   │   ├── srcvae_networks.py  # Encoders, Decoders, ConditionalPriorU
│   │   ├── srcvae_losses.py    # KL, MSE, BCE, Gaussian NLL, KL-to-MoG
│   │   ├── srcvae_model.py     # SRCVAEModel
│   │   └── srcvae_semi.py      # Semi-supervised training
│   ├── tarnet/                 # TARNet baseline
│   ├── dragonnet/              # DragonNet (propensity + targeted reg)
│   └── ganite/                 # GANITE (two-stage GAN)
├── data/
│   ├── ihdp.py                 # IHDP benchmark generator
│   └── synthetic.py            # Binary/continuous/multi-treatment generators
├── evaluation/
│   └── metrics.py              # ATE error, PEHE, ATT error, policy risk
├── tests/                      # 134 automated tests
│   ├── test_dml_core.py
│   ├── test_dml_extensions.py
│   ├── test_drcfr.py
│   ├── test_srcvae.py
│   ├── test_phase1.py
│   ├── test_phase2.py
│   ├── test_phase3.py
│   └── test_dragonnet.py
├── examples/                   # Runnable example scripts
├── app.py                      # Streamlit interactive UI
└── docs_content.py             # Algorithm theory documentation
```

---

## References

1. Chernozhukov, V., et al. (2018). "Double/debiased machine learning for treatment and structural parameters." *The Econometrics Journal*.
2. Nie, X. & Wager, S. (2021). "Quasi-oracle estimation of heterogeneous treatment effects." *Biometrika*.
3. Shalit, U., Johansson, F.D., & Sontag, D. (2017). "Estimating individual treatment effect: generalization bounds and algorithms." *ICML*.
4. Shi, C., Blei, D.M., & Veitch, V. (2019). "Adapting Neural Networks for the Estimation of Treatment Effects." *NeurIPS*.
5. Yoon, J., Jordon, J., & van der Schaar, M. (2018). "GANITE: Estimation of Individualized Treatment Effects Using Generative Adversarial Nets." *ICLR*.
6. "Learning Causal Effect Variational Autoencoder" (2021). https://arxiv.org/pdf/2109.04999
