"""
Algorithm documentation content for the Streamlit UI.
Each function writes Streamlit markdown to the current page.
"""
import streamlit as st


def show_dml_theory():
    st.markdown(r"""
## Principle

**Double/Debiased Machine Learning (DML)** estimates the causal effect of a treatment $T$ on an outcome $Y$ by "partialling out" the confounding effect of covariates $X$ using two machine learning models.

The core insight: if we remove the predictable part of both $Y$ and $T$ given $X$, the residuals retain only the causal relationship.

### Identification

Under **unconfoundedness** $(Y(0), Y(1)) \perp T \mid X$ and **overlap** $0 < P(T=1|X) < 1$:

$$\theta = E\left[\frac{\partial}{\partial t} E[Y | X, T=t]\right]$$

### Algorithm (Cross-fitted)

1. Split data into $K$ folds
2. For each fold $k$:
   - Train outcome model $\hat{m}(X) = \hat{E}[Y|X]$ on data **excluding** fold $k$
   - Train treatment model $\hat{e}(X) = \hat{E}[T|X]$ on data **excluding** fold $k$
   - Predict on fold $k$: $\tilde{Y}_i = Y_i - \hat{m}(X_i)$, $\tilde{T}_i = T_i - \hat{e}(X_i)$
3. Estimate $\theta$ by regressing $\tilde{Y}$ on $\tilde{T}$ across all folds:
   $$\hat{\theta} = \frac{\sum_i \tilde{T}_i \tilde{Y}_i}{\sum_i \tilde{T}_i^2}$$

### Standard Error (Neyman-Orthogonal)

$$SE(\hat{\theta}) = \sqrt{\frac{1}{n} \cdot \frac{E[\psi_i^2]}{(E[\tilde{T}_i^2])^2}}, \quad \psi_i = (\tilde{Y}_i - \hat{\theta} \tilde{T}_i) \tilde{T}_i$$

## Architecture

```
  X ──┬──► Outcome Model m(X)  ──► Y_res = Y - m(X)
      │
      └──► Treatment Model e(X) ──► T_res = T - e(X)

  θ = OLS(Y_res ~ T_res)   ──►  Causal Effect + SE + CI
```

## Extensions in This Project

| Extension | Module | Description |
|-----------|--------|-------------|
| **Cross-fitting** | `dml_core.py` | K-fold sample splitting with Neyman-orthogonal inference |
| **CATE** | `dml_cate.py` | R-Learner, DR-Learner & X-Learner for $\tau(x) = E[Y(1)-Y(0)|X=x]$ |
| **Multi-treatment** | `dml_core.py` | Categorical $T \in \{0,1,...,K\}$ via one-vs-reference |
| **Auto selection** | `dml_auto.py` | CV-based model selection from a catalogue |
| **IV-DML** | `dml_iv.py` | Instrumental variable DML for LATE |
| **DiD-DML** | `dml_iv.py` | Difference-in-Differences with ML nuisance |

## Usage

```python
from algorithms.dml import double_ml_crossfit
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

result = double_ml_crossfit(
    X, y, T,
    ml_model_y=RandomForestRegressor(),
    ml_model_t=RandomForestClassifier(),
    treatment_is_binary=True,
    n_folds=5
)
print(result.summary())
# θ = 2.49 ***, SE = 0.005, 95% CI = [2.48, 2.50]

# X-Learner for heterogeneous effects
from algorithms.dml import x_learner
from sklearn.linear_model import Ridge

cate = x_learner(
    X, y, T,
    ml_model_y0=LinearRegression(), ml_model_y1=LinearRegression(),
    ml_model_t=LogisticRegression(),
    cate_model_0=Ridge(), cate_model_1=Ridge(),
)
print(f"ATE = {cate.ate:.4f}, τ(x) std = {cate.tau_hat.std():.4f}")
```
""")


def show_drcfr_theory():
    st.markdown(r"""
## Principle

**MIM-DRCFR** (Mutual Information Minimized Disentangled Representation for Counterfactual Regression) learns **disentangled latent representations** to separate outcome-relevant features from treatment-assignment features.

### Key Idea

Not all features are equally relevant for outcome prediction vs. treatment assignment. DRCFR decomposes the learned representation into:
- $z_y$: features that predict the **outcome** (needed for ITE estimation)
- $z_s$: features that predict the **treatment assignment** (confounders to be balanced)

### Loss Function

$$\mathcal{L} = \underbrace{L_R}_{\text{factual MSE}} + \alpha \underbrace{L_{IPM}}_{\text{balance } z_y} + \beta \underbrace{L_I}_{\text{minimize MI}(z_s, T)}$$

- $L_R$: Factual outcome prediction error (separate heads for $Y(0)$ and $Y(1)$)
- $L_{IPM}$: MMD or Sinkhorn divergence between $P(z_y|T=1)$ and $P(z_y|T=0)$ — forces distributional balance
- $L_I$: Minimizes mutual information $I(z_s; T)$ — encourages $z_s$ to be independent of treatment

## Architecture

```
        x
        │
  ┌─────┴─────┐
  │  Encoder   │  φ(x) = shared hidden layers
  └─────┬─────┘
     ┌──┴──┐
    z_y   z_s     ← disentangled representations
     │      │
  ┌──┴──┐   │
  h₀  h₁   │     ← outcome prediction heads
 Y(0) Y(1)  │
             │
      MMD(z_y|T=0, z_y|T=1)   → L_IPM (balance)
      MMD(p(z_s,T), p(z_s)p(T)) → L_I (independence)
```

## Phase 1 & 2 Enhancements

| Feature | Description |
|---------|-------------|
| **Dropout + BatchNorm** | Optional regularization in Encoder & PredictionHead |
| **Early stopping** | Patience-based on validation loss |
| **LR scheduling** | `ReduceLROnPlateau` or `CosineAnnealing` |
| **Sinkhorn distance** | Entropic OT as alternative to MMD for $L_{IPM}$ |
| **Attention encoder** | Multi-head self-attention captures feature interactions |
| **MC Dropout uncertainty** | `predict_ite_with_uncertainty()` returns $(μ, σ)$ |

## Usage

```python
from algorithms.drcfr import DRCFRModel

model = DRCFRModel(
    input_dim=25, hidden_dims_phi=[128, 64],
    latent_dim_zy=20, latent_dim_zs=20,
    hidden_dims_h=[64, 32],
    alpha=1.0, beta=0.1,     # IPM and MI weights
    dropout=0.1, use_batchnorm=True,
    ipm_method='sinkhorn',   # or 'mmd'
)
history = model.fit(x_train, y_train, t_train,
                    num_epochs=100, batch_size=128,
                    patience=10, lr_scheduler='plateau')
ite = model.predict_ite(x_test)
ite_mean, ite_std = model.predict_ite_with_uncertainty(x_test, n_mc=50)
```
""")


def show_srcvae_theory():
    st.markdown(r"""
## Principle

**SRCVAE** (Causal Effect Variational Autoencoder) is a deep generative model that explicitly models **unobserved confounders** ($u$) and **instrumental variables** ($v$) as latent variables within a VAE framework.

### Structural Causal Model

$$v \to X, \quad v \to T, \quad u \to X, \quad u \to T, \quad u \to Y, \quad X \to T, \quad X \to Y, \quad T \to Y$$

- $u$: unobserved confounder (affects $X$, $T$, and $Y$)
- $v$: instrumental variable (affects $X$ and $T$, but **not** $Y$ directly)

### ELBO Objective

$$\log p(x, t, y) \geq E_q[\log p(x|u,v) + \log p(t|x,v) + \log p(y|x,u,t)] - KL(q(u|x,t,y) \| p(u)) - KL(q(v|x,t) \| p(v))$$

Plus auxiliary prediction terms $q(t|x)$ and $q(y|x,t)$ to aid representation learning.

### Causal Effect Estimation

$$\hat{\tau}(x) = E_{u \sim p(u)}[f_Y(x, u, t{=}1)] - E_{u \sim p(u)}[f_Y(x, u, t{=}0)]$$

By marginalizing over the prior $p(u)$ (via Monte Carlo sampling), we remove the confounding bias.

## Architecture

```
  Observed: (x, t, y_f)
                │
  ┌─────────────┼─────────────┐
  │             │             │
  EncoderU    EncoderV        │
  q(u|x,t,y)  q(v|x,t)      │
  │             │             │
  u_sampled   v_sampled       │
  │    │        │    │        │
  │  DecoderX ──┘    │        │
  │  p(x|u,v)       │        │
  │         DecoderT ─┘        │
  │         p(t|x,v)          │
  DecoderY ────────────────────┘
  p(y|x,u,t)

  Auxiliary: q(t|x), q(y|x,t)
```

## Phase 1 & 2 Enhancements

| Feature | Description |
|---------|-------------|
| **Heteroscedastic decoders** | DecoderX, DecoderY output $(μ, \log σ^2)$ with Gaussian NLL loss |
| **KL annealing** | Linear warmup of $\beta_u, \beta_v$ from 0 to target — prevents posterior collapse |
| **IWAE** | $K$-sample importance-weighted ELBO for tighter variational bound |
| **MoG prior** | Mixture-of-Gaussians $p(u) = \sum_m \pi_m \mathcal{N}(\mu_m, \sigma_m^2)$ |
| **Conditional prior** | $p(u|x)$ network instead of fixed $\mathcal{N}(0,I)$ |
| **Semi-supervised** | `semi_supervised_fit()` handles missing outcomes via pseudo-labels |

## Usage

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
)
history = model.fit(x_train, t_train, y_train,
                    num_epochs=100, batch_size=128,
                    kl_warmup_epochs=20, patience=10)
ite, y0, y1 = model.estimate_causal_effect(x_test, n_samples_u=100)
```
""")


def show_tarnet_theory():
    st.markdown(r"""
## Principle

**TARNet** (Treatment-Agnostic Representation Network) is the foundational deep learning architecture for ITE estimation. It learns a shared representation of covariates and predicts potential outcomes through treatment-specific heads.

### Key Idea

Instead of building separate models for $Y(0)$ and $Y(1)$, TARNet shares a representation network $\Phi(x)$ that captures common structure, then uses separate "heads" for each treatment arm.

### Loss Function

$$\mathcal{L} = \frac{1}{N_0}\sum_{i: T_i=0} (Y_i - h_0(\Phi(X_i)))^2 + \frac{1}{N_1}\sum_{i: T_i=1} (Y_i - h_1(\Phi(X_i)))^2$$

Simple factual outcome MSE — no distributional balancing.

## Architecture

```
        x (covariates)
        │
  ┌─────┴─────┐
  │  Shared    │  Φ(x) = Linear → ELU → Linear → ELU → ...
  │  Network   │
  └─────┬─────┘
     ┌──┴──┐
    h₀    h₁         ← treatment-specific heads
    │      │
   Y(0)   Y(1)       ← potential outcomes
    │      │
   ITE = Y(1) - Y(0)
```

### Strengths & Limitations

| ✅ Strengths | ❌ Limitations |
|-------------|---------------|
| Simple, fast to train | No distributional balancing between groups |
| Good baseline for comparison | Representation may capture treatment-predictive but outcome-irrelevant features |
| Few hyperparameters | May suffer from finite-sample bias when groups are imbalanced |

## Usage

```python
from algorithms.tarnet import TARNetModel

model = TARNetModel(
    input_dim=25,
    hidden_dims_shared=[128, 64],
    hidden_dims_head=[32],
    dropout=0.1
)
history = model.fit(x_train, y_train, t_train,
                    num_epochs=100, batch_size=128, patience=10)
ite = model.predict_ite(x_test)
y0, y1 = model.predict_potential_outcomes(x_test)
```
""")


def show_dragonnet_theory():
    st.markdown(r"""
## Principle

**DragonNet** extends TARNet by adding a **propensity score head** $\hat{\varepsilon}(x) = P(T=1|X)$ and a **targeted regularization** term inspired by TMLE (Targeted Maximum Likelihood Estimation).

### Key Insight

The sufficiency theorem states that any representation $\Phi(x)$ that is sufficient to predict the treatment assignment $T$ is also sufficient for unbiased causal effect estimation. By jointly predicting $T$ from $\Phi(x)$, DragonNet ensures the shared representation captures all confounding information.

### Loss Function

$$\mathcal{L} = \underbrace{L_{\text{outcome}}}_{\text{factual MSE}} + \alpha \underbrace{L_{\text{propensity}}}_{\text{BCE for } \hat{\varepsilon}(x)} + \beta \underbrace{L_{\text{targeted}}}_{\text{TMLE-style bias correction}}$$

**Targeted regularization:**
$$L_{\text{targeted}} = \frac{1}{n}\sum_i \left(\hat{Y}^f_i + \epsilon \cdot \left(\frac{T_i}{\hat{\varepsilon}(X_i)} - \frac{1-T_i}{1-\hat{\varepsilon}(X_i)}\right) - Y_i\right)^2$$

where $\epsilon$ is a learnable scalar that adjusts the bias correction magnitude.

## Architecture

```
        x (covariates)
        │
  ┌─────┴─────┐
  │  Shared    │  Φ(x)
  │  Network   │
  └─────┬─────┘
  ┌─────┼─────────┐
  │     │         │
 h₀    h₁       ε(x)      ← NEW: propensity head
 │      │         │
Y(0)   Y(1)   P(T=1|X)

Loss = MSE(Y) + α·BCE(T, ε) + β·Targeted(Y, ε, ε̂)
                                    │
                          Learnable ε parameter
                          for TMLE-style correction
```

### DragonNet vs TARNet vs DRCFR

| Feature | TARNet | DragonNet | DRCFR |
|---------|--------|-----------|-------|
| Shared representation | ✅ | ✅ | ✅ |
| Propensity modeling | ❌ | ✅ (3rd head) | ❌ |
| Distributional balance | ❌ | Implicit (via propensity) | Explicit (MMD/Sinkhorn) |
| Targeted regularization | ❌ | ✅ (TMLE-style) | ❌ |
| Disentanglement | ❌ | ❌ | ✅ ($z_y$, $z_s$) |

## Usage

```python
from algorithms.dragonnet import DragonNetModel

model = DragonNetModel(
    input_dim=25,
    hidden_dims_shared=[128, 64],
    hidden_dims_head=[32],
    alpha=1.0,    # propensity loss weight
    beta=1.0,     # targeted regularization weight
    dropout=0.1
)
history = model.fit(x_train, y_train, t_train,
                    num_epochs=100, batch_size=128, patience=10)
ite = model.predict_ite(x_test)
propensity = model.predict_propensity(x_test)  # P(T=1|X)
```
""")


def show_ganite_theory():
    st.markdown(r"""
## Principle

**GANITE** (Generative Adversarial Nets for Inference of Individualised Treatment Effects) uses a GAN framework to generate counterfactual outcomes and then train an ITE predictor on those generated outcomes.

### Key Insight

The fundamental problem of causal inference is that we never observe both $Y(0)$ and $Y(1)$ for the same individual. GANITE addresses this by **learning to generate the missing counterfactual** using adversarial training.

### Two-Stage Training

**Stage 1 — Counterfactual Generator + Discriminator:**
- Generator $G(x, t, y^f, z)$: takes covariates, observed treatment, factual outcome, and noise $z$, produces the counterfactual outcome $\hat{y}^{cf}$
- Discriminator $D(x, t, y_0, y_1)$: tries to distinguish real (factual, counterfactual) pairs from generated pairs
- $G$ is also supervised on the factual outcome for reconstruction

**Stage 2 — ITE Predictor:**
- Use the trained $G$ to generate pseudo-counterfactuals for all training data
- Construct pseudo-ITEs: $\hat{\tau}_i = \hat{Y}_1^i - \hat{Y}_0^i$
- Train a separate ITE network $I(x)$ to predict $\hat{\tau}$ from $x$ alone

## Architecture

```
  Stage 1: Counterfactual Block

  ┌─────────────────────────────────────┐
  │   x, t, y_f, z ──► Generator G     │
  │                      │              │
  │               ŷ_cf (counterfactual) │
  │                      │              │
  │   Construct: (y₀, y₁) pair         │
  │              │                      │
  │   ──────► Discriminator D           │
  │          real vs generated?         │
  └─────────────────────────────────────┘

  Stage 2: ITE Block

  ┌─────────────────────────────────────┐
  │   From Stage 1:                     │
  │   ŷ₀, ŷ₁ ──► τ_pseudo = ŷ₁ - ŷ₀   │
  │                                     │
  │   x ──► ITE Predictor I(x) ──► τ̂   │
  │                                     │
  │   Loss = MSE(τ̂, τ_pseudo)          │
  └─────────────────────────────────────┘
```

### Strengths & Limitations

| ✅ Strengths | ❌ Limitations |
|-------------|---------------|
| No functional form assumptions | GAN training can be unstable |
| Generates full counterfactual outcomes | Two-stage → error propagation |
| Works with complex outcome surfaces | Harder to tune (Generator + Discriminator balance) |
| Noise injection provides diversity | Slower than direct methods |

## Usage

```python
from algorithms.ganite import GANITEModel

model = GANITEModel(
    input_dim=25,
    hidden_dims=[64, 32],
    noise_dim=8,
    learning_rate=1e-3
)
history = model.fit(x_train, y_train, t_train,
                    num_epochs=100, batch_size=128)
# history contains 'g_loss', 'd_loss', 'i_loss'
ite = model.predict_ite(x_test)
y0, y1 = model.predict_potential_outcomes(x_test)
```
""")
