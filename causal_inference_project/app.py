"""
Causal Inference Algorithms – Interactive UI

Streamlit application for running and visualizing the three causal inference
algorithms implemented in this project: DML, MIM-DRCFR, and SRCVAE.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from algorithms.dml.dml_core import double_ml, double_ml_crossfit, double_ml_multi_treatment
from algorithms.dml.dml_cate import r_learner, dr_learner
from algorithms.dml.dml_auto import auto_dml
from algorithms.dml.dml_iv import iv_dml, did_dml
from algorithms.drcfr.drcfr_model import DRCFRModel
from algorithms.srcvae.srcvae_model import SRCVAEModel
from algorithms.tarnet.tarnet_model import TARNetModel
from algorithms.dragonnet.dragonnet_model import DragonNetModel
from algorithms.ganite.ganite_model import GANITEModel
import docs_content

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Causal Inference Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Synthetic-data generators (mirrors examples/)
# ---------------------------------------------------------------------------

def generate_dml_data(n_samples, n_features, true_effect, binary_treatment, seed):
    np.random.seed(seed)
    X = np.random.rand(n_samples, n_features)
    if binary_treatment:
        prop = 1 / (1 + np.exp(-X[:, 0] - X[:, 1]))
        T = np.random.binomial(1, prop, n_samples)
        y0 = X[:, 2] + X[:, 3] + np.random.normal(0, 0.1, n_samples)
        y1 = y0 + true_effect
        y = T * y1 + (1 - T) * y0
    else:
        T = X[:, 0] + 0.5 * X[:, 1] + np.random.normal(0, 0.1, n_samples)
        conf = X[:, 2] * 0.5 + X[:, 3] * 0.3
        y = conf + true_effect * T + np.random.normal(0, 0.1, n_samples)
    return X, y, T


def generate_drcfr_data(n_samples, n_features, true_ate_baseline, seed):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    prop = 1 / (1 + np.exp(-(X[:, 0] + X[:, 1])))
    T = np.random.binomial(1, prop, n_samples)
    Y0 = X[:, 0] * 0.5 + X[:, 1] * 0.2 + np.random.normal(0, 0.2, n_samples)
    Y1 = Y0 + X[:, 2] * 0.6 + X[:, 3] * 0.3 + true_ate_baseline + np.random.normal(0, 0.2, n_samples)
    Yf = T * Y1 + (1 - T) * Y0
    return X, T, Yf, Y0, Y1


def generate_srcvae_data(n_samples, x_dim, true_ate, seed):
    np.random.seed(seed)
    U = np.random.normal(0, 1, (n_samples, 1))
    V = np.random.normal(0, 1, (n_samples, 1))
    X = U * 0.7 + V * 0.5 + np.random.normal(0, 0.2, (n_samples, x_dim))
    t_logits = X[:, 0] * 0.5 + U[:, 0] * 0.6 + V[:, 0] * 0.8 - 0.5
    t_probs = 1 / (1 + np.exp(-t_logits))
    T = np.random.binomial(1, t_probs, n_samples).astype(np.float32)
    Y0 = X[:, 1 % x_dim] * 0.4 + U[:, 0] * 1.0 + np.random.normal(0, 0.1, n_samples)
    Y1 = Y0 + true_ate
    Yf = T * Y1 + (1 - T) * Y0
    return X, T, Yf, Y0, Y1

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_PRIMARY = "#4361ee"
C_SECONDARY = "#f72585"
C_SUCCESS = "#06d6a0"
C_WARNING = "#ffd166"
C_BG = "#0e1117"

# ---------------------------------------------------------------------------
# Sidebar – algorithm picker
# ---------------------------------------------------------------------------
st.sidebar.title("Causal Inference Lab")
algo = st.sidebar.radio(
    "Select Algorithm",
    ["Overview", "DML", "MIM-DRCFR", "SRCVAE", "TARNet", "DragonNet", "GANITE", "IHDP Benchmark"],
    index=0,
)
st.sidebar.markdown("---")

# ============================= OVERVIEW =====================================
if algo == "Overview":
    st.title("Causal Inference Algorithms")
    st.markdown(
        """
        Welcome to the **Causal Inference Lab** – an interactive playground for
        exploring three state-of-the-art causal effect estimation methods.

        Use the sidebar to select an algorithm, configure parameters, generate
        synthetic data, train models and visualise the results.

        ---

        ### Implemented Algorithms

        | Algorithm | Type | Key Idea |
        |-----------|------|----------|
        | **DML** | Statistical | Double/Debiased Machine Learning – partials out confounders via two ML models |
        | **MIM-DRCFR** | Deep Learning | Disentangled representations (z_y, z_s) with MMD regularisation |
        | **SRCVAE** | Deep Learning | Variational autoencoder with latent confounders (u) and instruments (v) |

        ---

        ### Quick Start

        1. Pick an algorithm from the left.
        2. Adjust the data & model parameters.
        3. Click **Run** and inspect the results.
        """
    )

# ============================= DML ==========================================
elif algo == "DML":
    st.title("Double / Debiased Machine Learning (DML)")
    with st.expander("📖 Algorithm Theory, Architecture & Usage", expanded=False):
        docs_content.show_dml_theory()

    from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression, Ridge
    from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                                  GradientBoostingRegressor)

    MODEL_MAP = {
        "LinearRegression": lambda: LinearRegression(),
        "Lasso": lambda: Lasso(alpha=0.1, max_iter=5000),
        "Ridge": lambda: Ridge(alpha=1.0),
        "RandomForestRegressor": lambda: RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        "GradientBoosting": lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
        "LogisticRegression": lambda: LogisticRegression(solver="liblinear", random_state=42),
        "RandomForestClassifier": lambda: RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    }

    dml_tab = st.tabs([
        "Cross-fit DML", "CATE (R/DR-Learner)", "Auto DML", "IV-DML", "DiD-DML",
    ])

    # ---- Tab 1: Cross-fitted DML with inference ----
    with dml_tab[0]:
        st.markdown("K-fold cross-fitted DML with Neyman-orthogonal standard errors and confidence intervals.")
        col_cfg, col_res = st.columns([1, 2])
        with col_cfg:
            st.subheader("Data")
            n_samples = st.slider("Samples", 500, 5000, 2000, 100, key="cf_n")
            n_features = st.slider("Features", 2, 20, 5, key="cf_f")
            true_effect = st.number_input("True effect", value=2.5, step=0.1, key="cf_te")
            binary = st.toggle("Binary treatment", True, key="cf_bin")
            seed = st.number_input("Seed", value=42, step=1, key="cf_seed")
            st.subheader("Settings")
            n_folds = st.slider("K folds", 2, 10, 5, key="cf_k")
            ym = st.selectbox("Outcome model", ["LinearRegression", "Lasso", "RandomForestRegressor", "GradientBoosting"], key="cf_ym")
            if binary:
                tm = st.selectbox("Treatment model", ["LogisticRegression", "RandomForestClassifier"], key="cf_tm")
            else:
                tm = st.selectbox("Treatment model", ["LinearRegression", "Lasso"], key="cf_tmc")
            run = st.button("Run Cross-fit DML", type="primary", use_container_width=True, key="cf_run")
        with col_res:
            if run:
                with st.spinner("Running …"):
                    X, y, T = generate_dml_data(n_samples, n_features, true_effect, binary, int(seed))
                    res = double_ml_crossfit(X, y, T, MODEL_MAP[ym](), MODEL_MAP[tm](),
                                            treatment_is_binary=binary, n_folds=n_folds)
                st.subheader("Results")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("True Effect", f"{true_effect:.4f}")
                m2.metric("θ̂", f"{res.theta:.4f}")
                m3.metric("SE", f"{res.se:.4f}")
                m4.metric("p-value", f"{res.p_value:.4f}")
                st.code(res.summary(), language="text")
                # CI visualisation
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[res.ci_lower, res.ci_upper], y=[0, 0],
                    mode="lines", line=dict(color=C_PRIMARY, width=6), name="95% CI"))
                fig.add_trace(go.Scatter(x=[res.theta], y=[0], mode="markers",
                    marker=dict(color=C_SECONDARY, size=14, symbol="diamond"), name="θ̂"))
                fig.add_vline(x=true_effect, line_dash="dash", line_color=C_SUCCESS, annotation_text="True θ")
                fig.update_layout(height=150, template="plotly_dark", showlegend=True,
                                  yaxis=dict(visible=False), xaxis_title="Causal Effect",
                                  margin=dict(t=30, b=30, l=30, r=30))
                st.plotly_chart(fig, use_container_width=True)
                # Residual scatter
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=res.t_residuals, y=res.y_residuals, mode="markers",
                    marker=dict(color=C_PRIMARY, opacity=0.35, size=3), name="Residuals"))
                xs = np.linspace(res.t_residuals.min(), res.t_residuals.max(), 100)
                fig2.add_trace(go.Scatter(x=xs, y=res.theta*xs, mode="lines",
                    line=dict(color=C_SECONDARY, width=3), name=f"θ = {res.theta:.4f}"))
                fig2.update_layout(title="Y_res vs T_res", height=350, template="plotly_dark",
                                   xaxis_title="T_res", yaxis_title="Y_res", margin=dict(t=40,b=30))
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Configure and click **Run Cross-fit DML**.")

    # ---- Tab 2: CATE ----
    with dml_tab[1]:
        st.markdown("Estimate heterogeneous treatment effects τ(x) = E[Y(1)−Y(0) | X=x].")
        col_cfg, col_res = st.columns([1, 2])
        with col_cfg:
            st.subheader("Data")
            n_s = st.slider("Samples", 500, 5000, 2000, 100, key="cate_n")
            n_f = st.slider("Features", 2, 20, 5, key="cate_f")
            te = st.number_input("True effect", value=2.0, step=0.1, key="cate_te")
            sd = st.number_input("Seed", value=42, step=1, key="cate_seed")
            st.subheader("Method")
            method = st.radio("CATE method", ["R-Learner", "DR-Learner", "X-Learner"], key="cate_m")
            final_model = st.selectbox("Final τ(x) model", ["Ridge", "RandomForestRegressor", "GradientBoosting"], key="cate_fm")
            run_cate = st.button("Run CATE", type="primary", use_container_width=True, key="cate_run")
        with col_res:
            if run_cate:
                with st.spinner("Estimating CATE …"):
                    X, y, T = generate_dml_data(n_s, n_f, te, True, int(sd))
                    cate_m = MODEL_MAP[final_model]()
                    from algorithms.dml.dml_cate import x_learner
                    if method == "R-Learner":
                        cres = r_learner(X, y, T, LinearRegression(),
                                         LogisticRegression(solver='liblinear', random_state=42),
                                         cate_m, n_folds=5)
                    elif method == "DR-Learner":
                        cres = dr_learner(X, y, T, LinearRegression(), LinearRegression(),
                                          LogisticRegression(solver='liblinear', random_state=42),
                                          cate_m, n_folds=5)
                    else:
                        cres = x_learner(X, y, T, LinearRegression(), LinearRegression(),
                                         LogisticRegression(solver='liblinear', random_state=42),
                                         cate_m, MODEL_MAP[final_model](), n_folds=5)
                st.subheader("Results")
                m1, m2, m3 = st.columns(3)
                m1.metric("ATE", f"{cres.ate:.4f}")
                m2.metric("τ(x) std", f"{cres.tau_hat.std():.4f}")
                m3.metric("True Effect", f"{te:.4f}")
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=cres.tau_hat, nbinsx=50,
                    marker_color=C_PRIMARY, opacity=0.7, name="τ̂(x)"))
                fig.add_vline(x=te, line_dash="dash", line_color=C_SECONDARY,
                              annotation_text=f"True θ = {te}")
                fig.update_layout(title=f"{method}: CATE Distribution",
                    xaxis_title="τ̂(x)", yaxis_title="Count",
                    height=380, template="plotly_dark", margin=dict(t=40,b=30))
                st.plotly_chart(fig, use_container_width=True)
                if cres.pseudo_outcomes is not None:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Histogram(x=cres.pseudo_outcomes, nbinsx=50,
                        marker_color=C_WARNING, opacity=0.7, name="Pseudo-outcomes Γ"))
                    fig2.update_layout(title="Doubly-Robust Pseudo-outcomes",
                        xaxis_title="Γ", height=300, template="plotly_dark", margin=dict(t=40,b=30))
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Configure and click **Run CATE**.")

    # ---- Tab 3: Auto DML ----
    with dml_tab[2]:
        st.markdown("Automatically select the best nuisance models via cross-validation, then run cross-fitted DML.")
        col_cfg, col_res = st.columns([1, 2])
        with col_cfg:
            st.subheader("Data")
            n_s = st.slider("Samples", 500, 5000, 2000, 100, key="auto_n")
            n_f = st.slider("Features", 2, 20, 5, key="auto_f")
            te = st.number_input("True effect", value=2.5, step=0.1, key="auto_te")
            binary = st.toggle("Binary treatment", True, key="auto_bin")
            sd = st.number_input("Seed", value=42, step=1, key="auto_seed")
            st.subheader("Settings")
            cv_folds = st.slider("CV folds (selection)", 2, 5, 3, key="auto_cvk")
            dml_folds = st.slider("DML folds", 2, 10, 5, key="auto_dk")
            run_auto = st.button("Run Auto DML", type="primary", use_container_width=True, key="auto_run")
        with col_res:
            if run_auto:
                with st.spinner("Selecting models & running DML …"):
                    X, y, T = generate_dml_data(n_s, n_f, te, binary, int(sd))
                    ares = auto_dml(X, y, T, treatment_is_binary=binary,
                                    cv_folds=cv_folds, dml_folds=dml_folds)
                st.subheader("Results")
                st.code(ares.summary(), language="text")
                m1, m2, m3 = st.columns(3)
                m1.metric("θ̂", f"{ares.dml_result.theta:.4f}")
                m2.metric("SE", f"{ares.dml_result.se:.4f}")
                m3.metric("True θ", f"{te:.4f}")
                # Model comparison bar charts
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Outcome Model CV R²", "Treatment Model CV Score"))
                names_y = list(ares.model_y_scores.keys())
                vals_y = [ares.model_y_scores[n] for n in names_y]
                colors_y = [C_SUCCESS if n == ares.best_model_y_name else C_PRIMARY for n in names_y]
                fig.add_trace(go.Bar(x=names_y, y=vals_y, marker_color=colors_y, name="Outcome"), row=1, col=1)
                names_t = list(ares.model_t_scores.keys())
                vals_t = [ares.model_t_scores[n] for n in names_t]
                colors_t = [C_SUCCESS if n == ares.best_model_t_name else C_SECONDARY for n in names_t]
                fig.add_trace(go.Bar(x=names_t, y=vals_t, marker_color=colors_t, name="Treatment"), row=1, col=2)
                fig.update_layout(height=350, template="plotly_dark", showlegend=False, margin=dict(t=40,b=80))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Configure and click **Run Auto DML**.")

    # ---- Tab 4: IV-DML ----
    with dml_tab[3]:
        st.markdown("DML with instrumental variables for Local Average Treatment Effect (LATE) when unconfoundedness fails.")
        col_cfg, col_res = st.columns([1, 2])
        with col_cfg:
            st.subheader("Data (synthetic IV)")
            n_s = st.slider("Samples", 500, 8000, 3000, 100, key="iv_n")
            late = st.number_input("True LATE", value=3.0, step=0.1, key="iv_late")
            sd = st.number_input("Seed", value=42, step=1, key="iv_seed")
            n_folds = st.slider("K folds", 2, 10, 5, key="iv_k")
            run_iv = st.button("Run IV-DML", type="primary", use_container_width=True, key="iv_run")
        with col_res:
            if run_iv:
                with st.spinner("Running IV-DML …"):
                    np.random.seed(int(sd))
                    X = np.random.randn(n_s, 3)
                    U = np.random.randn(n_s)
                    Z = (np.random.rand(n_s) > 0.5).astype(float)
                    T = (0.5*X[:,0] + 1.5*Z + 0.5*U + np.random.normal(0,0.3,n_s) > 1).astype(float)
                    y = X[:,1] + late*T + U + np.random.normal(0,0.3,n_s)
                    ivres = iv_dml(X, y, T, Z, LinearRegression(),
                                   LogisticRegression(solver='liblinear', random_state=42),
                                   LogisticRegression(solver='liblinear', random_state=42),
                                   n_folds=n_folds)
                st.subheader("Results")
                st.code(ivres.summary(), language="text")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("True LATE", f"{late:.4f}")
                m2.metric("θ̂", f"{ivres.theta:.4f}")
                m3.metric("SE", f"{ivres.se:.4f}")
                m4.metric("1st-stage F", f"{ivres.first_stage_f_stat:.1f}")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[ivres.ci_lower, ivres.ci_upper], y=[0,0],
                    mode="lines", line=dict(color=C_PRIMARY, width=6), name="95% CI"))
                fig.add_trace(go.Scatter(x=[ivres.theta], y=[0], mode="markers",
                    marker=dict(color=C_SECONDARY, size=14, symbol="diamond"), name="θ̂"))
                fig.add_vline(x=late, line_dash="dash", line_color=C_SUCCESS, annotation_text="True LATE")
                fig.update_layout(height=150, template="plotly_dark",
                    yaxis=dict(visible=False), xaxis_title="LATE", margin=dict(t=30,b=30))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Configure and click **Run IV-DML**.")

    # ---- Tab 5: DiD-DML ----
    with dml_tab[4]:
        st.markdown("Difference-in-Differences DML for panel data under conditional parallel trends.")
        col_cfg, col_res = st.columns([1, 2])
        with col_cfg:
            st.subheader("Data (synthetic DiD)")
            n_s = st.slider("Samples", 500, 5000, 2000, 100, key="did_n")
            att = st.number_input("True ATT", value=2.0, step=0.1, key="did_att")
            sd = st.number_input("Seed", value=42, step=1, key="did_seed")
            n_folds = st.slider("K folds", 2, 10, 5, key="did_k")
            run_did = st.button("Run DiD-DML", type="primary", use_container_width=True, key="did_run")
        with col_res:
            if run_did:
                with st.spinner("Running DiD-DML …"):
                    np.random.seed(int(sd))
                    X = np.random.randn(n_s, 3)
                    T = (X[:,0] + np.random.normal(0,0.5,n_s) > 0).astype(float)
                    y_pre = X[:,1] + np.random.normal(0, 0.2, n_s)
                    y_post = y_pre + 0.5 + att*T + np.random.normal(0, 0.2, n_s)
                    dres = did_dml(X, y_pre, y_post, T, LinearRegression(),
                                   LogisticRegression(solver='liblinear', random_state=42),
                                   n_folds=n_folds)
                st.subheader("Results")
                st.code(dres.summary(), language="text")
                m1, m2, m3 = st.columns(3)
                m1.metric("True ATT", f"{att:.4f}")
                m2.metric("θ̂", f"{dres.theta:.4f}")
                m3.metric("SE", f"{dres.se:.4f}")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[dres.ci_lower, dres.ci_upper], y=[0,0],
                    mode="lines", line=dict(color=C_PRIMARY, width=6), name="95% CI"))
                fig.add_trace(go.Scatter(x=[dres.theta], y=[0], mode="markers",
                    marker=dict(color=C_SECONDARY, size=14, symbol="diamond"), name="θ̂"))
                fig.add_vline(x=att, line_dash="dash", line_color=C_SUCCESS, annotation_text="True ATT")
                fig.update_layout(height=150, template="plotly_dark",
                    yaxis=dict(visible=False), xaxis_title="ATT", margin=dict(t=30,b=30))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Configure and click **Run DiD-DML**.")

# ============================= DRCFR ========================================
elif algo == "MIM-DRCFR":
    st.title("MIM-DRCFR")
    st.markdown(
        "Disentangled representations with MMD-based IPM and mutual-information "
        "regularisation for counterfactual regression."
    )
    with st.expander("📖 Algorithm Theory, Architecture & Usage", expanded=False):
        docs_content.show_drcfr_theory()

    col_cfg, col_res = st.columns([1, 2])

    with col_cfg:
        st.subheader("Data Settings")
        n_samples = st.slider("Samples", 500, 5000, 2000, 100, key="dr_n")
        n_features = st.slider("Features", 5, 30, 10, key="dr_f")
        true_ate = st.number_input("True ATE baseline", value=1.5, step=0.1, key="dr_ate")
        seed = st.number_input("Random seed", value=42, step=1, key="dr_seed")

        st.subheader("Architecture")
        latent_zy = st.slider("Latent dim z_y", 4, 64, 20, key="dr_zy")
        latent_zs = st.slider("Latent dim z_s", 4, 64, 20, key="dr_zs")
        alpha = st.number_input("α (IPM weight)", value=1.0, step=0.1, key="dr_a", min_value=0.0)
        beta = st.number_input("β (MI weight)", value=0.1, step=0.1, key="dr_b", min_value=0.0)

        st.subheader("Training")
        epochs = st.slider("Epochs", 10, 300, 100, 10, key="dr_ep")
        lr = st.select_slider("Learning rate", [1e-4, 5e-4, 1e-3, 5e-3], value=1e-3, key="dr_lr")
        batch_size = st.select_slider("Batch size", [32, 64, 128, 256], value=128, key="dr_bs")

        run_drcfr = st.button("Run MIM-DRCFR", type="primary", use_container_width=True)

    with col_res:
        if run_drcfr:
            from sklearn.model_selection import train_test_split
            progress = st.progress(0, text="Generating data …")

            X_np, T_np, Yf_np, Y0_np, Y1_np = generate_drcfr_data(
                n_samples, n_features, true_ate, int(seed)
            )
            (X_tr, X_val, T_tr, T_val,
             Yf_tr, Yf_val, Y0_tr, Y0_val, Y1_tr, Y1_val) = train_test_split(
                X_np, T_np, Yf_np, Y0_np, Y1_np, test_size=0.2, random_state=123
            )
            x_tr_t = torch.tensor(X_tr, dtype=torch.float32)
            x_val_t = torch.tensor(X_val, dtype=torch.float32)
            t_tr_t = torch.tensor(T_tr, dtype=torch.float32)
            t_val_t = torch.tensor(T_val, dtype=torch.float32)
            yf_tr_t = torch.tensor(Yf_tr, dtype=torch.float32).unsqueeze(1)
            yf_val_t = torch.tensor(Yf_val, dtype=torch.float32).unsqueeze(1)

            progress.progress(10, text="Initialising model …")

            model = DRCFRModel(
                input_dim=n_features,
                hidden_dims_phi=[128, 64, 32],
                latent_dim_zy=latent_zy,
                latent_dim_zs=latent_zs,
                hidden_dims_h=[64, 32],
                output_dim=1,
                alpha=alpha,
                beta=beta,
                learning_rate=lr,
                weight_decay=1e-5,
            )

            # Training with captured losses
            train_losses = {"total": [], "R": [], "IPM": [], "MI": []}
            val_losses = {"total": [], "R": [], "IPM": [], "MI": []}

            model.train()
            x_tr_t_d = x_tr_t.to(model.device)
            yf_tr_t_d = yf_tr_t.to(model.device)
            t_tr_t_d = t_tr_t.to(model.device)
            if t_tr_t_d.ndim == 1:
                t_tr_t_d = t_tr_t_d.unsqueeze(1)

            from torch.utils.data import TensorDataset, DataLoader
            train_ds = TensorDataset(x_tr_t_d, yf_tr_t_d, t_tr_t_d)
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            x_val_d = x_val_t.to(model.device)
            yf_val_d = yf_val_t.to(model.device)
            t_val_d = t_val_t.to(model.device)
            if t_val_d.ndim == 1:
                t_val_d = t_val_d.unsqueeze(1)
            val_ds = TensorDataset(x_val_d, yf_val_d, t_val_d)
            val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            for epoch in range(epochs):
                model.train()
                ep_t = ep_r = ep_i = ep_m = 0.0
                for bx, by, bt in train_dl:
                    model.optimizer.zero_grad()
                    h0, h1, zy, zs = model.forward(bx)
                    tl, lr_, li_, lm_ = model.compute_loss(bx, by, bt, h0, h1, zy, zs)
                    tl.backward()
                    model.optimizer.step()
                    ep_t += tl.item(); ep_r += lr_.item(); ep_i += li_.item(); ep_m += lm_.item()
                n_b = len(train_dl)
                train_losses["total"].append(ep_t / n_b)
                train_losses["R"].append(ep_r / n_b)
                train_losses["IPM"].append(ep_i / n_b)
                train_losses["MI"].append(ep_m / n_b)

                model.eval()
                vt = vr = vi = vm = 0.0
                with torch.no_grad():
                    for bx, by, bt in val_dl:
                        h0, h1, zy, zs = model.forward(bx)
                        tl, lr_, li_, lm_ = model.compute_loss(bx, by, bt, h0, h1, zy, zs)
                        vt += tl.item(); vr += lr_.item(); vi += li_.item(); vm += lm_.item()
                vn = len(val_dl)
                val_losses["total"].append(vt / vn)
                val_losses["R"].append(vr / vn)
                val_losses["IPM"].append(vi / vn)
                val_losses["MI"].append(vm / vn)

                pct = int(10 + 80 * (epoch + 1) / epochs)
                progress.progress(pct, text=f"Training epoch {epoch + 1}/{epochs} …")

            progress.progress(95, text="Evaluating …")

            # Evaluation
            ite_pred = model.predict_ite(x_val_t)
            true_ite = torch.tensor(Y1_val - Y0_val, dtype=torch.float32).unsqueeze(1)
            true_ate_val = (Y1_val - Y0_val).mean()
            est_ate = ite_pred.mean().item()
            pehe = torch.sqrt(torch.mean((true_ite.to(model.device) - ite_pred) ** 2)).item()

            progress.progress(100, text="Done!")

            # -- metrics --
            st.subheader("Results")
            m1, m2, m3 = st.columns(3)
            m1.metric("True ATE", f"{true_ate_val:.4f}")
            m2.metric("Estimated ATE", f"{est_ate:.4f}")
            m3.metric("PEHE", f"{pehe:.4f}")

            # -- training curves --
            ep_range = list(range(1, epochs + 1))
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Training Loss", "Validation Loss"))
            for name, color in [("total", C_PRIMARY), ("R", C_SUCCESS), ("IPM", C_WARNING), ("MI", C_SECONDARY)]:
                fig.add_trace(go.Scatter(x=ep_range, y=train_losses[name], mode="lines",
                                         name=name, line=dict(color=color)), row=1, col=1)
                fig.add_trace(go.Scatter(x=ep_range, y=val_losses[name], mode="lines",
                                         name=name, line=dict(color=color), showlegend=False),
                              row=1, col=2)
            fig.update_layout(height=380, template="plotly_dark", margin=dict(t=40, b=30))
            fig.update_xaxes(title_text="Epoch")
            fig.update_yaxes(title_text="Loss")
            st.plotly_chart(fig, use_container_width=True)

            # -- ITE distribution --
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=true_ite.numpy().flatten(), nbinsx=50,
                                         marker_color=C_PRIMARY, opacity=0.6, name="True ITE"))
            fig2.add_trace(go.Histogram(x=ite_pred.cpu().numpy().flatten(), nbinsx=50,
                                         marker_color=C_SECONDARY, opacity=0.6, name="Predicted ITE"))
            fig2.update_layout(barmode="overlay", title="ITE Distribution (Validation Set)",
                               xaxis_title="ITE", yaxis_title="Count",
                               height=380, template="plotly_dark",
                               margin=dict(t=40, b=30))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Configure parameters on the left and click **Run MIM-DRCFR**.")

# ============================= SRCVAE =======================================
elif algo == "SRCVAE":
    st.title("SRCVAE – Causal Effect VAE")
    st.markdown(
        "Variational autoencoder with latent confounders (**u**) and "
        "instrumental variables (**v**) for causal effect estimation under "
        "unobserved confounding."
    )
    with st.expander("📖 Algorithm Theory, Architecture & Usage", expanded=False):
        docs_content.show_srcvae_theory()

    col_cfg, col_res = st.columns([1, 2])

    with col_cfg:
        st.subheader("Data Settings")
        n_samples = st.slider("Samples", 500, 5000, 2000, 100, key="sv_n")
        x_dim = st.slider("X dimension", 3, 20, 5, key="sv_xd")
        true_ate = st.number_input("True ATE", value=2.0, step=0.1, key="sv_ate")
        seed = st.number_input("Random seed", value=42, step=1, key="sv_seed")

        st.subheader("Latent Dimensions")
        u_dim = st.slider("u dim (confounders)", 2, 20, 5, key="sv_ud")
        v_dim = st.slider("v dim (instruments)", 2, 20, 5, key="sv_vd")

        st.subheader("Loss Weights")
        c1, c2 = st.columns(2)
        beta_u = c1.number_input("β_u (KL u)", value=0.1, step=0.05, min_value=0.0, key="sv_bu")
        beta_v = c2.number_input("β_v (KL v)", value=0.1, step=0.05, min_value=0.0, key="sv_bv")

        st.subheader("Training")
        epochs = st.slider("Epochs", 10, 300, 50, 10, key="sv_ep")
        lr = st.select_slider("Learning rate", [1e-4, 5e-4, 1e-3, 5e-3], value=1e-3, key="sv_lr")
        batch_size = st.select_slider("Batch size", [32, 64, 128, 256], value=128, key="sv_bs")
        n_samples_u = st.slider("MC samples for ITE", 10, 200, 100, 10, key="sv_nsu")

        run_srcvae = st.button("Run SRCVAE", type="primary", use_container_width=True)

    with col_res:
        if run_srcvae:
            from sklearn.model_selection import train_test_split
            progress = st.progress(0, text="Generating data …")

            X_np, T_np, Yf_np, Y0_np, Y1_np = generate_srcvae_data(
                n_samples, x_dim, true_ate, int(seed)
            )
            (X_tr, X_val, T_tr, T_val,
             Yf_tr, Yf_val, Y0_tr, Y0_val, Y1_tr, Y1_val) = train_test_split(
                X_np, T_np, Yf_np, Y0_np, Y1_np, test_size=0.2, random_state=123
            )

            x_tr_t = torch.tensor(X_tr, dtype=torch.float32)
            t_tr_t = torch.tensor(T_tr, dtype=torch.float32)
            yf_tr_t = torch.tensor(Yf_tr, dtype=torch.float32)
            x_val_t = torch.tensor(X_val, dtype=torch.float32)
            t_val_t = torch.tensor(T_val, dtype=torch.float32)
            yf_val_t = torch.tensor(Yf_val, dtype=torch.float32)

            progress.progress(10, text="Initialising model …")

            model = SRCVAEModel(
                x_dim=x_dim, t_dim=1, y_dim=1,
                u_dim=u_dim, v_dim=v_dim,
                hidden_dims_encoder_u=[64, 32],
                hidden_dims_encoder_v=[64, 32],
                hidden_dims_decoder_x=[32, 64],
                hidden_dims_decoder_t=[32, 64],
                hidden_dims_decoder_y=[32, 64],
                hidden_dims_aux_qtx=[32],
                hidden_dims_aux_qyxt=[32],
                alpha_x=1.0, alpha_t=1.0, alpha_y=1.0,
                beta_u=beta_u, beta_v=beta_v,
                gamma_t=1.0, gamma_y=1.0,
                learning_rate=lr, weight_decay=1e-5,
            )

            # Custom training loop to capture losses
            from torch.utils.data import TensorDataset, DataLoader

            model.train()
            _t_tr = t_tr_t.to(model.device)
            _yf_tr = yf_tr_t.to(model.device)
            _x_tr = x_tr_t.to(model.device)
            if _t_tr.ndim == 1: _t_tr = _t_tr.unsqueeze(1)
            if _yf_tr.ndim == 1: _yf_tr = _yf_tr.unsqueeze(1)
            train_ds = TensorDataset(_x_tr, _t_tr, _yf_tr)
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            _t_val = t_val_t.to(model.device)
            _yf_val = yf_val_t.to(model.device)
            _x_val = x_val_t.to(model.device)
            if _t_val.ndim == 1: _t_val = _t_val.unsqueeze(1)
            if _yf_val.ndim == 1: _yf_val = _yf_val.unsqueeze(1)
            val_ds = TensorDataset(_x_val, _t_val, _yf_val)
            val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            loss_keys = ["total_loss", "loss_kl_u", "loss_kl_v",
                         "loss_recon_x", "loss_recon_t", "loss_recon_y",
                         "loss_aux_qt", "loss_aux_qy"]
            history_train = {k: [] for k in loss_keys}
            history_val = {k: [] for k in loss_keys}

            for epoch in range(epochs):
                model.train()
                ep_acc = {k: 0.0 for k in loss_keys}
                for bx, bt, by in train_dl:
                    model.optimizer.zero_grad()
                    outputs = model.forward(bx, bt, by)
                    u_m, u_lv, v_m, v_lv, xr, tr, yr, ta, ya, _, _, xlv, ylv = outputs
                    tl, comps = model.compute_loss(bx, bt, by,
                                                   u_m, u_lv, v_m, v_lv,
                                                   xr, tr, yr, ta, ya, xlv, ylv)
                    tl.backward()
                    model.optimizer.step()
                    for k in loss_keys:
                        ep_acc[k] += comps[k]
                nb = len(train_dl)
                for k in loss_keys:
                    history_train[k].append(ep_acc[k] / nb)

                model.eval()
                vep = {k: 0.0 for k in loss_keys}
                with torch.no_grad():
                    for bx, bt, by in val_dl:
                        outputs = model.forward(bx, bt, by)
                        u_m, u_lv, v_m, v_lv, xr, tr, yr, ta, ya, _, _, xlv, ylv = outputs
                        _, comps = model.compute_loss(bx, bt, by,
                                                      u_m, u_lv, v_m, v_lv,
                                                      xr, tr, yr, ta, ya, xlv, ylv)
                        for k in loss_keys:
                            vep[k] += comps[k]
                vnb = len(val_dl)
                for k in loss_keys:
                    history_val[k].append(vep[k] / vnb)

                pct = int(10 + 80 * (epoch + 1) / epochs)
                progress.progress(pct, text=f"Training epoch {epoch + 1}/{epochs} …")

            progress.progress(95, text="Estimating causal effects …")

            ite_pred, y0_pred, y1_pred = model.estimate_causal_effect(x_val_t, n_samples_u=n_samples_u)
            true_ate_val = (Y1_val - Y0_val).mean()
            est_ate = ite_pred.mean().item()
            true_ite_t = torch.tensor(Y1_val - Y0_val, dtype=torch.float32).unsqueeze(1)
            pehe = torch.sqrt(torch.mean((true_ite_t.to(model.device) - ite_pred) ** 2)).item()

            progress.progress(100, text="Done!")

            # -- metrics --
            st.subheader("Results")
            m1, m2, m3 = st.columns(3)
            m1.metric("True ATE", f"{true_ate_val:.4f}")
            m2.metric("Estimated ATE", f"{est_ate:.4f}")
            m3.metric("PEHE", f"{pehe:.4f}")

            # -- training curves --
            ep_range = list(range(1, epochs + 1))
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Training Loss", "Validation Loss"))
            colors_map = {
                "total_loss": C_PRIMARY, "loss_kl_u": C_SECONDARY, "loss_kl_v": C_WARNING,
                "loss_recon_x": C_SUCCESS, "loss_recon_y": "#8338ec",
                "loss_recon_t": "#fb5607", "loss_aux_qt": "#3a86a7", "loss_aux_qy": "#80b918",
            }
            for k in loss_keys:
                fig.add_trace(go.Scatter(x=ep_range, y=history_train[k], mode="lines",
                                         name=k.replace("loss_", ""),
                                         line=dict(color=colors_map.get(k, C_PRIMARY))),
                              row=1, col=1)
                fig.add_trace(go.Scatter(x=ep_range, y=history_val[k], mode="lines",
                                         name=k.replace("loss_", ""),
                                         line=dict(color=colors_map.get(k, C_PRIMARY)),
                                         showlegend=False),
                              row=1, col=2)
            fig.update_layout(height=400, template="plotly_dark", margin=dict(t=40, b=30))
            fig.update_xaxes(title_text="Epoch")
            fig.update_yaxes(title_text="Loss")
            st.plotly_chart(fig, use_container_width=True)

            # -- ITE distribution --
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=true_ite_t.numpy().flatten(), nbinsx=50,
                                         marker_color=C_PRIMARY, opacity=0.6, name="True ITE"))
            fig2.add_trace(go.Histogram(x=ite_pred.cpu().numpy().flatten(), nbinsx=50,
                                         marker_color=C_SECONDARY, opacity=0.6, name="Predicted ITE"))
            fig2.update_layout(barmode="overlay", title="ITE Distribution (Validation Set)",
                               xaxis_title="ITE", yaxis_title="Count",
                               height=380, template="plotly_dark",
                               margin=dict(t=40, b=30))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Configure parameters on the left and click **Run SRCVAE**.")

# ============================= TARNet ========================================
elif algo == "TARNet":
    st.title("TARNet — Treatment-Agnostic Representation Network")
    st.markdown(
        "Shared representation followed by two treatment-specific outcome heads. "
        "A simpler baseline without distributional balancing (no IPM/MMD)."
    )
    with st.expander("📖 Algorithm Theory, Architecture & Usage", expanded=False):
        docs_content.show_tarnet_theory()

    col_cfg, col_res = st.columns([1, 2])
    with col_cfg:
        st.subheader("Data Settings")
        n_samples = st.slider("Samples", 500, 5000, 2000, 100, key="tar_n")
        n_features = st.slider("Features", 5, 30, 10, key="tar_f")
        true_ate = st.number_input("True ATE baseline", value=1.5, step=0.1, key="tar_ate")
        seed = st.number_input("Random seed", value=42, step=1, key="tar_seed")
        st.subheader("Training")
        epochs = st.slider("Epochs", 10, 300, 100, 10, key="tar_ep")
        lr = st.select_slider("Learning rate", [1e-4, 5e-4, 1e-3, 5e-3], value=1e-3, key="tar_lr")
        dropout = st.slider("Dropout", 0.0, 0.5, 0.1, 0.05, key="tar_do")
        run_tar = st.button("Run TARNet", type="primary", use_container_width=True, key="tar_run")

    with col_res:
        if run_tar:
            from sklearn.model_selection import train_test_split
            progress = st.progress(0, text="Generating data …")
            X_np, T_np, Yf_np, Y0_np, Y1_np = generate_drcfr_data(n_samples, n_features, true_ate, int(seed))
            (X_tr, X_val, T_tr, T_val, Yf_tr, Yf_val,
             Y0_tr, Y0_val, Y1_tr, Y1_val) = train_test_split(
                X_np, T_np, Yf_np, Y0_np, Y1_np, test_size=0.2, random_state=123)

            x_tr_t = torch.tensor(X_tr, dtype=torch.float32)
            t_tr_t = torch.tensor(T_tr, dtype=torch.float32)
            yf_tr_t = torch.tensor(Yf_tr, dtype=torch.float32).unsqueeze(1)
            x_val_t = torch.tensor(X_val, dtype=torch.float32)
            t_val_t = torch.tensor(T_val, dtype=torch.float32)
            yf_val_t = torch.tensor(Yf_val, dtype=torch.float32).unsqueeze(1)

            progress.progress(10, text="Training TARNet …")
            model = TARNetModel(n_features, [128, 64], [32], dropout=dropout, learning_rate=lr)
            history = model.fit(x_tr_t, yf_tr_t, t_tr_t, num_epochs=epochs,
                                batch_size=128, x_val=x_val_t, y_val=yf_val_t, t_val=t_val_t,
                                print_every_epochs=9999, patience=15)
            progress.progress(90, text="Evaluating …")

            ite_pred = model.predict_ite(x_val_t).cpu().numpy().flatten()
            true_ite = Y1_val - Y0_val
            true_ate_val = true_ite.mean()
            est_ate = ite_pred.mean()
            pehe_val = np.sqrt(np.mean((true_ite - ite_pred) ** 2))
            progress.progress(100, text="Done!")

            st.subheader("Results")
            m1, m2, m3 = st.columns(3)
            m1.metric("True ATE", f"{true_ate_val:.4f}")
            m2.metric("Estimated ATE", f"{est_ate:.4f}")
            m3.metric("PEHE", f"{pehe_val:.4f}")

            ep_range = list(range(1, len(history['train_loss']) + 1))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ep_range, y=history['train_loss'], mode="lines",
                                     name="Train", line=dict(color=C_PRIMARY)))
            if history['val_loss']:
                fig.add_trace(go.Scatter(x=ep_range[:len(history['val_loss'])],
                                         y=history['val_loss'], mode="lines",
                                         name="Val", line=dict(color=C_SECONDARY)))
            fig.update_layout(title="Training Loss", xaxis_title="Epoch", yaxis_title="Loss",
                              height=350, template="plotly_dark", margin=dict(t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=true_ite, nbinsx=50, marker_color=C_PRIMARY, opacity=0.6, name="True ITE"))
            fig2.add_trace(go.Histogram(x=ite_pred, nbinsx=50, marker_color=C_SECONDARY, opacity=0.6, name="Predicted ITE"))
            fig2.update_layout(barmode="overlay", title="ITE Distribution",
                               xaxis_title="ITE", height=350, template="plotly_dark", margin=dict(t=40, b=30))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Configure parameters and click **Run TARNet**.")

# ============================= DragonNet =====================================
elif algo == "DragonNet":
    st.title("DragonNet — Adapted Neural Network for Treatment Effects")
    st.markdown(
        "TARNet extended with a **propensity score head** ε(x) and "
        "**targeted regularization** for doubly-robust-style bias correction."
    )
    with st.expander("📖 Algorithm Theory, Architecture & Usage", expanded=False):
        docs_content.show_dragonnet_theory()

    col_cfg, col_res = st.columns([1, 2])
    with col_cfg:
        st.subheader("Data Settings")
        n_samples = st.slider("Samples", 500, 5000, 2000, 100, key="drg_n")
        n_features = st.slider("Features", 5, 30, 10, key="drg_f")
        true_ate = st.number_input("True ATE baseline", value=1.5, step=0.1, key="drg_ate")
        seed = st.number_input("Random seed", value=42, step=1, key="drg_seed")
        st.subheader("DragonNet Settings")
        alpha_prop = st.number_input("α (propensity weight)", value=1.0, step=0.1, min_value=0.0, key="drg_a")
        beta_targ = st.number_input("β (targeted reg weight)", value=1.0, step=0.1, min_value=0.0, key="drg_b")
        st.subheader("Training")
        epochs = st.slider("Epochs", 10, 300, 100, 10, key="drg_ep")
        lr = st.select_slider("Learning rate", [1e-4, 5e-4, 1e-3, 5e-3], value=1e-3, key="drg_lr")
        dropout = st.slider("Dropout", 0.0, 0.5, 0.1, 0.05, key="drg_do")
        run_drg = st.button("Run DragonNet", type="primary", use_container_width=True, key="drg_run")

    with col_res:
        if run_drg:
            from sklearn.model_selection import train_test_split
            progress = st.progress(0, text="Generating data …")
            X_np, T_np, Yf_np, Y0_np, Y1_np = generate_drcfr_data(n_samples, n_features, true_ate, int(seed))
            (X_tr, X_val, T_tr, T_val, Yf_tr, Yf_val,
             Y0_tr, Y0_val, Y1_tr, Y1_val) = train_test_split(
                X_np, T_np, Yf_np, Y0_np, Y1_np, test_size=0.2, random_state=123)

            x_tr_t = torch.tensor(X_tr, dtype=torch.float32)
            t_tr_t = torch.tensor(T_tr, dtype=torch.float32).unsqueeze(1)
            yf_tr_t = torch.tensor(Yf_tr, dtype=torch.float32).unsqueeze(1)
            x_val_t = torch.tensor(X_val, dtype=torch.float32)
            t_val_t = torch.tensor(T_val, dtype=torch.float32).unsqueeze(1)
            yf_val_t = torch.tensor(Yf_val, dtype=torch.float32).unsqueeze(1)

            progress.progress(10, text="Training DragonNet …")
            model = DragonNetModel(n_features, [128, 64], [32],
                                   alpha=alpha_prop, beta=beta_targ,
                                   dropout=dropout, learning_rate=lr)
            history = model.fit(x_tr_t, yf_tr_t, t_tr_t, num_epochs=epochs,
                                batch_size=128, x_val=x_val_t, y_val=yf_val_t, t_val=t_val_t,
                                print_every_epochs=9999, patience=15)
            progress.progress(90, text="Evaluating …")

            ite_pred = model.predict_ite(x_val_t).cpu().numpy().flatten()
            prop_pred = model.predict_propensity(x_val_t).cpu().numpy().flatten()
            true_ite = Y1_val - Y0_val
            true_ate_val = true_ite.mean()
            est_ate = ite_pred.mean()
            pehe_val = np.sqrt(np.mean((true_ite - ite_pred) ** 2))
            progress.progress(100, text="Done!")

            st.subheader("Results")
            m1, m2, m3 = st.columns(3)
            m1.metric("True ATE", f"{true_ate_val:.4f}")
            m2.metric("Estimated ATE", f"{est_ate:.4f}")
            m3.metric("PEHE", f"{pehe_val:.4f}")

            ep_range = list(range(1, len(history['train_loss']) + 1))
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Total Loss", "Component Losses"))
            fig.add_trace(go.Scatter(x=ep_range, y=history['train_loss'], name="Train",
                                     line=dict(color=C_PRIMARY)), row=1, col=1)
            if history['val_loss']:
                fig.add_trace(go.Scatter(x=ep_range[:len(history['val_loss'])],
                                         y=history['val_loss'], name="Val",
                                         line=dict(color=C_SECONDARY)), row=1, col=1)
            fig.add_trace(go.Scatter(x=ep_range, y=history['train_outcome'], name="Outcome",
                                     line=dict(color=C_SUCCESS)), row=1, col=2)
            fig.add_trace(go.Scatter(x=ep_range, y=history['train_propensity'], name="Propensity",
                                     line=dict(color=C_WARNING)), row=1, col=2)
            fig.add_trace(go.Scatter(x=ep_range, y=history['train_targeted'], name="Targeted",
                                     line=dict(color=C_SECONDARY)), row=1, col=2)
            fig.update_layout(height=350, template="plotly_dark", margin=dict(t=40, b=30))
            fig.update_xaxes(title_text="Epoch")
            st.plotly_chart(fig, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(x=true_ite, nbinsx=50, marker_color=C_PRIMARY, opacity=0.6, name="True ITE"))
                fig2.add_trace(go.Histogram(x=ite_pred, nbinsx=50, marker_color=C_SECONDARY, opacity=0.6, name="Predicted ITE"))
                fig2.update_layout(barmode="overlay", title="ITE Distribution",
                                   xaxis_title="ITE", height=350, template="plotly_dark", margin=dict(t=40, b=30))
                st.plotly_chart(fig2, use_container_width=True)
            with col_b:
                fig3 = go.Figure()
                fig3.add_trace(go.Histogram(x=prop_pred, nbinsx=40, marker_color=C_SUCCESS, opacity=0.8, name="ε̂(x)"))
                fig3.update_layout(title="Predicted Propensity P(T=1|X)",
                                   xaxis_title="ε̂(x)", height=350, template="plotly_dark", margin=dict(t=40, b=30))
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Configure parameters and click **Run DragonNet**.")

# ============================= GANITE ========================================
elif algo == "GANITE":
    st.title("GANITE — GAN for Individualised Treatment Effects")
    st.markdown(
        "Two-stage GAN: **Stage 1** trains a counterfactual generator + discriminator, "
        "**Stage 2** trains an ITE predictor on the generator's pseudo-outcomes."
    )
    with st.expander("📖 Algorithm Theory, Architecture & Usage", expanded=False):
        docs_content.show_ganite_theory()

    col_cfg, col_res = st.columns([1, 2])
    with col_cfg:
        st.subheader("Data Settings")
        n_samples = st.slider("Samples", 500, 5000, 2000, 100, key="gan_n")
        n_features = st.slider("Features", 5, 30, 10, key="gan_f")
        true_ate = st.number_input("True ATE baseline", value=1.5, step=0.1, key="gan_ate")
        seed = st.number_input("Random seed", value=42, step=1, key="gan_seed")
        st.subheader("GANITE Settings")
        noise_dim = st.slider("Noise dim (z)", 2, 32, 8, key="gan_z")
        st.subheader("Training")
        epochs = st.slider("Total epochs (split S1/S2)", 10, 200, 60, 10, key="gan_ep")
        lr = st.select_slider("Learning rate", [1e-4, 5e-4, 1e-3, 5e-3], value=1e-3, key="gan_lr")
        run_gan = st.button("Run GANITE", type="primary", use_container_width=True, key="gan_run")

    with col_res:
        if run_gan:
            from sklearn.model_selection import train_test_split
            progress = st.progress(0, text="Generating data …")
            X_np, T_np, Yf_np, Y0_np, Y1_np = generate_drcfr_data(n_samples, n_features, true_ate, int(seed))
            (X_tr, X_val, T_tr, T_val, Yf_tr, Yf_val,
             Y0_tr, Y0_val, Y1_tr, Y1_val) = train_test_split(
                X_np, T_np, Yf_np, Y0_np, Y1_np, test_size=0.2, random_state=123)

            x_tr_t = torch.tensor(X_tr, dtype=torch.float32)
            t_tr_t = torch.tensor(T_tr, dtype=torch.float32).unsqueeze(1)
            yf_tr_t = torch.tensor(Yf_tr, dtype=torch.float32).unsqueeze(1)
            x_val_t = torch.tensor(X_val, dtype=torch.float32)

            progress.progress(10, text="Training GANITE (Stage 1 + 2) …")
            model = GANITEModel(n_features, hidden_dims=[64, 32], noise_dim=noise_dim, learning_rate=lr)
            history = model.fit(x_tr_t, yf_tr_t, t_tr_t, num_epochs=epochs,
                                batch_size=128, print_every_epochs=9999)
            progress.progress(90, text="Evaluating …")

            ite_pred = model.predict_ite(x_val_t).cpu().numpy().flatten()
            true_ite = Y1_val - Y0_val
            true_ate_val = true_ite.mean()
            est_ate = ite_pred.mean()
            pehe_val = np.sqrt(np.mean((true_ite - ite_pred) ** 2))
            progress.progress(100, text="Done!")

            st.subheader("Results")
            m1, m2, m3 = st.columns(3)
            m1.metric("True ATE", f"{true_ate_val:.4f}")
            m2.metric("Estimated ATE", f"{est_ate:.4f}")
            m3.metric("PEHE", f"{pehe_val:.4f}")

            fig = make_subplots(rows=1, cols=2, subplot_titles=("Stage 1: G & D Loss", "Stage 2: ITE Predictor Loss"))
            if history['g_loss']:
                s1_range = list(range(1, len(history['g_loss']) + 1))
                fig.add_trace(go.Scatter(x=s1_range, y=history['g_loss'], name="Generator",
                                         line=dict(color=C_PRIMARY)), row=1, col=1)
                fig.add_trace(go.Scatter(x=s1_range, y=history['d_loss'], name="Discriminator",
                                         line=dict(color=C_SECONDARY)), row=1, col=1)
            if history['i_loss']:
                s2_range = list(range(1, len(history['i_loss']) + 1))
                fig.add_trace(go.Scatter(x=s2_range, y=history['i_loss'], name="ITE Predictor",
                                         line=dict(color=C_SUCCESS)), row=1, col=2)
            fig.update_layout(height=350, template="plotly_dark", margin=dict(t=40, b=30))
            fig.update_xaxes(title_text="Epoch")
            fig.update_yaxes(title_text="Loss")
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=true_ite, nbinsx=50, marker_color=C_PRIMARY, opacity=0.6, name="True ITE"))
            fig2.add_trace(go.Histogram(x=ite_pred, nbinsx=50, marker_color=C_SECONDARY, opacity=0.6, name="Predicted ITE"))
            fig2.update_layout(barmode="overlay", title="ITE Distribution",
                               xaxis_title="ITE", height=350, template="plotly_dark", margin=dict(t=40, b=30))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Configure parameters and click **Run GANITE**.")

# ============================= IHDP BENCHMARK ================================
elif algo == "IHDP Benchmark":
    st.title("IHDP Benchmark — Algorithm Comparison")
    st.markdown(
        "Run all three algorithms on the IHDP semi-synthetic benchmark and "
        "compare ATE error, PEHE, and ATT error side by side."
    )

    from data.ihdp import generate_ihdp
    from evaluation.metrics import evaluation_report
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

    col_cfg, col_res = st.columns([1, 3])
    with col_cfg:
        st.subheader("Settings")
        n_ihdp = st.slider("IHDP samples", 200, 2000, 747, 50, key="bench_n")
        n_feat = st.slider("Features", 10, 30, 25, key="bench_f")
        seed = st.number_input("Seed", value=42, step=1, key="bench_seed")
        drcfr_epochs = st.slider("DRCFR epochs", 20, 200, 80, 10, key="bench_dr_ep")
        srcvae_epochs = st.slider("SRCVAE epochs", 20, 200, 50, 10, key="bench_sv_ep")
        run_bench = st.button("Run Benchmark", type="primary", use_container_width=True, key="bench_run")

    with col_res:
        if run_bench:
            progress = st.progress(0, text="Generating IHDP data …")
            ds = generate_ihdp(n_samples=n_ihdp, n_features=n_feat, seed=int(seed))
            train, test = ds.train_test_split(test_size=0.2, random_state=int(seed))

            results = {}

            # --- DML ---
            progress.progress(10, text="Running DML …")
            from algorithms.dml.dml_cate import dr_learner
            cres = dr_learner(
                train.X, train.Y, train.T,
                ml_model_y0=LinearRegression(), ml_model_y1=LinearRegression(),
                ml_model_t=LogisticRegression(solver='liblinear', random_state=42),
                cate_model=Ridge(alpha=1.0), n_folds=3,
            )
            dml_ite_test = cres.predict(test.X)
            dml_report = evaluation_report(test.ITE, dml_ite_test, test.T)
            results["DML (DR-Learner)"] = dml_report

            # --- DRCFR ---
            progress.progress(30, text="Training DRCFR …")
            import torch
            x_tr_t = torch.tensor(train.X, dtype=torch.float32)
            t_tr_t = torch.tensor(train.T, dtype=torch.float32)
            yf_tr_t = torch.tensor(train.Y, dtype=torch.float32).unsqueeze(1)
            x_te_t = torch.tensor(test.X, dtype=torch.float32)

            drcfr = DRCFRModel(
                input_dim=n_feat, hidden_dims_phi=[64, 32], latent_dim_zy=16,
                latent_dim_zs=16, hidden_dims_h=[32], output_dim=1,
                alpha=1.0, beta=0.1, learning_rate=1e-3, dropout=0.1,
            )
            drcfr.fit(x_tr_t, yf_tr_t, t_tr_t, num_epochs=drcfr_epochs,
                      batch_size=min(128, n_ihdp), print_every_epochs=9999)
            drcfr_ite = drcfr.predict_ite(x_te_t).cpu().numpy().flatten()
            drcfr_report = evaluation_report(test.ITE, drcfr_ite, test.T)
            results["MIM-DRCFR"] = drcfr_report

            # --- SRCVAE ---
            progress.progress(60, text="Training SRCVAE …")
            srcvae = SRCVAEModel(
                x_dim=n_feat, t_dim=1, y_dim=1, u_dim=5, v_dim=5,
                hidden_dims_encoder_u=[64, 32], hidden_dims_encoder_v=[64, 32],
                hidden_dims_decoder_x=[32, 64], hidden_dims_decoder_t=[32, 64],
                hidden_dims_decoder_y=[32, 64], hidden_dims_aux_qtx=[32],
                hidden_dims_aux_qyxt=[32],
                alpha_x=1.0, alpha_t=1.0, alpha_y=1.0,
                beta_u=0.1, beta_v=0.1, gamma_t=1.0, gamma_y=1.0,
                learning_rate=1e-3,
            )
            srcvae.fit(
                torch.tensor(train.X, dtype=torch.float32),
                torch.tensor(train.T, dtype=torch.float32),
                torch.tensor(train.Y, dtype=torch.float32),
                num_epochs=srcvae_epochs, batch_size=min(128, n_ihdp),
                print_every_epochs=9999, kl_warmup_epochs=10,
            )
            ite_sv, _, _ = srcvae.estimate_causal_effect(x_te_t, n_samples_u=50)
            srcvae_ite = ite_sv.cpu().numpy().flatten()
            srcvae_report = evaluation_report(test.ITE, srcvae_ite, test.T)
            results["SRCVAE"] = srcvae_report

            # --- DragonNet ---
            progress.progress(80, text="Training DragonNet …")
            dragon = DragonNetModel(
                input_dim=n_feat, hidden_dims_shared=[64, 32], hidden_dims_head=[32],
                output_dim=1, alpha=1.0, beta=1.0, learning_rate=1e-3, dropout=0.1,
            )
            dragon.fit(
                x_tr_t, yf_tr_t, t_tr_t,
                num_epochs=drcfr_epochs, batch_size=min(128, n_ihdp),
                print_every_epochs=9999,
            )
            dragon_ite = dragon.predict_ite(x_te_t).cpu().numpy().flatten()
            dragon_report = evaluation_report(test.ITE, dragon_ite, test.T)
            results["DragonNet"] = dragon_report

            progress.progress(100, text="Done!")

            # --- Display results ---
            st.subheader("Comparison Results")

            # Metrics table
            import pandas as pd
            rows = []
            for name, r in results.items():
                rows.append({
                    "Algorithm": name,
                    "ATE Error": f"{r.ate_abs_error:.4f}",
                    "PEHE": f"{r.pehe:.4f}",
                    "ATT Error": f"{r.att_abs_error:.4f}" if r.att_abs_error is not None else "N/A",
                    "Policy Risk": f"{r.policy_risk:.4f}" if r.policy_risk is not None else "N/A",
                })
            st.dataframe(pd.DataFrame(rows).set_index("Algorithm"), use_container_width=True)

            # Bar chart comparison
            algo_names = list(results.keys())
            fig = make_subplots(rows=1, cols=3,
                subplot_titles=("ATE Error (↓)", "PEHE (↓)", "ATT Error (↓)"))
            colors = [C_PRIMARY, C_SECONDARY, C_SUCCESS]
            for i, metric in enumerate(["ate_abs_error", "pehe", "att_abs_error"]):
                vals = [getattr(results[n], metric, 0) or 0 for n in algo_names]
                fig.add_trace(go.Bar(x=algo_names, y=vals,
                    marker_color=colors, name=metric), row=1, col=i+1)
            fig.update_layout(height=350, template="plotly_dark", showlegend=False,
                              margin=dict(t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

            # ITE distribution overlay
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=test.ITE, nbinsx=40,
                marker_color="white", opacity=0.3, name="True ITE"))
            ite_map = {
                "DML (DR-Learner)": dml_ite_test,
                "MIM-DRCFR": drcfr_ite,
                "SRCVAE": srcvae_ite,
                "DragonNet": dragon_ite,
            }
            colors = [C_PRIMARY, C_SECONDARY, C_SUCCESS, C_WARNING]
            for name, color in zip(algo_names, colors):
                ite = ite_map.get(name, dml_ite_test)
                fig2.add_trace(go.Histogram(x=ite, nbinsx=40,
                    marker_color=color, opacity=0.5, name=name))
            fig2.update_layout(barmode="overlay", title="ITE Distributions (Test Set)",
                xaxis_title="ITE", yaxis_title="Count",
                height=400, template="plotly_dark", margin=dict(t=40, b=30))
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("---")
            st.markdown(f"**IHDP dataset**: {ds.n_samples} samples, "
                        f"{ds.n_features} features | True ATE = {ds.ATE:.4f}")
        else:
            st.info("Configure settings and click **Run Benchmark**.")
