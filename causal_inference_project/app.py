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

from algorithms.dml.dml_core import double_ml
from algorithms.drcfr.drcfr_model import DRCFRModel
from algorithms.srcvae.srcvae_model import SRCVAEModel

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
    ["Overview", "DML", "MIM-DRCFR", "SRCVAE"],
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
    st.markdown(
        "Estimate causal effects by partialling out confounders with two "
        "ML models and regressing the residuals."
    )

    col_cfg, col_res = st.columns([1, 2])

    with col_cfg:
        st.subheader("Data Settings")
        n_samples = st.slider("Samples", 200, 5000, 2000, 100, key="dml_n")
        n_features = st.slider("Features (confounders)", 2, 20, 5, key="dml_f")
        true_effect = st.number_input("True causal effect", value=2.5, step=0.1, key="dml_te")
        binary = st.toggle("Binary treatment", value=True, key="dml_bin")
        seed = st.number_input("Random seed", value=42, step=1, key="dml_seed")

        st.subheader("Model Settings")
        from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        outcome_model_name = st.selectbox(
            "Outcome model (E[Y|X])",
            ["LinearRegression", "Lasso", "RandomForestRegressor"],
            key="dml_ym",
        )
        if binary:
            treatment_model_name = st.selectbox(
                "Treatment model (E[T|X])",
                ["LogisticRegression", "RandomForestClassifier"],
                key="dml_tm",
            )
        else:
            treatment_model_name = st.selectbox(
                "Treatment model (E[T|X])",
                ["LinearRegression", "Lasso"],
                key="dml_tm_c",
            )

        run_dml = st.button("Run DML", type="primary", use_container_width=True)

    with col_res:
        if run_dml:
            with st.spinner("Running DML …"):
                X, y, T = generate_dml_data(n_samples, n_features, true_effect, binary, int(seed))

                model_map = {
                    "LinearRegression": LinearRegression(),
                    "Lasso": Lasso(alpha=0.1),
                    "RandomForestRegressor": RandomForestRegressor(random_state=42),
                    "LogisticRegression": LogisticRegression(solver="liblinear", random_state=42),
                    "RandomForestClassifier": RandomForestClassifier(random_state=42),
                }
                model_y = model_map[outcome_model_name]
                model_t = model_map[treatment_model_name]

                est = double_ml(X, y, T, model_y, model_t, treatment_is_binary=binary)

            # -- metrics --
            st.subheader("Results")
            m1, m2, m3 = st.columns(3)
            m1.metric("True Effect", f"{true_effect:.4f}")
            m2.metric("Estimated Effect", f"{est:.4f}")
            m3.metric("Absolute Error", f"{abs(est - true_effect):.4f}")

            # -- residual plots --
            model_y2 = model_map[outcome_model_name].__class__(**model_map[outcome_model_name].get_params())
            model_t2 = model_map[treatment_model_name].__class__(**model_map[treatment_model_name].get_params())
            model_y2.fit(X, y)
            y_res = y - model_y2.predict(X)
            model_t2.fit(X, T)
            if binary and hasattr(model_t2, "predict_proba"):
                t_hat = model_t2.predict_proba(X)[:, 1]
            else:
                t_hat = model_t2.predict(X)
            t_res = T - t_hat

            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("Outcome Residuals (Y_res)", "Treatment Residuals (T_res)"))
            fig.add_trace(go.Histogram(x=y_res, nbinsx=50, marker_color=C_PRIMARY, opacity=0.8, name="Y_res"), row=1, col=1)
            fig.add_trace(go.Histogram(x=t_res, nbinsx=50, marker_color=C_SECONDARY, opacity=0.8, name="T_res"), row=1, col=2)
            fig.update_layout(height=350, showlegend=False,
                              template="plotly_dark", margin=dict(t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=t_res, y=y_res, mode="markers",
                                      marker=dict(color=C_PRIMARY, opacity=0.4, size=4),
                                      name="Residuals"))
            xs = np.linspace(t_res.min(), t_res.max(), 100)
            fig2.add_trace(go.Scatter(x=xs, y=est * xs, mode="lines",
                                      line=dict(color=C_SECONDARY, width=3),
                                      name=f"θ = {est:.4f}"))
            fig2.update_layout(title="Y_res vs T_res (slope = estimated effect)",
                               xaxis_title="T_res", yaxis_title="Y_res",
                               height=400, template="plotly_dark",
                               margin=dict(t=40, b=30))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Configure parameters on the left and click **Run DML**.")

# ============================= DRCFR ========================================
elif algo == "MIM-DRCFR":
    st.title("MIM-DRCFR")
    st.markdown(
        "Disentangled representations with MMD-based IPM and mutual-information "
        "regularisation for counterfactual regression."
    )

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
             Yf_tr, Yf_val, Y0_val, Y1_val) = train_test_split(
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
                    u_m, u_lv, v_m, v_lv, xr, tr, yr, ta, ya, _, _ = outputs
                    tl, comps = model.compute_loss(bx, bt, by,
                                                   u_m, u_lv, v_m, v_lv,
                                                   xr, tr, yr, ta, ya)
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
                        u_m, u_lv, v_m, v_lv, xr, tr, yr, ta, ya, _, _ = outputs
                        _, comps = model.compute_loss(bx, bt, by,
                                                      u_m, u_lv, v_m, v_lv,
                                                      xr, tr, yr, ta, ya)
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
