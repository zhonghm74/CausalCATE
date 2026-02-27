"""Tests for Phase 1 (基础加固) additions across DRCFR, SRCVAE, data, and evaluation."""

import numpy as np
import torch
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===================================================================
# DRCFR Phase 1: Dropout/BN, Early Stopping, History
# ===================================================================

from algorithms.drcfr.networks import Encoder, PredictionHead
from algorithms.drcfr.drcfr_model import DRCFRModel

DEVICE = torch.device("cpu")
torch.manual_seed(0)
np.random.seed(0)


class TestDRCFRDropoutBN:
    def test_encoder_with_dropout(self):
        enc = Encoder(5, [10, 8], 4, 3, dropout=0.3).to(DEVICE)
        x = torch.randn(10, 5, device=DEVICE)
        zy, zs = enc(x)
        assert zy.shape == (10, 4)
        assert zs.shape == (10, 3)

    def test_encoder_with_batchnorm(self):
        enc = Encoder(5, [10, 8], 4, 3, use_batchnorm=True).to(DEVICE)
        x = torch.randn(10, 5, device=DEVICE)
        zy, zs = enc(x)
        assert zy.shape == (10, 4)

    def test_encoder_dropout_and_bn(self):
        enc = Encoder(5, [10], 4, 3, dropout=0.2, use_batchnorm=True).to(DEVICE)
        x = torch.randn(10, 5, device=DEVICE)
        zy, zs = enc(x)
        assert zy.shape == (10, 4)

    def test_predhead_with_dropout(self):
        h = PredictionHead(4, [6], 1, dropout=0.3).to(DEVICE)
        z = torch.randn(10, 4, device=DEVICE)
        assert h(z).shape == (10, 1)

    def test_model_with_dropout_bn(self):
        model = DRCFRModel(
            input_dim=5, hidden_dims_phi=[10], latent_dim_zy=4, latent_dim_zs=3,
            hidden_dims_h=[6], output_dim=1, dropout=0.2, use_batchnorm=True,
        )
        x = torch.randn(10, 5, device=DEVICE)
        h0, h1, zy, zs = model.forward(x)
        assert h0.shape == (10, 1)


class TestDRCFREarlyStopping:
    def _make_data(self, n=100):
        x = torch.randn(n, 5)
        t = (torch.rand(n, 1) > 0.5).float()
        y = torch.randn(n, 1)
        return x, y, t

    def test_fit_returns_history(self):
        model = DRCFRModel(5, [10], 4, 3, [6], learning_rate=1e-3)
        x, y, t = self._make_data()
        history = model.fit(x, y, t, num_epochs=5, batch_size=50, print_every_epochs=100)
        assert isinstance(history, dict)
        assert 'train' in history
        assert len(history['train']['total']) == 5

    def test_early_stopping(self):
        model = DRCFRModel(5, [10], 4, 3, [6], learning_rate=1e-2)
        x, y, t = self._make_data(200)
        xv, yv, tv = self._make_data(50)
        history = model.fit(x, y, t, num_epochs=200, batch_size=50,
                            x_val=xv, y_val=yv, t_val=tv,
                            patience=5, print_every_epochs=1000)
        assert len(history['train']['total']) <= 200

    def test_lr_scheduler_plateau(self):
        model = DRCFRModel(5, [10], 4, 3, [6], learning_rate=1e-2)
        x, y, t = self._make_data(100)
        xv, yv, tv = self._make_data(30)
        history = model.fit(x, y, t, num_epochs=10, batch_size=50,
                            x_val=xv, y_val=yv, t_val=tv,
                            lr_scheduler='plateau', patience=5, print_every_epochs=1000)
        assert isinstance(history, dict)

    def test_lr_scheduler_cosine(self):
        model = DRCFRModel(5, [10], 4, 3, [6], learning_rate=1e-2)
        x, y, t = self._make_data(100)
        history = model.fit(x, y, t, num_epochs=10, batch_size=50,
                            lr_scheduler='cosine', print_every_epochs=1000)
        assert len(history['train']['total']) == 10


# ===================================================================
# SRCVAE Phase 1: Heteroscedastic, KL Annealing
# ===================================================================

from algorithms.srcvae.srcvae_model import SRCVAEModel
from algorithms.srcvae.srcvae_losses import gaussian_nll_loss


class TestSRCVAEHeteroscedastic:
    def _default_params(self, **overrides):
        p = dict(
            x_dim=5, t_dim=1, y_dim=1, u_dim=3, v_dim=3,
            hidden_dims_encoder_u=[8], hidden_dims_encoder_v=[8],
            hidden_dims_decoder_x=[6], hidden_dims_decoder_t=[6],
            hidden_dims_decoder_y=[6], hidden_dims_aux_qtx=[6],
            hidden_dims_aux_qyxt=[6],
            alpha_x=1.0, alpha_t=1.0, alpha_y=1.0,
            beta_u=0.1, beta_v=0.1, gamma_t=1.0, gamma_y=1.0,
            learning_rate=1e-3, device=DEVICE,
        )
        p.update(overrides)
        return p

    def test_homoscedastic_backward_compat(self):
        model = SRCVAEModel(**self._default_params())
        x = torch.randn(10, 5, device=DEVICE)
        t = torch.randint(0, 2, (10,), device=DEVICE, dtype=torch.float32)
        y = torch.randn(10, device=DEVICE)
        outputs = model.forward(x, t, y)
        assert len(outputs) == 13
        assert outputs[11] is None  # x_recon_logvar
        assert outputs[12] is None  # y_recon_logvar

    def test_heteroscedastic_outputs(self):
        model = SRCVAEModel(**self._default_params(heteroscedastic=True))
        x = torch.randn(10, 5, device=DEVICE)
        t = torch.randint(0, 2, (10,), device=DEVICE, dtype=torch.float32)
        y = torch.randn(10, device=DEVICE)
        outputs = model.forward(x, t, y)
        assert outputs[11] is not None  # x_recon_logvar
        assert outputs[12] is not None  # y_recon_logvar
        assert outputs[11].shape == (10, 5)  # x_dim
        assert outputs[12].shape == (10, 1)  # y_dim

    def test_heteroscedastic_fit(self):
        model = SRCVAEModel(**self._default_params(heteroscedastic=True))
        x = torch.randn(20, 5, device=DEVICE)
        t = torch.randint(0, 2, (20,), device=DEVICE, dtype=torch.float32)
        y = torch.randn(20, device=DEVICE)
        history = model.fit(x, t, y, num_epochs=3, batch_size=20, print_every_epochs=100)
        assert isinstance(history, dict)

    def test_gaussian_nll_loss(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        mean = torch.tensor([1.1, 2.0, 2.9])
        logvar = torch.zeros(3)
        loss = gaussian_nll_loss(y, mean, logvar)
        assert loss.item() > 0


class TestSRCVAEKLAnnealing:
    def _make_model_and_data(self, n=50):
        params = dict(
            x_dim=5, t_dim=1, y_dim=1, u_dim=3, v_dim=3,
            hidden_dims_encoder_u=[8], hidden_dims_encoder_v=[8],
            hidden_dims_decoder_x=[6], hidden_dims_decoder_t=[6],
            hidden_dims_decoder_y=[6], hidden_dims_aux_qtx=[6],
            hidden_dims_aux_qyxt=[6],
            alpha_x=1.0, alpha_t=1.0, alpha_y=1.0,
            beta_u=0.5, beta_v=0.5, gamma_t=1.0, gamma_y=1.0,
            learning_rate=1e-3, device=DEVICE,
        )
        model = SRCVAEModel(**params)
        x = torch.randn(n, 5, device=DEVICE)
        t = torch.randint(0, 2, (n,), device=DEVICE, dtype=torch.float32)
        y = torch.randn(n, device=DEVICE)
        return model, x, t, y

    def test_kl_warmup(self):
        model, x, t, y = self._make_model_and_data()
        history = model.fit(x, t, y, num_epochs=10, batch_size=50,
                            kl_warmup_epochs=5, print_every_epochs=100)
        assert len(history['train']['total_loss']) == 10

    def test_early_stopping(self):
        model, x, t, y = self._make_model_and_data(100)
        xv = torch.randn(30, 5, device=DEVICE)
        tv = torch.randint(0, 2, (30,), device=DEVICE, dtype=torch.float32)
        yv = torch.randn(30, device=DEVICE)
        history = model.fit(x, t, y, num_epochs=200, batch_size=50,
                            x_val=xv, t_val=tv, y_val=yv,
                            patience=5, print_every_epochs=1000)
        assert len(history['train']['total_loss']) <= 200

    def test_fit_returns_history(self):
        model, x, t, y = self._make_model_and_data()
        history = model.fit(x, t, y, num_epochs=3, batch_size=50, print_every_epochs=100)
        assert 'train' in history and 'val' in history
        assert 'total_loss' in history['train']
        assert len(history['train']['total_loss']) == 3


# ===================================================================
# Data Module
# ===================================================================

from data.ihdp import generate_ihdp, IHDPDataset
from data.synthetic import generate_binary_treatment, generate_continuous_treatment, generate_multi_treatment


class TestIHDP:
    def test_generate_default(self):
        ds = generate_ihdp()
        assert isinstance(ds, IHDPDataset)
        assert ds.n_samples == 747
        assert ds.n_features == 25
        assert ds.X.shape == (747, 25)
        assert len(ds.T) == 747
        assert len(ds.Y) == 747

    def test_custom_size(self):
        ds = generate_ihdp(n_samples=500, n_features=10, seed=99)
        assert ds.X.shape == (500, 10)

    def test_train_test_split(self):
        ds = generate_ihdp(n_samples=200, n_features=10)
        train, test = ds.train_test_split(test_size=0.3)
        assert train.n_samples + test.n_samples == 200
        assert train.X.shape[1] == 10

    def test_ate_att(self):
        ds = generate_ihdp(n_samples=1000)
        assert isinstance(ds.ATE, float)
        assert isinstance(ds.ATT, float)
        assert len(ds.ITE) == 1000

    def test_reproducibility(self):
        ds1 = generate_ihdp(seed=42)
        ds2 = generate_ihdp(seed=42)
        np.testing.assert_array_equal(ds1.X, ds2.X)


class TestSyntheticData:
    def test_binary(self):
        d = generate_binary_treatment(n_samples=100)
        assert d['X'].shape[0] == 100
        assert 'Y0' in d and 'Y1' in d

    def test_continuous(self):
        d = generate_continuous_treatment(n_samples=100)
        assert d['X'].shape[0] == 100

    def test_multi(self):
        d = generate_multi_treatment(n_samples=100)
        assert set(d['T']).issubset({0, 1, 2})


# ===================================================================
# Evaluation Metrics
# ===================================================================

from evaluation.metrics import ate_error, pehe, att_error, policy_risk, evaluation_report, EvalReport


class TestEvalMetrics:
    def test_ate_error_zero(self):
        ite = np.array([1.0, 2.0, 3.0])
        assert ate_error(ite, ite) == pytest.approx(0.0)

    def test_ate_error_nonzero(self):
        true = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, 2.0, 4.0])
        assert ate_error(true, pred) == pytest.approx(1.0 / 3)

    def test_pehe_zero(self):
        ite = np.array([1.0, 2.0])
        assert pehe(ite, ite) == pytest.approx(0.0)

    def test_pehe_nonzero(self):
        true = np.array([1.0, 2.0])
        pred = np.array([2.0, 2.0])
        assert pehe(true, pred) == pytest.approx(np.sqrt(0.5))

    def test_att_error(self):
        ite_t = np.array([1.0, 2.0, 3.0, 4.0])
        ite_p = np.array([1.0, 2.0, 3.0, 5.0])
        T = np.array([0, 1, 0, 1])
        err = att_error(ite_t, ite_p, T)
        assert err == pytest.approx(0.5)

    def test_policy_risk_perfect(self):
        ite = np.array([1.0, -1.0, 2.0])
        assert policy_risk(ite, ite) == pytest.approx(0.0)

    def test_policy_risk_worst(self):
        true = np.array([1.0, 1.0])
        pred = np.array([-1.0, -1.0])
        assert policy_risk(true, pred) == pytest.approx(1.0)

    def test_evaluation_report(self):
        true = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.1, 1.9, 3.2])
        T = np.array([0, 1, 1])
        report = evaluation_report(true, pred, T)
        assert isinstance(report, EvalReport)
        assert report.pehe > 0
        assert "Evaluation Report" in report.summary()
