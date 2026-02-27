"""Tests for DragonNet."""

import numpy as np
import torch
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn.functional as F
from algorithms.dragonnet.dragonnet_model import DragonNetModel

DEVICE = torch.device("cpu")
torch.manual_seed(0)
np.random.seed(0)


def _make_data(n=100, d=5):
    x = torch.randn(n, d)
    t = (torch.rand(n, 1) > 0.5).float()
    y = torch.randn(n, 1) + 2.0 * t
    return x, y, t


class TestDragonNetInit:
    def test_creates_three_heads(self):
        m = DragonNetModel(5, [10], [6])
        assert hasattr(m, 'h0')
        assert hasattr(m, 'h1')
        assert hasattr(m, 'propensity_head')
        assert hasattr(m, 'epsilon')

    def test_forward_shapes(self):
        m = DragonNetModel(5, [10], [6])
        x = torch.randn(8, 5, device=m.device)
        y0, y1, eps_logit = m(x)
        assert y0.shape == (8, 1)
        assert y1.shape == (8, 1)
        assert eps_logit.shape == (8, 1)

    def test_default_hyperparams(self):
        m = DragonNetModel(5, [10], [6])
        assert m.alpha == 1.0
        assert m.beta == 1.0

    def test_custom_hyperparams(self):
        m = DragonNetModel(5, [10], [6], alpha=0.5, beta=2.0, dropout=0.2)
        assert m.alpha == 0.5
        assert m.beta == 2.0


class TestDragonNetFit:
    def test_returns_history(self):
        m = DragonNetModel(5, [10], [6], learning_rate=1e-3)
        x, y, t = _make_data(50)
        history = m.fit(x, y, t, num_epochs=5, batch_size=25, print_every_epochs=100)
        assert 'train_loss' in history
        assert 'train_outcome' in history
        assert 'train_propensity' in history
        assert 'train_targeted' in history
        assert len(history['train_loss']) == 5

    def test_loss_decreases(self):
        m = DragonNetModel(5, [16, 8], [8], learning_rate=1e-3)
        x, y, t = _make_data(200)
        history = m.fit(x, y, t, num_epochs=50, batch_size=64, print_every_epochs=100)
        assert history['train_loss'][-1] < history['train_loss'][0]

    def test_with_validation(self):
        m = DragonNetModel(5, [10], [6])
        x, y, t = _make_data(100)
        xv, yv, tv = _make_data(30)
        history = m.fit(x, y, t, num_epochs=5, batch_size=50,
                        x_val=xv, y_val=yv, t_val=tv, print_every_epochs=100)
        assert len(history['val_loss']) == 5

    def test_early_stopping(self):
        m = DragonNetModel(5, [10], [6], learning_rate=1e-2)
        x, y, t = _make_data(150)
        xv, yv, tv = _make_data(50)
        history = m.fit(x, y, t, num_epochs=300, batch_size=50,
                        x_val=xv, y_val=yv, t_val=tv,
                        patience=10, print_every_epochs=9999)
        assert len(history['train_loss']) <= 300

    def test_lr_scheduler_cosine(self):
        m = DragonNetModel(5, [10], [6])
        x, y, t = _make_data(50)
        history = m.fit(x, y, t, num_epochs=5, batch_size=50,
                        lr_scheduler='cosine', print_every_epochs=100)
        assert len(history['train_loss']) == 5


class TestDragonNetPredict:
    def _trained_model(self):
        m = DragonNetModel(5, [16], [8], learning_rate=1e-3)
        x, y, t = _make_data(100)
        m.fit(x, y, t, num_epochs=10, batch_size=50, print_every_epochs=100)
        return m

    def test_predict_ite(self):
        m = self._trained_model()
        x = torch.randn(20, 5)
        ite = m.predict_ite(x)
        assert ite.shape == (20, 1)

    def test_predict_potential_outcomes(self):
        m = self._trained_model()
        x = torch.randn(15, 5)
        y0, y1 = m.predict_potential_outcomes(x)
        assert y0.shape == (15, 1)
        assert y1.shape == (15, 1)

    def test_predict_propensity(self):
        m = self._trained_model()
        x = torch.randn(10, 5)
        prop = m.predict_propensity(x)
        assert prop.shape == (10, 1)
        assert (prop >= 0).all() and (prop <= 1).all()

    def test_ite_consistent_with_potentials(self):
        m = self._trained_model()
        x = torch.randn(10, 5)
        ite = m.predict_ite(x)
        y0, y1 = m.predict_potential_outcomes(x)
        torch.testing.assert_close(ite, y1 - y0)


class TestDragonNetAccuracy:
    def test_recovers_constant_effect(self):
        torch.manual_seed(42)
        np.random.seed(42)
        n = 2000
        d = 5
        X = torch.randn(n, d)
        prop = torch.sigmoid(X[:, 0] + X[:, 1])
        T = torch.bernoulli(prop).unsqueeze(1)
        true_effect = 3.0
        Y0 = X[:, 2:3] + X[:, 3:4]
        Y1 = Y0 + true_effect
        Y = T * Y1 + (1 - T) * Y0 + torch.randn(n, 1) * 0.1

        m = DragonNetModel(d, [64, 32], [32], alpha=1.0, beta=1.0,
                           learning_rate=1e-3, dropout=0.1)
        m.fit(X, Y, T, num_epochs=100, batch_size=128, print_every_epochs=9999)

        ite = m.predict_ite(X[:200]).cpu()
        ate_pred = ite.mean().item()
        assert ate_pred == pytest.approx(true_effect, abs=1.0)

    def test_propensity_is_calibrated(self):
        torch.manual_seed(42)
        n = 1000
        X = torch.randn(n, 5)
        true_prop = torch.sigmoid(X[:, 0] + 0.5 * X[:, 1])
        T = torch.bernoulli(true_prop).unsqueeze(1)
        Y = torch.randn(n, 1) + 2.0 * T

        m = DragonNetModel(5, [32], [16], alpha=2.0, beta=0.5, learning_rate=1e-3)
        m.fit(X, Y, T, num_epochs=80, batch_size=128, print_every_epochs=9999)

        pred_prop = m.predict_propensity(X).cpu().squeeze()
        mse = F.mse_loss(pred_prop, true_prop).item()
        assert mse < 0.1
