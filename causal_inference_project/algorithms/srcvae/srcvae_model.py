"""
Main Model Class for the SRCVAE Algorithm.

Supports optional heteroscedastic decoders (learnable variance),
KL annealing, early stopping, LR scheduling, and structured training history.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from .srcvae_networks import EncoderU, EncoderV, DecoderX, DecoderT, DecoderY, AuxiliaryQTX, AuxiliaryQYXT
from .srcvae_losses import kl_gaussian_loss, reconstruction_mse_loss, reconstruction_bce_loss, gaussian_nll_loss

class SRCVAEModel(nn.Module):
    """
    SRCVAE (Learning Causal Effect Variational Autoencoder) Model.

    This model implements the SRCVAE algorithm designed to estimate causal effects
    from observational data, particularly addressing challenges like unobserved
    confounding and potentially unknown treatment assignment mechanisms. It leverages
    a variational autoencoder framework with two latent variables:
    - `u`: Represents unobserved confounders affecting x, t, and y.
    - `v`: Represents instrumental variables affecting x and t, but not y directly.

    The model consists of:
    - Encoders for `u` (q_phi_u(u|x,t,y_f)) and `v` (q_phi_v(v|x,t)).
    - Decoders for reconstructing `x` (p_theta_x(x|u,v)), `t` (p_theta_t(t|x,v)),
      and `y` (p_theta_y(y|x,u,t)).
    - Auxiliary networks for predicting `t` from `x` (q_phi_t(t|x)) and `y` from
      `x,t` (q_phi_y(y|x,t)) to aid representation learning.

    The total loss function is a weighted sum of KL divergences for `u` and `v`,
    reconstruction losses for `x`, `t`, and `y`, and auxiliary prediction losses for `t` and `y`.

    Attributes:
        encoder_u, encoder_v: Encoder networks.
        decoder_x, decoder_t, decoder_y: Decoder networks.
        aux_qtx, aux_qyxt: Auxiliary prediction networks.
        alpha_x, alpha_t, alpha_y: Weights for reconstruction losses.
        beta_u, beta_v: Weights for KL divergence losses.
        gamma_t, gamma_y: Weights for auxiliary prediction losses.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        device (torch.device): The device (CPU or CUDA) the model is on.
    """
    def __init__(self, x_dim, t_dim, y_dim, u_dim, v_dim,
                 hidden_dims_encoder_u, hidden_dims_encoder_v,
                 hidden_dims_decoder_x, hidden_dims_decoder_t, hidden_dims_decoder_y,
                 hidden_dims_aux_qtx, hidden_dims_aux_qyxt,
                 alpha_x, alpha_t, alpha_y,
                 beta_u, beta_v,
                 gamma_t, gamma_y,
                 learning_rate=1e-3, weight_decay=1e-4, device=None,
                 heteroscedastic=False):
        """
        Initializes the SRCVAEModel.

        Args:
            x_dim (int): Dimensionality of input features `x`.
            t_dim (int): Dimensionality of treatment `t` (typically 1 for binary).
            y_dim (int): Dimensionality of outcome `y` (typically 1 for continuous).
            u_dim (int): Dimensionality of latent variable `u`.
            v_dim (int): Dimensionality of latent variable `v`.
            hidden_dims_encoder_u (list of int): Hidden layer dimensions for `EncoderU`.
            hidden_dims_encoder_v (list of int): Hidden layer dimensions for `EncoderV`.
            hidden_dims_decoder_x (list of int): Hidden layer dimensions for `DecoderX`.
            hidden_dims_decoder_t (list of int): Hidden layer dimensions for `DecoderT`.
            hidden_dims_decoder_y (list of int): Hidden layer dimensions for `DecoderY`.
            hidden_dims_aux_qtx (list of int): Hidden layer dimensions for `AuxiliaryQTX`.
            hidden_dims_aux_qyxt (list of int): Hidden layer dimensions for `AuxiliaryQYXT`.
            alpha_x (float): Weight for the `x` reconstruction loss.
            alpha_t (float): Weight for the `t` reconstruction loss.
            alpha_y (float): Weight for the `y` reconstruction loss.
            beta_u (float): Weight for the KL divergence of `u` (vs. its prior).
            beta_v (float): Weight for the KL divergence of `v` (vs. its prior).
            gamma_t (float): Weight for the auxiliary `t` prediction loss (q_phi_t(t|x)).
            gamma_y (float): Weight for the auxiliary `y` prediction loss (q_phi_y(y|x,t)).
            learning_rate (float, optional): Learning rate for the Adam optimizer. Defaults to 1e-3.
            weight_decay (float, optional): Weight decay (L2 penalty) for the Adam optimizer. Defaults to 1e-4.
            device (torch.device, optional): Device to run the model on (e.g., 'cpu', 'cuda').
                                             If None, uses GPU if available, otherwise CPU. Defaults to None.
        """
        super().__init__()

        self.x_dim = x_dim
        self.t_dim = t_dim
        self.y_dim = y_dim
        self.u_dim = u_dim
        self.v_dim = v_dim
        self.heteroscedastic = heteroscedastic

        self.encoder_u = EncoderU(x_dim, t_dim, y_dim, u_dim, hidden_dims_encoder_u)
        self.encoder_v = EncoderV(x_dim, t_dim, v_dim, hidden_dims_encoder_v)
        self.decoder_x = DecoderX(u_dim, v_dim, x_dim, hidden_dims_decoder_x,
                                  heteroscedastic=heteroscedastic)
        self.decoder_t = DecoderT(x_dim, v_dim, t_dim, hidden_dims_decoder_t)
        self.decoder_y = DecoderY(x_dim, u_dim, t_dim, y_dim, hidden_dims_decoder_y,
                                  heteroscedastic=heteroscedastic)
        self.aux_qtx = AuxiliaryQTX(x_dim, t_dim, hidden_dims_aux_qtx)
        self.aux_qyxt = AuxiliaryQYXT(x_dim, t_dim, y_dim, hidden_dims_aux_qyxt)

        # Store loss weights
        self.alpha_x = alpha_x
        self.alpha_t = alpha_t
        self.alpha_y = alpha_y
        self.beta_u = beta_u
        self.beta_v = beta_v
        self.gamma_t = gamma_t
        self.gamma_y = gamma_y

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Device handling
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.to(self.device)

    def reparameterize(self, mean, logvar):
        """
        Implements the reparameterization trick for sampling from a Gaussian distribution.

        Given the mean and log-variance of a Gaussian, this method returns a sample
        from the distribution by adding scaled Gaussian noise to the mean.
        z = mean + std * epsilon, where std = exp(0.5 * logvar) and epsilon ~ N(0, I).

        Args:
            mean (torch.Tensor): The mean of the Gaussian distribution.
            logvar (torch.Tensor): The log-variance of the Gaussian distribution.

        Returns:
            torch.Tensor: A sample from the Gaussian distribution.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, t, y_f):
        """
        Performs a full forward pass of the SRCVAE model.

        This involves encoding inputs to latent spaces, sampling from these latent
        distributions, and then decoding these samples to reconstruct inputs and
        make auxiliary predictions.

        Args:
            x (torch.Tensor): Input features, shape (batch_size, x_dim).
            t (torch.Tensor): Observed treatment, shape (batch_size, t_dim) or (batch_size,).
            y_f (torch.Tensor): Observed factual outcome, shape (batch_size, y_dim) or (batch_size,).
                                This is used by `EncoderU`.

        Returns:
            tuple: A tuple containing the following tensors:
                - u_mean (torch.Tensor): Mean of the Gaussian for u from `EncoderU`.
                - u_logvar (torch.Tensor): Log-variance of the Gaussian for u from `EncoderU`.
                - v_mean (torch.Tensor): Mean of the Gaussian for v from `EncoderV`.
                - v_logvar (torch.Tensor): Log-variance of the Gaussian for v from `EncoderV`.
                - x_recon_mean (torch.Tensor): Reconstructed mean of x from `DecoderX`.
                - t_recon_logits (torch.Tensor): Reconstructed logits of t from `DecoderT`.
                - y_recon_mean (torch.Tensor): Reconstructed mean of y from `DecoderY`.
                - t_aux_logits (torch.Tensor): Predicted logits of t from `AuxiliaryQTX`.
                - y_aux_mean (torch.Tensor): Predicted mean of y from `AuxiliaryQYXT`.
                - u_sampled (torch.Tensor): Sampled `u` using reparameterization.
                - v_sampled (torch.Tensor): Sampled `v` using reparameterization.
        """
        # Encoders
        u_mean, u_logvar = self.encoder_u(x, t, y_f)
        v_mean, v_logvar = self.encoder_v(x, t)

        # Reparameterize
        u_sampled = self.reparameterize(u_mean, u_logvar)
        v_sampled = self.reparameterize(v_mean, v_logvar)

        # Decoders (may return tuples when heteroscedastic)
        x_dec_out = self.decoder_x(u_sampled, v_sampled)
        t_recon_logits = self.decoder_t(x, v_sampled)
        y_dec_out = self.decoder_y(x, u_sampled, t)

        if self.heteroscedastic:
            x_recon_mean, x_recon_logvar = x_dec_out
            y_recon_mean, y_recon_logvar = y_dec_out
        else:
            x_recon_mean = x_dec_out
            x_recon_logvar = None
            y_recon_mean = y_dec_out
            y_recon_logvar = None

        t_aux_logits = self.aux_qtx(x)
        y_aux_mean = self.aux_qyxt(x, t)

        return (u_mean, u_logvar, v_mean, v_logvar,
                x_recon_mean, t_recon_logits, y_recon_mean,
                t_aux_logits, y_aux_mean,
                u_sampled, v_sampled,
                x_recon_logvar, y_recon_logvar)

    def compute_loss(self, x_true, t_true, y_true,
                     u_mean, u_logvar, v_mean, v_logvar,
                     x_recon_mean, t_recon_logits, y_recon_mean,
                     t_aux_logits, y_aux_mean,
                     x_recon_logvar=None, y_recon_logvar=None,
                     kl_weight=1.0):
        """
        Computes the total SRCVAE loss and its individual components.

        The total loss is a weighted sum of:
        - KL divergence for u (beta_u * KL(q_u||p_u)).
        - KL divergence for v (beta_v * KL(q_v||p_v)).
        - Reconstruction loss for x (alpha_x * E[log p_x(x|u,v)]).
        - Reconstruction loss for t (alpha_t * E[log p_t(t|x,v)]).
        - Reconstruction loss for y (alpha_y * E[log p_y(y|x,u,t)]).
        - Auxiliary prediction loss for t (gamma_t * E[log q_t(t|x)]).
        - Auxiliary prediction loss for y (gamma_y * E[log q_y(y|x,t)]).

        Args:
            x_true (torch.Tensor): Ground truth features x.
            t_true (torch.Tensor): Ground truth treatment t.
            y_true (torch.Tensor): Ground truth (factual) outcome y.
            u_mean, u_logvar: Parameters of the Gaussian for u from `EncoderU`.
            v_mean, v_logvar: Parameters of the Gaussian for v from `EncoderV`.
            x_recon_mean: Reconstructed mean of x from `DecoderX`.
            t_recon_logits: Reconstructed logits of t from `DecoderT`.
            y_recon_mean: Reconstructed mean of y from `DecoderY`.
            t_aux_logits: Predicted logits of t from `AuxiliaryQTX`.
            y_aux_mean: Predicted mean of y from `AuxiliaryQYXT`.

        Returns:
            tuple:
                - total_loss (torch.Tensor): The total computed loss (scalar).
                - loss_components (dict): A dictionary containing the itemized values of
                                          individual loss components.
        """
        loss_kl_u = kl_gaussian_loss(u_mean, u_logvar)
        loss_kl_v = kl_gaussian_loss(v_mean, v_logvar)

        # Ensure t_true and y_true have correct shapes for loss functions if they are 1D
        if t_true.ndim == 1: t_true = t_true.unsqueeze(1)
        if y_true.ndim == 1: y_true = y_true.unsqueeze(1)

        if self.heteroscedastic and x_recon_logvar is not None:
            loss_recon_x = gaussian_nll_loss(x_true, x_recon_mean, x_recon_logvar)
        else:
            loss_recon_x = reconstruction_mse_loss(x_true, x_recon_mean)
        loss_recon_t = reconstruction_bce_loss(t_true, t_recon_logits)
        if self.heteroscedastic and y_recon_logvar is not None:
            loss_recon_y = gaussian_nll_loss(y_true, y_recon_mean, y_recon_logvar)
        else:
            loss_recon_y = reconstruction_mse_loss(y_true, y_recon_mean)
        
        loss_aux_qt = reconstruction_bce_loss(t_true, t_aux_logits)
        loss_aux_qy = reconstruction_mse_loss(y_true, y_aux_mean)

        total_loss = (self.alpha_x * loss_recon_x +
                      self.alpha_t * loss_recon_t +
                      self.alpha_y * loss_recon_y +
                      kl_weight * (self.beta_u * loss_kl_u + self.beta_v * loss_kl_v) +
                      self.gamma_t * loss_aux_qt +
                      self.gamma_y * loss_aux_qy)
        
        loss_components = {
            'total_loss': total_loss.item(), 'loss_kl_u': loss_kl_u.item(), 'loss_kl_v': loss_kl_v.item(),
            'loss_recon_x': loss_recon_x.item(), 'loss_recon_t': loss_recon_t.item(), 'loss_recon_y': loss_recon_y.item(),
            'loss_aux_qt': loss_aux_qt.item(), 'loss_aux_qy': loss_aux_qy.item()
        }
        return total_loss, loss_components

    def _unpack_forward(self, outputs):
        """Unpack the 13-element forward output tuple."""
        (u_mean, u_logvar, v_mean, v_logvar,
         x_recon_mean, t_recon_logits, y_recon_mean,
         t_aux_logits, y_aux_mean, u_s, v_s,
         x_recon_logvar, y_recon_logvar) = outputs
        return (u_mean, u_logvar, v_mean, v_logvar,
                x_recon_mean, t_recon_logits, y_recon_mean,
                t_aux_logits, y_aux_mean, x_recon_logvar, y_recon_logvar)

    def fit(self, x_train, t_train, y_train, num_epochs, batch_size,
            x_val=None, t_val=None, y_val=None, print_every_epochs=10,
            patience=None, lr_scheduler=None, kl_warmup_epochs=0):
        """Train the model.  Returns a ``history`` dict.

        New optional args (backward-compatible):
            patience: Early-stopping patience.  Requires validation data.
            lr_scheduler: ``'plateau'`` or ``'cosine'``, or ``None``.
            kl_warmup_epochs: Number of epochs to linearly anneal KL weight
                from 0 to 1.  ``0`` = no annealing (default).
        """
        self.train()

        x_train = x_train.to(self.device)
        t_train = t_train.to(self.device)
        y_train = y_train.to(self.device)
        if t_train.ndim == 1: t_train = t_train.unsqueeze(1)
        if y_train.ndim == 1: y_train = y_train.unsqueeze(1)

        train_dataset = TensorDataset(x_train, t_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if x_val is not None and t_val is not None and y_val is not None:
            x_val = x_val.to(self.device)
            t_val = t_val.to(self.device)
            y_val = y_val.to(self.device)
            if t_val.ndim == 1: t_val = t_val.unsqueeze(1)
            if y_val.ndim == 1: y_val = y_val.unsqueeze(1)
            val_dataset = TensorDataset(x_val, t_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # LR scheduler
        scheduler = None
        if lr_scheduler == 'plateau' and val_loader:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=max(1, (patience or 10) // 2))
        elif lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_state = None

        loss_keys = ['total_loss', 'loss_kl_u', 'loss_kl_v',
                     'loss_recon_x', 'loss_recon_t', 'loss_recon_y',
                     'loss_aux_qt', 'loss_aux_qy']
        history = {'train': {k: [] for k in loss_keys}, 'val': {k: [] for k in loss_keys}}

        for epoch in range(num_epochs):
            self.train()
            kl_weight = min(1.0, (epoch + 1) / max(kl_warmup_epochs, 1)) if kl_warmup_epochs > 0 else 1.0

            ep_acc = {k: 0.0 for k in loss_keys}
            for bx, bt, by in train_loader:
                self.optimizer.zero_grad()
                fwd = self._unpack_forward(self.forward(bx, bt, by))
                (u_m, u_lv, v_m, v_lv, xr, tr, yr, ta, ya, xlv, ylv) = fwd
                tl, comps = self.compute_loss(bx, bt, by, u_m, u_lv, v_m, v_lv,
                                              xr, tr, yr, ta, ya, xlv, ylv,
                                              kl_weight=kl_weight)
                tl.backward()
                self.optimizer.step()
                for k in loss_keys:
                    ep_acc[k] += comps[k]
            nb = len(train_loader)
            for k in loss_keys:
                history['train'][k].append(ep_acc[k] / nb)

            # Validation
            val_total = None
            if val_loader:
                self.eval()
                vep = {k: 0.0 for k in loss_keys}
                with torch.no_grad():
                    for bx, bt, by in val_loader:
                        fwd = self._unpack_forward(self.forward(bx, bt, by))
                        (u_m, u_lv, v_m, v_lv, xr, tr, yr, ta, ya, xlv, ylv) = fwd
                        _, comps = self.compute_loss(bx, bt, by, u_m, u_lv, v_m, v_lv,
                                                     xr, tr, yr, ta, ya, xlv, ylv,
                                                     kl_weight=kl_weight)
                        for k in loss_keys:
                            vep[k] += comps[k]
                vnb = len(val_loader)
                for k in loss_keys:
                    history['val'][k].append(vep[k] / vnb)
                val_total = vep['total_loss'] / vnb

            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau) and val_total is not None:
                    scheduler.step(val_total)
                elif not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()

            if patience is not None and val_total is not None:
                if val_total < best_val_loss - 1e-6:
                    best_val_loss = val_total
                    epochs_no_improve = 0
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if best_state is not None:
                        self.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
                    print(f"Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.4f})")
                    break

            if (epoch + 1) % print_every_epochs == 0:
                avg = ", ".join([f"{k}: {history['train'][k][-1]:.4f}" for k in loss_keys])
                print(f"Epoch {epoch+1}/{num_epochs} - Train Losses: {avg}")
                if val_loader:
                    avg_v = ", ".join([f"{k}: {history['val'][k][-1]:.4f}" for k in loss_keys])
                    print(f"Epoch {epoch+1}/{num_epochs} - Val Losses: {avg_v}")

        return history

    def estimate_causal_effect(self, x_factual, n_samples_u=100):
        """
        Estimates the Individual Treatment Effect (ITE) for given factual covariates `x_factual`.

        The ITE is estimated as E[Y|do(t=1), x] - E[Y|do(t=0), x].
        This is achieved by:
        1. Sampling `u` from its prior distribution p(u) = N(0, I) multiple times.
        2. For each sample of `u` and given `x_factual`:
           a. Predict Y(0) using `DecoderY` with `t=0`.
           b. Predict Y(1) using `DecoderY` with `t=1`.
        3. Averaging the predictions for Y(0) and Y(1) over the samples of `u`.

        Args:
            x_factual (torch.Tensor): Factual covariates, shape (batch_size, x_dim).
            n_samples_u (int, optional): Number of samples to draw from the prior p(u)
                                         for Monte Carlo estimation. Defaults to 100.

        Returns:
            tuple:
                - ite (torch.Tensor): Estimated ITEs, shape (batch_size, y_dim).
                - y0_pred_mean (torch.Tensor): Estimated potential outcomes Y(0),
                                               shape (batch_size, y_dim).
                - y1_pred_mean (torch.Tensor): Estimated potential outcomes Y(1),
                                               shape (batch_size, y_dim).
        """
        self.eval() # Set model to evaluation mode
        x_factual = x_factual.to(self.device)
        batch_size = x_factual.shape[0]

        # Create treatment tensors for intervention t=0 and t=1
        # Ensure they have shape (batch_size, t_dim)
        t_zeros = torch.zeros(batch_size, self.t_dim, device=self.device)
        t_ones = torch.ones(batch_size, self.t_dim, device=self.device)

        y_preds_do_t0_list = []
        y_preds_do_t1_list = []

        with torch.no_grad():
            for _ in range(n_samples_u):
                # Sample u from prior p(u) = N(0, I)
                u_prior_samples = torch.randn(batch_size, self.u_dim, device=self.device)
                
                # Predict Y(0) using DecoderY(x, u_sampled_from_prior, t=0)
                y_pred_do_t0_sample = self.decoder_y(x_factual, u_prior_samples, t_zeros)
                y_preds_do_t0_list.append(y_pred_do_t0_sample)

                # Predict Y(1) using DecoderY(x, u_sampled_from_prior, t=1)
                y_pred_do_t1_sample = self.decoder_y(x_factual, u_prior_samples, t_ones)
                y_preds_do_t1_list.append(y_pred_do_t1_sample)
        
        # Average predictions over the u_prior_samples
        y0_pred_mean = torch.stack(y_preds_do_t0_list).mean(dim=0)
        y1_pred_mean = torch.stack(y_preds_do_t1_list).mean(dim=0)
        
        ite = y1_pred_mean - y0_pred_mean
        return ite, y0_pred_mean, y1_pred_mean

# Example Usage (for testing purposes, can be removed or commented out later)
if __name__ == '__main__':
    params = {
        'x_dim': 10, 't_dim': 1, 'y_dim': 1, 'u_dim': 5, 'v_dim': 4,
        'hidden_dims_encoder_u': [32, 16], 'hidden_dims_encoder_v': [32, 16],
        'hidden_dims_decoder_x': [16, 32], 'hidden_dims_decoder_t': [16, 32], 
        'hidden_dims_decoder_y': [16, 32], 'hidden_dims_aux_qtx': [16], 
        'hidden_dims_aux_qyxt': [16],
        'alpha_x': 1.0, 'alpha_t': 1.0, 'alpha_y': 1.0,
        'beta_u': 0.1, 'beta_v': 0.1,
        'gamma_t': 0.5, 'gamma_y': 0.5,
        'learning_rate': 1e-3, 'weight_decay': 1e-5
    }
    model = SRCVAEModel(**params)
    print(f"SRCVAEModel instantiated on device: {next(model.parameters()).device}")

    # Dummy data for training
    n_train = 200
    x_train_ex = torch.randn(n_train, params['x_dim'])
    t_train_ex = (torch.rand(n_train, params['t_dim']) > 0.5).float()
    y_train_ex = torch.randn(n_train, params['y_dim'])

    n_val = 50
    x_val_ex = torch.randn(n_val, params['x_dim'])
    t_val_ex = (torch.rand(n_val, params['t_dim']) > 0.5).float()
    y_val_ex = torch.randn(n_val, params['y_dim'])

    print("\n--- Testing fit method ---")
    model.fit(x_train_ex, t_train_ex, y_train_ex, 
              num_epochs=3, batch_size=32,
              x_val=x_val_ex, t_val=t_val_ex, y_val=y_val_ex,
              print_every_epochs=1)

    print("\n--- Testing estimate_causal_effect method ---")
    x_test_ex = torch.randn(5, params['x_dim']) # Estimate ITE for 5 samples
    ite_preds, y0_preds, y1_preds = model.estimate_causal_effect(x_test_ex, n_samples_u=20)
    print(f"ITE predictions shape: {ite_preds.shape}")
    print(f"Y(0) predictions shape: {y0_preds.shape}")
    print(f"Y(1) predictions shape: {y1_preds.shape}")

    print(f"\nSRCVAEModel has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
