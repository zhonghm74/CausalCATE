"""
Main Model Class for MIM-DRCFR Algorithm.

Integrates Encoder, PredictionHeads, and regularised loss functions.
Supports optional Dropout/BatchNorm, early stopping, LR scheduling,
and returns structured training history.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from .networks import Encoder, PredictionHead
from .losses import factual_mse_loss, ipm_loss_zy, mi_loss_zs_t


class DRCFRModel(nn.Module):
    """MIM-DRCFR model with optional Dropout/BatchNorm, early stopping, and LR scheduling."""

    def __init__(self, input_dim,
                 hidden_dims_phi, latent_dim_zy, latent_dim_zs,
                 hidden_dims_h, output_dim=1,
                 alpha=1.0, beta=1.0,
                 learning_rate=1e-3, weight_decay=1e-4,
                 sigmas_ipm=None, sigmas_mi=None,
                 dropout=0.0, use_batchnorm=False,
                 ipm_method='mmd', sinkhorn_eps=0.1):
        super(DRCFRModel, self).__init__()

        self.encoder = Encoder(input_dim, hidden_dims_phi, latent_dim_zy, latent_dim_zs,
                               dropout=dropout, use_batchnorm=use_batchnorm)
        self.h0 = PredictionHead(latent_dim_zy, hidden_dims_h, output_dim,
                                 dropout=dropout, use_batchnorm=use_batchnorm)
        self.h1 = PredictionHead(latent_dim_zy, hidden_dims_h, output_dim,
                                 dropout=dropout, use_batchnorm=use_batchnorm)

        self.alpha = alpha
        self.beta = beta
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ipm_method = ipm_method
        self.sinkhorn_eps = sinkhorn_eps

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.sigmas_ipm = sigmas_ipm if sigmas_ipm is not None else [0.1, 1.0, 10.0]
        self.sigmas_mi = sigmas_mi if sigmas_mi is not None else [0.1, 1.0, 10.0]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


    def forward(self, x):
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): Input feature tensor of shape (batch_size, input_dim).

        Returns:
            tuple: Contains the following tensors:
                - y_pred_h0 (torch.Tensor): Predicted outcomes from the control head h0,
                                            shape (batch_size, output_dim).
                - y_pred_h1 (torch.Tensor): Predicted outcomes from the treated head h1,
                                            shape (batch_size, output_dim).
                - z_y (torch.Tensor): Latent representation for outcome prediction from the encoder,
                                      shape (batch_size, latent_dim_zy).
                - z_s (torch.Tensor): Latent representation for similarity/treatment mechanism
                                      from the encoder, shape (batch_size, latent_dim_zs).
        """
        z_y, z_s = self.encoder(x)
        y_pred_h0 = self.h0(z_y)
        y_pred_h1 = self.h1(z_y)
        return y_pred_h0, y_pred_h1, z_y, z_s

    def compute_loss(self, x, y_f, t_f, y_pred_h0, y_pred_h1, z_y, z_s):
        """
        Computes the total loss for the MIM-DRCFR model and its individual components.

        The total loss is a weighted sum of:
        - Factual MSE loss (L_R).
        - IPM loss on z_y (L_IPM).
        - MI minimization loss between z_s and t (L_I).
        Total Loss = L_R + alpha * L_IPM + beta * L_I.

        Args:
            x (torch.Tensor): Input features, shape (batch_size, input_dim).
                              (Currently unused in this specific loss calculation logic but passed for potential future use).
            y_f (torch.Tensor): Factual outcomes, shape (batch_size, output_dim) or (batch_size,).
            t_f (torch.Tensor): Factual binary treatments (0 or 1), shape (batch_size,) or (batch_size, 1).
                                Must be boolean or convertible to boolean.
            y_pred_h0 (torch.Tensor): Predictions from control head h0 for all samples.
            y_pred_h1 (torch.Tensor): Predictions from treated head h1 for all samples.
            z_y (torch.Tensor): Latent representation z_y for all samples.
            z_s (torch.Tensor): Latent representation z_s for all samples.

        Returns:
            tuple:
                - total_loss (torch.Tensor): The total computed loss (scalar).
                - loss_r (torch.Tensor): The factual MSE loss component (scalar).
                - loss_ipm (torch.Tensor): The IPM loss component for z_y balancing (scalar).
                - loss_mi (torch.Tensor): The MI minimization loss component for z_s and t (scalar).
        """
        # Ensure t_f is boolean for indexing if it's not already
        if t_f.dtype != torch.bool:
             if t_f.ndim > 1 and t_f.shape[1] == 1:
                t_f_bool = t_f.squeeze(1).bool()
             else:
                t_f_bool = t_f.bool() # Assumes t_f can be directly converted if 1D
        else:
            t_f_bool = t_f

        loss_r = factual_mse_loss(y_f, t_f_bool, y_pred_h0, y_pred_h1)
        
        z_y_control = z_y[~t_f_bool]
        z_y_treated = z_y[t_f_bool]
        
        loss_ipm = torch.tensor(0.0, device=self.device)
        if z_y_control.numel() > 0 and z_y_treated.numel() > 0:
            loss_ipm = ipm_loss_zy(z_y_treated, z_y_control, self.sigmas_ipm,
                                   method=self.ipm_method, sinkhorn_eps=self.sinkhorn_eps)
        
        loss_mi = mi_loss_zs_t(z_s, t_f_bool, self.sigmas_mi) # mi_loss_zs_t handles empty z_s internally

        total_loss = loss_r + self.alpha * loss_ipm + self.beta * loss_mi
        
        return total_loss, loss_r, loss_ipm, loss_mi

    def fit(self, x_train, y_train, t_train, num_epochs, batch_size,
            x_val=None, y_val=None, t_val=None, print_every_epochs=10,
            patience=None, lr_scheduler=None):
        """Train the model.  Returns a ``history`` dict with per-epoch losses.

        New optional args (backward-compatible – all default to None/off):
            patience: Early-stopping patience (epochs without val-loss
                improvement).  Requires validation data.  ``None`` = disabled.
            lr_scheduler: ``'plateau'`` for ReduceLROnPlateau, ``'cosine'``
                for CosineAnnealingLR, or ``None`` (no scheduling).
        """
        self.train()

        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)
        t_train = t_train.to(self.device)
        if y_train.ndim == 1: y_train = y_train.unsqueeze(1)
        if t_train.ndim == 1: t_train = t_train.unsqueeze(1)

        train_dataset = TensorDataset(x_train, y_train, t_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if x_val is not None and y_val is not None and t_val is not None:
            x_val = x_val.to(self.device)
            y_val = y_val.to(self.device)
            t_val = t_val.to(self.device)
            if y_val.ndim == 1: y_val = y_val.unsqueeze(1)
            if t_val.ndim == 1: t_val = t_val.unsqueeze(1)
            val_dataset = TensorDataset(x_val, y_val, t_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # LR scheduler
        scheduler = None
        if lr_scheduler == 'plateau' and val_loader:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=max(1, (patience or 10) // 2))
        elif lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)

        # Early stopping state
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_state = None

        keys = ['total', 'R', 'IPM', 'MI']
        history = {'train': {k: [] for k in keys}, 'val': {k: [] for k in keys}}

        for epoch in range(num_epochs):
            self.train()
            ep = {k: 0.0 for k in keys}
            for bx, by, bt in train_loader:
                self.optimizer.zero_grad()
                h0, h1, zy, zs = self.forward(bx)
                tl, lr_, li_, lm_ = self.compute_loss(bx, by, bt, h0, h1, zy, zs)
                tl.backward()
                self.optimizer.step()
                ep['total'] += tl.item(); ep['R'] += lr_.item()
                ep['IPM'] += li_.item(); ep['MI'] += lm_.item()
            nb = len(train_loader)
            for k in keys:
                history['train'][k].append(ep[k] / nb)

            # Validation
            val_total = None
            if val_loader:
                self.eval()
                vep = {k: 0.0 for k in keys}
                with torch.no_grad():
                    for bx, by, bt in val_loader:
                        h0, h1, zy, zs = self.forward(bx)
                        tl, lr_, li_, lm_ = self.compute_loss(bx, by, bt, h0, h1, zy, zs)
                        vep['total'] += tl.item(); vep['R'] += lr_.item()
                        vep['IPM'] += li_.item(); vep['MI'] += lm_.item()
                vnb = len(val_loader)
                for k in keys:
                    history['val'][k].append(vep[k] / vnb)
                val_total = vep['total'] / vnb

            # Scheduler step
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau) and val_total is not None:
                    scheduler.step(val_total)
                elif not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()

            # Early stopping
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
                    if (epoch + 1) % print_every_epochs == 0 or True:
                        print(f"Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.4f})")
                    break

            if (epoch + 1) % print_every_epochs == 0:
                tr = history['train']
                msg = (f"Epoch {epoch+1}/{num_epochs} - Train Loss: {tr['total'][-1]:.4f} "
                       f"(R: {tr['R'][-1]:.4f}, IPM: {tr['IPM'][-1]:.4f}, MI: {tr['MI'][-1]:.4f})")
                print(msg)
                if val_loader:
                    vr = history['val']
                    msg = (f"Epoch {epoch+1}/{num_epochs} - Val Loss: {vr['total'][-1]:.4f} "
                           f"(R: {vr['R'][-1]:.4f}, IPM: {vr['IPM'][-1]:.4f}, MI: {vr['MI'][-1]:.4f})")
                    print(msg)

        return history

    def predict_ite(self, x):
        """
        Predicts the Individual Treatment Effect (ITE) for given input features.

        ITE is estimated as E[Y(1)|x] - E[Y(0)|x], where predictions for Y(1) and Y(0)
        are obtained from the trained prediction heads h1 and h0, respectively.

        Args:
            x (torch.Tensor): Input feature tensor of shape (n_samples, input_dim).

        Returns:
            torch.Tensor: Predicted ITEs, shape (n_samples, output_dim).
        """
        self.eval() # Set model to evaluation mode
        x = x.to(self.device)
        with torch.no_grad():
            z_y, _ = self.encoder(x) # We only need z_y for ITE prediction via heads
            y0_pred = self.h0(z_y)
            y1_pred = self.h1(z_y)
        
        ite_pred = y1_pred - y0_pred
        return ite_pred

    def predict_potential_outcomes(self, x):
        """
        Predicts potential outcomes Y(0) and Y(1) for given input features.

        Args:
            x (torch.Tensor): Input feature tensor of shape (n_samples, input_dim).

        Returns:
            tuple:
                - y0_pred (torch.Tensor): Predicted potential outcomes Y(0),
                                          shape (n_samples, output_dim).
                - y1_pred (torch.Tensor): Predicted potential outcomes Y(1),
                                          shape (n_samples, output_dim).
        """
        self.eval() # Set model to evaluation mode
        x = x.to(self.device)
        with torch.no_grad():
            z_y, _ = self.encoder(x)
            y0_pred = self.h0(z_y)
            y1_pred = self.h1(z_y)
        return y0_pred, y1_pred

# Example Usage (for testing purposes, can be removed or commented out later)
if __name__ == '__main__':
    # Define some example parameters
    input_dim_ex = 25
    hidden_dims_phi_ex = [100, 100]
    latent_dim_zy_ex = 30
    latent_dim_zs_ex = 20
    hidden_dims_h_ex = [50, 50]
    output_dim_ex = 1
    alpha_ex = 0.5
    beta_ex = 0.2
    lr_ex = 0.001
    wd_ex = 0.0001

    # Instantiate the model
    model = DRCFRModel(input_dim_ex, hidden_dims_phi_ex, latent_dim_zy_ex, latent_dim_zs_ex,
                       hidden_dims_h_ex, output_dim_ex, alpha_ex, beta_ex, lr_ex, wd_ex)
    print("DRCFRModel instantiated successfully.")
    print(f"Model is on device: {next(model.parameters()).device}")


    # Create dummy data for training
    num_samples_train = 200
    batch_size_ex = 32
    x_train_ex = torch.randn(num_samples_train, input_dim_ex)
    y_train_ex = torch.randn(num_samples_train, 1)
    t_train_ex = (torch.rand(num_samples_train, 1) > 0.5).float() # Binary treatment

    # Create dummy data for validation
    num_samples_val = 50
    x_val_ex = torch.randn(num_samples_val, input_dim_ex)
    y_val_ex = torch.randn(num_samples_val, 1)
    t_val_ex = (torch.rand(num_samples_val, 1) > 0.5).float()
    
    print("\n--- Testing fit method ---")
    # Test fit method (run for a few epochs)
    model.fit(x_train_ex, y_train_ex, t_train_ex, 
              num_epochs=5, batch_size=batch_size_ex,
              x_val=x_val_ex, y_val=y_val_ex, t_val=t_val_ex,
              print_every_epochs=1)

    print("\n--- Testing predict_ite method ---")
    # Test predict_ite method
    num_samples_test = 10
    x_test_ex = torch.randn(num_samples_test, input_dim_ex)
    ite_predictions = model.predict_ite(x_test_ex)
    print(f"ITE predictions shape: {ite_predictions.shape}") # Expected: [num_samples_test, 1]

    y0_preds, y1_preds = model.predict_potential_outcomes(x_test_ex)
    print(f"Y(0) predictions shape: {y0_preds.shape}")
    print(f"Y(1) predictions shape: {y1_preds.shape}")

    print(f"\nDRCFRModel has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
