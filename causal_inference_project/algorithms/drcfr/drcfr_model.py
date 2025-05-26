"""
Main Model Class for MIM-DRCFR Algorithm.

This module defines the `DRCFRModel` class, which orchestrates the components
of the Mutual Information Regularized Disentangled Representation for
Counterfactual Regression (MIM-DRCFR) algorithm. It integrates the neural
network architectures (Encoder, PredictionHeads) and the loss functions
to enable training and inference for causal effect estimation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Relative imports for modules within the same package
from .networks import Encoder, PredictionHead
from .losses import factual_mse_loss, ipm_loss_zy, mi_loss_zs_t

class DRCFRModel(nn.Module):
    """
    Mutual Information Regularized Disentangled Representation for Counterfactual Regression (MIM-DRCFR) model.

    This model aims to estimate Individual Treatment Effects (ITEs) by learning
    disentangled representations of input features. It separates features into:
    - `z_y`: A latent representation primarily influencing the outcome.
    - `z_s`: A latent representation primarily influencing the treatment assignment mechanism.

    The model consists of an Encoder, two PredictionHeads (one for control outcome Y(0),
    one for treated outcome Y(1)), and a loss function that combines:
    1. Factual outcome prediction error (MSE).
    2. An Integral Probability Metric (IPM) term (MMD) to balance the distributions
       of `z_y` between treated and control groups.
    3. A Mutual Information (MI) minimization term (MMD-based) to reduce the
       statistical dependency between `z_s` and the treatment assignment `t`.

    Attributes:
        encoder (Encoder): The encoder network phi(x) -> (z_y, z_s).
        h0 (PredictionHead): Prediction head for control outcome Y(0) from z_y.
        h1 (PredictionHead): Prediction head for treated outcome Y(1) from z_y.
        alpha (float): Weight for the IPM loss term (L_IPM).
        beta (float): Weight for the Mutual Information loss term (L_I).
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        sigmas_ipm (list of float): RBF kernel bandwidths for the IPM MMD loss.
        sigmas_mi (list of float): RBF kernel bandwidths for the MI MMD loss.
        device (torch.device): The device (CPU or CUDA) the model is on.
    """
    def __init__(self, input_dim, 
                 hidden_dims_phi, latent_dim_zy, latent_dim_zs, 
                 hidden_dims_h, output_dim=1, 
                 alpha=1.0, beta=1.0, 
                 learning_rate=1e-3, weight_decay=1e-4,
                 sigmas_ipm=None, sigmas_mi=None):
        """
        Initializes the DRCFRModel.

        Args:
            input_dim (int): Dimensionality of the input features `x`.
            hidden_dims_phi (list of int): List of hidden layer sizes for the shared part
                                           of the `Encoder` network. Example: `[100, 100]`.
            latent_dim_zy (int): Dimensionality of the latent representation `z_y` (for outcome prediction).
            latent_dim_zs (int): Dimensionality of the latent representation `z_s` (for treatment mechanism).
            hidden_dims_h (list of int): List of hidden layer sizes for each `PredictionHead` network.
                                         Example: `[50, 50]`.
            output_dim (int, optional): Dimensionality of the predicted outcome by each `PredictionHead`.
                                        Defaults to 1 (for scalar outcomes).
            alpha (float, optional): Weight hyperparameter for the IPM loss term (L_IPM) that balances
                                     the `z_y` distributions. Defaults to 1.0.
            beta (float, optional): Weight hyperparameter for the Mutual Information loss term (L_I)
                                    that encourages independence between `z_s` and treatment `t`.
                                    Defaults to 1.0.
            learning_rate (float, optional): Learning rate for the Adam optimizer. Defaults to 1e-3.
            weight_decay (float, optional): Weight decay (L2 penalty) for the Adam optimizer.
                                            Defaults to 1e-4.
            sigmas_ipm (list of float, optional): List of RBF kernel bandwidths to use for the
                                                  IPM MMD loss calculation (balancing `z_y`).
                                                  Defaults to `[0.1, 1.0, 10.0]`.
            sigmas_mi (list of float, optional): List of RBF kernel bandwidths to use for the
                                                 MI MMD loss calculation (independence of `z_s` and `t`).
                                                 Defaults to `[0.1, 1.0, 10.0]`.
        """
        super(DRCFRModel, self).__init__()

        self.encoder = Encoder(input_dim, hidden_dims_phi, latent_dim_zy, latent_dim_zs)
        self.h0 = PredictionHead(latent_dim_zy, hidden_dims_h, output_dim) # For control group (t=0)
        self.h1 = PredictionHead(latent_dim_zy, hidden_dims_h, output_dim) # For treated group (t=1)

        self.alpha = alpha
        self.beta = beta
        # Store learning_rate and weight_decay for reference, though optimizer is main user
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

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
        
        loss_ipm = torch.tensor(0.0, device=self.device) # Default to 0 if a group is empty
        if z_y_control.numel() > 0 and z_y_treated.numel() > 0: # MMD requires samples from both groups
            loss_ipm = ipm_loss_zy(z_y_treated, z_y_control, self.sigmas_ipm)
        
        loss_mi = mi_loss_zs_t(z_s, t_f_bool, self.sigmas_mi) # mi_loss_zs_t handles empty z_s internally

        total_loss = loss_r + self.alpha * loss_ipm + self.beta * loss_mi
        
        return total_loss, loss_r, loss_ipm, loss_mi

    def fit(self, x_train, y_train, t_train, num_epochs, batch_size, 
            x_val=None, y_val=None, t_val=None, print_every_epochs=10):
        """
        Trains the MIM-DRCFR model using the provided training data.

        Args:
            x_train (torch.Tensor): Training features, shape (n_train_samples, input_dim).
            y_train (torch.Tensor): Training factual outcomes, shape (n_train_samples, output_dim) or (n_train_samples,).
            t_train (torch.Tensor): Training factual treatments (0 or 1), shape (n_train_samples,) or (n_train_samples,1).
            num_epochs (int): Number of epochs to train for.
            batch_size (int): Size of mini-batches for training.
            x_val (torch.Tensor, optional): Validation features. Defaults to None.
            y_val (torch.Tensor, optional): Validation factual outcomes. Defaults to None.
            t_val (torch.Tensor, optional): Validation factual treatments. Defaults to None.
            print_every_epochs (int, optional): Frequency of printing training and validation
                                                loss statistics. Defaults to 10.
        """
        self.train() # Set model to training mode

        # Move data to the model's device
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)
        t_train = t_train.to(self.device)

        # Ensure y_train and t_train have appropriate shapes for DataLoader and loss computation
        if y_train.ndim == 1: y_train = y_train.unsqueeze(1)
        if t_train.ndim == 1: t_train = t_train.unsqueeze(1) # Will be squeezed in compute_loss if needed for bool indexing


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

        for epoch in range(num_epochs):
            epoch_total_loss = 0
            epoch_loss_r = 0
            epoch_loss_ipm = 0
            epoch_loss_mi = 0
            
            for batch_x, batch_y, batch_t in train_loader:
                self.optimizer.zero_grad()
                y_pred_h0, y_pred_h1, z_y, z_s = self.forward(batch_x)
                total_loss, loss_r, loss_ipm, loss_mi = self.compute_loss(
                    batch_x, batch_y, batch_t, y_pred_h0, y_pred_h1, z_y, z_s
                )
                total_loss.backward()
                self.optimizer.step()

                epoch_total_loss += total_loss.item()
                epoch_loss_r += loss_r.item()
                epoch_loss_ipm += loss_ipm.item() 
                epoch_loss_mi += loss_mi.item()

            if (epoch + 1) % print_every_epochs == 0:
                avg_total_loss = epoch_total_loss / len(train_loader)
                avg_loss_r = epoch_loss_r / len(train_loader)
                avg_loss_ipm = epoch_loss_ipm / len(train_loader)
                avg_loss_mi = epoch_loss_mi / len(train_loader)
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_total_loss:.4f} "
                      f"(R: {avg_loss_r:.4f}, IPM: {avg_loss_ipm:.4f}, MI: {avg_loss_mi:.4f})")

                if val_loader:
                    self.eval() # Set model to evaluation mode for validation
                    val_epoch_total_loss = 0
                    val_epoch_loss_r = 0
                    val_epoch_loss_ipm = 0
                    val_epoch_loss_mi = 0
                    with torch.no_grad():
                        for batch_x_val, batch_y_val, batch_t_val in val_loader:
                            y_pred_h0_val, y_pred_h1_val, z_y_val, z_s_val = self.forward(batch_x_val)
                            total_loss_val, loss_r_val, loss_ipm_val, loss_mi_val = self.compute_loss(
                                batch_x_val, batch_y_val, batch_t_val, 
                                y_pred_h0_val, y_pred_h1_val, z_y_val, z_s_val
                            )
                            val_epoch_total_loss += total_loss_val.item()
                            val_epoch_loss_r += loss_r_val.item()
                            val_epoch_loss_ipm += loss_ipm_val.item()
                            val_epoch_loss_mi += loss_mi_val.item()
                    
                    avg_val_total_loss = val_epoch_total_loss / len(val_loader)
                    avg_val_loss_r = val_epoch_loss_r / len(val_loader)
                    avg_val_loss_ipm = val_epoch_loss_ipm / len(val_loader)
                    avg_val_loss_mi = val_epoch_loss_mi / len(val_loader)
                    print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {avg_val_total_loss:.4f} "
                          f"(R: {avg_val_loss_r:.4f}, IPM: {avg_val_loss_ipm:.4f}, MI: {avg_val_loss_mi:.4f})")
                    self.train() # Set back to training mode

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
