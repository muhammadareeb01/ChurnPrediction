"""
PyTorch Tabular Denoising Autoencoder for Advanced Data Imputation
Master Thesis Framework
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

class TabularAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(TabularAutoencoder, self).__init__()
        # Encoder: compress features to learn abstract representations
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LeakyReLU()
        )
        # Decoder: reconstruct original features
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.LeakyReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def pytorch_impute(df_clean, continuous_cols, epochs=30, batch_size=32):
    """
    Trains a Denoising Autoencoder to learn the feature distributions,
    and intelligently predicts the missing values for the continuous columns.
    """
    print(f"    [*] Initializing PyTorch Denoising Autoencoder for {len(continuous_cols)} continuous features...")
    df_imputed = df_clean.copy()
    
    # Extract only continuous columns for the neural network
    data_matrix = df_imputed[continuous_cols].values
    missing_mask = np.isnan(data_matrix)
    
    # Temporarily fill NaNs with column medians to allow tensor operations
    medians = np.nanmedian(data_matrix, axis=0)
    temp_filled_matrix = np.where(missing_mask, np.broadcast_to(medians, data_matrix.shape), data_matrix)
    
    # Standardize data for better neural network convergence
    means = np.mean(temp_filled_matrix, axis=0)
    stds = np.std(temp_filled_matrix, axis=0)
    stds[stds == 0] = 1.0  # Prevent division by zero
    scaled_data = (temp_filled_matrix - means) / stds
    
    # Prepare PyTorch Dataloaders
    tensor_data = torch.FloatTensor(scaled_data)
    dataset = TensorDataset(tensor_data, tensor_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Model, Loss (MSE), and Optimizer
    model = TabularAutoencoder(input_dim=len(continuous_cols))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training Loop
    print(f"    [*] Training Autoencoder on {len(data_matrix)} records for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            # Denoising target: Add random noise to input, ask model to predict clean output
            noise = torch.randn_like(batch_x) * 0.1
            noisy_x = batch_x + noise
            
            optimizer.zero_grad()
            outputs = model(noisy_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    # Inference phase (predict missing values)
    model.eval()
    with torch.no_grad():
        reconstructed_scaled = model(tensor_data).numpy()
        
    # Destandardize exactly back to original feature scale
    reconstructed_data = (reconstructed_scaled * stds) + means
    
    # Map PyTorch predictions ONLY to the missing spots (keeping original known data completely untouched)
    imputation_log = {}
    for idx, col in enumerate(continuous_cols):
        missing_count = int(np.sum(missing_mask[:, idx]))
        if missing_count > 0:
            # Extract predicted values for the NaNs
            predicted_values = reconstructed_data[missing_mask[:, idx], idx]
            # Replace NaNs in df
            df_imputed.loc[missing_mask[:, idx], col] = projected_vals = np.round(predicted_values, 1)
            
            # Log results
            avg_predicted = float(np.mean(projected_vals))
            imputation_log[col] = {
                "missing_filled": missing_count,
                "imputed_value": f"Dyn. PyTorch Avg: {avg_predicted:.1f}"
            }
            print(f"        -> {col}: PyTorch successfully predicted {missing_count} missing values (Avg: {avg_predicted:.1f})")
            
    return df_imputed, imputation_log
