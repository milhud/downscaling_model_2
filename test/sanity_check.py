# test/sanity_check.py
import sys
import os
import torch
import torch.nn.functional as F
import torch.optim as optim

# Go up one level from 'test/' to the root directory so 'src' can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataloader import get_dataloader
from src.preprocess import TemperaturePreprocessor
from src.model import ClimateTranslatorVAE

def run_sanity_check():
    print("--- Starting Pipeline Sanity Check ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/6] Using device: {device}")

    try:
        # Initialize components
        preprocessor = TemperaturePreprocessor(t_min=200.0, t_max=320.0)
        
        model = ClimateTranslatorVAE(
            era5_in_channels=1,
            wrf_out_channels=1,
            latent_channels=8
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        print("[2/6] Model, Preprocessor, and Optimizer initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize components: {e}")
        return

    try:
        # Load exactly one batch of real data (using just 1980 to be fast)
        dataloader = get_dataloader(data_dir='./data', years=[1980], batch_size=2, shuffle=True)
        batch = next(iter(dataloader))
        
        era5_raw = batch["era5"].to(device)
        conus_raw = batch["conus"].to(device)
        print(f"[3/6] Data loaded. ERA5 shape: {era5_raw.shape}, CONUS shape: {conus_raw.shape}")
    except Exception as e:
        print(f"Failed to load data. Ensure ./data contains 1980 .nc files. Error: {e}")
        return

    try:
        # Preprocess
        era5_norm = preprocessor.normalize(era5_raw)
        conus_norm = preprocessor.normalize(conus_raw)
        print("[4/6] Data normalized successfully.")
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return

    try:
        # Forward pass and Loss
        optimizer.zero_grad()
        
        # Extract target shape (Height, Width) from the CONUS tensor
        target_shape = conus_norm.shape[2:] 
        
        # Pass the target shape into the model
        wrf_pred, posterior = model(era5_norm, target_shape=target_shape)
        
        # Calculate standard VAE Loss: MSE + beta * KL Divergence
        recon_loss = F.mse_loss(wrf_pred, conus_norm)
        kl_loss = posterior.kl().mean()
        beta = 0.01 # KL weight
        total_loss = recon_loss + beta * kl_loss
        
        print(f"[5/6] Forward pass complete. Recon Loss: {recon_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}")
    except Exception as e:
        print(f"Forward pass or loss calculation failed: {e}")
        return

    try:
        # Backward pass and Optimization
        total_loss.backward()
        
        # Check if gradients exist and weights update
        initial_weight = model.encoder.conv_in.weight.clone()
        optimizer.step()
        updated_weight = model.encoder.conv_in.weight.clone()
        
        if torch.equal(initial_weight, updated_weight):
            raise RuntimeError("Weights did not update after optimizer.step(). The computation graph might be detached.")
            
        print("[6/6] Backward pass successful. Gradients computed and weights updated.")
        print("\n✅ SANITY CHECK PASSED! The model is ready for training.")
    except Exception as e:
        print(f"Backward pass failed: {e}")
        return

if __name__ == "__main__":
    run_sanity_check()
