# train.py
import sys
import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.dataloader import get_dataloader
from src.preprocess import TemperaturePreprocessor
from src.model import ClimateTranslatorVAE

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on {device} | Test Mode: {args.test}")

    # 1. Initialize Dataset & Preprocessor
    # If testing, just use one year to keep the xarray lazy load extremely fast
    years = [1980] if args.test else range(1980, 2021)
    dataloader = get_dataloader(data_dir='./data', years=years, batch_size=args.batch_size)
    preprocessor = TemperaturePreprocessor()

    # 2. Initialize Model
    model = ClimateTranslatorVAE(era5_in_channels=1, wrf_out_channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 3. Training Loop Setup
    epochs = 2 if args.test else args.epochs
    max_batches = 50 if args.test else len(dataloader) # Limit batches if testing
    beta = 0.01 # KL Divergence weighting

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Create progress bar
        pbar = tqdm(enumerate(dataloader), total=max_batches, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in pbar:
            if args.test and batch_idx >= max_batches:
                break # Exit early in test mode
                
            era5_data = preprocessor.normalize(batch["era5"].to(device))
            conus_data = preprocessor.normalize(batch["conus"].to(device))

            # Forward
            optimizer.zero_grad()
            wrf_pred, posterior = model(era5_data)

            # Loss
            recon_loss = F.mse_loss(wrf_pred, conus_data)
            kl_loss = posterior.kl().mean()
            loss = recon_loss + beta * kl_loss

            # Backward
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            # Update progress bar with current loss to visually verify it goes down
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Recon": f"{recon_loss.item():.4f}"})

        avg_loss = epoch_loss / max_batches
        print(f"End of Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")

    # Save the model
    os.makedirs("./checkpoints", exist_ok=True)
    model.save_pretrained("./checkpoints/climate_vae_latest")
    print("Training complete. Model saved to ./checkpoints/climate_vae_latest")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Asymmetric Climate VAE")
    parser.add_argument("--test", action="store_true", help="Run a quick test to verify loss reduction.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    
    args = parser.parse_args()
    train(args)
