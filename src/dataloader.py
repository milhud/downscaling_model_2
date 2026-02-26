# src/dataloader.py
import os
import torch
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader
import warnings

# Suppress standard numpy warnings about calculating the mean of empty/NaN slices,
# which is common and expected around the boundaries of climate data maps.
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

class ClimateDataset(Dataset):
    """
    PyTorch Dataset for pairing ERA5 and WRF CONUS404 NetCDF data.
    Lazily loads data using xarray to minimize memory overhead and dynamically
    handles dimension mismatches.
    """
    def __init__(self, data_dir='./data', years=None):
        self.data_dir = data_dir
        
        if years is None:
            years = range(1980, 2021)
            
        era5_files = []
        conus_files = []
        
        # Discover and pair files
        for year in years:
            era_path = os.path.join(data_dir, f"era5_{year}.nc")
            conus_path = os.path.join(data_dir, f"conus404_yearly_{year}.nc")
            
            if os.path.exists(era_path) and os.path.exists(conus_path):
                era5_files.append(era_path)
                conus_files.append(conus_path)
            else:
                print(f"Warning: Missing paired data for year {year}. Skipping.")
        
        if not era5_files:
            raise ValueError(f"No matching data files found in {data_dir}.")
        
        # Open multi-file datasets lazily
        self.era5_data = xr.open_mfdataset(era5_files, combine='by_coords', engine='netcdf4')
        self.conus_data = xr.open_mfdataset(conus_files, combine='by_coords', engine='netcdf4')
        
        # Isolate the targeted temperature variables
        self.era5_t2 = self.era5_data['t2m']
        self.conus_t2m = self.conus_data['T2']
        
        # Dynamically identify the time dimension (prioritize 'valid_time' if it exists)
        self.era5_time_dim = 'valid_time' if 'valid_time' in self.era5_t2.dims else 'time'
        self.conus_time_dim = 'valid_time' if 'valid_time' in self.conus_t2m.dims else 'time'
        
        # Verify temporal alignment
        self._verify_alignment()
        
        # Determine total length by the correct time dimension
        self.time_dim_length = len(self.era5_t2[self.era5_time_dim])

    def _verify_alignment(self):
        """Checks if the loaded datasets correctly align in length."""
        era_len = len(self.era5_data[self.era5_time_dim])
        conus_len = len(self.conus_data[self.conus_time_dim])
        
        if era_len != conus_len:
            raise ValueError(
                f"Time dimension mismatch: ERA5 has {era_len} steps (using '{self.era5_time_dim}'), "
                f"but CONUS404 has {conus_len} steps (using '{self.conus_time_dim}')."
            )
            
        print(f"Data alignment verified: Temporal dimensions match ({era_len} steps).")

    def __len__(self):
        return self.time_dim_length

    def __getitem__(self, idx):
        """Fetches a single paired time-step, formats dimensions, and converts to a PyTorch tensor."""
        # Dynamically slice using the correct time dimension name
        era5_step = self.era5_t2.isel({self.era5_time_dim: idx}).values.astype(np.float32)
        conus_step = self.conus_t2m.isel({self.conus_time_dim: idx}).values.astype(np.float32)
        
        # If ERA5 has extra dimensions (e.g., 12 intra-day steps), average them to a single 2D daily grid
        if era5_step.ndim > 2:
            axes_to_mean = tuple(range(era5_step.ndim - 2))
            era5_step = np.nanmean(era5_step, axis=axes_to_mean)
            
        # Do the same for CONUS just to be completely safe against unexpected dimensions
        if conus_step.ndim > 2:
            axes_to_mean = tuple(range(conus_step.ndim - 2))
            conus_step = np.nanmean(conus_step, axis=axes_to_mean)
        
        # Add channel dimensions (C, H, W) for HuggingFace diffusers
        # Shape goes from (Lat, Lon) -> (1, Lat, Lon)
        era5_tensor = torch.from_numpy(era5_step).unsqueeze(0)
        conus_tensor = torch.from_numpy(conus_step).unsqueeze(0)
        
        # Clean up any residual NaNs (e.g., from ocean boundaries) to prevent loss explosions
        era5_tensor = torch.nan_to_num(era5_tensor, nan=0.0)
        conus_tensor = torch.nan_to_num(conus_tensor, nan=0.0)
        
        return {
            "era5": era5_tensor, 
            "conus": conus_tensor
        }
        
def get_dataloader(data_dir='./data', years=None, batch_size=4, shuffle=True, num_workers=2):
    """Utility function to spin up the DataLoader."""
    dataset = ClimateDataset(data_dir=data_dir, years=years)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
