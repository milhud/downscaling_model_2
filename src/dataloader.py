# src/dataloader.py
import os
import torch
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader

class ClimateDataset(Dataset):
    """
    PyTorch Dataset for pairing ERA5 and WRF CONUS404 NetCDF data.
    Lazily loads data using xarray to minimize memory overhead.
    """
    def __init__(self, data_dir='./data', years=None):
        self.data_dir = data_dir
        
        if years is None:
            # Generate the list of years based on your directory (1980 - 2020)
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
        
        # Verify temporal alignment
        self._verify_alignment()
        
        # Isolate the targeted temperature variables
        self.era5_t2 = self.era5_data['T2']
        self.conus_t2m = self.conus_data['t2m']
        
        # Determine total length by the time dimension
        self.time_dim_length = len(self.era5_t2.time)

    def _verify_alignment(self):
        """Checks if the loaded datasets correctly align in time."""
        era_times = self.era5_data.time.values
        conus_times = self.conus_data.time.values
        
        # 1. Check total lengths
        if len(era_times) != len(conus_times):
            raise ValueError(
                f"Time dimension mismatch: ERA5 has {len(era_times)} steps, "
                f"but CONUS404 has {len(conus_times)} steps."
            )
        
        # 2. Check bounds to ensure they represent the same temporal window
        if era_times[0] != conus_times[0] or era_times[-1] != conus_times[-1]:
            raise ValueError("Time bounds mismatch! The datasets do not start/end on the same timestamps.")
            
        print("Data alignment verified: Temporal dimensions perfectly match.")

    def __len__(self):
        return self.time_dim_length

    def __getitem__(self, idx):
        """Fetches a single paired time-step and converts it to a PyTorch tensor."""
        # .values forces the lazy loader to fetch the data into memory
        era5_step = self.era5_t2.isel(time=idx).values.astype(np.float32)
        conus_step = self.conus_t2m.isel(time=idx).values.astype(np.float32)
        
        # Add channel dimensions (C, H, W) for HuggingFace diffusers
        era5_tensor = torch.from_numpy(era5_step).unsqueeze(0)
        conus_tensor = torch.from_numpy(conus_step).unsqueeze(0)
        
        return {
            "era5": era5_tensor, 
            "conus": conus_tensor
        }
        
def get_dataloader(data_dir='./data', years=None, batch_size=4, shuffle=True, num_workers=2):
    """Utility function to spin up the DataLoader."""
    dataset = ClimateDataset(data_dir=data_dir, years=years)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
