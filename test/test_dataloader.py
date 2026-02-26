# test/test_dataloader.py
import sys
import os
import pytest
import torch
import numpy as np
import xarray as xr

# Dynamically add the parent directory to sys.path so 'src' can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataloader import ClimateDataset, get_dataloader

@pytest.fixture
def mock_climate_data(tmp_path):
    """Creates dummy NetCDF files for the test suite."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    time_idx = xr.DataArray(np.arange(10), dims=["time"], name="time")
    
    era5_ds = xr.Dataset(
        {"T2": (("time", "lat", "lon"), np.random.rand(10, 32, 32))}, 
        coords={"time": time_idx}
    )
    
    conus_ds = xr.Dataset(
        {"t2m": (("time", "lat", "lon"), np.random.rand(10, 256, 256))}, 
        coords={"time": time_idx}
    )
    
    era5_ds.to_netcdf(data_dir / "era5_1980.nc")
    conus_ds.to_netcdf(data_dir / "conus404_yearly_1980.nc")
    
    return str(data_dir)

def test_dataset_loading_and_shapes(mock_climate_data):
    dataset = ClimateDataset(data_dir=mock_climate_data, years=[1980])
    assert len(dataset) == 10
    
    sample = dataset[0]
    assert sample["era5"].shape == (1, 32, 32)
    assert sample["conus"].shape == (1, 256, 256)

def test_dataloader_batching(mock_climate_data):
    dataloader = get_dataloader(data_dir=mock_climate_data, years=[1980], batch_size=2)
    batch = next(iter(dataloader))
    
    assert batch["era5"].shape == (2, 1, 32, 32)
    assert batch["conus"].shape == (2, 1, 256, 256)
