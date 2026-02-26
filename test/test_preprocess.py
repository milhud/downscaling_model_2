# test/test_preprocess.py
import sys
import os
import torch

# Dynamically add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import TemperaturePreprocessor

def test_temperature_normalization_bounds():
    """Tests that physical bounds map perfectly to [-1, 1]."""
    preprocessor = TemperaturePreprocessor(t_min=200.0, t_max=320.0)
    
    # Create a tensor with the min, exact middle, and max temperatures
    physical_temps = torch.tensor([200.0, 260.0, 320.0])
    
    normalized = preprocessor.normalize(physical_temps)
    
    # Check expected bounds
    assert torch.allclose(normalized[0], torch.tensor(-1.0)), "Min temp should map to -1.0"
    assert torch.allclose(normalized[1], torch.tensor(0.0)), "Mid temp should map to 0.0"
    assert torch.allclose(normalized[2], torch.tensor(1.0)), "Max temp should map to 1.0"

def test_temperature_denormalization_reversibility():
    """Tests that normalizing and then denormalizing returns the exact original data."""
    preprocessor = TemperaturePreprocessor(t_min=200.0, t_max=320.0)
    
    # Generate random temperatures between 200K and 320K
    original_temps = torch.rand(4, 4) * 120.0 + 200.0
    
    normalized = preprocessor.normalize(original_temps)
    reconstructed_temps = preprocessor.denormalize(normalized)
    
    # Assert they are equal within standard floating point tolerance
    assert torch.allclose(original_temps, reconstructed_temps, atol=1e-5), "Denormalization failed to reconstruct original values."
