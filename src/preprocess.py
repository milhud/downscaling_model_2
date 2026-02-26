# src/preprocess.py
import torch

class TemperaturePreprocessor:
    """
    Scales temperature data (in Kelvin) to the [-1, 1] range expected by 
    HuggingFace Diffusers, and provides a method to reverse the scaling.
    """
    def __init__(self, t_min: float = 200.0, t_max: float = 320.0):
        # 200K (-73C) to 320K (46C) safely covers typical Earth surface temps
        self.t_min = t_min
        self.t_max = t_max

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Scales input tensor from [t_min, t_max] to [-1, 1]."""
        # First scale to [0, 1]
        tensor_norm = (tensor - self.t_min) / (self.t_max - self.t_min)
        # Then scale to [-1, 1]
        return tensor_norm * 2.0 - 1.0

    def denormalize(self, tensor_norm: torch.Tensor) -> torch.Tensor:
        """Reverts a [-1, 1] scaled tensor back to physical Kelvin values."""
        # First revert to [0, 1]
        tensor = (tensor_norm + 1.0) / 2.0
        # Then revert to [t_min, t_max]
        return tensor * (self.t_max - self.t_min) + self.t_min
