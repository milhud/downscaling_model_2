# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.autoencoders.vae import Encoder, Decoder, DiagonalGaussianDistribution

class ClimateTranslatorVAE(ModelMixin, ConfigMixin):
    """
    An Asymmetric VAE mapping low-res ERA5 to high-res WRF CONUS404.
    Fully compatible with HuggingFace Diffusers.
    """
    
    @register_to_config
    def __init__(
        self,
        era5_in_channels: int = 1,   # Defaulting to 1 for Temperature
        wrf_out_channels: int = 1,   
        latent_channels: int = 8,    
        
        # Encoder: 3 blocks = 2 downsample steps (e.g., 32x32 -> 16x16 -> 8x8)
        encoder_block_out_channels: tuple = (64, 128, 256),
        encoder_down_block_types: tuple = (
            "DownEncoderBlock2D", 
            "DownEncoderBlock2D", 
            "DownEncoderBlock2D"
        ),
        
        # Decoder: 5 blocks = 4 upsample steps 
        decoder_block_out_channels: tuple = (256, 128, 64, 32, 32),
        decoder_up_block_types: tuple = (
            "UpDecoderBlock2D", 
            "UpDecoderBlock2D", 
            "UpDecoderBlock2D", 
            "UpDecoderBlock2D", 
            "UpDecoderBlock2D"
        ),
        
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
    ):
        super().__init__()

        # 1. ERA5 Encoder
        self.encoder = Encoder(
            in_channels=era5_in_channels,
            out_channels=encoder_block_out_channels[-1],
            down_block_types=encoder_down_block_types,
            block_out_channels=encoder_block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            double_z=False, 
        )

        # 2. Latent Mappings (Mean & Log-Variance for VAE)
        self.quant_conv = nn.Conv2d(
            encoder_block_out_channels[-1], 
            2 * latent_channels, 
            kernel_size=1
        )

        # Post-latent mapping back to decoder space
        self.post_quant_conv = nn.Conv2d(
            latent_channels, 
            decoder_block_out_channels[0], 
            kernel_size=1
        )

        # 3. WRF CONUS404 Decoder
        self.decoder = Decoder(
            in_channels=decoder_block_out_channels[0],
            out_channels=wrf_out_channels,
            up_block_types=decoder_up_block_types,
            block_out_channels=decoder_block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
        )

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        """Compresses ERA5 data into the diffusion latent space."""
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z: torch.Tensor, target_shape: tuple = None) -> torch.Tensor:
        """Expands diffusion latent space into WRF CONUS404 data."""
        z = self.post_quant_conv(z)
        wrf_reconstruction = self.decoder(z)
        
        # Dynamically interpolate to match the exact CONUS grid size
        if target_shape is not None:
            wrf_reconstruction = F.interpolate(
                wrf_reconstruction, 
                size=target_shape, 
                mode="bilinear", 
                align_corners=False
            )
            
        return wrf_reconstruction

    def forward(self, x: torch.Tensor, target_shape: tuple = None, sample_posterior: bool = True) -> tuple:
        """
        Forward pass for training. 
        Returns the WRF prediction and the posterior for the KL-divergence loss.
        """
        posterior = self.encode(x)
        
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
            
        wrf_pred = self.decode(z, target_shape=target_shape)
        return wrf_pred, posterior
