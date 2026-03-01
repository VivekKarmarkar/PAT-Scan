"""
U-Net Architecture with Mexican Hat Wavelet Activation
"""

import torch
import torch.nn as nn


class MexicanHatActivation(nn.Module):
    """
    Mexican Hat (Ricker) Wavelet Activation Function
    ψ(x) = (1 - x²)exp(-x²/2)
    
    With learnable scale and shift parameters:
    ψ(x) = (1 - ((x - shift)/scale)²) * exp(-((x - shift)/scale)²/2)
    """
    def __init__(self, num_features=None):
        super(MexicanHatActivation, self).__init__()
        # Learnable parameters for scale and shift
        # Initialize scale to 1.0 and shift to 0.0
        if num_features is not None:
            # Channel-wise parameters
            self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.shift = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            # Global parameters
            self.scale = nn.Parameter(torch.ones(1))
            self.shift = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        # Normalize input
        x_normalized = (x - self.shift) / (self.scale + 1e-8)
        
        # Mexican hat wavelet: (1 - x²)exp(-x²/2)
        x_squared = x_normalized ** 2
        wavelet = (1.0 - x_squared) * torch.exp(-x_squared / 2.0)
        
        return wavelet


def double_conv(in_c, out_c):
    """Double convolution block with Mexican Hat Wavelet activation"""
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        MexicanHatActivation(num_features=out_c),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        MexicanHatActivation(num_features=out_c)
    )
    return conv


class UNet(nn.Module):
    """U-Net for material property segmentation with Wavelet Activations"""
    def __init__(self, in_channels=2, out_channels=1, base_features=32):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = double_conv(in_channels, base_features)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = double_conv(base_features, base_features*2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = double_conv(base_features*2, base_features*4)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = double_conv(base_features*4, base_features*8)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(base_features*8, base_features*4, kernel_size=2, stride=2)
        self.dec3 = double_conv(base_features*8, base_features*4)
        
        self.upconv2 = nn.ConvTranspose2d(base_features*4, base_features*2, kernel_size=2, stride=2)
        self.dec2 = double_conv(base_features*4, base_features*2)
        
        self.upconv1 = nn.ConvTranspose2d(base_features*2, base_features, kernel_size=2, stride=2)
        self.dec1 = double_conv(base_features*2, base_features)
        
        # Output
        self.out = nn.Conv2d(base_features, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        # Decoder with skip connections
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output
        out = self.out(dec1)
        return self.sigmoid(out)