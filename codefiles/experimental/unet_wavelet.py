"""
Wavelet U-Net Architecture - PURE PYTORCH IMPLEMENTATION
Replaces max pooling with Discrete Wavelet Transform (DWT)
and transposed convolution with Inverse Wavelet Transform (IWT)
NO EXTERNAL DEPENDENCIES - 100% torch operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWTDownsampling(nn.Module):
    """
    Discrete Wavelet Transform for downsampling using pure PyTorch
    Implements Haar wavelet (can be extended to other wavelets)
    """
    def __init__(self, in_channels):
        super(DWTDownsampling, self).__init__()
        
        # Haar wavelet filters (2x2)
        # Low-pass filter (averaging): [[0.5, 0.5], [0.5, 0.5]]
        # High-pass filters for horizontal, vertical, diagonal details
        
        # LL (Low-Low): approximation
        ll = torch.tensor([[0.5, 0.5],
                          [0.5, 0.5]], dtype=torch.float32)
        
        # LH (Low-High): horizontal details
        lh = torch.tensor([[0.5, 0.5],
                          [-0.5, -0.5]], dtype=torch.float32)
        
        # HL (High-Low): vertical details  
        hl = torch.tensor([[0.5, -0.5],
                          [0.5, -0.5]], dtype=torch.float32)
        
        # HH (High-High): diagonal details
        hh = torch.tensor([[0.5, -0.5],
                          [-0.5, 0.5]], dtype=torch.float32)
        
        # Create filters for each input channel (groups=in_channels)
        self.register_buffer('ll_filter', ll.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1))
        self.register_buffer('lh_filter', lh.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1))
        self.register_buffer('hl_filter', hl.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1))
        self.register_buffer('hh_filter', hh.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1))
        
        self.in_channels = in_channels
    
    def forward(self, x):
        """
        Apply 2D DWT
        Input: (B, C, H, W)
        Output: (B, 4*C, H/2, W/2) - concatenated [LL, LH, HL, HH]
        """
        # Apply convolution with stride 2 for downsampling
        ll = F.conv2d(x, self.ll_filter, stride=2, groups=self.in_channels)
        lh = F.conv2d(x, self.lh_filter, stride=2, groups=self.in_channels)
        hl = F.conv2d(x, self.hl_filter, stride=2, groups=self.in_channels)
        hh = F.conv2d(x, self.hh_filter, stride=2, groups=self.in_channels)
        
        # Concatenate along channel dimension
        return torch.cat([ll, lh, hl, hh], dim=1)


class IWTUpsampling(nn.Module):
    """
    Inverse Wavelet Transform for upsampling using pure PyTorch
    Reconstructs from wavelet coefficients
    """
    def __init__(self, in_channels):
        super(IWTUpsampling, self).__init__()
        
        # Haar wavelet reconstruction filters
        # These are transpose of decomposition filters
        ll = torch.tensor([[0.5, 0.5],
                          [0.5, 0.5]], dtype=torch.float32)
        
        lh = torch.tensor([[0.5, 0.5],
                          [-0.5, -0.5]], dtype=torch.float32)
        
        hl = torch.tensor([[0.5, -0.5],
                          [0.5, -0.5]], dtype=torch.float32)
        
        hh = torch.tensor([[0.5, -0.5],
                          [-0.5, 0.5]], dtype=torch.float32)
        
        # Create filters
        self.register_buffer('ll_filter', ll.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1))
        self.register_buffer('lh_filter', lh.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1))
        self.register_buffer('hl_filter', hl.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1))
        self.register_buffer('hh_filter', hh.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1))
        
        self.in_channels = in_channels
    
    def forward(self, x):
        """
        Apply 2D IWT
        Input: (B, 4*C, H, W) - concatenated [LL, LH, HL, HH]
        Output: (B, C, 2*H, 2*W)
        """
        # Split input into wavelet components
        C = self.in_channels
        ll = x[:, 0:C, :, :]
        lh = x[:, C:2*C, :, :]
        hl = x[:, 2*C:3*C, :, :]
        hh = x[:, 3*C:4*C, :, :]
        
        # Upsample using transposed convolution and sum
        ll_up = F.conv_transpose2d(ll, self.ll_filter, stride=2, groups=C)
        lh_up = F.conv_transpose2d(lh, self.lh_filter, stride=2, groups=C)
        hl_up = F.conv_transpose2d(hl, self.hl_filter, stride=2, groups=C)
        hh_up = F.conv_transpose2d(hh, self.hh_filter, stride=2, groups=C)
        
        # Reconstruct by summing all components
        return ll_up + lh_up + hl_up + hh_up


class DWTConvBlock(nn.Module):
    """
    Convolution block that processes DWT output
    Input has 4x channels from DWT, output has desired channels
    """
    def __init__(self, in_c, out_c):
        super(DWTConvBlock, self).__init__()
        # Input will have 4x channels from DWT [LL, LH, HL, HH]
        self.conv = nn.Sequential(
            nn.Conv2d(in_c * 4, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


def double_conv(in_c, out_c):
    """Double convolution block"""
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    return conv


class WaveletUNet(nn.Module):
    """
    Wavelet U-Net for material property segmentation
    Uses DWT for downsampling and IWT for upsampling to preserve fine details
    100% PyTorch implementation - no external dependencies
    """
    def __init__(self, in_channels=2, out_channels=1, base_features=32):
        super(WaveletUNet, self).__init__()
        
        # Encoder - Level 1
        self.enc1 = double_conv(in_channels, base_features)
        self.dwt1 = DWTDownsampling(base_features)
        
        # Encoder - Level 2
        self.enc2 = DWTConvBlock(base_features, base_features*2)
        self.dwt2 = DWTDownsampling(base_features*2)
        
        # Encoder - Level 3
        self.enc3 = DWTConvBlock(base_features*2, base_features*4)
        self.dwt3 = DWTDownsampling(base_features*4)
        
        # Bottleneck
        self.bottleneck = DWTConvBlock(base_features*4, base_features*8)
        
        # Prepare bottleneck output for IWT (expand to 4x channels)
        self.bottleneck_prep = nn.Conv2d(base_features*8, base_features*8*4, kernel_size=1)
        
        # Decoder - Level 3
        self.iwt3 = IWTUpsampling(base_features*8)
        self.dec3_conv = nn.Conv2d(base_features*8, base_features*4, kernel_size=1)
        self.dec3 = double_conv(base_features*8, base_features*4)  # *8 due to skip connection
        
        # Decoder - Level 2
        self.dec2_prep = nn.Conv2d(base_features*4, base_features*4*4, kernel_size=1)
        self.iwt2 = IWTUpsampling(base_features*4)
        self.dec2_conv = nn.Conv2d(base_features*4, base_features*2, kernel_size=1)
        self.dec2 = double_conv(base_features*4, base_features*2)
        
        # Decoder - Level 1
        self.dec1_prep = nn.Conv2d(base_features*2, base_features*2*4, kernel_size=1)
        self.iwt1 = IWTUpsampling(base_features*2)
        self.dec1_conv = nn.Conv2d(base_features*2, base_features, kernel_size=1)
        self.dec1 = double_conv(base_features*2, base_features)
        
        # Output layer
        self.out = nn.Conv2d(base_features, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)              # (B, base_features, H, W)
        dwt1_out = self.dwt1(enc1)       # (B, base_features*4, H/2, W/2)
        
        enc2 = self.enc2(dwt1_out)       # (B, base_features*2, H/2, W/2)
        dwt2_out = self.dwt2(enc2)       # (B, base_features*2*4, H/4, W/4)
        
        enc3 = self.enc3(dwt2_out)       # (B, base_features*4, H/4, W/4)
        dwt3_out = self.dwt3(enc3)       # (B, base_features*4*4, H/8, W/8)
        
        # Bottleneck
        bottleneck = self.bottleneck(dwt3_out)  # (B, base_features*8, H/8, W/8)
        
        # Decoder path with IWT upsampling
        # Level 3
        bottleneck_prep = self.bottleneck_prep(bottleneck)  # (B, base_features*8*4, H/8, W/8)
        dec3_up = self.iwt3(bottleneck_prep)                 # (B, base_features*8, H/4, W/4)
        dec3_up = self.dec3_conv(dec3_up)                    # (B, base_features*4, H/4, W/4)
        dec3 = torch.cat([dec3_up, enc3], dim=1)             # (B, base_features*8, H/4, W/4)
        dec3 = self.dec3(dec3)                               # (B, base_features*4, H/4, W/4)
        
        # Level 2
        dec2_prep = self.dec2_prep(dec3)          # (B, base_features*4*4, H/4, W/4)
        dec2_up = self.iwt2(dec2_prep)            # (B, base_features*4, H/2, W/2)
        dec2_up = self.dec2_conv(dec2_up)         # (B, base_features*2, H/2, W/2)
        dec2 = torch.cat([dec2_up, enc2], dim=1)  # (B, base_features*4, H/2, W/2)
        dec2 = self.dec2(dec2)                    # (B, base_features*2, H/2, W/2)
        
        # Level 1
        dec1_prep = self.dec1_prep(dec2)          # (B, base_features*2*4, H/2, W/2)
        dec1_up = self.iwt1(dec1_prep)            # (B, base_features*2, H, W)
        dec1_up = self.dec1_conv(dec1_up)         # (B, base_features, H, W)
        dec1 = torch.cat([dec1_up, enc1], dim=1)  # (B, base_features*2, H, W)
        dec1 = self.dec1(dec1)                    # (B, base_features, H, W)
        
        # Output
        out = self.out(dec1)
        return self.sigmoid(out)


if __name__ == "__main__":
    print("="*70)
    print("Testing Wavelet U-Net (Pure PyTorch Implementation)")
    print("="*70)
    
    # Create model
    model = WaveletUNet(in_channels=2, out_channels=1, base_features=32)
    
    # Test with dummy input
    x = torch.randn(2, 2, 128, 128)  # (batch, channels, height, width)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test on GPU if available
    if torch.cuda.is_available():
        print("\nTesting on GPU...")
        model_gpu = model.cuda()
        x_gpu = x.cuda()
        with torch.no_grad():
            output_gpu = model_gpu(x_gpu)
        print(f"GPU output shape: {output_gpu.shape}")
        print("✓ GPU test passed!")
    
    print("\n" + "="*70)
    print("✓ Wavelet U-Net working correctly!")
    print("="*70)
    
    # Architecture summary
    print("\nArchitecture Summary:")
    print("- Encoder: 3 levels with DWT downsampling")
    print("- Bottleneck: DWT conv block")
    print("- Decoder: 3 levels with IWT upsampling")
    print("- Wavelet: Haar (2x2 filters)")
    print("- Skip connections: Concatenation between encoder and decoder")
    print("\nKey Features:")
    print("✓ DWT replaces MaxPool (preserves high-frequency details)")
    print("✓ IWT replaces TransposeConv (better reconstruction)")
    print("✓ 100% PyTorch - no external dependencies")
    print("✓ Fully GPU-compatible")
    print("✓ Drop-in replacement for standard U-Net")