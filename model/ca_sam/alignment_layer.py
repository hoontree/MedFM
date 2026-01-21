"""
Alignment Layer Implementation for CA-SAM

논문의 핵심 컴포넌트: SAM encoder와 decoder 사이에서 
feature distribution을 의료 도메인에 맞게 정렬하는 경량 모듈
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CAResBlock(nn.Module):
    """
    Channel Attention Residual Block
    
    구성:
    - 두 개의 3x3 convolution layers
    - Channel attention mechanism (global average pooling + 1D conv)
    - Residual connection
    - LayerNorm for stability
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Main convolution path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Channel attention
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.channel_attention = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        
        # Residual connection (if dimensions match)
        self.residual = nn.Identity() if in_channels == out_channels else \
                       nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        # Layer Normalization for stable training
        self.layer_norm = nn.GroupNorm(1, out_channels)  # GroupNorm with 1 group = LayerNorm2d
        
    def forward(self, x):
        identity = self.residual(x)
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Channel attention
        B, C, H, W = out.shape
        att = self.gap(out)  # [B, C, 1, 1]
        att = att.squeeze(-1).permute(0, 2, 1)  # [B, 1, C]
        att = self.channel_attention(att)  # [B, 1, C]
        att = att.permute(0, 2, 1).unsqueeze(-1)  # [B, C, 1, 1]
        
        out = out * att
        
        # Residual connection
        out = out + identity
        out = self.relu(out)
        
        # Layer normalization
        out = self.layer_norm(out)
        
        return out


class AlignmentLayer(nn.Module):
    """
    Alignment Layer: 여러 CAResBlock을 stack하여 feature distribution 정렬
    
    Args:
        in_channels: SAM encoder의 출력 채널 수 (default: 256 for ViT-B)
        hidden_channels: 중간 hidden layer의 채널 수
        num_blocks: CAResBlock의 개수 (논문: 3-5개)
    """
    def __init__(
        self, 
        in_channels: int = 256,
        hidden_channels: int = 256,
        num_blocks: int = 4
    ):
        super().__init__()
        
        self.num_blocks = num_blocks
        
        # Initial projection (if needed)
        self.input_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1) \
                         if in_channels != hidden_channels else nn.Identity()
        
        # Stack of CAResBlocks
        blocks = []
        for i in range(num_blocks):
            blocks.append(CAResBlock(hidden_channels, hidden_channels))
        self.blocks = nn.Sequential(*blocks)
        
        # Output projection (if needed)
        self.output_proj = nn.Conv2d(hidden_channels, in_channels, kernel_size=1) \
                          if in_channels != hidden_channels else nn.Identity()
        
    def forward(self, x):
        """
        Args:
            x: SAM encoder output features [B, C, H, W]
            
        Returns:
            Aligned features [B, C, H, W]
        """
        out = self.input_proj(x)
        out = self.blocks(out)
        out = self.output_proj(out)
        return out
    
    def get_num_params(self):
        """모델의 파라미터 수 계산"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class IdentityAlignmentLayer(nn.Module):
    """
    Identity Alignment Layer for OOD fallback
    OOD 샘플에 대해 frozen SAM으로 zero-shot inference 수행
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x
    
    def get_num_params(self):
        return 0


if __name__ == "__main__":
    # Test Alignment Layer
    print("=" * 60)
    print("Alignment Layer Test")
    print("=" * 60)
    
    # SAM ViT-B encoder output: [B, 256, 64, 64] (for 1024x1024 input)
    batch_size = 2
    channels = 256
    height, width = 64, 64
    
    x = torch.randn(batch_size, channels, height, width)
    
    # Test with different number of blocks
    for num_blocks in [2, 3, 4, 5]:
        print(f"\n{num_blocks} blocks:")
        alignment_layer = AlignmentLayer(
            in_channels=channels,
            hidden_channels=channels,
            num_blocks=num_blocks
        )
        
        output = alignment_layer(x)
        num_params = alignment_layer.get_num_params()
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Number of parameters: {num_params:,}")
        print(f"  Parameters (M): {num_params / 1e6:.2f}M")
        
    # Test Identity Alignment Layer
    print("\n" + "=" * 60)
    print("Identity Alignment Layer (for OOD)")
    identity_layer = IdentityAlignmentLayer()
    output = identity_layer(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Number of parameters: {identity_layer.get_num_params()}")
