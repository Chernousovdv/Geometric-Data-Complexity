"""
U-Net for MNIST diffusion model.

Architecture for 28x28 grayscale images with time conditioning.
Predicts noise eps given noisy image x_t and time t.

Encoder:  28x28 -> 14x14 -> 7x7
Middle:   self-attention at 7x7
Decoder:  7x7 -> 14x14 -> 28x28  (with skip connections)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion time t in [0, 1]."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / half
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """Residual block with FiLM time conditioning (scale + shift)."""

    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        # FiLM: project time embedding to scale and shift (2 * out_ch)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2),
        )
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        # FiLM conditioning: scale and shift after first conv
        scale_shift = self.time_mlp(t_emb)[:, :, None, None]
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention for 2D feature maps."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + h


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for 28x28 grayscale images.

    Encoder: [1] -> [ch] -> [ch] -> down -> [2ch] -> [2ch] -> down -> [4ch] -> [4ch]
    Middle:  [4ch] -> attention -> [4ch]
    Decoder: mirror with skip connections

    Args:
        in_channels: input image channels (1 for MNIST)
        base_channels: base channel count (default 64)
        channel_mults: channel multipliers per resolution level
        dropout: dropout rate in residual blocks
    """

    def __init__(self, in_channels=1, base_channels=32,
                 channel_mults=(1, 2, 4), dropout=0.1):
        super().__init__()

        time_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Initial projection
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch_in = base_channels
        self.skip_channels = [base_channels]

        for i, mult in enumerate(channel_mults):
            ch_out = base_channels * mult
            self.encoder_blocks.append(ResBlock(ch_in, ch_out, time_dim, dropout))
            self.encoder_blocks.append(ResBlock(ch_out, ch_out, time_dim, dropout))
            self.skip_channels.extend([ch_out, ch_out])
            ch_in = ch_out

            if i < len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch_out))
                self.skip_channels.append(ch_out)
            else:
                self.downsamples.append(nn.Identity())

        # Middle
        self.mid_block1 = ResBlock(ch_in, ch_in, time_dim, dropout)
        self.mid_attn = SelfAttention(ch_in)
        self.mid_block2 = ResBlock(ch_in, ch_in, time_dim, dropout)

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_mults))):
            ch_out = base_channels * mult

            # Three ResBlocks per level (to consume skip connections)
            for j in range(3):
                skip_ch = self.skip_channels.pop()
                self.decoder_blocks.append(
                    ResBlock(ch_in + skip_ch, ch_out, time_dim, dropout)
                )
                ch_in = ch_out

            if i > 0:
                self.upsamples.append(Upsample(ch_out))
            else:
                self.upsamples.append(nn.Identity())

        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, ch_in),
            nn.SiLU(),
            nn.Conv2d(ch_in, in_channels, 3, padding=1),
        )

    def forward(self, x, t):
        """
        Args:
            x: (B, 1, 28, 28) noisy image
            t: (B,) diffusion time in [0, 1]

        Returns:
            eps_pred: (B, 1, 28, 28) predicted noise
        """
        t_emb = self.time_embed(t)
        h = self.conv_in(x)

        # Encoder with skip connections
        skips = [h]
        block_idx = 0
        for i in range(len(self.downsamples)):
            h = self.encoder_blocks[block_idx](h, t_emb)
            skips.append(h)
            block_idx += 1
            h = self.encoder_blocks[block_idx](h, t_emb)
            skips.append(h)
            block_idx += 1
            h = self.downsamples[i](h)
            if not isinstance(self.downsamples[i], nn.Identity):
                skips.append(h)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decoder
        block_idx = 0
        for i in range(len(self.upsamples)):
            for j in range(3):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.decoder_blocks[block_idx](h, t_emb)
                block_idx += 1
            h = self.upsamples[i](h)

        return self.conv_out(h)
