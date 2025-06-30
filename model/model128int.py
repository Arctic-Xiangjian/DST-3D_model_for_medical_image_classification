import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SinusoidalPositionalEncoding(nn.Module):
    """Adds fixed sinusoidal PE: works for variable sequence length, handles extrapolation."""
    def __init__(self, embed_dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embed_dim)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"PositionalEncoding supports length up to {self.max_len}, got {seq_len}")
        return x + self.pe[:, :seq_len, :]

# class DepthAttentionBlock(nn.Module):
#     def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
#         super().__init__()
#         self.qwkv_proj  = nn.Conv1d(embed_dim, embed_dim * 3, kernel_size=1, bias=False)
#         self.qwkv_proj2 = nn.Conv1d(embed_dim * 3, embed_dim * 3, kernel_size=3,
#                                     padding=1, groups=embed_dim * 3, bias=False)
#         self.attn       = nn.MultiheadAttention(embed_dim, num_heads, dropout=0., batch_first=True)
#         self.out_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.norm1      = nn.LayerNorm(embed_dim)
#         self.ff         = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim * 4),
#             nn.GELU(),
#             nn.Linear(embed_dim * 4, embed_dim),
#         )
#         self.norm2      = nn.LayerNorm(embed_dim)
#         self.dropout    = nn.Dropout(dropout)

#     def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         # x: (B, L, C)
#         B, L, C = x.shape
#         x1 = x.permute(0, 2, 1)           # (B, C, L)
#         x1 = self.qwkv_proj(x1)            # (B, 3C, L)
#         x1 = self.qwkv_proj2(x1)           # (B, 3C, L)
#         x1 = x1.permute(0, 2, 1)           # (B, L, 3C)
#         q, w, k = x1.chunk(3, dim=-1)      # each (B, L, C)

#         attn_out, _ = self.attn(q, k, w, key_padding_mask=key_padding_mask)
#         attn_out = self.out_proj(attn_out)
#         x = self.norm1(x + self.dropout(attn_out))
#         ff_out = self.ff(x)
#         x = self.norm2(x + self.dropout(ff_out))
#         return x


class DepthAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.qwkv_proj  = nn.Conv1d(embed_dim, embed_dim * 3, kernel_size=1, bias=False)
        self.qwkv_proj2 = nn.Conv1d(embed_dim * 3, embed_dim * 3, kernel_size=3,
                                    padding=1, groups=embed_dim * 3, bias=False)
        self.attn       = nn.MultiheadAttention(embed_dim, num_heads, dropout=0., batch_first=True)
        self.out_proj   = nn.Linear(embed_dim, embed_dim, bias=False)

        self.norm1      = nn.LayerNorm(embed_dim)
        self.norm2      = nn.LayerNorm(embed_dim)
        self.ff         = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 3),
            nn.GELU(),
            nn.Linear(embed_dim * 3, embed_dim),
        )
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, L, C)
        B, L, C = x.shape

        # === Attention block with Pre-Norm ===
        x_norm = self.norm1(x)
        x1 = x_norm.permute(0, 2, 1)                    # (B, C, L)
        x1 = self.qwkv_proj(x1)                         # (B, 3C, L)
        x1 = self.qwkv_proj2(x1)                        # (B, 3C, L)
        x1 = x1.permute(0, 2, 1)                        # (B, L, 3C)
        q, w, k = x1.chunk(3, dim=-1)                   # each (B, L, C)

        attn_out, _ = self.attn(q, k, w, key_padding_mask=key_padding_mask)
        attn_out = self.out_proj(attn_out)
        x = x + attn_out              # ← Residual after norm

        # === FFN block with Pre-Norm ===
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x

# ---------------------------------------------------------
# DepthTransformer with [CLS], PE, padding mask, dual outputs
# ---------------------------------------------------------
class DepthTransformer(nn.Module):
    """
    Takes feature map (B, C, 1, 1, D) → returns
        • slice_scores: (B, 1, D)             – same as before
        • logits:       (B, num_classes)      – optional classification logits (if num_classes>0)
    """
    def __init__(
        self,
        in_channels:    int  = 4096,
        embed_dim:      int  = 4096,
        num_heads:      int  = 8,
        num_layers:     int  = 8,
        max_len:        int  = 96,
        dropout:        float= 0,
        num_classes:    int  = 0,     # 0 = no classification head
    ):
        super().__init__()
        self.max_len     = max_len
        self.num_classes = num_classes

        self.input_norm = nn.LayerNorm(embed_dim)
        self.pos_enc    = SinusoidalPositionalEncoding(embed_dim, max_len + 1)  # +1 for CLS
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            DepthAttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ]) if num_layers > 0 else nn.ModuleList([nn.Identity()])
        self.norm_final = nn.LayerNorm(embed_dim)

        self.slice_head = nn.Linear(embed_dim, 1)
        self.cls_head   = nn.Linear(embed_dim, num_classes) if num_classes > 0 else None

    # --------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        B, C, _, _, D = x.shape
        tok = x.permute(0, 4, 1, 2, 3).reshape(B, D, C) 

        pad_len = self.max_len - D
        if pad_len < 0:
            raise ValueError(f"D={D} exceeds max_len={self.max_len}. Increase max_len or crop volume.")
        if pad_len > 0:
            pad_tok = torch.zeros(B, pad_len, C, dtype=tok.dtype, device=tok.device)
            tok     = torch.cat([pad_tok, tok], dim=1)  # (B, max_len, C)
        L = tok.size(1)

        cls = self.cls_token.expand(B, -1, -1) 
        tok = torch.cat([cls, tok], dim=1)

        if pad_len > 0:
            mask = torch.zeros(B, 1 + L, dtype=torch.bool, device=tok.device)
            mask[:,  1:pad_len + 1] = True  # padding mask for CLS and padded tokens
        else:
            mask = None
        # tok = self.pos_enc(tok)
        tok = self.input_norm(tok)
        tok = self.pos_enc(tok)
        for blk in self.blocks:
            if isinstance(blk, nn.Identity):
                # If no transformer layers, just pass through
                continue
            tok = blk(tok, key_padding_mask=mask)
        tok = self.norm_final(tok)                         # (B, 1+L, C)

        # ------------- heads ---------------------------
        # slice scores (exclude cls & paddings)
        slice_tokens = tok[:, 1+pad_len:, :]  # (B, D, C)
        slice_scores = self.slice_head(slice_tokens)       # (B, D, 1)
        slice_scores = slice_scores.permute(0, 2, 1)       # (B, 1, D)

        logits = None
        if self.cls_head is not None:
            cls_emb = tok[:, 0, :]                         # (B, C)
            logits  = self.cls_head(cls_emb)               # (B, num_classes)

        return logits, slice_scores

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=(2, 2, 1), padding=1)
        self.bn1   = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, groups=out_channels)
        self.bn2   = nn.BatchNorm3d(out_channels)
        self.downsample = (
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                           stride=(2, 2, 1), padding=0),
                nn.BatchNorm3d(out_channels)
            ) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out

# ---------------------------------------------------------
class ResNet3D(nn.Module):
    """3D backbone + DepthTransformer.

    Returns:
        logits: (B, num_classes) or None
        slice_scores: (B, 1, D)
    """
    def __init__(
        self,
        block_counts:  int = 6,
        init_channels: int = 1,
        embed_dim:     int = 1024,
        num_heads:     int = 4,
        num_layers:    int = 8,
        max_len:       int = 96,
        num_classes:   int = 3,   # 0 disables classification head
    ):
        super().__init__()
        ch = 16
        self.entry = nn.Conv3d(init_channels, ch, kernel_size=7, stride=(1, 1, 1), padding=3)
        self.bn    = nn.BatchNorm3d(ch)
        # self.entry_pool = nn.AvgPool3d((2, 2, 1))
        self.relu  = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        blocks = []
        for _ in range(block_counts):
            blocks.append(ResBlock(ch, ch * 2))
            ch *= 2
        self.res_blocks = nn.Sequential(*blocks)   # final ch = 16 * 2^block_counts
        self.spatial_pool = nn.AvgPool3d((2, 2, 1))  # (B, C, 1, 1, D)

        self.depth_transformer = DepthTransformer(
            in_channels = ch,
            embed_dim    = embed_dim,
            num_heads    = num_heads,
            num_layers   = num_layers,
            max_len      = max_len,
            num_classes  = num_classes,
        )

    # --------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        # x: (B, 1, H, W, D)
        x = self.entry(x)
        x = self.bn(x)
        # x = self.entry_pool(x)
        x = self.relu(x)
        x = self.res_blocks(x)
        x = self.spatial_pool(x)  # (B, C, 1, 1, D)
        logits, slice_scores = self.depth_transformer(x)
        return logits, slice_scores

import torch
import time
from thop import profile




def benchmark_single_case(model: torch.nn.Module,
                           input_tensor: torch.Tensor,
                           device: torch.device = torch.device("cuda:0"),
                           warmup: int = 10,
                           repeat: int = 50):
    """返回 (avg_ms, peak_mem_MB, curr_mem_MB)"""
    model = model.to(device).eval()
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
        torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)

    timings = []
    with torch.no_grad():
        for _ in range(repeat):
            starter.record()
            _ = model(input_tensor)
            ender.record()
            torch.cuda.synchronize(device)
            timings.append(starter.elapsed_time(ender))  # ms

    avg_ms     = sum(timings) / len(timings)
    peak_memMB = torch.cuda.max_memory_allocated(device) / 1024**2
    curr_memMB = torch.cuda.memory_allocated(device)    / 1024**2
    return avg_ms, peak_memMB, curr_memMB


def compute_flops(model: torch.nn.Module, input_tensor: torch.Tensor) -> float:
    """返回 FLOPs (GFLOPs)"""
    flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
    return flops / 1e9


if __name__ == "__main__":
    B, C_in, H, W, D = 1, 1, 128, 128, 64 
    dummy = torch.randn(B, C_in, H, W, D)
    model = ResNet3D(max_len=64, num_classes=3)
    gflops = compute_flops(model, dummy)
    print(f"FLOPs            : {gflops:.2f} GFLOPs")

    avg_ms, peakMB, currMB = benchmark_single_case(model, dummy)
    print(f"Average latency  : {avg_ms:.2f} ms")
    print(f"Peak GPU memory  : {peakMB:.1f} MB")
    print(f"Current GPU mem. : {currMB:.1f} MB")

    # cout parms
    def count_parameters(model: torch.nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print(f"Number of params : {num_params:,}")


    import timm_3d
    model = timm_3d.create_model(
    "resnet101.tv_in1k",
    pretrained=False,
    in_chans=1,
    num_classes=3
    )
    gflops = compute_flops(model, dummy)
    print(f"FLOPs (timm resnet50): {gflops:.2f} GFLOPs")
    avg_ms, peakMB, currMB = benchmark_single_case(model, dummy)
    print(f"Average latency (timm resnet50): {avg_ms:.2f} ms")
    print(f"Peak GPU memory (timm resnet50): {peakMB:.1f} MB")
    print(f"Current GPU mem. (timm resnet50): {currMB:.1f} MB")