from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential

try:
    from monai.networks.nets import SwinUNETR  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SwinUNETR = None


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "mish":
        return nn.Mish(inplace=True)
    raise ValueError(f"Unsupported activation {name}")


def get_norm(norm: str, num_features: int) -> nn.Module:
    norm = norm.lower()
    if norm == "instance":
        return nn.InstanceNorm3d(num_features, affine=True)
    if norm == "batch":
        return nn.BatchNorm3d(num_features)
    if norm == "group":
        return nn.GroupNorm(8, num_features)
    raise ValueError(f"Unsupported norm {norm}")


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str, activation: str, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = get_norm(norm, out_channels)
        self.act1 = get_activation(activation)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = get_norm(norm, out_channels)
        self.act2 = get_activation(activation)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.dropout(out)
        out += identity
        out = self.act2(out)
        return out


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str, activation: str, dropout: float):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.block = ResidualBlock3D(in_channels, out_channels, norm, activation, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str, activation: str, dropout: float):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        self.block = ResidualBlock3D(in_channels, out_channels, norm, activation, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            x = nn.functional.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ResidualUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        channel_multipliers: Optional[List[int]] = None,
        blocks_per_stage: int = 2,
        norm: str = "instance",
        activation: str = "mish",
        dropout: float = 0.0,
        deep_supervision: bool = False,
        activation_checkpointing: Optional[Dict] = None,
    ):
        super().__init__()
        channel_multipliers = channel_multipliers or [1, 2, 4, 8]
        self.deep_supervision = deep_supervision
        self.ckpt_cfg = activation_checkpointing or {}
        self.ckpt_enabled = bool(self.ckpt_cfg.get("enabled", False))
        apply_to = self.ckpt_cfg.get("apply_to", ["bottleneck", "decoder"])
        if isinstance(apply_to, str):
            apply_to = [apply_to]
        self.ckpt_targets = set(apply_to)
        segments_cfg = self.ckpt_cfg.get("segments", 2)
        default_segments = int(segments_cfg) if isinstance(segments_cfg, int) else int(self.ckpt_cfg.get("default_segments", 2))
        self.ckpt_segments: Dict[str, int] = {"default": max(1, default_segments)}
        if isinstance(segments_cfg, dict):
            for key, value in segments_cfg.items():
                self.ckpt_segments[key] = max(1, int(value))

        encoder_channels = [base_channels * m for m in channel_multipliers]
        self.encoder_blocks = nn.ModuleList()
        in_ch = in_channels
        for ch in encoder_channels:
            stage = [ResidualBlock3D(in_ch, ch, norm, activation, dropout)]
            for _ in range(blocks_per_stage - 1):
                stage.append(ResidualBlock3D(ch, ch, norm, activation, dropout))
            self.encoder_blocks.append(nn.Sequential(*stage))
            in_ch = ch
        self.pools = nn.ModuleList([nn.MaxPool3d(2) for _ in range(len(encoder_channels) - 1)])

        bottleneck_layers = [ResidualBlock3D(in_ch, in_ch * 2, norm, activation, dropout)]
        for _ in range(blocks_per_stage - 1):
            bottleneck_layers.append(ResidualBlock3D(in_ch * 2, in_ch * 2, norm, activation, dropout))
        self.bottleneck = nn.Sequential(*bottleneck_layers)
        decoder_channels = list(reversed(encoder_channels[:-1]))
        current_channels = in_ch * 2
        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.decoder_out_channels: List[int] = []
        for ch in decoder_channels:
            self.upconvs.append(nn.ConvTranspose3d(current_channels, ch, kernel_size=2, stride=2))
            stage = [ResidualBlock3D(ch * 2, ch, norm, activation, dropout)]
            for _ in range(blocks_per_stage - 1):
                stage.append(ResidualBlock3D(ch, ch, norm, activation, dropout))
            self.decoder_blocks.append(nn.Sequential(*stage))
            current_channels = ch
            self.decoder_out_channels.append(ch)

        self.head = nn.Conv3d(current_channels, out_channels, 1)
        if self.deep_supervision:
            aux_channels = self.decoder_out_channels[:-1] if len(self.decoder_out_channels) > 1 else []
            self.aux_heads = nn.ModuleList([nn.Conv3d(ch, out_channels, 1) for ch in aux_channels])
        else:
            self.aux_heads = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        skips = []
        for idx, block in enumerate(self.encoder_blocks):
            x = block(x)
            if idx < len(self.encoder_blocks) - 1:
                skips.append(x)
                x = self.pools[idx](x)
        x = self._maybe_checkpoint(self.bottleneck, x, "bottleneck")

        aux_outputs = []
        for idx, (up, dec) in enumerate(zip(self.upconvs, self.decoder_blocks)):
            skip = skips[-(idx + 1)]
            x = up(x)
            if x.shape[-3:] != skip.shape[-3:]:
                x = nn.functional.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self._maybe_checkpoint(dec, x, "decoder")
            if self.deep_supervision and self.aux_heads and idx < len(self.aux_heads):
                aux_outputs.append(self.aux_heads[idx](x))
        logits = self.head(x)
        return {"logits": logits, "aux_logits": aux_outputs}

    def _maybe_checkpoint(self, module: nn.Sequential, tensor: torch.Tensor, stage: str) -> torch.Tensor:
        if not isinstance(module, nn.Sequential):
            return module(tensor)
        if self.ckpt_enabled and self.training and stage in self.ckpt_targets:
            segments = self.ckpt_segments.get(stage, self.ckpt_segments["default"])
            segments = max(1, min(len(module), segments))
            return checkpoint_sequential(module, segments, tensor)
        return module(tensor)


class LightUNet3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(4, base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(base_channels, base_channels * 2, 3, padding=1),
            nn.GroupNorm(4, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(base_channels * 2, base_channels, 2, stride=2),
            nn.GroupNorm(4, base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(base_channels, base_channels // 2, 2, stride=2),
            nn.GroupNorm(4, max(base_channels // 2, 1)),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv3d(max(base_channels // 2, 1), out_channels, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.encoder(x)
        x = self.decoder(x)
        return {"logits": self.head(x), "aux_logits": []}


class SwinUNETRWrapper(nn.Module):
    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        if SwinUNETR is None:
            raise ImportError("MONAI is required for Swin UNETR. Install `monai` to enable this model.")
        self.model = SwinUNETR(
            img_size=tuple(cfg.get("img_size", [96, 160, 160])),
            in_channels=cfg.get("in_channels", 1),
            out_channels=cfg.get("out_channels", 1),
            feature_size=cfg.get("embed_dim", 96),
            use_checkpoint=cfg.get("use_checkpoint", False),
            spatial_dims=3,
            drop_rate=cfg.get("drop_rate", 0.0),
            attn_drop_rate=cfg.get("attn_drop_rate", 0.0),
            dropout_path_rate=cfg.get("stochastic_depth_prob", 0.0),
            depths=cfg.get("depths", [2, 2, 6, 2]),
            num_heads=cfg.get("num_heads", [3, 6, 12, 24]),
            window_size=cfg.get("window_size", [7, 7, 7]),
            patch_size=cfg.get("patch_size", [2, 2, 2]),
            normalize=True,
        )
        ckpt = cfg.get("pretrained_backbone_ckpt")
        if ckpt:
            state = torch.load(ckpt, map_location="cpu")
            self.model.load_from(state_dict=state)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.model(x)
        return {"logits": logits, "aux_logits": []}


def build_model(model_cfg: Dict) -> nn.Module:
    model_type = model_cfg.get("type", "unet3d_residual").lower()
    if model_type == "unet3d_residual":
        return ResidualUNet3D(
            in_channels=model_cfg.get("in_channels", 1),
            out_channels=model_cfg.get("out_channels", 1),
            base_channels=model_cfg.get("base_channels", 32),
            channel_multipliers=model_cfg.get("channel_multipliers", [1, 2, 4, 8]),
            blocks_per_stage=model_cfg.get("blocks_per_stage", 2),
            norm=model_cfg.get("norm", "instance"),
            activation=model_cfg.get("activation", "mish"),
            dropout=model_cfg.get("dropout", 0.0),
            deep_supervision=model_cfg.get("deep_supervision", False),
            activation_checkpointing=model_cfg.get("activation_checkpointing"),
        )
    if model_type == "light_unet":
        return LightUNet3D(
            in_channels=model_cfg.get("in_channels", 1),
            out_channels=model_cfg.get("out_channels", 1),
            base_channels=model_cfg.get("base_channels", 16),
        )
    if model_type == "swin_unetr":
        return SwinUNETRWrapper(model_cfg)
    raise ValueError(f"Unknown model type: {model_type}")


__all__ = ["build_model", "ResidualUNet3D", "LightUNet3D", "SwinUNETRWrapper"]

