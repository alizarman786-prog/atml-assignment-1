"""AdaIN style transfer (Huang & Belongie, ICCV 2017), used for Task 1's
shape/texture cue-conflict generation.

Architecture and pretrained weights follow the standard reference PyTorch
implementation (naoto0804/pytorch-AdaIN, MIT licensed), attributed here per
the assignment's requirement to attribute materially reused external code:
  https://github.com/naoto0804/pytorch-AdaIN
Pretrained weights (decoder.pth, vgg_normalised.pth) are downloaded from that
repository's GitHub release (v0.0.0) on first use and cached locally.

Encoder: a "normalized" VGG-19 (a 1x1 conv absorbs the usual ImageNet
mean-subtraction, so inputs are plain [0,1]-range tensors, no separate
normalization needed), truncated to relu4_1 (the first 31 layers).
Decoder: the mirrored architecture trained by the paper's authors to invert
that encoder's relu4_1 features back into an image.
AdaIN layer: rescales content feature statistics (channel-wise mean/std) to
match the style feature's statistics -- this IS the mechanism that transfers
texture/style while (approximately) preserving content/shape structure.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

DECODER_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth"
VGG_URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth"

# Full "normalized" VGG-19 architecture (53 layers, through relu5_4). Only the
# first 31 layers (through relu4_1) are used as the encoder at inference
# time; the remaining layers exist purely so the checkpoint's state_dict keys
# line up (the checkpoint was saved from this exact full architecture).
def _build_vgg() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(3, 64, (3, 3)), nn.ReLU(),      # relu1_1
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),     # relu1_2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 128, (3, 3)), nn.ReLU(),    # relu2_1
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),   # relu2_2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 256, (3, 3)), nn.ReLU(),   # relu3_1
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),   # relu3_2
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),   # relu3_3
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),   # relu3_4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 512, (3, 3)), nn.ReLU(),   # relu4_1 <- encoder ends here (index 30)
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),   # relu4_2
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),   # relu4_3
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),   # relu4_4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),   # relu5_1
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),   # relu5_2
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),   # relu5_3
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),   # relu5_4
    )


def _build_decoder() -> nn.Sequential:
    return nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 256, (3, 3)), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 128, (3, 3)), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 64, (3, 3)), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 3, (3, 3)),
    )


ENCODER_CUTOFF = 31  # vgg[:31] ends exactly after relu4_1


def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5):
    n, c = feat.size()[:2]
    var = feat.view(n, c, -1).var(dim=2) + eps
    std = var.sqrt().view(n, c, 1, 1)
    mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    return mean, std


def adaptive_instance_normalization(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


def _download_if_missing(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        print(f"[adain] downloading {url} -> {dst}")
        torch.hub.download_url_to_file(url, str(dst))


class AdaINStyleTransfer(nn.Module):
    """Loads the pretrained encoder (truncated VGG) and decoder, and performs
    style_transfer(content, style, alpha). Inputs/outputs are [0,1]-range
    RGB tensors (N,3,H,W) -- no ImageNet mean/std normalization needed, the
    encoder's first 1x1 conv absorbs that."""

    def __init__(self, cache_dir: str = "./data_raw/adain_weights"):
        super().__init__()
        cache_dir = Path(cache_dir)
        vgg_path = cache_dir / "vgg_normalised.pth"
        decoder_path = cache_dir / "decoder.pth"
        _download_if_missing(VGG_URL, vgg_path)
        _download_if_missing(DECODER_URL, decoder_path)

        vgg_full = _build_vgg()
        vgg_full.load_state_dict(torch.load(vgg_path, map_location="cpu"))
        self.encoder = vgg_full[:ENCODER_CUTOFF]

        decoder = _build_decoder()
        decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
        self.decoder = decoder

        self.encoder.eval().requires_grad_(False)
        self.decoder.eval().requires_grad_(False)

    @torch.no_grad()
    def style_transfer(self, content: torch.Tensor, style: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        assert 0.0 <= alpha <= 1.0
        content_f = self.encoder(content)
        style_f = self.encoder(style)
        feat = adaptive_instance_normalization(content_f, style_f)
        feat = feat * alpha + content_f * (1 - alpha)
        out = self.decoder(feat)
        return out.clamp(0.0, 1.0)
