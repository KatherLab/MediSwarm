import argparse
import contextlib
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import (
    mc3_18,
    mvit_v1_b,
    mvit_v2_s,
    r2plus1d_18,
    r3d_18,
    s3d,
    swin3d_b,
    swin3d_s,
    swin3d_t,
)

from models.base_model import BasicClassifier, ModelWrapper


def _resolve_checkpoint_file(path_like: str) -> str:
    """
    Resolve either:
      - a direct checkpoint file path, or
      - a directory containing checkpoint-like files.

    Returns a concrete file path.
    """
    if os.path.isfile(path_like):
        return path_like

    if os.path.isdir(path_like):
        picks = [
            f for f in os.listdir(path_like)
            if f.lower().endswith((".pth", ".pt", ".ckpt", ".pyth"))
        ]
        if not picks:
            raise FileNotFoundError(f"No checkpoint file found in: {path_like}")
        if len(picks) > 1:
            print(f"⚠️  Multiple checkpoint files found in {path_like} — using {picks[0]}")
        return os.path.join(path_like, picks[0])

    raise FileNotFoundError(f"Checkpoint path not found: {path_like}")


def resolve_pretrained_path(
    arch: str,
    pretrained_path: str | None,
    base_dir: str | Path,
) -> str | None:
    """
    Resolve pretrained checkpoint from local filesystem only.
    The checkpoint must be cached in the Docker image.

    NVFlare deploys jobs to a workspace directory, so the Python files
    run from a copy — not the Docker baked-in path. When the checkpoint
    is not found next to the workspace copy, we fall back to the known
    Docker baked-in location.

    `arch` is kept for interface compatibility.
    """
    del arch  # unused, kept to preserve function signature
    base_dir = Path(base_dir)

    if not pretrained_path:
        return None

    candidate = Path(pretrained_path)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()

    if candidate.exists():
        resolved = _resolve_checkpoint_file(str(candidate))
        print(f"Using local pretrained checkpoint: {resolved}")
        return resolved

    # Fallback: NVFlare workspace copy won't contain the .pth file;
    # try the Docker baked-in location instead.
    docker_path = Path(
        "/MediSwarm/application/jobs/challenge_3agaldran/app/custom/models"
    ) / Path(pretrained_path).name
    if docker_path.exists():
        resolved = _resolve_checkpoint_file(str(docker_path))
        print(f"Using Docker baked-in pretrained checkpoint: {resolved}")
        return resolved

    raise FileNotFoundError(
        f"Pretrained checkpoint not found: {candidate}. "
        f"Also checked Docker path: {docker_path}. "
        f"Make sure it is cached in the Docker image."
    )


def bn_to_in(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm3d):
            inorm = nn.InstanceNorm3d(
                child.num_features,
                affine=True,
                track_running_stats=False,
            )
            inorm.weight.data.copy_(child.weight.data)
            inorm.bias.data.copy_(child.bias.data)
            setattr(module, name, inorm)
        else:
            bn_to_in(child)


def load_mvit_v1_b(
    pretrained_path: str | None = None,
    num_classes: int = 3,
    in_ch: int = 3,
    target_frames: int = 16,
    target_hw: int = 224,
    temporal_kernel: int = 5,
) -> nn.Module:
    core = mvit_v1_b(weights=None)

    if pretrained_path:
        ck = _resolve_checkpoint_file(pretrained_path)
        print(f"🚀 loading MViT-B weights from {ck}")
        core.load_state_dict(torch.load(ck, map_location="cpu"), strict=False)
    else:
        print("⚙️  model initialised without pretrained weights")

    proj = core.conv_proj
    if in_ch != proj.in_channels:
        w = proj.weight
        if in_ch == 1:
            new_w = w.mean(1, keepdim=True)
        elif in_ch > 3:
            reps = (in_ch + 2) // 3
            new_w = w.repeat(1, reps, 1, 1, 1)[:, :in_ch]
            new_w *= 3.0 / in_ch
        else:
            new_w = w[:, :in_ch]

        new_proj = nn.Conv3d(
            in_ch,
            proj.out_channels,
            kernel_size=proj.kernel_size,
            stride=proj.stride,
            padding=proj.padding,
            bias=False,
        )
        new_proj.weight = nn.Parameter(new_w.clone())
        core.conv_proj = new_proj
        print(f"✓ patched conv_proj → {in_ch}-channel")

    emb_dim = core.head[1].in_features
    core.head[1] = nn.Linear(emb_dim, num_classes)

    class MVITBWrapper(nn.Module):
        def __init__(self, backbone: nn.Module):
            super().__init__()
            stride_t = math.ceil(64 / target_frames)
            self.reduce = nn.Conv3d(
                in_ch,
                in_ch,
                kernel_size=(temporal_kernel, 1, 1),
                stride=(stride_t, 1, 1),
                padding=(temporal_kernel // 2, 0, 0),
                groups=in_ch,
                bias=False,
            )
            with torch.no_grad():
                t = torch.arange(temporal_kernel) - (temporal_kernel - 1) / 2
                window = 0.54 - 0.46 * torch.cos(
                    2 * math.pi * (t + (temporal_kernel - 1) / 2)
                    / (temporal_kernel - 1)
                )
                sinc = torch.where(
                    t == 0,
                    torch.tensor(1.0),
                    torch.sin(math.pi * t) / (math.pi * t),
                )
                k = (sinc * window).view(1, 1, temporal_kernel, 1, 1)
                k = k.repeat(in_ch, 1, 1, 1, 1)
                k /= k.sum()
                self.reduce.weight.copy_(k)

            self.up = nn.Upsample(
                size=(target_frames, target_hw, target_hw),
                mode="trilinear",
                align_corners=False,
            )
            self.core = backbone

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.reduce(x)
            x = self.up(x)
            return self.core(x)

    print(f"✓ temporal blur 64→{target_frames} & spatial ↑ to {target_hw}²")
    return MVITBWrapper(core)


def load_mvit_v2_s(
    pretrained_path: str | None = None,
    num_classes: int = 3,
    in_ch: int = 3,
    target_frames: int = 16,
    target_hw: int = 224,
) -> nn.Module:
    core = mvit_v2_s(weights=None)

    if pretrained_path:
        ckpt = _resolve_checkpoint_file(pretrained_path)
        print("🚀 loading MViT-V2 weights from", ckpt)
        state = torch.load(ckpt, map_location="cpu")
        core.load_state_dict(state, strict=False)
    else:
        print("⚙️  model initialised without pretrained weights")

    proj = core.conv_proj
    if in_ch != proj.in_channels:
        w = proj.weight
        if in_ch == 1:
            new_w = w.mean(1, keepdim=True)
        elif in_ch > 3:
            reps = (in_ch + 2) // 3
            new_w = w.repeat(1, reps, 1, 1, 1)[:, :in_ch]
            new_w *= 3.0 / in_ch
        else:
            new_w = w[:, :in_ch]

        new_proj = nn.Conv3d(
            in_ch,
            proj.out_channels,
            kernel_size=proj.kernel_size,
            stride=proj.stride,
            padding=proj.padding,
            bias=False,
        )
        new_proj.weight = nn.Parameter(new_w.clone())
        core.conv_proj = new_proj
        print(f"✓ patched conv_proj → {in_ch}-channel")

    emb_dim = core.head[1].in_features
    core.head[1] = nn.Linear(emb_dim, num_classes)

    class Wrapper(nn.Module):
        def __init__(self, backbone: nn.Module):
            super().__init__()
            stride_t = math.ceil(64 / target_frames)
            self.reduce = nn.Conv3d(
                in_ch,
                in_ch,
                kernel_size=(5, 1, 1),
                stride=(stride_t, 1, 1),
                padding=(2, 0, 0),
                groups=in_ch,
                bias=False,
            )
            with torch.no_grad():
                k = torch.tensor([1, 2, 4, 2, 1], dtype=torch.float32) / 10.0
                k = k.view(1, 1, 5, 1, 1).repeat(in_ch, 1, 1, 1, 1)
                self.reduce.weight.copy_(k)
            self.core = backbone

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.reduce(x)
            x = F.interpolate(
                x,
                size=(target_frames, target_hw, target_hw),
                mode="trilinear",
                align_corners=False,
            )
            return self.core(x)

    print(f"✓ temporal blur 64→{target_frames} & spatial ↑ → {target_hw}²")
    return Wrapper(core)


def load_swin3d_s(
    pretrained_path: str | None = None,
    num_classes: int = 2,
    in_ch: int = 1,
) -> nn.Module:
    model = swin3d_s(weights=None)

    if pretrained_path:
        ckpt = _resolve_checkpoint_file(pretrained_path)
        print("🚀 Loading pretrained weights from:", ckpt)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    else:
        print("⚙️  Model initialised without pretrained weights.")

    proj, w = model.patch_embed.proj, model.patch_embed.proj.weight
    if in_ch == 1:
        new_w = w.mean(1, keepdim=True)
    elif in_ch > 3:
        reps = (in_ch + 2) // 3
        new_w = w.repeat(1, reps, 1, 1, 1)[:, :in_ch]
        new_w *= 3.0 / in_ch
    else:
        new_w = w[:, :in_ch]

    new_proj = nn.Conv3d(
        in_ch,
        proj.out_channels,
        kernel_size=proj.kernel_size,
        stride=proj.stride,
        padding=proj.padding,
        bias=False,
    )
    new_proj.weight = nn.Parameter(new_w.clone())
    model.patch_embed.proj = new_proj

    embed_dim = model.head.in_features
    model.head = nn.Linear(embed_dim, num_classes)
    return model


def load_swin3d_b(
    pretrained_path: str | None = None,
    num_classes: int = 2,
    in_ch: int = 1,
) -> nn.Module:
    model = swin3d_b(weights=None)

    if pretrained_path:
        ckpt = _resolve_checkpoint_file(pretrained_path)
        print("🚀 Loading pretrained weights from:", ckpt)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    else:
        print("⚙️  Model initialised without pretrained weights.")

    proj, w = model.patch_embed.proj, model.patch_embed.proj.weight
    if in_ch == 1:
        new_w = w.mean(1, keepdim=True)
    elif in_ch > 3:
        reps = (in_ch + 2) // 3
        new_w = w.repeat(1, reps, 1, 1, 1)[:, :in_ch]
        new_w *= 3.0 / in_ch
    else:
        new_w = w[:, :in_ch]

    new_proj = nn.Conv3d(
        in_ch,
        proj.out_channels,
        kernel_size=proj.kernel_size,
        stride=proj.stride,
        padding=proj.padding,
        bias=False,
    )
    new_proj.weight = nn.Parameter(new_w.clone())
    model.patch_embed.proj = new_proj

    embed_dim = model.head.in_features
    model.head = nn.Linear(embed_dim, num_classes)
    return model


def load_swin3d_t(
    pretrained_path: str | None = None,
    num_classes: int = 2,
    in_ch: int = 1,
) -> nn.Module:
    model = swin3d_t(weights=None)

    if pretrained_path:
        ckpt = _resolve_checkpoint_file(pretrained_path)
        print("🚀 Loading pretrained weights from:", ckpt)
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
    else:
        print("⚙️  Model initialised without pretrained weights.")

    proj = model.patch_embed.proj
    w = proj.weight
    if in_ch == 1:
        new_w = w.mean(1, keepdim=True)
    elif in_ch > 3:
        reps = (in_ch + 2) // 3
        new_w = w.repeat(1, reps, 1, 1, 1)[:, :in_ch]
        new_w *= 3.0 / in_ch
    else:
        new_w = w[:, :in_ch]

    new_proj = nn.Conv3d(
        in_ch,
        proj.out_channels,
        kernel_size=proj.kernel_size,
        stride=proj.stride,
        padding=proj.padding,
        bias=False,
    )
    new_proj.weight = nn.Parameter(new_w.clone())
    model.patch_embed.proj = new_proj

    embed_dim = model.head.in_features
    model.head = nn.Linear(embed_dim, num_classes)
    return model


def load_mc3_18(
    pretrained_path: str | None = None,
    num_classes: int = 3,
    in_ch: int = 3,
) -> nn.Module:
    model = mc3_18(weights=None)

    def adapt_stem(conv: nn.Conv3d, in_channels: int) -> nn.Conv3d:
        w = conv.weight

        if in_channels == conv.in_channels:
            return conv
        elif in_channels == 1:
            new_w = w.mean(dim=1, keepdim=True)
        elif in_channels > conv.in_channels:
            reps = (in_channels + 2) // 3
            new_w = w.repeat(1, reps, 1, 1, 1)[:, :in_channels]
            new_w *= conv.in_channels / in_channels
        else:
            new_w = w[:, :in_channels]

        new_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=False,
        )
        new_conv.weight = nn.Parameter(new_w.clone())
        return new_conv

    if pretrained_path:
        ckpt = _resolve_checkpoint_file(pretrained_path)
        print("🚀 Loading pretrained weights from:", ckpt)
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
    else:
        print("⚙️  Model initialised without pretrained weights.")

    bn_to_in(model)
    model.stem[0] = adapt_stem(model.stem[0], in_ch)

    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model


def load_s3d(
    pretrained_path: str | None = None,
    num_classes: int = 3,
    in_ch: int = 3,
) -> nn.Module:
    full = s3d(weights=None)

    if pretrained_path:
        ckpt = _resolve_checkpoint_file(pretrained_path)
        print("🚀 Loading pretrained weights from:", ckpt)
        full.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    else:
        print("⚙️  Model initialised without pretrained weights.")

    bn_to_in(full)

    stem_conv = full.features[0][0][0]
    if in_ch != stem_conv.in_channels:
        old_w = stem_conv.weight

        if in_ch == 1:
            new_w = old_w.mean(1, keepdim=True)
        elif in_ch > 3:
            reps = (in_ch + 2) // 3
            new_w = old_w.repeat(1, reps, 1, 1, 1)[:, :in_ch]
            new_w *= 3.0 / in_ch
        else:
            new_w = old_w[:, :in_ch]

        new_conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=stem_conv.out_channels,
            kernel_size=stem_conv.kernel_size,
            stride=stem_conv.stride,
            padding=stem_conv.padding,
            bias=False,
        )
        new_conv.weight = nn.Parameter(new_w.clone())
        full.features[0][0][0] = new_conv

    backbone = nn.Sequential(*full.features)
    feat_dim = full.classifier[1].in_channels

    class S3DHead(nn.Module):
        def __init__(self, core: nn.Module):
            super().__init__()
            self.core = core
            self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.head = nn.Linear(feat_dim, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.core(x)
            x = self.pool(x).flatten(1)
            return self.head(x)

    return S3DHead(backbone)


def load_r2plus1d_18(
    pretrained_path: str | None = None,
    num_classes: int = 2,
    in_ch: int = 1,
) -> nn.Module:
    full = r2plus1d_18(weights=None)

    if pretrained_path:
        ckpt = _resolve_checkpoint_file(pretrained_path)
        print("🚀 Loading pretrained weights from:", ckpt)
        full.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    else:
        print("⚙️  Model initialized without pretrained weights.")

    stem_conv = full.stem[0]
    w_rgb = stem_conv.weight

    if in_ch == 1:
        new_w = w_rgb.mean(1, keepdim=True)
    elif in_ch > 3:
        reps = (in_ch + 2) // 3
        new_w = w_rgb.repeat(1, reps, 1, 1, 1)[:, :in_ch]
        new_w *= 3.0 / in_ch
    elif in_ch == 3:
        new_w = w_rgb
    else:
        new_w = w_rgb[:, :in_ch]

    new_stem_conv = nn.Conv3d(
        in_ch,
        stem_conv.out_channels,
        kernel_size=stem_conv.kernel_size,
        stride=stem_conv.stride,
        padding=stem_conv.padding,
        bias=False,
    )
    new_stem_conv.weight = nn.Parameter(new_w.clone())
    full.stem[0] = new_stem_conv

    backbone = nn.Sequential(
        full.stem,
        full.layer1,
        full.layer2,
        full.layer3,
        full.layer4,
    )

    feat_dim = full.fc.in_features

    class R2Plus1dHead(nn.Module):
        def __init__(self, core: nn.Module):
            super().__init__()
            self.core = core
            self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.classif = nn.Linear(feat_dim, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.core(x)
            x = self.pool(x).flatten(1)
            return self.classif(x)

    return R2Plus1dHead(backbone)


def load_r3d_18(
    pretrained_path: str | None = None,
    num_classes: int = 2,
    in_ch: int = 1,
) -> nn.Module:
    model = r3d_18(weights=None)

    if pretrained_path is not None:
        ckpt_file = _resolve_checkpoint_file(pretrained_path)
        print("🚀 Loading pretrained weights from:", ckpt_file)

        state = torch.load(ckpt_file, map_location="cpu")
        conv1_w = state.get("stem.0.weight")
        if conv1_w is None:
            raise KeyError("Checkpoint missing 'stem.0.weight'")

        if in_ch == 1:
            collapsed = conv1_w.mean(dim=1, keepdim=True)
        elif in_ch > 3:
            reps = (in_ch + 2) // 3
            collapsed = conv1_w.repeat(1, reps, 1, 1, 1)[:, :in_ch]
            collapsed *= 3.0 / in_ch
        else:
            collapsed = conv1_w[:, :in_ch]

        orig = model.stem[0]
        new_stem = nn.Conv3d(
            in_channels=in_ch,
            out_channels=orig.out_channels,
            kernel_size=orig.kernel_size,
            stride=orig.stride,
            padding=orig.padding,
            bias=False,
        )
        new_stem.weight = nn.Parameter(collapsed.clone())
        model.stem[0] = new_stem

        for k in ["stem.0.weight", "stem.0.bias", "fc.weight", "fc.bias"]:
            state.pop(k, None)

        model.load_state_dict(state, strict=False)
    else:
        print("⚙️  Model initialized without pretrained weights.")
        orig = model.stem[0]
        w_orig = orig.weight

        if in_ch == 1:
            collapsed = w_orig.mean(dim=1, keepdim=True)
        elif in_ch > 3:
            reps = (in_ch + 2) // 3
            collapsed = w_orig.repeat(1, reps, 1, 1, 1)[:, :in_ch]
            collapsed *= 3.0 / in_ch
        else:
            collapsed = w_orig[:, :in_ch]

        new_stem = nn.Conv3d(
            in_channels=in_ch,
            out_channels=orig.out_channels,
            kernel_size=orig.kernel_size,
            stride=orig.stride,
            padding=orig.padding,
            bias=False,
        )
        new_stem.weight = nn.Parameter(collapsed.clone())
        model.stem[0] = new_stem

    feat_dim = model.fc.in_features
    model.fc = nn.Linear(feat_dim, num_classes)
    return model


VIDEO_BACKBONES = {
    "r3d_18": load_r3d_18,
    "r2plus1d_18": load_r2plus1d_18,
    "s3d": load_s3d,
    "mc3_18": load_mc3_18,
    "swin3d_t": load_swin3d_t,
    "swin3d_s": load_swin3d_s,
    "swin3d_b": load_swin3d_b,
    "mvit_v2_s": load_mvit_v2_s,
    "mvit_v1_b": load_mvit_v1_b,
}


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def model_factory(
    arch: str,
    pretrained_path: str | None = None,
    num_classes: int = 2,
    in_ch: int = 3,
    freeze_backbone: bool = False,
    seed: int | None = 42,
    **loss_kwargs,
) -> BasicClassifier:
    """
    Builds and returns a video classification model for 3D medical volumes.

    Pretrained weights are loaded from the local filesystem only.
    Ensure the checkpoint file is cached in the Docker image at build time.
    """
    set_global_seed(seed)
    arch = arch.lower()

    if arch not in VIDEO_BACKBONES:
        raise ValueError(
            f"Unknown architecture '{arch}'. Available: {list(VIDEO_BACKBONES.keys())}"
        )

    base_dir = os.path.dirname(os.path.abspath(__file__))

    resolved_pretrained_path = resolve_pretrained_path(
        arch=arch,
        pretrained_path=pretrained_path,
        base_dir=base_dir,
    )

    if pretrained_path and resolved_pretrained_path is None:
        raise FileNotFoundError(
            f"Could not resolve pretrained checkpoint for requested path: {pretrained_path}"
        )

    model = VIDEO_BACKBONES[arch](
        pretrained_path=resolved_pretrained_path,
        num_classes=num_classes,
        in_ch=in_ch,
    )

    def freeze_backbone_only(m: nn.Module) -> None:
        for p in m.parameters():
            p.requires_grad = False

        for mod in reversed(list(m.modules())):
            if isinstance(mod, nn.Linear):
                for p in mod.parameters():
                    p.requires_grad = True
                break

    if freeze_backbone:
        freeze_backbone_only(model)

    wrapped_model = ModelWrapper(
        backbone=model,
        in_ch=in_ch,
        num_classes=num_classes,
        **loss_kwargs,
    )
    return wrapped_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video model factory smoke test for 3D volumes."
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        help="Model key: " + ", ".join(VIDEO_BACKBONES.keys()),
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to local pretrained weights.",
    )
    parser.add_argument("--num_classes", type=int, default=1, help="Number of classes.")
    parser.add_argument("--in_ch", type=int, default=3, help="Input channels.")
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze model weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on, e.g., 'cpu' or 'cuda:0'.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for dummy input.")
    parser.add_argument("--depth", type=int, default=16, help="Temporal depth.")
    parser.add_argument("--height", type=int, default=64, help="Spatial height.")
    parser.add_argument("--width", type=int, default=64, help="Spatial width.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    args = parser.parse_args()

    print(f"🔧 Building model '{args.arch}'...")

    model = model_factory(
        arch=args.arch,
        pretrained_path=args.pretrained,
        num_classes=args.num_classes,
        in_ch=args.in_ch,
        freeze_backbone=args.freeze_backbone,
        seed=args.seed,
    )
    model = model.to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dummy = torch.randn(
        args.batch_size,
        args.in_ch,
        args.depth,
        args.height,
        args.width,
        device=args.device,
    )

    model.eval()
    with torch.no_grad():
        for _ in range(2):
            _ = model(dummy)

        t0 = time.perf_counter()
        for _ in range(5):
            _ = model(dummy)
        elapsed = (time.perf_counter() - t0) / 5 * 1e3

    out = model(dummy)
    print(f"✅ Output shape : {tuple(out.shape)}")
    print(f"📦 Params       : {total_params:,} ({trainable_params:,} trainable)")
    print(f"⚡ Fwd-time     : {elapsed:,.1f} ms on {args.device}")