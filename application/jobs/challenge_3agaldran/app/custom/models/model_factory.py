import argparse
import os, sys
import math
import random
import time
import contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from pytorchvideo.models.hub import i3d_r50, slowfast_r50#, r2plus1d_18
from torchvision.models.video import r3d_18, r2plus1d_18, s3d, mc3_18
#from pytorchvideo.models.hub import slowfast_r101
from torchvision.models.video import swin3d_t          # tiny / patch=(2,4,4)
from torchvision.models.video import swin3d_b
from torchvision.models.video import swin3d_s
from torchvision.models.video import mvit_v1_b, mvit_v2_s
#from pytorchvideo.models.hub import x3d_s

# utils/model_factory.py  (add near the other loaders)
#from pytorchvideo.models.hub import x3d_s
import torch, torch.nn as nn, torch.nn.functional as F
import os, math

from models.base_model import BasicClassifier, ModelWrapper

def bn_to_in(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm3d):
            inorm = nn.InstanceNorm3d(child.num_features,
                                       affine=True,
                                       track_running_stats=False)
            inorm.weight.data.copy_(child.weight.data)
            inorm.bias.data.copy_(child.bias.data)
            setattr(module, name, inorm)
        else:
            bn_to_in(child)

# def load_x3d_s(
#     pretrained_path: str | None = "mvit_v2_s-ae3be167",   # folder OR .pth/.pt/.ckpt file
#     num_classes:    int        = 2,
#     target_frames:  int        = 16,
#     target_hw:      int        = 224,
# ) -> nn.Module:
#     """
#     X3D-S adapted for 1-channel 64×128×128 CT volumes (no auto-download).

#     Steps
#     -----
#     1.  Build bare backbone (`x3d_s(pretrained=False)`).
#     2.  Optionally load local checkpoint before surgery.
#     3.  Collapse first spatial conv (RGB→1ch) — works for both
#         Conv3d stems (legacy) *and* Conv2plus1d stems (new).
#     4.  Replace classifier head with fresh `Linear(num_classes)`.
#     5.  Wrap with:
#           • learned temporal blur 64→`target_frames`
#           • trilinear up-sample to `target_hw`².
#     """

#     # ---------------------------------------------------------------- 1) build
#     core = x3d_s(pretrained=False)

#     # ---------------------------------------------------------------- 2) ckpt
#     if pretrained_path:
#         print(f"🚀 loading X3D-S weights from {pretrained_path}")
#         ckpt_file = pretrained_path
#         if os.path.isdir(pretrained_path):
#             picks = [f for f in os.listdir(pretrained_path)
#                      if f.lower().endswith((".pth", ".pt", ".ckpt"))]
#             if not picks:
#                 raise FileNotFoundError(f"No checkpoint in {pretrained_path}")
#             if len(picks) > 1:
#                 print(f"⚠️ multiple ckpts — using {picks[0]}")
#             ckpt_file = os.path.join(pretrained_path, picks[0])
#         state = torch.load(ckpt_file, map_location="cpu")
#         core.load_state_dict(state, strict=False)
#     else:
#         print("⚙️ model initialised **without** pretrained weights")

#     # ---------------------------------------------------------------- 3) patch first conv → 1-channel
#     #
#     #  • Older releases:   core.stem.conv  (Conv3d)
#     #  • Newer releases:   first block is Conv2plus1d  ⇒ it has attribute `conv_s`
#     #
#     def _find_spatial_conv(net):
#         # Case A – classic Conv3d stem
#         if hasattr(net, "stem") and hasattr(net.stem, "conv"):
#             return net.stem, "conv", net.stem.conv

#         # Case B – Conv2plus1d stem (new X3D)
#         first_block = net.blocks[0] if hasattr(net, "blocks") else None
#         if first_block is not None and hasattr(first_block, "conv_s"):
#             return first_block, "conv_s", first_block.conv_s

#         # Fallback: walk all modules until we hit a Conv3d-like module
#         for m in net.modules():
#             if isinstance(m, nn.Conv3d):
#                 return None, None, m
#             # allow detection by class-name to avoid importing the class
#             if m.__class__.__name__.lower().startswith("conv2plus1d"):
#                 return m, "conv_s", m.conv_s

#         raise RuntimeError("Could not locate a spatial conv in X3D stem")

#     parent, attr, old = _find_spatial_conv(core)

#     w_gray = old.weight.mean(1, keepdim=True)            # (C_out,1,kT,kH,kW)
#     new = nn.Conv3d(
#         in_channels=1,
#         out_channels=old.out_channels,
#         kernel_size=old.kernel_size,
#         stride=old.stride,
#         padding=old.padding,
#         bias=False,
#     )
#     new.weight = nn.Parameter(w_gray)

#     if parent is None:  # extremely old rare case
#         raise RuntimeError("Unhandled stem layout (no parent module)")
#     else:
#         setattr(parent, attr, new)

#     print("✓ patched first spatial conv → 1-channel")

#     # ---------------------------------------------------------------- 4) new classifier
#     try:                            # current API
#         in_dim = core.head.projection.in_features
#         core.head.projection = nn.Identity()
#     except AttributeError:          # very old API
#         in_dim = core.head.in_features
#         core.head = nn.Identity()

#     new_fc = nn.Linear(in_dim, num_classes)

#     # ---------------------------------------------------------------- 5) wrapper with temporal blur & up-sample
#     class X3DWrapper(nn.Module):
#         def __init__(self, backbone, classifier):
#             super().__init__()
#             stride_t = math.ceil(64 / target_frames)        # ≈4
#             self.reduce = nn.Conv3d(
#                 1, 1, kernel_size=(5,1,1), stride=(stride_t,1,1),
#                 padding=(2,0,0), bias=False,
#             )
#             with torch.no_grad():
#                 blur = torch.tensor([1,2,4,2,1], dtype=torch.float32) / 10
#                 self.reduce.weight.zero_()
#                 self.reduce.weight[:, :, :, 0, 0] = blur.view(1,1,5)

#             self.backbone   = backbone
#             self.classifier = classifier
#             self.up         = nn.Upsample(size=(target_frames, target_hw, target_hw),
#                                           mode="trilinear", align_corners=False)

#         def forward(self, x):                       # x : [B,1,64,128,128]
#             x = self.reduce(x)                      #      [B,1,16,128,128]
#             x = self.up(x)                          #      [B,1,16,224,224]
#             feats = self.backbone(x)                # (already GAP pooled)
#             return self.classifier(feats)           # logits

#     print(f"✓ temporal blur 64→{target_frames} & spatial ↑ to {target_hw}²")
#    return X3DWrapper(core, new_fc)

def load_mvit_v1_b(pretrained_path: str | None = None,
                   num_classes:    int = 3,
                   in_ch:          int = 3,     # Pre, Sub1, T2
                   target_frames:  int = 16,
                   target_hw:      int = 224,
                   temporal_kernel:int = 5) -> nn.Module:
    """
    Offline-safe MViT-v1-B wrapper that digests `in_ch` channels and adds
    a learned 1-D blur to compress the slice/time axis.

    ── Pipeline ────────────────────────────────────────────────────────────
    input  : B × in_ch × 64 × 128 × 128  (unilateral crop, full-rez)
    reduce : depth-wise Conv3d, stride 4 along T  → B x in_ch x 16 x 128 x 128
    up     : trilinear upsample                → B x in_ch x 16 x x 224 x 224
    core   : original MViT-v1-B                → logits  (B x num_classes)
    """

    # ───────────────────────── 1) backbone skeleton ───────────────────────
    core = mvit_v1_b(weights=None)         # never auto-downloads

    # ───────────────────────── 2) optional checkpoint ─────────────────────
    if pretrained_path:
        print(f"🚀 loading MViT-B weights from {pretrained_path}")
        ck = pretrained_path
        if os.path.isdir(pretrained_path):
            cand = [f for f in os.listdir(pretrained_path)
                    if f.lower().endswith((".pth", ".pt", ".ckpt", ".pyth"))]
            if not cand:
                raise FileNotFoundError("no ckpt in " + pretrained_path)
            if len(cand) > 1:
                print("⚠️  multiple files – using", cand[0])
            ck = os.path.join(pretrained_path, cand[0])
        core.load_state_dict(torch.load(ck, map_location="cpu"),
                             strict=False)
    else:
        print("⚙️  model initialised without pretrained weights")

    # ───────────────────────── 3) patch conv_proj for in_ch ───────────────
    proj = core.conv_proj                       # Conv3d(3,128,3,7,7)
    if in_ch != proj.in_channels:
        w = proj.weight                         # (128,3,3,7,7)
        if in_ch == 1:                          # RGB → mono
            new_w = w.mean(1, keepdim=True)
        elif in_ch > 3:                         # replicate & scale
            reps = (in_ch + 2) // 3
            new_w = w.repeat(1, reps, 1, 1, 1)[:, :in_ch]
            new_w *= 3.0 / in_ch
        else:                                   # 2-channel
            new_w = w[:, :in_ch]

        new_proj = nn.Conv3d(
            in_ch, proj.out_channels,
            kernel_size=proj.kernel_size,
            stride=proj.stride,
            padding=proj.padding,
            bias=False
        )
        new_proj.weight = nn.Parameter(new_w.clone())
        core.conv_proj = new_proj
        print(f"✓ patched conv_proj → {in_ch}-channel")

    # ───────────────────────── 4) replace classifier head ──────────────────
    emb_dim = core.head[1].in_features         # 1024 for v1-B
    core.head[1] = nn.Linear(emb_dim, num_classes)

    # ───────────────────────── 5) wrapper with temporal blur ───────────────
    class MVIT_B_Wrapper(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            stride_t = math.ceil(64 / target_frames)
            # depth-wise 1×1×k blur per channel
            self.reduce = nn.Conv3d(
                in_ch, in_ch,
                kernel_size=(temporal_kernel, 1, 1),
                stride=(stride_t, 1, 1),
                padding=(temporal_kernel // 2, 0, 0),
                groups=in_ch,
                bias=False)
            # sinc-Hann init (same kernel for every channel)
            with torch.no_grad():
                t = torch.arange(temporal_kernel) - (temporal_kernel - 1) / 2
                window = 0.54 - 0.46 * torch.cos(
                    2 * math.pi * (t + (temporal_kernel - 1) / 2)
                    / (temporal_kernel - 1))
                sinc = torch.where(t == 0, torch.tensor(1.),
                                   torch.sin(math.pi * t) / (math.pi * t))
                k = (sinc * window).view(1, 1, temporal_kernel, 1, 1)
                k = k.repeat(in_ch, 1, 1, 1, 1)
                k /= k.sum()
                self.reduce.weight.copy_(k)

            self.up = nn.Upsample(size=(target_frames,
                                        target_hw, target_hw),
                                   mode="trilinear",
                                   align_corners=False)
            self.core = backbone

        def forward(self, x):                # x: B×in_ch×64×128×128
            x = self.reduce(x)               #          ↓
            x = self.up(x)                   # B×in_ch×16×224×224
            return self.core(x)              # logits

    print(f"✓ temporal blur 64→{target_frames} & spatial ↑ to {target_hw}²")
    return MVIT_B_Wrapper(core)

def load_mvit_v2_s(pretrained_path: str | None = None,
                   num_classes:    int = 3,
                   in_ch:          int = 3,      # Pre, Sub1, T2
                   target_frames:  int = 16,
                   target_hw:      int = 224) -> nn.Module:
    """
    Offline-safe MViT-V2-S wrapper with
      • channel-agnostic stem (in_ch = 1, 3, …)
      • depth-wise stride-4 blur Conv3d to compress 64→target_frames
      • spatial up-sample to target_hw
      • fresh Linear head (num_classes)
    """

    # -------- 1) backbone skeleton ---------------------------------------
    core = mvit_v2_s(weights=None)          # never auto-downloads

    # -------- 2) optional checkpoint -------------------------------------
    if pretrained_path:
        ckpt = (pretrained_path if os.path.isfile(pretrained_path)
                else os.path.join(pretrained_path, next(f for f in os.listdir(pretrained_path) if f.lower().endswith((".pth", ".pt", ".ckpt", ".pyth")))))
        print("🚀 loading MViT-V2 weights from", ckpt)
        state = torch.load(ckpt, map_location="cpu")
        core.load_state_dict(state, strict=False)
    else:
        print("⚙️  model initialised without pretrained weights")

    # -------- 3) patch conv_proj for in_ch --------------------------------
    proj = core.conv_proj                       # (96,3,3,7,7)
    if in_ch != proj.in_channels:
        w = proj.weight                         # rgb weights
        if in_ch == 1:                          # RGB → mono
            new_w = w.mean(1, keepdim=True)
        elif in_ch > 3:                         # replicate & scale
            reps = (in_ch + 2) // 3
            new_w = w.repeat(1, reps, 1, 1, 1)[:, :in_ch]
            new_w *= 3.0 / in_ch
        else:                                   # 2-channel
            new_w = w[:, :in_ch]

        new_proj = nn.Conv3d(
            in_ch, proj.out_channels,
            kernel_size=proj.kernel_size,
            stride=proj.stride,
            padding=proj.padding,
            bias=False
        )
        new_proj.weight = nn.Parameter(new_w.clone())
        core.conv_proj = new_proj
        print(f"✓ patched conv_proj → {in_ch}-channel")

    # -------- 4) replace classifier head ----------------------------------
    emb_dim = core.head[1].in_features
    core.head[1] = nn.Linear(emb_dim, num_classes)

    # -------- 5) wrapper with temporal blur & upsample --------------------
    class Wrapper(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            stride_t = math.ceil(64 / target_frames)
            self.reduce = nn.Conv3d(
                in_ch, in_ch,
                kernel_size=(5, 1, 1),
                stride=(stride_t, 1, 1),
                padding=(2, 0, 0),
                groups=in_ch,
                bias=False
            )
            # blur kernel [1 2 4 2 1] / 10 replicated per channel
            with torch.no_grad():
                k = torch.tensor([1, 2, 4, 2, 1],
                                 dtype=torch.float32) / 10.0
                k = k.view(1, 1, 5, 1, 1).repeat(in_ch, 1, 1, 1, 1)
                self.reduce.weight.copy_(k)
            self.core = backbone

        def forward(self, x):                # B×C×64×128×128
            x = self.reduce(x)               # B×C×16×128×128
            x = F.interpolate(
                x, size=(target_frames, target_hw, target_hw),
                mode="trilinear", align_corners=False)
            return self.core(x)              # logits

    print(f"✓ temporal blur 64→{target_frames} & spatial ↑ → {target_hw}²")
    return Wrapper(core)

def load_swin3d_s(pretrained_path: str | None = None,
                  num_classes: int = 2) -> nn.Module:
    model = swin3d_s(weights=None)

    # --- checkpoint first -----------
    if pretrained_path:
        print("🚀 Loading pretrained weights from:", pretrained_path)
        ckpt = (next((os.path.join(pretrained_path, f)
                      for f in os.listdir(pretrained_path)
                      if f.lower().endswith((".pth", ".pt", ".ckpt", ".pyth"))),
                      pretrained_path))
        model.load_state_dict(torch.load(ckpt, map_location="cpu"),
                              strict=False)
    else:
        print("⚙️  Model initialised without pretrained weights.")

    # --- RGB → grayscale ------------
    proj, w = model.patch_embed.proj, model.patch_embed.proj.weight
    new_proj = nn.Conv3d(1, proj.out_channels,
                         kernel_size=proj.kernel_size,
                         stride=proj.stride,
                         padding=proj.padding,
                         bias=False)
    new_proj.weight = nn.Parameter(w.mean(1, keepdim=True))
    model.patch_embed.proj = new_proj

    # --- fresh classifier head ------
    embed_dim = model.head.in_features   # 1024 for Small
    model.head = nn.Linear(embed_dim, num_classes)

    return model

def load_swin3d_b(pretrained_path: str | None = None,
                  num_classes: int = 2) -> nn.Module:
    model = swin3d_b(weights=None)

    # checkpoint (same logic as above)
    if pretrained_path:
        print("🚀 Loading pretrained weights from:", pretrained_path)
        ckpt = (next((os.path.join(pretrained_path, f)
                      for f in os.listdir(pretrained_path)
                      if f.lower().endswith((".pth", ".pt", ".ckpt", ".pyth"))),
                      pretrained_path))
        model.load_state_dict(torch.load(ckpt, map_location="cpu"),
                              strict=False)
    else:
        print("⚙️  Model initialised without pretrained weights.")

    # RGB → 1-channel
    proj, w = model.patch_embed.proj, model.patch_embed.proj.weight
    new_proj = nn.Conv3d(1, proj.out_channels,
                         kernel_size=proj.kernel_size,
                         stride=proj.stride,
                         padding=proj.padding,
                         bias=False)
    new_proj.weight = nn.Parameter(w.mean(1, keepdim=True))
    model.patch_embed.proj = new_proj

    # new head
    embed_dim = model.head.in_features   # 1280 for Base
    model.head = nn.Linear(embed_dim, num_classes)

    return model

def load_swin3d_t(pretrained_path: str | None = None,
                  num_classes: int = 2) -> nn.Module:
    """
    Swin-3D Tiny adapted to 1-channel CT cubes + new classifier head.
    """
    # 1️⃣ build bare model (never auto-download)
    model = swin3d_t(weights=None)

    # 2️⃣ optional checkpoint (load BEFORE surgery)
    if pretrained_path:
        print("🚀 Loading pretrained weights from:", pretrained_path)
        ckpt = pretrained_path
        if os.path.isdir(pretrained_path):
            files = [f for f in os.listdir(pretrained_path)
                     if f.lower().endswith((".pth", ".pt", ".ckpt", ".pyth"))]
            if not files:
                raise FileNotFoundError(f"No ckpt in {pretrained_path}")
            if len(files) > 1:
                print(f"⚠️  Multiple ckpts — using {files[0]}")
            ckpt = os.path.join(pretrained_path, files[0])
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
    else:
        print("⚙️  Model initialised without pretrained weights.")

    # 3️⃣ collapse patch-embedding conv (RGB → 1-channel)
    proj   = model.patch_embed.proj                  # Conv3d(3,dim,2,4,4)
    w_rgb  = proj.weight                             # (C_out, 3, 2, 4, 4)
    w_gray = w_rgb.mean(1, keepdim=True)             # (C_out, 1, 2, 4, 4)
    new_proj = nn.Conv3d(
        1, proj.out_channels,
        kernel_size=proj.kernel_size,
        stride=proj.stride,
        padding=proj.padding,
        bias=False
    )
    new_proj.weight = nn.Parameter(w_gray)
    model.patch_embed.proj = new_proj

    # 4️⃣ replace the classification head (token mean is inside forward)
    embed_dim   = model.head.in_features             # 768
    model.head  = nn.Linear(embed_dim, num_classes)

    return model

def load_mc3_18(pretrained_path: str | None = None,
                num_classes:    int = 3,
                in_ch:          int = 3) -> nn.Module:
    """
    Build an MC3-18 backbone that works offline and ingests `in_ch` channels.

    Parameters
    ----------
    pretrained_path : str | None
        Local .pt/.pth checkpoint (file or folder).  Never downloads.
    num_classes : int
        Output classes (3 for ODELIA).
    in_ch : int
        Input channels: 3 = Pre+Sub+T2, 1 = mono, etc.
    """
    # 1) create base model (no torchvision weights)
    model = mc3_18(weights=None)

    # --------------------------------------------------------------
    # 2) helper to adapt stem weights for any in_ch
    # --------------------------------------------------------------
    def adapt_stem(conv: nn.Conv3d, in_channels: int) -> nn.Conv3d:
        w = conv.weight  # shape (64, old_in, 3, 7, 7)

        if in_channels == conv.in_channels:
            return conv                       # nothing to change
        elif in_channels == 1:                # RGB→mono (average)
            new_w = w.mean(dim=1, keepdim=True)
        elif in_channels > conv.in_channels:  # replicate & scale
            reps = (in_channels + 2) // 3     # ceil
            new_w = w.repeat(1, reps, 1, 1, 1)[:, :in_channels]
            new_w *= conv.in_channels / in_channels
        else:                                 # 2-channel or fewer
            new_w = w[:, :in_channels]

        new_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=False,
        )
        new_conv.weight = nn.Parameter(new_w)
        return new_conv

    # --------------------------------------------------------------
    # 3) optionally load checkpoint BEFORE stem surgery
    # --------------------------------------------------------------
    if pretrained_path:
        print("🚀 Loading pretrained weights from:", pretrained_path)
        ckpt = pretrained_path
        if os.path.isdir(pretrained_path):
            picks = [f for f in os.listdir(pretrained_path)
                     if f.lower().endswith((".pth", ".pt", ".ckpt", ".pyth"))]
            if not picks:
                raise FileNotFoundError(f"No ckpt in {pretrained_path}")
            if len(picks) > 1:
                print(f"⚠️  Multiple ckpts — using {picks[0]}")
            ckpt = os.path.join(pretrained_path, picks[0])
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
    else:
        print("⚙️  Model initialised without pretrained weights.")
    
    
    # convert to instance3d norm
    bn_to_in(model)


    # --------------------------------------------------------------
    # 4) adapt stem to in_ch AFTER checkpoint is loaded
    # --------------------------------------------------------------
    model.stem[0] = adapt_stem(model.stem[0], in_ch)

    # --------------------------------------------------------------
    # 5) replace classifier head
    # --------------------------------------------------------------
    in_feats = model.fc.in_features          # 512 for MC3-18
    model.fc = nn.Linear(in_feats, num_classes)

    return model

def load_s3d(pretrained_path: str | None = None,
             num_classes: int = 3,
             in_ch:      int = 3) -> nn.Module:
    """
    Build an S3D model that works offline and accepts `in_ch` channels.

    Parameters
    ----------
    pretrained_path : str | None
        Folder *or* file containing a local .pth / .pt / .ckpt checkpoint.
        If None, the model starts from scratch.
    num_classes : int
        Number of output classes (3 for ODELIA).
    in_ch : int
        Expected input channels.  Use 3 for (Pre, Sub_1, T2).
    """
    # 1) bare backbone, no internet access
    full = s3d(weights=None)

    # 2) optionally load local checkpoint *before* any surgery
    if pretrained_path:
        print("🚀 Loading pretrained weights from:", pretrained_path)
        ckpt = pretrained_path
        if os.path.isdir(pretrained_path):
            picks = [f for f in os.listdir(pretrained_path)
                     if f.lower().endswith((".pth", ".pt", ".ckpt", ".pyth"))]
            if not picks:
                raise FileNotFoundError(f"No ckpt in {pretrained_path}")
            if len(picks) > 1:
                print(f"⚠️  Multiple ckpts — using {picks[0]}")
            ckpt = os.path.join(pretrained_path, picks[0])
        full.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    else:
        print("⚙️  Model initialised **without** pretrained weights.")


    # convert to instance3d norm
    bn_to_in(full)

    # 3) adapt the first spatial Conv3d if channel count differs
    stem_conv = full.features[0][0][0]      # (64, 3, 1, 7, 7) in vanilla
    if in_ch != stem_conv.in_channels:
        old_w = stem_conv.weight            # (64, 3, kT, 7, 7)

        if in_ch == 1:                      # RGB -> mono
            new_w = old_w.mean(1, keepdim=True)
        elif in_ch > 3:                     # replicate & scale
            reps = (in_ch + 2) // 3         # ceil
            new_w = old_w.repeat(1, reps, 1, 1, 1)[:, :in_ch]
            new_w *= 3.0 / in_ch            # keep variance similar
        else:                               # e.g. 2-channel
            new_w = old_w[:, :in_ch]

        new_conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=stem_conv.out_channels,
            kernel_size=stem_conv.kernel_size,
            stride=stem_conv.stride,
            padding=stem_conv.padding,
            bias=False,
        )
        new_conv.weight = nn.Parameter(new_w)
        full.features[0][0][0] = new_conv

    # 4) strip original classifier and add ours
    backbone = nn.Sequential(*full.features)
    feat_dim = full.classifier[1].in_channels  # 1024 for S3D

    class S3DHead(nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core
            self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.head = nn.Linear(feat_dim, num_classes)
        def forward(self, x):
            x = self.core(x)            # B, C, T', H', W'
            x = self.pool(x).flatten(1) # B, C
            return self.head(x)

    return S3DHead(backbone)

def load_r2plus1d_18(pretrained_path: str | None = None,
                     num_classes: int = 2) -> nn.Module:
    import os, torch, torch.nn as nn

    # 1️⃣ build bare model (no hub download)
    full = r2plus1d_18(weights=None)

    # 2️⃣ optional checkpoint (load before surgery)
    if pretrained_path:
        print("🚀 Loading pretrained weights from:", pretrained_path)
        ckpt = pretrained_path
        if os.path.isdir(pretrained_path):
            files = [f for f in os.listdir(pretrained_path)
                     if f.lower().endswith((".pth", ".pt", ".ckpt", ".pyth"))]
            if not files:
                raise FileNotFoundError(f"No ckpt in {pretrained_path}")
            if len(files) > 1:
                print(f"⚠️  Multiple ckpts — using {files[0]}")
            ckpt = os.path.join(pretrained_path, files[0])
        full.load_state_dict(torch.load(ckpt, map_location="cpu"),
                             strict=False)
    else:
        print("⚙️  Model initialized without pretrained weights.")

    # 3️⃣ collapse RGB → 1-channel in the stem
    stem_conv         = full.stem[0]                 # Conv3d
    w_rgb             = stem_conv.weight             # (64,3,3,7,7)
    w_gray            = w_rgb.mean(1, keepdim=True)  # (64,1,3,7,7)
    new_stem_conv     = nn.Conv3d(
        1, 64, kernel_size=stem_conv.kernel_size,
        stride=stem_conv.stride, padding=stem_conv.padding, bias=False
    )
    new_stem_conv.weight = nn.Parameter(w_gray)
    full.stem[0] = new_stem_conv

    # 4️⃣ build backbone = stem + layer1…layer4  (no pool / flatten)
    backbone = nn.Sequential(
        full.stem,
        full.layer1,
        full.layer2,
        full.layer3,
        full.layer4,
    )

    feat_dim = full.fc.in_features   # 512

    class R2Plus1dHead(nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core   = core
            self.pool   = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.classif= nn.Linear(feat_dim, num_classes)
        def forward(self, x):
            x = self.core(x)                 # B,C,T',H',W'
            x = self.pool(x).flatten(1)      # B,C
            return self.classif(x)

    return R2Plus1dHead(backbone)

def load_r3d_18(pretrained_path: str | None = None, num_classes: int = 2) -> nn.Module:
    """
    Loads ResNet3D-18, adapts it to 1-channel input by collapsing the
    pretrained 3-channel stem weights, skips loading the original fc,
    then swaps in a new Linear head.
    
    Args:
      pretrained_path: None, or path to a folder (one .pth/.pt/.ckpt inside)
                       or directly to a checkpoint file.
      num_classes:     number of output classes for the new head.
    """
    # 1) Build a bare r3d_18 (no built-in weights)
    model = r3d_18(weights=None)

    # 2) If user supplied a checkpoint, load it and patch stem from it
    if pretrained_path is not None:
        print("🚀 Loading pretrained weights from:", pretrained_path)
        # Resolve file
        if os.path.isdir(pretrained_path):
            candidates = [f for f in os.listdir(pretrained_path)
                          if f.lower().endswith((".pth",".pt",".ckpt"))]
            if not candidates:
                raise FileNotFoundError(f"No checkpoint in {pretrained_path}")
            if len(candidates) > 1:
                print(f"⚠️  Multiple files found—using {candidates[0]}")
            ckpt_file = os.path.join(pretrained_path, candidates[0])
        elif os.path.isfile(pretrained_path):
            ckpt_file = pretrained_path
        else:
            raise FileNotFoundError(f"Pretrained path '{pretrained_path}' not found")

        state = torch.load(ckpt_file, map_location="cpu")

        # 2a) Collapse the 3-channel stem weight by mean → 1-channel
        conv1_w = state.get("stem.0.weight")
        if conv1_w is None:
            raise KeyError("Checkpoint missing 'stem.0.weight'")
        # mean over the channel dimension
        collapsed = conv1_w.mean(dim=1, keepdim=True)  # (64,1,3,7,7)

        # 2b) Patch the model.stem[0] conv to 1→64 with the collapsed weight
        orig = model.stem[0]
        new_stem = nn.Conv3d(
            in_channels=1,
            out_channels=orig.out_channels,
            kernel_size=orig.kernel_size,
            stride=orig.stride,
            padding=orig.padding,
            bias=False
        )
        new_stem.weight = nn.Parameter(collapsed)
        model.stem[0] = new_stem

        # 2c) Remove stem & fc entries so we can load the rest
        for k in ["stem.0.weight", "stem.0.bias", "fc.weight", "fc.bias"]:
            state.pop(k, None)

        # 2d) Load remaining params (strict=False to ignore missing keys)
        model.load_state_dict(state, strict=False)

    else:
        print("⚙️  Model initialized without pretrained weights.")
        #  If you still want to collapse the random init to 1-channel:
        orig = model.stem[0]
        w_orig = orig.weight
        collapsed = w_orig.mean(dim=1, keepdim=True)
        new_stem = nn.Conv3d(
            in_channels=1,
            out_channels=orig.out_channels,
            kernel_size=orig.kernel_size,
            stride=orig.stride,
            padding=orig.padding,
            bias=False
        )
        new_stem.weight = nn.Parameter(collapsed)
        model.stem[0] = new_stem

    # 3) Now replace the classifier head with fresh Linear
    feat_dim      = model.fc.in_features
    model.fc      = nn.Linear(feat_dim, num_classes)

    return model

# def load_slowfast_r50(pretrained_path: str | None = None,
#                   num_classes: int = 2,
#                   alpha: int = 4,
#                   dummy_depth: int = 32,
#                   dummy_hw: int = 128) -> nn.Module:
#     base = slowfast_r50(pretrained=False)

#     # Load weights first
#     if pretrained_path is not None:
#         print("🚀 Loading pretrained weights from:", pretrained_path)
#         if os.path.isdir(pretrained_path):
#             ckpts = [f for f in os.listdir(pretrained_path) if f.lower().endswith((".pth", ".pt", ".ckpt", ".pyth"))]
#             if not ckpts:
#                 raise FileNotFoundError(f"No checkpoint found in {pretrained_path}")
#             if len(ckpts) > 1:
#                 print(f"⚠️  Multiple files found—using {ckpts[0]}")
#             ckpt_file = os.path.join(pretrained_path, ckpts[0])
#         else:
#             ckpt_file = pretrained_path
#         state = torch.load(ckpt_file, map_location="cpu")
#         base.load_state_dict(state, strict=False)
#     else:
#         print("⚙️  Model initialized without pretrained weights.")

#     # Adapt both stems to 1-channel input
#     for path_i in [0, 1]:  # 0: slow, 1: fast
#         conv = base.blocks[0].multipathway_blocks[path_i].conv
#         w = conv.weight  # (C_out, 3, kt, kh, kw)
#         w1 = w.mean(dim=1, keepdim=True)  # (C_out, 1, kt, kh, kw)
#         new_stem = nn.Conv3d(
#             in_channels=1,
#             out_channels=conv.out_channels,
#             kernel_size=conv.kernel_size,
#             stride=conv.stride,
#             padding=conv.padding,
#             bias=False
#         )
#         new_stem.weight = nn.Parameter(w1)
#         base.blocks[0].multipathway_blocks[path_i].conv = new_stem

#     # Wrap blocks[0:5] and infer channel dimensions
#     backbone = nn.Sequential(*base.blocks[:5])
#     with torch.no_grad():
#         dummy = torch.zeros(1, 1, dummy_depth, dummy_hw, dummy_hw)
#         slow = dummy[:, :, ::alpha]
#         fast = dummy
#         s_out, f_out = backbone([slow, fast])
#         C_s, C_f = s_out.shape[1], f_out.shape[1]

#     class SlowFastHead(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.backbone = backbone
#             self.pool_s = nn.AdaptiveAvgPool3d((1, 1, 1))
#             self.pool_f = nn.AdaptiveAvgPool3d((1, 1, 1))
#             self.classifier = nn.Linear(C_s + C_f, num_classes)

#         def forward(self, x):
#             s = x[:, :, ::alpha]
#             f = x
#             s_out, f_out = self.backbone([s, f])
#             s_feat = self.pool_s(s_out).view(x.size(0), -1)
#             f_feat = self.pool_f(f_out).view(x.size(0), -1)
#             return self.classifier(torch.cat([s_feat, f_feat], dim=1))

#     return SlowFastHead()

# def load_slowfast_r101(pretrained_path: str | None = None,
#                        num_classes: int = 2,
#                        alpha: int = 4,
#                        dummy_depth: int = 32,
#                        dummy_hw: int = 128) -> nn.Module:
#     """
#     SlowFast-R101 adapted for grayscale 3-D CT volumes.
#     """
#     base = slowfast_r101(pretrained=False)        # never auto-download

#     # ---------- load checkpoint ----------
#     if pretrained_path:
#         print("🚀 Loading pretrained weights from:", pretrained_path)
#         ckpt = pretrained_path
#         if os.path.isdir(pretrained_path):
#             files = [f for f in os.listdir(pretrained_path)
#                      if f.lower().endswith((".pth", ".pt", ".ckpt", ".pyth"))]
#             if not files:
#                 raise FileNotFoundError(f"No ckpt in {pretrained_path}")
#             if len(files) > 1:
#                 print(f"⚠️  Multiple ckpts — using {files[0]}")
#             ckpt = os.path.join(pretrained_path, files[0])
#         base.load_state_dict(torch.load(ckpt, map_location="cpu"),
#                              strict=False)
#     else:
#         print("⚙️  Model initialized without pretrained weights.")

#     # ---------- collapse both stems to 1 channel ----------
#     for pathway in (0, 1):                                    # 0 slow, 1 fast
#         conv = base.blocks[0].multipathway_blocks[pathway].conv
#         w_rgb = conv.weight                                    # (C_out, 3, …)
#         w_g   = w_rgb.mean(1, keepdim=True)                    # (C_out, 1, …)
#         new_conv = nn.Conv3d(
#             1, conv.out_channels, kernel_size=conv.kernel_size,
#             stride=conv.stride, padding=conv.padding, bias=False
#         )
#         new_conv.weight = nn.Parameter(w_g)
#         base.blocks[0].multipathway_blocks[pathway].conv = new_conv

#     # ---------- build backbone (blocks 0–4) ----------
#     backbone = nn.Sequential(*base.blocks[:5])

#     # ---------- infer channel dims ----------
#     with torch.no_grad():
#         dummy = torch.zeros(1, 1, dummy_depth, dummy_hw, dummy_hw)
#         slow  = dummy[:, :, ::alpha]        # temporal stride for slow path
#         fast  = dummy
#         s_out, f_out = backbone([slow, fast])
#         C_s, C_f = s_out.shape[1], f_out.shape[1]

#     # ---------- classification head ----------
#     class SlowFastR101Head(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.backbone = backbone
#             self.pool_s   = nn.AdaptiveAvgPool3d((1, 1, 1))
#             self.pool_f   = nn.AdaptiveAvgPool3d((1, 1, 1))
#             self.classif  = nn.Linear(C_s + C_f, num_classes)
#         def forward(self, x):
#             s = x[:, :, ::alpha]
#             f = x
#             s_out, f_out = self.backbone([s, f])
#             s_feat = self.pool_s(s_out).flatten(1)
#             f_feat = self.pool_f(f_out).flatten(1)
#             return self.classif(torch.cat([s_feat, f_feat], dim=1))

#     return SlowFastR101Head()

# def load_i3d(pretrained_path: str | None = None, num_classes: int = 2) -> nn.Module:
#     base = i3d_r50(pretrained=False)

#     # Load pretrained weights if provided
#     if pretrained_path is not None:
#         print("🚀 Loading pretrained weights from:", pretrained_path)
#         if os.path.isdir(pretrained_path):
#             ckpts = [f for f in os.listdir(pretrained_path)
#                      if f.lower().endswith((".pth", ".pt", ".ckpt", ".pyth"))]
#             if not ckpts:
#                 raise FileNotFoundError(f"No checkpoint found in {pretrained_path}")
#             if len(ckpts) > 1:
#                 print(f"⚠️  Multiple files found—using {ckpts[0]}")
#             ckpt_file = os.path.join(pretrained_path, ckpts[0])
#         else:
#             ckpt_file = pretrained_path
#         state = torch.load(ckpt_file, map_location="cpu")
#         base.load_state_dict(state, strict=False)
#     else:
#         print("⚙️  Model initialized without pretrained weights.")

#     # Adapt 3-channel stem to 1-channel
#     w_orig = base.blocks[0].conv.weight  # (64,3,5,7,7)
#     w_1ch = w_orig.mean(dim=1, keepdim=True)  # (64,1,5,7,7)
#     new_stem = nn.Conv3d(
#         in_channels=1,
#         out_channels=64,
#         kernel_size=(5, 7, 7),
#         stride=(1, 2, 2),
#         padding=(2, 3, 3),
#         bias=False
#     )
#     new_stem.weight = nn.Parameter(w_1ch)
#     base.blocks[0].conv = new_stem

#     # Replace classifier head
#     feat_dim = base.blocks[-1].proj.in_features
#     class I3DHead(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.backbone = nn.Sequential(*base.blocks[:-1])
#             self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
#             self.classifier = nn.Linear(feat_dim, num_classes)

#         def forward(self, x):
#             x = self.backbone(x)
#             x = self.pool(x).view(x.size(0), -1)
#             return self.classifier(x)

#     return I3DHead()


# Registry of available video models
VIDEO_BACKBONES = {
    #"i3d_r50":      load_i3d,
    #"slowfast_r50": load_slowfast_r50,
    #"slowfast_r101": load_slowfast_r101,
    "r3d_18":       load_r3d_18,
    "r2plus1d_18":  load_r2plus1d_18,
    "s3d":          load_s3d,
    "mc3_18":       load_mc3_18,
    "swin3d_t":     load_swin3d_t,
    "swin3d_s":     load_swin3d_s,
    "swin3d_b":     load_swin3d_b,
    "mvit_v2_s":    load_mvit_v2_s,
    "mvit_v1_b":    load_mvit_v1_b, 
    #"x3d_s":        load_x3d_s,
    #"x3d_xs":       load_x3d_xs,
}


def set_global_seed(seed: int | None):
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



import hashlib
import os
from pathlib import Path
from typing import Optional

import gdown
def _extract_google_drive_id(google_drive_path: str) -> str:
    """
    Accept either a Google Drive file ID or a typical Google Drive sharing URL,
    and return the file ID.
    """
    google_drive_path = google_drive_path.strip()

    # If it already looks like a raw file ID, use it directly
    if "/" not in google_drive_path and "http" not in google_drive_path:
        return google_drive_path

    patterns = [
        "/file/d/",
        "id=",
    ]

    if "/file/d/" in google_drive_path:
        return google_drive_path.split("/file/d/")[1].split("/")[0]

    if "id=" in google_drive_path:
        return google_drive_path.split("id=")[1].split("&")[0]

    raise ValueError(
        f"Could not extract Google Drive file ID from: {google_drive_path}"
    )


def _sha256sum(file_path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute SHA-256 for a file.
    """
    file_path = Path(file_path)
    sha256 = hashlib.sha256()

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def download_verify_pretrained_model(
    google_drive_path: str = "https://drive.google.com/file/d/1gFLJmwWsfGAXnApacKjf6jnL3pVkaF5j/view?usp=sharing",
    expected_sha256: str = "ae3be16733081f6d1cd40e4ab980ca23d6df6dc6486d15ada05a5e8ab8c9b975",
    cache_dir: str | Path = "./models",
    output_filename: str = "mvit_v2_s-ae3be167.pth",
    force_download: bool = False,
) -> str:
    """
    Download a pretrained model from Google Drive, verify its SHA-256 hash,
    and return the local file path.

    Args:
        google_drive_path:
            Google Drive file URL or file ID.
        expected_sha256:
            Expected SHA-256 checksum of the model file.
        cache_dir:
            Local directory where the model will be stored.
        output_filename:
            Local filename for the downloaded model.
        force_download:
            If True, re-download even if the file already exists.

    Returns:
        str: Local path to the verified checkpoint file.

    Raises:
        ValueError: If the downloaded file hash does not match expected_sha256.
        FileNotFoundError: If download did not produce the expected file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_path = cache_dir / output_filename
    expected_sha256 = expected_sha256.lower().strip()

    # Reuse existing file if present and valid
    if output_path.exists() and not force_download:
        actual_sha256 = _sha256sum(output_path)
        if actual_sha256 == expected_sha256:
            return str(output_path)
        else:
            print(
                f"Existing file hash mismatch for {output_path}. "
                f"Expected {expected_sha256}, got {actual_sha256}. Re-downloading."
            )
            output_path.unlink()

    file_id = _extract_google_drive_id(google_drive_path)

    # Use gdown's uc URL form
    download_url = f"https://drive.google.com/uc?id={file_id}"

    gdown.download(
        url=download_url,
        output=str(output_path),
        quiet=False,
        fuzzy=True,
    )

    if not output_path.exists():
        raise FileNotFoundError(f"Download failed, file not found: {output_path}")

    actual_sha256 = _sha256sum(output_path)
    if actual_sha256 != expected_sha256:
        # Remove bad file so it is not reused accidentally
        output_path.unlink(missing_ok=True)
        raise ValueError(
            "Downloaded model hash mismatch.\n"
            f"Expected: {expected_sha256}\n"
            f"Actual:   {actual_sha256}"
        )

    return str(output_path)

# ----------------------- Factory -----------------------
def model_factory(
    arch: str,
    pretrained_path: str | None = None,
    num_classes: int = 2,
    in_ch: int = 3,
    freeze_backbone: bool = False,
    seed: int | None = 42,
    **classifier_kwargs,
) -> BasicClassifier:  # TODO adaption: previous -> nn.Module:
    """
    Builds and returns a video classification model for 3D medical volumes.

    Args:
        arch: Key of the model in the registry.
        pretrained_path: Local path to pretrained weights.
        num_classes: Number of output classes.
        freeze: If True, freezes all model parameters.
        seed: Random seed for reproducibility.
    """
    set_global_seed(seed)
    arch = arch.lower()
    if arch not in VIDEO_BACKBONES:
        raise ValueError(f"Unknown architecture '{arch}'. Available: {list(VIDEO_BACKBONES.keys())}")
    
    cache_dir = './models'
    if not os.path.isabs(pretrained_path):
        cache_dir = os.path.dirname(os.path.abspath(__file__))
        pretrained_path = os.path.join(cache_dir, pretrained_path)

    pretrained_path = download_verify_pretrained_model(cache_dir=cache_dir, output_filename=pretrained_path)
    model = VIDEO_BACKBONES[arch](pretrained_path=pretrained_path, num_classes=num_classes, in_ch=in_ch)
    
    def freeze_backbone_only(model: nn.Module) -> None:
        # 1) freeze everything
        for p in model.parameters():
            p.requires_grad = False

        # 2) un-freeze the *last* nn.Linear encountered
        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Linear):
                for p in m.parameters():
                    p.requires_grad = True
                break

    if freeze_backbone:
        freeze_backbone_only(model)

    # if freeze_backbone:
    #     for name, param in model.named_parameters():
    #         if not name.startswith("classifier"):
    #             param.requires_grad = False    

    # TODO adaption: added: 
    wrapped_model = ModelWrapper(backbone=model, in_ch=in_ch, num_classes=num_classes, **classifier_kwargs)        
    return wrapped_model   # TODO adaption: previous -> model

# ----------------------- CLI Smoke Test -----------------------
# Example:
#   python utils/model_factory.py --arch i3d_r50      --pretrained pretrained_weights/i3d_r50/
#   python utils/model_factory.py --arch slowfast_r50 --pretrained pretrained_weights/slowfast_r50/
#   python utils/model_factory.py --arch slowfast_r101 --pretrained pretrained_weights/slowfast_r101/
#   python utils/model_factory.py --arch r3d_18 --pretrained pretrained_weights/r3d_18/
#   python utils/model_factory.py --arch r2plus1d_18 --pretrained pretrained_weights/r2plus1d_18/
#   python utils/model_factory.py --arch s3d --in_ch 3  --pretrained pretrained_weights/s3d/
#   python utils/model_factory.py --arch mc3_18 --in_ch 3 --pretrained pretrained_weights/mc3_18/
#   python utils/model_factory.py --arch swin3d_t --pretrained pretrained_weights/swin3d_t/
#   python utils/model_factory.py --arch swin3d_s --pretrained pretrained_weights/swin3d_s/
#   python utils/model_factory.py --arch mvit_v2_s --in_ch 3 --pretrained pretrained_weights/mvit_v2_s/
#   python utils/model_factory.py --arch mvit_v1_b --in_ch 3 --pretrained pretrained_weights/mvit_v1_b/
#   python utils/model_factory.py --arch timesformer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video model factory smoke test for 3D volumes.")
    parser.add_argument("--arch", type=str, required=True, help="Model key: " + ", ".join(VIDEO_BACKBONES.keys()))
    parser.add_argument("--pretrained", type=str, default=None, help="Path to local pretrained weights.")
    parser.add_argument("--num_classes", type=int, default=1, help="Number of classes.")
    parser.add_argument("--in_ch", type=int, default=3, help="Number of classes.")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze model weights.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on, e.g., 'cpu' or 'cuda:0'.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for dummy input.")
    parser.add_argument("--depth", type=int, default=16, help="Temporal depth (z-axis slices).")
    parser.add_argument("--height", type=int, default=64, help="Spatial height.")
    parser.add_argument("--width", type=int, default=64, help="Spatial width.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    args = parser.parse_args()

    print(f"🔧 Building model '{args.arch}'...")

    model = model_factory(
        arch=args.arch,
        pretrained_path=args.pretrained,
        num_classes=args.num_classes,
        in_ch = args.in_ch,
        freeze_backbone=args.freeze_backbone,
        seed=args.seed,
    )
    model = model.to(args.device)
    
    # ------------- count parameters -------------
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ------------- warm-up & timing -------------
    dummy = torch.randn(
        args.batch_size, args.in_ch,
        args.depth, args.height, args.width,
        device=args.device
    )
    model.eval()
    with torch.no_grad():
        # warm-up
        for _ in range(2):
            _ = model(dummy)

        # timed passes
        t0 = time.perf_counter()
        for _ in range(5):
            _ = model(dummy)
        elapsed = (time.perf_counter() - t0) / 5 * 1e3   # → ms

    # # ------------- FLOPs (optional) -------------
    # flops_str = ""
    # with contextlib.suppress(ModuleNotFoundError):
    #     from fvcore.nn import FlopCountAnalysis, flop_count_table
    #     flops = FlopCountAnalysis(model, dummy).total()
    #     flops_str = f"\n📊 FLOPs (1 pass): {flops/1e9:,.2f} G"

    # ------------- pretty summary -------------
    out = model(dummy)
    print(f"✅ Output shape : {tuple(out.shape)}")
    print(f"📦 Params       : {total_params:,} "
          f"({trainable_params:,} trainable)")
    print(f"⚡ Fwd-time     : {elapsed:,.1f} ms on {args.device}")

    