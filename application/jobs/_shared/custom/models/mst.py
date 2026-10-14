import contextlib
import contextvars
import hashlib
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from x_transformers import Encoder

from .base_model import BasicClassifier

logger = logging.getLogger(__name__)

# SAM-Med2D ViT-B encoder (contributed by Zhongying Deng, Cambridge): the published checkpoint
# was trained at 256 x 256 with the encoder adapter enabled, so both are fixed here.
SAM_MED2D_IMAGE_SIZE = 256
SAM_MED2D_CHECKPOINT_NAME = "sam-med2d_b.pth"
SAM_MED2D_CHECKPOINT_CANDIDATES = (
    "/MediSwarm/pretrained_weights/" + SAM_MED2D_CHECKPOINT_NAME,
    os.path.join(os.environ.get("TORCH_HOME", "/torch_home"), "hub", "checkpoints", SAM_MED2D_CHECKPOINT_NAME),
)
# sha256 of the SAM-Med2D release file (2,561,829,701 bytes; the copy in jefftud/odelia:1.8.1-cam.261009).
# Logged against, not enforced: whether a file is usable is decided by the encoder key check below.
SAM_MED2D_RELEASE_SHA256 = "921d8c130cc4bcc4b7220c064e60f207f15ad59f17ab36de340e41c67a490d18"
# The only way to run MST_SAMMed2D without its pretrained encoder weights: unit tests and CI images
# built without the file. Never for a real run.
SAM_MED2D_ALLOW_RANDOM_INIT_ENV = "SAM_MED2D_ALLOW_RANDOM_INIT"
_SAM_ENCODER_PREFIX = "image_encoder."
_SAM_MED2D_FIX = (
    "Fix: run an image that ships /MediSwarm/pretrained_weights/" + SAM_MED2D_CHECKPOINT_NAME
    + " (restart the client so that it pulls the current image), or set SAM_MED2D_CHECKPOINT to a copy of"
    " the SAM-Med2D release file (sha256 " + SAM_MED2D_RELEASE_SHA256 + "). " + SAM_MED2D_ALLOW_RANDOM_INIT_ENV
    + "=1 starts from random encoder weights instead; it exists for tests only."
)
# Set (with a reason) while a caller builds MST_SAMMed2D only to load a complete checkpoint into it.
_ARCHITECTURE_ONLY = contextvars.ContextVar("sam_med2d_architecture_only", default=None)


def resolve_sam_med2d_checkpoint():
    """Path of the SAM-Med2D weights, or None if none is present.

    `SAM_MED2D_CHECKPOINT` wins when set; otherwise the image's pretrained-weights directory and
    the torch hub cache are tried, in that order.
    """
    env = os.environ.get("SAM_MED2D_CHECKPOINT")
    if env:
        if os.path.isfile(env):
            return env
        raise FileNotFoundError(f"SAM_MED2D_CHECKPOINT={env} does not exist. {_SAM_MED2D_FIX}")
    for candidate in SAM_MED2D_CHECKPOINT_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return None


def sam_med2d_random_init_allowed():
    return os.environ.get(SAM_MED2D_ALLOW_RANDOM_INIT_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


@contextlib.contextmanager
def sam_med2d_architecture_only(reason):
    """Build MST_SAMMed2D inside this block without reading the pretrained SAM-Med2D weights.

    Only for callers that load a COMPLETE MST_SAMMed2D state dict right after construction and
    check that every key matched (scripts/evaluation/predict.py): every pretrained weight would be
    overwritten anyway, so reading and checking the 2.56 GB file would only cost time, and an
    evaluation must not fail because an image lacks a file the evaluated checkpoint replaces.
    `reason` is logged instead of the weights line.
    """
    token = _ARCHITECTURE_ONLY.set(reason)
    try:
        yield
    finally:
        _ARCHITECTURE_ONLY.reset(token)


def _new_sam_med2d_encoder():
    """The SAM-Med2D ViT-B image encoder with adapters, randomly initialised; no weights file is read."""
    from .segment_anything import sam_model_registry

    class _Args:
        image_size = SAM_MED2D_IMAGE_SIZE
        sam_checkpoint = None  # build_sam_med2d_encoder loads and checks the weights itself
        encoder_adapter = True

    sam = sam_model_registry["vit_b"](_Args())
    return sam.image_encoder  # prompt encoder and mask decoder are dropped with `sam`


def _sha256(path, block_size=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _encoder_state_from_checkpoint(path):
    """The image-encoder tensors of a SAM(-Med2D) checkpoint, keyed as in the encoder's own state dict."""
    # weights_only=False keeps the loading the SAM-Med2D code always did (torch 2.2's default): the
    # release file also carries the optimizer state. It comes from the image or from
    # SAM_MED2D_CHECKPOINT, both set by whoever runs the site.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise TypeError(f"expected a state dict or {{'model': state_dict}}, found {type(checkpoint).__name__}")
    prefix = _SAM_ENCODER_PREFIX
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


def _first(keys, limit=5):
    keys = list(keys)
    return ", ".join(keys[:limit]) + (f", ... ({len(keys) - limit} more)" if len(keys) > limit else "")


def _encoder_mismatch(encoder, state):
    """None if `state` fits `encoder` exactly, else a description of every difference."""
    expected = encoder.state_dict()
    missing = [key for key in expected if key not in state]
    unexpected = [key for key in state if key not in expected]
    reshaped = [
        f"{key} (file {tuple(getattr(state[key], 'shape', ()))}, model {tuple(expected[key].shape)})"
        for key in expected
        if key in state and tuple(getattr(state[key], "shape", ())) != tuple(expected[key].shape)
    ]
    problems = []
    if missing:
        problems.append(f"{len(missing)} of {len(expected)} encoder tensors missing ({_first(missing)})")
    if unexpected:
        problems.append(f"{len(unexpected)} unexpected encoder tensors ({_first(unexpected)})")
    if reshaped:
        problems.append(f"{len(reshaped)} encoder tensors with another shape ({_first(reshaped, 3)})")
    return "; ".join(problems) or None


def _refuse_or_allow_random(problem, cause=None):
    """Raise RuntimeError for `problem`, unless SAM_MED2D_ALLOW_RANDOM_INIT=1 explicitly allows random weights."""
    if not sam_med2d_random_init_allowed():
        raise RuntimeError(f"MST_SAMMed2D refuses to start: {problem}. {_SAM_MED2D_FIX}") from cause
    banner = "*" * 100
    logger.warning(
        "\n%s\n*** %s=1: the MST_SAMMed2D backbone starts from RANDOM weights.\n*** Why: %s.\n"
        "*** This override is for tests only: a model trained like this is not the pretrained SAM-Med2D model.\n%s",
        banner, SAM_MED2D_ALLOW_RANDOM_INIT_ENV, problem, banner,
    )


def build_sam_med2d_encoder():
    """The SAM-Med2D ViT-B image encoder with adapter layers, initialised from the pretrained weights.

    Refuses to start without them. Only the swarm's starting client seeds round 0, so a single site
    without the file used to start every site from a random encoder, with nothing but a warning in
    one log, and the run still ended as COMPLETED. A missing file, an unreadable one, or one whose
    image-encoder tensors do not match this encoder exactly (missing, unexpected or reshaped) now
    raises RuntimeError naming the file(s) and the fix; the old loader accepted partial and
    non-strict loads silently. SAM_MED2D_ALLOW_RANDOM_INIT=1 is the only override, and it yields a
    fully random encoder: a partially loaded one is never returned.
    """
    reason = _ARCHITECTURE_ONLY.get()
    if reason:
        logger.info("SAM-Med2D pretrained weights not read: %s", reason)
        return _new_sam_med2d_encoder()

    checkpoint = resolve_sam_med2d_checkpoint()
    if checkpoint is None:
        searched = ", ".join(["SAM_MED2D_CHECKPOINT (not set)", *SAM_MED2D_CHECKPOINT_CANDIDATES])
        _refuse_or_allow_random(
            f"the pretrained SAM-Med2D weights ({SAM_MED2D_CHECKPOINT_NAME}) were not found; searched {searched}"
        )
        return _new_sam_med2d_encoder()

    size = os.path.getsize(checkpoint)
    digest = _sha256(checkpoint)  # read before torch.load, which then reads from the page cache
    described = f"{checkpoint} ({size:,} bytes, sha256 {digest})"
    try:
        state = _encoder_state_from_checkpoint(checkpoint)
    except Exception as exc:  # torch.load of a truncated or foreign file fails in many different ways
        _refuse_or_allow_random(
            f"{described} cannot be read as a SAM-Med2D checkpoint ({type(exc).__name__}: {exc}); "
            "a truncated copy fails like this", cause=exc,
        )
        return _new_sam_med2d_encoder()

    encoder = _new_sam_med2d_encoder()
    mismatch = _encoder_mismatch(encoder, state)
    if mismatch:
        _refuse_or_allow_random(
            f"{described} does not fit the SAM-Med2D ViT-B encoder with adapters at "
            f"{SAM_MED2D_IMAGE_SIZE} px: {mismatch}"
        )
        return encoder  # nothing was loaded into it: still entirely random
    encoder.load_state_dict(state, strict=True)
    logger.info(
        "SAM-Med2D backbone weights: %s (%s bytes, sha256 %s); loaded %d/%d encoder tensors",
        checkpoint, f"{size:,}", digest, len(state), len(state),
    )
    if digest != SAM_MED2D_RELEASE_SHA256:
        logger.warning(
            "SAM-Med2D weights %s are not the published release file (sha256 %s); every encoder tensor "
            "matched, so they are used.", checkpoint, SAM_MED2D_RELEASE_SHA256,
        )
    return encoder


class TransformerEncoder(Encoder):
    """Override the default forward to match input formatting."""

    def forward(self, x, mask=None, src_key_padding_mask=None):
        src_key_padding_mask = ~src_key_padding_mask if src_key_padding_mask is not None else None
        mask = ~mask if mask is not None else None
        return super().forward(x=x, context=None, mask=src_key_padding_mask, context_mask=None, attn_mask=mask)


class _MST(nn.Module):
    """Multi-slice transformer for 3D volume input classification or regression."""

    def __init__(
            self,
            out_ch=1,
            backbone_type="dinov2",
            model_size=None,
            slice_fusion_type="transformer"
    ):
        super().__init__()
        self.backbone_type = backbone_type
        self.slice_fusion_type = slice_fusion_type

        if backbone_type == "dinov2":
            torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
            self.backbone = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_size}14')
            self.backbone.mask_token = None
            emb_ch = self.backbone.num_features
        elif backbone_type == "sammed2d":
            self.backbone = build_sam_med2d_encoder()
            emb_ch = 256  # the encoder neck's output channels (B, 256, 16, 16) at 256 px
        else:
            raise ValueError("Unknown backbone_type")

        self.emb_ch = emb_ch

        if slice_fusion_type == "transformer":
            self.slice_fusion = TransformerEncoder(
                dim=emb_ch,
                heads=12 if emb_ch % 12 == 0 else 8,
                ff_mult=1,
                attn_dropout=0.0,
                pre_norm=True,
                depth=1,
                attn_flash=True,
                ff_no_bias=True,
                rotary_pos_emb=True,
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, emb_ch))
        elif slice_fusion_type in ["average", "none"]:
            self.slice_fusion = None
        else:
            raise ValueError("Unknown slice_fusion_type")

        self.linear = nn.Linear(emb_ch, out_ch)

    def forward(self, x):
        B, *_ = x.shape
        x = rearrange(x, 'b c d h w -> (b c d) h w')
        x = x[:, None].repeat(1, 3, 1, 1)  # Gray to RGB

        if self.backbone_type == "sammed2d":
            # The dataset delivers 224 x 224 slices; the SAM-Med2D encoder is fixed at 256 x 256.
            # Cambridge's experiment cropped to 256 at load time instead; resizing here keeps
            # the shared data pipeline unchanged for every other model.
            if x.shape[-2:] != (SAM_MED2D_IMAGE_SIZE, SAM_MED2D_IMAGE_SIZE):
                x = F.interpolate(x, size=(SAM_MED2D_IMAGE_SIZE, SAM_MED2D_IMAGE_SIZE),
                                  mode="bilinear", align_corners=False)
            x = self.backbone(x)  # (B * D, 256, 16, 16)
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B * D, 256)
        else:
            x = self.backbone(x)  # (B * D, E)
        x = rearrange(x, '(b d) e -> b d e', b=B)

        if self.slice_fusion_type == 'none':
            return x
        elif self.slice_fusion_type == 'transformer':
            x = torch.cat([x, self.cls_token.repeat(B, 1, 1)], dim=1)
            x = self.slice_fusion(x)
        elif self.slice_fusion_type == 'average':
            x = x.mean(dim=1, keepdim=True)

        x = self.linear(x[:, -1])
        return x


class MST(BasicClassifier):
    """MST-based classifier: a 2D slice encoder (DINOv2 ViT-S by default, or the SAM-Med2D
    ViT-B with `backbone_type="sammed2d"`) followed by transformer slice fusion."""

    def __init__(
            self,
            n_input_channels: int,
            num_classes: int,
            spatial_dims: int,
            backbone_type="dinov2",
            model_size="s",
            slice_fusion_type="transformer",
            optimizer_kwargs={'lr': 1e-6},
            **kwargs
    ):
        in_ch = 'in_ch'
        if in_ch in kwargs.keys():
            assert(kwargs[in_ch] == n_input_channels)  # avoid discrepancy
        out_ch = 'out_ch'
        if out_ch in kwargs.keys():
            assert(kwargs[out_ch] == num_classes)      # avoid discrepancy

        super().__init__(n_input_channels, num_classes, spatial_dims, optimizer_kwargs=optimizer_kwargs, **{k:v for k, v in kwargs.items() if k not in {in_ch, out_ch}})

        self.mst = _MST(out_ch=num_classes, backbone_type=backbone_type, model_size=model_size,
                        slice_fusion_type=slice_fusion_type)

    def forward(self, x):
        return self.mst(x)
