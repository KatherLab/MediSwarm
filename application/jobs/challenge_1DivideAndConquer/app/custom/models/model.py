from typing import List, Tuple, Type, Union

import torch
import torch.nn as nn
from models.base_model import BasicClassifier, ModelWrapper
from dynamic_network_architectures.building_blocks.helper import (
    convert_conv_op_to_dim,
    get_matching_dropout,
)
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from dynamic_network_architectures.building_blocks.residual_encoders import (
    ResidualEncoder,
)
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

import os
import gdown
import hashlib
import subprocess
from pathlib import Path


class ClassificationHead(nn.Module):
    """Global average pooling followed by linear classification layer."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResidualEncoderClsNetwork(nn.Module):
    """
    3D Residual Encoder for Classification.

    This is the core network architecture extracted from the nnunet4cls framework.
    It consists of a ResidualEncoder backbone with a classification head.
    """

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        block: Union[Type[BasicBlockD]] = BasicBlockD,
        bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
        stem_channels: int = None,
        final_layer_dropout: float = 0.0,
        squeeze_excitation: bool = False,
        squeeze_excitation_reduction_ratio: float = 1.0 / 16,
    ):
        super().__init__()

        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        if dropout_op_kwargs is not None and dropout_op is None:
            dropout_op = nn.Dropout

        assert len(n_blocks_per_stage) == n_stages, (
            "n_blocks_per_stage must have as many entries as we have "
            f"resolution stages. here: {n_stages}. "
            f"n_blocks_per_stage: {n_blocks_per_stage}"
        )
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), (
            "n_conv_per_stage_decoder must have one less entries "
            f"as we have resolution stages. here: {n_stages} "
            f"stages, so it should have {n_stages - 1} entries. "
            f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        )

        self.encoder = ResidualEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            block,
            bottleneck_channels,
            return_skips=False,
            disable_default_stem=False,
            stem_channels=stem_channels,
            squeeze_excitation=squeeze_excitation,
            squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio,
        )
        self.cls_head = ClassificationHead(features_per_stage[-1], num_classes)
        self.final_layer_dropout = get_matching_dropout(
            dimension=convert_conv_op_to_dim(conv_op)
        )(p=final_layer_dropout)

    def forward(self, x):
        skips = self.encoder(x)
        skips = self.final_layer_dropout(skips)
        return self.cls_head(skips)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


class ResidualEncoderClsLightning(BasicClassifier):
    """
    PyTorch Lightning wrapper for ResidualEncoderCls.
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 2,
        spatial_dims: int = 3,
        loss_kwargs: dict = {},
        optimizer=torch.optim.AdamW,
        optimizer_kwargs: dict = {"lr": 1e-4, "weight_decay": 1e-2},
        lr_scheduler=None,
        lr_scheduler_kwargs: dict = {},
        aucroc_kwargs: dict = {},
        acc_kwargs: dict = {},
        save_hyperparameters: bool = True,
        final_layer_dropout: float = 0.0,
    ):
        super().__init__(
            in_ch=in_ch,
            out_ch=out_ch,
            spatial_dims=spatial_dims,
            loss_kwargs=loss_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            aucroc_kwargs=aucroc_kwargs,
            acc_kwargs=acc_kwargs,
            save_hyperparameters=save_hyperparameters,
        )

        self.network_config = {
            "n_stages": 6,
            "features_per_stage": [32, 64, 128, 256, 320, 320],
            "conv_op": nn.Conv3d,
            "kernel_sizes": [
                [1, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
            ],
            "strides": [
                [1, 1, 1],
                [1, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
            ],
            "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
            "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
            "conv_bias": True,
            "norm_op": nn.InstanceNorm3d,
            "norm_op_kwargs": {"eps": 1e-05, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": nn.LeakyReLU,
            "nonlin_kwargs": {"inplace": True},
        }

        self.model = ResidualEncoderClsNetwork(
            input_channels=in_ch,
            num_classes=out_ch,
            final_layer_dropout=final_layer_dropout,
            **self.network_config,
        )

        self.model.apply(self.model.initialize)

    def forward(self, x, cond=None):
        return self.model(x)

    def load_pretrained_unet_encoder(
        self,
        checkpoint_path: str,
        strict: bool = False,
        map_location=None,
        verbose: bool = True,
        print_not_loaded_layers: bool = False,
        max_not_loaded_to_print: int = 20,
    ):
        import re
        from pathlib import Path

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location or "cpu",
            weights_only=False,
        )

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            pretrained_state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "network_weights" in checkpoint:
            pretrained_state_dict = checkpoint["network_weights"]
        else:
            pretrained_state_dict = checkpoint

        current_state_dict = self.state_dict()

        def filter_encoder_weights(key):
            patterns_to_keep = [
                r"^network\.encoder\.",
                r"^encoder\.",
            ]
            patterns_to_skip = [
                r"decoder",
                r"seg_layers",
                r"seg_head",
                r"output",
            ]

            for pattern in patterns_to_skip:
                if re.search(pattern, key, re.IGNORECASE):
                    return False

            for pattern in patterns_to_keep:
                if re.search(pattern, key):
                    return True

            return False

        def map_key(unet_key):
            mapped_key = unet_key.replace("network.encoder.", "model.encoder.")
            if mapped_key.startswith("encoder."):
                mapped_key = "model." + mapped_key
            return mapped_key

        filtered_weights = {}
        skipped_details = []

        for key, value in pretrained_state_dict.items():
            if filter_encoder_weights(key):
                mapped_key = map_key(key)
                if mapped_key in current_state_dict:
                    if current_state_dict[mapped_key].shape == value.shape:
                        filtered_weights[mapped_key] = value
                    else:
                        skipped_details.append(
                            {
                                "source_key": key,
                                "mapped_key": mapped_key,
                                "reason": (
                                    "shape_mismatch: "
                                    f"pretrained={tuple(value.shape)} model={tuple(current_state_dict[mapped_key].shape)}"
                                ),
                            }
                        )
                else:
                    skipped_details.append(
                        {
                            "source_key": key,
                            "mapped_key": mapped_key,
                            "reason": "mapped_key_not_found_in_classifier",
                        }
                    )
            else:
                skipped_details.append(
                    {
                        "source_key": key,
                        "mapped_key": None,
                        "reason": "filtered_out_non_encoder_or_decoder_head",
                    }
                )

        if verbose and print_not_loaded_layers and len(skipped_details) > 0:
            if max_not_loaded_to_print == -1:
                to_print = skipped_details
            else:
                to_print = skipped_details[: max(0, max_not_loaded_to_print)]
            for item in to_print:
                source_key = item["source_key"]
                mapped_key = item["mapped_key"]
                if mapped_key is None:
                    print(f"skip: {source_key}")
                else:
                    print(f"skip: {source_key} -> {mapped_key}")

        if len(filtered_weights) > 0:
            result = self.load_weights(
                filtered_weights,
                strict=strict,
                filter=lambda key: key in filtered_weights,
            )
            if verbose:
                print(
                    f"Loaded encoder weights: {len(filtered_weights)}/{len(pretrained_state_dict)}"
                )
            return result
        else:
            if verbose:
                print("No matching encoder weights found.")
            return self


def _extract_google_drive_id(google_drive_path: str) -> str:
    google_drive_path = google_drive_path.strip()

    if "/" not in google_drive_path and "http" not in google_drive_path:
        return google_drive_path

    if "/file/d/" in google_drive_path:
        return google_drive_path.split("/file/d/")[1].split("/")[0]

    if "id=" in google_drive_path:
        return google_drive_path.split("id=")[1].split("&")[0]

    raise ValueError(
        f"Could not extract Google Drive file ID from: {google_drive_path}"
    )


def _sha256sum(file_path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    file_path = Path(file_path)
    sha256 = hashlib.sha256()

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def _scp_fetch_from_cosmos(
    remote_filename: str,
    local_path: str | Path,
    remote_user: str = "jeff",
    remote_host: str = "172.24.4.65",
    remote_dir: str = "/home/mediswarm-upload/odelia_challenge_ckpt",
    port: int = 22,
) -> str:
    """
    Fetch checkpoint from 172.24.4.65 using SSH public key authentication.
    """
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    remote_path = f"{remote_user}@{remote_host}:{remote_dir.rstrip('/')}/{remote_filename}"

    cmd = [
        "scp",
        "-P", str(port),
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        remote_path,
        str(local_path),
    ]

    print(f"Fetching checkpoint via SCP: {remote_path} -> {local_path}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"SCP failed for remote file {remote_path}. "
            f"Make sure the SSH key is available in this environment and the file exists."
        ) from e

    if not local_path.exists():
        raise FileNotFoundError(f"SCP finished but file not found locally: {local_path}")

    return str(local_path)


def download_verify_pretrained_model(
    google_drive_path: str = "https://drive.google.com/file/d/1bVmZHvI7H1H9YTIMy11zwU2p95W4Y_W6/view?usp=sharing",
    expected_sha256: str = "ed686907205fb0cb752dc987851eb9d0191034599c5d204c7ec1ad9ff91dd758",
    cache_dir: str | Path = "./models",
    output_filename: str = "checkpoint_final.pth",
    force_download: bool = False,
    try_scp_first: bool = True,
) -> str:
    """
    Resolve a pretrained model in this order:
    1) existing local file with valid SHA256
    2) SCP from 172.24.4.65 via public-key SSH
    3) Google Drive download
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_filename = Path(output_filename).name
    output_path = cache_dir / output_filename
    expected_sha256 = expected_sha256.lower().strip()

    if output_path.exists() and not force_download:
        actual_sha256 = _sha256sum(output_path)
        if actual_sha256 == expected_sha256:
            print(f"Using existing verified checkpoint: {output_path}")
            return str(output_path)
        else:
            print(
                f"Existing file hash mismatch for {output_path}. "
                f"Expected {expected_sha256}, got {actual_sha256}. Removing and re-resolving."
            )
            output_path.unlink()

    if try_scp_first:
        try:
            _scp_fetch_from_cosmos(
                remote_filename=output_filename,
                local_path=output_path,
            )
            actual_sha256 = _sha256sum(output_path)
            if actual_sha256 == expected_sha256:
                print(f"Using SCP-fetched verified checkpoint: {output_path}")
                return str(output_path)
            else:
                print(
                    f"SCP-fetched file hash mismatch for {output_path}. "
                    f"Expected {expected_sha256}, got {actual_sha256}. Removing and falling back to Google Drive."
                )
                output_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"SCP fetch skipped/failed: {e}")

    file_id = _extract_google_drive_id(google_drive_path)
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
        output_path.unlink(missing_ok=True)
        raise ValueError(
            "Downloaded model hash mismatch.\n"
            f"Expected: {expected_sha256}\n"
            f"Actual:   {actual_sha256}"
        )

    return str(output_path)


def create_model(
    num_classes: int = 3,
    n_input_channels: int = 1,
    spatial_dims: int = 3,
    pretrained_path=None,
) -> BasicClassifier:
    model = ResidualEncoderClsLightning(
        in_ch=n_input_channels,
        out_ch=num_classes,
        spatial_dims=spatial_dims,
    )

    if pretrained_path:
        model_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        requested_path = Path(pretrained_path)

        # 1) local file first
        if requested_path.is_absolute():
            resolved_path = requested_path
        else:
            resolved_path = model_dir / requested_path

        if resolved_path.exists():
            print(f"Using local checkpoint: {resolved_path}")
            model.load_pretrained_unet_encoder(str(resolved_path), verbose=True)
            return model

        # 2) otherwise resolve by filename through SCP -> GDrive fallback
        resolved_path = download_verify_pretrained_model(
            cache_dir=model_dir,
            output_filename=requested_path.name,
            try_scp_first=True,
        )
        model.load_pretrained_unet_encoder(resolved_path, verbose=True)

    return model