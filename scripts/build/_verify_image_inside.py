"""Checks that run INSIDE a built ODELIA image (driven by scripts/build/verify_release_image.sh).

    python3 _verify_image_inside.py packages    -> JSON {distribution: version or null}
    python3 _verify_image_inside.py imports     -> imports every module of the job code
    python3 _verify_image_inside.py models      -> builds MST and MST_SAMMed2D on CPU, checks the
                                                   SAM-Med2D encoder equals the shipped checkpoint
    python3 _verify_image_inside.py hf-offline  -> resolves the baked MedicalNet ResNet18 offline

Prints one line per finding prefixed PASS/FAIL (JSON for `packages`); exits 1 on any FAIL.
"""

import importlib
import json
import logging
import os
import sys
import traceback
from importlib import metadata
from pathlib import Path

JOB_CUSTOM = Path("/MediSwarm/application/jobs/ODELIA_ternary_classification/app/custom")
SAM_CHECKPOINT = "/MediSwarm/pretrained_weights/sam-med2d_b.pth"

# Everything one swarm must agree on across images, the frozen training stack, and what 1.9.0 removed.
PACKAGES = ["nvflare_mediswarm", "grpcio", "protobuf", "msgpack", "aiohttp", "websockets", "cryptography",
            "torch", "torchvision", "numpy", "monai", "pytorch_lightning", "torchmetrics", "timm",
            "SimpleITK", "torchio", "pandas", "scikit-learn", "huggingface_hub", "x-transformers",
            "wandb", "datasets", "GitPython", "pyarrow", "pyOpenSSL", "PyJWT", "Werkzeug", "urllib3",
            "setuptools", "pillow"]

failed = False


def report(ok, message):
    global failed
    failed = failed or not ok
    print(("PASS " if ok else "FAIL ") + message, flush=True)


def packages():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    print(json.dumps(versions, sort_keys=True))


def imports():
    sys.path.insert(0, str(JOB_CUSTOM))
    modules = sorted(p.stem for p in JOB_CUSTOM.glob("*.py"))
    modules += ["models", "models.models_config", "data.datamodules", "data.datasets"]
    for name in modules:
        try:
            importlib.import_module(name)
            report(True, f"import {name}")
        except Exception as exc:  # report every broken module, not just the first
            report(False, f"import {name}: {type(exc).__name__}: {exc}")


def models():
    import torch

    sys.path.insert(0, str(JOB_CUSTOM))
    os.environ.setdefault("MEDISWARM_ALLOW_CPU_MODEL", "1")
    os.environ.setdefault("SITE_NAME", "VERIFY_1")
    os.environ.setdefault("SCRATCH_DIR", "/tmp")
    os.environ.setdefault("DATA_DIR", "/tmp")
    logging.basicConfig(level=logging.INFO, format="  log: %(message)s")
    from models.models_config import create_model

    torch.manual_seed(0)
    volume = torch.randn(1, 1, 32, 224, 224)
    built = {}
    for name in ["MST", "MST_SAMMed2D"]:
        try:
            model = create_model(logger=logging.getLogger("verify"), model_name=name, num_classes=3)
            model.eval()
            with torch.no_grad():
                out = model(volume)
            report(tuple(out.shape) == (1, 3), f"{name}: forward on 1x1x32x224x224 gives {tuple(out.shape)}")
            built[name] = model
        except Exception as exc:
            traceback.print_exc()
            report(False, f"{name}: {type(exc).__name__}: {exc}")

    if "MST_SAMMed2D" not in built:
        return
    state = torch.load(SAM_CHECKPOINT, map_location="cpu")
    state = state.get("model", state)
    encoder = {k[len("image_encoder."):]: v for k, v in state.items() if k.startswith("image_encoder.")}
    backbone = {k.split("backbone.", 1)[1]: v for k, v in built["MST_SAMMed2D"].state_dict().items()
                if "backbone." in k}
    shared = [k for k in backbone if k in encoder]
    equal = [k for k in shared if torch.equal(backbone[k].cpu(), encoder[k])]
    report(len(shared) == len(backbone) and len(equal) == len(shared) and len(shared) > 100,
           f"MST_SAMMed2D encoder = checkpoint: {len(equal)}/{len(shared)} tensors equal, "
           f"{len(backbone)} in the model, {len(encoder)} in the checkpoint")


def hf_offline():
    os.environ["HF_HUB_OFFLINE"] = "1"
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download("TencentMedicalNet/MedicalNet-Resnet18", "resnet_18_23dataset.pth")
        report(os.path.isfile(path), f"MedicalNet ResNet18 resolves offline: {path}")
    except Exception as exc:
        report(False, f"MedicalNet ResNet18 offline: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    {"packages": packages, "imports": imports, "models": models, "hf-offline": hf_offline}[sys.argv[1]]()
    sys.exit(1 if failed else 0)
