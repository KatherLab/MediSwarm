# Making Your Training Code MediSwarm-Compatible

A step-by-step guide for turning standalone PyTorch training code into a
federated/swarm learning application that runs on MediSwarm.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Step-by-Step Integration Guide](#step-by-step-integration-guide)
   - [Step 1: Understand the Model Class Hierarchy](#step-1-understand-the-model-class-hierarchy)
   - [Step 2: Create Your Model Backbone](#step-2-create-your-model-backbone)
   - [Step 3: Write a Model Factory Function](#step-3-write-a-model-factory-function)
   - [Step 4: Register Your Model](#step-4-register-your-model)
   - [Step 5: Create an NVFlare Job Configuration](#step-5-create-an-nvflare-job-configuration)
   - [Step 6: Data Integration](#step-6-data-integration)
   - [Step 7: Local Testing](#step-7-local-testing)
   - [Step 8: Simulation Testing](#step-8-simulation-testing)
   - [Step 9: Swarm Deployment](#step-9-swarm-deployment)
5. [Complete Example](#complete-example)
6. [Reference Tables](#reference-tables)
7. [FAQ / Troubleshooting](#faq--troubleshooting)

---

## Overview

**MediSwarm** is a federated/swarm learning platform for medical imaging
built on top of [NVIDIA FLARE (NVFlare)](https://nvflare.readthedocs.io/).
It enables multiple clinical sites to collaboratively train deep learning
models without sharing raw patient data.

The platform provides:

- **Swarm learning** via NVFlare's `SwarmServerController` / `SwarmClientController`
- **Peer-to-peer aggregation** (no single point of failure)
- **Cross-site evaluation** for comparing model performance across institutions
- **Differential privacy** via percentile-based gradient clipping
- **PyTorch Lightning integration** for clean training loops
- A **centralized model factory** that dynamically loads any registered model

If you have standalone PyTorch training code (e.g., a model that classifies
3D medical images), this guide shows you how to wrap it so that MediSwarm
can orchestrate training across multiple sites.

---

## Architecture

```
+---------------------------------------------------------------------+
|                         MediSwarm Platform                          |
+---------------------------------------------------------------------+
|                                                                     |
|  Your Code                  MediSwarm Glue           NVFlare        |
|  ─────────                  ──────────────           ──────         |
|                                                                     |
|  nn.Module          -->  ModelWrapper           SwarmClientController|
|  (backbone)               (BasicClassifier)     SwarmServerController|
|                              |                       |              |
|                              v                       v              |
|                         PyTorch Lightning        flare.patch()      |
|                         Trainer                  flare.receive()    |
|                              |                   flare.send()       |
|                              v                       |              |
|                         create_model()               v              |
|                         (factory fn)            Aggregation         |
|                              |                  (weighted avg)      |
|                              v                       |              |
|                         models_config.py             v              |
|                         (model registry)        Swarm rounds        |
|                                                                     |
+---------------------------------------------------------------------+
```

### Data Flow

1. Each site runs `main.py`, which reads `TRAINING_MODE` from the environment.
2. `threedcnn_ptl.py` calls `create_model()` to instantiate the model.
3. For **swarm** mode, NVFlare patches the PyTorch Lightning `Trainer` so that
   after each local training round, model weights are exchanged with peers.
4. The `SwarmClientController` handles peer discovery, weight aggregation,
   and model selection (best model by accuracy).

---

## Prerequisites

### Software

| Component           | Version   | Notes                                      |
|---------------------|-----------|--------------------------------------------|
| Python              | 3.10+     | Tested with 3.10                           |
| PyTorch             | 2.2.2+    | CUDA 12.1 support required                 |
| PyTorch Lightning   | 2.4.0+    | Training loop framework                    |
| NVIDIA FLARE        | 2.7.2     | Federated learning framework               |
| MONAI               | 1.4.0+    | Medical imaging transforms (optional)      |
| torchmetrics        | 1.7.1+    | Metrics (AUROC, Accuracy)                  |
| scikit-learn        | 1.5.2+    | Used for data splitting                    |
| CUDA                | 12.1+     | GPU required for training                  |

### Hardware

- **GPU**: NVIDIA GPU with CUDA support (required — MediSwarm raises
  `RuntimeError` if no GPU is detected)
- **RAM**: 16 GB+ recommended for 3D medical imaging
- **Storage**: Sufficient space for datasets and model checkpoints

### Docker

MediSwarm ships as a Docker image based on
`pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`. All dependencies are
pre-installed. You typically only need to add your model code.

---

## Step-by-Step Integration Guide

### Step 1: Understand the Model Class Hierarchy

MediSwarm uses a layered class hierarchy built on PyTorch Lightning:

```
pl.LightningModule
  └── VeryBasicModel          # Abstract base: forward(), _step(), checkpoint I/O
        └── BasicModel        # + configure_optimizers() with lr_scheduler
              └── BasicClassifier  # + CrossEntropyLoss, AUROC/Accuracy metrics
                    └── ModelWrapper   # Wraps any nn.Module backbone
```

**You do NOT need to subclass any of these directly.** Instead, you:

1. Write a standard `nn.Module` (your backbone).
2. Wrap it with `ModelWrapper`, which inherits all the training logic.

`ModelWrapper` provides:

- `forward(x)` — calls `self.backbone(x)`
- `training_step()` / `validation_step()` / `test_step()` — extract
  `batch['source']` and `batch['target']`, compute loss, log metrics
- `configure_optimizers()` — AdamW with configurable LR scheduler
- `_epoch_end()` — computes and logs accuracy and AUC-ROC
- Checkpoint save/load compatible with NVFlare's model persistor

### Step 2: Create Your Model Backbone

Write your model as a standard `torch.nn.Module`. The only requirements:

1. **Input**: A tensor of shape `(batch, channels, [spatial_dims...])`.
   For 3D medical imaging, this is typically `(B, 1, D, H, W)`.
2. **Output**: Logits of shape `(batch, num_classes)`.

```python
# my_model.py
import torch
import torch.nn as nn


class MyBackbone(nn.Module):
    """Example: a simple 3D CNN for ternary classification."""

    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

> **Tip**: Your backbone can use any PyTorch-compatible library (MONAI,
> timm, torchvision, etc.) as long as all dependencies are available in the
> Docker image.

### Step 3: Write a Model Factory Function

MediSwarm's model registry calls a **factory function** to instantiate your
model. This function must:

1. Accept keyword arguments (`num_classes`, `n_input_channels`,
   `spatial_dims`, plus any model-specific args).
2. Create your backbone.
3. Wrap it with `ModelWrapper`.
4. Return the wrapped model.

```python
# my_model.py (continued)
import sys
from pathlib import Path

# Ensure the shared custom directory is on the path
_CUSTOM_DIR = Path(__file__).resolve().parents[2]  # adjust depth as needed
if str(_CUSTOM_DIR) not in sys.path:
    sys.path.insert(0, str(_CUSTOM_DIR))

from models.base_model import ModelWrapper


def create_model(
    num_classes: int = 3,
    n_input_channels: int = 1,
    spatial_dims: int = 3,
    **kwargs,
):
    """Factory function called by MediSwarm's model registry."""
    backbone = MyBackbone(
        in_channels=n_input_channels,
        num_classes=num_classes,
    )
    return ModelWrapper(
        backbone=backbone,
        in_ch=n_input_channels,
        num_classes=num_classes,
        spatial_dims=spatial_dims,
        **kwargs,
    )
```

**Key points:**

- The function name is typically `create_model`, but can be anything — it is
  referenced by its dotted path in the configuration.
- `ModelWrapper` takes `backbone`, `in_ch`, `num_classes`, and optionally
  `spatial_dims`, `loss`, `loss_kwargs`, `optimizer`, `lr`.
- The factory function must be importable from the path you register (see
  next step).

### Step 4: Register Your Model

You have two options for registering your model:

#### Option A: Challenge Model (recommended for new models)

Place your model code under the challenge models directory:

```
application/jobs/_shared/custom/models/challenge/
  my_team_name/
    __init__.py
    my_model.py     # Contains MyBackbone + create_model()
```

Then register it in `models_config.py`:

```python
# In application/jobs/_shared/custom/models/models_config.py

CHALLENGE_MODELS = {
    # ... existing models ...
    "my_team_name": {
        "team_name": "my_team_name",
        "persistor_path": "challenge.my_team_name.my_model.create_model",
        "persistor_args": {
            "num_classes": 3,
            "n_input_channels": 1,
            "spatial_dims": 3,
            # Add any model-specific kwargs here
        },
    },
}
```

The `persistor_path` is a dot-separated path **relative to the `models/`
directory**. The last segment is the function name; everything before it
is the module path.

For example, `challenge.my_team_name.my_model.create_model` resolves to:
```
models/challenge/my_team_name/my_model.py → create_model()
```

#### Option B: Top-Level Model

For models that should be first-class options alongside MST, ResNet, and
Swin3D, add them directly to the `create_model()` function in
`models_config.py`:

```python
elif model_name == "MyModel":
    from models.my_model import MyModel
    model = MyModel(num_classes=num_classes, in_ch=in_ch, ...)
```

This approach requires modifying the factory function itself and is best
reserved for models that will be widely used.

### Step 5: Create an NVFlare Job Configuration

Each MediSwarm job is a directory under `application/jobs/` with this
structure:

```
application/jobs/my_new_job/
  meta.conf                           # Job metadata
  app/
    config/
      config_fed_client.conf          # Client-side configuration
      config_fed_server.conf          # Server-side configuration
    custom -> ../../_shared/custom    # Symlink to shared code
```

#### meta.conf

```hocon
{
  name = "my_new_job"
  resource_spec {}
  deploy_map {
    app = ["@ALL"]
  }
  min_clients = 2
  mandatory_clients = []
}
```

- `min_clients`: Minimum number of sites required to start training.
- `deploy_map`: `"@ALL"` deploys the app to every connected client.

#### config_fed_client.conf

This is the most complex configuration file. Here is a minimal working
example:

```hocon
{
  format_version = 2

  executors = [
    {
      tasks = ["train", "validate", "submit_model"]
      executor {
        path = "nvflare.app_opt.pt.client_api_launcher_executor.PTClientAPILauncherExecutor"
        args {
          launcher_id = "launcher"
          pipe_id = "pipe"
          heartbeat_timeout = 600
          params_exchange_format = "pytorch"
          params_transfer_type = "DIFF"
          train_with_evaluation = true
          launch_once = true
        }
      }
    }
    {
      tasks = ["swarm_*"]
      executor {
        path = "nvflare.app_common.ccwf.SwarmClientController"
        args {
          learn_task_name = "train"
          persistor_id = "persistor"
          aggregator_id = "aggregator"
          shareable_generator_id = "shareable_generator"
          min_responses_required = 2
          wait_time_after_min_resps_received = 300
          learn_task_timeout = 7200
          learn_task_ack_timeout = 120
          final_result_ack_timeout = 300
          metric_comparator_id = "metric_comparator"
        }
      }
    }
  ]

  components = [
    {
      id = "launcher"
      path = "nvflare.app_common.launchers.subprocess_launcher.SubprocessLauncher"
      args {
        launch_command = "python3 custom/main.py"
      }
    }
    {
      id = "pipe"
      path = "nvflare.fuel.utils.pipe.cell_pipe.CellPipe"
      args {
        mode = "PASSIVE"
        site_name = "{SITE_NAME}"
        token = "{JOB_ID}"
        root_url = "{ROOT_URL}"
        secure_mode = "{SECURE_MODE}"
        workspace_dir = "{WORKSPACE}"
      }
    }
    {
      id = "persistor"
      path = "nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor"
      args {
        model {
          path = "models.models_config.create_model"
          args {}
        }
      }
    }
    {
      id = "shareable_generator"
      path = "nvflare.app_common.shareablegenerators.full_model_shareable_generator.FullModelShareableGenerator"
      args {}
    }
    {
      id = "aggregator"
      path = "nvflare.app_common.aggregators.intime_accumulate_model_aggregator.InTimeAccumulateWeightedAggregator"
      args {
        expected_data_kind = "WEIGHT_DIFF"
      }
    }
    {
      id = "metric_comparator"
      path = "nvflare.app_common.utils.numeric_comparator.NumberMetricComparator"
      args {}
    }
    {
      id = "model_selector"
      path = "nvflare.app_common.widgets.intime_model_selector.IntimeModelSelector"
      args {
        key_metric = "accuracy"
      }
    }
    {
      id = "metric_relay"
      path = "nvflare.app_common.ccwf.comps.metric_relay.MetricRelay"
      args {
        pipe_id = "pipe"
        event_type = "fed.analytix_log_stats"
        read_interval = 0.1
      }
    }
  ]

  task_result_filters = [
    {
      tasks = ["train"]
      filters = [
        {
          path = "nvflare.app_common.filters.percentile_privacy.PercentilePrivacy"
          args {
            percentile = 10
            gamma = 0.01
          }
        }
      ]
    }
  ]
}
```

**Key points:**

- The **persistor** `model.path` must point to your `create_model()` factory.
  For a challenge model, use `models.models_config.create_model` (the
  centralized factory reads `MODEL_NAME` from the environment).
- For a challenge model, set the environment variable
  `MODEL_NAME=challenge_<team_name>` at runtime.
- `task_result_filters` with `PercentilePrivacy` adds differential privacy.

#### config_fed_server.conf

```hocon
{
  format_version = 2

  workflows = [
    {
      id = "swarm_controller"
      path = "nvflare.app_common.ccwf.SwarmServerController"
      args {
        num_rounds = 20
        start_task_timeout = 600
        progress_timeout = 7200
        configure_task_timeout = 300
        max_status_report_interval = 300
      }
    }
  ]

  components = [
    {
      id = "json_generator"
      path = "nvflare.app_common.widgets.validation_json_generator.ValidationJsonGenerator"
      args {}
    }
  ]
}
```

#### Create the symlink

```bash
cd application/jobs/my_new_job/app/
ln -s ../../_shared/custom custom
```

This symlink points to the shared custom code directory, so all jobs use
the same `main.py`, `env_config.py`, model code, etc.

### Step 6: Data Integration

MediSwarm expects datasets to return dictionaries with these keys:

```python
{
    "source": torch.Tensor,  # Input data (e.g., shape [1, D, H, W] for 3D)
    "target": torch.Tensor,  # Label (integer class index)
    "uid": str,              # Unique identifier for the sample
}
```

#### Using the Built-in ODELIA Dataset

If your data is in the ODELIA format (NIfTI 3D breast MRI), you can use
`ODELIA_Dataset3D` directly. Set the `DATA_DIR` environment variable to
point to your data directory.

#### Using a Custom Dataset

To use your own dataset:

1. **Create a dataset class** that returns the dictionary format above.
2. **Modify `env_config.py`** (or create a wrapper) to load your dataset
   instead of `ODELIA_Dataset3D`.

```python
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.samples = self._load_samples(data_dir, split)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, label, uid = self.samples[idx]

        if self.transform:
            image = self.transform(image)

        return {
            "source": image,       # Tensor: [C, D, H, W]
            "target": torch.tensor(label, dtype=torch.long),
            "uid": uid,
        }
```

The `DataModule` class (in `data/datamodules/datamodule.py`) wraps your
dataset into PyTorch `DataLoader`s with configurable batch size, sampling
strategy (random or weighted), and worker processes.

### Step 7: Local Testing

Before deploying to a swarm, test your model locally:

#### Preflight Check

```bash
export TRAINING_MODE="preflight_check"
export SITE_NAME="LOCAL_TEST"
export MODEL_NAME="challenge_my_team_name"   # or "MST", "ResNet101", etc.
export DATA_DIR="/path/to/your/data"
export SCRATCH_DIR="/path/to/scratch"
export MAX_EPOCHS="1"

cd application/jobs/_shared/custom/
python3 main.py
```

This runs a single epoch to verify that:
- The model can be instantiated
- Data loading works
- Forward/backward passes succeed
- Metrics are computed correctly

#### Local Training

```bash
export TRAINING_MODE="local_training"
export MAX_EPOCHS="5"
# ... same env vars as above ...

cd application/jobs/_shared/custom/
python3 main.py
```

This runs full local training (without swarm aggregation) to verify
convergence behavior.

### Step 8: Simulation Testing

NVFlare provides a simulator for testing federated workflows on a single
machine:

```bash
nvflare simulator \
  -w /tmp/nvflare_sim \
  -n 2 \
  -t 2 \
  application/jobs/my_new_job/
```

- `-n 2`: Simulate 2 clients
- `-t 2`: Use 2 threads

This tests the full NVFlare workflow (model exchange, aggregation, model
selection) without requiring multiple physical machines.

### Step 9: Swarm Deployment

For full swarm deployment:

1. **Build the Docker image** using the provided build scripts:
   ```bash
   ./scripts/build/buildDockerImageAndStartupKits.sh
   ```

2. **Distribute startup kits** to each participating site.

3. **Set environment variables** on each site:
   ```bash
   export TRAINING_MODE="swarm"
   export SITE_NAME="SITE_A"        # Unique per site
   export MODEL_NAME="challenge_my_team_name"
   export DATA_DIR="/local/data"
   export SCRATCH_DIR="/local/scratch"
   ```

4. **Launch** the Docker container on each site. The NVFlare framework
   handles peer discovery, training coordination, and model aggregation.

---

## Complete Example

This walkthrough integrates a ResNet18-based classifier (similar to the
`5Pimed` challenge model) into MediSwarm.

### 1. Create the model file

```
application/jobs/_shared/custom/models/challenge/my_resnet/
  __init__.py
  model.py
```

**`model.py`**:

```python
"""ResNet18 backbone for 3D medical image classification."""

import sys
from pathlib import Path

import torch.nn as nn

# Add shared custom dir to path for base_model import
_CUSTOM_DIR = Path(__file__).resolve().parents[3]
if str(_CUSTOM_DIR) not in sys.path:
    sys.path.insert(0, str(_CUSTOM_DIR))

from models.base_model import ModelWrapper


class ResNet3D(nn.Module):
    """Simple 3D ResNet using MONAI's implementation."""

    def __init__(self, model_name="resnet18", num_classes=3,
                 n_input_channels=1, spatial_dims=3, norm="batch"):
        super().__init__()
        from monai.networks.nets import resnet18, resnet50

        model_map = {
            "resnet18": resnet18,
            "resnet50": resnet50,
        }

        builder = model_map[model_name]
        self.model = builder(
            pretrained=False,
            n_input_channels=n_input_channels,
            num_classes=num_classes,
            spatial_dims=spatial_dims,
            norm=norm,
        )

    def forward(self, x):
        return self.model(x)


def create_model(
    model_name="resnet18",
    num_classes=3,
    n_input_channels=1,
    spatial_dims=3,
    norm="batch",
    **kwargs,
):
    """Factory function for MediSwarm model registry."""
    backbone = ResNet3D(
        model_name=model_name,
        num_classes=num_classes,
        n_input_channels=n_input_channels,
        spatial_dims=spatial_dims,
        norm=norm,
    )
    return ModelWrapper(
        backbone=backbone,
        in_ch=n_input_channels,
        num_classes=num_classes,
        spatial_dims=spatial_dims,
        **kwargs,
    )
```

**`__init__.py`**:

```python
from .model import create_model
```

### 2. Register in models_config.py

Add to the `CHALLENGE_MODELS` dictionary:

```python
"my_resnet": {
    "team_name": "my_resnet",
    "persistor_path": "challenge.my_resnet.model.create_model",
    "persistor_args": {
        "model_name": "resnet18",
        "num_classes": 3,
        "n_input_channels": 1,
        "spatial_dims": 3,
        "norm": "batch",
    },
},
```

### 3. Create the NVFlare job

```bash
# Create job directory
mkdir -p application/jobs/my_resnet_job/app/config

# Symlink shared custom code
cd application/jobs/my_resnet_job/app/
ln -s ../../_shared/custom custom
cd ..

# Copy config templates from ODELIA job
cp ../ODELIA_ternary_classification/meta.conf .
cp ../ODELIA_ternary_classification/app/config/config_fed_client.conf app/config/
cp ../ODELIA_ternary_classification/app/config/config_fed_server.conf app/config/

# Edit meta.conf to set the job name
sed -i 's/ODELIA_ternary_classification/my_resnet_job/' meta.conf
```

### 4. Test locally

```bash
export TRAINING_MODE="preflight_check"
export SITE_NAME="LOCAL"
export MODEL_NAME="challenge_my_resnet"
export DATA_DIR="/path/to/data"
export SCRATCH_DIR="/tmp/mediswarm_test"
export MAX_EPOCHS="1"

cd application/jobs/_shared/custom/
python3 main.py
```

### 5. Run in swarm

```bash
export TRAINING_MODE="swarm"
export MODEL_NAME="challenge_my_resnet"
# ... launch via Docker / NVFlare dashboard
```

---

## Reference Tables

### Environment Variables

| Variable              | Required | Default       | Description                                          |
|-----------------------|----------|---------------|------------------------------------------------------|
| `TRAINING_MODE`       | Yes      | —             | `preflight_check`, `local_training`, or `swarm`      |
| `SITE_NAME`           | Yes      | —             | Unique identifier for this site (e.g., `"TUD_1"`)    |
| `DATA_DIR`            | Yes      | —             | Path to training data directory                      |
| `SCRATCH_DIR`         | Yes      | —             | Path for checkpoints, logs, and output                |
| `MODEL_NAME`          | No       | `"ResNet101"` | Model to train (e.g., `"MST"`, `"challenge_5Pimed"`) |
| `MAX_EPOCHS`          | No       | `100`         | Maximum training epochs per round                    |
| `MIN_PEERS`           | No       | `2`           | Minimum peers for swarm aggregation                  |
| `MAX_PEERS`           | No       | `10`          | Maximum peers for swarm aggregation                  |
| `LOCAL_COMPARE_FLAG`  | No       | `"False"`     | Enable local model comparison                        |
| `USE_ADAPTIVE_SYNC`   | No       | `"False"`     | Enable adaptive sync frequency                       |
| `SYNC_FREQUENCY`      | No       | `1024`        | Steps between sync events                            |
| `PREDICT_FLAG`        | No       | `"ext"`       | Prediction mode (`"ext"` for external test set)       |
| `MEDISWARM_VERSION`   | No       | —             | Version string for logging                           |

### Model Names

| `MODEL_NAME` Value            | Description                              |
|-------------------------------|------------------------------------------|
| `MST`                         | Medical Swin Transformer (default)       |
| `ResNet101`                   | ResNet-101 via MONAI                     |
| `ResNet50`                    | ResNet-50 via MONAI                      |
| `Swin3D`                      | Swin Transformer 3D                      |
| `challenge_1DivideAndConquer` | Residual encoder network                 |
| `challenge_2BCN_AIM`          | Swin UNETR architecture                  |
| `challenge_3agaldran`         | MobileViT v2 / video backbone model      |
| `challenge_4LME_ABMIL`       | Attention-based MIL                      |
| `challenge_5Pimed`            | ResNet18-based classifier                |

### Directory Structure

```
application/jobs/_shared/custom/
  main.py                          # Entry point (reads TRAINING_MODE)
  env_config.py                    # Environment variable loading, dataset setup
  threedcnn_ptl.py                 # Training orchestration (Trainer config)
  data/
    datasets/                      # Dataset classes (ODELIA_Dataset3D, etc.)
    datamodules/
      datamodule.py                # PyTorch Lightning DataModule
  models/
    base_model.py                  # VeryBasicModel -> BasicModel -> BasicClassifier -> ModelWrapper
    models_config.py               # Model registry + create_model() factory
    resnet.py                      # Built-in ResNet variants
    mst.py                         # Medical Swin Transformer
    swin3D.py                      # Swin Transformer 3D
    challenge/                     # Challenge model submissions
      1DivideAndConquer/
      2BCN_AIM/
      3agaldran/
      4LME_ABMIL/
      5pimed/
  utils/                           # Utility functions
```

### NVFlare Config Fields

#### config_fed_client.conf — Key Fields

| Field                                     | Description                                       |
|-------------------------------------------|---------------------------------------------------|
| `executors[0].tasks`                      | Tasks this executor handles (`train`, `validate`)  |
| `executors[0].executor.args.pipe_id`      | Communication pipe for client-server messaging     |
| `components[persistor].model.path`        | Dotted path to your `create_model()` function      |
| `components[persistor].model.args`        | Kwargs passed to `create_model()`                 |
| `components[model_selector].args.key_metric` | Metric for best-model selection (`"accuracy"`)  |
| `task_result_filters`                     | Privacy/quality filters on model updates          |

#### config_fed_server.conf — Key Fields

| Field                                    | Description                                        |
|------------------------------------------|----------------------------------------------------|
| `workflows[0].args.num_rounds`           | Number of swarm learning rounds                    |
| `workflows[0].args.start_task_timeout`   | Timeout for clients to configure (seconds)         |
| `workflows[0].args.progress_timeout`     | Max time without progress before abort (seconds)   |

---

## FAQ / Troubleshooting

### Q: My model works standalone but fails in MediSwarm

**Check these common issues:**

1. **Missing `create_model()` factory**: MediSwarm requires a factory
   function, not direct class instantiation.
2. **Wrong `persistor_path`**: The path is relative to the `models/`
   directory. Verify with:
   ```python
   # The path "challenge.my_team.model.create_model" resolves to:
   # models/challenge/my_team/model.py → create_model()
   ```
3. **Import errors**: Ensure all your model's dependencies are available
   in the Docker image. Check `docker_config/Dockerfile_ODELIA` for the
   installed packages.

### Q: `RuntimeError: This example requires a GPU`

MediSwarm's `create_model()` checks `torch.cuda.is_available()` before
instantiating any model. You need a CUDA-capable GPU. For local testing
without a GPU, use the unit test mocking approach (see
`tests/unit_tests/conftest.py`).

### Q: How do I add custom dependencies?

Edit `docker_config/Dockerfile_ODELIA` to add `pip install` commands for
your dependencies. Then rebuild the Docker image:

```bash
./scripts/build/buildDockerImageAndStartupKits.sh
```

### Q: How do I handle class imbalance?

`ModelWrapper` accepts `loss_kwargs` which are passed to
`torch.nn.CrossEntropyLoss`. Use the `weight` parameter:

```python
import torch

# In your create_model() factory:
class_weights = torch.tensor([1.0, 2.0, 3.0])  # Adjust per your data
return ModelWrapper(
    backbone=backbone,
    in_ch=1,
    num_classes=3,
    loss_kwargs={"weight": class_weights},
)
```

### Q: How do I use a different optimizer or learning rate?

`ModelWrapper` (via `BasicModel`) supports configurable optimizers:

```python
return ModelWrapper(
    backbone=backbone,
    in_ch=1,
    num_classes=3,
    optimizer=torch.optim.SGD,
    lr=0.001,
)
```

### Q: My model has module names starting with digits (e.g., `3agaldran`)

Python module names cannot start with digits. MediSwarm handles this via
`importlib.util.spec_from_file_location()`, which loads modules by file
path rather than by Python import. Just ensure your `persistor_path` is
correct (e.g., `"challenge.3agaldran.model_factory.model_factory"`), and
the dynamic loader will handle it.

### Q: How do I monitor training?

MediSwarm logs to TensorBoard. Logs are saved under the `SCRATCH_DIR`:

```bash
tensorboard --logdir /path/to/scratch/runs/
```

### Q: Swarm training hangs or times out

Check these settings in `config_fed_server.conf`:
- `start_task_timeout`: Increase if clients take long to start
- `progress_timeout`: Increase for slow training rounds
- `num_rounds`: Reduce for faster testing

Also verify:
- All sites can reach each other via VPN
- `min_clients` in `meta.conf` matches the number of connected sites
- GPU is available on all sites (`nvidia-smi`)

### Q: How do I test without real patient data?

Use synthetic data or publicly available datasets. The key requirement is
that your dataset returns the expected dictionary format:

```python
{"source": tensor, "target": tensor, "uid": str}
```

For unit testing, see `tests/unit_tests/conftest.py` for the `MockDataset`
class that generates random tensors.

---

## Further Reading

- [NVFlare Documentation](https://nvflare.readthedocs.io/)
- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [MONAI Documentation](https://docs.monai.io/)
- [MediSwarm Repository](https://github.com/KatherLab/MediSwarm)
