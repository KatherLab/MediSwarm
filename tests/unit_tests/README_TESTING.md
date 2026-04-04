# Challenge Models Testing Guide

This guide explains how to use the automated testing scripts for all 5 challenge models without manual intervention.

## Overview

The testing suite provides:
- **Automated model loading validation** - Ensures all models can be instantiated correctly
- **Local training tests** - Runs preflight_check and local_training modes for each model
- **Complete CI/CD workflow** - Tests, commits, pushes, and builds in one command

## Quick Start

### Run Complete Test & Build Workflow

```bash
cd <repo-root>  # e.g. /home/user/Projects/MediSwarm

# Test all models, update configs, and build startup kits
./scripts/build/test_and_build_all_models.sh

# Test specific models only
./scripts/build/test_and_build_all_models.sh --models "2BCN_AIM,3agaldran"

# Test without pushing changes to git
./scripts/build/test_and_build_all_models.sh --no-push

# Test without building startup kits
./scripts/build/test_and_build_all_models.sh --skip-build

# Test specific models, skip push and build
./scripts/build/test_and_build_all_models.sh --models "1DivideAndConquer" --no-push --skip-build
```

### Challenge Models Tested

1. **1DivideAndConquer** - Residual encoder network
2. **2BCN_AIM** - Swin UNETR architecture
3. **3agaldran** - MobileViT v2 Small model
4. **4LME_ABMIL** - Attention-based MIL
5. **5Pimed** - ResNet18-based classifier

## Configuration Management

### Shared Configuration File

All model configurations are centralized in `application/jobs/ODELIA_ternary_classification/app/custom/models/challenge/challenge_models_config.py` to eliminate duplication between scripts. This file contains:

- Model persistor paths
- Model-specific arguments
- Helper functions for accessing configurations

### Update Config for Specific Model

```bash
cd <repo-root>  # e.g. /home/user/Projects/MediSwarm

# List available models and their configs
python3 scripts/update_config_fed_client.py --list

# Specify custom config path
python3 scripts/update_config_fed_client.py 3agaldran \
  /custom/path/config_fed_client.conf
```

The script automatically creates backups of your config:
- `config_fed_client.conf.2BCN_AIM.backup`
- `config_fed_client.conf.3agaldran.backup`
- etc.

### Model Persistor Configurations

Each model has a specific persistor path and arguments configured in the config file:
# application.jobs.ODELIA_ternary_classification.app.custom.

#### 1DivideAndConquer
```
Path: models.challenge.1DivideAndConquer.model.create_model
Args: n_input_channels=3, num_classes=3
```

#### 2BCN_AIM
```
Path: models.challenge.2bcnaim.swinunetr.create_model
Args: img_size=224, num_classes=3, n_input_channels=1, spatial_dims=3
```

#### 3agaldran
```
Path: models.challenge.3agaldran.model_factory.model_factory
Args: arch="mvit_v2_s", num_classes=3, in_ch=1, seed=123
```

#### 4LME_ABMIL
```
Path: models.challenge.4abmil.model.create_model
Args: n_input_channels=3, num_classes=3
```

#### 5Pimed
```
Path: models.challenge.5pimed.model.create_model
Args: model_name="resnet18", num_classes=3, n_input_channels=1, spatial_dims=3, norm="batch"
```

## Unit Testing

### Run Individual Model Tests

```bash
# Run Python unit tests
cd <repo-root>  # e.g. /home/user/Projects/MediSwarm
python3 tests/unit_tests/test_challenge_models.py

# Or with pytest if available
pytest tests/unit_tests/test_challenge_models.py -v
```

### Test Results

Test results are saved to:
`<repo-root>/application/jobs/ODELIA_ternary_classification/test_results.json`

Example output:
```json
{
  "summary": {
    "1DivideAndConquer": {
      "instantiation": true,
      "preflight_check": true,
      "local_training": true,
      "overall": true
    },
    ...
  }
}
```

## Local Training Tests

### Manual Testing

Each test runs with:
- **NUM_EPOCHS**: 1 (minimal for quick testing)
- **Timeout**: 600 seconds per test
- **Output**: Saved to `/tmp/test_<model>_<mode>.log`

To manually test a model:

```bash
export TRAINING_MODE="preflight_check"  # or "local_training"
export SITE_NAME="TEST_SITE"
export MODEL_NAME="challenge_3agaldran"
export MODEL_VARIANT="3agaldran"
export NUM_EPOCHS="1"

cd application/jobs/ODELIA_ternary_classification/app/custom
python3 application/jobs/ODELIA_ternary_classification/app/custom/main.py
```

### View Test Logs

```bash
# View preflight check logs for a model
tail -100 /tmp/test_2BCN_AIM_preflight_check.log

# View local training logs
tail -100 /tmp/test_3agaldran_local_training.log
```

## Workflow without Manual Intervention

The complete workflow now goes like this:

### Before (Manual)
1. ❌ Push changes manually
2. ❌ Create startup kits manually
3. ❌ Test preflight_check for each model manually
4. ❌ Test local_training for each model manually
5. ❌ Verify models load correctly manually

### After (Automated)
```bash
./scripts/build/test_and_build_all_models.sh
```
1. ✓ Tests all 5 models automatically
2. ✓ Updates config files automatically  
3. ✓ Commits changes automatically
4. ✓ Pushes to git automatically
5. ✓ Builds startup kits automatically

## Troubleshooting

### Test Timeouts

If tests timeout (>600 seconds), the model may be loading a large pretrained weight file. You can:

1. Increase timeout in `scripts/build/test_and_build_all_models.sh` (change `600` to higher value)
2. Pre-download weights in the container
3. Check `/tmp/test_<model>_<mode>.log` for detailed error messages

### Config Update Fails

If `update_config_fed_client.py` fails to update the config:

1. Restore from backup:
   ```bash
   cp config/config_fed_client.conf.2BCN_AIM.backup \
      config/config_fed_client.conf
   ```

2. Try manual update - the config persistor section looks like:
   ```hocon
   {
     id = "persistor"
     path = "nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor"
     args {
       model {
         path = "models.challenge.2bcnaim.swinunetr.create_model"
         args {
           img_size=224
           num_classes=3
           n_input_channels=1
           spatial_dims=3
         }
       }
     }
   }
   ```

### Model Instantiation Errors

Check the error message in the test log:
```bash
grep "MODEL_INSTANTIATION_FAILED" /tmp/test_*.log
```

Common issues:
- Missing imports or dependencies
- Incorrect model path
- Missing pretrained weight files (for 3agaldran)
- GPU not available (requires CUDA)

## CI/CD Integration

To integrate into CI/CD (GitHub Actions, GitLab CI, etc.):

```bash
#!/bin/bash
set -e

cd <repo-root>  # e.g. /home/user/Projects/MediSwarm

# Run tests
python3 tests/unit_tests/test_challenge_models.py

# Update all configs and build
./scripts/build/test_and_build_all_models.sh

# Check results
if [ -f "application/jobs/ODELIA_ternary_classification/test_results.json" ]; then
    echo "Tests completed"
else
    echo "Tests failed"
    exit 1
fi
```

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `TRAINING_MODE` | preflight_check, local_training, or swarm | `preflight_check` |
| `SITE_NAME` | Site identifier | `TEST_SITE`, `TUD_1` |
| `MODEL_NAME` | Model name with challenge_ prefix | `challenge_2BCN_AIM` |
| `MODEL_VARIANT` | Just the team name | `2BCN_AIM` |
| `NUM_EPOCHS` | Number of training epochs | `1` |

## Files Created

- `/scripts/build/test_and_build_all_models.sh` - Main automation script
- `/scripts/update_config_fed_client.py` - Config updater utility
- `/tests/unit_tests/test_challenge_models.py` - Unit test suite
- `/tests/unit_tests/README_TESTING.md` - This file

## Support

For issues or questions:
1. Check test logs in `/tmp/test_*.log`
2. Review test results in `test_results.json`
3. Re-run with `--no-push --skip-build` for debugging
4. Check environment variables are set correctly
