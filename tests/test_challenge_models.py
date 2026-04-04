#!/usr/bin/env python3
"""
Test suite for ODELIA challenge models.
Tests all 5 challenge models with preflight_check and local_training modes.

NOTE: This is an *integration*-level test — it actually runs main.py and
requires a CUDA GPU, real training data, and the full dependency stack.
For lightweight CI-friendly tests see tests/unit_tests/.
"""

import os
import sys
import subprocess
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Tuple, List
import logging

# ---------------------------------------------------------------------------
# Dynamic path resolution — works from any checkout location
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_CUSTOM_DIR = REPO_ROOT / "application" / "jobs" / "_shared" / "custom"
MODELS_DIR = SHARED_CUSTOM_DIR / "models"

sys.path.insert(0, str(SHARED_CUSTOM_DIR))
sys.path.insert(0, str(MODELS_DIR))
from models_config import CHALLENGE_MODELS, create_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTester:
    """Test runner for challenge models."""
    
    def __init__(self, odelia_app_dir: str):
        self.odelia_app_dir = Path(odelia_app_dir)
        self.custom_dir = self.odelia_app_dir / "custom"
        #self.config_path = self.odelia_app_dir / "config" / "config_fed_client.conf"
        self.results = {}
        
    def set_environment_variables(self, model_variant: str, training_mode: str, site_name: str = "UKA_1"):
        """Set required environment variables for training."""
        env = os.environ.copy()
        env["TRAINING_MODE"] = training_mode
        env["SITE_NAME"] = site_name
        env["MODEL_NAME"] = f"challenge_{model_variant}"
        env["NUM_EPOCHS"] = "1"  # Minimal epochs for testing
        env["PYTHONUNBUFFERED"] = "1"
        # Use DATA_DIR from env if set, otherwise fall back to a sensible default
        env.setdefault("SCRATCH_DIR", f"./results/{model_variant}_{training_mode}")
        env.setdefault("DATA_DIR", os.environ.get(
            "DATA_DIR",
            str(REPO_ROOT / "data"),  # fallback — override via DATA_DIR env var
        ))

        print(f"Currently set environment variables for {model_variant} - {training_mode}:")
        for key in ["TRAINING_MODE", "SITE_NAME", "MODEL_NAME", "NUM_EPOCHS", "SCRATCH_DIR", "DATA_DIR"]:
            print(f"  {key}: {env.get(key, '<not set>')}")
        print(f"  Current dir: {os.getcwd()}")
        return env
    
    def run_test(self, model_variant: str, training_mode: str, 
                 timeout: int = 600) -> Tuple[bool, str]:
        """Run test for a specific model and training mode."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {model_variant} - {training_mode}")
        logger.info(f"{'='*60}")
        
        try:
            env = self.set_environment_variables(model_variant, training_mode)
            
            # Change to custom directory
            os.chdir(self.custom_dir)
            
            # Run main.py
            cmd = [sys.executable, "main.py"]
            logger.info(f"Running: {cmd} in {str(self.custom_dir)}")
            
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.custom_dir)
            )
            
            success = result.returncode == 0
            output = result.stdout + result.stderr
            
            if success:
                logger.info(f"✓ {model_variant} {training_mode} PASSED")
            else:
                logger.error(f"✗ {model_variant} {training_mode} FAILED")
                logger.error(f"Return code: {result.returncode}")

                logger.error(f"STDOUT: {result.stdout[-2000:]}")
                logger.error(f"STDERR: {result.stderr[-2000:]}")
                            
            return success, output
            
        except subprocess.TimeoutExpired:
            logger.error(f"✗ {model_variant} {training_mode} TIMEOUT (>{timeout}s)")
            return False, "Test timed out"
        except Exception as e:
            logger.error(f"✗ {model_variant} {training_mode} ERROR: {str(e)}")
            return False, str(e)
    
    
    def run_all_tests(self) -> Dict:
        """Run all tests for all models."""
        logger.info("Starting comprehensive challenge model tests...")
        logger.info(f"ODELIA app directory: {self.odelia_app_dir}")
        
        self.results = {
            "summary": {},
            "details": {}
        }
        
        for model_name, model_config in CHALLENGE_MODELS.items():
            model_variant = model_config["team_name"]
            self.results["details"][model_name] = {}
            
            # Test preflight_check
            preflight_success, preflight_output = self.run_test(
                model_variant, "preflight_check", timeout=600
            )
            self.results["details"][model_name]["preflight_check"] = {
                "success": preflight_success,
                "output": preflight_output[-500:] if preflight_output else ""
            }
            
            # Test local_training
            local_success, local_output = self.run_test(
                model_variant, "local_training", timeout=600
            )
            self.results["details"][model_name]["local_training"] = {
                "success": local_success,
                "output": local_output[-500:] if local_output else ""
            }
            
            # Summary
            self.results["summary"][model_name] = {
                "preflight_check": preflight_success,
                "local_training": local_success,
                "overall": preflight_success and local_success
            }
        
        self._print_summary()
        return self.results
    
    def _print_summary(self):
        """Print test summary."""
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        
        for model_name, results in self.results["summary"].items():
            status = "✓ PASS" if results["overall"] else "✗ FAIL"
            logger.info(f"\n{model_name}: {status}")
            logger.info(f"  Instantiation: {'✓' if results.get('instantiation') else '✗'}")
            logger.info(f"  Preflight:     {'✓' if results.get('preflight_check') else '✗'}")
            logger.info(f"  Local Train:   {'✓' if results.get('local_training') else '✗'}")
        
        overall_pass = all(r["overall"] for r in self.results["summary"].values())
        logger.info("\n" + "="*60)
        if overall_pass:
            logger.info("ALL TESTS PASSED ✓")
        else:
            logger.info("SOME TESTS FAILED ✗")
        logger.info("="*60)
        
        return overall_pass


def main():
    """Main test entry point."""
    test_dir = Path(__file__).parent

    # Dynamically resolve the ODELIA app directory from the repo root.
    # After code deduplication (PR 241), all jobs share custom/ via symlinks.
    # We point at the ODELIA job (the canonical one).
    odelia_app_dir = REPO_ROOT / "application" / "jobs" / "ODELIA_ternary_classification" / "app"

    if not odelia_app_dir.exists():
        logger.error(f"ODELIA app directory not found: {odelia_app_dir}")
        logger.error("Make sure you are running from the repository root.")
        sys.exit(1)
    
    # Run tests
    logger.info(f"init Model Tester with app dir: {odelia_app_dir}")
    tester = ModelTester(str(odelia_app_dir))
    results = tester.run_all_tests()
    
    # Save results to file
    results_file = test_dir / "test_results.json"
    with open(results_file, 'w') as f:
        # Convert to serializable format
        serializable_results = {
            "summary": tester.results["summary"],
            "details": {
                k: {kk: str(vv) for kk, vv in v.items()}
                for k, v in tester.results["details"].items()
            }
        }
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    
    # Exit with appropriate code
    all_passed = all(r["overall"] for r in tester.results["summary"].values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
