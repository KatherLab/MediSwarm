#!/usr/bin/env python3
"""
Test suite for ODELIA challenge models.
Tests all 5 challenge models with preflight_check and local_training modes.
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

# Import shared model configurations
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'application', 'jobs', 'ODELIA_ternary_classification', 'app', 'custom', 'models'))
from challenge.challenge_models_config import CHALLENGE_MODELS, get_model_config

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
        env["MODEL_VARIANT"] = model_variant
        env["NUM_EPOCHS"] = "1"  # Minimal epochs for testing
        env["PYTHONUNBUFFERED"] = "1"
        # Add required environment variables for instantiation
        env["SCRATCH_DIR"] = f"./results/{model_variant}_{training_mode}"
        env["DATA_DIR"] = '/home/swarm/Documents/ODELIA/LocalSL/ChallengeData/'
        
        print(f"Currently set environment variables for {model_variant} - {training_mode}:")
        for key in ["TRAINING_MODE", "SITE_NAME", "MODEL_NAME", "MODEL_VARIANT", "NUM_EPOCHS", "SCRATCH_DIR", "DATA_DIR"]:
            print(f"  {key}: {env[key]}")
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
            logger.info(f"Running: {' '.join(cmd)}")
            
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
    # Determine ODELIA app directory
    test_dir = Path(__file__).parent
    odelia_app_dir = Path(
        "/home/swarm/Documents/MediSwarmChallenge/MediSwarm/"
        "application/jobs/ODELIA_ternary_classification/app"
    )

    if not odelia_app_dir.exists():
        logger.error(f"ODELIA app directory not found: {odelia_app_dir}")
        sys.exit(1)
    
    # Run tests
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
