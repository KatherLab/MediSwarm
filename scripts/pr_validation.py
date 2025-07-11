# scripts/pr_validation.py

import os
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
print("Script is running")

import os

print("PWD:", os.getcwd())
print("Files in current dir:", os.listdir())


def get_latest_workspace():
    root = Path.cwd()
    candidates = list(root.rglob("odelia_0.9-dev.*_MEVIS_test"))
    if not candidates:
        raise RuntimeError("No workspace found matching pattern 'odelia_0.9-dev.*_MEVIS_test'")
    return sorted(candidates, reverse=True)[0]


def run_command(cmd, cwd=None):
    print(f"\n>>> Running: {' '.join(cmd)} in {cwd}")
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    site = os.environ.get("SITE_NAME", "UKA")
    datadir = os.environ["DATADIR"]
    scratchdir = os.environ["SCRATCHDIR"]

    workspace_version = get_latest_workspace()
    startup_dir = workspace_version / "prod_00" / site / "startup"

    print(f"Using workspace: {workspace_version}")
    print(f"Startup directory: {startup_dir}")

    # Run dummy training
    run_command(["./docker.sh", "--scratch_dir", scratchdir, "--GPU", "device=0", "--dummy_training"], cwd=startup_dir)

    # Run preflight check
    run_command(
        ["./docker.sh", "--data_dir", datadir, "--scratch_dir", scratchdir, "--GPU", "device=0", "--preflight_check"],
        cwd=startup_dir)


if __name__ == "__main__":
    main()
