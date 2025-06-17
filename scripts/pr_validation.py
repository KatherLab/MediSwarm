# scripts/pr_validation.py

import os
import subprocess
from pathlib import Path

def get_latest_workspace():
    workspace_root = Path("workspace")
    versions = sorted([p for p in workspace_root.iterdir() if "odelia" in p.name], reverse=True)
    if not versions:
        raise RuntimeError("No odelia version found in workspace/")
    return versions[0]

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
    run_command(["./docker.sh", "--data_dir", datadir, "--scratch_dir", scratchdir, "--GPU", "device=0", "--preflight_check"], cwd=startup_dir)

if __name__ == "__main__":
    main()
