"""
Batch orchestration script to analyze 6 datasets using existing config files:
- configs/np_assay/high/covid
- configs/np_assay/high/control
- configs/np_assay/medium/covid
- configs/np_assay/medium/control
- configs/np_assay/low/covid
- configs/np_assay/low/control

How it works:
- Runs main.py for each dataset using the corresponding config subfolder.
- Set PROFILE_ANALYSIS to True to enable profiling (serial mode).
- Set STOP_ON_ERROR to False to continue processing after errors.

Run (PowerShell):
    python scripts/orchestrate_datasets.py

Notes:
- Config files must already exist in the subfolders listed above.
- Preprocess datasets separately before running this script if needed.
"""
from __future__ import annotations
import os
import sys
import subprocess
from dataclasses import dataclass
from typing import List, Optional

# ======= User-adjustable flags =======
PROFILE_ANALYSIS: bool = False   # Set True to run main.py with profiling (serial)
STOP_ON_ERROR: bool = True       # Stop the whole batch on first failure

# ======= Repo-relative paths =======
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_SCRIPT = os.path.join(REPO_ROOT, 'main.py')
CONFIGS_ROOT = os.path.join(REPO_ROOT, 'configs')


@dataclass
class Dataset:
    level: str                   # high | medium | low
    label: str                   # covid | control
    config_subdir: str           # relative subfolder under configs/ (e.g., "np_assay/high/covid")


def build_datasets() -> List[Dataset]:
    """
    Define the 6 datasets to analyze.
    Each dataset corresponds to a config subfolder under configs/np_assay/<level>/<label>.
    """
    datasets: List[Dataset] = [
        Dataset(
            level='high',
            label='covid',
            config_subdir=os.path.join('np_assay', 'high', 'covid'),
        ),
        Dataset(
            level='high',
            label='control',
            config_subdir=os.path.join('np_assay', 'high', 'control'),
        ),
        Dataset(
            level='medium',
            label='covid',
            config_subdir=os.path.join('np_assay', 'medium', 'covid'),
        ),
        Dataset(
            level='medium',
            label='control',
            config_subdir=os.path.join('np_assay', 'medium', 'control'),
        ),
        Dataset(
            level='low',
            label='covid',
            config_subdir=os.path.join('np_assay', 'low', 'covid'),
        ),
        Dataset(
            level='low',
            label='control',
            config_subdir=os.path.join('np_assay', 'low', 'control'),
        ),
    ]
    return datasets


def run_cmd(cmd: List[str], env: Optional[dict] = None, cwd: Optional[str] = None) -> int:
    print(f"\n>>> Running: {' '.join(cmd)}\n")
    proc = subprocess.Popen(cmd, env=env, cwd=cwd)
    proc.wait()
    return proc.returncode


def run_analysis(dataset: Dataset) -> int:
    config_dir = os.path.join(CONFIGS_ROOT, dataset.config_subdir)
    if not os.path.isdir(config_dir):
        print(f"[Analyze] ERROR: Config directory not found: {config_dir}")
        return 1

    cmd = [sys.executable, ANALYSIS_SCRIPT, '--config-file-folder', config_dir]
    if PROFILE_ANALYSIS:
        cmd.extend(['--profile', 'True'])

    print(f"[Analyze] Level={dataset.level} Label={dataset.label}")
    print(f"  Config folder: {config_dir}")
    rc = run_cmd(cmd)
    if rc != 0:
        print(f"[Analyze] FAILED for {dataset.level}/{dataset.label} with exit code {rc}")
    return rc


def main():
    datasets = build_datasets()

    # Analyze each dataset using its config subfolder
    for d in datasets:
        rc = run_analysis(d)
        if rc != 0 and STOP_ON_ERROR:
            print("Stopping due to analysis error.")
            sys.exit(rc)

    print("\nAll datasets analyzed.")


if __name__ == '__main__':
    main()
