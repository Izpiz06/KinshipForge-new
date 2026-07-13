"""
Experiment Logger for KinshipForge.
Auto-appends to experiment_history.csv with full reproducibility info.
"""
import csv
import os
import json
import hashlib
import subprocess
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).parent.parent / "experiments"
HISTORY_FILE = EXPERIMENT_DIR / "experiment_history.csv"


class ExperimentLogger:
    """Logs experiment results to CSV with full reproducibility metadata."""
    
    def __init__(self, experiment_name: str, config: Dict[str, Any]):
        self.experiment_name = experiment_name
        self.config = config
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.exp_dir = EXPERIMENT_DIR / f"{self.timestamp}_{experiment_name}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Capture git commit
        self.git_commit = self._get_git_commit()
        self.config_hash = self._hash_config(config)
        
        # Initialize results storage
        self.results = []
    
    def _get_git_commit(self) -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], 
                capture_output=True, text=True, cwd=Path(__file__).parent.parent
            )
            return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"
    
    def _hash_config(self, config: Dict) -> str:
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def log_result(self, 
                   variant: str,
                   pair_id: str,
                   seed: int,
                   age: str,
                   metrics: Dict[str, Any],
                   status: str = "success"):
        """Log a single generation result."""
        row = {
            "timestamp": self.timestamp,
            "experiment": self.experiment_name,
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "variant": variant,
            "pair_id": pair_id,
            "seed": seed,
            "age": age,
            "status": status,
            **metrics
        }
        self.results.append(row)
        
        # Append to CSV immediately
        self._append_csv(row)
    
    def _append_csv(self, row: Dict):
        """Append a row to the master history CSV."""
        file_exists = HISTORY_FILE.exists()
        
        # Flatten nested metrics
        flat_row = {}
        for k, v in row.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    flat_row[f"{k}.{sk}"] = sv
            else:
                flat_row[k] = v
        
        with open(HISTORY_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(flat_row)
    
    def get_experiment_dir(self) -> Path:
        return self.exp_dir
    
    def save_config(self):
        """Save experiment configuration."""
        config_file = self.exp_dir / "experiment_configuration.md"
        with open(config_file, 'w') as f:
            f.write(f"# Experiment Configuration\n\n")
            f.write(f"**Experiment**: {self.experiment_name}\n")
            f.write(f"**Timestamp**: {self.timestamp}\n")
            f.write(f"**Git Commit**: {self.git_commit}\n")
            f.write(f"**Config Hash**: {self.config_hash}\n\n")
            f.write("## Configuration\n\n")
            f.write("```json\n")
            f.write(json.dumps(self.config, indent=2, default=str))
            f.write("\n```\n")


def create_experiment_logger(experiment_name: str, 
                             config: Dict[str, Any]) -> ExperimentLogger:
    """Factory function to create and initialize experiment logger."""
    logger = ExperimentLogger(experiment_name, config)
    logger.save_config()
    return logger


def load_experiment_history() -> list:
    """Load all experiment history from CSV."""
    if not HISTORY_FILE.exists():
        return []
    import pandas as pd
    df = pd.read_csv(HISTORY_FILE)
    return df.to_dict('records')


# ============================================================================
# Reproducibility Utilities
# ============================================================================

def set_deterministic(seed: int = 42):
    """Set all random seeds for deterministic behavior."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_system_info() -> Dict[str, str]:
    """Capture system information for reproducibility."""
    import sys
    import platform
    
    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    return info


def verify_reproducibility(func, *args, seed: int = 42, n_runs: int = 3, **kwargs) -> bool:
    """Verify that a function produces identical results across runs."""
    results = []
    for _ in range(n_runs):
        set_deterministic(seed)
        result = func(*args, **kwargs)
        results.append(result)
    
    # Compare all results
    for i in range(1, n_runs):
        if isinstance(results[0], torch.Tensor):
            if not torch.allclose(results[0], results[i], atol=1e-6):
                return False
        elif isinstance(results[0], np.ndarray):
            if not np.allclose(results[0], results[i], atol=1e-6):
                return False
        elif results[0] != results[i]:
            return False
    return True