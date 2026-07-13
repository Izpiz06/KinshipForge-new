"""
Experiment Logger for KinshipForge
Auto-logs every experiment to CSV history with full reproducibility metadata.
"""
import os
import csv
import json
import hashlib
import subprocess
import datetime
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ExperimentRecord:
    """Single experiment record for CSV logging."""
    timestamp: str
    experiment_name: str
    git_commit: str
    config_hash: str
    random_seed: int
    parents: str  # JSON of father/mother paths
    child_age: str
    child_gender: str
    gamma: float
    eta: float
    arcs_lambda: float
    father_weight: float
    mother_weight: float
    mix_mode: str
    crossover_mode: str
    mutation_mode: str
    status: str  # running, completed, failed
    metrics: str  # JSON string of all metrics
    decision: str  # keep, revert, iterate
    notes: str


class ExperimentLogger:
    """Manages experiment logging with automatic CSV history."""
    
    HISTORY_PATH = Path("experiments/experiment_history.csv")
    EXPERIMENTS_DIR = Path("experiments")
    
    # CSV fieldnames matching ExperimentRecord
    FIELDNAMES = [
        "timestamp", "experiment_name", "git_commit", "config_hash",
        "random_seed", "parents", "child_age", "child_gender",
        "gamma", "eta", "arcs_lambda", "father_weight", "mother_weight",
        "mix_mode", "crossover_mode", "mutation_mode", "status",
        "metrics", "decision", "notes"
    ]
    
    def __init__(self):
        self.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self._init_history()
    
    def _init_history(self):
        """Initialize history CSV with headers if not exists."""
        if not self.HISTORY_PATH.exists():
            with open(self.HISTORY_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.FIELDNAMES)
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, cwd=Path.cwd()
            )
            return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"
    
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Create deterministic hash of configuration."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]
    
    def create_experiment_dir(self, experiment_name: str) -> Path:
        """Create timestamped experiment directory."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_dir = self.EXPERIMENTS_DIR / f"{timestamp}_{experiment_name}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "figures").mkdir(exist_ok=True)
        return exp_dir
    
    def log_start(self, experiment_name: str, config: Dict[str, Any], 
                  parents: Dict[str, str]) -> tuple[Path, ExperimentRecord]:
        """Log experiment start, return experiment directory and record."""
        exp_dir = self.create_experiment_dir(experiment_name)
        
        record = ExperimentRecord(
            timestamp=datetime.datetime.now().isoformat(),
            experiment_name=experiment_name,
            git_commit=self._get_git_commit(),
            config_hash=self._hash_config(config),
            random_seed=config.get("seed", 42),
            parents=json.dumps(parents),
            child_age=config.get("age", "5-10"),
            child_gender=config.get("gender", "male"),
            gamma=config.get("gamma", 0.47),
            eta=config.get("eta", 0.4),
            arcs_lambda=config.get("arcs_lambda", 0.0),
            father_weight=config.get("father_weight", 0.5),
            mother_weight=config.get("mother_weight", 0.5),
            mix_mode=config.get("mix_mode", "fixed_50_50"),
            crossover_mode=config.get("crossover_mode", "rfg_linear"),
            mutation_mode=config.get("mutation_mode", "brdas"),
            status="running",
            metrics="{}",
            decision="",
            notes=""
        )
        
        # Save initial record
        self._append_record(record)
        
        # Save config to experiment dir
        with open(exp_dir / "experiment_config.json", 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        return exp_dir, record
    
    def log_complete(self, record: ExperimentRecord, metrics: Dict[str, Any],
                     decision: str, notes: str = "") -> None:
        """Log experiment completion with metrics and decision."""
        record.metrics = json.dumps(metrics, default=str)
        record.decision = decision
        record.notes = notes
        record.status = "completed"
        self._append_record(record)
    
    def log_failed(self, record: ExperimentRecord, error: str) -> None:
        """Log experiment failure."""
        record.status = "failed"
        record.notes = error
        self._append_record(record)
    
    def _append_record(self, record: ExperimentRecord) -> None:
        """Append record to history CSV."""
        with open(self.HISTORY_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                record.timestamp, record.experiment_name, record.git_commit,
                record.config_hash, record.random_seed, record.parents,
                record.child_age, record.child_gender, record.gamma,
                record.eta, record.arcs_lambda, record.father_weight,
                record.mother_weight, record.mix_mode, record.crossover_mode,
                record.mutation_mode, record.status, record.metrics,
                record.decision, record.notes
            ])


# Convenience function for scripts
def get_logger() -> ExperimentLogger:
    """Get singleton logger instance."""
    return ExperimentLogger()


if __name__ == "__main__":
    # Test the logger
    logger = get_logger()
    config = {
        "seed": 42,
        "age": "5-10",
        "gender": "male",
        "gamma": 0.47,
        "eta": 0.4,
        "arcs_lambda": 0.0,
        "father_weight": 0.7,
        "mother_weight": 0.3,
        "mix_mode": "gender_biased",
        "crossover_mode": "rfg_linear",
        "mutation_mode": "brdas"
    }
    parents = {"father": "father_p1.jpg", "mother": "mother_p1.jpg"}
    
    exp_dir, record = logger.log_start("test_mix_fix", config, parents)
    print(f"Created experiment dir: {exp_dir}")
    print(f"Record: {record}")
    
    # Simulate metrics
    metrics = {
        "geometry": {"width_height_ratio": 0.85, "jaw_width": 0.42, "cheek_width": 0.38},
        "identity": {"arcface_father": 0.62, "arcface_mother": 0.58},
        "image": {"ssim": 0.72, "lpips": 0.18, "mae": 0.045},
        "performance": {"runtime_sec": 12.3, "gpu_memory_gb": 3.2}
    }
    
    logger.log_complete(record, metrics, "keep", "Width reduced 5%, identity preserved")
    print("Logged completion")